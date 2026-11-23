# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""mla_q_tail_and_cache against aiter's fused_qk_rope_concat_and_cache_mla
under an identity rope: same q_out rope slice and KV-cache bytes, and q_out's
nope half left as it was."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aiter import dtypes, fused_qk_rope_concat_and_cache_mla

from atom.model_ops.triton_mla_q_tail_cache import mla_q_tail_and_cache

L, R, P = 512, 64, 128
FP8 = dtypes.fp8
DEV = "cuda"


def _inputs(tokens, heads, block_size, owned_every, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(tokens, heads, P + R, device=DEV, generator=g).to(torch.bfloat16)
    q[0, 0, P:] = -0.0  # a signed zero whose partner is negative after rotation
    kv_a = torch.randn(tokens, L + R, device=DEV, generator=g).to(torch.bfloat16)
    own = torch.arange(tokens, device=DEV) % owned_every == 0
    n_own = int(own.sum())
    blocks = n_own // block_size + 2
    slot = torch.full((tokens,), -1, dtype=torch.int64, device=DEV)
    slot[own] = torch.randperm(blocks * block_size, device=DEV, generator=g)[:n_own]
    return q[..., P:], kv_a[:, :L], kv_a[:, L:], slot, blocks


def _aiter(q_pe, k_nope, k_pe, slot, blocks, block_size, q_out, scale):
    cache = torch.zeros(blocks, block_size, L + R, dtype=FP8, device=DEV)
    tokens, heads = q_pe.shape[:2]
    cos = torch.ones(4096, 1, 1, R // 2, dtype=torch.bfloat16, device=DEV)
    sin = torch.zeros_like(cos)
    fused_qk_rope_concat_and_cache_mla(
        q_pe.new_empty(L).expand(tokens, heads, -1),
        q_pe,
        k_nope,
        k_pe,
        cache,
        q_out,
        slot,
        scale,
        scale,
        torch.arange(tokens, device=DEV, dtype=torch.int64),
        cos,
        sin,
        is_neox=False,
        is_nope_first=True,
        compute_all_q_rope=True,
    )
    return cache


def _assert_rope_slice_matches(got, ref):
    # aiter's x * cos - y * sin turns -0.0 into +0.0; nothing else may differ
    diff = got != ref
    assert torch.equal(diff, diff & (ref == 0) & (got == 0x80))


@pytest.mark.parametrize("tokens", [1, 7, 300, 2051])
@pytest.mark.parametrize("heads", [12, 96])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize(
    "host_scale", [False, True], ids=["device-scale", "host-scale"]
)
def test_matches_aiter_identity_rope(tokens, heads, block_size, host_scale):
    q_pe, k_nope, k_pe, slot, blocks = _inputs(
        tokens, heads, block_size, 8, tokens + heads
    )
    dscale = torch.tensor(1.0, dtype=torch.float32, device=DEV)
    ref_q = torch.zeros(tokens, heads, L + R, dtype=FP8, device=DEV)
    ref_cache = _aiter(q_pe, k_nope, k_pe, slot, blocks, block_size, ref_q, dscale)

    sentinel = torch.full((tokens, heads, L + R), 0x3C, dtype=torch.uint8, device=DEV)
    q_out = sentinel.clone().view(FP8)
    cache = torch.zeros(blocks, block_size, L + R, dtype=FP8, device=DEV)
    scale = dscale.cpu() if host_scale else dscale
    mla_q_tail_and_cache(q_pe, k_nope, k_pe, cache, q_out, slot, scale, scale)
    torch.cuda.synchronize()

    got, ref = q_out.view(torch.uint8), ref_q.view(torch.uint8)
    _assert_rope_slice_matches(got[..., L:], ref[..., L:])
    assert torch.equal(got[..., :L], sentinel[..., :L])
    assert torch.equal(cache.view(torch.uint8), ref_cache.view(torch.uint8))


def test_strided_q_out_and_scale():
    tokens, heads, block_size = 40, 12, 1
    q_pe, k_nope, k_pe, slot, blocks = _inputs(tokens, heads, block_size, 3, 5)
    padded = torch.zeros(tokens, 16, L + R, dtype=FP8, device=DEV)
    q_out = padded[:, :heads]
    scale = torch.tensor(0.5, dtype=torch.float32, device=DEV)
    ref_q = torch.zeros(tokens, heads, L + R, dtype=FP8, device=DEV)
    ref_cache = _aiter(q_pe, k_nope, k_pe, slot, blocks, block_size, ref_q, scale)
    cache = torch.zeros(blocks, block_size, L + R, dtype=FP8, device=DEV)
    mla_q_tail_and_cache(q_pe, k_nope, k_pe, cache, q_out, slot, scale, scale)
    torch.cuda.synchronize()
    got = q_out.contiguous().view(torch.uint8)[..., L:]
    _assert_rope_slice_matches(got, ref_q.view(torch.uint8)[..., L:])
    assert torch.equal(cache.view(torch.uint8), ref_cache.view(torch.uint8))
    assert not padded[:, heads:].view(torch.uint8).any()
