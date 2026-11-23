# SPDX-License-Identifier: MIT
import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter import dtypes, fused_qk_rope_concat_and_cache_mla
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding
from atom.model_ops.triton_mla_decode_qtail import mla_decode_qtail

# Kimi-K3.1 DSA decode under QREP: 96 replicated heads, 128 nope, 64 rope,
# 512 latent; DCP8 rank 0 owns every eighth row's K.
HEADS, NOPE, PE, LORA = 96, 128, 64, 512


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """e4m3 codes as signed step counts from zero, so +0 and -0 coincide."""
    c = codes.view(torch.uint8).to(torch.int32)
    return torch.where(c >= 128, 128 - c, c)


@pytest.mark.parametrize("q_scale", [1.0, 2.0**-10])
@pytest.mark.parametrize("tokens", [1, 20, 35, 96, 257])
def test_qtail_matches_bmm_rope_path(tokens, q_scale):
    torch.manual_seed(tokens)
    # q_proj's output and the fused_qkv_a slices, at the decode strides.
    q = torch.randn(tokens, HEADS, NOPE + PE, device="cuda", dtype=torch.bfloat16)
    q_nope, q_pe = q.split([NOPE, PE], dim=-1)
    w_k = 0.05 * torch.randn(HEADS, LORA, NOPE, device="cuda", dtype=torch.bfloat16)
    proj = torch.randn(tokens, LORA + PE + 160, device="cuda", dtype=torch.bfloat16)
    k_nope = torch.randn(tokens, LORA, device="cuda", dtype=torch.bfloat16)
    k_pe = proj[:, LORA : LORA + PE]
    slots = torch.randperm(4 * tokens, device="cuda")[:tokens]
    positions = 131072 + torch.arange(tokens, device="cuda")
    slots[positions % 8 != 0] = -1
    rope = NoPositionalRotaryEmbedding(PE, PE, 140000, 10000.0, dtype=torch.bfloat16).cuda()
    qs = torch.tensor([q_scale], device="cuda")
    ks = torch.ones(1, device="cuda")

    def buffers():
        return (
            torch.full((tokens, HEADS, LORA + PE), 0x7F, device="cuda", dtype=torch.uint8).view(dtypes.fp8),
            torch.zeros(4 * tokens, 1, LORA + PE, device="cuda", dtype=dtypes.fp8),
        )

    ref_q, ref_kv = buffers()
    ql_nope = batched_gemm_bf16(q_nope.transpose(0, 1), w_k).transpose(0, 1)
    fused_qk_rope_concat_and_cache_mla(
        ql_nope, q_pe, k_nope, k_pe, ref_kv, ref_q, slots, ks, qs, positions,
        rope.cos_cache, rope.sin_cache, is_neox=rope.is_neox_style, is_nope_first=True,
        compute_all_q_rope=True,
    )
    new_q, new_kv = buffers()
    mla_decode_qtail(q_nope, q_pe, w_k, new_q, k_nope, k_pe, new_kv, slots, qs, ks)

    torch.testing.assert_close(new_kv.view(torch.uint8), ref_kv.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        new_q[..., LORA:].view(torch.uint8), ref_q[..., LORA:].view(torch.uint8), rtol=0, atol=0
    )
    assert not new_q.float().isnan().any(), "q_out holds an FP8 NaN (an unwritten or overflowed code)"
    steps = (_ordinal(new_q[..., :LORA]) - _ordinal(ref_q[..., :LORA])).abs()
    assert steps.max().item() <= 1, f"nope half off by {steps.max().item()} e4m3 steps"
    assert (steps == 1).float().mean().item() < 0.06
