# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 ``q_out`` rope slice and MLA KV-cache write for an identity rotary.

Under ``NoPositionalRotaryEmbedding`` (cos 1, sin 0) the rope of
``fused_qk_rope_concat_and_cache_mla`` is the identity, so the op reduces to
casts: ``q_out[..., L:] = e4m3(q_pe / q_scale)`` for every token and
``kv_cache[slot] = e4m3([kv_c, k_pe] / k_scale)`` for a slot >= 0. Paired with
``mla_absorb_q_pack``, which stores ``q_out[..., :L]``, this skips the aiter
op's store of a stand-in nope half that the absorb then overwrites -- 512 of
the 576 bytes it writes per head.

Numerics: identical codes for finite in-range input. aiter computes
``x * 1 - y * 0``, which turns a -0.0 into +0.0 when its partner is negative;
values past the e4m3 range are clamped here, as ``mla_absorb_q_pack`` does.
"""

import torch
import triton
import triton.language as tl

# (tokens, heads) per program. Picked on gfx950 at 96 heads, T 2K-16K
# (runs/PF3/t0); any value is correct.
_BLOCK_T = 4
_BLOCK_H = 32
_NUM_WARPS = 4


@triton.jit
def _mla_q_tail_and_cache_kernel(
    q_pe_ptr,  # [T, H, R]
    kv_c_ptr,  # [T, L]
    k_pe_ptr,  # [T, R]
    kv_cache_ptr,  # [blocks, block_size, L + R] through its strides
    q_out_ptr,  # [T, H, >= L + R]
    slot_ptr,  # int64 [T]
    q_scale_ptr,
    k_scale_ptr,
    inv_q_host,
    inv_k_host,
    num_tokens,
    num_heads,
    block_size,
    stride_qpe_t,
    stride_qpe_h,
    stride_kvc_t,
    stride_kpe_t,
    stride_qo_t,
    stride_qo_h,
    stride_cache_block,
    stride_cache_entry,
    L: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALES_ON_DEVICE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_ok = tok < num_tokens
    tok64 = tok.to(tl.int64)
    head = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    col = tl.arange(0, R)
    mask = tok_ok[:, None, None] & (head < num_heads)[None, :, None]

    x = tl.load(
        q_pe_ptr
        + tok64[:, None, None] * stride_qpe_t
        + head[None, :, None] * stride_qpe_h
        + col[None, None, :],
        mask=mask,
        other=0.0,
    )
    if SCALES_ON_DEVICE:
        inv_q = 1.0 / tl.load(q_scale_ptr)
    else:
        inv_q = inv_q_host
    x = tl.clamp(x.to(tl.float32) * inv_q, -FP8_MAX, FP8_MAX)
    tl.store(
        q_out_ptr
        + tok64[:, None, None] * stride_qo_t
        + head[None, :, None] * stride_qo_h
        + L
        + col[None, None, :],
        x.to(q_out_ptr.dtype.element_ty),
        mask=mask,
    )

    if pid_h == 0:
        slot = tl.load(slot_ptr + tok, mask=tok_ok, other=-1)
        own = (slot >= 0)[:, None]
        entry = (
            kv_cache_ptr
            + (slot // block_size) * stride_cache_block
            + (slot % block_size) * stride_cache_entry
        )[:, None]
        if SCALES_ON_DEVICE:
            inv_k = 1.0 / tl.load(k_scale_ptr)
        else:
            inv_k = inv_k_host
        lat = tl.arange(0, L)
        k_c = tl.load(kv_c_ptr + tok64[:, None] * stride_kvc_t + lat[None, :], mask=own)
        k_c = tl.clamp(k_c.to(tl.float32) * inv_k, -FP8_MAX, FP8_MAX)
        tl.store(entry + lat[None, :], k_c.to(kv_cache_ptr.dtype.element_ty), mask=own)
        k_r = tl.load(k_pe_ptr + tok64[:, None] * stride_kpe_t + col[None, :], mask=own)
        k_r = tl.clamp(k_r.to(tl.float32) * inv_k, -FP8_MAX, FP8_MAX)
        tl.store(
            entry + L + col[None, :], k_r.to(kv_cache_ptr.dtype.element_ty), mask=own
        )


def mla_q_tail_and_cache(
    q_pe: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    q_out: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    q_scale: torch.Tensor,
    block_t: int = _BLOCK_T,
    block_h: int = _BLOCK_H,
    num_warps: int = _NUM_WARPS,
) -> None:
    """``fused_qk_rope_concat_and_cache_mla`` for an identity rope with
    ``compute_all_q_rope``, minus the nope half of ``q_out``.

    Args:
        q_pe: ``[T, H, R]``; last dim contiguous.
        kv_c: ``[T, L]`` or ``[T, 1, L]``; last dim contiguous.
        k_pe: ``[T, R]`` or ``[T, 1, R]``; last dim contiguous.
        kv_cache: ``[blocks, block_size, L + R]`` FP8, entries ``[nope | pe]``.
        q_out: ``[T, H, >= L + R]`` FP8; only columns ``L:L+R`` are written.
        slot_mapping: int64 ``[T]``; a negative slot writes no KV.
        k_scale, q_scale: fp32 per-tensor scales, both on the device or both
            on the host (MLAAttention keeps them there); a host scale is read
            on the host, so neither way synchronizes.
    """
    num_tokens, num_heads, r = q_pe.shape
    kv_c = kv_c.reshape(num_tokens, -1)
    k_pe = k_pe.reshape(num_tokens, -1)
    l = kv_c.shape[1]
    assert k_pe.shape[1] == r and kv_cache.shape[-1] == l + r
    assert q_out.shape[:2] == (num_tokens, num_heads) and q_out.shape[2] >= l + r
    assert slot_mapping.dtype == torch.int64 and slot_mapping.shape[0] == num_tokens
    for t in (q_pe, kv_c, k_pe, kv_cache, q_out):
        assert t.stride(-1) == 1
    if num_tokens == 0:
        return
    on_device = q_scale.is_cuda
    assert k_scale.is_cuda == on_device
    if on_device:
        inv_q = inv_k = 1.0
    else:
        inv_q, inv_k = 1.0 / float(q_scale), 1.0 / float(k_scale)
        q_scale = k_scale = slot_mapping
    _mla_q_tail_and_cache_kernel[
        (triton.cdiv(num_tokens, block_t), triton.cdiv(num_heads, block_h))
    ](
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        q_out,
        slot_mapping,
        q_scale,
        k_scale,
        inv_q,
        inv_k,
        num_tokens,
        num_heads,
        kv_cache.shape[1],
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c.stride(0),
        k_pe.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        L=l,
        R=r,
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        FP8_MAX=torch.finfo(q_out.dtype).max,
        SCALES_ON_DEVICE=on_device,
        num_warps=num_warps,
    )
