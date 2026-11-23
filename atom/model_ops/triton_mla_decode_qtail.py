# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The NoPE MLA decode query tail in one launch.

For a layer whose rotary embedding is the identity (``NoPositionalRotaryEmbedding``)
the decode path's ``W_K`` absorb (``batched_gemm_bf16`` into a bf16 intermediate)
and ``fused_qk_rope_concat_and_cache_mla`` (quantize the intermediate and q_pe
into ``q_out``, write the step's K into the cache) collapse into this kernel:

    q_out[t, h, :L] = e4m3(q_nope[t, h] @ w_k[h].T / q_scale)
    q_out[t, h, L:] = e4m3(q_pe[t, h] / q_scale)
    cache[slot[t]]  = e4m3([k_nope[t], k_pe[t]] / k_scale)    where slot[t] >= 0

Each program owns BLOCK_N absorb columns of one head for BLOCK_M rows, so a
decode step's few rows still spread over heads x L / BLOCK_N programs; the
programs at column block 0 also store the head's rope slice, and those of head 0
the rows' K.

Numerics: the bf16 intermediate of the two-kernel path is not formed, so the
nope half of q_out takes one rounding (fp32 to e4m3) instead of two and moves by
at most one e4m3 step. The rope half and the cache are bit-identical for the
identity rope. Values are clamped to the finite e4m3 range.
"""

import torch
import triton
import triton.language as tl

_BLOCK_N = 128


def mla_decode_qtail_supports(
    nope_head_dim: int, rope_head_dim: int, kv_lora_rank: int
) -> bool:
    """``tl.arange`` over P, PE and L compiles only for powers of two, and the
    column blocks have no tail mask."""
    return (
        all(
            d == triton.next_power_of_2(d)
            for d in (nope_head_dim, rope_head_dim, kv_lora_rank)
        )
        and kv_lora_rank % _BLOCK_N == 0
    )


@triton.jit
def _mla_decode_qtail_kernel(
    q_nope_ptr,  # [T, H, P], last dim contiguous
    q_pe_ptr,  # [T, H, PE], last dim contiguous
    w_k_ptr,  # [H, L, P], last dim contiguous
    q_out_ptr,  # [T, H, L + PE], last dim contiguous
    k_nope_ptr,  # [T, L], last dim contiguous
    k_pe_ptr,  # [T, PE], last dim contiguous
    cache_ptr,  # [slots, L + PE], rows contiguous
    slot_ptr,  # [T] int64, < 0: not this rank's
    q_scale_ptr,
    k_scale_ptr,
    num_tokens,
    stride_qn_t,
    stride_qn_h,
    stride_qp_t,
    stride_qp_h,
    stride_w_h,
    stride_w_l,
    stride_o_t,
    stride_o_h,
    stride_kn_t,
    stride_kp_t,
    stride_c_s,
    P: tl.constexpr,
    PE: tl.constexpr,
    L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens
    rows = offs_m.to(tl.int64)[:, None]
    offs_p = tl.arange(0, P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(
        q_nope_ptr + rows * stride_qn_t + head * stride_qn_h + offs_p[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    w = tl.load(
        w_k_ptr + head * stride_w_h + offs_n[:, None] * stride_w_l + offs_p[None, :]
    )
    acc = tl.dot(a, tl.trans(w))
    inv_q = 1.0 / tl.load(q_scale_ptr)
    out_row = q_out_ptr + rows * stride_o_t + head * stride_o_h
    tl.store(
        out_row + offs_n[None, :],
        tl.clamp(acc * inv_q, -FP8_MAX, FP8_MAX).to(q_out_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )
    if pid_n == 0:
        offs_r = tl.arange(0, PE)
        pe = tl.load(
            q_pe_ptr + rows * stride_qp_t + head * stride_qp_h + offs_r[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_row + L + offs_r[None, :],
            tl.clamp(pe * inv_q, -FP8_MAX, FP8_MAX).to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
        if head == 0:
            slot = tl.load(slot_ptr + offs_m, mask=m_mask, other=-1)
            ok = (m_mask & (slot >= 0))[:, None]
            inv_k = 1.0 / tl.load(k_scale_ptr)
            c_row = cache_ptr + slot.to(tl.int64)[:, None] * stride_c_s
            offs_l = tl.arange(0, L)
            kn = tl.load(
                k_nope_ptr + rows * stride_kn_t + offs_l[None, :], mask=ok, other=0.0
            ).to(tl.float32)
            tl.store(
                c_row + offs_l[None, :],
                tl.clamp(kn * inv_k, -FP8_MAX, FP8_MAX).to(cache_ptr.dtype.element_ty),
                mask=ok,
            )
            kp = tl.load(
                k_pe_ptr + rows * stride_kp_t + offs_r[None, :], mask=ok, other=0.0
            ).to(tl.float32)
            tl.store(
                c_row + L + offs_r[None, :],
                tl.clamp(kp * inv_k, -FP8_MAX, FP8_MAX).to(cache_ptr.dtype.element_ty),
                mask=ok,
            )


def mla_decode_qtail(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    w_k: torch.Tensor,
    q_out: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """Fill ``q_out`` and write the step's K, in place (see the module docstring).

    Args:
        q_nope: ``[T, H, P]`` bf16 view of ``q_proj``'s output.
        q_pe: ``[T, H, PE]`` bf16 view of the same output; identity rope.
        w_k: ``[H, L, P]`` bf16 absorb weight (``W_K_qrep``).
        q_out: ``[T, H, L + PE]`` FP8.
        k_nope, k_pe: ``[T, L]`` and ``[T, PE]`` bf16 (rows may be strided).
        kv_cache: FP8 cache whose rows, flattened to ``[-1, L + PE]``, are slots.
        slot_mapping: ``[T]`` int64; rows with a negative slot write no K.
        q_scale, k_scale: fp32 per-tensor scales.
    """
    t, h, p = q_nope.shape
    l, pe = w_k.shape[1], q_pe.shape[-1]
    assert w_k.shape == (h, l, p) and q_pe.shape[:2] == (t, h)
    assert q_out.shape == (t, h, l + pe)
    k_nope = k_nope.reshape(t, l)
    k_pe = k_pe.reshape(t, pe)
    cache = kv_cache.view(-1, l + pe)
    for x in (q_nope, q_pe, w_k, q_out, k_nope, k_pe, cache):
        assert x.stride(-1) == 1
    if t == 0:
        return
    block_m = max(16, min(64, 1 << (t - 1).bit_length()))
    _mla_decode_qtail_kernel[(-(-t // block_m), h, l // _BLOCK_N)](
        q_nope,
        q_pe,
        w_k,
        q_out,
        k_nope,
        k_pe,
        cache,
        slot_mapping,
        q_scale,
        k_scale,
        t,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        w_k.stride(0),
        w_k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        k_nope.stride(0),
        k_pe.stride(0),
        cache.stride(0),
        P=p,
        PE=pe,
        L=l,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        FP8_MAX=torch.finfo(q_out.dtype).max,
        num_warps=4,
        num_stages=1,
    )
