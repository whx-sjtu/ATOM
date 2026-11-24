# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MFMA FP4 index staging for a DCP prefill in two launches
(``ATOM_SPG_INDEXER_STAGE``).

`_dcp_stage_indexer_fp4_prefill` reads the local shard with two advanced
indexes and a cat, and after the all-gather reorders the rows with an
``index_select``, zero-fills two page buffers and scatters data and scales
with two ``index_put``. Here one kernel packs the local rows and one writes
every staged byte from its source row, or zero where the stage has none: the
same bytes, without the reordered copy or the zero fill.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .layout import fp4_scale_byte

# Bytes per row: four 16-byte E2M1 chunks, then four e8m0 scales.
_CHUNKS, _CHUNK_BYTES, _SCALES = 4, 16, 4
_DATA = _CHUNKS * _CHUNK_BYTES
ROW_BYTES = _DATA + _SCALES
_PAGE_ROWS = 64


@triton.jit
def _scale_byte(row, chunk):
    """`layout.fp4_scale_byte`, int64 ``row``."""
    return (
        row // 64 * 256
        + chunk % 2 * 128
        + row % 32 * 4
        + chunk // 2 * 2
        + row % 64 // 32
    )


@triton.jit(do_not_specialize=["n_rows"])
def _fp4_pack_rows_kernel(
    cache_ptr,
    scale_ptr,
    page_ptr,
    row_ptr,
    scale_at_ptr,
    out_ptr,
    n_rows,
    stride_page,
    stride_chunk,
    stride_row,
    stride_scale_at,
    CHUNKS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    SCALES: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    keep = r < n_rows
    page = tl.load(page_ptr + r, mask=keep, other=0).to(tl.int64)
    row = tl.load(row_ptr + r, mask=keep, other=0).to(tl.int64)
    col = tl.arange(0, CHUNKS * CHUNK_BYTES)
    src = (
        page[:, None] * stride_page
        + (col // CHUNK_BYTES)[None, :] * stride_chunk
        + row[:, None] * stride_row
        + (col % CHUNK_BYTES)[None, :]
    )
    dst = out_ptr + r[:, None] * (CHUNKS * CHUNK_BYTES + SCALES)
    tl.store(
        dst + col[None, :],
        tl.load(cache_ptr + src, mask=keep[:, None]),
        mask=keep[:, None],
    )
    j = tl.arange(0, SCALES)
    at = tl.load(
        scale_at_ptr + r[:, None] * stride_scale_at + j[None, :],
        mask=keep[:, None],
        other=0,
    )
    scale = tl.load(scale_ptr + at, mask=keep[:, None])
    tl.store(dst + CHUNKS * CHUNK_BYTES + j[None, :], scale, mask=keep[:, None])


@triton.jit(do_not_specialize=["n_slots"])
def _fp4_stage_rows_kernel(
    rows_ptr,
    row_of_ptr,
    gather_ptr,
    staged_ptr,
    staged_scale_ptr,
    n_slots,
    PAGE_ROWS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    SCALES: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S)
    keep = s < n_slots
    i = tl.load(row_of_ptr + s, mask=keep, other=-1)
    has = i >= 0
    src = tl.load(gather_ptr + tl.where(has, i, 0), mask=has, other=0).to(tl.int64)
    width = CHUNKS * CHUNK_BYTES + SCALES
    base = rows_ptr + src[:, None] * width
    col = tl.arange(0, CHUNKS * CHUNK_BYTES)
    data = tl.load(base + col[None, :], mask=has[:, None], other=0)
    dst = (
        (s // PAGE_ROWS)[:, None] * (CHUNKS * PAGE_ROWS * CHUNK_BYTES)
        + (col // CHUNK_BYTES)[None, :] * (PAGE_ROWS * CHUNK_BYTES)
        + (s % PAGE_ROWS)[:, None] * CHUNK_BYTES
        + (col % CHUNK_BYTES)[None, :]
    )
    tl.store(staged_ptr + dst, data, mask=keep[:, None])
    j = tl.arange(0, SCALES)
    scale = tl.load(
        base + CHUNKS * CHUNK_BYTES + j[None, :], mask=has[:, None], other=0
    )
    tl.store(
        staged_scale_ptr + _scale_byte(s[:, None], j[None, :]),
        scale,
        mask=keep[:, None],
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def stage_row_of(stage_slots: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Each staged slot's global key row, or -1 for a page's unwritten tail:
    the inverse of the stage slots, which are distinct. Once per forward."""
    row_of = torch.full((num_slots,), -1, dtype=torch.int32, device=stage_slots.device)
    row_of[stage_slots.long()] = torch.arange(
        stage_slots.numel(), dtype=torch.int32, device=stage_slots.device
    )
    return row_of


def supports(kv_cache: torch.Tensor, kv_cache_scale: torch.Tensor) -> bool:
    """The OPUS gfx950 byte planes these kernels address: 64-row pages of
    ``[4, 64, 16]`` data and 256 dense scale bytes."""
    return (
        kv_cache.element_size() == 1
        and kv_cache_scale.element_size() == 1
        and tuple(kv_cache.shape[1:]) == (_CHUNKS, _PAGE_ROWS, _CHUNK_BYTES)
        and kv_cache.stride(-1) == 1
        and kv_cache_scale.is_contiguous()
        and kv_cache_scale[0].numel() == _PAGE_ROWS * _SCALES
    )


def pack_rows(
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    page: torch.Tensor,
    row: torch.Tensor,
    scale_at: torch.Tensor,
) -> torch.Tensor:
    """``cat(kv_cache[page, :, row, :].flatten(1),
    kv_cache_scale.view(-1)[scale_at], 1)``: ``[n, 68]`` bytes."""
    n = page.numel()
    out = torch.empty((n, ROW_BYTES), dtype=torch.uint8, device=kv_cache.device)
    if n == 0:
        return out
    cache = _bytes(kv_cache)
    block = 64
    _fp4_pack_rows_kernel[(triton.cdiv(n, block),)](
        cache,
        _bytes(kv_cache_scale),
        page,
        row,
        scale_at,
        out,
        n,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        scale_at.stride(0),
        CHUNKS=_CHUNKS,
        CHUNK_BYTES=_CHUNK_BYTES,
        SCALES=_SCALES,
        BLOCK_R=block,
    )
    return out


def stage_rows(
    rows: torch.Tensor,
    gather_index: torch.Tensor,
    row_of: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The staged ``(data, scale)`` pages from the all-gathered ``rows``:
    slot ``s`` holds ``rows[gather_index[row_of[s]]]``, or zeros when
    ``row_of[s]`` is -1."""
    num_slots = row_of.numel()
    pages = num_slots // _PAGE_ROWS
    staged = kv_cache.new_empty(pages, *kv_cache.shape[1:])
    staged_scale = kv_cache_scale.new_empty(pages, *kv_cache_scale.shape[1:])
    if num_slots == 0:
        return staged, staged_scale
    block = 64
    _fp4_stage_rows_kernel[(triton.cdiv(num_slots, block),)](
        rows,
        row_of,
        gather_index,
        _bytes(staged),
        _bytes(staged_scale),
        num_slots,
        PAGE_ROWS=_PAGE_ROWS,
        CHUNKS=_CHUNKS,
        CHUNK_BYTES=_CHUNK_BYTES,
        SCALES=_SCALES,
        BLOCK_S=block,
    )
    return staged, staged_scale


def warmup(device) -> None:
    """Compile both kernels: warmup never reaches a long DCP prefill."""
    cache = torch.zeros(
        (1, _CHUNKS, _PAGE_ROWS, _CHUNK_BYTES), dtype=torch.uint8, device=device
    )
    scale = torch.zeros((1, 2, 32, 4), dtype=torch.uint8, device=device)
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    scale_at = fp4_scale_byte(slots[:, None], torch.arange(_SCALES, device=device))
    rows = pack_rows(cache, scale, slots, slots, scale_at)
    gather = torch.zeros(1, dtype=torch.int32, device=device)
    stage_rows(rows, gather, stage_row_of(gather, _PAGE_ROWS), cache, scale)
