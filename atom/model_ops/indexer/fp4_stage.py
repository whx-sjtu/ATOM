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

With ``ATOM_DCP_PREFILL_INDEX_PULL`` (`prepare_pull`, `stage_from_peers`) the
rows skip the all-gather: each rank packs its rows into its own peer memory,
and after one barrier the stage kernel reads each key's row from its owner.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from .layout import fp4_scale_byte

logger = logging.getLogger("atom")

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


@triton.jit(do_not_specialize=["n_rows"])
def _fp4_pack_planes_kernel(
    cache_ptr,
    scale_ptr,
    page_ptr,
    row_ptr,
    scale_at_ptr,
    data_ptr,
    scales_ptr,
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
    """`_fp4_pack_rows_kernel` into two planes: a row's data bytes at
    ``data_ptr``, its scale bytes at ``scales_ptr``."""
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
    tl.store(
        data_ptr + r[:, None] * (CHUNKS * CHUNK_BYTES) + col[None, :],
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
    tl.store(scales_ptr + r[:, None] * SCALES + j[None, :], scale, mask=keep[:, None])


@triton.jit(do_not_specialize=["n_slots", "n_local"])
def _fp4_stage_peers_kernel(
    data_tab,  # [ranks] int64: each rank's [n_local, 64] data plane
    scales_tab,  # [ranks] int64: each rank's [n_local, 4] scale plane
    row_of_ptr,
    gather_ptr,
    staged_ptr,
    staged_scale_ptr,
    n_slots,
    n_local,
    PAGE_ROWS: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    SCALES: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """`_fp4_stage_rows_kernel` reading gathered row ``i`` from row
    ``i % n_local`` of rank ``i // n_local``'s planes. The plane addresses stay
    raw 64-bit pointers: as offsets from a pointer argument the AMD backend
    makes them 32-bit buffer offsets, and peers lie far apart."""
    s = tl.program_id(0).to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S)
    keep = s < n_slots
    i = tl.load(row_of_ptr + s, mask=keep, other=-1)
    has = i >= 0
    src = tl.load(gather_ptr + tl.where(has, i, 0), mask=has, other=0)
    owner = src // n_local
    at = (src - owner * n_local).to(tl.int64)
    width = CHUNKS * CHUNK_BYTES
    data_base = tl.multiple_of(
        tl.load(data_tab + owner).to(tl.pointer_type(tl.uint8)), 16
    )
    col = tl.arange(0, CHUNKS * CHUNK_BYTES)
    data = tl.load(
        data_base[:, None] + at[:, None] * width + col[None, :],
        mask=has[:, None],
        other=0,
    )
    dst = (
        (s // PAGE_ROWS)[:, None] * (CHUNKS * PAGE_ROWS * CHUNK_BYTES)
        + (col // CHUNK_BYTES)[None, :] * (PAGE_ROWS * CHUNK_BYTES)
        + (s % PAGE_ROWS)[:, None] * CHUNK_BYTES
        + (col % CHUNK_BYTES)[None, :]
    )
    tl.store(staged_ptr + dst, data, mask=keep[:, None])
    j = tl.arange(0, SCALES)
    scale_base = tl.load(scales_tab + owner).to(tl.pointer_type(tl.uint8))
    scale = tl.load(
        scale_base[:, None] + at[:, None] * SCALES + j[None, :],
        mask=has[:, None],
        other=0,
    )
    tl.store(
        staged_scale_ptr + _scale_byte(s[:, None], j[None, :]),
        scale,
        mask=keep[:, None],
    )


def _pack_planes(kv_cache, kv_cache_scale, page, row, scale_at, data, scales):
    """`pack_rows` into ``data`` [>= n, 64] and ``scales`` [>= n, 4]."""
    n = page.numel()
    cache = _bytes(kv_cache)
    block = 64
    _fp4_pack_planes_kernel[(triton.cdiv(n, block),)](
        cache,
        _bytes(kv_cache_scale),
        page,
        row,
        scale_at,
        data,
        scales,
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


def _stage_peers(
    data_tab, scales_tab, n_local, gather_index, row_of, kv_cache, kv_cache_scale
):
    """`stage_rows` of the rows each rank packed into its planes."""
    num_slots = row_of.numel()
    pages = num_slots // _PAGE_ROWS
    staged = kv_cache.new_empty(pages, *kv_cache.shape[1:])
    staged_scale = kv_cache_scale.new_empty(pages, *kv_cache_scale.shape[1:])
    if num_slots == 0:
        return staged, staged_scale
    block = 64
    _fp4_stage_peers_kernel[(triton.cdiv(num_slots, block),)](
        data_tab,
        scales_tab,
        row_of,
        gather_index,
        _bytes(staged),
        _bytes(staged_scale),
        num_slots,
        n_local,
        PAGE_ROWS=_PAGE_ROWS,
        CHUNKS=_CHUNKS,
        CHUNK_BYTES=_CHUNK_BYTES,
        SCALES=_SCALES,
        BLOCK_S=block,
    )
    return staged, staged_scale


# Peer memory of a DCP group (`prepare_pull`): None when it was tried and the
# staging keeps the all-gather.
_PULLS: dict = {}
# Rows a rank, about a million keys a step over 8 ranks; a longer step keeps
# the all-gather.
PULL_MAX_ROWS = 1 << 17


class _Pull:
    """Two slots of up to ``rows`` packed rows a rank, data and scale bytes in
    planes of their own, and the barrier's signal words. Consecutive stagings
    alternate between the slots, so none needs a barrier after its reads: a
    rank packs a slot again only past the next staging's barrier, which every
    peer reaches after its reads of that slot, queued before it on the same
    stream (`stage_from_peers` keeps the stagings on one stream)."""

    def __init__(self, cp_group, rows):
        from atom.model_ops.dcp_ops import _PULL_SIGNAL_BYTES, PeerBuffers

        self.rows = rows
        self.peer = PeerBuffers(
            cp_group, (2 * rows * _DATA, 2 * rows * _SCALES, _PULL_SIGNAL_BYTES)
        )
        self.data = self.peer.local(0, torch.uint8).view(2, rows, _DATA)
        self.scales = self.peer.local(1, torch.uint8).view(2, rows, _SCALES)
        data_tab, scales_tab = self.peer.tabs[:2]
        # [slot, rank]
        self.data_tabs = torch.stack((data_tab, data_tab + rows * _DATA))
        self.scales_tabs = torch.stack((scales_tab, scales_tab + rows * _SCALES))
        self.stagings = 0
        self.stream = None


def _warmup_pull(pull: _Pull) -> None:
    """Compile both pull kernels on this rank's own planes."""
    device = pull.data.device
    cache = torch.zeros(
        (1, _CHUNKS, _PAGE_ROWS, _CHUNK_BYTES), dtype=torch.uint8, device=device
    )
    scale = torch.zeros((1, 2, 32, 4), dtype=torch.uint8, device=device)
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    scale_at = fp4_scale_byte(slots[:, None], torch.arange(_SCALES, device=device))
    _pack_planes(cache, scale, slots, slots, scale_at, pull.data[0], pull.scales[0])
    # Tables of their own: production's are 16-byte aligned rows of the slots'.
    own = slice(pull.peer.rank, pull.peer.rank + 1)
    gather = torch.zeros(1, dtype=torch.int32, device=device)
    _stage_peers(
        pull.data_tabs[0, own].clone(),
        pull.scales_tabs[0, own].clone(),
        1,
        gather,
        stage_row_of(gather, _PAGE_ROWS),
        cache,
        scale,
    )
    torch.cuda.synchronize(device)


def prepare_pull(cp_group) -> bool:
    """Create ``cp_group``'s peer memory for `stage_from_peers`.

    Collective (it maps every rank's buffers): call it on every rank of
    ``cp_group`` in the same order. Its memory sits outside torch's allocator,
    so it must exist before the KV cache is sized. Returns False on every rank,
    and the staging keeps the all-gather, when it cannot be allocated or
    mapped.
    """
    key = id(cp_group)
    if key in _PULLS:
        return _PULLS[key] is not None
    _PULLS[key] = None
    ranks = cp_group.world_size
    if ranks < 2 or ranks & (ranks - 1):
        logger.warning(
            "DCP index-key staging pull: needs a power-of-two group over 1; "
            "got %d ranks, staging with all-gather",
            ranks,
        )
        return False
    try:
        pull = _Pull(cp_group, PULL_MAX_ROWS)
    except RuntimeError as error:
        logger.warning(
            "DCP index-key staging pull peer memory: %s; staging with all-gather",
            error,
        )
        return False
    _warmup_pull(pull)
    _PULLS[key] = pull
    logger.info(
        "DCP prefill index-key staging pulls from peer memory: up to %d rows a "
        "rank, %.1f MB of peer memory a rank",
        pull.rows,
        pull.peer.nbytes / 1e6,
    )
    return True


def stage_from_peers(
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    page: torch.Tensor,
    row: torch.Tensor,
    scale_at: torch.Tensor,
    local_total: int,
    gather_index: torch.Tensor,
    row_of: torch.Tensor,
    cp_group,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``stage_rows(cp_group.all_gather(pack_rows(kv_cache, kv_cache_scale,
    page, row, scale_at)), gather_index, row_of, kv_cache, kv_cache_scale)``,
    bit for bit, through the group's peer memory; or None when this call cannot
    take it: no peer memory, a shard of none or more rows than it holds, or a
    graph capture. Every rank must make the same calls: each takes a barrier
    paired with the same call on every peer, and every condition above is the
    same on every rank."""
    pull = _PULLS.get(id(cp_group))
    if (
        pull is None
        or not 0 < local_total <= pull.rows
        or page.numel() != local_total
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    from atom.model_ops.dcp_ops import peer_barrier

    slot = pull.stagings % 2
    pull.stagings += 1
    stream = torch.cuda.current_stream()
    if pull.stream is not None and pull.stream != stream:
        stream.wait_stream(pull.stream)
    pull.stream = stream
    _pack_planes(
        kv_cache,
        kv_cache_scale,
        page,
        row,
        scale_at,
        pull.data[slot],
        pull.scales[slot],
    )
    peer_barrier(pull.peer)
    return _stage_peers(
        pull.data_tabs[slot],
        pull.scales_tabs[slot],
        local_total,
        gather_index,
        row_of,
        kv_cache,
        kv_cache_scale,
    )


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
