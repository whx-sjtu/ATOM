# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP decode top-k merge: this rank's owned KV slots of the global top-k.

The contract and the output are aiter's ``flydsl_dcp_topk_merge``, bit for
bit: the same threshold (the k-th largest of the row's ``W * k_loc``
candidates in the fp32 order map with NaN at the bottom), the same admission
of threshold ties in flat candidate order, and the same emit order (this
rank's plane in position order). Only the schedule differs:

* the row is staged in a whole block's registers (``threads``, default 1024,
  16 candidates a thread at W=8, k_loc=2048) and selected in ``radix`` passes
  (default 11/11/10 bits). The first pass's histogram has 2048 bins: the
  leading 8 bits are the sign and 7 exponent bits, into which a row's
  candidates fall in a handful of bins, so every lane of a wave added into
  one LDS word;
* this rank's plane -- its scores, local indices and their block-table slots
  -- is loaded at the start, contiguous per thread, so the emit waits on no
  global load and is one block scan, not one scan and two dependent loads per
  256 candidates.

The candidates come either row-major (``layout="row"``, aiter's
``[rows, W * k_loc]``) or as the dim-0 all-gather leaves them
(``layout="plane"``, ``[W * rows, k_loc]``), which saves the transposing copy
into the row-major form.

The pack (one block per row, its offset from a scan of the counts) is
aiter's.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    get_warp_size,
    uint32_to_int32,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import Int32, arith, const_expr, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

DEFAULT_THREADS = 1024
DEFAULT_RADIX = (11, 11, 10)

_WAVE = 64
_INT32_MIN = -(1 << 31)
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1
_PACK_THREADS = 256
_DPP_ROW_SHR_1 = 0x111
_DPP_ROW_SHR_2 = 0x112
_DPP_ROW_SHR_4 = 0x114
_DPP_ROW_SHR_8 = 0x118
_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF
# s_acc slots
_PREFIX, _MASK, _REM, _PRIOR_EQ = 0, 1, 2, 3


def parse_config(spec: str) -> tuple[int, tuple[int, ...]]:
    """``"<threads>:<b0>,<b1>,..."`` -> (threads, radix); empty -> defaults."""
    if not spec:
        return DEFAULT_THREADS, DEFAULT_RADIX
    threads, _, bits = spec.partition(":")
    radix = tuple(int(b) for b in bits.split(",")) if bits else DEFAULT_RADIX
    return int(threads), radix


def _f32_to_ord(val):
    """aiter's merge map: order-preserving fp32 -> int32, NaN at INT32_MIN."""
    bits = val.bitcast(Int32)
    ords = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    abs_bits = bits & fx.Int32(0x7FFFFFFF)
    is_nan = fx.Uint32(abs_bits) > 0x7F800000
    return is_nan.select(fx.Int32(uint32_to_int32(0x80000000)), ords)


def _unwrap(val):
    return val.ir_value() if hasattr(val, "ir_value") else arith.unwrap(val)


def _warp_inclusive_prefix_i32(val, lane):
    val_raw = _unwrap(val)
    zero_raw = _unwrap(0)
    for dpp_op, threshold in (
        (_DPP_ROW_SHR_1, 1),
        (_DPP_ROW_SHR_2, 2),
        (_DPP_ROW_SHR_4, 4),
        (_DPP_ROW_SHR_8, 8),
    ):
        remote = fly_rocdl.update_dpp(
            T.i32, zero_raw, val_raw, dpp_op, _DPP_ROW_MASK, _DPP_BANK_MASK, True
        )
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap(val)
    src16 = (lane & fx.Int32(0x30)) - 1
    r16 = fly_rocdl.ds_bpermute(T.i32, src16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(r16), val)
    src32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    r32 = fly_rocdl.ds_bpermute(T.i32, src32 * fx.Int32(4), val)
    return (lane >= fx.Int32(32)).select(val + fx.Int32(r32), val)


@flyc.jit
def _block_exclusive_prefix_i32(val, scan, tid, NWAVES: fx.Constexpr[int]):
    """(exclusive prefix, block total) of ``val``; uniform control flow only.
    ``scan`` holds NWAVES + 1 words."""
    lane = tid % fx.Int32(_WAVE)
    warp = tid // fx.Int32(_WAVE)
    inclusive = _warp_inclusive_prefix_i32(val, lane)
    exclusive = inclusive - val
    if lane == fx.Int32(_WAVE - 1):
        scan[warp] = inclusive
    gpu.barrier()
    if warp == 0:
        wv = fx.Int32(0)
        if lane < fx.Int32(NWAVES):
            wv = scan[lane]
        wi = _warp_inclusive_prefix_i32(wv, lane)
        if lane < fx.Int32(NWAVES):
            scan[lane] = wi - wv
        if lane == fx.Int32(NWAVES - 1):
            scan[NWAVES] = wi
    gpu.barrier()
    res = scan[warp] + exclusive
    tot = scan[NWAVES]
    # a fast wave's next call must not overwrite what a slow one still reads
    gpu.barrier()
    return res, tot


def _check_config(n_cand, k_loc, threads, radix):
    if threads % _WAVE or not _WAVE <= threads <= 1024:
        raise ValueError(f"threads must be a multiple of {_WAVE} in [64, 1024]")
    if sum(radix) != 32 or any(not 1 <= b <= 14 for b in radix):
        raise ValueError(f"radix digits must be 1..14 bits summing to 32: {radix}")
    if k_loc >= 1 << _PACK_SHIFT:
        raise ValueError("k_loc must fit the 16-bit packed emit counts")


def build_dcp_topk_merge(
    n_cand, k_loc, topk_tokens, page_size, threads, radix, plane_major=False
):
    _check_config(n_cand, k_loc, threads, radix)
    waves = threads // _WAVE
    steps = -(-n_cand // threads)
    own = -(-k_loc // threads)
    take = min(topk_tokens, n_cand)
    shifts = tuple(sum(radix[i + 1 :]) for i in range(len(radix)))
    hist_bins = 1 << max(radix)
    tag = "_".join(str(b) for b in radix) + ("_p" if plane_major else "")

    def cand_offset(step, tid, row, rows_m):
        """Flat offset of candidate ``step * threads + tid`` of ``row``; only
        meaningful below n_cand."""
        c0 = step * threads
        if not plane_major:
            return row * fx.Int32(n_cand) + fx.Int32(c0) + tid
        if k_loc % threads == 0:
            plane = fx.Int32(c0 // k_loc)
            return (plane * rows_m + row) * fx.Int32(k_loc) + fx.Int32(c0 % k_loc) + tid
        c = fx.Int32(c0) + tid
        plane = c // fx.Int32(k_loc)
        return (plane * rows_m + row) * fx.Int32(k_loc) + (c - plane * fx.Int32(k_loc))

    def own_offset(t, row, rank, rows_m):
        if not plane_major:
            return row * fx.Int32(n_cand) + rank * fx.Int32(k_loc) + t
        return (rank * rows_m + row) * fx.Int32(k_loc) + t

    @fx.struct
    class Storage:
        bins: fx.Array[Int32, hist_bins, 16]
        scan: fx.Array[Int32, waves + 1, 16]
        acc: fx.Array[Int32, 4, 4]

    @flyc.kernel(
        name=f"atom_dcp_merge_select_t{threads}_r{tag}",
        known_block_size=[threads, 1, 1],
    )
    def select_kernel(
        gathered_scores: fx.Tensor,
        local_idx: fx.Tensor,
        block_table: fx.Tensor,
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        rows_m: fx.Int32,
        rank: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        row_local = fx.slice(local_idx, (row, None))
        row_bt = fx.slice(block_table, (row, None))
        row_stage = fx.slice(staging, (row, None))
        base = rank * fx.Int32(k_loc)

        # This rank's plane first, so its block-table loads (which wait on the
        # local indices) are in flight while the row is selected.
        own_j = fx.make_rmem_tensor(own, Int32)
        own_ord = fx.make_rmem_tensor(own, Int32)
        own_slot = fx.make_rmem_tensor(own, Int32)
        for i in range_constexpr(own):
            t = tid * fx.Int32(own) + fx.Int32(i)
            in_plane = t < fx.Int32(k_loc)
            safe_t = in_plane.select(t, fx.Int32(0))
            own_j[i] = in_plane.select(row_local[safe_t], fx.Int32(-1))
        ukeys = fx.make_rmem_tensor(steps, Int32)
        for step in range_constexpr(steps):
            c = step * threads + tid
            in_cand = c < fx.Int32(n_cand)
            off = in_cand.select(cand_offset(step, tid, row, rows_m), fx.Int32(0))
            ord_c = in_cand.select(
                _f32_to_ord(gathered_scores[off]), fx.Int32(_INT32_MIN)
            )
            # sign flipped: the digits below are unsigned
            ukeys[step] = ord_c ^ fx.Int32(_INT32_MIN)
        for i in range_constexpr(own):
            t = tid * fx.Int32(own) + fx.Int32(i)
            in_plane = t < fx.Int32(k_loc)
            safe_t = in_plane.select(t, fx.Int32(0))
            own_ord[i] = _f32_to_ord(
                gathered_scores[own_offset(safe_t, row, rank, rows_m)]
            )
            j = own_j[i]
            safe_j = (j >= 0).select(j, fx.Int32(0))
            own_slot[i] = row_bt[safe_j // fx.Int32(page_size)] * fx.Int32(
                page_size
            ) + (safe_j % fx.Int32(page_size))

        storage = fx.SharedAllocator().allocate(Storage)
        s_hist = storage.bins.peek().view(fx.make_layout(hist_bins, 1))
        s_scan = storage.scan.peek().view(fx.make_layout(waves + 1, 1))
        s_acc = storage.acc.peek().view(fx.make_layout(4, 1))

        if tid == 0:
            s_acc[_PREFIX] = 0
            s_acc[_MASK] = 0
            s_acc[_REM] = fx.Int32(take)
        gpu.barrier()

        for p in range_constexpr(len(radix)):
            bins = 1 << radix[p]
            shift = shifts[p]
            digit_mask = bins - 1
            per_thread = -(-bins // threads)
            for i in range_constexpr(per_thread):
                b = tid * fx.Int32(per_thread) + fx.Int32(i)
                if b < fx.Int32(bins):
                    s_hist[b] = 0
            gpu.barrier()

            prefix = s_acc[_PREFIX]
            dmask = s_acc[_MASK]
            for step in range_constexpr(steps):
                c = step * threads + tid
                if c < fx.Int32(n_cand):
                    key = ukeys[step]
                    bv = (key >> fx.Int32(shift)) & fx.Int32(digit_mask)
                    if const_expr(p == 0):
                        atomic_add_i32(s_hist, 1, bv, "workgroup")
                    else:
                        if (key & dmask) == prefix:
                            atomic_add_i32(s_hist, 1, bv, "workgroup")
            gpu.barrier()

            # A thread's bins are consecutive, ascending: the bucket of the
            # rem-th largest is the one whose ascending prefix straddles
            # total - rem.
            counts = fx.make_rmem_tensor(per_thread, Int32)
            local_total = fx.Int32(0)
            for i in range_constexpr(per_thread):
                b = tid * fx.Int32(per_thread) + fx.Int32(i)
                in_bins = b < fx.Int32(bins)
                cnt = in_bins.select(
                    s_hist[in_bins.select(b, fx.Int32(0))], fx.Int32(0)
                )
                counts[i] = cnt
                local_total = local_total + cnt
            excl, total = _block_exclusive_prefix_i32(local_total, s_scan, tid, waves)
            rem = s_acc[_REM]
            # every thread has read rem before the hit thread rewrites it
            gpu.barrier()
            target = total - rem
            run = excl
            for i in range_constexpr(per_thread):
                incl = run + counts[i]
                if (run <= target) & (incl > target):
                    b = tid * fx.Int32(per_thread) + fx.Int32(i)
                    s_acc[_PREFIX] = prefix | (b << fx.Int32(shift))
                    s_acc[_MASK] = dmask | fx.Int32(
                        uint32_to_int32(digit_mask << shift)
                    )
                    s_acc[_REM] = rem - (total - incl)
                run = incl
            gpu.barrier()

        t_ukey = s_acc[_PREFIX]
        threshold = t_ukey ^ fx.Int32(_INT32_MIN)
        remaining_k = s_acc[_REM]

        # Threshold ties in the planes before this one, liveness not checked:
        # they take the admission room first.
        prior_local = fx.Int32(0)
        for step in range_constexpr(steps):
            c = step * threads + tid
            before = (c < base) & (c < fx.Int32(n_cand))
            prior_local = prior_local + (before & (ukeys[step] == t_ukey)).select(
                fx.Int32(1), fx.Int32(0)
            )
        _, prior_eq = _block_exclusive_prefix_i32(prior_local, s_scan, tid, waves)

        above_cnt = fx.Int32(0)
        equal_cnt = fx.Int32(0)
        for i in range_constexpr(own):
            live = own_j[i] >= 0
            o = own_ord[i]
            above_cnt = above_cnt + (live & (o > threshold)).select(
                fx.Int32(1), fx.Int32(0)
            )
            equal_cnt = equal_cnt + (live & (o == threshold)).select(
                fx.Int32(1), fx.Int32(0)
            )
        packed = (above_cnt << fx.Int32(_PACK_SHIFT)) + equal_cnt
        pack_before, pack_total = _block_exclusive_prefix_i32(
            packed, s_scan, tid, waves
        )
        ab_run = pack_before >> fx.Int32(_PACK_SHIFT)
        eq_run = pack_before & fx.Int32(_PACK_MASK)
        ab_total = pack_total >> fx.Int32(_PACK_SHIFT)
        eq_total = pack_total & fx.Int32(_PACK_MASK)
        room = remaining_k - prior_eq
        room = (room < 0).select(fx.Int32(0), room)
        for i in range_constexpr(own):
            live = own_j[i] >= 0
            o = own_ord[i]
            above = live & (o > threshold)
            equal = live & (o == threshold)
            keep = above | (equal & (eq_run < room))
            dst = ab_run + (eq_run < room).select(eq_run, room)
            if keep:
                row_stage[dst] = own_slot[i]
            ab_run = ab_run + above.select(fx.Int32(1), fx.Int32(0))
            eq_run = eq_run + equal.select(fx.Int32(1), fx.Int32(0))
        if tid == 0:
            owned_counts[row] = ab_total + (eq_total < room).select(eq_total, room)

    pack_waves = _PACK_THREADS // _WAVE
    pack_steps = -(-k_loc // _PACK_THREADS)

    @fx.struct
    class PackStorage:
        scan: fx.Array[Int32, pack_waves + 1, 16]
        dst: fx.Array[Int32, 1, 4]

    @flyc.kernel(name="atom_dcp_merge_pack", known_block_size=[_PACK_THREADS, 1, 1])
    def pack_kernel(
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        out_kv_indices: fx.Tensor,
        out_kv_indptr: fx.Tensor,
        rows_n: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        storage = fx.SharedAllocator().allocate(PackStorage)
        s_scan = storage.scan.peek().view(fx.make_layout(pack_waves + 1, 1))
        s_dst = storage.dst.peek().view(fx.make_layout(1, 1))

        if tid == 0:
            s_dst[0] = 0
        gpu.barrier()
        chunk = fx.Int32(0)
        running = fx.Int32(0)
        step_blk = fx.Int32(_PACK_THREADS)
        for _ in range(fx.Int32(0), (rows_n + step_blk - 1) // step_blk, fx.Int32(1)):
            r = chunk + tid
            cnt = (r < rows_n).select(
                owned_counts[(r < rows_n).select(r, fx.Int32(0))], fx.Int32(0)
            )
            excl, chunk_total = _block_exclusive_prefix_i32(
                cnt, s_scan, tid, pack_waves
            )
            if r == row:
                s_dst[0] = running + excl
            if row == 0 and r < rows_n:
                out_kv_indptr[r + fx.Int32(1)] = running + excl + cnt
            running = running + chunk_total
            chunk = chunk + step_blk
        if row == 0 and tid == 0:
            out_kv_indptr[0] = 0
        gpu.barrier()
        dst = s_dst[0]
        n = owned_counts[row]
        row_stage = fx.slice(staging, (row, None))
        for step in range_constexpr(pack_steps):
            t = step * _PACK_THREADS + tid
            if t < n:
                out_kv_indices[dst + t] = row_stage[t]

    @flyc.jit
    def launch(
        gathered_scores: fx.Tensor,
        local_idx: fx.Tensor,
        block_table: fx.Tensor,
        out_kv_indices: fx.Tensor,
        out_kv_indptr: fx.Tensor,
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        rows_m: fx.Int32,
        rank_i32: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008  framework idiom
    ):
        sel = select_kernel(
            gathered_scores,
            local_idx,
            block_table,
            staging,
            owned_counts,
            rows_m,
            rank_i32,
        )
        sel.launch(grid=(rows_m, 1, 1), block=(threads, 1, 1), stream=stream)
        pk = pack_kernel(staging, owned_counts, out_kv_indices, out_kv_indptr, rows_m)
        pk.launch(grid=(rows_m, 1, 1), block=(_PACK_THREADS, 1, 1), stream=stream)

    return launch


@functools.cache
def _get_launcher(n_cand, k_loc, topk_tokens, page_size, threads, radix, plane_major):
    return build_dcp_topk_merge(
        n_cand, k_loc, topk_tokens, page_size, threads, radix, plane_major
    )


@functools.lru_cache(maxsize=128)
def _validate(sc, li, bt, idx, indptr, counts, stage, dcp_rank, world, topk, layout):
    """aiter's preconditions (``aiter.ops.flydsl.dcp_topk_merge._validate``),
    plus the plane-major candidate layout."""
    (sc_shape, sc_dtype, sc_contig), (li_shape, li_dtype), (bt_shape, bt_dtype) = (
        sc,
        li,
        bt,
    )
    if len(sc_shape) != 2 or sc_dtype != torch.float32 or not sc_contig:
        raise ValueError("gathered_scores must be a contiguous 2D float32 tensor")
    if world <= 0:
        raise ValueError("world_size must be positive")
    if layout == "row":
        rows, n_cand = sc_shape
        if n_cand % world:
            raise ValueError("gathered_scores.shape[1] must be divisible by world_size")
        k_loc = n_cand // world
    elif layout == "plane":
        if sc_shape[0] % world:
            raise ValueError("gathered_scores.shape[0] must be divisible by world_size")
        rows, k_loc = sc_shape[0] // world, sc_shape[1]
    else:
        raise ValueError(f"layout must be 'row' or 'plane', got {layout!r}")
    if not 0 <= dcp_rank < world:
        raise ValueError("dcp_rank out of range")
    if li_shape != (rows, k_loc) or li_dtype != torch.int32:
        raise ValueError("local_idx must be int32 [rows, n_cand // world_size]")
    if len(bt_shape) != 2 or bt_shape[0] != rows or bt_dtype != torch.int32:
        raise ValueError("block_table must be int32 [rows, max_blocks]")
    if idx[1] != torch.int32 or indptr[1] != torch.int32 or counts[1] != torch.int32:
        raise ValueError("out_kv_indices, out_kv_indptr, owned_counts must be int32")
    if stage != ((rows, k_loc), torch.int32):
        raise ValueError("staging must be int32 [rows, n_cand // world_size]")
    if indptr[0][0] < rows + 1 or counts[0][0] < rows:
        raise ValueError("out_kv_indptr / owned_counts too short")
    if idx[0] < rows * min(k_loc, topk):
        raise ValueError("out_kv_indices too small for the worst case")
    if get_warp_size() != _WAVE:
        raise ValueError("the DCP merge kernel requires a wave64 GPU")


def dcp_topk_merge(
    gathered_scores: torch.Tensor,
    local_idx: torch.Tensor,
    block_table: torch.Tensor,
    out_kv_indices: torch.Tensor,
    out_kv_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
    staging: torch.Tensor,
    dcp_rank: int,
    world_size: int,
    topk_tokens: int,
    page_size: int,
    threads: int = DEFAULT_THREADS,
    radix: tuple[int, ...] = DEFAULT_RADIX,
    layout: str = "row",
) -> None:
    """``aiter.ops.topk.flydsl_dcp_topk_merge``, same arguments and output.

    ``layout="plane"`` takes ``gathered_scores`` as ``[W * rows, k_loc]``, rank
    ``w``'s candidates for row ``r`` in row ``w * rows + r`` (a dim-0
    all-gather of ``[rows, k_loc]``); ``"row"`` is aiter's ``[rows, W * k_loc]``.
    Unused candidate slots must score -inf (``top_k_per_row_decode(values=)``
    pads so). The first call for a shape JIT-compiles: warm it up before a
    CUDAGraph capture.
    """
    _validate(
        (
            tuple(gathered_scores.shape),
            gathered_scores.dtype,
            gathered_scores.is_contiguous(),
        ),
        (tuple(local_idx.shape), local_idx.dtype),
        (tuple(block_table.shape), block_table.dtype),
        (out_kv_indices.numel(), out_kv_indices.dtype),
        (tuple(out_kv_indptr.shape), out_kv_indptr.dtype),
        (tuple(owned_counts.shape), owned_counts.dtype),
        (tuple(staging.shape), staging.dtype),
        dcp_rank,
        world_size,
        topk_tokens,
        layout,
    )
    rows, k_loc = local_idx.shape
    launcher = _get_launcher(
        world_size * k_loc,
        k_loc,
        topk_tokens,
        page_size,
        threads,
        tuple(radix),
        layout == "plane",
    )
    _run_compiled(
        launcher,
        gathered_scores.view(-1),
        local_idx,
        block_table,
        out_kv_indices,
        out_kv_indptr,
        staging,
        owned_counts,
        rows,
        dcp_rank,
        torch.cuda.current_stream(gathered_scores.device),
    )
