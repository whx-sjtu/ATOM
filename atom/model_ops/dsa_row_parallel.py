# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Row-parallel DSA prefill under DCP (``ATOM_DSA_ROW_PARALLEL_PREFILL``).

A sparse prefill under DCP replicates its query side: with query replication
(QREP) every rank projects every row's query heads, attends the 1/dcp of each
row's top-k it holds, returning the LSE, and the group merges the partials.
Row-parallel splits the rows instead, contiguously as the indexer already does
(``RowSplit.of``: rank r takes ``[r * ceil(T / dcp), ...)``). Each rank

  1. scores its rows only and maps their whole top-k to rows of the gathered
     KV (no candidate exchange, no ownership filter);
  2. writes every new token's KV at the slots it owns, as QREP does, and
     projects and absorbs its rows' queries for all the group's heads;
  3. all-gathers the step's KV rows -- every prefill request's
     ``[0, context_len)`` from the DCP-sharded fp8 cache, 576 B a token, as
     #3547's "kv" transport does;
  4. attends its rows over their whole top-k: no LSE, nothing to merge;
  5. projects V for its rows (all heads) and hands each rank its heads of
     every row back in one bf16 all-to-all -- the layout QREP's merge leaves
     for the gate and o_proj.

With ``ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT`` (on by default) the front of the
layer runs on this rank's rows too (``torch.ops.aiter.kimi_k3_dsa_front``):
``fused_qkv_a``, the q/kv norms and the indexer's ``wq_b``. Only the index
keys -- 128 of ``fused_qkv_a``'s 2304 columns -- are projected for every row:
the indexer's fused writer caches them at the slots this rank owns, which lie
in every rank's rows. The new tokens' KV then comes from their own ranks: step
2 writes this rank's rows' codes after its shard of the gathered KV (``block_
rows`` = shard rows + ``new_stride``), step 3 gathers both at once, and each
rank copies its owned slots out of the gathered rows (``write_new_kv``). With
it off the front runs on every row, as without row-parallel; it is traced
into the compiled graph that decode shares.

Numerics: a row's softmax runs over its whole top-k in one kernel call, as
without DCP, instead of eight LSE-merged partials. Every other rounding (fp8
query and KV codes, bf16 latent, bf16 V projection) is the same as QREP's;
the front's GEMMs see this rank's rows, or the index-key columns, only, so
their kernels may sum in another order.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
import triton
import triton.language as tl

from atom.utils import envs, upload_numpy

logger = logging.getLogger("atom")

ENV = "ATOM_DSA_ROW_PARALLEL_PREFILL"


def requested() -> bool:
    """Whether `ENV` is set, rather than on by default: only then does a
    configuration that cannot run it warn."""
    return ENV in os.environ


def min_tokens_env() -> int:
    n = envs.ATOM_DSA_ROW_PARALLEL_PREFILL_MIN_TOKENS
    if n < 1:
        raise ValueError(f"{ENV}_MIN_TOKENS={n}: must be >= 1")
    return n


def max_kv_rows_env() -> int:
    cap = envs.ATOM_DSA_ROW_PARALLEL_PREFILL_MAX_KV_ROWS
    if cap < 1:
        raise ValueError(f"{ENV}_MAX_KV_ROWS={cap}: must be >= 1")
    return cap


def builder_blocker(
    *,
    is_sparse: bool,
    dcp_world_size: int,
    kv_cache_dtype: str,
    mla_page_size: int,
    indexer_fp4: bool,
    index_kpool: int,
    persistent_plan: bool,
) -> str | None:
    """Why a metadata builder cannot plan row-parallel steps, or None.

    The gathered KV is fp8 cache rows read at page size 1; the indexer's
    rows must be FP4 paged rows, which score each row against the whole key
    set in one pass; the plan goes into the persistent sparse-prefill work
    buffers. Plugins and PCP keep the default path: PCP rewrites the token
    order the row split reads.
    """
    from atom.distributed.pcp_utils import pcp_is_enabled
    from atom.plugin import is_plugin_mode

    if not is_sparse:
        return "no sparse (DSA) attention"
    if dcp_world_size <= 1:
        return "DCP is off"
    if not kv_cache_dtype.startswith("fp8"):
        return "a non-fp8 KV cache"
    if mla_page_size != 1:
        return f"MLA page size {mla_page_size}"
    if not indexer_fp4:
        return "a non-FP4 indexer"
    if index_kpool > 1:
        return "a pooled indexer"
    if not persistent_plan:
        return "no persistent DCP sparse prefill plan"
    if is_plugin_mode():
        return "plugin mode"
    if pcp_is_enabled():
        return "PCP"
    return None


# ───────────────────────────── the step's KV rows, all-gathered (#3547) ──


@dataclass(frozen=True)
class GatheredKV:
    """The gathered KV's layout: rank ``r``'s rows of request ``i`` land at
    ``r * block_rows + shard_base[i] + local_index``. With new rows
    (``new_stride > 0``), the step's token ``j`` follows its rank's shard, at
    ``(j // new_stride) * block_rows + rows_pad + j % new_stride``, and the
    shard holds each request's positions before ``new_meta[0]``."""

    slot_ids: torch.Tensor  # int32 [rows_pad], this rank's rows, padded per request
    shard_base: torch.Tensor  # int32 [num_requests + 1], exclusive cumsum
    rows_pad: int
    new_stride: int = 0
    # int32 [3, num_requests]: cached lens, first token, new tokens.
    new_meta: torch.Tensor | None = None

    @property
    def block_rows(self) -> int:
        return self.rows_pad + self.new_stride


def padded_local_lens(context_lens, world: int, block_size: int) -> np.ndarray:
    """Each request's length on one DCP rank padded to whole local blocks: the
    same on every rank, since every rank holds ceil(len / (block_size *
    world)) blocks of it."""
    context_lens = np.asarray(context_lens, dtype=np.int64)
    return -(-context_lens // (block_size * world)) * block_size


def gathered_kv_rows(context_lens, world: int, block_size: int) -> int:
    """Rows of the gathered KV: every rank's padded share."""
    return world * int(padded_local_lens(context_lens, world, block_size).sum())


def gathered_kv_plan(
    context_lens, block_tables, world: int, block_size: int, device
) -> GatheredKV:
    """The gathered KV's layout for a prefill step.

    Every input is the step's batch, identical on every DCP rank, so every
    rank gathers the same row count. Padding rows read allocated slots and
    are never selected. Raises if a block table is shorter than its context:
    those blocks already hold the step's KV.
    """
    padded = padded_local_lens(context_lens, world, block_size)
    if len(block_tables) < len(padded):
        raise ValueError(f"{len(block_tables)} block tables for {len(padded)} requests")
    # Upload each request's local blocks, not its slots, and expand on the
    # GPU: 4 B a local block keeps the upload under the pageable-copy cliff.
    blocks = []
    for i, n in enumerate(padded.tolist()):
        block_ids = np.asarray(block_tables[i], dtype=np.int32)
        if len(block_ids) * block_size < n:
            raise ValueError(f"request {i}: {len(block_ids)} blocks for {n} local rows")
        blocks.append(block_ids[: n // block_size])
    shard_base = np.zeros(len(padded) + 1, dtype=np.int32)
    np.cumsum(padded, out=shard_base[1:])
    block_ids = upload_numpy(np.concatenate(blocks), device)
    offsets = torch.arange(block_size, dtype=torch.int32, device=device)
    return GatheredKV(
        slot_ids=(block_ids[:, None] * block_size + offsets).view(-1),
        shard_base=upload_numpy(shard_base, device),
        rows_pad=int(shard_base[-1]),
    )


def with_new_rows(
    kv: GatheredKV, cached_lens, seqlens_q, new_stride: int, device
) -> GatheredKV:
    """``kv`` -- planned over ``cached_lens`` -- with the step's new tokens
    after each rank's shard, ``new_stride`` rows a rank (``RowSplit.of``)."""
    cached_lens = np.asarray(cached_lens, dtype=np.int64)
    seqlens_q = np.asarray(seqlens_q, dtype=np.int64)
    if new_stride < 1 or cached_lens.shape != seqlens_q.shape:
        raise ValueError(
            f"new rows: stride {new_stride}, {cached_lens.shape} cached lens, "
            f"{seqlens_q.shape} new lens"
        )
    first = np.zeros_like(seqlens_q)
    np.cumsum(seqlens_q[:-1], out=first[1:])
    meta = np.stack([cached_lens, first, seqlens_q]).astype(np.int32)
    return replace(
        kv, new_stride=new_stride, new_meta=upload_numpy(meta.reshape(-1), device)
    )


def gather_kv(cp_group, kv_cache: torch.Tensor, plan: GatheredKV) -> torch.Tensor:
    """All-gather the step's KV rows: ``[world * rows_pad, 1, d]`` in the
    cache dtype, rank-major. fp8 rows travel as bytes; the gather is a copy."""
    from atom.model_ops.dcp_ops import dcp_all_gather, dcp_gather_compressed_kv

    local = dcp_gather_compressed_kv(kv_cache, plan.slot_ids)
    return dcp_all_gather(cp_group, local, 0).unsqueeze(1)


def local_kv(kv_cache: torch.Tensor, plan: GatheredKV) -> torch.Tensor:
    """This rank's block of a gathered KV with new rows, ``[block_rows, d]``:
    its shard read from ``kv_cache``, then ``new_stride`` rows for the caller
    to write its rows' new KV into."""
    d = kv_cache.shape[-1]
    local = torch.empty(
        (plan.block_rows, d), dtype=kv_cache.dtype, device=kv_cache.device
    )
    torch.index_select(
        kv_cache.view(-1, d).view(torch.uint8),
        0,
        plan.slot_ids,
        out=local[: plan.rows_pad].view(torch.uint8),
    )
    return local


def gather_local_kv(cp_group, local: torch.Tensor) -> torch.Tensor:
    """All-gather every rank's ``local_kv`` block: ``[world * block_rows, 1, d]``."""
    from atom.model_ops.dcp_ops import dcp_all_gather

    return dcp_all_gather(cp_group, local, 0).unsqueeze(1)


@triton.jit(do_not_specialize=["num_tokens", "block_rows", "rows_pad", "new_stride"])
def _write_new_kv_kernel(
    kv,  # int32 words [world * block_rows, WORDS]
    kv_cache,  # int32 words [slots, WORDS]
    slot_mapping,  # int64 [num_tokens]
    num_tokens,
    block_rows,
    rows_pad,
    new_stride,
    kv_stride: tl.int64,
    cache_stride: tl.int64,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slot_mapping + token)
    row = (token // new_stride) * block_rows + rows_pad + token % new_stride
    cols = tl.arange(0, BLOCK)
    mask = (cols < WORDS) & (slot >= 0)
    words = tl.load(kv + row * kv_stride + cols, mask=mask)
    tl.store(kv_cache + slot * cache_stride + cols, words, mask=mask)


def write_new_kv(
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    plan: GatheredKV,
) -> None:
    """``kv_cache[slot_mapping[j]] = `` token ``j``'s row of the gathered
    ``kv`` for every slot >= 0: this rank's owned slots of the step's new
    tokens, the codes their own ranks wrote. Bytes are copied."""
    d = kv_cache.shape[-1]
    assert kv.shape[-1] == d and d % 4 == 0 and plan.new_stride > 0
    assert slot_mapping.dtype == torch.int64
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    src = kv.reshape(-1, d).view(torch.int32)
    dst = kv_cache.view(-1, d).view(torch.int32)
    words = d // 4
    _write_new_kv_kernel[(num_tokens,)](
        src,
        dst,
        slot_mapping,
        num_tokens,
        plan.block_rows,
        plan.rows_pad,
        plan.new_stride,
        src.stride(0),
        dst.stride(0),
        WORDS=words,
        BLOCK=triton.next_power_of_2(words),
    )


@triton.jit(do_not_specialize=["block_rows", "rows_pad", "new_stride", "out_numel"])
def _gathered_kv_index_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1], absolute offsets into the output
    batch_id_per_q_token,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS], positions in the request
    shard_base,  # int32 [num_requests + 1]
    new_meta,  # int32 [3, num_requests] (NEW_ROWS)
    out_kv_indices,  # int32
    block_rows,
    rows_pad,
    new_stride,
    num_requests,
    out_numel,
    DCP_WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEW_ROWS: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    """Each candidate's row in the gathered KV, written where the dcp=1
    prefill writes its cache slot: the same positions and the same zero for an
    invalid one (negative, or past the request's padded rows -- with new
    rows, past its new tokens), so the attention reads the same rows in the
    same order."""
    token_id = tl.program_id(0)
    col_id = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    req_id = tl.load(batch_id_per_q_token + token_id)
    valid_req = req_id >= 0
    kv_start = tl.load(dsa_kv_indptr + token_id)
    kv_len = tl.load(dsa_kv_indptr + token_id + 1) - kv_start
    indice = tl.load(topk_indices + token_id * ti_stride0 + col_id * ti_stride1)
    base = tl.load(shard_base + req_id, mask=valid_req, other=0)
    end = tl.load(shard_base + req_id + 1, mask=valid_req, other=0)
    owner = (indice // INTERLEAVE) % DCP_WORLD
    local = (indice // (INTERLEAVE * DCP_WORLD)) * INTERLEAVE + indice % INTERLEAVE
    row = owner * block_rows + base + local
    in_range = base + local < end
    if NEW_ROWS:
        cached = tl.load(new_meta + req_id, mask=valid_req, other=0)
        first = tl.load(new_meta + num_requests + req_id, mask=valid_req, other=0)
        count = tl.load(new_meta + 2 * num_requests + req_id, mask=valid_req, other=0)
        new = indice >= cached
        token = first + indice - cached
        new_row = (token // new_stride) * block_rows + rows_pad + token % new_stride
        row = tl.where(new, new_row, row)
        in_range = tl.where(new, indice - cached < count, in_range)
    out_offset = kv_start + col_id
    store_mask = (col_id < kv_len) & (out_offset < out_numel)
    valid = valid_req & store_mask & (indice >= 0) & in_range
    tl.store(out_kv_indices + out_offset, tl.where(valid, row, 0), mask=store_mask)


def gathered_kv_index(
    dsa_kv_indptr: torch.Tensor,
    batch_id_per_q_token: torch.Tensor,
    topk_indices: torch.Tensor,
    plan: GatheredKV,
    world: int,
    out: torch.Tensor,
    num_topk_tokens: int,
    interleave: int = 1,
    block_n: int = 128,
) -> torch.Tensor:
    """Map a sparse-prefill top-k (positions in each token's request) to rows
    of the gathered KV (``gather_kv``), at ``dsa_kv_indptr``'s offsets into
    ``out``. Any run of rows: slice all three row inputs alike."""
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[1] == num_topk_tokens
    assert num_topk_tokens % block_n == 0
    num_tokens = min(
        dsa_kv_indptr.shape[0] - 1,
        batch_id_per_q_token.shape[0],
        topk_indices.shape[0],
    )
    if num_tokens == 0:
        return out
    ti_stride0, ti_stride1 = topk_indices.stride()
    new_rows = plan.new_meta is not None
    _gathered_kv_index_kernel[(num_tokens, num_topk_tokens // block_n)](
        dsa_kv_indptr,
        batch_id_per_q_token,
        topk_indices,
        plan.shard_base,
        plan.new_meta if new_rows else plan.shard_base,
        out,
        plan.block_rows,
        plan.rows_pad,
        plan.new_stride,
        plan.shard_base.numel() - 1,
        out.numel(),
        world,
        interleave,
        num_topk_tokens,
        block_n,
        new_rows,
        ti_stride0,
        ti_stride1,
    )
    return out


def warm_gathered_kv_index(
    world: int, interleave: int, num_topk_tokens: int, device, new_rows: bool = False
):
    """Compile the index kernel with one candidate-free token: dummy runs
    build no plan, so the first live row-parallel step would JIT it. With
    ``new_rows``, that variant and ``write_new_kv`` too."""
    zeros = torch.zeros(3, dtype=torch.int32, device=device)
    plan = GatheredKV(slot_ids=zeros[:0], shard_base=zeros[:2], rows_pad=0)
    if new_rows:
        plan = replace(plan, new_stride=1, new_meta=zeros)
        no_slot = torch.full((1,), -1, dtype=torch.int64, device=device)
        rows = torch.zeros(1, 4, dtype=torch.uint8, device=device)
        write_new_kv(rows, rows, no_slot, plan)
    gathered_kv_index(
        zeros[:2],
        zeros[:1],
        torch.zeros(1, num_topk_tokens, dtype=torch.int32, device=device),
        plan,
        world,
        zeros[:0],
        num_topk_tokens,
        interleave,
    )


# ─────────────────────────────────────────────── the step's plan ──


@dataclass
class DsaRowParallel:
    """A row-parallel prefill step, built by the metadata builder: this
    rank's rows ``[first, end)`` of ``num_tokens``, ``stride`` rows a rank
    (``RowSplit.of``), the gathered KV, and the attention's arguments over
    this rank's rows. ``kv_indptr`` is the rows' ``sparse_kv_indptr`` rebased
    to 0; their candidates start at ``kv_base`` of ``sparse_kv_indices``.
    ``front``: the front runs on these rows (``ATOM_DSA_ROW_PARALLEL_PREFILL_
    FRONT``); ``kv`` then carries the new rows, which the q tail writes at
    ``new_slots`` of this rank's ``local_kv`` block."""

    num_tokens: int
    first: int
    end: int
    stride: int
    kv: GatheredKV
    kv_base: int
    kv_indptr: torch.Tensor  # int32 [end - first + 1]
    cu_seqlens_q: torch.Tensor  # int32 [end - first + 1]
    last_page_lens: torch.Tensor  # int32 [end - first]
    no_slots: torch.Tensor  # int64 [end - first], all -1
    work: Mapping[str, torch.Tensor]
    new_slots: torch.Tensor | None = None  # int64 [end - first], kv.rows_pad + i

    @property
    def rows(self) -> int:
        return self.end - self.first

    @property
    def front(self) -> bool:
        return self.new_slots is not None


def own_kv_indptr(
    host_indptr: np.ndarray, first: int, end: int
) -> tuple[np.ndarray, int]:
    """Rows ``[first, end)``'s slice of the step's host ``sparse_kv_indptr``,
    rebased to 0, and the offset it was rebased by."""
    own = host_indptr[first : end + 1].astype(np.int64)
    base = int(own[0])
    return (own - base).astype(np.int32), base


# ───────────────────────────────────────── layer helpers ──


def write_kv(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """``kv_cache[slot] = e4m3([kv_c, k_pe] / k_scale)`` for every slot >= 0:
    ``mla_q_tail_and_cache``'s KV half, by the same kernel with no query
    heads, so the same codes.

    ``kv_cache`` is ``[blocks, block_size, L + R]``; scales as there."""
    from atom.model_ops.triton_mla_q_tail_cache import (
        _BLOCK_H,
        _BLOCK_T,
        _NUM_WARPS,
        _mla_q_tail_and_cache_kernel,
    )

    num_tokens = kv_c.shape[0]
    kv_c = kv_c.reshape(num_tokens, -1)
    k_pe = k_pe.reshape(num_tokens, -1)
    l, r = kv_c.shape[1], k_pe.shape[1]
    assert kv_cache.shape[-1] == l + r and slot_mapping.dtype == torch.int64
    assert slot_mapping.shape[0] == num_tokens
    for t in (kv_c, k_pe, kv_cache):
        assert t.stride(-1) == 1
    if num_tokens == 0:
        return
    on_device = k_scale.is_cuda
    if on_device:
        inv_k = 1.0
    else:
        inv_k, k_scale = 1.0 / float(k_scale), slot_mapping
    _mla_q_tail_and_cache_kernel[(triton.cdiv(num_tokens, _BLOCK_T), 1)](
        k_pe,  # q_pe: no heads, never read
        kv_c,
        k_pe,
        kv_cache,
        kv_cache,  # q_out: no heads, never written
        slot_mapping,
        k_scale,  # q_scale: never read
        k_scale,
        1.0,
        inv_k,
        num_tokens,
        0,
        kv_cache.shape[1],
        0,
        0,
        kv_c.stride(0),
        k_pe.stride(0),
        0,
        0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        L=l,
        R=r,
        BLOCK_T=_BLOCK_T,
        BLOCK_H=_BLOCK_H,
        # The q-tail call's bound is its q_out's, the same fp8 type here.
        FP8_MAX=torch.finfo(kv_cache.dtype).max,
        SCALES_ON_DEVICE=on_device,
        num_warps=_NUM_WARPS,
    )


def rows_to_heads(projected: torch.Tensor, num_rows: int, cp_group) -> torch.Tensor:
    """Each rank's rows of all heads to every row of this rank's heads.

    ``projected``: ``[world * h, stride, v]`` head-major, this rank's
    ``stride`` rows. Returns ``[num_rows, h * v]``, rows in order: rank r's
    rows are rows ``[r * stride, (r + 1) * stride)``. Rows past the step's
    tokens -- a rank's tail, or the layer input's padding -- are not
    defined."""
    world = cp_group.world_size
    heads, stride, v = projected.shape
    send = projected.view(world, heads // world, stride, v)
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=cp_group.device_group)
    out = recv.permute(0, 2, 1, 3).reshape(world * stride, -1)
    if out.shape[0] >= num_rows:
        return out[:num_rows]
    return torch.nn.functional.pad(out, (0, 0, 0, num_rows - out.shape[0]))


@triton.jit(do_not_specialize=["stride", "num_rows", "num_tokens", "head0"])
def _rows_to_heads_pull_kernel(
    values_tab,  # [world] int64: each rank's [world * h, stride, V] BF16 projection
    out,  # [num_rows, h * V] BF16, rows of out_stride
    stride,
    num_rows,
    num_tokens,
    head0,
    out_stride: tl.int64,
    V: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """``out[peer * stride + s, h * V:(h + 1) * V] = peer's projection[head0 +
    h, s]`` for the rows below ``num_tokens``, and zeros for the others below
    ``num_rows``. A copy. The source pointers are raw 64-bit peer addresses,
    16-byte aligned (`dcp_ops.PeerBuffers`)."""
    peer = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    row = peer * stride + s
    store = (s < stride) & (row < num_rows)
    load = store & (row < num_tokens)
    src = tl.multiple_of(
        tl.load(values_tab + peer).to(tl.pointer_type(tl.bfloat16)), 16
    )
    d = tl.arange(0, V)
    vals = tl.load(
        src + ((head0 + h) * stride + s)[:, None] * V + d[None, :],
        mask=load[:, None],
        other=0.0,
    )
    tl.store(
        out + row[:, None] * out_stride + h * V + d[None, :], vals, mask=store[:, None]
    )


_ROWS_PULL_BLOCK_S = 16
_rows_pull_logged = False


def rows_to_heads_pull_buffer(cp_group, heads: int, stride: int, v: int):
    """This rank's peer-pull workspace (`dcp_ops.prepare_dcp_prefill_pull` of
    ``(cp_group, heads, v)``) as the ``[heads, stride, v]`` BF16 projection
    `rows_to_heads_pull` reads, or None when the step keeps `rows_to_heads`:
    no workspace, or rows a rank past its chunk. Reads only shapes, the same
    on every DCP rank, so every rank takes the same path."""
    from atom.model_ops.dcp_ops import _PULL_WORKSPACES, _pull_key

    ws = _PULL_WORKSPACES.get(_pull_key(cp_group, heads, v))
    if ws is None or not 0 < stride <= ws.rows or heads % ws.ranks:
        return None
    return ws.values[: heads * stride * v].view(heads, stride, v)


def rows_to_heads_pull(
    projected: torch.Tensor, num_rows: int, num_tokens: int, cp_group
) -> torch.Tensor:
    """`rows_to_heads` through the peer-pull workspace: ``projected`` is
    `rows_to_heads_pull_buffer`'s view, written on every rank. After a barrier
    each rank copies its heads of every rank's rows from that rank's
    workspace, then a second barrier keeps the next writer of the workspace
    from overwriting rows a peer still reads. Returns ``[num_rows, h * v]``:
    bit for bit `rows_to_heads`' rows below ``num_tokens`` (a copy), zeros in
    the rows past them, which `rows_to_heads` leaves undefined."""
    from atom.model_ops.dcp_ops import _PULL_WORKSPACES, _pull_barrier, _pull_key

    global _rows_pull_logged
    heads, stride, v = projected.shape
    key = _pull_key(cp_group, heads, v)
    ws = _PULL_WORKSPACES[key]
    assert projected.data_ptr() == ws.values.data_ptr() and projected.is_contiguous()
    assert projected.dtype == torch.bfloat16
    h_local = heads // ws.ranks
    out = torch.empty(
        (num_rows, h_local * v), dtype=torch.bfloat16, device=projected.device
    )
    _pull_barrier(ws, key)
    _rows_to_heads_pull_kernel[
        (ws.ranks, triton.cdiv(stride, _ROWS_PULL_BLOCK_S), h_local)
    ](
        ws.values_tab,
        out,
        stride,
        num_rows,
        num_tokens,
        ws.rank * h_local,
        out.stride(0),
        V=v,
        BLOCK_S=_ROWS_PULL_BLOCK_S,
    )
    _pull_barrier(ws, key)
    if num_rows > ws.ranks * stride:
        out[ws.ranks * stride :].zero_()
    if not _rows_pull_logged:
        _rows_pull_logged = True
        logger.info(
            "Row-parallel DSA prefill pulls rows_to_heads from peer memory: first "
            "step %d tokens, %d rows a rank",
            num_tokens,
            stride,
        )
    return out


def scratch_bytes(max_kv_rows: int, world: int, entry_bytes: int) -> int:
    """The gathered KV at the cap and this rank's share of it, for the memory
    profile, whose dummy run attends nothing. Every other row-parallel buffer
    is smaller than the QREP one it replaces, which is not profiled either.
    New rows count against the cap (``block_rows``)."""
    return (max_kv_rows + -(-max_kv_rows // world)) * entry_bytes
