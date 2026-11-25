# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Point a native DCP sparse MLA work plan at the current layer's keys.

The metadata builder plans the forward's query rows once, one whole row a work
item. Decode templates also reserve an item for every empty row. Only each row's
rank-owned key range changes from one DSA layer to the next, so a layer rewrites
the items' KV start and end instead of planning again.
"""

import math

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 256
# aiter's v1.2 planner charges every work item 16 blocks on top of its KV
# blocks at page size 1 (fixed_over_head_num_blocks), and in auto-split mode
# deals sum_blocks over min(round(1.2 * sqrt(sum_blocks * cus)), cus) parts.
_PLAN_OVERHEAD_BLOCKS = 16
_PLAN_SPLIT_COEF = 1.2


@triton.jit
def _patch_work_kv_ranges(
    work_info, items, kv_indptr, GROUPS: tl.constexpr, BLOCK: tl.constexpr
):
    # aiter's MlaWorkInfo (csrc/include/mla.h) is eight int32: batch_idx (the
    # row) at 0, kv_start and kv_end at 4, 5.
    item = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = item < tl.load(items)
    info = work_info + item * 8
    row = tl.load(info, mask=live, other=0) // GROUPS
    # The planner counts KV from the indptr's first entry.
    first = tl.load(kv_indptr)
    tl.store(info + 4, tl.load(kv_indptr + row, mask=live) - first, mask=live)
    tl.store(info + 5, tl.load(kv_indptr + row + 1, mask=live) - first, mask=live)


def patch_work_kv_ranges(
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    rows: int,
    head_groups: int = 1,
) -> None:
    """Give each work item of a whole-row plan over ``rows`` query rows its
    row's range in ``kv_indptr``.

    The plan must hold at most one unsplit work item a row, as a plan at a KV
    granularity no row exceeds does. A row with no keys can get no item, so
    the plan's own item count (``work_indptr[-1]``) bounds the patch. aiter's
    persistent MLA kernel takes a row's keys from its work item alone, not
    from the indptr it is passed.

    A plan over head groups has ``head_groups`` consecutive rows a ``kv_indptr``
    row, each holding an equal share of its heads; they share its keys.
    """
    assert work_info_set.dtype == torch.int32 and work_info_set.shape[-1] == 8, (
        "aiter's MlaWorkInfo layout changed: work_info_set is "
        f"{tuple(work_info_set.shape)} {work_info_set.dtype}, the patch writes [*, 8] int32"
    )
    _patch_work_kv_ranges[(-(-rows // BLOCK),)](
        work_info_set, work_indptr[-1:], kv_indptr, GROUPS=head_groups, BLOCK=BLOCK
    )


class TemplatePlans:
    """Which native decode template each set of persistent work buffers holds,
    by the address of its ``work_info_set``.

    The builder plans a template from its geometry alone (rows, heads, head
    groups, dtypes and split parameters) over constant inputs: unit query and
    key ranges and unit last pages. A layer then rewrites only its items' KV
    start and end, from each item's batch_idx and the item count
    (`patch_work_kv_ranges`, the q tail's plan patch), so buffers that held a
    geometry's template still hold it on the next step over that geometry.
    Anything that may leave them otherwise must forget them. It keeps the
    buffers it vouches for, so no reallocation can pass for them at their
    address.
    """

    def __init__(self):
        self._held = {}

    def holds(self, buffers: tuple[torch.Tensor, ...], template) -> bool:
        """Whether ``buffers`` (``work_info_set`` first, then the planner's
        other outputs and inputs) hold ``template``; never for None."""
        held = self._held.get(buffers[0].data_ptr())
        return (
            template is not None
            and held is not None
            and held[0] == template
            and all(a is b for a, b in zip(held[1], buffers, strict=True))
        )

    def wrote(self, buffers: tuple[torch.Tensor, ...], template) -> None:
        """Record a planner write into ``buffers``: a template, or None for
        any other plan."""
        if template is None:
            self._held.pop(buffers[0].data_ptr(), None)
        else:
            self._held[buffers[0].data_ptr()] = (template, buffers)

    def forget(self) -> None:
        self._held.clear()


# The target's sparse MTP verify templates (ATOM_MLA_SPARSE_MTP_PLAN_REUSE).
sparse_mtp_templates = TemplatePlans()


@triton.jit
def _plan_whole_rows(
    work_info,
    work_indptr,
    reduce_indptr,
    work_meta,
    kv_indptr,
    indptr_addr,
    info_addr,
    rows,
    rows_per_cu,
    num_cu,
    reduce_size,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < rows
    first = tl.load(kv_indptr)
    kv_start = tl.load(kv_indptr + i, mask=live, other=0) - first
    kv_end = tl.load(kv_indptr + i + 1, mask=live, other=0) - first
    # MlaWorkInfo: batch_idx, partial_qo_loc, qo_start, qo_end, kv_start,
    # kv_end, kv_offset, padding.
    f = tl.arange(0, 8)[None, :]
    r = i[:, None]
    info = tl.where(f == 0, r, 0)
    info = tl.where(f == 1, -1, info)
    info = tl.where(f == 2, r, info)
    info = tl.where(f == 3, r + 1, info)
    info = tl.where(f == 4, kv_start[:, None], info)
    info = tl.where(f == 5, kv_end[:, None], info)
    tl.store(work_info + r * 8 + f, info, mask=live[:, None])
    tl.store(work_indptr + i, tl.minimum(i * rows_per_cu, rows), mask=i <= num_cu)
    tl.store(reduce_indptr + i, 0, mask=i < reduce_size)
    if tl.program_id(0) == 0:
        j = tl.arange(0, 2)
        tl.store(work_meta + j, tl.where(j == 0, indptr_addr, info_addr))


def whole_row_plan_rows_per_cu(
    host_kv_indptr: np.ndarray, kv_granularity: int, num_cu: int
) -> int | None:
    """How many rows each workgroup takes in aiter's fast-mode, auto-split
    plan over ``host_kv_indptr``'s rows at page size 1, or None where that
    plan is not one whole-row item a row dealt out evenly: a row with no keys
    gets no item, one past ``kv_granularity`` splits, and too few rows split
    over fewer workgroups than ``num_cu``."""
    rows = host_kv_indptr.shape[0] - 1
    if rows < 1:
        return None
    lens = np.diff(host_kv_indptr)
    if lens.min() < 1 or lens.max() > kv_granularity:
        return None
    unit = 1 + _PLAN_OVERHEAD_BLOCKS
    sum_blocks = rows * unit
    if _PLAN_SPLIT_COEF * math.sqrt(sum_blocks * num_cu) < num_cu + 1:
        return None
    payload = -(-sum_blocks // num_cu) + _PLAN_OVERHEAD_BLOCKS
    return payload // unit


def plan_whole_rows(
    kv_indptr: torch.Tensor,
    rows: int,
    rows_per_cu: int,
    work: dict[str, torch.Tensor],
) -> None:
    """Write the plan ``get_mla_metadata_v1`` makes for ``rows`` query rows of
    one token each when ``whole_row_plan_rows_per_cu`` gives ``rows_per_cu``:
    item i is row i whole, workgroups take consecutive runs of rows, nothing
    is reduced. ``work`` holds aiter's six plan buffers; the reduce maps are
    left as they are, as aiter leaves them for a plan without splits."""
    work_indptr = work["work_indptr"]
    info = work["work_info_set"]
    reduce_indptr = work["reduce_indptr"]
    assert info.dtype == torch.int32 and info.shape[-1] == 8
    num_cu = work_indptr.numel() - 1
    span = max(rows, num_cu + 1, reduce_indptr.numel())
    _plan_whole_rows[(-(-span // BLOCK),)](
        info,
        work_indptr,
        reduce_indptr,
        work["work_meta_data"].view(torch.int64),
        kv_indptr,
        work_indptr.data_ptr(),
        info.data_ptr(),
        rows,
        rows_per_cu,
        num_cu,
        reduce_indptr.numel(),
        BLOCK=BLOCK,
    )
