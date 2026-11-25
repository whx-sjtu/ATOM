# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stable per-row top-k of a wide prefill band through AITER's sampled
selector (`ATOM_DSA_PREFILL_TOPK_SAMPLED`).

The sampled selector reads a row about once -- a sampled threshold, one
filtering pass, an exact select of the candidates -- where the one-block
kernel reads a long row twice. With `tie_low` it keeps the stable emit's set:
the k largest, ties to the smaller column, NaN by sign. It writes them in no
promised order; the ids are then put in ascending order, -1 last, the order
the stable emit writes. One exception: the stable emit takes a row's negative
NaNs (as if above +inf), the sampled selector ranks them last. Index logits
are finite.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from aiter.ops.topk import top_k_per_row_prefill_sampled, topk_sampled_supports

# The smallest band it takes: below either bound the one-block kernel was as
# fast or faster (MI355X, k=2048, causal rows, the ascending pass included:
# 1018 rows over 72K columns, 256 over 152K), and the sampled plan follows
# the pitch, not where rows end. Also clear of the small-width path, which
# tie_low lacks.
MIN_ROWS = 384
MIN_WIDTH = 81920

# Above every column id, so a row's -1 padding sorts last.
_PAD = 2**31 - 1

_zero_starts: dict[torch.device, torch.Tensor] = {}


@triton.jit
def _ascending_ids_kernel(ids_ptr, stride, K: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, K)
    ids = tl.load(ids_ptr + row * stride + offs)
    ids = tl.sort(tl.where(ids < 0, PAD, ids))
    tl.store(ids_ptr + row * stride + offs, tl.where(ids == PAD, -1, ids))


@lru_cache(maxsize=1024)
def _supports(rows: int, pitch: int, k: int, device: int) -> bool:
    return bool(topk_sampled_supports(rows, pitch, k, device))


def takes(scores: torch.Tensor, k: int) -> bool:
    """Whether `stable_sampled_topk` serves this band: tall and wide, eager
    (its workspace is shared scratch), a shape the sampled kernels take."""
    rows, width = scores.shape
    return (
        rows >= MIN_ROWS
        and width >= MIN_WIDTH
        and not torch.cuda.is_current_stream_capturing()
        and _supports(rows, scores.stride(0), k, scores.get_device())
    )


def _starts(rows: int, device: torch.device) -> torch.Tensor:
    zeros = _zero_starts.get(device)
    if zeros is None or zeros.numel() < rows:
        zeros = _zero_starts[device] = torch.zeros(
            max(rows, 4096), dtype=torch.int32, device=device
        )
    return zeros[:rows]


def stable_sampled_topk(scores, row_ends, out, k):
    """`out[r]`: the k largest of `scores[r, :row_ends[r]]`, ties to the smaller
    column, ascending, -1 past a short row -- `top_k_per_row_decode(next_n=1,
    stable=True)`'s ids. `k` a power of two; `takes(scores, k)` first."""
    rows = scores.shape[0]
    top_k_per_row_prefill_sampled(
        scores,
        _starts(rows, scores.device),
        row_ends,
        out,
        None,
        rows,
        scores.stride(0),
        1,
        k=k,
        ragged=True,
        tie_low=True,
    )
    _ascending_ids_kernel[(rows,)](out, out.stride(0), K=k, PAD=_PAD, num_warps=4)
