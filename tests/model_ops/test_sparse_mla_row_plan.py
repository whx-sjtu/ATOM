# SPDX-License-Identifier: MIT
"""ATOM_MLA_SPARSE_PREFILL_ROW_PLAN: the whole-row step plan, against aiter's."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

HEADS, TOPK, SENTINEL = 96, 2048, 0x5A5A5A5A


def _buffers(dtype, capacity):
    from aiter import get_mla_metadata_info_v1

    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )

    sizes = get_mla_metadata_info_v1(
        capacity, 1, HEADS, dtype, dtype, is_sparse=True, fast_mode=True
    )
    work = {}
    for name, (shape, dt) in zip(PERSISTENT_WORK_BUFFERS, sizes):
        buf = torch.empty(shape, dtype=dt, device="cuda")
        buf.view(torch.uint8).fill_(0x5A)
        work[name] = buf
    return work


def _aiter_plan(kv_indptr, work, dtype):
    from aiter import get_mla_metadata_v1

    rows = kv_indptr.numel() - 1
    get_mla_metadata_v1(
        torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
        kv_indptr,
        torch.ones(rows, dtype=torch.int32, device="cuda"),
        HEADS,
        1,
        True,
        work["work_meta_data"],
        work["work_info_set"],
        work["work_indptr"],
        work["reduce_indptr"],
        work["reduce_final_map"],
        work["reduce_partial_map"],
        page_size=1,
        dtype_q=dtype,
        dtype_kv=dtype,
        kv_granularity=TOPK,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=1,
        max_split_per_batch=-1,
    )


def _lens(rows, seed):
    gen = np.random.default_rng(seed)
    lens = gen.integers(1, TOPK + 1, rows)
    lens[: min(rows, 2048)] = np.minimum(np.arange(1, min(rows, 2048) + 1), TOPK)
    lens[::7] = TOPK
    return lens


def _assert_same_plan(mine, ref):
    for name in mine:
        if name == "work_meta_data":
            got = mine[name].view(torch.int64)[:2].tolist()
            want = ref[name].view(torch.int64)[:2].tolist()
            assert got == [
                mine["work_indptr"].data_ptr(),
                mine["work_info_set"].data_ptr(),
            ]
            assert want == [
                ref["work_indptr"].data_ptr(),
                ref["work_info_set"].data_ptr(),
            ]
            assert torch.equal(
                mine[name].view(torch.int64)[2:], ref[name].view(torch.int64)[2:]
            )
        else:
            assert torch.equal(mine[name], ref[name]), name


@pytest.mark.parametrize("cache_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize(
    "rows", [1, 10, 11, 12, 64, 255, 256, 257, 300, 2048, 4097, 8192, 12345, 16384]
)
def test_whole_row_plan_is_aiters_plan(cache_dtype, rows):
    from aiter import dtypes

    from atom.model_ops.sparse_mla_work_plan import (
        plan_whole_rows,
        whole_row_plan_rows_per_cu,
    )

    dtype = dtypes.fp8 if cache_dtype == "fp8" else torch.bfloat16
    host = np.zeros(rows + 1, dtype=np.int32)
    host[1:] = np.cumsum(_lens(rows, rows))
    kv_indptr = torch.from_numpy(host).cuda()
    for capacity in (rows, 16384):
        ref = _buffers(dtype, capacity)
        _aiter_plan(kv_indptr, ref, dtype)
        mine = _buffers(dtype, capacity)
        per_cu = whole_row_plan_rows_per_cu(host, TOPK, mine["work_indptr"].numel() - 1)
        if per_cu is None:
            # Too few rows for the plan to use every workgroup.
            assert rows <= 10
            continue
        plan_whole_rows(kv_indptr, rows, per_cu, mine)
        _assert_same_plan(mine, ref)
        assert int(mine["work_indptr"][-1]) == rows


def test_rows_without_keys_or_past_the_granularity_keep_aiters_planner():
    from atom.model_ops.sparse_mla_work_plan import whole_row_plan_rows_per_cu

    lens = _lens(4096, 1)
    for bad in (0, TOPK + 1):
        lens2 = lens.copy()
        lens2[100] = bad
        host = np.concatenate([[0], np.cumsum(lens2)]).astype(np.int32)
        assert whole_row_plan_rows_per_cu(host, TOPK, 256) is None


def test_builder_writes_the_row_plan_and_checks_it_once(monkeypatch):
    from aiter import dtypes

    from atom.model_ops.attentions import aiter_mla
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder

    dtype, rows = dtypes.fp8, 16384
    host = np.zeros(rows + 1, dtype=np.int32)
    host[1:] = np.cumsum(_lens(rows, 3))
    gpu = torch.from_numpy(host).cuda()
    work = _buffers(dtype, rows)
    builder = SimpleNamespace(
        dcp_sparse_prefill_nonps=False,
        block_size=1,
        dtype_q=dtype,
        dtype_kv=dtype,
        _sparse_prefill_kv_granularity=TOPK,
        model_runner=SimpleNamespace(
            forward_vars={
                "sparse_kv_indptr": SimpleNamespace(np=host, gpu=gpu),
                **{f"sparse_prefill_{n}": b for n, b in work.items()},
            }
        ),
    )
    aiter_calls = []
    planner = aiter_mla.get_mla_metadata_v1

    def counted(*args, **kwargs):
        aiter_calls.append(1)
        return planner(*args, **kwargs)

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", counted)
    monkeypatch.setenv("ATOM_MLA_SPARSE_PREFILL_ROW_PLAN", "1")
    last = torch.ones(rows, dtype=torch.int32, device="cuda")
    meta = SimpleNamespace(
        sparse_cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
        sparse_kv_indptr=gpu,
    )

    ref = _buffers(dtype, rows)
    _aiter_plan(gpu, ref, dtype)
    for step in range(2):
        AiterMLAMetadataBuilder._sparse_prefill_work_plan(
            builder, meta, "", HEADS, last
        )
        # The first step plans with aiter too, on copies, to check the plan.
        assert len(aiter_calls) == 1
        assert builder._whole_row_plan_checked and builder._whole_row_plan_ok
        _assert_same_plan(work, ref)
        assert meta.sparse_prefill_work_info_set is work["work_info_set"]

    # A step the indptr was not built on the host for keeps aiter's planner.
    meta.sparse_kv_indptr = gpu.clone()
    AiterMLAMetadataBuilder._sparse_prefill_work_plan(builder, meta, "", HEADS, last)
    assert len(aiter_calls) == 2
