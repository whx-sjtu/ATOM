# SPDX-License-Identifier: MIT
"""Sparse decode's builder/consumer contract; CPU stubs stop at GPU operations."""

from __future__ import annotations

import ast
import base64
import contextlib
import json
import logging
import math
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.sparse_mla_work_plan import TemplatePlans

WORK = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
BUILD = {
    "_persistent_work_buffer_sizes",
    "_sparse_mtp_work_buffer_sizes",
    "_head_split_work_buffer_sizes",
    "_widest_work_buffers",
    "_work_buffers_hold",
    "_active_work_indptr",
    "_sparse_dcp_decode_plan",
    "_sparse_dcp_decode_template",
    "set_mla_persistent_worker_buffers",
    "_set_mla_persistent_worker_buffers_sparse_mtp",
}
DECODE = {
    "_forward_decode",
    "_sparse_dcp_decode_reusable_plan",
    "_decode_qtail_work_plan",
    "_sparse_dcp_decode_work_buffers",
    "_rebuild_sparse_dcp_persistent_metadata",
}


def methods(path, names, namespace):
    tree = ast.parse((Path(__file__).parents[2] / path).read_text(encoding="utf-8"))
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in nodes} == names
    exec(  # noqa: S102 -- real production bodies without importing GPU dependencies
        compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace
    )
    return {name: namespace[name] for name in names}


@pytest.fixture
def runtime():
    ns = {
        "math": math,
        "torch": torch,
        "PERSISTENT_WORK_BUFFERS": WORK,
        "_MLA_SPLIT_BUDGET_AUTO": -1,
        "_mla_seg_meta_kwargs": dict,
        "envs": SimpleNamespace(
            ATOM_MLA_PAGE_SIZE=1,
            ATOM_DMLA_MLA_HEAD_SPLIT=False,
            ATOM_DMLA_MLA_NO_REDUCE=False,
            ATOM_DMLA_QTAIL_PLAN_PATCH=False,
            ATOM_MLA_SPARSE_MTP_PLAN_REUSE=False,
        ),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "logger": logging.getLogger(__name__),
    }
    ns["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=True, supported=True
    )
    ns["get_mla_decode_fwd_max_splits"] = lambda *a: 256
    ns["Builder"] = type(
        "Builder", (), methods("atom/model_ops/attentions/aiter_mla.py", BUILD, ns)
    )
    ns["Attention"] = type(
        "Attention", (), methods("atom/model_ops/attention_mla.py", DECODE, ns)
    )
    calls = []

    def sizes(rows, max_q, heads, dtype_q, dtype_kv, *, is_sparse, fast_mode):
        assert fast_mode is True and (is_sparse or max_q == 1)
        groups = ns["get_mla_decode_fwd_max_splits"](heads, max_q, dtype_q, dtype_kv)
        rows *= max_q if is_sparse else 1
        return [
            (2, torch.int64),
            (groups + 1, torch.int32),
            ((rows + groups, 8), torch.int32),
            (rows + 1, torch.int32),
            ((rows, 2), torch.int32),
            (rows + 2 * groups, torch.int32),
        ]

    def planner(qptr, kvptr, last, heads, _, causal, wmd, info, wi, ri, rfm, rpm, **kw):
        rows = qptr.numel() - 1
        groups = ns["get_mla_decode_fwd_max_splits"](
            heads, kw["max_seqlen_qo"], kw["dtype_q"], kw["dtype_kv"]
        )
        assert wi.numel() >= groups + 1, "planner work-indptr allocation is too small"
        calls.append((heads, kvptr.clone(), kw, info, (wmd, wi, info, ri, rfm, rpm)))
        info.zero_()
        info[:rows, 0] = torch.arange(rows)
        info[:rows, 1] = -1
        info[:rows, 2] = torch.arange(rows)
        info[:rows, 3] = torch.arange(1, rows + 1)
        info[:rows, 4] = kvptr[:-1] - kvptr[0]
        info[:rows, 5] = kvptr[1:] - kvptr[0]
        wi[: groups + 1].copy_(torch.arange(groups + 1).clamp_max(rows))
        for tensor in (wmd, ri, rfm, rpm):
            tensor.fill_(len(calls))

    patches = []

    def patch(info, wi, ptr, rows, groups=1):
        patches.append((rows, groups))
        count = int(wi[-1])
        ids = info[:count, 0].long()
        assert count <= rows and (ids < rows).all()
        ids //= groups
        info[:count, 4] = ptr[ids] - ptr[0]
        info[:count, 5] = ptr[ids + 1] - ptr[0]

    def launch(q, kv, out, qptr, ptr, indices, last, *a, **kw):
        info, wi = kw["work_info_set"], kw["work_indptr"]
        lse = torch.full(q.shape[:2], -torch.inf)
        out.zero_()
        for row, _, start, end, lo, hi, *_ in info[: int(wi[-1])].tolist():
            assert start == row and end == row + 1
            if hi > lo:
                keys = kv.float().flatten(0, 2)[indices[lo:hi].long()]
                scores = q[row].float() @ keys.T * kw["sm_scale"]
                out[row] = (scores.softmax(-1) @ keys[:, : out.shape[-1]]).to(out.dtype)
                lse[row] = scores.logsumexp(-1)
        return out, lse

    reduce_skips = []

    def skipping_reduce(skip):
        reduce_skips.append(skip)
        return contextlib.nullcontext()

    ns.update(
        get_mla_metadata_v1=planner,
        get_mla_metadata_info_v1=sizes,
        decode_update_mla_metadata_v1=planner,
        patch_work_kv_ranges=patch,
        mla_decode_fwd=launch,
        _skipping_mla_decode_reduce=skipping_reduce,
        sparse_mtp_templates=TemplatePlans(),
        calls=calls,
        patches=patches,
        reduce_skips=reduce_skips,
    )
    return ns


def setup(
    ns,
    rows=16,
    max_q=1,
    dtype=torch.bfloat16,
    device="cpu",
    sizes=None,
    heads=64,
    allocation_q=1,
):
    builder = ns["Builder"].__new__(ns["Builder"])
    builder.__dict__.update(
        is_sparse=True,
        sparse_dcp_metadata_rebuild=True,
        block_size=1,
        persistent_num_heads=heads,
        max_bs=rows,
        padded_num_attention_heads=16,
        dtype_q=dtype,
        dtype_kv=dtype,
        index_topk_out=8,
        kv_shuffled=False,
    )
    if sizes is None:
        sizes = builder._persistent_work_buffer_sizes(allocation_q)
    var = {
        prefix + name: torch.empty(shape, dtype=dt, device=device)
        for prefix in ("", "sparse_mtp_")
        for name, (shape, dt) in zip(WORK, sizes)
    }
    ptr = torch.arange(rows + 1, dtype=torch.int32, device=device)
    # Static per-token buffers, sized like the runner's max_num_batched_tokens.
    tokens = max(rows, 256)
    for name, value in {
        "cu_seqlens_q": ptr,
        "sparse_cu_seqlens_q": torch.arange(
            tokens + 1, dtype=torch.int32, device=device
        ),
        "kv_indptr": ptr * 16,
        "sparse_kv_indptr": ptr * 8,
        "kv_last_page_lens": torch.ones(rows, dtype=torch.int32, device=device),
        "sparse_kv_last_page_lens": torch.ones(
            tokens, dtype=torch.int32, device=device
        ),
    }.items():
        var[name] = SimpleNamespace(gpu=value)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    md = SimpleNamespace(
        max_seqlen_q=max_q,
        causal=True,
        kv_indices=None,
        **{
            name: value.gpu
            for name, value in var.items()
            if isinstance(value, SimpleNamespace)
        },
    )
    result = (
        builder._set_mla_persistent_worker_buffers_sparse_mtp(rows)
        if max_q > 1
        else builder.set_mla_persistent_worker_buffers(rows, 1)
    )
    md.__dict__.update(result)
    layer = ns["Attention"].__new__(ns["Attention"])
    if isinstance(layer, torch.nn.Module):
        torch.nn.Module.__init__(layer)
    layer.__dict__.update(
        num_heads=heads,
        kv_lora_rank=512 if device == "cuda" else 4,
        dtype=torch.bfloat16,
        use_seg_mla=False,
        use_triton_mla=False,
        kv_shuffled=False,
        is_sparse_mla=True,
        dcp_world_size=8,
        dcp_head_pad=0,
        sparse_dcp_metadata_rebuild=True,
        owns_sparse_indexer=True,
        scale=0.25,
        _q_scale=torch.ones(1, device=device),
        _k_scale=torch.ones(1, device=device),
        _head_group_indptr=torch.arange(257, dtype=torch.int32, device=device),
        _head_group_last_page_lens=torch.ones(256, dtype=torch.int32, device=device),
    )
    layer._pad_decode_query_heads = lambda x: x
    layer._restore_decode_query_heads = lambda x, _: x
    layer.decode_is_persistent = lambda: True
    return builder, md, layer


def selection(md, rows, seed, device="cpu"):
    lengths = (torch.arange(rows, device=device) + seed) % 5
    lengths[0] = 0
    md.dcp_sparse_kv_indptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).int()
    md.sparse_kv_indices = (
        (torch.arange(rows * 8, device=device) * 7 + seed) % 97
    ).int()
    return lengths


def native_fp8_probability_scale(support, q_dtype, kv_dtype):
    if (
        q_dtype == kv_dtype == torch.float8_e4m3fn
        and getattr(support, "arch", None) == "gfx950"
        and getattr(support, "backend", None) == "asm"
        and getattr(support, "asm_status", None) == "ok"
        and getattr(support, "native", None) is True
        and getattr(support, "supported", None) is True
        and getattr(support, "kernel_num_heads", None) == 64
        # AITER's query returns the generated map key: arch + symbol.
        and getattr(support, "asm_kernel", None)
        == "gfx950_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE"
    ):
        return 448.0
    return None


def reference(q, kv, md, rank, scale, *, fp8_probability_scale=None):
    if fp8_probability_scale is not None:
        assert q.dtype == kv.dtype == torch.float8_e4m3fn
    out = torch.zeros((*q.shape[:2], rank), device=q.device)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device)
    ptr = md.dcp_sparse_kv_indptr.tolist()
    for row, (lo, hi) in enumerate(pairwise(ptr)):
        if hi > lo:
            keys = kv.float()[md.sparse_kv_indices[lo:hi].long()].reshape(
                -1, q.shape[-1]
            )
            scores = torch.einsum("hd,kd->hk", q[row].float(), keys) * scale
            lse[row] = torch.logsumexp(scores, -1)
            if fp8_probability_scale is None:
                out[row] = torch.einsum(
                    "hk,kd->hd", torch.exp(scores - lse[row, :, None]), keys[:, :rank]
                )
            else:
                # ROCm/aiter#6422: round scaled P before PV, retaining the FP32 Z.
                weights = torch.exp(scores - scores.amax(-1, keepdim=True))
                denominator = weights.sum(-1, keepdim=True)
                weights = (
                    (weights * fp8_probability_scale).to(torch.float8_e4m3fn).float()
                )
                value = torch.einsum("hk,kd->hd", weights, keys[:, :rank])
                out[row] = (
                    (value / denominator / fp8_probability_scale).bfloat16().float()
                )
    return out, lse


def test_fp8_reference_matches_saved_native_vectors():
    data = json.loads(
        Path(__file__).with_name("sparse_mla_fp8_contract.json").read_text()
    )

    def tensor(name, dtype, width):
        raw = bytearray(base64.b64decode(data[name]))
        return (
            torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(-1, 1, width)
        )

    q = tensor("q_fp8", torch.float8_e4m3fn, 576)
    kv = tensor("kv_fp8", torch.float8_e4m3fn, 576)
    native = tensor("native_bf16", torch.bfloat16, 512).float()
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor(data["kv_indptr"], dtype=torch.int32),
        sparse_kv_indices=torch.tensor(data["kv_indices"], dtype=torch.int32),
    )
    original, old_lse = reference(q, kv, md, 512, data["scale"])
    failures = (~torch.isclose(native, original, atol=0.015, rtol=0.03)).nonzero()
    assert failures.tolist() == [
        [row, 0, dim] for row, dim in enumerate(data["failed_dims"])
    ]
    torch.testing.assert_close(
        original[torch.arange(3), 0, data["failed_dims"]],
        torch.tensor(data["gpu_reference"]),
        atol=2e-7,
        rtol=0,
    )
    corrected, lse = reference(q, kv, md, 512, data["scale"], fp8_probability_scale=448)
    torch.testing.assert_close(native, corrected, atol=0.015, rtol=0.03)
    torch.testing.assert_close(lse, old_lse, atol=0, rtol=0)
    assert torch.equal(corrected, corrected.bfloat16().float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_probability_reference_keeps_default_attention(dtype):
    q = torch.tensor([[[0, 1]]], dtype=dtype)
    kv = torch.tensor([[[1, 0]], [[1, -0.5]]], dtype=dtype)
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor([0, 2]),
        sparse_kv_indices=torch.tensor([0, 1]),
    )
    out, lse = reference(q, kv, md, 1, 1)
    torch.testing.assert_close(out, torch.ones(1, 1, 1))
    torch.testing.assert_close(lse, torch.tensor([[math.log1p(math.exp(-0.5))]]))


def test_probability_reference_uses_unrounded_denominator():
    q = torch.tensor([[[0, 1]]], dtype=torch.float8_e4m3fn)
    kv = torch.tensor([[[1, 0]], [[1, -0.5]]], dtype=torch.float8_e4m3fn)
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor([0, 2]),
        sparse_kv_indices=torch.tensor([0, 1]),
    )
    out, lse = reference(q, kv, md, 1, 1, fp8_probability_scale=448)
    # 448*exp(-0.5) rounds to 256; the denominator still uses exp(-0.5).
    expected = torch.tensor((448 + 256) / 448 / (1 + math.exp(-0.5))).bfloat16().float()
    torch.testing.assert_close(out, expected.expand_as(out), atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.tensor([[math.log1p(math.exp(-0.5))]]))


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"arch": "gfx942"},
        {"backend": "hk"},
        {"backend": "opus"},
        {"backend": "flydsl_ps1"},
        {"native": False},
        {"supported": None},
        {"asm_status": "unknown"},
        {"kernel_num_heads": 16},
        {"asm_kernel": "_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE"},
        {"asm_kernel": None},
    ],
)
def test_native_probability_scale_is_route_specific(change):
    values = {
        "arch": "gfx950",
        "backend": "asm",
        "asm_status": "ok",
        "native": True,
        "supported": True,
        "kernel_num_heads": 64,
        "asm_kernel": "gfx950_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE",
    }
    values.update(change)
    support = SimpleNamespace(**values)
    fp8 = torch.float8_e4m3fn
    assert native_fp8_probability_scale(support, fp8, fp8) == (None if change else 448)
    assert native_fp8_probability_scale(support, torch.bfloat16, fp8) is None
    assert native_fp8_probability_scale(support, fp8, torch.bfloat16) is None
    assert native_fp8_probability_scale(SimpleNamespace(), fp8, fp8) is None


@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (8, 256)])
def test_three_owners_and_shared_layers_reuse_one_builder_plan(runtime, max_q, rows):
    _, md, layer = setup(runtime, rows, max_q)
    call = runtime["calls"][0]
    assert call[0] == 64 and torch.equal(call[1], torch.arange(rows + 1))
    assert call[2]["kv_granularity"] == 8 and call[2]["max_split_per_batch"] == 1
    torch.manual_seed(7)
    q, kv = torch.randn(rows + 2, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    for owner in range(3):
        selection(md, rows, owner)
        for owns in (True, False):
            layer.owns_sparse_indexer = owns
            out, lse = layer._forward_decode(q, kv, md, return_lse=True)
            expected, expected_lse = reference(q[:rows], kv, md, 4, layer.scale)
            torch.testing.assert_close(
                out[:rows], expected.bfloat16(), atol=0.016, rtol=0.01
            )
            torch.testing.assert_close(lse[:rows], expected_lse)
    assert len(runtime["calls"]) == 1


@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 64), (1, 64)])
def test_head_split_plans_16_head_rows_that_match_whole_rows(runtime, max_q, rows):
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    _, md, layer = setup(runtime, rows, max_q)
    plan = getattr(
        md, "sparse_mtp_dcp_decode_plan" if max_q > 1 else "sparse_dcp_decode_plan"
    )
    call = runtime["calls"][0]
    assert plan[:2] == (rows * 4, 16) and plan[4] == 4
    assert call[0] == 16 and torch.equal(call[1], torch.arange(rows * 4 + 1))
    torch.manual_seed(7)
    q, kv = torch.randn(rows + 2, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    for owner in range(3):
        selection(md, rows, owner)
        for owns in (True, False):
            layer.owns_sparse_indexer = owns
            out, lse = layer._forward_decode(q, kv, md, return_lse=True)
            assert out.shape == (rows + 2, 64, 4) and lse.shape == (rows + 2, 64)
            expected, expected_lse = reference(q[:rows], kv, md, 4, layer.scale)
            torch.testing.assert_close(
                out[:rows], expected.bfloat16(), atol=0.016, rtol=0.01
            )
            torch.testing.assert_close(lse[:rows], expected_lse)
    assert len(runtime["calls"]) == 1


@pytest.mark.parametrize("rows,heads", [(65, 64), (16, 40), (16, 16)])
def test_head_split_keeps_whole_rows_it_cannot_split(runtime, rows, heads):
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    _, md, _ = setup(runtime, rows, heads=heads)
    assert md.sparse_dcp_decode_plan == (rows, heads, torch.bfloat16, torch.bfloat16, 1)
    assert runtime["calls"][0][0] == heads


def aiter_gfx950_fp8(ns):
    """aiter's sizing and metadata stores on gfx950 for FP8 q and KV at 16 and
    96 heads, as runs/DMLA/hsfix/hs_probe.py measured them on MI355X: 256
    workers, one reduce tile a q_len=1 row (sparse sizing folds the query
    length into the rows), one work item and one reduce_indptr entry a planned
    row. The stores fail where the kernel's would land past a buffer."""
    workers = 256
    stores = []

    def sizes(rows, max_q, heads, dtype_q, dtype_kv, *, is_sparse, fast_mode):
        assert heads in (16, 96) and is_sparse and fast_mode is True
        tiles = rows * max_q
        return [
            (2, torch.int64),
            (workers + 1, torch.int32),
            ((tiles + workers - 1, 8), torch.int32),
            (tiles + 1, torch.int32),
            ((tiles, 2), torch.int32),
            (min(tiles + workers - 1, 2 * (workers - 1)), torch.int32),
        ]

    def planner(qptr, kvptr, last, heads, _, causal, wmd, info, wi, ri, rfm, rpm, **kw):
        rows = qptr.numel() - 1
        stores.append((heads, rows))
        for name, tensor, count in (
            ("work_indptr", wi, workers + 1),
            ("work_info_set", info, rows * 8),
            ("reduce_indptr", ri, rows + 1),
        ):
            assert tensor.numel() >= count, (
                f"{heads}-head plan of {rows} rows stores {count} {name} "
                f"entries into {tensor.numel()}"
            )

    ns.update(
        get_mla_metadata_info_v1=sizes,
        get_mla_metadata_v1=planner,
        decode_update_mla_metadata_v1=planner,
        get_mla_decode_fwd_max_splits=lambda *a: workers,
    )
    return stores


def head_split_builder(ns, mns, k, mtp_sizes=None):
    """A Kimi-K3.1 DCP8 builder (96 gathered heads, FP8) at ``max_num_seqs``
    ``mns`` and K=``k``, its work buffers as the builder sizes them."""
    builder = ns["Builder"].__new__(ns["Builder"])
    builder.__dict__.update(
        is_sparse=True,
        sparse_dcp_metadata_rebuild=True,
        block_size=1,
        persistent_num_heads=96,
        max_bs=mns,
        padded_num_attention_heads=12,
        dtype_q=torch.float8_e4m3fn,
        dtype_kv=torch.float8_e4m3fn,
        index_topk_out=2048,
        kv_shuffled=False,
    )
    if mtp_sizes is None:
        mtp_sizes = builder._sparse_mtp_work_buffer_sizes(k + 1)
    var = {
        prefix + name: torch.empty(shape, dtype=dt)
        for prefix, sizes in (
            ("", builder._persistent_work_buffer_sizes(k + 1)),
            ("sparse_mtp_", mtp_sizes),
        )
        for name, (shape, dt) in zip(WORK, sizes)
    }
    tokens = 16384  # --max-num-batched-tokens
    ptr = torch.arange(mns + 1, dtype=torch.int32)
    for name, value in {
        "cu_seqlens_q": ptr,
        "sparse_cu_seqlens_q": torch.arange(tokens + 1, dtype=torch.int32),
        "kv_indptr": ptr * 64,
        "sparse_kv_indptr": torch.arange(tokens + 1, dtype=torch.int32) * 8,
        "kv_last_page_lens": torch.ones(mns, dtype=torch.int32),
        "sparse_kv_last_page_lens": torch.ones(tokens, dtype=torch.int32),
    }.items():
        var[name] = SimpleNamespace(gpu=value)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    return builder


@pytest.mark.parametrize("k", [0, 3, 4])
@pytest.mark.parametrize("mns", [8, 10, 12, 16])
def test_head_split_templates_stay_inside_their_work_buffers(runtime, mns, k):
    # 25 Nov C16 daily, mns 10 / K4: the bs-8 capture split 40 rows into 240
    # and stored 241 reduce_indptr entries into 51, then faulted.
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    stores = aiter_gfx950_fp8(runtime)
    builder = head_split_builder(runtime, mns, k)
    for bs in range(1, mns + 1):
        if k:
            rows = bs * (k + 1)
            plan = builder._set_mla_persistent_worker_buffers_sparse_mtp(rows)
            assert plan["sparse_mtp_dcp_decode_plan"][4] == (6 if rows <= 42 else 1)
        plan = builder.set_mla_persistent_worker_buffers(bs, 1)
        assert plan["sparse_dcp_decode_plan"][4] == (6 if bs <= 42 else 1)
    assert (16, 6 * mns) in stores
    if k:
        assert (16, 6 * (k + 1) * min(mns, 42 // (k + 1))) in stores
        held = builder.model_runner.forward_vars["sparse_mtp_reduce_indptr"].numel()
        assert held == min(6 * mns * (k + 1), 256) + 1


@pytest.mark.parametrize("mns,k", [(10, 4), (8, 4), (16, 4), (8, 3)])
def test_head_split_keeps_whole_rows_its_work_buffers_cannot_hold(
    runtime, caplog, mns, k
):
    # INT3b's sparse_mtp_* sizing: whole 96-head rows only.
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    aiter_gfx950_fp8(runtime)
    whole = runtime["get_mla_metadata_info_v1"](
        mns * (k + 1), 1, 96, None, None, is_sparse=True, fast_mode=True
    )
    builder = head_split_builder(runtime, mns, k, mtp_sizes=whole)
    with caplog.at_level(logging.WARNING):
        for bs in range(1, mns + 1):
            rows = bs * (k + 1)
            plan = builder._set_mla_persistent_worker_buffers_sparse_mtp(rows)
            split = rows <= 42 and 6 * bs <= mns
            assert plan["sparse_mtp_dcp_decode_plan"][4] == (6 if split else 1)
    assert "stay whole" in caplog.text


def test_head_split_off_keeps_the_whole_row_allocation(runtime):
    aiter_gfx950_fp8(runtime)
    builder = head_split_builder(runtime, 10, 4)
    sizes = runtime["get_mla_metadata_info_v1"]
    assert list(builder._sparse_mtp_work_buffer_sizes(5)) == [
        tuple(s) for s in sizes(50, 1, 96, None, None, is_sparse=True, fast_mode=True)
    ]
    assert list(builder._persistent_work_buffer_sizes(5)) == [
        tuple(s) for s in sizes(10, 5, 96, None, None, is_sparse=True, fast_mode=True)
    ]


def test_head_split_plan_falls_back_without_the_layers_row_tensors(runtime):
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    _, md, layer = setup(runtime, 16)
    del layer._head_group_indptr, layer._head_group_last_page_lens
    selection(md, 16, 2)
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    assert len(runtime["calls"]) == 2 and runtime["calls"][-1][0] == 64
    expected, expected_lse = reference(q, kv, md, 4, layer.scale)
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.016, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse)


@pytest.mark.parametrize("no_reduce", [False, True])
@pytest.mark.parametrize("head_split", [False, True])
@pytest.mark.parametrize("max_q", [1, 8])
def test_no_reduce_covers_only_the_builders_template(
    runtime, no_reduce, head_split, max_q
):
    runtime["envs"].ATOM_DMLA_MLA_NO_REDUCE = no_reduce
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = head_split
    _, md, layer = setup(runtime, 16, max_q)
    selection(md, 16, 1)
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    layer._forward_decode(q, kv, md, return_lse=True)
    # A rebuilt plan may split rows over workgroups and keeps the reduce.
    layer._forward_decode(q[:, :32], kv, md, return_lse=True)
    assert len(runtime["calls"]) == 2
    assert runtime["reduce_skips"] == [no_reduce, False]


@pytest.mark.parametrize("head_split", [False, True])
@pytest.mark.parametrize("max_q", [1, 8])
def test_qtail_plan_patch_stands_in_for_the_layers_patch(runtime, head_split, max_q):
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = head_split
    runtime["envs"].ATOM_DMLA_QTAIL_PLAN_PATCH = True
    _, md, layer = setup(runtime, 16, max_q)
    prefix = "sparse_mtp_" if max_q > 1 else ""
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    patches = runtime["patches"]
    for owner in range(3):
        selection(md, 16, owner)
        plan = layer._decode_qtail_work_plan(md, q, kv)
        assert plan[0] is getattr(md, prefix + "work_info_set")
        assert plan[1] is getattr(md, prefix + "work_indptr")
        assert torch.equal(plan[2], md.dcp_sparse_kv_indptr)
        assert plan[3:] == ((64, 4) if head_split else (16, 1))
        # What mla_decode_qtail stores given the plan.
        runtime["patch_work_kv_ranges"](*plan)
        made = len(patches)
        out, lse = layer._forward_decode(q, kv, md, return_lse=True)
        assert len(patches) == made, "the decode patched what its q tail did"
        expected, expected_lse = reference(q, kv, md, 4, layer.scale)
        torch.testing.assert_close(out, expected.bfloat16(), atol=0.016, rtol=0.01)
        torch.testing.assert_close(lse, expected_lse)
        # A decode its q tail did not patch patches for itself.
        layer._forward_decode(q, kv, md, return_lse=True)
        assert len(patches) == made + 1
    assert len(runtime["calls"]) == 1


@pytest.mark.parametrize("change", ["off", "heads", "head_pad", "no_template"])
def test_qtail_plan_patch_only_where_the_layer_patches(runtime, change):
    runtime["envs"].ATOM_DMLA_QTAIL_PLAN_PATCH = change != "off"
    _, md, layer = setup(runtime, 16)
    selection(md, 16, 1)
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    if change == "heads":
        q = q[:, :32]
    elif change == "head_pad":
        layer.dcp_head_pad = 32
    elif change == "no_template":
        md.sparse_dcp_decode_plan = None
    layer._decode_kv_ranges_patched = "stale"
    assert layer._decode_qtail_work_plan(md, q, kv) is None
    assert layer._decode_kv_ranges_patched is None


def test_whole_row_plan_is_not_reused_for_a_multiple_of_its_heads(runtime):
    _, md, layer = setup(runtime, 16, heads=16)
    selection(md, 16, 1)
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    assert len(runtime["calls"]) == 2 and runtime["calls"][-1][0] == 64
    expected, expected_lse = reference(q, kv, md, 4, layer.scale)
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.016, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse)


@pytest.mark.parametrize(
    "change",
    ["folded", "unknown", "unsupported", "q_dtype", "kv_dtype", "heads", "rows"],
)
@pytest.mark.parametrize("max_q", [1, 8])
def test_fallback_preserves_native_template(runtime, change, max_q):
    _, md, layer = setup(runtime, 16, max_q)
    selection(md, 16, 1)
    prefix = "sparse_mtp_" if max_q > 1 else ""
    saved = [getattr(md, prefix + name).clone() for name in WORK]
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    if change in ("folded", "unknown", "unsupported"):
        runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
            native=change != "folded",
            supported={"folded": True, "unknown": None, "unsupported": False}[change],
        )
    elif change == "heads":
        q = q[:, :32]
    elif change == "rows":
        q = q[:12]
    elif change == "q_dtype":
        q = q.float()
    else:
        kv = kv.float()
    layer._forward_decode(q, kv, md, return_lse=True)
    assert len(runtime["calls"]) == 2
    assert runtime["calls"][-1][0] == q.shape[1]
    assert runtime["calls"][-1][3] is not getattr(md, prefix + "work_info_set")
    for name, value in zip(WORK, saved):
        torch.testing.assert_close(getattr(md, prefix + name), value)
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=True, supported=True
    )
    selection(md, 16, 3)
    layer.owns_sparse_indexer = False
    q, kv = torch.randn(16, 64, 8).bfloat16(), kv.bfloat16()
    out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    expected, expected_lse = reference(q, kv, md, 4, layer.scale)
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.016, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse)
    assert len(runtime["calls"]) == 2


@pytest.mark.parametrize(
    "rows,groups,capacity",
    [(0, 256, 257), (257, 304, 305), (65, 64, 65), (16, 512, 257)],
)
def test_unqualified_capacity_keeps_layer_rebuild(runtime, rows, groups, capacity):
    builder, _, _ = setup(runtime, max(1, rows))
    runtime["get_mla_decode_fwd_max_splits"] = lambda *a: groups
    assert builder._sparse_dcp_decode_plan(rows, 64, torch.empty(capacity)) is None


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
def test_unqualified_builder_keeps_existing_geometry(runtime, native, supported):
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=native, supported=supported
    )
    _, md, layer = setup(runtime)
    assert md.sparse_dcp_decode_plan is None
    assert runtime["calls"][0][2]["kv_granularity"] == 16
    for owner in range(3):
        selection(md, 16, owner)
        layer._forward_decode(
            torch.randn(16, 64, 8).bfloat16(),
            torch.randn(97, 1, 8).bfloat16(),
            md,
            return_lse=True,
        )
    assert len(runtime["calls"]) == 4
    assert all(call[3] is md.work_info_set for call in runtime["calls"])


def test_next_step_refreshes_rows_without_reusing_old_template(runtime):
    builder, _, _ = setup(runtime, 128, 8)
    next_step = builder._set_mla_persistent_worker_buffers_sparse_mtp(7)
    assert next_step["sparse_mtp_dcp_decode_plan"][0] == 7
    assert next_step["sparse_mtp_num_tokens"] == 7
    assert int(next_step["sparse_mtp_work_indptr"][-1]) == 7
    assert len(runtime["calls"]) == 2
    updated = builder.set_mla_persistent_worker_buffers(7, 1, only_update=True)
    assert updated["sparse_dcp_decode_plan"] is None
    assert runtime["calls"][-1][2]["kv_granularity"] == 16


def test_fallback_sizes_for_consumers_with_more_workgroups(runtime):
    builder, md, layer = setup(runtime)
    fp8 = torch.float8_e4m3fn
    runtime["get_mla_decode_fwd_max_splits"] = lambda heads, _, qdt, kdt: (
        512 if heads == 64 and qdt == kdt == fp8 else 256
    )
    builder.persistent_num_heads = 96
    builder.dtype_q = builder.dtype_kv = fp8
    md.__dict__.update(builder.set_mla_persistent_worker_buffers(16, 1))
    selection(md, 16, 2)
    original = md.work_info_set.clone()
    buffers, groups, whole_row_items = layer._sparse_dcp_decode_work_buffers(
        md,
        torch.empty(16, 64, 8, dtype=fp8),
        torch.empty(97, 1, 8, dtype=fp8),
        md.cu_seqlens_q,
        md.dcp_sparse_kv_indptr,
        md.sparse_kv_last_page_lens,
        "",
    )
    assert groups == 1 and not whole_row_items
    assert buffers[1].numel() == 513 and md.work_indptr.numel() == 257
    expected = runtime["get_mla_metadata_info_v1"](
        16, 1, 64, fp8, fp8, is_sparse=False, fast_mode=True
    )
    for buffer, (shape, dtype) in zip(buffers, expected, strict=True):
        assert tuple(buffer.shape) == (shape if isinstance(shape, tuple) else (shape,))
        assert buffer.dtype == dtype
    torch.testing.assert_close(md.work_info_set, original)


@pytest.mark.parametrize("heads,allocation_q", [(16, 4), (64, 8)])
def test_spec_allocation_covers_both_occupancies_and_publishes_active_prefix(
    runtime, heads, allocation_q
):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    dtype = torch.float8_e4m3fn
    builder, md, layer = setup(
        runtime, heads=heads, allocation_q=allocation_q, dtype=dtype
    )
    backing = builder.model_runner.forward_vars["work_indptr"]
    assert backing.numel() == 513
    backing.fill_(-12345)
    md.__dict__.update(builder.set_mla_persistent_worker_buffers(16, 1))
    groups = runtime["get_mla_decode_fwd_max_splits"](heads, 1)
    assert md.sparse_dcp_decode_plan is not None
    assert md.work_indptr.numel() == groups + 1
    assert int(md.work_indptr[-1]) == 16
    assert torch.all(backing[groups + 1 :] == -12345)
    selection(md, 16, 2)
    layer._forward_decode(
        torch.randn(16, heads, 8).to(dtype),
        torch.randn(97, 1, 8).to(dtype),
        md,
        return_lse=True,
    )
    assert len(runtime["calls"]) == 2


@pytest.mark.parametrize("heads,actual_q", [(64, 1), (64, 8), (16, 4), (16, 8)])
def test_every_planner_publishes_active_workers(runtime, heads, actual_q):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    fp8 = torch.float8_e4m3fn
    builder, _, _ = setup(runtime, heads=heads, allocation_q=8, dtype=fp8)
    var = builder.model_runner.forward_vars
    # Every configured width must fit all buffers, including an interior peak.
    for q_len in range(1, 9):
        sizes = runtime["get_mla_metadata_info_v1"](
            16, q_len, heads, fp8, fp8, is_sparse=True, fast_mode=True
        )
        for name, (shape, dtype) in zip(WORK, sizes, strict=True):
            dims = shape if isinstance(shape, tuple) else (shape,)
            assert all(a >= b for a, b in zip(var[name].shape, dims, strict=True))
            assert var[name].dtype == dtype
    var["cu_seqlens_q"].gpu = torch.arange(17, dtype=torch.int32) * actual_q
    backing = var["work_indptr"]
    backing.fill_(-12345)
    result = builder.set_mla_persistent_worker_buffers(16, actual_q)
    groups = runtime["get_mla_decode_fwd_max_splits"](heads, actual_q)
    assert (
        result["work_indptr"].numel() == groups + 1
    ), "unwritten capacity launches workers"
    assert torch.all(backing[groups + 1 :] == -12345)
    for name, buffer in zip(WORK, runtime["calls"][-1][4], strict=True):
        assert buffer is result[name] and buffer.data_ptr() == var[name].data_ptr()
    assert (result["sparse_dcp_decode_plan"] is not None) == (actual_q == 1)

    def update(*args, **kwargs):
        # The incremental updater reads the terminal entry of the original plan.
        assert kwargs["max_seqlen_qo"] == actual_q
        assert args[8].numel() == groups + 1 and int(args[8][-1]) == 16

    runtime["decode_update_mla_metadata_v1"] = update
    updated = builder.set_mla_persistent_worker_buffers(16, actual_q, only_update=True)
    assert updated["work_indptr"].numel() == groups + 1


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
def test_unqualified_mtp_still_publishes_active_workers(runtime, native, supported):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=native, supported=supported
    )
    builder, _, _ = setup(runtime, heads=16, allocation_q=4, max_q=8)
    backing = builder.model_runner.forward_vars["sparse_mtp_work_indptr"]
    backing.fill_(-12345)
    result = builder._set_mla_persistent_worker_buffers_sparse_mtp(16)
    assert result["sparse_mtp_dcp_decode_plan"] is None
    assert result["sparse_mtp_work_indptr"].numel() == 257
    assert torch.all(backing[257:] == -12345)


def test_decode_ubatch_publishes_the_plan(runtime):
    builder, _, _ = setup(runtime)
    var = builder.model_runner.forward_vars
    var.update({"ub0_" + name: value for name, value in tuple(var.items())})
    path = Path(__file__).parents[2] / "atom/model_ops/attentions/aiter_mla.py"
    method = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "finish_ubatch"
    )
    cls = ast.ClassDef(
        name="Child",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    ns = {"Parent": type("Parent", (), {"finish_ubatch": lambda *a: None})}
    exec(  # noqa: S102 -- actual caller, with only its parent boundary stubbed
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    child = ns["Child"]()
    child.model_runner = builder.model_runner
    child.set_mla_persistent_worker_buffers = builder.set_mla_persistent_worker_buffers
    child._decode_index_rows = lambda *a, **kw: None
    child._mtp_index_rows = lambda *a: None
    md = SimpleNamespace()
    child.finish_ubatch(0, SimpleNamespace(running_bs=16, running_tokens=16), md, 1)
    assert md.sparse_dcp_decode_plan is not None
    assert md.work_indptr.data_ptr() == var["ub0_work_indptr"].data_ptr()


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (8, 256), (1, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("head_split", [False, True])
def test_native_decode_and_graph_against_direct_attention(
    max_q, rows, dtype, head_split, monkeypatch
):
    from aiter import get_mla_decode_shape_support

    from atom.model_ops import attention_mla
    from atom.model_ops.attentions import aiter_mla

    monkeypatch.setenv("ATOM_DMLA_MLA_HEAD_SPLIT", "1" if head_split else "0")
    ns = {
        "Builder": aiter_mla.AiterMLAMetadataBuilder,
        "Attention": attention_mla.MLAAttention,
    }
    builds = []
    original = aiter_mla.get_mla_metadata_v1

    def build(*a, **kw):
        builds.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", build)
    _, md, layer = setup(ns, rows, max_q, dtype, "cuda")
    plan = getattr(
        md, "sparse_mtp_dcp_decode_plan" if max_q > 1 else "sparse_dcp_decode_plan"
    )
    if plan is None:
        pytest.skip("runtime does not qualify this native plan geometry")
    prefix = "sparse_mtp_" if max_q > 1 else ""
    planned, heads, *_, groups = plan
    assert groups == (4 if head_split and rows * 4 <= 256 else 1)
    assert (planned, heads) == (rows * groups, 64 // groups)
    wi = getattr(md, prefix + "work_indptr")
    assert int(wi[-1]) == planned and int((wi[1:] - wi[:-1]).max()) == 1
    probability_scale = None
    if dtype == torch.float8_e4m3fn:
        support = get_mla_decode_shape_support(
            heads, 1, dtype, dtype, persistent=True, return_lse=True, page_size=1
        )
        probability_scale = native_fp8_probability_scale(support, dtype, dtype)
        if probability_scale is not None:
            assert (getattr(md, prefix + "work_info_set")[:planned, 1] == -1).all()
    monkeypatch.setattr(
        attention_mla,
        "get_mla_metadata_v1",
        lambda *a, **kw: pytest.fail("layer rebuilt native plan"),
    )
    torch.manual_seed(92)
    q = (torch.randn(rows + 2, 64, 576, device="cuda") / 4).to(dtype)
    kv = (torch.randn(97, 1, 576, device="cuda") / 4).to(dtype)
    layer.scale = 576**-0.5
    selection(md, rows, 0, "cuda")
    layer._forward_decode(q, kv, md, return_lse=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    # Without the FP8 probability emulation the reference is exact attention.
    atol = (
        0.025 if dtype == torch.float8_e4m3fn and probability_scale is None else 0.015
    )
    for owner in range(3):
        next_md = SimpleNamespace()
        selection(next_md, rows, owner, "cuda")
        md.dcp_sparse_kv_indptr.copy_(next_md.dcp_sparse_kv_indptr)
        md.sparse_kv_indices.copy_(next_md.sparse_kv_indices)
        graph.replay()
        expected, expected_lse = reference(
            q[:rows], kv, md, 512, layer.scale, fp8_probability_scale=probability_scale
        )
        # Raw empty-rank partials may be NaN; the DCP merge uses owned counts
        # to drop them. Check every live row, including those following empties.
        live = md.dcp_sparse_kv_indptr.diff() > 0
        torch.testing.assert_close(
            out[:rows][live].float(), expected[live], atol=atol, rtol=0.03
        )
        torch.testing.assert_close(
            lse[:rows][live], expected_lse[live], atol=0.02, rtol=0.01
        )
    assert len(builds) == 1


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (1, 64)])
@pytest.mark.parametrize("heads", [64, 96])
@pytest.mark.parametrize("head_split", [False, True])
def test_native_decode_without_reduce_is_bit_identical(
    max_q, rows, heads, head_split, monkeypatch
):
    from atom.model_ops import attention_mla
    from atom.model_ops.attentions import aiter_mla

    dtype = torch.float8_e4m3fn
    monkeypatch.setenv("ATOM_DMLA_MLA_HEAD_SPLIT", "1" if head_split else "0")
    ns = {
        "Builder": aiter_mla.AiterMLAMetadataBuilder,
        "Attention": attention_mla.MLAAttention,
    }
    _, md, layer = setup(ns, rows, max_q, dtype, "cuda", heads=heads)
    plan_name = "sparse_mtp_dcp_decode_plan" if max_q > 1 else "sparse_dcp_decode_plan"
    if getattr(md, plan_name) is None:
        pytest.skip("runtime does not qualify this native plan geometry")
    groups = heads // 16 if head_split and rows * heads // 16 <= 256 else 1
    assert getattr(md, plan_name)[4] == groups
    reduces = []
    reduce = attention_mla._aiter_mla._mla_decode_reduce_v1_dispatch

    def counted(*a, **kw):
        reduces.append(1)
        return reduce(*a, **kw)

    monkeypatch.setattr(
        attention_mla._aiter_mla, "_mla_decode_reduce_v1_dispatch", counted
    )
    empty = torch.empty

    def poisoned(*a, **kw):
        out = empty(*a, **kw)
        return out.fill_(math.nan) if out.is_floating_point() else out

    torch.manual_seed(5)
    q = (torch.randn(rows + 2, heads, 576, device="cuda") / 4).to(dtype)
    kv = (torch.randn(97, 1, 576, device="cuda") / 4).to(dtype)
    layer.scale = 576**-0.5
    selection(md, rows, 1, "cuda")
    outs = {}
    for no_reduce in (False, True):
        monkeypatch.setenv("ATOM_DMLA_MLA_NO_REDUCE", "1" if no_reduce else "0")
        # Anything the decode leaves unwritten reads NaN.
        monkeypatch.setattr(torch, "empty", poisoned)
        outs[no_reduce] = layer._forward_decode(q, kv, md, return_lse=True)
        monkeypatch.setattr(torch, "empty", empty)
        torch.cuda.synchronize()
    assert len(reduces) == 1
    live = md.dcp_sparse_kv_indptr.diff() > 0
    assert torch.isfinite(outs[True][0][:rows][live].float()).all()
    assert torch.isfinite(outs[True][1][:rows][live]).all()
    for got, want in zip(outs[True], outs[False], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    assert len(reduces) == 1
    out[:rows][live] = math.nan
    lse[:rows][live] = math.nan
    graph.replay()
    torch.testing.assert_close(
        out[:rows][live], outs[False][0][:rows][live], rtol=0, atol=0
    )
    torch.testing.assert_close(
        lse[:rows][live], outs[False][1][:rows][live], rtol=0, atol=0
    )
