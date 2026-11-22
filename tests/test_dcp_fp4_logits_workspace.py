# SPDX-License-Identifier: MIT
"""Real CPU planning/storage and source-bound GPU consumer boundaries."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import (
    LogitsWorkspace,
    decode_band_rows,
    prefill_workspace_capacity,
    qsplit_rows,
)
from atom.utils import envs

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "atom/model_ops/attentions/aiter_mla.py"
RUNNER = ROOT / "atom/model_engine/model_runner.py"


def load(path, names, scope):
    nodes = [
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    exec(  # noqa: S102 - unchanged source at controlled CPU boundaries
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope
    )
    return [scope[name] for name in names]


def builder(monkeypatch, enabled=True, **changes):
    monkeypatch.setenv("ATOM_DCP_FP4_PREFILL_LOGITS_WORKSPACE", "1" if enabled else "0")
    config = NS(
        index_cache_layout="mfma",
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        max_model_len=1024,
        compilation_config=NS(static_forward_context={}),
    )
    vars(config).update(changes)
    scope = {
        "envs": envs,
        "MFMA": "mfma",
        "prefill_workspace_capacity": prefill_workspace_capacity,
    }
    (method,) = load(BUILDER, ["_dcp_fp4_prefill_workspace_capacity"], scope)
    result = NS(
        model_runner=NS(config=config),
        _indexer_fp4=True,
        dcp_world_size=8,
        max_num_batched_tokens=64,
        device="cpu",
        is_sparse=True,
        index_logits_workspace=None,
        sparse_dcp_metadata_rebuild=False,
    )
    result._dcp_fp4_prefill_workspace_capacity = lambda: method(result)
    result._publishes_decode_index_rows = lambda: False
    result._concat_prefill_scored = lambda: False
    return result


def allocate_from_constructor(obj):
    init = next(
        n
        for n in ast.walk(ast.parse(BUILDER.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    block = next(
        n
        for n in init.body
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Name)
            and x.func.id == "LogitsWorkspace"
            for x in ast.walk(n)
        )
    )
    scope = {
        "self": obj,
        "config": obj.model_runner.config,
        "LogitsWorkspace": LogitsWorkspace,
        "logger": logging.getLogger(__name__),
    }
    exec(  # noqa: S102 - unchanged source at controlled CPU boundaries
        compile(ast.Module(body=[block], type_ignores=[]), str(BUILDER), "exec"), scope
    )
    return obj.index_logits_workspace


def test_default_off_and_constructor_owns_one_live_allocation(monkeypatch):
    monkeypatch.delenv("ATOM_DCP_FP4_PREFILL_LOGITS_WORKSPACE", raising=False)
    assert not envs.ATOM_DCP_FP4_PREFILL_LOGITS_WORKSPACE
    assert allocate_from_constructor(builder(monkeypatch, False)) is None
    obj = builder(monkeypatch)
    workspace = allocate_from_constructor(obj)
    assert workspace.capacity == 8 * 1024 and workspace.nbytes == 8 * 1024 * 4
    assert workspace is obj.index_logits_workspace
    assert workspace.logits(8, 1024).dtype == torch.float32


@pytest.mark.parametrize("budget_mb", [1, 4])
def test_tiny_budget_constructor_keeps_the_legacy_dynamic_path(monkeypatch, budget_mb):
    monkeypatch.setenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", str(budget_mb))
    assert (
        allocate_from_constructor(builder(monkeypatch, max_model_len=1 << 20)) is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"index_cache_layout": "preshuffle"},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"enable_tbo": True},
        {"enable_tbo_decode": True},
    ],
)
def test_unsupported_config_keeps_dynamic_path(monkeypatch, changes):
    assert allocate_from_constructor(builder(monkeypatch, **changes)) is None


@pytest.mark.parametrize("fp4,world", [(False, 8), (True, 1), (True, 4), (True, 16)])
def test_other_cache_or_dcp_does_not_add_a_workspace(monkeypatch, fp4, world):
    obj = builder(monkeypatch)
    obj._indexer_fp4, obj.dcp_world_size = fp4, world
    assert obj._dcp_fp4_prefill_workspace_capacity() == 0


@pytest.mark.parametrize("mb", [0, 1, 4, 8, 64, 2048, 4096])
def test_real_band_planner_matches_every_original_context_transition(monkeypatch, mb):
    monkeypatch.setenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", str(mb))
    scope = {"decode_band_rows": decode_band_rows}
    _, bands = load(
        ROOT / "atom/model_ops/indexer/rows.py",
        ["_uniform_bands", "_plane_bands"],
        scope,
    )
    capacity = prefill_workspace_capacity(1024, 1 << 20)
    workspace = None
    if capacity:
        workspace = LogitsWorkspace.__new__(LogitsWorkspace)
        workspace._logits = NS(numel=lambda: capacity)  # allocation boundary only
    for count in [0, 1, 7, 8, 9, 255, 8191, 8192]:
        for rank in range(8):
            _, _, first, end = qsplit_rows(count, rank, 8)
            for width in [
                1,
                4096,
                8192,
                131071,
                131072,
                507904,
                524287,
                524288,
                524289,
                1048576,
            ]:
                assert bands(first, end, width, workspace) == bands(
                    first, end, width, None
                ), (mb, count, rank, width)


def test_real_scorer_reuses_one_plane_with_exact_views_and_results(monkeypatch):
    monkeypatch.setenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", "1")
    workspace = LogitsWorkspace(prefill_workspace_capacity(8, 65537), "cpu")
    addresses = []
    scope = {"torch": torch, "decode_band_rows": decode_band_rows}
    _, bands = load(
        ROOT / "atom/model_ops/indexer/rows.py",
        ["_uniform_bands", "_plane_bands"],
        scope,
    )

    def score(
        values, scales, weights, cache, scores, offsets, weight_scale, index, rows
    ):
        assert scores.dtype == torch.float32 and scores.stride() == (rows.width, 1)
        addresses.append(scores.data_ptr())
        scores.copy_(
            -(torch.arange(rows.width)[None, :] - values.reshape(-1, 1) * 7).abs()
        )

    def topk(scores, group, ends, out, count, stride0, stride1, k, stable):
        assert (stride0, stride1) == scores.stride() and count == len(out)
        out.copy_(scores.topk(k, dim=1).indices)

    scope.update(_fp4_band_logits=score, top_k_per_row_decode=topk)
    (consume,) = load(ROOT / "atom/model_ops/indexer/score.py", ["_score_paged"], scope)
    query = (torch.arange(64).view(64, 1, 1, 1), torch.zeros(64, 1))
    _, _, first, end = qsplit_rows(64, 2, 8)
    for width in [17, 65537, 31, 257]:
        outputs = []
        for held in [None, workspace]:
            rows = NS(
                workspace=held,
                bands=bands(first, end, width, held),
                width=width,
                next_n=1,
                packed_offsets=None,
                row_ends=torch.full((64,), width),
            )
            out = torch.full((64, 4), -1, dtype=torch.int32)
            start = len(addresses)
            consume(
                query, torch.ones(64, 1), NS(fp4=True), rows, 4, 1.0, False, None, out
            )
            if held is not None:
                assert set(addresses[start:]) == {workspace._logits.data_ptr()}
            outputs.append(out)
        assert torch.equal(*outputs)
    with pytest.raises(ValueError, match="exceeds its workspace"):
        workspace.logits(1, workspace.capacity + 1)


def test_ctor_precedes_profile_and_existing_budget_charges_live_bytes_once(monkeypatch):
    tree = ast.parse(RUNNER.read_text())
    runner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    init = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    created = next(
        n.lineno
        for n in ast.walk(init)
        if isinstance(n, ast.Attribute)
        and n.attr == "attn_metadata_builder"
        and isinstance(n.ctx, ast.Store)
    )
    warmup = next(
        n.lineno
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_maybe_warmup"
    )
    assert created < warmup
    budget = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_num_blocks"
    )
    stop = next(
        i
        for i, n in enumerate(budget.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "available_for_kv_budget"
            for t in n.targets
        )
    )
    held = allocate_from_constructor(builder(monkeypatch)).nbytes

    def available(live):
        base, total = 4 << 30, 16 << 30
        cuda = NS(
            mem_get_info=lambda: ((10 << 30) - live, total),
            memory_stats=lambda: {
                "allocated_bytes.all.peak": base,
                "allocated_bytes.all.current": base + live,
            },
            memory_reserved=lambda: base + live,
        )
        obj = NS(
            device="cpu",
            config=NS(hf_config=NS(head_dim=1), gpu_memory_utilization=0.9),
            _estimate_cudagraph_overhead=lambda: 0,
        )
        scope = {"self": obj, "torch": NS(cuda=cuda, set_default_device=lambda _: None)}
        exec(  # noqa: S102 - unchanged source at controlled CPU boundaries
            compile(
                ast.Module(body=budget.body[: stop + 1], type_ignores=[]),
                str(RUNNER),
                "exec",
            ),
            scope,
        )
        return scope["available_for_kv_budget"]

    assert available(0) - available(held) == held


def test_mono_dcp_abi_ignores_the_new_prefill_plane(monkeypatch):
    workspace = allocate_from_constructor(builder(monkeypatch))
    ptr = torch.zeros(2, dtype=torch.int32)
    md = NS(
        context_lens=ptr,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        block_tables=ptr.view(2, 1),
        batch_id_per_q_token=None,
        max_seqlen_q=1,
    )
    module = NS(
        dcp_sparse_kv_indptr_buffer=ptr,
        dcp_owned_counts_buffer=ptr,
        sparse_kv_indices_buffer=ptr,
        rope_dim=0,
        wq_b=NS(weight=ptr),
        k_norm=NS(weight=ptr, bias=ptr),
        k_cache=NS(kv_cache=[ptr], kv_cache_scale=ptr),
        _indexer_fp4=True,
    )
    scope = {
        "torch": torch,
        "get_forward_context": lambda: NS(
            context=NS(is_dummy_run=False), attn_metadata=md
        ),
        "kix": NS(ABI=NS(pack=lambda values: values)),
    }
    plane, launch = load(
        ROOT / "atom/mono/dsa_indexer/bind.py", ["logits_plane", "launch_args"], scope
    )
    ilog = plane(NS(index_logits_workspace=workspace))
    assert ilog.data_ptr() == workspace._logits.data_ptr()
    kw = torch.zeros(2, 4)
    args = launch(
        module, kw, ptr, kw, kw, False, ptr, ptr, ilog, NS(addresses=ptr), 0, 0, 1
    )
    assert args["ilog"] == 0 and args["plain"] == ptr.data_ptr()
