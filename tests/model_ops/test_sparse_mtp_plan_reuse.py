# SPDX-License-Identifier: MIT
"""ATOM_MLA_SPARSE_MTP_PLAN_REUSE: the target's sparse MTP verify template stays
in its buffers while the step's template geometry repeats (TemplatePlans).

The CPU tests run the production builder and layer bodies on the stubs of
test_sparse_mla_decode_plan; the GPU tests run aiter's planner and decode.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.model_ops.test_sparse_mla_decode_plan import (  # noqa: F401 -- fixture
    WORK,
    runtime,
    selection,
    setup,
)

ROOT = Path(__file__).parents[2]
AITER_MLA = "atom/model_ops/attentions/aiter_mla.py"
ATTENTION_MLA = "atom/model_ops/attention_mla.py"
GPU = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def step(builder, md, tokens):
    """A decode step's sparse MTP plan, published on ``md`` as finish_decode does."""
    md.__dict__.update(builder._set_mla_persistent_worker_buffers_sparse_mtp(tokens))
    return md.sparse_mtp_dcp_decode_plan


@pytest.fixture
def reuse(runtime):  # noqa: F811
    runtime["envs"].ATOM_MLA_SPARSE_MTP_PLAN_REUSE = True
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    return runtime


def test_off_plans_every_step(runtime):  # noqa: F811
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    builder, md, _ = setup(runtime, 16, 8)
    for _ in range(3):
        assert step(builder, md, 16) is not None
    assert len(runtime["calls"]) == 4
    assert not runtime["sparse_mtp_templates"]._held


@pytest.mark.parametrize("head_split", [False, True])
def test_repeated_geometry_keeps_the_template(reuse, head_split):
    reuse["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = head_split
    builder, md, _ = setup(reuse, 16, 8)
    plan = md.sparse_mtp_dcp_decode_plan
    assert plan[4] == (4 if head_split else 1)
    for _ in range(3):
        assert step(builder, md, 16) == plan
        assert int(md.sparse_mtp_work_indptr[-1]) == plan[0]
    assert len(reuse["calls"]) == 1


# (tokens, selection seed): the batch's composition changes every step.
STEPS = ((16, 0), (16, 1), (16, 2), (12, 3), (12, 4), (16, 5))


def decode_steps(ns, steps):
    """Each step's outputs and plan after two layers over their own selections."""
    builder, md, layer = setup(ns, steps[0][0], 8)
    torch.manual_seed(3)
    q, kv = torch.randn(18, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    seen = []
    for n, (tokens, seed) in enumerate(steps):
        if n:
            step(builder, md, tokens)
        for layer_seed in (seed, seed + 7):
            selection(md, tokens, layer_seed)
            out, lse = layer._forward_decode(q[:tokens], kv, md, return_lse=True)
        seen.append(
            (
                out.clone(),
                lse.clone(),
                md.sparse_mtp_work_info_set.clone(),
                md.sparse_mtp_work_indptr.clone(),
            )
        )
    return seen


def test_reused_template_decodes_as_a_fresh_one(runtime):  # noqa: F811
    runtime["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    fresh = decode_steps(runtime, STEPS)
    assert len(runtime["calls"]) == len(STEPS)
    runtime["envs"].ATOM_MLA_SPARSE_MTP_PLAN_REUSE = True
    reused = decode_steps(runtime, STEPS)
    # The first step and the two row changes.
    assert len(runtime["calls"]) == len(STEPS) + 3
    for want, got in zip(fresh, reused, strict=True):
        for a, b in zip(want, got, strict=True):
            assert torch.equal(a, b)


def test_row_change_plans_again(reuse):
    builder, md, _ = setup(reuse, 16, 8)
    for tokens in (12, 12, 16, 16, 7):
        plan = step(builder, md, tokens)
        assert plan[0] == tokens * 4 and int(md.sparse_mtp_work_indptr[-1]) == plan[0]
    assert len(reuse["calls"]) == 4


def test_head_split_switch_plans_again(reuse):
    builder, md, _ = setup(reuse, 16, 8)
    assert md.sparse_mtp_dcp_decode_plan[:2] == (64, 16)
    reuse["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = False
    assert step(builder, md, 16)[:2] == (16, 64)
    assert step(builder, md, 16)[:2] == (16, 64)
    reuse["envs"].ATOM_DMLA_MLA_HEAD_SPLIT = True
    assert step(builder, md, 16)[:2] == (64, 16)
    assert len(reuse["calls"]) == 3
    assert reuse["calls"][-1][0] == 16


def test_switching_reuse_off_plans_and_drops_the_template(reuse):
    builder, md, _ = setup(reuse, 16, 8)
    reuse["envs"].ATOM_MLA_SPARSE_MTP_PLAN_REUSE = False
    step(builder, md, 16)
    assert not reuse["sparse_mtp_templates"]._held
    reuse["envs"].ATOM_MLA_SPARSE_MTP_PLAN_REUSE = True
    step(builder, md, 16)
    step(builder, md, 16)
    assert len(reuse["calls"]) == 3


@pytest.mark.parametrize("change", ["unsupported", "no_rebuild"])
def test_other_plans_plan_every_step_and_drop_the_template(reuse, change):
    """A plan over the step's key ranges, which a layer may also rebuild into
    the step's buffers, never stands in for the template."""
    builder, md, _ = setup(reuse, 16, 8)
    support = reuse["get_mla_decode_shape_support"]
    if change == "unsupported":
        reuse["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
            native=False, supported=True
        )
    else:
        builder.sparse_dcp_metadata_rebuild = False
    assert step(builder, md, 16) is None
    assert step(builder, md, 16) is None
    assert not reuse["sparse_mtp_templates"]._held
    reuse["get_mla_decode_shape_support"] = support
    builder.sparse_dcp_metadata_rebuild = True
    assert step(builder, md, 16) is not None
    assert len(reuse["calls"]) == 4


def test_forgotten_template_is_planned_again(reuse):
    builder, md, _ = setup(reuse, 16, 8)
    step(builder, md, 16)
    reuse["sparse_mtp_templates"].forget()
    step(builder, md, 16)
    step(builder, md, 16)
    assert len(reuse["calls"]) == 2


@pytest.mark.parametrize(
    "name",
    [f"sparse_mtp_{name}" for name in WORK]
    + ["sparse_cu_seqlens_q", "sparse_kv_last_page_lens"],
)
def test_reallocated_buffer_plans_again(reuse, name):
    builder, md, _ = setup(reuse, 16, 8)
    var = builder.model_runner.forward_vars
    if isinstance(var[name], SimpleNamespace):
        var[name] = SimpleNamespace(gpu=var[name].gpu.clone())
    else:
        var[name] = var[name].clone()
    step(builder, md, 16)
    step(builder, md, 16)
    assert len(reuse["calls"]) == 2


def function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    (node,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    return node


def names(node):
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


@pytest.mark.parametrize(
    "path,kernel,plan,count",
    [
        pytest.param(
            "atom/model_ops/sparse_mla_work_plan.py",
            "_patch_work_kv_ranges",
            "work_info",
            "items",
            id="patch_work_kv_ranges",
        ),
        pytest.param(
            "atom/model_ops/triton_mla_decode_qtail.py",
            "_mla_decode_qtail_kernel",
            "work_info_ptr",
            "work_items_ptr",
            id="q tail plan patch",
        ),
    ],
)
def test_layer_patches_read_batch_idx_and_write_only_the_kv_range(
    path, kernel, plan, count
):
    """TemplatePlans: a patch reads an item's batch_idx (MlaWorkInfo field 0)
    and the item count, and writes kv_start and kv_end (fields 4, 5) alone,
    so patching a reused template equals patching a fresh one. A patch that
    writes another field, or reads one it writes, must make the builder plan
    again (forget sparse_mtp_templates) before this list changes."""
    fn = function(path, kernel)
    (address,) = [
        n for n in ast.walk(fn) if isinstance(n, ast.Assign) and plan in names(n.value)
    ]
    assert names(fn).count(plan) == 1
    assert ast.unparse(address.value) == f"{plan} + item * 8"
    (target,) = address.targets
    info = target.id
    accesses, covered = [], {info: 0, count: 0}
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        for name in covered:
            if name in names(call.args[0]):
                accesses.append((ast.unparse(call.func), ast.unparse(call.args[0])))
                covered[name] += names(call.args[0]).count(name)
    for name, n in covered.items():
        assert names(fn).count(name) == n + (name == info), f"{name} escapes its loads"
    assert sorted(accesses) == sorted(
        [
            ("tl.load", count),
            ("tl.load", info),
            ("tl.store", f"{info} + 4"),
            ("tl.store", f"{info} + 5"),
        ]
    )


WRITERS = {
    "get_mla_metadata_v1",
    "decode_update_mla_metadata_v1",
    "patch_work_kv_ranges",
    "plan_whole_rows",
    "mla_decode_qtail",
}


def plan_writers(path):
    """(innermost function, callee) of every MLA work-plan write in ``path``."""
    found, stack = set(), []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node):
            func = node.func
            callee = getattr(func, "attr", getattr(func, "id", None))
            if callee in WRITERS and (
                callee != "mla_decode_qtail"
                or any(k.arg == "work_plan" for k in node.keywords)
            ):
                found.add((stack[-1], callee))
            self.generic_visit(node)

    Visitor().visit(ast.parse((ROOT / path).read_text(encoding="utf-8")))
    return found


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            AITER_MLA,
            {
                ("_aiter_sparse_prefill_plan", "get_mla_metadata_v1"),
                (
                    "_set_mla_persistent_worker_buffers_sparse_mtp",
                    "get_mla_metadata_v1",
                ),
                ("set_mla_persistent_worker_buffers", "decode_update_mla_metadata_v1"),
                ("set_mla_persistent_worker_buffers", "get_mla_metadata_v1"),
                ("_sparse_prefill_work_plan", "plan_whole_rows"),
                ("_dsa_row_parallel_work_plan", "plan_whole_rows"),
            },
        ),
        (
            ATTENTION_MLA,
            {
                ("_forward_prefill_mla", "patch_work_kv_ranges"),
                ("_rebuild_sparse_dcp_persistent_metadata", "get_mla_metadata_v1"),
                ("_sparse_dcp_decode_work_buffers", "patch_work_kv_ranges"),
                ("forward_impl", "mla_decode_qtail"),
            },
        ),
    ],
)
def test_mla_work_plan_writers_are_the_known_ones(path, expected):
    """A new or moved work-plan writer here: if it can write the sparse_mtp_
    buffers, it must forget sparse_mtp_templates (TemplatePlans) or meet the
    patch contract above. Then update this list."""
    assert plan_writers(path) == expected


def neighbours(fn, callee):
    """The statements before and after the one calling ``callee``."""
    for node in ast.walk(fn):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if isinstance(stmt, (ast.Assign, ast.Expr)) and callee in ast.unparse(
                    stmt
                ):
                    after = body[i + 1] if i + 1 < len(body) else None
                    return body[i - 1] if i else None, stmt, after
    raise AssertionError(f"{fn.name} does not call {callee}")


def test_capture_prefill_and_graph_width_wiring():
    forget = "sparse_mtp_templates.forget()"
    sparse_mtp = "_set_mla_persistent_worker_buffers_sparse_mtp"
    # A capture may record the planner without running it.
    before, _, after = neighbours(
        function(AITER_MLA, "build_for_cudagraph_capture"), sparse_mtp
    )
    assert [s and ast.unparse(s) for s in (before, after)] == [forget, forget]
    prefill = function(AITER_MLA, "_finish_prefill_batch")
    assert ast.unparse(prefill.body[0]) == forget
    # The template's rows are the step's real tokens, whatever graph width
    # the step replays at.
    decode = function(AITER_MLA, "finish_decode")
    _, call, _ = neighbours(decode, sparse_mtp)
    assert [ast.unparse(a) for a in call.value.args] == ["scheduled_tokens"]
    assert "scheduled_tokens = batch.total_tokens_num_decode" in [
        ast.unparse(n) for n in ast.walk(decode) if isinstance(n, ast.Assign)
    ]


@pytest.fixture
def templates():
    from atom.model_ops.sparse_mla_work_plan import sparse_mtp_templates

    sparse_mtp_templates.forget()
    yield sparse_mtp_templates
    sparse_mtp_templates.forget()


def same_bits(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    return a == b


# Kimi-K3.1 DCP8 verify at five tokens a request: B7 and B8 plan six 16-head
# rows a token (210, 240 rows); B9's 270 exceed 256 and plan 45 whole rows.
K31_STEPS = (
    (35, 0),
    (35, 1),
    (35, 2),
    (40, 3),
    (40, 4),
    (35, 5),
    (45, 6),
    (45, 7),
    (35, 8),
    (35, 9),
)
K31_HEADS, K31_MAX_Q = 96, 5


@GPU
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_reused_template_is_bit_identical_on_gpu(dtype, monkeypatch, templates):
    """Plan buffers, outputs and LSEs of every step, eager and on graph replay,
    with and without reuse; the decode reads the template's items alone."""
    from aiter import get_mla_metadata_info_v1

    from atom.model_ops import attention_mla
    from atom.model_ops.attentions import aiter_mla

    monkeypatch.setenv("ATOM_DMLA_MLA_HEAD_SPLIT", "1")
    sizes = None
    for query_length in range(1, K31_MAX_Q + 1):
        more = get_mla_metadata_info_v1(
            64, query_length, K31_HEADS, dtype, dtype, is_sparse=True, fast_mode=True
        )
        sizes = more if sizes is None else [
            (
                tuple(map(max, shape, other)) if isinstance(shape, tuple)
                else max(shape, other),
                dt,
            )
            for (shape, dt), (other, _) in zip(sizes, more, strict=True)
        ]  # fmt: skip
    ns = {
        "Builder": aiter_mla.AiterMLAMetadataBuilder,
        "Attention": attention_mla.MLAAttention,
    }
    plans = []
    planner = aiter_mla.get_mla_metadata_v1

    def counted(*a, **kw):
        plans.append(a[0].numel() - 1)
        return planner(*a, **kw)

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", counted)
    torch.manual_seed(31)
    q = (torch.randn(48, K31_HEADS, 576, device="cuda") / 4).to(dtype)
    kv = (torch.randn(97, 1, 576, device="cuda") / 4).to(dtype)

    def run(reuse):
        monkeypatch.setenv("ATOM_MLA_SPARSE_MTP_PLAN_REUSE", "1" if reuse else "0")
        del plans[:]
        builder, md, layer = setup(
            ns, K31_STEPS[0][0], K31_MAX_Q, dtype, "cuda", sizes, heads=K31_HEADS
        )
        layer.scale = 576**-0.5
        # Both arms plan step 0 again on the same bytes, so whole buffers,
        # unplanned tails included, compare.
        for name in WORK:
            var = builder.model_runner.forward_vars[f"sparse_mtp_{name}"]
            var.view(torch.uint8).fill_(0xA5)
        templates.forget()
        step(builder, md, K31_STEPS[0][0])
        seen = []

        def record(out, lse):
            torch.cuda.synchronize()
            # A row without keys leaves o and the LSE unwritten (the DCP merge
            # drops it by its owned count).
            live = md.dcp_sparse_kv_indptr.diff() > 0
            rows = live.numel()
            meta, *buffers = (getattr(md, f"sparse_mtp_{name}") for name in WORK)
            # Addresses of this arm's own buffers, so checked, not compared.
            assert meta[:2].view(torch.int64).tolist() == [
                md.sparse_mtp_work_indptr.data_ptr(),
                md.sparse_mtp_work_info_set.data_ptr(),
            ]
            seen.append([live, out[:rows][live], lse[:rows][live]])
            seen[-1] += [md.sparse_mtp_dcp_decode_plan]
            seen[-1] += [buffer.clone() for buffer in buffers]

        for n, (tokens, seed) in enumerate(K31_STEPS):
            if n:
                step(builder, md, tokens)
            for layer_seed in (seed, seed + 50):
                selection(md, tokens, layer_seed, "cuda")
                out, lse = layer._forward_decode(q[:tokens], kv, md, return_lse=True)
            record(out, lse)
        tokens = K31_STEPS[0][0]
        step(builder, md, tokens)
        selection(md, tokens, 0, "cuda")
        layer._forward_decode(q[:tokens], kv, md, return_lse=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = layer._forward_decode(q[:tokens], kv, md, return_lse=True)
        for seed in range(3):
            step(builder, md, tokens)
            fresh = SimpleNamespace()
            selection(fresh, tokens, 100 + seed, "cuda")
            md.dcp_sparse_kv_indptr.copy_(fresh.dcp_sparse_kv_indptr)
            md.sparse_kv_indices.copy_(fresh.sparse_kv_indices)
            graph.replay()
            record(out, lse)
        return seen, list(plans)

    fresh, fresh_plans = run(False)
    reused, reused_plans = run(True)
    assert all(row[0].any() and row[3] is not None for row in fresh)
    assert [row[3][4] for row in fresh[: len(K31_STEPS)]] == [
        6 if tokens * 6 <= 256 else 1 for tokens, _ in K31_STEPS
    ]
    # Off: every builder call. On: setup, step 0 again, each row change, and
    # none of the graph's.
    assert len(fresh_plans) == len(K31_STEPS) + 5
    assert reused_plans == [210, 210, 240, 210, 45, 210]
    for want, got in zip(fresh, reused, strict=True):
        for a, b in zip(want, got, strict=True):
            assert same_bits(a, b)


def _kvptr(tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    counts = torch.randint(0, 300, (tokens,), device="cuda", generator=gen)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).int() + seed


def _stale_plan(tokens, groups):
    gen = torch.Generator(device="cuda").manual_seed(tokens * groups)
    rows = tokens * groups
    info = torch.randint(
        -9, 9, (rows + 300, 8), dtype=torch.int32, device="cuda", generator=gen
    )
    info[:rows, 0] = torch.randperm(rows, device="cuda", generator=gen).int()
    work_indptr = torch.tensor([0, rows // 2, rows], dtype=torch.int32, device="cuda")
    return info, work_indptr


@GPU
@pytest.mark.parametrize("tokens,groups", [(40, 6), (45, 1)])
def test_patching_a_patched_template_equals_patching_a_fresh_one(tokens, groups):
    from atom.model_ops.sparse_mla_work_plan import patch_work_kv_ranges

    rows = tokens * groups
    fresh, work_indptr = _stale_plan(tokens, groups)
    want, reused = fresh.clone(), fresh.clone()
    for seed in (1, 2, 3):
        patch_work_kv_ranges(reused, work_indptr, _kvptr(tokens, seed), rows, groups)
    patch_work_kv_ranges(want, work_indptr, _kvptr(tokens, 3), rows, groups)
    assert torch.equal(reused, want)
    kept = [0, 1, 2, 3, 6, 7]
    assert torch.equal(want[:, kept], fresh[:, kept])
    assert torch.equal(want[rows:], fresh[rows:])


@GPU
@pytest.mark.parametrize("tokens,groups", [(40, 6), (45, 1)])
def test_qtail_patching_a_patched_template_equals_patching_a_fresh_one(tokens, groups):
    from aiter import dtypes

    from atom.model_ops.triton_mla_decode_qtail import mla_decode_qtail

    heads, nope, pe, lora = K31_HEADS, 128, 64, 512
    gen = torch.Generator(device="cuda").manual_seed(tokens)
    q = torch.randn(
        tokens, heads, nope + pe, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    w_k = 0.05 * torch.randn(
        heads, lora, nope, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    k_nope = torch.randn(tokens, lora, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.randn(tokens, pe, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(4 * tokens, device="cuda", generator=gen)[:tokens]
    one = torch.ones(1, device="cuda")
    rows = tokens * groups

    def run(info, kvptr):
        q_out = torch.zeros(tokens, heads, lora + pe, device="cuda", dtype=dtypes.fp8)
        cache = torch.zeros(4 * tokens, 1, lora + pe, device="cuda", dtype=dtypes.fp8)
        mla_decode_qtail(
            q[..., :nope], q[..., nope:], w_k, q_out, k_nope, k_pe, cache, slots,
            one, one, work_plan=(info, work_indptr, kvptr, rows, groups),
        )  # fmt: skip

    fresh, work_indptr = _stale_plan(tokens, groups)
    want, reused = fresh.clone(), fresh.clone()
    for seed in (1, 2, 3):
        run(reused, _kvptr(tokens, seed))
    run(want, _kvptr(tokens, 3))
    assert torch.equal(reused, want)
    kept = [0, 1, 2, 3, 6, 7]
    assert torch.equal(want[:, kept], fresh[:, kept])
    assert torch.equal(want[rows:], fresh[rows:])
