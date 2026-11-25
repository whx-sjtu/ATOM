# SPDX-License-Identifier: MIT
"""Kimi's query layout travels from its constructor through binding to MLA."""

import ast
import functools
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch import nn

from atom.config import ParallelConfig

ROOT = Path(__file__).resolve().parents[2]


def _load(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
    ]
    if methods is not None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                node.body = [
                    item for item in node.body if getattr(item, "name", None) in methods
                ]
    tree.body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *body]
    exec(  # noqa: S102 - only production definitions from this checkout
        compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace
    )


@pytest.fixture
def runtime(monkeypatch):
    context = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "contextmanager": contextmanager,
        "_forward_context_local": threading.local(),
    }
    _load(
        "atom/utils/forward_context.py",
        {"ForwardContext", "get_forward_context", "side_stream"},
        context,
    )
    ctx = context["ForwardContext"](
        context=SimpleNamespace(
            is_prefill=True,
            is_dummy_run=False,
            scheduled_tokens=5,
            forward_mode=SimpleNamespace(prefill_token_graph=True),
        ),
        attn_metadata=SimpleNamespace(
            slot_mapping=torch.tensor([7, 3, 11, 2, 13, -1, -1, -1])
        ),
    )
    context["_forward_context"] = ctx
    ctx.tr_graph = "replay"
    graph = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "functools": functools,
        "_local": threading.local(),
        "forward_context": SimpleNamespace(side_stream=context["side_stream"]),
    }
    _load(
        "atom/utils/region_graph.py",
        {"TokenGraphQueryLayout", "_recorder", "eager_region"},
        graph,
    )
    layout = graph["TokenGraphQueryLayout"]
    mode = SimpleNamespace(
        plugin=False, quantized=False, sparse=True, fail=False, pcp=False
    )
    seen = []

    class MLA(nn.Module):
        def __init__(self, layer_num, sparse):
            super().__init__()
            self.layer_num, self.is_sparse_mla = layer_num, sparse
            self.split_prefill_recording = False

        def prefill_split_supported(self):
            return not self.is_sparse_mla

        def records_prefill_split(self):
            return False

        def forward(self, query, k_nope, k_rope, positions, q_scale):
            slots = ctx.attn_metadata.slot_mapping
            seen.append(
                (
                    query.shape[0],
                    k_nope.shape[0],
                    k_rope.shape[0],
                    positions.shape[-1],
                    slots,
                    positions,
                    q_scale,
                )
            )
            if mode.fail:
                raise RuntimeError("attention failed")
            return query[:, :3] + k_nope[:, :3] + k_rope[:, :3]

    class Projection(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Attention(nn.Module):
        def __init__(self, *args, layer_num, mla_modules, **kwargs):
            super().__init__()
            self.impl = MLA(layer_num, mla_modules.is_sparse)
            self.layer_num, self.base_attention, self.use_mla = layer_num, None, True
            self.indexer = mla_modules.indexer

    quant = Enum("QuantType", {"No": 0, "per_Token": 1, "per_1x128": 2})
    fp8, fp4 = torch.float8_e4m3fn, torch.uint8
    hf = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        dsa_layer_indices=(0, 2, 4),
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        num_hidden_layers=4,
    )
    config = SimpleNamespace(
        hf_config=hf,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        max_model_len=32,
        # As the native engine builds it: the DCP size is the top-level one,
        # ParallelConfig keeps its default.
        parallel_config=ParallelConfig(),
        decode_context_parallel_size=8,
        prefill_context_parallel_size=1,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    envs = SimpleNamespace(
        ATOM_DSA_ROW_PARALLEL_PREFILL=False,
        ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT=True,
        ATOM_FUSB_QKVG=False,
        ATOM_FUSB_SKIP_WQB=False,
    )
    normed = []

    def fuse_rmsnorm_quant(q_c, q_w, q_eps, kv_c, kv_w, kv_eps, res, **kw):
        normed.append(kw)
        return (q_c * q_w, None), None, kv_c * kv_w, None

    namespace = {
        "torch": torch,
        "nn": nn,
        "envs": envs,
        "pcp_is_enabled": lambda: mode.pcp,
        "get_current_atom_config": lambda: config,
        "get_forward_context": context["get_forward_context"],
        "_fuse_rmsnorm_quant": fuse_rmsnorm_quant,
        "MLAAttention": MLA,
        "MLAModules": SimpleNamespace,
        "Attention": Attention,
        "TokenGraphQueryLayout": layout,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "_extract_layer_idx": lambda prefix: int(prefix.split(".")[1]),
        "indexer_projection_fusable": lambda *args: True,
        "_is_e8m0_blockscale": lambda *args: False,
        "qrep_tp_override": lambda *args: {},
        "QuantType": quant,
        "dtypes": SimpleNamespace(fp8=fp8, fp4x2=fp4, bf16=torch.bfloat16),
        "logging": logging,
        "lru_cache": functools.lru_cache,
        "_RMS_FUSABLE_QUANT_TYPES": (),
        "is_plugin_mode": lambda: mode.plugin,
        "effective_layer_quant": lambda config, prefix: (
            (quant.per_Token, fp8)
            if mode.quantized and prefix.endswith("q_b_proj")
            else (quant.No, None)
        ),
    }
    for name in (
        "FusedIndexerProjection",
        "MergedReplicatedLinear",
        "RMSNorm",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "NoPositionalRotaryEmbedding",
        "Indexer",
    ):
        namespace[name] = Projection
    # The output gate in the merge has its own tests (test_dcp_gate_in_merge.py);
    # other switches may share this envs.
    namespace.setdefault("envs", SimpleNamespace()).ATOM_DMLA_GATE_IN_MERGE = False
    _load(
        "atom/models/kimi_k3/model.py",
        {"_text_config", "KimiFullAttention", "_linear_into"},
        namespace,
        {"__init__", "dsa_front", "_gated_core"},
    )
    # ATOM_FUSB_QKVG's merge, whole (its class attributes too).
    _load(
        "atom/models/kimi_k3/model.py",
        {"_QkvgMerge", "_qkvg_refusal", "_mono_serves", "_log_qkvg_merged"},
        namespace,
    )
    receiver = {
        "torch": torch,
        "MLAAttention": MLA,
        "TokenGraphQueryLayout": layout,
        "get_forward_context": context["get_forward_context"],
        "get_current_atom_config": lambda: config,
        "is_plugin_mode": lambda: mode.plugin,
        "eager_region": graph["eager_region"],
        "mark_spliting_op": lambda **kw: lambda f: f,
        "fake_": lambda *args: None,
        # SPG's region-out recording path has its own tests (test_spg_region_out.py)
        "envs": SimpleNamespace(ATOM_SPG_REGION_OUT=False),
    }
    _load(
        "atom/model_ops/base_attention.py",
        {
            "_mla_token_graph_forward",
            "_fits",
            "unified_attention_with_output_base",
            "_merge_gate_impl",
            "_unified_attention",
            "_attention",
            "_attention_region",
        },
        receiver,
    )
    builder_ns = {
        "torch": torch,
        "MLAAttention": MLA,
        "TokenGraphQueryLayout": layout,
        "PageUnitGeometryMixin": type("Geometry", (), {}),
        "GDNStateMixin": type("State", (), {}),
        "MLA_ROWS": "mla",
        "is_plugin_mode": lambda: mode.plugin,
    }
    _load(
        "atom/model_ops/attentions/kimi_mla_gdn_attn.py",
        {"_KimiMLAGDNCommon"},
        builder_ns,
        {"build_kv_cache_tensor", "supports_token_graph"},
    )

    def make(layer=0):
        hf.dsa_layer_indices = (0, 2, 4) if mode.sparse else ()
        owner = namespace["KimiFullAttention"](
            config, None, prefix=f"layers.{layer}.self_attn"
        )
        config.compilation_config.static_forward_context["layer"] = owner.attn
        return owner

    def bind(owner, target=None):
        builder = builder_ns["_KimiMLAGDNCommon"]()
        builder.model_runner = SimpleNamespace(
            config=config,
            hf_text_config=hf,
            has_mla_indexer=False,
            model=nn.ModuleList([owner]) if target is None else target,
        )
        builder.pool_rows = {"mla": {owner.attn: 0}}
        builder.kv_pool = SimpleNamespace(
            entry_dim=4, layer=lambda *args: torch.zeros(2, 4)
        )
        case.builder = builder
        with monkeypatch.context() as patch:
            patch.setitem(
                sys.modules,
                "atom.config",
                SimpleNamespace(KVCacheTensor=SimpleNamespace),
            )
            return builder.build_kv_cache_tensor(owner.attn)

    args = [
        torch.arange(64, dtype=torch.bfloat16).view(8, 8),
        None,
        torch.ones(8, 8, dtype=torch.bfloat16),
        torch.ones(8, 8, dtype=torch.bfloat16),
        torch.arange(8),
        "layer",
        True,
        None,
    ]
    case = SimpleNamespace(
        make=make,
        bind=bind,
        mode=mode,
        envs=envs,
        namespace=namespace,
        normed=normed,
        config=config,
        ctx=ctx,
        args=args,
        call=receiver["unified_attention_with_output_base"],
        seen=seen,
        layout=layout,
        receiver=receiver,
    )
    return case


@pytest.mark.parametrize("layer", [0, 2], ids=["first-stage", "later-stage"])
@pytest.mark.parametrize("position_axes", [1, 3])
def test_constructor_builder_and_receiver_preserve_live_rows(
    runtime, layer, position_axes
):
    owner = runtime.make(layer)
    declaration = owner.attn.impl.prefill_query_layout
    assert declaration == runtime.layout(layer, torch.bfloat16)
    bound = runtime.bind(owner)
    assert bound.layer_num == layer
    assert owner.attn.impl.token_graph_query_layout is declaration
    if position_axes > 1:
        runtime.args[4] = runtime.args[4].expand(position_axes, -1)
    slots = runtime.ctx.attn_metadata.slot_mapping
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (5, 5, 5, 5)
    torch.testing.assert_close(runtime.seen[-1][4], slots[:5])
    torch.testing.assert_close(runtime.seen[-1][5], runtime.args[4][..., :5])
    assert runtime.ctx.attn_metadata.slot_mapping is slots
    torch.testing.assert_close(out[:5], runtime.args[0][:5, :3] + 2)
    assert out.shape == (8, 3) and not out[5:].any()


@pytest.mark.parametrize("case", ["quantized", "dense", "draft", "plugin"])
def test_constructor_declines_unsupported_owners(runtime, case):
    runtime.mode.quantized = case == "quantized"
    runtime.mode.sparse = case != "dense"
    runtime.mode.plugin = case == "plugin"
    owner = runtime.make(4 if case == "draft" else 0)
    assert getattr(owner.attn.impl, "prefill_query_layout", None) is None


@pytest.mark.parametrize("case", ["off", "sparse", "no_dcp", "quantized", "dense"])
def test_constructor_opts_bf16_sparse_layers_into_row_parallel_prefill(runtime, case):
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = case != "off"
    runtime.config.decode_context_parallel_size = 1 if case == "no_dcp" else 8
    runtime.mode.quantized = case == "quantized"
    runtime.mode.sparse = case != "dense"
    owner = runtime.make()
    opted = case == "sparse"
    assert getattr(owner.attn.impl, "dsa_row_parallel", False) is opted
    if owner.indexer is not None:
        assert getattr(owner.indexer, "row_parallel", False) is opted


def test_a_native_engine_config_opts_layers_into_row_parallel_prefill(runtime):
    # EngineArgs.parallel_config() never sets the DCP size; only the top-level
    # Config field has it (-dcp 8 at -tp 8).
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = True
    assert runtime.config.parallel_config.decode_context_parallel_size == 1
    assert runtime.config.decode_context_parallel_size == 8
    owner = runtime.make()
    assert owner.attn.impl.dsa_row_parallel is True
    assert owner.indexer.row_parallel is True
    assert owner.dsa_front_name == "layers.0.self_attn.dsa_front"


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_native_engine_kwargs_opt_layers_in_under_dcp(runtime, dcp):
    # The kwargs LLMEngine builds Config from (-tp 8 -dcp <dcp>).
    from atom.model_engine.arg_utils import EngineArgs

    kwargs = EngineArgs(
        model="m", tensor_parallel_size=8, decode_context_parallel_size=dcp
    )._get_engine_kwargs()
    runtime.config.parallel_config = kwargs["parallel_config"]
    runtime.config.decode_context_parallel_size = kwargs[
        "decode_context_parallel_size"
    ]
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = True
    owner = runtime.make()
    assert getattr(owner.attn.impl, "dsa_row_parallel", False) is (dcp > 1)
    assert getattr(owner.indexer, "row_parallel", False) is (dcp > 1)


@pytest.mark.parametrize(
    "case", ["front", "off", "front_off", "no_dcp", "pcp", "quantized"]
)
def test_constructor_registers_the_front_op_of_row_parallel_layers(runtime, case):
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = case != "off"
    runtime.config.decode_context_parallel_size = 1 if case == "no_dcp" else 8
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL_FRONT = case != "front_off"
    runtime.mode.pcp = case == "pcp"
    runtime.mode.quantized = case == "quantized"
    owner = runtime.make()
    names = runtime.config.compilation_config.static_forward_context
    if case == "front":
        assert owner.dsa_front_name == "layers.0.self_attn.dsa_front"
        assert names[owner.dsa_front_name] is owner
    else:
        assert owner.dsa_front_name is None
        assert not any(name.endswith(".dsa_front") for name in names)


class _Linear(nn.Module):
    """``x @ weight.T``, each call recorded; ``out=`` where ``takes_out``."""

    def __init__(self, weight, calls, name, takes_out=True):
        super().__init__()
        self.weight, self.calls, self.name = weight, calls, name
        self.takes_out = takes_out

    def supports_out(self):
        return self.takes_out

    def make_row_view(self, start, length):
        return _Linear(
            self.weight[start : start + length],
            self.calls,
            f"{self.name}[{start}:{start + length}]",
            self.takes_out,
        )

    def forward(self, x, x_scale=None, out=None):
        assert x_scale is None
        self.calls.append((self.name, x.shape[0], out is not None))
        y = x @ self.weight.T
        return y if out is None else out.copy_(y)


def _front(runtime, takes_out=True):
    """A row-parallel layer with exact integer weights: any row's front is
    the same whichever rows a GEMM runs over."""
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = True
    owner = runtime.make()
    g = torch.Generator().manual_seed(0)

    def ints(*shape):
        return torch.randint(-3, 4, shape, generator=g).float()

    calls = []
    owner.fused_qkv_a_proj = _Linear(ints(16, 8), calls, "fused", takes_out)
    owner.indexer.wq_b = _Linear(ints(8, 4), calls, "wq_b", takes_out)
    owner.indexer.n_head, owner.indexer.head_dim = 2, 4
    for norm in (owner.q_a_layernorm, owner.kv_a_layernorm):
        norm.weight, norm.eps = ints(4), 1e-6
    return owner, ints(8, 8), calls


def test_the_front_op_runs_every_row_off_a_front_step(runtime):
    owner, hidden, calls = _front(runtime)
    for plan in (None, SimpleNamespace(front=False, first=2, end=5)):
        runtime.ctx.attn_metadata.dsa_row_parallel = plan
        calls.clear()
        runtime.normed.clear()
        projected, q, kv, index_q = owner.dsa_front(hidden)
        full = hidden @ owner.fused_qkv_a_proj.weight.T
        assert torch.equal(projected, full)
        assert torch.equal(q, full[:, :4] * owner.q_a_layernorm.weight)
        assert torch.equal(kv, full[:, 4:8] * owner.kv_a_layernorm.weight)
        assert torch.equal(index_q, q @ owner.indexer.wq_b.weight.T)
        assert calls == [("fused", 8, False), ("wq_b", 8, False)]
        # `_gated_core`'s norm call for a BF16 q.
        assert runtime.normed == [
            {
                "dtype_quant": torch.bfloat16,
                "shuffle": False,
                "scale_shuffle_padding": False,
                "group_size": 128,
                "quant_type": 0,
                "output_unquantized_inp1": False,
                "transpose_scale": True,
            }
        ]


@pytest.mark.parametrize("takes_out", [True, False])
def test_a_front_step_runs_the_front_on_its_rows(runtime, takes_out):
    owner, hidden, calls = _front(runtime, takes_out)
    runtime.ctx.attn_metadata.dsa_row_parallel = None
    full = owner.dsa_front(hidden)
    calls.clear()
    runtime.ctx.attn_metadata.dsa_row_parallel = SimpleNamespace(
        front=True, first=2, end=5
    )
    got = owner.dsa_front(hidden)
    own = slice(2, 5)
    for out, want in zip(got, full):
        assert out.shape == want.shape and out.is_contiguous()
        assert torch.equal(out[own], want[own])
    # Every row's index keys, for the indexer's cache writer.
    assert torch.equal(got[0][:, 10:14], full[0][:, 10:14])
    assert calls == [
        ("fused", 3, takes_out),
        ("fused[10:14]", 8, False),
        ("wq_b", 3, takes_out),
    ]


class _Stop(Exception):
    pass


def test_gated_core_takes_the_front_from_its_op(runtime):
    owner, hidden, _calls = _front(runtime)
    outs = (
        torch.arange(8 * 16.0).view(8, 16),
        torch.ones(8, 4),
        torch.ones(8, 4) * 2,
        torch.ones(8, 8) * 3,
    )
    fronts, indexed = [], []

    class Torch:
        ops = SimpleNamespace(
            aiter=SimpleNamespace(kimi_k3_dsa_front=lambda *a: fronts.append(a) or outs)
        )

        def __getattr__(self, name):
            return getattr(torch, name)

    class Indexer(nn.Module):
        def forward(self, *args, **kwargs):
            indexed.append((args, kwargs))
            raise _Stop

    runtime.namespace["torch"] = Torch()
    owner.indexer = Indexer()
    positions = torch.arange(8)
    with pytest.raises(_Stop):
        owner._gated_core(positions, hidden)
    ((h, name),) = fronts
    assert h is hidden and name == owner.dsa_front_name
    (((h2, q, q_scale, pos), kw),) = indexed
    assert h2 is hidden and q is outs[1] and q_scale is None and pos is positions
    k, weights = kw["projected_k_weights"]
    assert torch.equal(k, outs[0][:, 10:14]) and torch.equal(weights, outs[0][:, 14:])
    assert kw["projected_q"] is outs[3]


def _qkvg(runtime):
    """`_front` under ATOM_FUSB_QKVG: BF16 integer weights (exact whatever
    the summation order), a g_proj, and aiter's GEMM recorded. The merge's
    post-load hook is the test's to run."""
    runtime.envs.ATOM_FUSB_QKVG = True
    owner, _, calls = _front(runtime)
    g = torch.Generator().manual_seed(1)

    def ints(*shape):
        return torch.randint(-3, 4, shape, generator=g).to(torch.bfloat16)

    owner.fused_qkv_a_proj = _Linear(ints(16, 8), calls, "fused")
    owner.g_proj = _Linear(ints(6, 8), calls, "g")
    owner.indexer.wq_b = _Linear(ints(8, 4), calls, "wq_b")
    for norm in (owner.q_a_layernorm, owner.kv_a_layernorm):
        norm.weight = ints(4)
    for lin in (owner.fused_qkv_a_proj, owner.g_proj):
        lin.runs_bf16, lin.bias, lin.prefix = True, None, lin.name
        lin.quant_type = runtime.namespace["QuantType"].No

    def mm(x, w, bias, otype):
        assert bias is None and otype is torch.bfloat16
        calls.append(("tgemm", x.shape[0], w.shape[0]))
        return x @ w.T

    runtime.namespace["tgemm"] = SimpleNamespace(mm=mm)
    runtime.ctx.attn_metadata.dsa_row_parallel = None
    return owner, ints(8, 8), calls


def test_the_qkvg_merge_makes_both_weights_views_of_one(runtime):
    owner, _, _ = _qkvg(runtime)
    a, g = owner.fused_qkv_a_proj.weight.clone(), owner.g_proj.weight.clone()
    merge = owner._qkvg_merge
    # The loader's parent-first walk reaches it after every projection.
    assert list(dict(owner.named_children()))[-1] == "_qkvg_merge"
    assert not list(merge.children()) and not merge.ready
    merge.process_weights_after_loading()
    assert merge.ready and (merge.split, merge.width) == (16, 22)
    assert torch.equal(merge.weight, torch.cat([a, g]))
    assert owner.fused_qkv_a_proj.weight.data_ptr() == merge.weight.data_ptr()
    assert owner.g_proj.weight.data_ptr() == merge.weight[16:].data_ptr()
    assert torch.equal(owner.fused_qkv_a_proj.weight, a)
    assert torch.equal(owner.g_proj.weight, g)


@pytest.mark.parametrize(
    "change", ["bias", "shuffled", "fp32", "padded", "gate_quant", "rows"]
)
def test_a_refused_qkvg_merge_keeps_the_two_weights(runtime, change):
    owner, _, _ = _qkvg(runtime)
    a, g = owner.fused_qkv_a_proj, owner.g_proj
    if change == "bias":
        g.bias = torch.zeros(6)
    elif change == "shuffled":
        a.weight.is_shuffled = True
    elif change == "fp32":
        g.weight = g.weight.float()
    elif change == "padded":
        g.is_output_padded = True
    elif change == "gate_quant":
        owner.fuse_sigmoid_mul_quant = True
    else:
        a.weight = a.weight[:12]
    wa, wg = a.weight, g.weight
    owner._qkvg_merge.process_weights_after_loading()
    assert not owner._qkvg_merge.ready and owner._qkvg_merge.weight is None
    assert a.weight is wa and g.weight is wg


def test_a_short_step_runs_one_gemm_over_the_merged_weight(runtime):
    owner, hidden, calls = _qkvg(runtime)
    owner._qkvg_merge.process_weights_after_loading()
    calls.clear()
    both, q, kv, index_q = owner.dsa_front(hidden, with_gate=True)
    full = hidden @ owner.fused_qkv_a_proj.weight.T
    assert both.shape == (8, 22) and both.is_contiguous()
    assert torch.equal(both[:, :16], full)
    assert torch.equal(both[:, 16:], hidden @ owner.g_proj.weight.T)
    assert torch.equal(q, full[:, :4] * owner.q_a_layernorm.weight)
    assert torch.equal(kv, full[:, 4:8] * owner.kv_a_layernorm.weight)
    assert torch.equal(index_q, q @ owner.indexer.wq_b.weight.T)
    assert calls == [("tgemm", 8, 22), ("wq_b", 8, False)]


def test_a_longer_step_runs_the_two_gemms_into_the_same_layout(runtime):
    owner, hidden, calls = _qkvg(runtime)
    merge = owner._qkvg_merge
    merge.process_weights_after_loading()
    short = owner.dsa_front(hidden, with_gate=True)
    merge.MAX_ROWS = 7
    calls.clear()
    longer = owner.dsa_front(hidden, with_gate=True)
    assert calls == [("fused", 8, False), ("g", 8, False), ("wq_b", 8, False)]
    for got, want in zip(longer, short):
        assert got.shape == want.shape and got.is_contiguous()
        assert torch.equal(got, want)


def test_the_plain_front_op_keeps_its_calls_beside_the_merge(runtime):
    owner, hidden, calls = _qkvg(runtime)
    owner._qkvg_merge.process_weights_after_loading()
    calls.clear()
    projected, *_ = owner.dsa_front(hidden)
    assert projected.shape == (8, 16)
    assert torch.equal(projected, hidden @ owner.fused_qkv_a_proj.weight.T)
    assert calls == [("fused", 8, False), ("wq_b", 8, False)]


def test_a_front_step_carries_every_rows_gate(runtime):
    owner, hidden, calls = _qkvg(runtime)
    owner._qkvg_merge.process_weights_after_loading()
    full = owner.dsa_front(hidden, with_gate=True)
    runtime.ctx.attn_metadata.dsa_row_parallel = SimpleNamespace(
        front=True, first=2, end=5
    )
    calls.clear()
    got = owner.dsa_front(hidden, with_gate=True)
    own = slice(2, 5)
    assert got[0].shape == (8, 22) and got[0].is_contiguous()
    assert torch.equal(got[0][own], full[0][own])
    assert torch.equal(got[0][:, 10:14], full[0][:, 10:14])
    assert torch.equal(got[0][:, 16:], full[0][:, 16:])
    assert calls[-1] == ("g", 8, False)


@pytest.mark.parametrize("served", [False, True])
def test_the_front_skips_wq_b_on_a_step_the_launch_serves(runtime, served):
    runtime.envs.ATOM_FUSB_SKIP_WQB = True
    owner, hidden, calls = _front(runtime)
    assert owner._front_skips_wq_b
    asked = []
    runtime.namespace["_mono_serves"] = lambda m: asked.append(m) or served
    runtime.ctx.attn_metadata.dsa_row_parallel = None
    projected, q, kv, index_q = owner.dsa_front(hidden)
    assert asked == [owner.indexer]
    assert index_q.shape == (8, 8) and index_q.dtype == hidden.dtype
    if served:
        assert calls == [("fused", 8, False)]
    else:
        assert calls == [("fused", 8, False), ("wq_b", 8, False)]
        assert torch.equal(index_q, q @ owner.indexer.wq_b.weight.T)
    # A row-parallel front step is never one the launch serves.
    runtime.ctx.attn_metadata.dsa_row_parallel = SimpleNamespace(
        front=True, first=2, end=5
    )
    asked.clear()
    calls.clear()
    owner.dsa_front(hidden)
    assert not asked and calls[-1] == ("wq_b", 3, True)


@pytest.mark.parametrize("row_parallel", [False, True])
def test_the_fusb_switches_off_build_nothing(runtime, row_parallel):
    runtime.envs.ATOM_DSA_ROW_PARALLEL_PREFILL = row_parallel
    owner = runtime.make()
    assert (owner.dsa_front_name is not None) == row_parallel
    assert owner._qkvg_merge is None and owner._front_skips_wq_b is False
    assert "_qkvg_merge" not in dict(owner.named_children())


def test_the_fusb_switches_act_only_where_the_front_op_runs(runtime):
    runtime.envs.ATOM_FUSB_QKVG = runtime.envs.ATOM_FUSB_SKIP_WQB = True
    owner = runtime.make()
    assert owner.dsa_front_name is None
    assert owner._qkvg_merge is None and owner._front_skips_wq_b is False


@pytest.mark.parametrize("ready", [True, False])
def test_gated_core_takes_the_gate_from_the_qkvg_op(runtime, ready):
    owner, hidden, _calls = _qkvg(runtime)
    if ready:
        owner._qkvg_merge.process_weights_after_loading()
    both = torch.arange(8 * 22.0).view(8, 22)
    outs = (both, torch.ones(8, 4), torch.ones(8, 4) * 2, torch.ones(8, 8) * 3)
    plain = (both[:, :16].contiguous(), *outs[1:])
    ops, attended, gates = [], [], []

    class Torch:
        ops = SimpleNamespace(
            aiter=SimpleNamespace(
                kimi_k3_dsa_front=lambda *a: ops.append("front") or plain,
                kimi_k3_dsa_front_qkvg=lambda *a: ops.append("qkvg") or outs,
            )
        )

        def __getattr__(self, name):
            return getattr(torch, name)

    class Indexer(nn.Module):
        def forward(self, *args, **kwargs):
            pass

    class Attn(nn.Module):
        def forward(self, q, kv, k_rope, positions, q_scale=None, qkv=None):
            attended.append((k_rope, qkv))
            return "gated"

    class Gate(nn.Module):
        def forward(self, x, x_scale=None):
            gates.append(x)
            return "g_proj"

    runtime.namespace["torch"] = Torch()
    owner.indexer, owner.attn, owner.g_proj = Indexer(), Attn(), Gate()
    owner.gate_in_merge = True
    assert owner._gated_core(torch.arange(8), hidden) == ("gated", None)
    ((k_rope, gate),) = attended
    if ready:
        assert ops == ["qkvg"] and not gates
        assert torch.equal(gate, both[:, 16:])
        assert gate.data_ptr() == both[:, 16:].data_ptr()
        assert torch.equal(k_rope, both[:, 8:10])
    else:
        assert ops == ["front"] and gates == [hidden] and gate == "g_proj"


@pytest.mark.parametrize("projected", [False, True])
def test_the_indexer_takes_a_projected_q_in_place_of_wq_b(projected):
    namespace = {
        "torch": torch,
        "nn": nn,
        "pcp_is_enabled": lambda: False,
        "_pcp_active": lambda: False,
        "IndexerDecoratorForPluginMode": lambda cls: cls,
    }
    _load(
        "atom/models/deepseek_v2.py",
        {"Indexer"},
        namespace,
        {"forward", "forward_impl"},
    )
    indexer = object.__new__(namespace["Indexer"])
    nn.Module.__init__(indexer)
    projections, scored = [], []
    for name, value in {
        "rotary_emb": SimpleNamespace(is_neox_style=False),
        "wq_b": lambda x, s: projections.append((x, s)) or torch.ones(5, 8),
        "n_head": 2,
        "head_dim": 4,
        "external_k_weights": True,
        "use_nope_cache_fusion": True,
        "_indexer_opus": False,
        "use_qk_rope_cache_fusion": True,
        "sparse_attn_indexer_impl": lambda *a, **k: scored.append(a),
        "k_cache": SimpleNamespace(prefix="L.k_cache", kv_cache=[torch.zeros(1)]),
        "quant_block_size": 128,
        "scale_fmt": None,
        "topk_tokens": 4,
        "max_model_len": 8,
        "max_total_seq_len": 8,
        "sparse_kv_indices_buffer": None,
        "dcp_sparse_kv_indptr_buffer": None,
        "dcp_owned_counts_buffer": None,
        "k_norm": SimpleNamespace(weight=None, bias=None, eps=1e-6),
        "rope_dim": 0,
        "_empty_rope_cache": torch.zeros(0),
        "_weights_scale": 1.0,
        "stable_topk": False,
    }.items():
        object.__setattr__(indexer, name, value)
    qr, k, weights = torch.zeros(5, 4), torch.zeros(5, 4), torch.zeros(5, 2)
    index_q = torch.arange(40.0).view(5, 8) if projected else None
    indexer.forward(
        qr,
        qr,
        None,
        torch.arange(5),
        projected_k_weights=(k, weights),
        projected_q=index_q,
    )
    ((args),) = scored
    q_input = args[3]
    assert args[4] is k and args[5] is weights and q_input.shape == (5, 2, 4)
    if projected:
        assert projections == [] and q_input.data_ptr() == index_q.data_ptr()
    else:
        assert len(projections) == 1 and projections[0][0] is qr


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "wrong_layer",
        "wrong_dtype",
        "draft_identity",
        "unknown_target",
        "plugin",
        "dp",
        "pcp",
    ],
)
def test_builder_revokes_layout_without_target_provenance(runtime, case):
    owner = runtime.make()
    if case == "unknown":
        owner.attn.impl.prefill_query_layout = True
    elif case == "wrong_layer":
        owner.attn.impl.prefill_query_layout = runtime.layout(1, torch.bfloat16)
    elif case == "wrong_dtype":
        owner.attn.impl.prefill_query_layout = runtime.layout(0, torch.float8_e4m3fn)
    elif case == "plugin":
        runtime.mode.plugin = True
    elif case == "dp":
        runtime.config.parallel_config.data_parallel_size = 2
    elif case == "pcp":
        runtime.config.prefill_context_parallel_size = 2
    target = nn.ModuleList([]) if case == "draft_identity" else None
    if case == "unknown_target":
        target = SimpleNamespace()
    runtime.bind(owner, target)
    assert owner.attn.impl.token_graph_query_layout is None
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots


def test_receiver_requires_target_binding_before_using_declared_layout(runtime):
    owner = runtime.make()
    assert owner.attn.impl.prefill_query_layout is not None
    assert getattr(owner.attn.impl, "token_graph_query_layout", None) is None
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots


def test_prefill_graph_gate_accepts_bound_native_sparse_queries(runtime):
    runtime.bind(runtime.make())
    assert runtime.builder.supports_token_graph


@pytest.mark.parametrize("reason", ["quantized", "draft", "plugin", "dp", "pcp"])
def test_prefill_graph_gate_keeps_unsupported_sparse_queries_eager(runtime, reason):
    runtime.mode.quantized = reason == "quantized"
    runtime.mode.plugin = reason == "plugin"
    if reason == "dp":
        runtime.config.parallel_config.data_parallel_size = 2
    if reason == "pcp":
        runtime.config.prefill_context_parallel_size = 2
    runtime.bind(runtime.make(4 if reason == "draft" else 0))
    assert not runtime.builder.supports_token_graph


def test_plain_attention_does_not_enter_the_token_graph_helper(runtime):
    runtime.bind(runtime.make())
    runtime.ctx.tr_graph = None
    runtime.ctx.context.forward_mode.prefill_token_graph = False
    runtime.receiver["_mla_token_graph_forward"] = lambda *args: pytest.fail(
        "plain call entered graph helper"
    )
    runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)


@pytest.mark.parametrize(
    "case", ["scale", "dtype", "eager", "decode", "dummy", "full", "plugin"]
)
def test_receiver_preserves_non_sliceable_or_unpadded_calls(runtime, case):
    runtime.bind(runtime.make())
    if case == "scale":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
        runtime.args[1] = torch.ones(1, 8)
    elif case == "dtype":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
        runtime.args[0] = runtime.args[0].float()
    elif case == "eager":
        runtime.ctx.context.forward_mode.prefill_token_graph = False
    elif case == "decode":
        runtime.ctx.context.is_prefill = False
    elif case == "dummy":
        runtime.ctx.context.is_dummy_run = True
    elif case == "full":
        runtime.ctx.context.scheduled_tokens = 8
    else:
        runtime.mode.plugin = True
    slots = runtime.ctx.attn_metadata.slot_mapping
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (8, 8, 8, 8)
    assert runtime.seen[-1][4] is slots
    assert runtime.seen[-1][6] is runtime.args[1]
    assert out.dtype == runtime.args[0].dtype


@pytest.mark.parametrize("case", ["scale", "dtype"])
def test_bound_padded_query_refuses_changed_runtime_layout(runtime, case):
    runtime.bind(runtime.make())
    if case == "scale":
        runtime.args[1] = torch.ones(1, 8)
    else:
        runtime.args[0] = runtime.args[0].float()
    slots = runtime.ctx.attn_metadata.slot_mapping
    with pytest.raises(ValueError, match="bound token-graph query layout"):
        runtime.call(*runtime.args)
    assert not runtime.seen
    assert runtime.ctx.attn_metadata.slot_mapping is slots


def test_slots_are_restored_on_failure_and_next_live_step(runtime):
    runtime.bind(runtime.make())
    slots = runtime.ctx.attn_metadata.slot_mapping
    runtime.mode.fail = True
    with pytest.raises(RuntimeError, match="attention failed"):
        runtime.call(*runtime.args)
    assert runtime.ctx.attn_metadata.slot_mapping is slots
    runtime.mode.fail = False
    runtime.ctx.context.scheduled_tokens = 2
    new_slots = torch.tensor([12, 4, -1, -1, -1, -1, -1, -1])
    runtime.ctx.attn_metadata.slot_mapping = new_slots
    out = runtime.call(*runtime.args)
    assert runtime.seen[-1][:4] == (2, 2, 2, 2)
    torch.testing.assert_close(runtime.seen[-1][4], new_slots[:2])
    assert runtime.ctx.attn_metadata.slot_mapping is new_slots
    assert not out[2:].any()
