# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_DMLA_GATE_IN_MERGE: a Kimi-K3 full-attention layer hands its output
gate g_proj(x) to the attention op, which returns sigmoid(gate) * attention.
A DCP decode offers the gate to its merge, and a merge that takes the mailbox
applies it in aiter's combine; every other call gets the BF16 gate kernel
after the attention, as the layer ran it before. Kernels are recorders."""

import contextlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from aiter_stub import atom_without_aiter


@contextlib.contextmanager
def _einops():
    if importlib.util.find_spec("einops") is not None:
        yield
        return
    einops = ModuleType("einops")
    einops.__getattr__ = lambda name: MagicMock(name=f"einops.{name}")
    sys.modules["einops"] = einops
    try:
        yield
    finally:
        sys.modules.pop("einops", None)


with _einops(), atom_without_aiter() as _load:
    base = _load("atom.model_ops.base_attention")
    mla = _load("atom.model_ops.attention_mla")
    kimi = _load("atom.models.kimi_k3.model")


@pytest.fixture
def gate_kernel(monkeypatch):
    """The BF16 gate kernel `forward_gated` falls back to, as a recorder."""
    calls = []

    def sigmoid_mul(out, gate):
        calls.append((out, gate))
        return ("gated", out, gate)

    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.triton_fused_sigmoid_mul_quant",
        SimpleNamespace(sigmoid_mul=sigmoid_mul),
    )
    return calls


def _op_layers(monkeypatch, impl):
    compilation = SimpleNamespace(
        static_forward_context={"layer": SimpleNamespace(impl=impl)}
    )
    monkeypatch.setattr(
        base,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )
    monkeypatch.setattr(base, "_unified_attention", lambda *args: ("attention", args))


def test_the_attention_op_hands_a_layers_gate_to_its_impl(monkeypatch):
    gated = []

    def forward_gated(attend, gate):
        gated.append(gate)
        return ("gated", attend())

    _op_layers(
        monkeypatch, SimpleNamespace(gate_in_merge=True, forward_gated=forward_gated)
    )
    q, k, v, positions, gate = (torch.zeros(4, 2) for _ in range(5))
    out = base.unified_attention_with_output_base(
        q, None, k, v, positions, "layer", True, gate
    )
    assert gated == [gate]
    kind, (attention, args) = out
    assert kind == "gated" and attention == "attention"
    # The gate does not reach the attention itself.
    assert args == (q, None, k, v, positions, "layer", True, None)


@pytest.mark.parametrize("case", ["not gated", "no gate", "not MLA"])
def test_the_attention_op_without_a_gate_is_unchanged(monkeypatch, case):
    impl = SimpleNamespace(
        gate_in_merge=case != "not gated",
        forward_gated=lambda *a: pytest.fail("gate applied"),
    )
    _op_layers(monkeypatch, impl)
    q, k, v, positions, gate = (torch.zeros(4, 2) for _ in range(5))
    qkv = None if case == "no gate" else gate
    use_mla = case != "not MLA"
    kind, args = base.unified_attention_with_output_base(
        q, None, k, v, positions, "layer", use_mla, qkv
    )
    assert kind == "attention"
    assert args == (q, None, k, v, positions, "layer", use_mla, qkv)


def _impl():
    impl = object.__new__(mla.MLAAttention)
    assert impl.gate_in_merge is False and impl._merge_gate is None
    return impl


def test_a_merge_that_takes_the_gate_returns_the_gated_rows(gate_kernel):
    impl, gate = _impl(), torch.ones(2, 4)
    taken = []

    def attend():
        taken.append(impl._take_merge_gate())
        return "merged and gated"

    assert impl.forward_gated(attend, gate) == "merged and gated"
    assert taken == [gate] and gate_kernel == []
    assert impl._merge_gate is None


def test_a_call_whose_merge_leaves_the_gate_gets_the_gate_kernel(gate_kernel):
    impl, gate = _impl(), torch.ones(2, 4)
    assert impl.forward_gated(lambda: "attn", gate) == ("gated", "attn", gate)
    assert gate_kernel == [("attn", gate)]
    assert impl._merge_gate is None


def test_a_failing_attention_leaves_no_pending_gate(gate_kernel):
    impl = _impl()

    def fails():
        raise RuntimeError("attention failed")

    with pytest.raises(RuntimeError, match="attention failed"):
        impl.forward_gated(fails, torch.ones(2, 4))
    assert impl._merge_gate is None and gate_kernel == []


@pytest.mark.parametrize("pending", [True, False])
def test_dcp_decode_offers_a_pending_gate_to_its_merge(monkeypatch, pending):
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.dcp_ops",
        SimpleNamespace(dcp_all_gather_query_heads=lambda *a: pytest.fail("gather")),
    )
    impl = _impl()
    impl.is_sparse_mla, impl._cp_triton_ctx = True, None
    attn = (torch.zeros(2, 16, 128), torch.zeros(2, 16))
    impl._forward_decode = lambda *a, **k: attn
    offered = []
    impl._dcp_project_merge_out = lambda o, lse, **kw: offered.append(kw["gate"])
    metadata = SimpleNamespace(dcp_owned_counts=None)
    gate = torch.ones(2, 4)
    if pending:
        impl._merge_gate = [gate, False]
    impl._dcp_decode(torch.zeros(2, 16, 576), None, metadata, use_qrep=True)
    if not pending:
        assert offered == [None]
        return
    [take] = offered
    assert impl._merge_gate == [gate, False]
    assert take() is gate and impl._merge_gate[1] is True


def _merge_rig(monkeypatch, pbm=True, backend="mailbox"):
    seen = []

    def dcp_lse_merge(o, lse, group, backend, **kw):
        seen.append((backend, kw["gate"]))
        return o

    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.dcp_ops",
        SimpleNamespace(dcp_lse_merge=dcp_lse_merge),
    )
    impl = _impl()
    impl.dcp_group, impl.dcp_comm_backend = "group", backend
    impl.pbm_enabled, impl.num_heads, impl.v_head_dim = pbm, 2, 4
    impl.W_V_dcp = impl.W_V_dcp_scale = None
    impl._v_up_proj = lambda o, *a, **k: o
    impl._v_up_proj_and_o_proj = lambda o: o
    impl._dcp_fused_quant_dtype = lambda: None
    impl.o_proj = lambda o: o
    return impl, seen


@pytest.mark.parametrize(
    "pbm,backend,decode,fp32,offered",
    [
        (True, "mailbox", True, False, True),
        (True, "mailbox", False, False, False),  # prefill merges with a2a
        (True, "a2a", True, False, False),
        (False, "mailbox", True, False, False),  # rows still need the V proj
        (True, "mailbox", True, True, False),  # the fp32 merge keeps its store
    ],
)
def test_only_a_mailbox_merge_ahead_of_o_proj_is_offered_the_gate(
    monkeypatch, pbm, backend, decode, fp32, offered
):
    impl, seen = _merge_rig(monkeypatch, pbm, backend)
    take = object()
    impl._dcp_project_merge_out(
        torch.zeros(2, 2, 4),
        torch.zeros(2, 2),
        merge_in_fp32=fp32,
        decode=decode,
        gate=take,
    )
    [(_, gate)] = seen
    assert gate is (take if offered else None)


def _gated_core_layer(monkeypatch, gate_in_merge):
    calls = []
    hidden = torch.zeros(3, 8)
    q, kv = torch.zeros(3, 6), torch.zeros(3, 4)
    projected = torch.zeros(3, 6 + 4 + 2)

    def attn(*args, **kwargs):
        calls.append(("attn", args, kwargs))
        return "attn_out"

    layer = SimpleNamespace(
        gate_in_merge=gate_in_merge,
        dsa_front_name=None,
        fused_qkv_a_proj=lambda x, scale: projected,
        projection_split_sizes=[6, 4, 2],
        qknorm_dtype=torch.bfloat16,
        q_a_layernorm=SimpleNamespace(weight=None, eps=1e-6),
        kv_a_layernorm=SimpleNamespace(weight=None, eps=1e-6),
        qknorm_quant_type_value=0,
        indexer=None,
        attn=attn,
        g_proj=lambda x, scale: calls.append(("g_proj", x, scale)) or "gate",
        fuse_sigmoid_mul_quant=False,
        o_proj_quant_type_value=0,
    )
    norm = SimpleNamespace(
        _fuse_rmsnorm_quant=lambda *a, **k: ((q, None), None, kv, None)
    )
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", norm)
    monkeypatch.setattr(
        kimi,
        "fused_sigmoid_mul_maybe_quant",
        lambda out, gate, **kw: calls.append(("sigmoid", out, gate)) or ("gated", None),
    )
    return layer, hidden, calls


def test_a_gate_in_merge_layer_hands_its_gate_to_the_attention_op(monkeypatch):
    layer, hidden, calls = _gated_core_layer(monkeypatch, gate_in_merge=True)
    out = kimi.KimiFullAttention._gated_core(layer, "positions", hidden)
    assert out == ("attn_out", None)
    assert [c[0] for c in calls] == ["g_proj", "attn"]
    assert calls[0][1] is hidden
    _, args, kwargs = calls[1]
    assert args[-1] == "positions"
    assert kwargs["qkv"] == "gate" and kwargs["q_scale"] is None


def test_an_ungated_layer_runs_g_proj_and_the_sigmoid_itself(monkeypatch):
    layer, hidden, calls = _gated_core_layer(monkeypatch, gate_in_merge=False)
    out = kimi.KimiFullAttention._gated_core(layer, "positions", hidden)
    assert out == ("gated", None)
    assert [c[0] for c in calls] == ["attn", "g_proj", "sigmoid"]
    assert "qkv" not in calls[0][2]
