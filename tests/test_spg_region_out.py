# SPDX-License-Identifier: MIT
"""``ATOM_SPG_REGION_OUT``: a prefill token-graph recording hands the sparse
MLA attention region and the KDA recurrence region an output buffer, which
they write and return, so a replay reads their result in place. Kernels are
replaced by references."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo  # before Triton may be faked below
from aiter_stub import atom_without_aiter
from torch import nn

from atom.utils import forward_context
from atom.utils.region_graph import TokenGraphQueryLayout


@contextlib.contextmanager
def _einops():
    """The Kimi model imports einops, which the CPU runner may not have."""
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
    mla = _load("atom.model_ops.attention_mla")
    base = _load("atom.model_ops.base_attention")
    model = _load("atom.models.kimi_k3.model")

HEADS, V_DIM, HIDDEN = 2, 8, 32


@pytest.fixture
def step(monkeypatch):
    ctx = SimpleNamespace(
        context=None, tr_graph=None, in_hipgraph=False, attn_metadata=None
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    monkeypatch.setattr(mla, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(base, "is_plugin_mode", lambda: False)
    return ctx


def _sparse_impl(**over):
    impl = object.__new__(mla.MLAAttention)
    impl.__dict__.update(
        split_prefill_recording=True,
        is_sparse_mla=True,
        o_proj=nn.Identity(),
        num_heads=HEADS,
        v_head_dim=V_DIM,
        layer_num=0,
        token_graph_query_layout=TokenGraphQueryLayout(0, torch.bfloat16),
        forward=lambda query, k_nope, k_rope, positions, q_scale: (
            query[:, : HEADS * V_DIM] * 2 + 1
        ),
        **over,
    )
    return impl


def _atom_config(monkeypatch, impl):
    layers = {"layer": SimpleNamespace(impl=impl)}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers),
        hf_config=SimpleNamespace(hidden_size=HIDDEN),
        torch_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(base, "get_current_atom_config", lambda: config)


@pytest.mark.parametrize(
    "switch, tr_graph, in_hipgraph, handed",
    [
        ("1", "record", True, True),
        ("0", "record", True, False),
        ("1", "replay", False, False),  # a forward under a replaying step
        ("1", None, True, False),  # the warmup forward before the recording
    ],
)
def test_a_sparse_mla_recording_hands_its_region_the_op_output(
    monkeypatch, step, switch, tr_graph, in_hipgraph, handed
):
    monkeypatch.setenv("ATOM_SPG_REGION_OUT", switch)
    step.tr_graph, step.in_hipgraph = tr_graph, in_hipgraph
    step.context = SimpleNamespace(is_prefill=True, is_dummy_run=True)
    _atom_config(monkeypatch, _sparse_impl())
    monkeypatch.setattr(base, "_attention_region", lambda *args: args)
    q = torch.zeros(5, 40, dtype=torch.bfloat16)
    args = base.unified_attention_with_output_base(
        q, None, q, q, None, "layer", True, None
    )
    assert len(args) == (9 if handed else 8)
    if handed:
        out = args[-1]
        assert out.shape == (5, HEADS * V_DIM) and out.dtype == torch.bfloat16


def _live(step, rows, total):
    step.context = SimpleNamespace(
        is_prefill=True,
        is_dummy_run=False,
        forward_mode=SimpleNamespace(prefill_token_graph=True),
        scheduled_tokens=rows,
    )
    step.attn_metadata = SimpleNamespace(slot_mapping=torch.arange(total))


def _query(total):
    generator = torch.Generator().manual_seed(total)
    return torch.randn(total, 40, generator=generator).to(torch.bfloat16)


def test_a_live_bucket_writes_its_rows_and_zeros_only_the_pad(step):
    impl = _sparse_impl()
    _live(step, rows=3, total=5)
    q = _query(5)
    want = base._mla_token_graph_forward(impl, q, None, q, q, None)
    out = torch.full((5, HEADS * V_DIM), 7.0, dtype=torch.bfloat16)
    got = base._mla_token_graph_forward(impl, q, None, q, q, None, out)
    assert got is out
    assert torch.equal(got, want)
    assert step.attn_metadata.slot_mapping.shape == (5,)


def test_a_full_bucket_and_the_recording_write_the_buffer(step):
    impl = _sparse_impl()
    for context in (
        SimpleNamespace(is_prefill=True, is_dummy_run=True),  # the recording
        None,
    ):
        _live(step, rows=5, total=5)
        if context is not None:
            step.context = context
        q = _query(5)
        want = base._mla_token_graph_forward(impl, q, None, q, q, None)
        out = torch.full((5, HEADS * V_DIM), 7.0, dtype=torch.bfloat16)
        got = base._mla_token_graph_forward(impl, q, None, q, q, None, out)
        assert got is out and torch.equal(got, want)


def test_a_buffer_that_does_not_fit_is_left_alone(step):
    impl = _sparse_impl()
    q = _query(5)
    for rows in (3, 5):
        _live(step, rows=rows, total=5)
        want = base._mla_token_graph_forward(impl, q, None, q, q, None)
        for out in (
            torch.zeros(5, HEADS * V_DIM + 1, dtype=torch.bfloat16),
            torch.zeros(5, HEADS * V_DIM, dtype=torch.float32),
        ):
            got = base._mla_token_graph_forward(impl, q, None, q, q, None, out)
            assert got is not out and torch.equal(got, want)
            assert not out.any()


# ---- KDA ----

H, D = 2, 4


def _kda_layer(**over):
    layer = SimpleNamespace(num_local_heads=H, head_dim=D, layer_name="kda", **over)
    return layer


@pytest.mark.parametrize(
    "switch, tr_graph, handed",
    [("1", "record", True), ("0", "record", False), ("1", None, False)],
)
def test_a_kda_recording_hands_its_recurrence_a_buffer(
    monkeypatch, step, switch, tr_graph, handed
):
    monkeypatch.setenv("ATOM_SPG_REGION_OUT", switch)
    step.tr_graph = tr_graph
    mixed = torch.zeros(6, 3 * H * D, dtype=torch.bfloat16)
    layer = _kda_layer(
        _project_in=lambda hs, scale: (mixed, "gate_out", "beta", "gate"),
        _project_out=lambda out, gate: out,
    )
    monkeypatch.setattr(model, "_kda_recurrence", lambda *args: args)
    args = model.KimiKDAAttention._forward_prefill_split(layer, None, None)
    assert len(args) == (5 if handed else 4)
    if handed:
        assert args[-1].shape == (12, H, D) and args[-1].dtype == torch.bfloat16


def test_the_recordings_dummy_prefill_zeros_the_buffer(step):
    step.context = SimpleNamespace(is_dummy_run=True)
    layer = _kda_layer()
    mixed = torch.zeros(6, 3 * H * D, dtype=torch.bfloat16)
    out = torch.full((12, H, D), 7.0, dtype=torch.bfloat16)
    got = model.KimiKDAAttention._live_prefill_recurrence(layer, mixed, None, None, out)
    assert got.data_ptr() == out.data_ptr() and got.shape == (6, H, D)
    assert not got.any() and bool((out[6:] == 7).all())


def _kda_inputs(total):
    generator = torch.Generator().manual_seed(total)

    def rand(*shape):
        return torch.randn(1, total, *shape, generator=generator).to(torch.bfloat16)

    return rand(H, D), rand(H, D), rand(H, D), rand(H, D), rand(H)


def _run_kda(q, k, v, g, beta, initial, cu, final, out=None):
    """A reference recurrence: output a function of each token, final states
    one per sequence."""
    o = torch.empty_like(v) if out is None else out
    o.copy_(q * 2 + v)
    return o, initial + cu.shape[0]


@pytest.mark.parametrize(
    "heads, bucket",
    [
        ([(2, 3), (6, 2)], 9),  # rows and heads overrun the bucket
        ([(6, 1)], 12),  # they do not fill it: pad rows past them
    ],
)
def test_prefill_kda_writes_the_buffer_and_returns_its_rows(heads, bucket):
    num_tokens = 8
    q, k, v, g, beta = _kda_inputs(bucket)
    tokens = torch.cat(
        [torch.arange(num_tokens)] + [torch.arange(s, s + n) for s, n in heads]
    )
    kda_cu = [0, num_tokens]
    for _, n in heads:
        kda_cu.append(kda_cu[-1] + n)
    ckpt = {
        "kda_tokens": tokens,
        "kda_cu": torch.tensor(kda_cu),
        "kda_rows": torch.zeros(1 + len(heads), dtype=torch.long),
    }
    initial = torch.zeros(1, H, D, D)
    layer = _kda_layer(
        _run_kda=_run_kda,
        _kda_out_rows=lambda out, q, v, cu: out[: q.shape[1]].unsqueeze(0),
    )
    attn = model.KimiKDAAttention
    want, want_last, want_heads = attn._prefill_kda(
        layer, q, k, v, g, beta, initial, ckpt["kda_cu"], ckpt
    )
    out = torch.full((2 * bucket, H, D), 7.0, dtype=torch.bfloat16)
    got, last, head_states = attn._prefill_kda(
        layer, q, k, v, g, beta, initial, ckpt["kda_cu"], ckpt, out=out
    )
    assert got.shape == want.shape == (1, bucket, H, D)
    assert got.data_ptr() == out.data_ptr() and got.is_contiguous()
    written = min(bucket, len(tokens))
    assert torch.equal(got[:, :written], want[:, :written])
    assert torch.equal(last, want_last) and torch.equal(head_states, want_heads)


def test_prefill_kda_without_room_keeps_its_own_output():
    q, k, v, g, beta = _kda_inputs(4)
    layer = _kda_layer(_run_kda=_run_kda, _kda_out_rows=lambda *a: None)
    cu = torch.tensor([0, 2, 4])
    rows = torch.zeros(2, dtype=torch.long)
    ckpt = {"kda_tokens": torch.arange(4), "kda_cu": cu, "kda_rows": rows}
    out = torch.zeros(8, H, D, dtype=torch.bfloat16)
    got, _, _ = model.KimiKDAAttention._prefill_kda(
        layer, q, k, v, g, beta, torch.zeros(2, H, D, D), cu, ckpt, out=out
    )
    assert got.data_ptr() != out.data_ptr() and not out.any()


def test_kda_out_rows_stays_off_the_cpu():
    layer = _kda_layer(_kda_gate_lower_bound=-5.0, A_log=torch.zeros(H))
    q = torch.zeros(1, 6, H, D, dtype=torch.bfloat16)
    out = torch.zeros(12, H, D, dtype=torch.bfloat16)
    rows = model.KimiKDAAttention._kda_out_rows
    # CPU tensors never reach FlashKDA.
    assert rows(layer, out, q, q, torch.tensor([0, 3, 6])) is None
    assert rows(layer, None, q, q, torch.tensor([0, 3, 6])) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="asks FlashKDA")
@pytest.mark.parametrize("cu", [[0, 96], [0, 40, 96]])
def test_every_flash_kda_call_writes_the_buffer(cu):
    """One sequence goes through `chunk_kimi_delta_attn`, several (short ones,
    below the balanced segmenting) too: both reach FlashKDA, which takes it."""
    layer = _kda_layer(A_log=torch.rand(12, device="cuda"), _kda_gate_lower_bound=-5.0)
    q = torch.randn(1, 96, 12, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(192, 12, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(cu, device="cuda", dtype=torch.int32)
    rows = model.KimiKDAAttention._kda_out_rows(layer, out, q, q, cu)
    assert rows is not None and rows.data_ptr() == out.data_ptr()
    assert rows.shape == q.shape


@pytest.mark.parametrize("given", [False, True])
def test_the_paged_recurrence_writes_the_buffer(given):
    seen = {}

    def chunk_kda(**kwargs):
        seen.update(kwargs)
        return kwargs["out"], None

    layer = _kda_layer(
        _chunk_kda=chunk_kda,
        A_log=None,
        dt_bias=None,
        _kda_gate_lower_bound=-5.0,
    )
    q, k, v, g, beta = _kda_inputs(6)
    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor([0, 6]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        has_initial_state=torch.tensor([True]),
    )
    out = torch.zeros(12, H, D, dtype=torch.bfloat16) if given else None
    extra = {"out": out} if given else {}
    got = model.KimiKDAAttention._run_kda_paged(
        layer, q, k, v, g, beta, metadata, "pool", **extra
    )
    assert got.shape == v.shape and got.is_contiguous()
    assert (
        (got.data_ptr() == out.data_ptr())
        if given
        else got.data_ptr()
        not in (
            q.data_ptr(),
            v.data_ptr(),
        )
    )
