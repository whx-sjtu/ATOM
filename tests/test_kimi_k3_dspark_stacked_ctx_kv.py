# SPDX-License-Identifier: MIT
"""ATOM_DSPARK_STACKED_CTX_KV_GEMM: the context rows of every draft layer take
one GEMM over the stacked kv rows of their bf16 fused_qkv_a_proj weights; any
layer whose weight is not plain bf16 sends all of them down the per-layer
projection."""

from types import SimpleNamespace

import pytest
import torch

try:
    from aiter import QuantType

    from atom.models import kimi_k3_dspark
    from atom.models.kimi_k3_dspark import (
        K3DSparkDecoderLayer,
        K3DSparkMLAAttention,
        KimiK3DSpark,
    )
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the DSpark draft imports AITER")

Q_LORA, KV, HIDDEN, LAYERS, N = 24, 9, 16, 3, 5


class _Proj(torch.nn.Module):
    def __init__(self, quant_type=None):
        super().__init__()
        self.quant_type = QuantType.No if quant_type is None else quant_type
        self.weight = torch.nn.Parameter(
            torch.randn(Q_LORA + KV, HIDDEN).to(torch.bfloat16), requires_grad=False
        )
        self.bias = None
        self.is_output_padded = False
        self.native_a8_group_rows = None
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.float() @ self.weight.float().t()


def _layer(proj, written):
    attn = K3DSparkMLAAttention.__new__(K3DSparkMLAAttention)
    torch.nn.Module.__init__(attn)
    attn.fused_qkv_a_proj = proj
    attn.q_lora_rank = Q_LORA
    attn.kv_a_layernorm = None
    attn.mla_attn = SimpleNamespace(
        kv_cache=None,
        impl=SimpleNamespace(
            write_context_kv_latent=lambda cache, kv, *a: written.append(kv)
        ),
    )
    layer = K3DSparkDecoderLayer.__new__(K3DSparkDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attn = attn
    return layer


def _model(projs, stacked=True):
    model = KimiK3DSpark.__new__(KimiK3DSpark)
    torch.nn.Module.__init__(model)
    written = []
    model.layers = torch.nn.ModuleList([_layer(p, written) for p in projs])
    model._ctx_kv_src = None
    if stacked:
        model.register_buffer(
            "_ctx_kv_weight",
            torch.empty(len(projs) * KV, HIDDEN, dtype=torch.bfloat16),
            persistent=False,
        )
    else:
        model._ctx_kv_weight = None
    return model, written


@pytest.fixture
def gemms(monkeypatch):
    seen = []

    def mm(x, w):
        seen.append(w)
        return x.float() @ w.float().t()

    monkeypatch.setattr(kimi_k3_dspark, "tgemm", SimpleNamespace(mm=mm))
    return seen


def _write(model):
    x = torch.randn(N, HIDDEN).to(torch.bfloat16)
    model.write_context_rows(x, torch.arange(N), torch.arange(N))
    return x


def _per_layer(x, projs):
    return [(x.float() @ p.weight.float().t())[:, Q_LORA:] for p in projs]


def test_one_gemm_over_the_stacked_kv_rows(gemms):
    projs = [_Proj() for _ in range(LAYERS)]
    model, written = _model(projs)
    x = _write(model)
    (w,) = gemms
    assert w is model._ctx_kv_weight and w.shape == (LAYERS * KV, HIDDEN)
    assert all(p.calls == 0 for p in projs)
    assert len(written) == LAYERS
    for got, want in zip(written, _per_layer(x, projs)):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("replace", [False, True], ids=["reload", "new tensor"])
def test_the_stack_follows_a_weight_update(gemms, replace):
    projs = [_Proj() for _ in range(LAYERS)]
    model, written = _model(projs)
    _write(model)
    new = torch.randn(Q_LORA + KV, HIDDEN).to(torch.bfloat16)
    if replace:
        projs[1].weight = torch.nn.Parameter(new, requires_grad=False)
    else:
        # The loader writes in place, through .data, then runs the hooks.
        projs[1].weight.data.copy_(new)
        model.process_weights_after_loading()
    written.clear()
    x = _write(model)
    for got, want in zip(written, _per_layer(x, projs)):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize(
    "stacked, projs",
    [
        pytest.param(False, lambda: [_Proj() for _ in range(LAYERS)], id="off"),
        pytest.param(
            True,
            lambda: [_Proj(), _Proj(QuantType.per_Token), _Proj()],
            id="one quantized layer",
        ),
    ],
)
def test_otherwise_every_layer_projects_its_own(gemms, stacked, projs):
    projs = projs()
    model, written = _model(projs, stacked=stacked)
    _write(model)
    assert gemms == []
    assert [p.calls for p in projs] == [1] * LAYERS
    assert [kv.shape for kv in written] == [(N, KV)] * LAYERS


@pytest.mark.parametrize(
    "flag, allocated", [(None, True), ("1", True), ("0", False), ("", False)]
)
def test_the_stack_is_on_by_default(monkeypatch, flag, allocated):
    if flag is None:
        monkeypatch.delenv("ATOM_DSPARK_STACKED_CTX_KV_GEMM", raising=False)
    else:
        monkeypatch.setenv("ATOM_DSPARK_STACKED_CTX_KV_GEMM", flag)
    from atom.utils import envs

    assert bool(envs.ATOM_DSPARK_STACKED_CTX_KV_GEMM) is allocated
