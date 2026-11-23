# SPDX-License-Identifier: MIT
"""ATOM_MLA_BF16_ABSORB_BMM_CONFIG: a decode-sized BF16 W_K absorb bmm of an MLA
layer whose owner opted in (the Kimi-K3 DSpark draft) gets an explicit Triton
tile on gfx950; every other layer, the target's included, keeps aiter's own
choice."""

from types import SimpleNamespace

import pytest
import torch

try:
    from aiter import QuantType
    from aiter.dist import parallel_state

    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention
    from atom.models import kimi_k3_dspark
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the MLA attention imports AITER")

HEADS = 4


def _layer(opted_in):
    # The real method on a bare instance: only the opt-in attribute is read.
    layer = MLAAttention.__new__(MLAAttention)
    object.__setattr__(layer, "bf16_absorb_bmm_tile", opted_in)
    return layer


def _kwargs(layer, rows, k, n=512):
    x, w = torch.empty(HEADS, rows, k), torch.empty(HEADS, n, k)
    return layer._bf16_absorb_bmm_kwargs(x, w)


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(attention_mla, "_runs_on_gfx950", lambda: True)


@pytest.mark.parametrize("rows", [1, 48, 64])
def test_an_opted_in_layer_gets_the_tile(monkeypatch, gfx950, rows):
    monkeypatch.setenv("ATOM_MLA_BF16_ABSORB_BMM_CONFIG", "1")
    kw = _kwargs(_layer(True), rows, 128)
    assert kw["backend"] == "triton"
    cfg = kw["config"]
    assert (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]) == (
        32,
        64,
        128,
    )
    assert cfg["NUM_KSPLIT"] == 1 and cfg["SPLITK_BLOCK_SIZE"] == 128
    assert "SPLITK_BLOCK_SIZE" not in MLAAttention._BF16_ABSORB_BMM_CONFIG


@pytest.mark.parametrize(
    "opted_in, flag, rows, k, on_gfx950",
    [
        pytest.param(False, "1", 48, 128, True, id="not opted in (the target)"),
        pytest.param(True, "0", 48, 128, True, id="off"),
        pytest.param(True, "1", 65, 128, True, id="prefill-sized"),
        pytest.param(True, "1", 48, 64, True, id="K below the tile"),
        pytest.param(True, "1", 48, 128, False, id="not gfx950"),
    ],
)
def test_otherwise_aiter_chooses(monkeypatch, opted_in, flag, rows, k, on_gfx950):
    monkeypatch.setenv("ATOM_MLA_BF16_ABSORB_BMM_CONFIG", flag)
    monkeypatch.setattr(attention_mla, "_runs_on_gfx950", lambda: on_gfx950)
    assert _kwargs(_layer(opted_in), rows, k) == {}


class _Module(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _Attention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # What MLAAttention.__init__ leaves for every layer.
        self.impl = SimpleNamespace(bf16_absorb_bmm_tile=False)


def test_the_dspark_draft_opts_its_own_layers_in(monkeypatch):
    m = kimi_k3_dspark
    for name in (
        "MergedReplicatedLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "RMSNorm",
    ):
        monkeypatch.setattr(m, name, _Module)
    monkeypatch.setattr(m, "Attention", _Attention)
    monkeypatch.setattr(m, "get_rope", lambda *a, **k: None)
    monkeypatch.setattr(m, "draft_rope_kwargs", lambda *a, **k: {"rope_scaling": None})
    monkeypatch.setattr(m, "is_neox_rope_style", lambda *a, **k: False)
    monkeypatch.setattr(m, "qrep_tp_override", lambda tp: {})
    monkeypatch.setattr(m, "mla_min_query_heads", lambda *a, **k: 16)
    monkeypatch.setattr(
        m, "effective_layer_quant", lambda *a, **k: (QuantType.No, torch.bfloat16)
    )
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rms_norm_eps=1e-6,
    )
    atom_config = SimpleNamespace(
        quant_config=None, kv_cache_dtype="bf16", speculative_config=None
    )
    attn = m.K3DSparkMLAAttention(atom_config, config, layer_num=0, prefix="l0")
    assert attn.mla_attn.impl.bf16_absorb_bmm_tile is True
