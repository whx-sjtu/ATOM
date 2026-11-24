# SPDX-License-Identifier: MIT
"""ATOM_DSPARK_SHARDED_CONTEXT_PROJ: context_proj split by output column
across TP, the rank's columns gathered along the last dim before
context_norm."""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.models import kimi_k3_dspark
    from atom.models.kimi_k3_dspark import KimiK3DSpark
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the DSpark draft imports AITER")

TP, HIDDEN, TAPS, N = 4, 8, 3, 5


class _Group:
    """A TP group of which this process plays every rank: the gather asks
    each rank's projection for its columns."""

    world_size = TP

    def __init__(self, weight):
        self.weight, self.calls, self.x = weight, [], None

    def all_gather(self, t, use_custom, dim):
        self.calls.append((tuple(t.shape), use_custom, dim))
        cols = HIDDEN // TP
        return torch.cat(
            [self.x @ self.weight[r * cols : (r + 1) * cols].t() for r in range(TP)],
            dim=dim,
        )


def _model(sharded, weight, group, rank=1):
    model = KimiK3DSpark.__new__(KimiK3DSpark)
    torch.nn.Module.__init__(model)
    model.context_proj_sharded = sharded
    cols = HIDDEN // TP if sharded else HIDDEN
    w = weight[rank * cols : (rank + 1) * cols] if sharded else weight

    def proj(x):
        group.x = x
        return x @ w.t()

    model.context_proj = proj
    model.context_norm = lambda x: x * 2
    return model


@pytest.fixture
def weight():
    return torch.randn(
        HIDDEN, TAPS * HIDDEN, generator=torch.Generator().manual_seed(0)
    )


@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "replicated"])
def test_the_context_is_the_whole_projection(monkeypatch, weight, sharded):
    group = _Group(weight)
    monkeypatch.setattr(kimi_k3_dspark, "get_tp_group", lambda: group)
    taps = [torch.randn(N, HIDDEN) for _ in range(TAPS)]
    got = _model(sharded, weight, group).project_context(taps)
    torch.testing.assert_close(got, (torch.cat(taps, -1) @ weight.t()) * 2)
    from atom.utils import envs

    want = [((N, HIDDEN // TP), envs.ATOM_USE_CUSTOM_ALL_GATHER, -1)] if sharded else []
    assert group.calls == want


@pytest.mark.parametrize(
    "flag, tp, hidden, sp, on",
    [
        pytest.param("1", 8, 7168, False, True, id="on"),
        pytest.param(None, 8, 7168, False, True, id="default"),
        pytest.param("0", 8, 7168, False, False, id="off"),
        pytest.param("1", 1, 7168, False, False, id="tp 1"),
        pytest.param("1", 8, 7170, False, False, id="uneven"),
        pytest.param("1", 8, 7168, True, False, id="ulysses sp"),
    ],
)
def test_when_the_projection_is_sharded(monkeypatch, flag, tp, hidden, sp, on):
    if flag is None:
        monkeypatch.delenv("ATOM_DSPARK_SHARDED_CONTEXT_PROJ", raising=False)
    else:
        monkeypatch.setenv("ATOM_DSPARK_SHARDED_CONTEXT_PROJ", flag)
    monkeypatch.setattr(
        kimi_k3_dspark, "get_tp_group", lambda: SimpleNamespace(world_size=tp)
    )
    monkeypatch.setattr(kimi_k3_dspark, "sp_is_enabled", lambda: sp)
    model = KimiK3DSpark.__new__(KimiK3DSpark)
    model.hidden_size = hidden
    assert bool(model._can_shard_context_proj()) is on
