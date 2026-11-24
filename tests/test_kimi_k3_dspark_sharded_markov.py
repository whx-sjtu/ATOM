# SPDX-License-Identifier: MIT
"""ATOM_DSPARK_SHARDED_MARKOV: Kimi-K3's greedy draft block keeps the LM head
sharded and samples every position with `dspark_markov_argmax_tp`; anything
it cannot cover takes the base head and block."""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.models import kimi_k3_dspark
    from atom.models.dspark_draft import DSparkDraftModel
    from atom.models.kimi_k3_dspark import KimiK3DSpark
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the DSpark draft imports AITER")

TP, RANK, V, R, H, B, T = 4, 2, 32, 3, 8, 3, 4
SHARD = V // TP


def _whole_vocab():
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(V, H, generator=g),
        torch.randn(V, R, generator=g),
        torch.randn(V, R, generator=g),
    )


def _model(sharded=True, tp=TP, sampling=None):
    head_w, w1, w2 = _whole_vocab()
    model = KimiK3DSpark.__new__(KimiK3DSpark)
    torch.nn.Module.__init__(model)
    model.sharded_markov = sharded
    model.draft_sampling = sampling
    model.lm_head = SimpleNamespace(
        tp_size=tp,
        bias=None,
        weight=head_w[RANK * SHARD : (RANK + 1) * SHARD],
        vocab_start_idx=RANK * SHARD,
        full=head_w,
    )
    model.markov_head = SimpleNamespace(
        fused_sample=True,
        markov_w1=SimpleNamespace(weight=w1),
        markov_w2=SimpleNamespace(weight=w2),
    )
    return model


@pytest.fixture
def ops(monkeypatch):
    """The TP op answers what the whole-vocab argmax would, off the full head
    the test keeps aside, and checks it was handed this rank's shard."""
    calls = []
    hidden = {}

    def mm(x, w):
        hidden["x"] = x
        return x @ w.t()

    def argmax_tp(base, prev, w1, w2, out, vocab_start, group, use_custom):
        k = len(calls)
        calls.append({"prev": prev.clone(), "vocab_start": vocab_start, "group": group})
        full = (hidden["x"] @ head_full.t()).view(B, T, V)[:, k]
        torch.testing.assert_close(base, full[:, vocab_start : vocab_start + SHARD])
        out.copy_((full + w1[prev] @ w2.t()).argmax(-1))

    group = object()
    head_full = _whole_vocab()[0]
    monkeypatch.setattr(kimi_k3_dspark, "tgemm", SimpleNamespace(mm=mm))
    monkeypatch.setattr(kimi_k3_dspark, "dspark_markov_argmax_tp", argmax_tp)
    monkeypatch.setattr(kimi_k3_dspark, "get_tp_group", lambda: group)
    monkeypatch.setattr(kimi_k3_dspark, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(kimi_k3_dspark, "markov_weights_unsupported", lambda *a: None)
    return calls, group


@pytest.fixture
def base(monkeypatch):
    seen = []

    def head_and_sample(self, out, anchor_ids, num_draft, *, greedy=False):
        seen.append(greedy)
        return "base", None

    monkeypatch.setattr(DSparkDraftModel, "head_and_sample", head_and_sample)
    return seen


def _reference(model, out, anchor):
    """`DSparkMarkovHead.sample_block`'s greedy loop over the whole vocab."""
    logits = (out @ model.lm_head.full.t()).view(B, T, V)
    w1, w2 = model.markov_head.markov_w1.weight, model.markov_head.markov_w2.weight
    ids = [anchor]
    for k in range(T):
        ids.append((logits[:, k] + w1[ids[-1]] @ w2.t()).argmax(-1))
    return torch.stack(ids[1:], dim=1)


def test_the_greedy_block_samples_off_this_ranks_shard(ops, base):
    calls, group = ops
    model = _model()
    out = torch.randn(B * T, H)
    anchor = torch.tensor([1, 7, 30])
    drafts, confidence = model.head_and_sample(out, anchor, T)
    assert confidence is None and base == []
    assert torch.equal(drafts, _reference(model, out, anchor))
    assert len(calls) == T
    assert all(c["vocab_start"] == RANK * SHARD and c["group"] is group for c in calls)
    # each position starts from the previous one's id, the anchor first
    assert torch.equal(calls[0]["prev"], anchor)
    for k in range(1, T):
        assert torch.equal(calls[k]["prev"], drafts[:, k - 1])


@pytest.mark.parametrize(
    "model, greedy",
    [
        pytest.param(lambda: _model(sharded=False), False, id="switch off"),
        pytest.param(lambda: _model(tp=1), False, id="tp 1"),
        pytest.param(lambda: _model(sampling=object()), False, id="lossless armed"),
    ],
)
def test_otherwise_the_base_head_and_block(ops, base, model, greedy):
    calls, _ = ops
    assert model().head_and_sample(torch.randn(B * T, H), torch.zeros(B), T) == (
        "base",
        None,
    )
    assert calls == [] and base == [greedy]


def test_a_greedy_step_of_an_armed_draft_stays_sharded(ops, base):
    calls, _ = ops
    model = _model(sampling=object())
    model.head_and_sample(
        torch.randn(B * T, H), torch.zeros(B, dtype=torch.long), T, greedy=True
    )
    assert len(calls) == T and base == []


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param(
            lambda m: setattr(m.lm_head, "bias", torch.zeros(SHARD)), id="head bias"
        ),
        pytest.param(
            lambda m: setattr(m.markov_head, "fused_sample", False), id="unfused markov"
        ),
        pytest.param(lambda m: setattr(m, "lm_head", None), id="no head"),
    ],
)
def test_what_the_sharded_block_cannot_cover(ops, patch):
    model = _model()
    patch(model)
    assert not model._sharded_markov_greedy()


def test_not_under_ulysses_sp(ops, monkeypatch):
    monkeypatch.setattr(kimi_k3_dspark, "sp_is_enabled", lambda: True)
    assert not _model()._sharded_markov_greedy()


@pytest.mark.parametrize("flag, on", [(None, False), ("1", True), ("0", False)])
def test_the_switch_is_off_by_default(monkeypatch, flag, on):
    if flag is None:
        monkeypatch.delenv("ATOM_DSPARK_SHARDED_MARKOV", raising=False)
    else:
        monkeypatch.setenv("ATOM_DSPARK_SHARDED_MARKOV", flag)
    from atom.utils import envs

    assert bool(envs.ATOM_DSPARK_SHARDED_MARKOV) is on
