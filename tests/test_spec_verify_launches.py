# SPDX-License-Identifier: MIT
"""The MTP verify's launch-saving switches keep its rows, draws and ids:
host-built verification rows (ATOM_SPEC_VERIFY_HOST_ROWS), one logits gather
(ATOM_SPEC_VERIFY_ONE_GATHER) and one TP broadcast
(ATOM_SPEC_VERIFY_ONE_BROADCAST). These run on CPU."""

from types import MethodType, SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

mods = SimpleNamespace()


@pytest.fixture(scope="module", autouse=True)
def _modules():
    with atom_without_aiter() as load:
        mods.runner = load("atom.model_engine.model_runner")
        mods.drafter = load("atom.spec_decode.drafter")
        mods.sampler = load("atom.model_ops.sampler")
        mods.utils = load("atom.utils")
        yield


class _Counts:
    def __init__(self):
        self.counts = {}

    def set_counts(self, counts):
        self.counts.update(counts)


def _drafter(monkeypatch, host_rows, one_gather, max_bs=8, mtp_k=4):
    monkeypatch.setenv("ATOM_SPEC_VERIFY_HOST_ROWS", str(int(host_rows)))
    monkeypatch.setenv("ATOM_SPEC_VERIFY_ONE_GATHER", str(int(one_gather)))
    Drafter = mods.drafter.Drafter

    class MetadataDrafter(Drafter):
        def _resolve_mtp_k(self):
            return mtp_k

        def _propose(self, *args):
            raise NotImplementedError

    runner = SimpleNamespace(arange_np=np.arange(64, dtype=np.int32))
    runner._get_cumsum_and_arange = MethodType(
        mods.runner.ModelRunner._get_cumsum_and_arange, runner
    )
    runner.forward_vars = {
        **Drafter._allocate_metadata_buffers(max_bs, mtp_k, "cpu"),
        "temperatures": mods.utils.CpuGpuBuffer(
            max_bs, dtype=torch.float32, device="cpu", pin_memory=False
        ),
    }
    drafter = MetadataDrafter.__new__(MetadataDrafter)
    drafter.runner = runner
    drafter.mtp_k = mtp_k
    return drafter, runner.forward_vars


@pytest.mark.parametrize(
    "lengths", [[5, 1, 3, 5], [1, 1], [5] * 8, [2, 5, 1], [3]], ids=str
)
def test_host_rows_are_the_device_lookup(monkeypatch, lengths):
    drafter, var = _drafter(monkeypatch, host_rows=True, one_gather=True)
    bs = len(lengths)
    temperatures = np.linspace(0.3, 1.7, bs, dtype=np.float32)
    var["temperatures"].np[:bs] = temperatures
    lengths = np.array(lengths, dtype=np.int32)
    group = _Counts()
    _, drafted = drafter.prepare_spec_decode_indices(lengths, np.cumsum(lengths), group)

    cu = torch.from_numpy(var["cu_num_draft_tokens"].np[:bs].copy())
    rows = torch.searchsorted(cu, torch.arange(drafted, dtype=cu.dtype), right=True)
    host_rows = torch.from_numpy(var["verify_request_indices"].np[:drafted].copy())
    assert host_rows.dtype == rows.dtype and torch.equal(host_rows, rows)
    assert torch.equal(
        torch.from_numpy(var["verify_temperatures"].np[:drafted].copy()),
        torch.from_numpy(temperatures)[rows],
    )
    both = var["verify_logits_indices"].np[: bs + drafted]
    assert both[:bs].tolist() == var["bonus_logits_indices"].np[:bs].tolist()
    assert both[bs:].tolist() == var["target_logits_indices"].np[:drafted].tolist()
    assert group.counts == {
        "target_logits_indices": drafted,
        "cu_num_draft_tokens": bs,
        "bonus_logits_indices": bs,
        "verify_request_indices": drafted,
        "verify_temperatures": drafted,
        "verify_logits_indices": bs + drafted,
    }

    metadata = drafter.calc_spec_decode_metadata(
        torch.zeros(int(lengths.sum()), dtype=torch.int32), (lengths - 1, drafted)
    )
    assert metadata.verify_request_indices.shape == (drafted,)
    assert metadata.verify_temperatures.shape == (drafted,)
    assert metadata.verify_logits_indices.shape == (bs + drafted,)


def test_off_keeps_the_metadata_buffers_and_counts(monkeypatch):
    drafter, var = _drafter(monkeypatch, host_rows=False, one_gather=False)
    assert set(var) == {
        "cu_num_draft_tokens",
        "target_logits_indices",
        "bonus_logits_indices",
        "temperatures",
    }
    lengths = np.array([5, 2], dtype=np.int32)
    group = _Counts()
    prepared = drafter.prepare_spec_decode_indices(lengths, np.cumsum(lengths), group)
    assert group.counts == {
        "target_logits_indices": 5,
        "cu_num_draft_tokens": 2,
        "bonus_logits_indices": 2,
    }
    metadata = drafter.calc_spec_decode_metadata(torch.zeros(7), prepared)
    assert metadata.verify_request_indices is None
    assert metadata.verify_temperatures is None
    assert metadata.verify_logits_indices is None


def test_on_by_default_the_drafter_stages_both(monkeypatch):
    _drafter(monkeypatch, host_rows=True, one_gather=True)
    monkeypatch.delenv("ATOM_SPEC_VERIFY_HOST_ROWS")
    monkeypatch.delenv("ATOM_SPEC_VERIFY_ONE_GATHER")
    buffers = mods.drafter.Drafter._allocate_metadata_buffers(8, 4, "cpu")
    assert {
        "verify_request_indices",
        "verify_temperatures",
        "verify_logits_indices",
    } <= set(buffers)


def test_the_verification_draw_takes_host_rows_without_a_device_lookup(monkeypatch):
    Sampler = mods.sampler.Sampler
    seen = []

    def capture(self, logits, temperatures, top_ks, top_ps, **kwargs):
        seen.append((temperatures, top_ks, top_ps, kwargs))
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    monkeypatch.setattr(Sampler, "forward", capture)
    temperatures = torch.tensor([0.5, 1.0, 1.5])
    args = (
        torch.zeros(5, 8),
        torch.tensor([2, 2, 5], dtype=torch.int32),
        temperatures,
        torch.tensor([3]),
        torch.tensor([0.6, 0.7, 0.8]),
    )
    Sampler().sample_verification_tokens(*args)
    rows = torch.tensor([0, 0, 2, 2, 2])
    with mock.patch.object(torch, "searchsorted", side_effect=AssertionError):
        Sampler().sample_verification_tokens(
            *args, request_indices=rows, row_temperatures=temperatures[rows]
        )
    (t0, k0, p0, kw0), (t1, k1, p1, kw1) = seen
    assert torch.equal(t0, t1) and torch.equal(k0, k1) and torch.equal(p0, p1)
    assert kw0 == kw1


# Request 0 drafts rows 0-1 and takes its bonus from row 2; request 1 drafts
# row 3 and takes its bonus from row 4.
TARGET_ROWS, BONUS_ROWS = torch.tensor([0, 1, 3]), torch.tensor([2, 4])


def _verify_step(monkeypatch, *, rank, one_gather, one_broadcast, received=None):
    """One sampled verify step on TP rank ``rank`` of two. Rank 1 draws
    different ids than rank 0 and receives, from its k-th broadcast, what
    rank 0 sent in its k-th (``received``). Returns the emitted ids, what
    each broadcast sent, and the draws' logits rows."""
    runner_mod = mods.runner
    spec = SimpleNamespace(
        bonus_logits_indices=BONUS_ROWS,
        target_logits_indices=TARGET_ROWS,
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32),
        verify_logits_indices=(
            torch.cat((BONUS_ROWS, TARGET_ROWS)) if one_gather else None
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    sent = []

    def broadcast(tensor, src):
        assert src == 0
        sent.append(tensor.clone())
        return tensor if rank == 0 else received[len(sent) - 1].clone()

    monkeypatch.setattr(
        runner_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, broadcast=broadcast),
    )
    draws = []

    def verification_draw(target_logits, *args, **kwargs):
        draws.append(("verification", target_logits.clone()))
        return torch.tensor([5, 6, 7], dtype=torch.int32) + 10 * rank

    def bonus_draw(logits, **kwargs):
        draws.append(("bonus", logits.clone()))
        return torch.tensor([1, 2], dtype=torch.int32) + 10 * rank

    def verify(metadata, target_logits, bonus, *, target_token_ids, ratio):
        assert ratio is None
        t, b = target_token_ids, bonus
        sampled = torch.stack(
            (torch.stack((t[0], t[1], b[0])), torch.stack((t[2], b[1], b[1] * 0)))
        )
        return sampled, torch.zeros(3, 2, dtype=torch.int32)

    class Emitted(Exception):
        pass

    def prepare_sampled_ids(batch, sampled_tokens, *args, **kwargs):
        raise Emitted(sampled_tokens.clone())

    sampler = mock.Mock(side_effect=bonus_draw)
    sampler.sample_verification_tokens = mock.Mock(side_effect=verification_draw)
    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(
            is_deferred_out=True,
            prev_batch=None,
            prepare_sampled_ids=prepare_sampled_ids,
        ),
        sampler=sampler,
        rejection_sampler=SimpleNamespace(forward=verify),
        drafter=SimpleNamespace(lossless=None),
        forward_done_event=mock.Mock(),
        _served=None,
        structured_output=None,
        _verify_one_broadcast=one_broadcast,
    )
    logits = torch.arange(5 * 8, dtype=torch.float32).view(5, 8)
    with pytest.raises(Emitted) as emitted:
        runner_mod.ModelRunner.postprocess(
            runner,
            SimpleNamespace(
                total_seqs_num=2,
                return_logprobs=[False, False],
                any_logprobs=False,
                required_channel=None,
                structured_requests=None,
                num_spec_step=2,
            ),
            logits,
            torch.ones(2),
            None,
            None,
            False,
            hidden_states=None,
        )
    return emitted.value.args[0], sent, draws


@pytest.fixture
def no_pcp(monkeypatch):
    monkeypatch.setattr(mods.runner, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(mods.runner, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mods.runner, "get_pcp_world_size", lambda: 1)


def _two_ranks(monkeypatch, **flags):
    ids0, sent0, draws0 = _verify_step(monkeypatch, rank=0, **flags)
    ids1, _, draws1 = _verify_step(monkeypatch, rank=1, received=sent0, **flags)
    return ids0, ids1, sent0, draws0, draws1


@pytest.mark.parametrize("one_gather", [False, True])
@pytest.mark.parametrize("one_broadcast", [False, True])
def test_every_rank_emits_rank_zeros_ids(
    monkeypatch, no_pcp, one_gather, one_broadcast
):
    ids0, ids1, sent0, draws0, draws1 = _two_ranks(
        monkeypatch, one_gather=one_gather, one_broadcast=one_broadcast
    )
    reference, _, _, reference_draws, _ = _two_ranks(
        monkeypatch, one_gather=False, one_broadcast=False
    )
    assert torch.equal(ids0, reference) and torch.equal(ids1, reference)
    # Both draws, on every rank, in the same order and from the same rows.
    for draws in (draws0, draws1):
        assert [kind for kind, _ in draws] == ["verification", "bonus"]
        for (_, rows), (_, expected) in zip(draws, reference_draws):
            assert torch.equal(rows, expected)
    if one_broadcast:
        assert len(sent0) == 1
        assert sent0[0].tolist() == [5, 6, 7, 1, 2]
    else:
        assert len(sent0) == 2


def test_one_broadcast_stands_down_under_pcp(monkeypatch, no_pcp):
    monkeypatch.setattr(mods.runner, "pcp_is_enabled", lambda: True)
    monkeypatch.setattr(
        mods.runner,
        "get_pcp_group",
        lambda: SimpleNamespace(broadcast=lambda tensor, src: tensor),
    )
    _, sent, _ = _verify_step(monkeypatch, rank=0, one_gather=False, one_broadcast=True)
    assert [t.tolist() for t in sent] == [[5, 6, 7], [[5, 6, 1], [7, 2, 0]]]
