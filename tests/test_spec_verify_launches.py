# SPDX-License-Identifier: MIT
"""The MTP verify's launch-saving switches keep its rows, draws and ids:
host-built verification rows (ATOM_SPEC_VERIFY_HOST_ROWS). These run on
CPU."""

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


def _drafter(monkeypatch, *, host_rows, max_bs=8, mtp_k=4):
    monkeypatch.setenv("ATOM_SPEC_VERIFY_HOST_ROWS", str(int(host_rows)))
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
    drafter, var = _drafter(monkeypatch, host_rows=True)
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
    assert group.counts == {
        "target_logits_indices": drafted,
        "cu_num_draft_tokens": bs,
        "bonus_logits_indices": bs,
        "verify_request_indices": drafted,
        "verify_temperatures": drafted,
    }

    metadata = drafter.calc_spec_decode_metadata(
        torch.zeros(int(lengths.sum()), dtype=torch.int32), (lengths - 1, drafted)
    )
    assert metadata.verify_request_indices.shape == (drafted,)
    assert metadata.verify_temperatures.shape == (drafted,)


def test_off_keeps_the_metadata_buffers_and_counts(monkeypatch):
    drafter, var = _drafter(monkeypatch, host_rows=False)
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
