# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono held off while LMCache MP transfers are in flight (`mono.runtime.kv_hold`).

CPU only: the scheduler's mark, what every TP rank decides from it, the
recordings a capture asks for, and the K3 wrapper's ``hold_off``.
"""

import pickle
from types import SimpleNamespace

import torch

from atom.kv_transfer.offload.mp import connector as mp_connector
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp.metadata import MPOffloadMetadata
from atom.mono.runtime import kv_hold
from atom.mono.runtime.kv_hold import HOLD_REASON, MONO_OFF_PHASE, MonoStepRouter

MONO_BUCKET = (4, 5)
WIDE_BUCKET = (13, 5)


class _Impl:
    """A scheduler whose live transfers a test sets."""

    def __init__(self):
        self.live = set()

    def build_connector_meta(self):
        return MPOffloadMetadata()

    def _live_transfers(self):
        return set(self.live)


def _shell(impl):
    shell = mp_connector.LMCacheMPConnectorScheduler.__new__(
        mp_connector.LMCacheMPConnectorScheduler
    )
    shell._impl = impl
    return shell


def _batch(shell):
    return SimpleNamespace(connector_meta_output=shell.build_connector_meta())


class _Wrapper:
    """What the router reads of the K3 mono wrapper."""

    def __init__(self):
        self.hold_off = None
        self.last_served = False
        self.last_decline = None


def _captured_router(log=False):
    """A router after a capture: one mono bucket, one the build declined."""
    wrapper = _Wrapper()
    router = MonoStepRouter(wrapper, enabled=True, log=log)
    wrapper.last_served = True
    assert router.note_capture(MONO_BUCKET)
    wrapper.last_served, wrapper.last_decline = False, "65 rows > the 64-row buffers"
    assert not router.note_capture(WIDE_BUCKET)
    return router


def test_the_scheduler_marks_a_step_while_a_transfer_is_live():
    impl = _Impl()
    shell = _shell(impl)
    assert shell.build_connector_meta().transfers_live is False
    impl.live.add("save-207")
    assert shell.build_connector_meta().transfers_live is True
    impl.live.clear()
    assert shell.build_connector_meta().transfers_live is False


def test_the_real_scheduler_counts_saves_and_dispatched_loads():
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._save_inflight, sched._save_operation_owner = {}, {}
    sched._active_load_operations = {}
    assert not sched._live_transfers()
    sched._save_inflight["207"] = "save-op"
    assert sched._live_transfers() == {"save-op"}
    del sched._save_inflight["207"]
    sched._active_load_operations["208"] = (object(), "load-op")
    assert sched._live_transfers() == {"load-op"}


def test_every_rank_decides_the_same_from_its_copy_of_the_batch():
    """Each rank unpickles its own copy of the scheduler's batch; eight
    routers captured alike pick the same recording on every step."""
    impl = _Impl()
    shell = _shell(impl)
    ranks = [_captured_router() for _ in range(8)]
    for live in (False, True, True, False, True, False):
        impl.live = {"op"} if live else set()
        wire = pickle.dumps(_batch(shell))
        for key in (MONO_BUCKET, WIDE_BUCKET):
            phases = {
                router.graph_phase(pickle.loads(wire), key, key == MONO_BUCKET)
                for router in ranks
            }
            assert len(phases) == 1
    assert len({tuple(sorted(r.counts.items())) for r in ranks}) == 1


def test_mono_returns_once_the_scheduler_has_seen_the_completion():
    impl = _Impl()
    shell = _shell(impl)
    router = _captured_router()
    phases = []
    for live in (False, True, True, False, False):
        impl.live = {"save-207"} if live else set()
        phases.append(router.graph_phase(_batch(shell), MONO_BUCKET, True))
    assert phases == [None, MONO_OFF_PHASE, MONO_OFF_PHASE, None, None]
    assert router.counts == {kv_hold.SERVED: 3, f"declined: {HOLD_REASON}": 2}


def test_a_bucket_mono_never_served_keeps_its_one_recording():
    impl = _Impl()
    impl.live = {"op"}
    router = _captured_router()
    assert router.graph_phase(_batch(_shell(impl)), WIDE_BUCKET, False) is None
    assert router.counts == {"declined: 65 rows > the 64-row buffers": 1}


def test_a_bucket_without_a_production_recording_stays_on_mono():
    """`has_off` False (no recording to switch to) never picks the phase."""
    impl = _Impl()
    impl.live = {"op"}
    router = _captured_router()
    assert router.graph_phase(_batch(_shell(impl)), MONO_BUCKET, False) is None


def test_eager_decode_holds_off_and_prefill_is_not_counted():
    impl = _Impl()
    shell = _shell(impl)
    router = _captured_router()
    impl.live = {"op"}
    assert router.eager_hold(_batch(shell), is_prefill=True) is None
    assert router.eager_hold(_batch(shell), is_prefill=False) == HOLD_REASON
    impl.live = set()
    assert router.eager_hold(_batch(shell), is_prefill=False) is None
    assert router.counts == {f"eager, declined: {HOLD_REASON}": 1, "eager": 1}


def test_held_off_restores_the_wrapper():
    router = _captured_router()
    with router.held_off():
        assert router.model.hold_off == HOLD_REASON
    assert router.model.hold_off is None


def test_inactive_without_a_mono_wrapper_or_when_switched_off():
    assert not MonoStepRouter(torch.nn.Identity(), enabled=True).active
    assert not MonoStepRouter(_Wrapper(), enabled=False).active
    assert MonoStepRouter(_Wrapper(), enabled=True).active


def test_a_batch_without_connector_metadata_is_not_held():
    assert not kv_hold.transfers_live(SimpleNamespace(connector_meta_output=None))
    assert not kv_hold.transfers_live(SimpleNamespace())


def test_counts_are_logged_every_log_every_steps(monkeypatch, caplog):
    monkeypatch.setattr(kv_hold, "LOG_EVERY", 2)
    router = _captured_router(log=True)
    batch = SimpleNamespace(connector_meta_output=None)
    with caplog.at_level("INFO", logger="atom"):
        for _ in range(4):
            router.graph_phase(batch, MONO_BUCKET, True)
    lines = [r.message for r in caplog.records if "mono step routing" in r.message]
    assert lines == [
        "Kimi-K3 mono step routing after 2 decode steps: served=2",
        "Kimi-K3 mono step routing after 4 decode steps: served=4",
    ]


def test_each_switch_is_logged_once_with_its_time(caplog):
    impl = _Impl()
    shell = _shell(impl)
    router = _captured_router(log=True)
    with caplog.at_level("INFO", logger="atom"):
        for live in (False, True, True, False, False, True):
            impl.live = {"op"} if live else set()
            router.graph_phase(_batch(shell), MONO_BUCKET, True)
    switches = [
        r.message.split(" at ")[0]
        for r in caplog.records
        if r.message.startswith("Kimi-K3 mono held off")
        or r.message.startswith("Kimi-K3 mono resumed")
    ]
    held = f"Kimi-K3 mono held off: {HOLD_REASON}"
    assert switches == [held, "Kimi-K3 mono resumed", held]


def _k3_wrapper(monkeypatch, requests=2, qlen=5):
    from atom.models.kimi_k3.mono import dispatch

    metadata = SimpleNamespace(
        spec_sequence_masks=object(),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=requests,
        num_actual_tokens=requests * qlen,
        spec_state_indices_tensor=torch.zeros(requests, qlen, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
        replayssm=False,
        has_state_fork=False,
    )
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: context)
    return dispatch.MonoDecodeModel(torch.nn.Identity()), torch.zeros(requests * qlen)


def test_the_k3_wrapper_declines_under_hold_off_and_serves_after(monkeypatch, caplog):
    wrapper, ids = _k3_wrapper(monkeypatch)
    assert wrapper._qlen(ids, None, None) == 5
    with caplog.at_level("INFO", logger="atom"):
        for _ in range(3):
            wrapper.hold_off = HOLD_REASON
            assert wrapper._qlen(ids, None, None) is None
            assert wrapper.last_decline == HOLD_REASON
            wrapper.hold_off = None
            assert wrapper._qlen(ids, None, None) == 5
    declined = [r.message for r in caplog.records if HOLD_REASON in r.message]
    assert declined == [f"Kimi-K3 mono decode declined a step: {HOLD_REASON}"]
