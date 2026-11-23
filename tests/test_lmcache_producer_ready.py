# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The LMCache producer-ready gate (tools/lmcache_producer_ready).

The GPU wait LMCache issues on ATOM's producer fence must reach the stream only
after a host query has seen the fence complete, for the event imported for the
STORE / RETRIEVE in progress and for no other wait.
"""

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path
from typing import ClassVar

import pytest

GATE_DIR = Path(__file__).resolve().parent.parent / "tools" / "lmcache_producer_ready"
sys.path.insert(0, str(GATE_DIR))

import producer_ready as gate

event_ipc = pytest.importorskip("lmcache.v1.platform.base.event_ipc")


class _Event:
    """A torch.cuda.Event stand-in that is ready after ``pending`` queries."""

    log: ClassVar[list] = []

    def __init__(self, pending=1):
        self.pending = pending

    @classmethod
    def from_ipc_handle(cls, device, handle):
        event = cls(pending=2 if handle == b"producer" else 0)
        cls.log.append(("import", handle, device, event))
        return event

    def query(self):
        self.log.append(("query", self))
        if self.pending:
            self.pending -= 1
            return False
        return True

    def wait(self, stream):
        self.log.append(("wait", self, stream))


@pytest.fixture
def backend():
    class Backend(event_ipc.DefaultEventIPCBackend):
        pass

    Backend.import_event = gate._wrap_import(Backend.import_event)
    Backend.wait_event = gate._wrap_wait(Backend.wait_event)
    _Event.log = []
    return Backend(
        event_module=type("Module", (), {"Event": _Event}), device_type="cuda"
    )


def _kinds():
    return [entry[0] for entry in _Event.log]


def _store(backend, stream, *, extra=None):
    """The shape of LMCacheDrivenTransferModule.store: stage, import the
    producer handle, make the stream wait, then transfer on it."""

    def store(self, key, instance_id, gpu_block_ids, event_ipc_handle):
        producer = backend.import_event(event_ipc_handle, 3)
        backend.wait_event(producer, stream)
        if extra is not None:
            extra()
        return ("stored", key)

    return gate._wrap_transfer(store)


def test_the_producer_wait_is_admitted_only_after_the_fence_is_ready(backend):
    stream = object()
    assert _store(backend, stream)(None, "k", 0, [[1]], b"producer") == ("stored", "k")
    assert _kinds() == ["import", "query", "query", "query", "wait"]
    imported = _Event.log[0][3]
    assert _Event.log[-1][1] is imported and _Event.log[-1][2] is stream


def test_the_handle_is_found_when_passed_by_keyword(backend):
    _store(backend, object())(None, "k", 0, [[1]], event_ipc_handle=b"producer")
    assert _kinds() == ["import", "query", "query", "query", "wait"]


def test_a_wait_outside_a_transfer_is_untouched(backend):
    stream = object()
    event = _Event(pending=5)
    backend.wait_event(event, stream)
    assert _kinds() == ["wait"]


def test_only_the_event_imported_for_the_transfers_handle_is_gated(backend):
    stream = object()

    def other_waits():
        # another handle imported inside the same STORE, then a second wait on
        # the already-admitted producer: neither is polled
        other = backend.import_event(b"other", 3)
        backend.wait_event(other, stream)

    _store(backend, stream, extra=other_waits)(None, "k", 0, [[1]], b"producer")
    assert _kinds() == ["import", "query", "query", "query", "wait", "import", "wait"]


def test_nested_transfers_gate_their_own_handles(backend):
    stream = object()
    inner = _store(backend, stream)

    def nested():
        inner(None, "inner", 0, [[2]], b"producer")

    _store(backend, stream, extra=nested)(None, "outer", 0, [[1]], b"producer")
    assert _kinds() == ["import", "query", "query", "query", "wait"] * 2


def test_the_gate_keeps_no_reference_to_the_imported_event(backend):
    # the event's destructor must still run where LMCache drops it
    _store(backend, object())(None, "k", 0, [[1]], b"producer")
    ref = weakref.ref(_Event.log[0][3])
    _Event.log.clear()
    gc.collect()
    assert ref() is None


def test_a_query_error_propagates_and_the_wait_is_not_issued(backend, monkeypatch):
    error = RuntimeError("hipErrorInvalidHandle")

    def broken(self):
        raise error

    monkeypatch.setattr(_Event, "query", broken)
    with pytest.raises(RuntimeError) as got:
        _store(backend, object())(None, "k", 0, [[1]], b"producer")
    assert got.value is error
    assert "wait" not in _kinds()


def test_a_transfer_without_a_handle_parameter_is_refused():
    def store(self, key, instance_id, gpu_block_ids):
        pass

    with pytest.raises(TypeError, match="event_ipc_handle"):
        gate._wrap_transfer(store)


def test_the_real_transfer_methods_take_the_handle():
    transfer = pytest.importorskip(gate.TRANSFER)
    cls = transfer.LMCacheDrivenTransferModule
    for name in ("store", "retrieve"):
        gate._wrap_transfer(getattr(cls, name))


def test_polling_has_no_deadline_and_warns_while_it_waits(monkeypatch):
    clock = [0.0]
    pauses = []
    warnings = []

    def pause(seconds):
        pauses.append(seconds)
        clock[0] += 0.5

    class Log:
        @staticmethod
        def warning(fmt, waited, where):
            warnings.append(waited)

    answers = iter([False] * 200 + [True])
    monkeypatch.setattr(gate, "_logger", Log)
    polls, waited = gate.poll_ready(
        lambda: next(answers), clock=lambda: clock[0], pause=pause
    )
    assert polls == 201 and waited == pytest.approx(100.0)
    # fast polls for the first 0.1 s, then the slow interval
    assert pauses[0] == gate.POLL_S and set(pauses[1:]) == {gate.SLOW_POLL_S}
    # at 10 s, then every 60 s
    assert [round(w) for w in warnings] == [10, 70]


def test_counts_are_logged_at_the_first_gated_wait_then_once_a_minute(monkeypatch):
    lines = []

    class Log:
        @staticmethod
        def info(fmt, snapshot):
            lines.append(snapshot)

    monkeypatch.setattr(gate, "_logger", Log)
    monkeypatch.setattr(gate, "_STATS", dict.fromkeys(gate._STATS, 0))
    monkeypatch.setattr(gate, "_stats_logged_at", None)
    for t in (0.0, 1.0, 59.0, 61.0):
        gate._count(clock=lambda t=t: t, transfers=1)
        gate._count(clock=lambda t=t: t, gated=1, blocked=1, wait_s=t / 100)
    assert [s["gated"] for s in lines] == [1, 4]
    assert lines[-1]["transfers"] == 4 and lines[-1]["max_wait_s"] == pytest.approx(
        0.61
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(gate.ENABLED_ENV, raising=False)
    before = list(sys.meta_path)
    assert gate.install_from_env() is False
    assert sys.meta_path == before


def _run(code, env_extra, pythonpath):
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", gate.ENABLED_ENV)
    }
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


_PATCHED = f"""
import importlib, json
mods = [importlib.import_module(n) for n in ({gate.EVENT_IPC!r}, {gate.TRANSFER!r})]
print(json.dumps([bool(getattr(m, "_atom_producer_ready_patched", False)) for m in mods]))
"""


@pytest.mark.parametrize("enabled", [True, False])
def test_the_startup_hook_patches_lmcache_when_it_is_imported(enabled):
    pytest.importorskip(gate.TRANSFER)
    env = {gate.ENABLED_ENV: "1"} if enabled else {}
    res = _run(_PATCHED, env, [GATE_DIR])
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == (
        "[true, true]" if enabled else "[false, false]"
    )
    assert ("producer-ready gate installed" in res.stderr) is enabled


def test_the_startup_hook_chains_behind_step_timing_and_to_the_next_one(tmp_path):
    marker = tmp_path / "order.txt"
    steptiming = tmp_path / "harness" / "steptiming"
    steptiming.mkdir(parents=True)
    # the harness hook: record, then chain to the next sitecustomize but itself
    (steptiming / "sitecustomize.py").write_text(textwrap.dedent(f"""
            import importlib.machinery, importlib.util, os, sys
            open({str(marker)!r}, "a").write("steptiming\\n")
            here = os.path.dirname(os.path.abspath(__file__))
            path = [p for p in sys.path if os.path.abspath(p or ".") != here]
            spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            """))
    system = tmp_path / "system"
    system.mkdir()
    (system / "sitecustomize.py").write_text(
        f"open({str(marker)!r}, 'a').write('system\\n')\n"
    )
    code = f"""
    import sys
    installed = getattr(sys, "_atom_producer_ready_installed", False)
    open({str(marker)!r}, "a").write(f"installed={{installed}}\\n")
    """
    res = _run(code, {gate.ENABLED_ENV: "1"}, [steptiming, GATE_DIR, system])
    assert res.returncode == 0, res.stderr
    assert marker.read_text().split() == ["steptiming", "system", "installed=True"]
