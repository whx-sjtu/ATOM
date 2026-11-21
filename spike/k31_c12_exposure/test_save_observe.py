"""CPU-only differential checks of native save observation and controller bounds."""

import copy
import gc
import importlib.util
import json
import logging
import sys
import threading
import time
import types
import weakref
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
import plan
import save_observe as S


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    value = plan.make_plan(tmp_path / "out", "save_completion_observer")
    out = Path(value["output"])
    out.mkdir()
    (out / "plan.json").write_text(json.dumps(value))
    (out / "save-observer-control.json").write_text(
        json.dumps(
            {
                "head": value["head"],
                "launch_mono": time.monotonic(),
                "evidence_end_mono": time.monotonic() + 1800,
                "stop": None,
            }
        )
    )
    monkeypatch.setattr(S, "PLAN", value)
    monkeypatch.setattr(S, "RUNTIME", None)
    monkeypatch.setattr(S, "BOOT_ERRORS", [])
    S.TLS.context = None
    return value


def recorder(frozen, role="worker", rank=0):
    S.RUNTIME = S.Recorder(frozen, role, None if role == "manager" else rank)
    S.RUNTIME.enter("worker" if role == "worker" else "engine_core")
    if role == "manager":
        S.RUNTIME.enter("manager")
    return S.RUNTIME


@pytest.fixture
def native(monkeypatch):
    from tests.aiter_stub import atom_under_stubs

    installed = S.LMC
    with atom_under_stubs():
        packages = [
            "lmcache",
            "lmcache.v1",
            "lmcache.v1.platform",
            "lmcache.v1.platform.base",
            "lmcache.v1.multiprocess",
            "lmcache.v1.mp_observability",
            "lmcache.integration",
            "lmcache.integration.atom",
            "lmcache.v1.multiprocess.transport",
        ]
        for name in packages:
            module = types.ModuleType(name)
            module.__path__ = [str(installed / Path(*name.split(".")[1:]))]
            monkeypatch.setitem(sys.modules, name, module)

        def forbidden(*args, **kwargs):
            raise AssertionError("implicit device/network boundary")

        stubs = {
            "lmcache.utils": {
                "lmcache_deprecate": lambda message: lambda fn: fn,
                "EngineType": NS(ATOM="atom", VLLM="vllm"),
                "init_logger": logging.getLogger,
            },
            "lmcache.logging": {"init_logger": logging.getLogger},
            "lmcache.v1.mp_observability.event": {
                "Event": object,
                "EventType": NS(TIMEOUT_RAISED="timeout"),
            },
            "lmcache.v1.mp_observability.event_bus": {
                "is_observability_enabled": lambda: False,
                "get_event_bus": forbidden,
            },
            "lmcache.v1.multiprocess.custom_types": {"IPCCacheServerKey": object},
            "lmcache.v1.multiprocess.group_view": {
                "EngineGroupInfo": object,
                "expand_engine_block_ids": lambda infos, blocks: blocks,
            },
            "lmcache.v1.multiprocess.transfer_context": {
                "TransferContext": object,
                "create_transfer_context": forbidden,
            },
            "lmcache.v1.multiprocess.transport.base": {"RequestClient": object},
            "lmcache.v1.multiprocess.transport.factory": {
                "RequestClientFactory": object
            },
            "lmcache.v1.periodic_thread": {
                "PeriodicThread": object,
                "ThreadLevel": NS(CRITICAL="critical"),
                "ThreadRunSummary": NS,
            },
        }
        for name, fields in stubs.items():
            module = types.ModuleType(name)
            module.__dict__.update(fields)
            monkeypatch.setitem(sys.modules, name, module)
        sys.modules["lmcache"].torch_dev = NS(current_device=forbidden)
        sys.modules["lmcache"].torch_device_type = "cpu-fixture"
        sys.modules["lmcache.v1.platform"].get_device_spec = forbidden

        def load(name, relative):
            spec = importlib.util.spec_from_file_location(name, installed / relative)
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            return module

        load("lmcache.v1.mp_observability.errors", "v1/mp_observability/errors.py")
        events = load(
            "lmcache.v1.platform.base.event_ipc", "v1/platform/base/event_ipc.py"
        )
        futures = load("lmcache.v1.multiprocess.futures", "v1/multiprocess/futures.py")
        adapter = load(
            "lmcache.integration.atom.multi_process_adapter",
            "integration/atom/multi_process_adapter.py",
        )
        sys.modules["lmcache.integration.atom"].AtomMPTransferSpec = (
            adapter.AtomMPTransferSpec
        )
        from atom.kv_transfer.disaggregation import aggregator
        from atom.kv_transfer.disaggregation import types as kvtypes
        from atom.kv_transfer.offload.mp import native_state_worker, transfer

        yield NS(
            events=events,
            futures=futures,
            adapter=adapter,
            transfer=transfer,
            worker=native_state_worker,
            aggregator=aggregator,
            types=kvtypes,
        )


class FixtureFault(RuntimeError):
    pass


def future_case(
    native,
    *,
    pending=False,
    values=None,
    value=True,
    fail=None,
    prepared=False,
    missing=False,
    extra_query=False,
):
    calls, event_refs = [], []

    class Event:
        def __init__(self, interprocess=False):
            self.values = list(values or [True])
            event_refs.append(weakref.ref(self))

        @classmethod
        def from_ipc_handle(cls, device, handle):
            calls.append(("event.import", device))
            if fail == "import":
                raise FixtureFault("import")
            return cls()

        def query(self):
            calls.append(("event.query", None))
            if fail == "event_query":
                raise FixtureFault("event_query")
            return self.values.pop(0) if len(self.values) > 1 else self.values[0]

        def synchronize(self):
            calls.append(("event.synchronize", None))
            if fail == "sync":
                raise FixtureFault("sync")

        def wait(self, *args):
            raise AssertionError("added event wait")

        def record(self, *args):
            raise AssertionError("added event record")

    class Raw(native.futures.MessagingFuture):
        def query(self):
            calls.append(("raw.query", None))
            if fail == "raw_query":
                raise FixtureFault("raw_query")
            return super().query()

        def result(self, timeout=None):
            calls.append(("raw.result", timeout))
            return super().result(timeout)

        def wait(self, timeout=None):
            calls.append(("raw.wait", timeout))
            return super().wait(timeout)

    class Device(native.futures.DeviceMessagingFuture):
        def query(self):
            calls.append(("device.query", None))
            return super().query()

        def result(self, timeout=None):
            calls.append(("device.result", timeout))
            return super().result(timeout)

        def wait(self, timeout=None):
            calls.append(("device.wait", timeout))
            return super().wait(timeout)

    raw = Raw()
    if not pending:
        if fail == "raw_result":
            raw.set_exception(FixtureFault("raw_result"))
        else:
            raw.set_result((b"fixture-event" if values is not None else b"", value))
    backend = native.events.DefaultEventIPCBackend(
        event_module=NS(Event=Event), device_type="cpu-fixture"
    )
    future = Device(raw, device=0, event_backend=backend)
    if prepared:
        future._prepare(timeout=0)
        calls.clear()
    results = []
    try:
        if extra_query:
            future.query()
        for unused in range(len(values) if values and len(values) > 1 else 1):
            results.append(
                native.transfer._terminal_future_result(None if missing else future)
            )
    except native.transfer.LMCacheTransferUnprovable as error:
        results.append((type(error).__name__, type(error.__cause__).__name__))
        if error.__cause__ is not None:
            error.__cause__.__traceback__ = None
        error.__traceback__ = None
    refs = [weakref.ref(raw), weakref.ref(future), *event_refs]
    del raw, future, backend
    gc.collect()
    assert all(ref() is None for ref in refs)
    return calls, results


CASES = [
    {"pending": True},
    {"values": [False, False, True]},
    {},
    {"value": False},
    {"values": [False, True], "prepared": True},
    {"missing": True},
    {"fail": "raw_query"},
    {"fail": "raw_result"},
    {"fail": "import", "values": [True]},
    {"fail": "event_query", "values": [True]},
    {"fail": "sync", "values": [True]},
]


@pytest.mark.parametrize("case", CASES)
def test_native_query_order_and_lifetime(frozen, native, case):
    S.RUNTIME = None
    baseline = future_case(native, **case)
    for module in [native.events, native.futures, native.transfer]:
        S.patch(module)
    rec = recorder(frozen)
    with S.context((130, 71), phase="poll"):
        observed = future_case(native, **case)
    assert getattr(S.TLS, "context", None) is None
    assert observed == baseline
    assert S.plain(rec.data["operations"]) == rec.data["operations"]
    if case == {"values": [False, False, True]}:
        stages = rec.data["operations"]["71"]["stages"]
        assert stages["raw_query"]["count"] == 1
        assert stages["event_query"]["count"] == 3
        assert sum(name == "event.synchronize" for name, _ in observed[0]) == 1


def test_extra_query_control_and_submit_scope(frozen, native):
    for module in [native.events, native.futures, native.transfer]:
        S.patch(module)
    rec = recorder(frozen)
    with S.context((130, 1), phase="submit"):
        baseline = future_case(native, pending=True)
    assert rec.data["operations"] == {}
    with S.context((130, 1), phase="poll"):
        extra = future_case(native, pending=True, extra_query=True)
    with pytest.raises(AssertionError, match="native call ledger differs"):
        assert extra == baseline, "native call ledger differs"


def test_selector_caps_writer_failure_and_cutoff(frozen, monkeypatch):
    rec = recorder(frozen)
    rec.sample("x", (129, 1), {"value": True})
    assert not rec.data["operations"]
    for gen in range(1, 10):
        rec.sample("x", (130, gen), {"value": False})
    assert set(rec.data["operations"]) == {str(g) for g in range(1, 9)}
    assert "generation cap" in rec.data["unavailable"]
    for unused in range(100):
        rec.sample("x", (130, 1), {"value": False})
    assert len(rec.data["records"]) == 8
    assert rec.data["operations"]["1"]["stages"]["x"]["count"] == 101
    rec.max_records = len(rec.data["records"])
    rec.sample("changed", (130, 1), {"value": True})
    assert "record cap" in rec.data["unavailable"]
    rec.max_bytes = 128
    rec.publish()
    assert rec.path.stat().st_size <= 128
    original = Path.open

    def fail_write(path, mode="r", *args, **kwargs):
        if path == rec.path and "w" in mode:
            raise OSError("fixture write failure")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_write)
    rec.publish()
    assert "write:OSError" in rec.data["unavailable"]
    control = json.loads(rec.control_path.read_text())
    control["evidence_end_mono"] = time.monotonic() - 1
    rec.control_path.write_text(json.dumps(control))
    rec.publish()
    assert rec.disabled and S.runtime() is None


def test_modes_and_clock_latch(frozen):
    assert frozen["invocations"] == ["save_completion_observer"]
    for role, env in frozen["environments"].items():
        assert ("K31_SAVE_OBSERVER_PLAN" in env) == (role == "server")
        assert "K31_OBSERVER_PLAN" not in env
        assert "TORCH_NCCL_TRACE_BUFFER_SIZE" not in env
    clock = S.SaveClock(frozen["protocol"]["save_observer"], 100)
    assert clock.deadline() == 1900
    clock.start_client(1400)
    assert clock.deadline() == 1900
    clock.phases["warmup"] = {"observed_mono": 1500}
    assert clock.deadline() == 1900
    clock.latch("native-failure", 1898)
    clock.latch("invocation-end", 1900)
    assert clock.reason["reason"] == "native-failure" and clock.grace_end() == 1913
    expired = S.SaveClock(clock.config, 100)
    expired.start_client(1901)
    assert expired.reason["reason"] == "deadline" and expired.grace_end() == 1915


def native_save_case(native, mode, clear_logs):
    from atom.kv_transfer.offload.metadata import NativeStateTransfer, SaveSpec

    calls = []
    raw = native.futures.MessagingFuture()
    if mode == "query_error":
        raw.set_exception(FixtureFault("raw result"))
    else:
        raw.set_result((b"", mode != "false"))
    backend = native.events.DefaultEventIPCBackend(
        event_module=NS(Event=object), device_type="cpu-fixture"
    )
    future = native.futures.DeviceMessagingFuture(raw, device=0, event_backend=backend)

    class Transport:
        def __init__(self, selected_future):
            self.future = None if mode == "submit_error" else selected_future

        def submit_store(self, session, key, caches, blocks, event, chunk):
            calls.append(("transport.submit_store", session))
            if mode == "submit_error":
                raise FixtureFault("transport")
            return self.future

    adapter = native.adapter.AtomMPWorkerAdapter.__new__(
        native.adapter.AtomMPWorkerAdapter
    )
    adapter._state_lock = threading.RLock()
    adapter._state_changed = threading.Condition(adapter._state_lock)
    adapter._closed = False
    adapter._health_event = threading.Event()
    if mode != "missing":
        adapter._health_event.set()
    adapter._transfer_context = Transport(future)
    adapter._kv_caches, adapter._engine_group_infos = {}, []
    adapter._context_submission_leases, adapter._context_operation_futures = {}, {}
    adapter.blocks_in_chunk = 1
    adapter._create_key = lambda session, spec: ("fixture-key", session)
    worker = native.worker.NativeStateLMCacheMPConnector.__new__(
        native.worker.NativeStateLMCacheMPConnector
    )
    worker._lock = threading.RLock()
    worker._native_saves, worker._native_loads = {}, {}
    worker._completed_save_operations, worker._completed_load_operations = set(), set()
    worker._completed_save_operation_order = deque()
    worker._is_kv_writer, worker._max_pending_saves = True, 4
    worker._adapter, worker.chunk_size = adapter, 4
    worker._transfer_deadline_s = 30.0
    worker._config = NS(
        parallel_config=NS(server_instance_id="fixture", data_parallel_rank=0)
    )
    worker._native_block_ids = lambda *args, **kwargs: [[0, 1, 2, 3]]
    worker._submit_kwargs = dict
    op = native.types.SaveOperationId(130, 17)
    req = native.worker.MPReqMeta(
        req_id=130,
        token_ids=list(range(16)),
        block_ids=[0, 1, 2, 3],
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        save_operation=op,
        native_state=NativeStateTransfer((1,), 16, 123),
        is_last_prefill=True,
    )
    producer = NS()
    worker._submit_native(req, producer, loading=False)
    output = native.types.KVConnectorOutput()
    result = None
    try:
        worker._poll_native_save(
            output, "save:130:17", worker._native_saves["save:130:17"]
        )
    except native.transfer.LMCacheTransferUnprovable as error:
        result = type(error).__name__
        if error.__cause__ is not None:
            error.__cause__.__traceback__ = None
        error.__traceback__ = None
    result = {
        "calls": calls,
        "exception": result,
        "pending_keys": sorted(worker._native_saves),
        "terminal": sorted(
            (c.channel, c.operation_id.req_id, c.operation_id.generation, c.succeeded)
            for c in output.connector_completions
            if c.channel == S.CHANNEL
        ),
    }
    refs = [weakref.ref(future), weakref.ref(raw)]
    del worker, adapter, future, raw, backend, req, output, producer
    # pytest owns exception LogRecords; dropping caller-owned capture is not runtime machinery.
    clear_logs()
    gc.collect()
    assert all(ref() is None for ref in refs)
    return result


@pytest.mark.parametrize(
    "mode", ["true", "false", "missing", "submit_error", "query_error"]
)
def test_actual_native_submission_poll_and_report(frozen, native, mode, caplog):
    S.RUNTIME = None
    baseline = native_save_case(native, mode, caplog.clear)
    for module in [
        native.events,
        native.futures,
        native.adapter,
        native.transfer,
        native.worker,
    ]:
        S.patch(module)
    rec = recorder(frozen)
    rec.data["current"] = {"ordinal": 3}
    observed = native_save_case(native, mode, caplog.clear)
    assert observed == baseline
    assert getattr(S.TLS, "context", None) is None
    stages = rec.data["operations"]["17"]["stages"]
    assert stages["adapter_enter"]["value"]["session"] == "atom-offload-fixture-dp0:130"
    if mode in ("true", "false", "missing"):
        assert stages["worker_terminal"]["value"]["succeeded"] is (mode == "true")
        assert stages["worker_terminal"]["value"]["aggregation_ordinal"] == 3
    if mode == "submit_error":
        assert stages["adapter_return"]["value"]["exception"] == "FixtureFault"
        assert stages["terminal"]["value"]["terminal"] is False
    assert S.plain(rec.data["operations"]) == rec.data["operations"]


def test_native_quorum_duplicate_generations_and_settlement(frozen, native):
    from atom.kv_transfer.offload.mp import native_state_scheduler as N

    for module in [native.aggregator, N]:
        S.patch(module)
    rec = recorder(frozen, "manager")
    rec.data["current"] = {"ordinal": 9, "rank_mask": 255, "receipts": {}}
    op = native.types.SaveOperationId(130, 23)
    outputs = [
        native.types.KVConnectorOutput(
            connector_completions={
                native.types.ConnectorCompletion(S.CHANNEL, op, succeeded=(rank != 0))
            }
        )
        for rank in range(8)
    ]
    outputs[0].connector_completions.add(
        native.types.ConnectorCompletion(S.CHANNEL, op, succeeded=True)
    )
    aggregator = native.aggregator.KVOutputAggregator(world_size=8)
    result = aggregator.aggregate(outputs)
    assert result.connector_completions == {
        native.types.ConnectorCompletion(S.CHANNEL, op, succeeded=False)
    }
    assert not aggregator.aggregate(outputs).connector_completions
    calls = []
    scheduler = N.NativeStateLMCacheMPConnectorScheduler.__new__(
        N.NativeStateLMCacheMPConnectorScheduler
    )
    scheduler._native_saves = {op: N._NativeSave(NS(), "fixture-source", 0, 16)}
    scheduler._source_group_finished = lambda group: calls.append("source_safe")
    scheduler._checkpoints = NS(
        release_offload_store_source=lambda source: calls.append("release"),
        settle_offload_store=lambda source: calls.append("settle"),
    )
    scheduler._store_finished = lambda operation, succeeded: calls.append(
        ("store_finished", succeeded)
    )
    scheduler._save_failures, scheduler._save_tracker = {}, {}
    with S.context(None, snapshot=9):
        scheduler._complete_native_save(operation=op, succeeded=False)
    assert calls == ["source_safe", "release", "settle", ("store_finished", False)]
    stages = rec.data["operations"]["23"]["stages"]
    assert stages["tp_quorum"]["value"]["rank_mask"] == 255
    assert stages["tp_quorum"]["value"]["succeeded"] is False
    assert stages["settlement_enter"]["value"]["lease_present"] is True
    assert stages["settlement_return"]["value"]["lease_present"] is False


def test_no_tensor_access_and_source_binding(frozen, native):
    class NoRead:
        def __len__(self):
            raise AssertionError("tensor/opaque container length read")

    with pytest.raises(TypeError):
        S.request_fields(NS(token_ids=NoRead()))
    with pytest.raises(TypeError):
        S.plain(native.futures.MessagingFuture())
    with pytest.raises(RuntimeError, match="unbound native method"):
        S.verify_method(lambda: None)


@pytest.mark.parametrize(
    "outcome", ["normal", "client_error", "deadline", "native_failure"]
)
def test_existing_controller_lifecycle(frozen, tmp_path, monkeypatch, outcome):
    import os
    import signal

    import run

    subject = copy.deepcopy(frozen)
    subject["output"] = str(tmp_path / "controlled")
    subject["protocol"]["save_observer"].update(
        workload_s=18, client_s=9, evidence_s=0.15, evidence_absolute_s=18.15
    )
    private = tmp_path / "private"
    private.mkdir()
    for name in ("runtime-ready", "dataset-ready"):
        (private / f"{name}.json").write_text('{"ready":true}')
    now = [100.0]
    alarms, events, children, handlers = [], [], {}, {}
    deadline = [None]
    fd = os.open(tmp_path / "owned.lock", os.O_CREAT | os.O_RDWR, 0o600)

    class MissingProcess(Exception):
        pass

    class Process:
        def __init__(self, role):
            self.pid = 1000000000 + len(children)
            self.role = role
            self.returncode = 0 if role == "metadata" else None
            self.started = now[0]
            children[self.pid] = self

        def poll(self):
            if (
                self.role == "client"
                and now[0] >= self.started + 0.3
                and outcome in ("normal", "client_error")
            ):
                self.returncode = 0 if outcome == "normal" else 7
            return self.returncode

        def wait(self, timeout):
            events.append(("metadata_wait", timeout))
            return self.returncode

        def create_time(self):
            return float(self.pid)

        def status(self):
            return "sleeping" if self.poll() is None else "zombie"

        def send_signal(self, sig):
            events.append(("signal", self.role, sig))
            self.returncode = -sig

    def get_process(pid):
        if pid not in children:
            raise MissingProcess()
        return children[pid]

    def spawn(argv, **kwargs):
        role = (
            "metadata"
            if "--capture-metadata" in argv
            else (
                "cache"
                if argv == subject["cache"]
                else "server" if argv == subject["server"] else "client"
            )
        )
        events.append(("spawn", role))
        if role == "client":
            directory = kwargs["log_path"].parent
            (directory / "console.log").write_text(
                "Phase warmup (warmup) started\nPhase profiling (profiling) started\n"
            )
            (directory / "profile_export.jsonl").write_text("")
        return Process(role)

    def timer(which, seconds):
        alarms.append(seconds)
        deadline[0] = now[0] + seconds if seconds else None

    def sleep(seconds):
        target = now[0] + seconds
        if deadline[0] is not None and target >= deadline[0]:
            now[0] = deadline[0]
            deadline[0] = None
            handlers[signal.SIGALRM](signal.SIGALRM, None)
        else:
            now[0] = target

    def reports(*args):
        values = {
            name: {"ready": True}
            for name in ["manager", *[f"rank-{r}" for r in range(8)]]
        }
        if outcome == "native_failure" and any(
            p.role == "client" for p in children.values()
        ):
            values["manager"]["native_failure"] = {"exception": "FixtureNativeFailure"}
        return values

    release_checks = []

    def release(timeout):
        os.fstat(fd)
        release_checks.append(timeout)
        return True, {"cards": {str(r): {} for r in range(8)}}

    lifecycle = NS(
        GpuLock=lambda path: NS(acquire=lambda: fd),
        assert_gpus_idle=dict,
        assert_ports_free=lambda ports: events.append(("ports", ports)),
        descendants=lambda pid: [],
        spawn=spawn,
        wait_gpus_released=release,
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        NS(
            Process=get_process,
            NoSuchProcess=MissingProcess,
            STATUS_ZOMBIE="zombie",
            process_iter=lambda fields: [],
        ),
    )
    monkeypatch.setattr(run, "PRIVATE", private)
    monkeypatch.setattr(run, "make_plan", lambda *args: subject)
    monkeypatch.setattr(run, "git", lambda *args: "")
    monkeypatch.setenv("K31_TASK6_GPU_GRANT", subject["head"])
    monkeypatch.setenv("K31_TASK6_PLAN_SHA", "fixture")
    monkeypatch.setenv("KFD_GUARD_LOG", "")
    monkeypatch.setattr(run.os, "nice", lambda increment: 0)
    monkeypatch.setattr(
        run.importlib.util,
        "spec_from_file_location",
        lambda *args: NS(loader=NS(exec_module=lambda module: None)),
    )
    monkeypatch.setattr(run.importlib.util, "module_from_spec", lambda spec: lifecycle)
    monkeypatch.setattr(
        run.signal, "signal", lambda sig, handler: handlers.update({sig: handler})
    )
    monkeypatch.setattr(run.signal, "setitimer", timer)
    monkeypatch.setattr(run.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(run.time, "time", lambda: 1800000000 + now[0])
    monkeypatch.setattr(run.time, "sleep", sleep)
    monkeypatch.setattr(
        run.os, "killpg", lambda pid, sig: children[pid].send_signal(sig)
    )
    monkeypatch.setattr(
        run,
        "request",
        lambda url: (
            '{"status":"healthy"}'
            if "healthcheck" in url
            else json.dumps({"data": [{"id": run.P["served_name"]}]})
        ),
    )
    monkeypatch.setattr(run, "metrics", lambda *args: None)
    monkeypatch.setattr(S, "reports", reports)
    monkeypatch.setattr(
        S,
        "summarize",
        lambda *args: {
            "join_available": False,
            "complete_chains": False,
            "generations": {},
            "incomplete": ["fixture"],
        },
    )
    try:
        if outcome == "normal":
            run.execute(subject, "fixture")
        else:
            with pytest.raises(SystemExit):
                run.execute(subject, "fixture")
        state = json.loads((Path(subject["output"]) / "status.json").read_text())
        assert state["gpu_released"] and state["alive"] == []
        assert release_checks == [10]
        assert [e[1] for e in events if e[0] == "spawn"] == [
            "metadata",
            "cache",
            "server",
            "client",
        ]
        assert ("metadata_wait", 60) in events
        assert alarms[0] == 18 and 9 in alarms and alarms[-1] == 0
        receipt = state["save_observer_client"]
        assert receipt["instrumented"] and receipt["performance_accepted"] is False
        assert receipt["censored"] is (outcome in ("deadline", "native_failure"))
        assert (
            state["save_observer_stop"]["reason"]
            == {
                "normal": "invocation-end",
                "client_error": "client-failure",
                "deadline": "deadline",
                "native_failure": "native-failure",
            }[outcome]
        )
        assert state["state"] == ("COMPLETED" if outcome == "normal" else "FAILED")
        assert now[0] <= 118.15
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def test_terminal_receipt_failure_and_expired_evidence(tmp_path, monkeypatch):
    import run

    clock = S.SaveClock(plan.P["save_observer"], 0)
    clock.client_spawn = 3
    clock.client_directory = tmp_path
    clock.client_process = NS(poll=lambda: -9)
    clock.client_censored = True
    clock.latch("deadline", 900)
    state = {}
    monkeypatch.setattr(
        run,
        "observer_client_ledger",
        lambda directory: (_ for _ in ()).throw(OSError("fixture")),
    )
    run.save_completion_receipt(clock, state)
    assert state["save_observer_client"]["exit_code"] == -9
    assert state["save_observer_client"]["ledger"]["terminal"] is None
    (tmp_path / "client-result.json").unlink()
    monkeypatch.setattr(
        run,
        "save",
        lambda *args: (_ for _ in ()).throw(AssertionError("post-grace I/O")),
    )
    run.save_completion_receipt(clock, state, collect=False)
    assert state["save_observer_client"]["censored"] is True


@pytest.mark.parametrize("mismatch", ["range", "ordinal"])
def test_reducer_rejects_semantic_or_ordinal_repair(frozen, monkeypatch, mismatch):
    fields = {
        "req_id": 130,
        "start": 0,
        "end": 16,
        "boundary": 16,
        "last_prefill": True,
    }
    manager_stages = {
        "scheduler_dispatch": {"value": fields},
        "tp_quorum": {"value": {"rank_mask": 255, "succeeded": True}},
        "settlement_return": {"value": {"lease_present": False, "succeeded": True}},
    }
    values = {
        "manager": {
            "ready": True,
            "aggregation_count": 5,
            "operations": {"1": {"stages": manager_stages}},
        }
    }
    for rank in range(8):
        manager_stages[f"rank_{rank}_receipt"] = {
            "value": {"snapshot": 4, "succeeded": True}
        }
        values[f"rank-{rank}"] = {
            "ready": True,
            "aggregation_count": 5,
            "operations": {
                "1": {
                    "stages": {
                        "submit_enter": {"value": dict(fields)},
                        "worker_terminal": {
                            "value": {"aggregation_ordinal": 4, "succeeded": True}
                        },
                    }
                }
            },
        }
    monkeypatch.setattr(S, "reports", lambda *args: values)
    assert S.summarize("fixture", frozen)["complete_chains"]
    stages = values["rank-0"]["operations"]["1"]["stages"]
    if mismatch == "range":
        stages["submit_enter"]["value"]["end"] = 17
    else:
        stages["worker_terminal"]["value"]["aggregation_ordinal"] = 3
    result = S.summarize("fixture", frozen)
    assert not result["join_available"] and not result["complete_chains"]
