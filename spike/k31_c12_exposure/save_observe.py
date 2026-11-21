"""Passive scalar observation of one request's native save completion chain."""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import math
import os
import sys
import threading
import time
import types
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LMC = Path("/opt/venv/lib/python3.12/site-packages/lmcache")
SOURCE_FILES = {
    "atom.model_engine.async_proc": ROOT / "atom/model_engine/async_proc.py",
    "atom.model_engine.engine_core": ROOT / "atom/model_engine/engine_core.py",
    "atom.model_engine.model_runner": ROOT / "atom/model_engine/model_runner.py",
    "atom.kv_transfer.offload.metadata": ROOT / "atom/kv_transfer/offload/metadata.py",
    "atom.kv_transfer.offload.mp.native_state_worker": ROOT
    / "atom/kv_transfer/offload/mp/native_state_worker.py",
    "atom.kv_transfer.offload.mp.native_state_scheduler": ROOT
    / "atom/kv_transfer/offload/mp/native_state_scheduler.py",
    "atom.kv_transfer.offload.mp.scheduler": ROOT
    / "atom/kv_transfer/offload/mp/scheduler.py",
    "atom.kv_transfer.offload.mp.worker": ROOT
    / "atom/kv_transfer/offload/mp/worker.py",
    "atom.kv_transfer.offload.mp.transfer": ROOT
    / "atom/kv_transfer/offload/mp/transfer.py",
    "atom.kv_transfer.offload.mp.deployment": ROOT
    / "atom/kv_transfer/offload/mp/deployment.py",
    "atom.kv_transfer.disaggregation.aggregator": ROOT
    / "atom/kv_transfer/disaggregation/aggregator.py",
    "atom.kv_transfer.disaggregation.types": ROOT
    / "atom/kv_transfer/disaggregation/types.py",
    "lmcache.v1.multiprocess.futures": LMC / "v1/multiprocess/futures.py",
    "lmcache.integration.atom.multi_process_adapter": LMC
    / "integration/atom/multi_process_adapter.py",
    "lmcache.v1.platform.base.event_ipc": LMC / "v1/platform/base/event_ipc.py",
    "lmcache.v1.platform.devices.cuda.timeline_semaphore_event_ipc": LMC
    / "v1/platform/devices/cuda/timeline_semaphore_event_ipc.py",
    "lmcache.v1.platform.devices.cuda": LMC / "v1/platform/devices/cuda/__init__.py",
    "queue": Path("/usr/lib/python3.12/queue.py"),
}
PLAN = None
RUNTIME = None
TLS = threading.local()
BOUND = set()
BOOT_ERRORS = []
CODE_BINDINGS = {}
CHANNEL = "native_state_mp_store"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if value is None or type(value) is bool:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    if type(value) is str and len(value.encode()) <= 1024:
        return value
    if type(value) in (list, tuple) and len(value) <= 64:
        return [plain(v) for v in value]
    if type(value) is dict and len(value) <= 64 and all(type(k) is str for k in value):
        return {k: plain(v) for k, v in value.items()}
    raise TypeError("non-scalar or oversized observation")


def kind(value):
    return type(value).__module__ + "." + type(value).__qualname__


def op_key(value):
    if type(value) is str:
        parts = value.split(":")
        if (
            len(parts) == 3
            and parts[0] == "save"
            and parts[1].isdigit()
            and parts[2].isdigit()
        ):
            return int(parts[1]), int(parts[2])
        return None
    if type(value).__name__ != "SaveOperationId":
        return None
    req, generation = value.req_id, value.generation
    if type(req) not in (int, str) or not isinstance(generation, Integral):
        return None
    return req, int(generation)


def request_fields(req):
    tokens = req.token_ids
    if type(tokens) not in (list, tuple) and (
        type(tokens).__module__,
        type(tokens).__name__,
    ) != ("numpy", "ndarray"):
        raise TypeError("logical token length requires a host container")
    native, spec = req.native_state, req.save_spec
    return plain(
        {
            "req_id": req.req_id,
            "start": None if spec is None else spec.skip_leading_tokens,
            "end": len(tokens),
            "boundary": None if native is None else native.boundary_tokens,
            "last_prefill": req.is_last_prefill,
            "after_forward": req.after_forward,
        }
    )


def costs():
    # Reuse only the existing pure CPU accounting class; never install its hooks.
    name = "_k31_save_costs"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, HERE / "observe.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    class Costs(module.ObserverCosts):
        NAMES = ("capture", "json_encode", "file_write", "publication")
        BYTE_CATEGORIES = frozenset(("json_encode", "file_write"))

        def snapshot(self):
            value = super().snapshot()
            value["limits"] = (
                "Recorder-only scopes exclude original native calls. Scopes nest; "
                "do not sum them as native/GPU time. Cumulative write bytes differ "
                "from the bounded current payload. This publication's tail is "
                "absent from its own snapshot; perturbation is not zero."
            )
            return value

    return Costs()


class Recorder:
    def __init__(self, plan, role, rank):
        self.plan = plan
        self.config = plan["protocol"]["save_observer"]
        self.lock = threading.RLock()
        self.costs = costs()
        name = "manager" if role == "manager" else f"rank-{rank}"
        self.path = Path(plan["output"]) / "save_observer" / f"{name}.json"
        self.control_path = Path(plan["output"]) / "save-observer-control.json"
        self.disabled = False
        self.evidence_end = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = self.config[
            "manager_records" if role == "manager" else "worker_records"
        ]
        self.max_bytes = self.config[
            "manager_bytes" if role == "manager" else "worker_bytes"
        ]
        self.data = {
            "schema": 1,
            "role": role,
            "rank": rank,
            "pid": os.getpid(),
            "head": plan["head"],
            "config_sha256": plan["observer_config_sha256"],
            "sources": plan["sources"],
            "native_sources": plan["native_sources"],
            "activated_mono": time.monotonic(),
            "activation_ordinal": 0,
            "entry_points": [],
            "ready": False,
            "unavailable": list(BOOT_ERRORS),
            "dropped": 0,
            "operations": {},
            "records": [],
            "aggregation_count": 0,
            "current": None,
            "previous": None,
            "consumed": None,
        }
        if self.path.exists():
            raise RuntimeError("preexisting recorder path")
        self.publish()

    def fail(self, reason):
        with self.lock:
            if (
                reason not in self.data["unavailable"]
                and len(self.data["unavailable"]) < 16
            ):
                self.data["unavailable"].append(reason)
            self.data["dropped"] += 1

    def select(self, key):
        if self.disabled or (
            self.evidence_end is not None and time.monotonic() >= self.evidence_end
        ):
            return False
        if key is None or str(key[0]) != str(self.config["req_id"]):
            return False
        generation = str(key[1])
        with self.lock:
            if generation not in self.data["operations"]:
                if len(self.data["operations"]) >= self.config["max_generations"]:
                    self.fail("generation cap")
                    return False
                self.data["operations"][generation] = {
                    "req_id": key[0],
                    "generation": key[1],
                    "stages": {},
                }
            return True

    def sample(self, stage, key, fields, *, signature=None):
        if not self.select(key):
            return
        with self.costs.measure("capture"), self.lock:
            fields = plain(fields)
            tag = fields if signature is None else plain(signature)
            stages = self.data["operations"][str(key[1])]["stages"]
            now = time.monotonic()
            row = stages.setdefault(
                stage,
                {
                    "first_mono": now,
                    "last_mono": now,
                    "count": 0,
                    "value": None,
                    "signature": None,
                },
            )
            changed = row["count"] == 0 or row["signature"] != tag
            row.update(
                last_mono=now, count=row["count"] + 1, value=fields, signature=tag
            )
            if changed:
                if len(self.data["records"]) < self.max_records:
                    self.data["records"].append(
                        {
                            "stage": stage,
                            "req_id": key[0],
                            "generation": key[1],
                            "mono": now,
                            "value": fields,
                        }
                    )
                else:
                    self.fail("record cap")

    def enter(self, name):
        with self.lock:
            if name not in self.data["entry_points"]:
                self.data["entry_points"].append(name)
            required = (
                {"engine_core", "manager"}
                if self.data["role"] == "manager"
                else {"worker"}
            )
            self.data["ready"] = (
                required <= set(self.data["entry_points"])
                and not self.data["unavailable"]
            )
        self.publish()

    def publish(self):
        try:
            control = json.loads(self.control_path.read_text())
            if control["head"] != self.plan["head"]:
                raise ValueError("control head differs")
            self.evidence_end = control["evidence_end_mono"]
            if time.monotonic() >= self.evidence_end:
                self.disabled = True
                return
        except (OSError, ValueError, KeyError) as error:
            self.fail("control:" + type(error).__name__)
        # One capped snapshot per role: no temporary duplicate payload, and no
        # file write at an individual raw/event query. Interrupted JSON is unavailable.
        with self.costs.measure("publication"), self.lock:
            self.data["bound_modules"] = sorted(BOUND)
            self.data["published_mono"] = time.monotonic()
            self.data["costs"] = self.costs.snapshot()
            try:
                with self.costs.measure("json_encode") as cost:
                    raw = json.dumps(
                        self.data, allow_nan=False, separators=(",", ":")
                    ).encode()
                    while len(raw) > self.max_bytes and self.data["records"]:
                        self.data["records"].pop()
                        self.fail("byte cap")
                        raw = json.dumps(
                            self.data, allow_nan=False, separators=(",", ":")
                        ).encode()
                    if len(raw) > self.max_bytes:
                        self.fail("byte cap")
                        header = {
                            key: self.data[key]
                            for key in (
                                "schema",
                                "role",
                                "rank",
                                "pid",
                                "head",
                                "config_sha256",
                                "sources",
                                "native_sources",
                                "activated_mono",
                                "activation_ordinal",
                                "entry_points",
                                "native_failure",
                            )
                            if key in self.data
                        }
                        header.update(
                            ready=False,
                            unavailable=["byte cap"],
                            dropped=self.data["dropped"],
                        )
                        raw = json.dumps(header).encode()
                    if len(raw) > self.max_bytes:
                        raw = b""
                    cost["bytes"] = len(raw)
                with self.costs.measure("file_write") as cost:
                    with self.path.open("wb") as target:
                        written = target.write(raw)
                    if written != len(raw):
                        raise OSError("short observer write")
                    cost["bytes"] = written
            # Recorder failure must preserve native work.
            except Exception as error:  # noqa: BLE001
                self.fail("write:" + type(error).__name__)
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass


def runtime():
    global RUNTIME
    if RUNTIME is not None and RUNTIME.data["pid"] != os.getpid():
        RUNTIME = None
        TLS.context = None
    return None if RUNTIME is not None and RUNTIME.disabled else RUNTIME


def activate(role, rank, entry):
    global RUNTIME
    if PLAN is None:
        return
    value = runtime()
    if value is None:
        RUNTIME = value = Recorder(PLAN, role, rank)
    if (value.data["role"], value.data["rank"]) != (role, rank):
        value.fail("role/rank changed")
    value.enter(entry)


def safe(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    # Observation must not replace native behavior.
    except Exception as error:  # noqa: BLE001
        value = runtime()
        if value is not None:
            value.fail(type(error).__name__)
        elif PLAN is not None and len(BOOT_ERRORS) < 16:
            BOOT_ERRORS.append(type(error).__name__)
        return None


def native_failure(stage, error):
    value = runtime()
    if value is not None and value.data.get("native_failure") is None:
        value.data["native_failure"] = {
            "stage": stage,
            "exception": type(error).__name__,
            "observed_mono": time.monotonic(),
        }
        safe(value.publish)


def sample(stage, key, fields, **kwargs):
    value = runtime()
    if value is not None:
        safe(value.sample, stage, key, fields, **kwargs)


@contextmanager
def context(key, **fields):
    previous = getattr(TLS, "context", None)
    TLS.context = {"key": key, **plain(fields)}
    try:
        yield
    finally:
        TLS.context = previous


def current_key():
    if runtime() is None:
        return None
    return (getattr(TLS, "context", None) or {}).get("key")


def selected_output(output):
    found = []
    for item in output.connector_completions:
        key = op_key(item.operation_id)
        if (
            item.channel == CHANNEL
            and key is not None
            and str(key[0]) == str(PLAN["protocol"]["save_observer"]["req_id"])
        ):
            value = runtime()
            if value is not None and value.select(key):
                found.append([item.channel, key[0], key[1], plain(item.succeeded)])
    return sorted(found, key=lambda row: (str(row[1]), row[2], row[3]))


def verify_method(function):
    if PLAN is None:
        return
    target = inspect.unwrap(function)
    code = target.__code__
    path = str(Path(code.co_filename).resolve())
    if (
        path not in PLAN["native_sources"]
        or digest(path) != PLAN["native_sources"][path]
    ):
        raise RuntimeError("unbound native method source")
    if path not in CODE_BINDINGS:
        compiled = compile(
            Path(path).read_text(), code.co_filename, "exec", dont_inherit=True
        )
        found = {}

        def walk(value):
            found[value.co_qualname] = value
            for child in value.co_consts:
                if isinstance(child, types.CodeType):
                    walk(child)

        walk(compiled)
        CODE_BINDINGS[path] = found
    if CODE_BINDINGS[path].get(code.co_qualname) != code:
        raise RuntimeError("native method differs from frozen source")


def wrap(owner, name, factory):
    original = getattr(owner, name)
    if getattr(original, "_k31_save_wrapped", False):
        return
    verify_method(original)
    replacement = functools.wraps(original)(factory(original))
    replacement._k31_save_wrapped = True
    setattr(owner, name, replacement)


def query_key():
    ctx = getattr(TLS, "context", None) or {}
    return current_key() if ctx.get("phase") in ("poll", "device_query") else None


def query_factory(stage):
    def factory(original):
        def observed(self, *args, **kwargs):
            key = query_key()
            if key is None:
                return original(self, *args, **kwargs)
            ctx = getattr(TLS, "context", None)
            if ctx is not None and stage in ("raw_query", "event_query"):
                ctx[stage + "_calls"] = ctx.get(stage + "_calls", 0) + 1
            try:
                result = original(self, *args, **kwargs)
            except BaseException as error:
                sample(stage, key, {"exception": type(error).__name__})
                raise
            fields = {
                "result": result if result is None or type(result) is bool else None,
                "result_kind": kind(result),
                "exception": None,
            }
            sample(stage, key, fields)
            return result

        return observed

    return factory


def device_query_factory(original):
    def observed(self, *args, **kwargs):
        key = query_key()
        if key is None:
            return original(self, *args, **kwargs)
        before = safe(
            lambda: {
                "raw_processed": self._raw_response_processed,
                "event_present": self.event_ is not None,
                "backend_kind": kind(self._event_backend),
            }
        )
        backend_bound = safe(
            lambda: getattr(
                type(self._event_backend).query_event, "_k31_save_wrapped", False
            )
        )
        if not backend_bound:
            value = runtime()
            if value is not None:
                value.fail("unbound event backend")
        with context(key, phase="device_query", raw_query_calls=0, event_query_calls=0):
            try:
                result = original(self, *args, **kwargs)
            except BaseException as error:
                ctx = TLS.context
                sample(
                    "device_query",
                    key,
                    {
                        **(before or {}),
                        "raw_calls": ctx["raw_query_calls"],
                        "event_calls": ctx["event_query_calls"],
                        "exception": type(error).__name__,
                    },
                )
                raise
            ctx = TLS.context
            sample(
                "device_query",
                key,
                {
                    **(before or {}),
                    "raw_calls": ctx["raw_query_calls"],
                    "event_calls": ctx["event_query_calls"],
                    "raw_processed_after": self._raw_response_processed,
                    "event_present_after": self.event_ is not None,
                    "result": result if type(result) is bool else None,
                    "result_kind": kind(result),
                    "exception": None,
                },
            )
            return result

    return observed


def terminal_factory(original):
    def observed(future):
        key = query_key()
        if key is None:
            return original(future)
        try:
            result = original(future)
        except BaseException as error:
            safe(native_failure, "terminal", error)
            sample(
                "terminal",
                key,
                {"future_kind": kind(future), "exception": type(error).__name__},
            )
            raise
        sample(
            "terminal",
            key,
            {
                "future_kind": kind(future),
                "terminal": result[0],
                "result": (
                    result[1] if result[1] is None or type(result[1]) is bool else None
                ),
                "result_kind": kind(result[1]),
                "exception": None,
            },
        )
        return result

    return observed


def deadline_factory(original):
    def observed(operation, started_at, deadline_s):
        key = op_key(operation) or current_key()
        value = runtime()
        if value is None or not safe(value.select, key):
            try:
                return original(operation, started_at, deadline_s)
            except BaseException as error:
                safe(native_failure, "deadline", error)
                raise
        entered = time.monotonic()
        ctx = getattr(TLS, "context", None) or {}
        fields = {
            "started_at": started_at,
            "deadline_s": deadline_s,
            "observer_entry_age_s": entered - started_at,
            "consumed_snapshot": ctx.get("snapshot"),
        }
        try:
            result = original(operation, started_at, deadline_s)
        except BaseException as error:
            safe(native_failure, "deadline", error)
            text = error.args[0] if error.args and type(error.args[0]) is str else ""
            import re

            match = re.search(r"after (\d+)s", text)
            fields.update(
                observer_exit_age_s=time.monotonic() - started_at,
                exception=type(error).__name__,
                native_rounded_age_s=int(match[1]) if match else None,
            )
            sample(
                "deadline",
                key,
                fields,
                signature=[type(error).__name__, started_at, deadline_s],
            )
            value = runtime()
            if value is not None:
                safe(value.publish)
            raise
        fields.update(observer_exit_age_s=time.monotonic() - started_at, exception=None)
        sample("deadline", key, fields, signature=[None, started_at, deadline_s])
        return result

    return observed


def local_key_factory(original):
    def observed(kind, completion):
        result = original(kind, completion)
        if kind == "save":
            sample("local_key", op_key(completion), {"local_key": result})
        return result

    return observed


def adapter_factory(original):
    def observed(self, request_id, spec, event):
        key = current_key()
        if key is None:
            return original(self, request_id, spec, event)
        ctx = getattr(TLS, "context", None)
        if ctx is not None:
            ctx["adapter_called"] = True
        fields = (
            safe(lambda: {"session": request_id, "start": spec.start, "end": spec.end})
            or {}
        )
        sample("adapter_enter", key, fields)
        try:
            future = original(self, request_id, spec, event)
        except BaseException as error:
            sample("adapter_return", key, {**fields, "exception": type(error).__name__})
            raise
        sample(
            "adapter_return",
            key,
            {**fields, "future_kind": kind(future), "exception": None},
        )
        return future

    return observed


def pending_fields(worker, local_key):
    pending = worker._native_saves.get(local_key)
    return (
        {"pending": False}
        if pending is None
        else {
            "pending": True,
            "future_kind": kind(pending.future),
            "immediate_success": pending.immediate_success,
            "started_at": pending.started_at,
        }
    )


def submit_factory(original):
    def observed(self, req, event, *, loading):
        key = None if loading else op_key(req.save_operation)
        value = runtime()
        selected = value is not None and safe(value.select, key)
        fields = safe(request_fields, req) if selected else None
        sample("submit_enter", key, fields or {})
        with context(key if selected else None, phase="submit", adapter_called=False):
            try:
                result = original(self, req, event, loading=loading)
            except BaseException as error:
                sample("submit_return", key, {"exception": type(error).__name__})
                raise
            if selected:
                local = f"save:{req.req_id}:{req.save_operation.generation}"
                after = safe(pending_fields, self, local) or {}
                sample(
                    "submit_return",
                    key,
                    {
                        **after,
                        "adapter_called": TLS.context["adapter_called"],
                        "exception": None,
                    },
                )
        return result

    return observed


def poll_factory(original):
    def observed(self, output, operation_id, pending):
        key = op_key(pending.request.save_operation)
        value = runtime()
        selected = value is not None and safe(value.select, key)
        if selected:
            before = safe(selected_output, output) or []
            sample(
                "poll_enter",
                key,
                {
                    **(safe(request_fields, pending.request) or {}),
                    "local_key": operation_id,
                    "future_kind": kind(pending.future),
                    "started_at": pending.started_at,
                    "immediate_success": pending.immediate_success,
                },
            )
        with context(key if selected else None, phase="poll"):
            try:
                result = original(self, output, operation_id, pending)
            except BaseException as error:
                sample("poll_return", key, {"exception": type(error).__name__})
                raise
            if selected:
                keys = [
                    row
                    for row in (safe(selected_output, output) or [])
                    if row not in before
                    and (str(row[1]), row[2]) == (str(key[0]), key[1])
                ]
                sample("poll_return", key, {"emitted": keys, "exception": None})
                for channel, req, gen, succeeded in keys:
                    ordinal = (value.data["current"] or {}).get("ordinal")
                    sample(
                        "worker_terminal",
                        (req, gen),
                        {
                            "channel": channel,
                            "succeeded": succeeded,
                            "aggregation_ordinal": ordinal,
                        },
                    )
        return result

    return observed


def build_save_factory(original):
    def observed(self, seq, saved, aligned, operation, block_ids, is_last_prefill):
        key = op_key(operation)
        sample(
            "scheduler_save_enter",
            key,
            {"saved": saved, "boundary": aligned, "last_prefill": is_last_prefill},
        )
        try:
            result = original(
                self, seq, saved, aligned, operation, block_ids, is_last_prefill
            )
        except BaseException as error:
            sample("scheduler_save_return", key, {"exception": type(error).__name__})
            raise
        sample(
            "scheduler_save_return",
            key,
            {"returned_none": result is None, "exception": None},
        )
        return result

    return observed


def build_meta_factory(original):
    def observed(self):
        result = original(self)
        for req in result.requests:
            key = op_key(req.save_operation)
            value = runtime()
            if value is not None and safe(value.select, key):
                sample("scheduler_dispatch", key, safe(request_fields, req) or {})
        return result

    return observed


def lease_fields(scheduler, op):
    lease = scheduler._native_saves.get(op)
    return (
        {"lease_present": False}
        if lease is None
        else {
            "lease_present": True,
            "saved_before": lease.saved_before,
            "boundary": lease.boundary,
            "source_present": lease.source is not None,
        }
    )


def settle_factory(original):
    def observed(self, operation, *, succeeded):
        key = op_key(operation)
        fields = safe(lease_fields, self, operation) or {}
        fields.update(
            succeeded=succeeded,
            consumed_snapshot=(getattr(TLS, "context", None) or {}).get("snapshot"),
        )
        sample("settlement_enter", key, fields)
        try:
            result = original(self, operation, succeeded=succeeded)
        except BaseException as error:
            sample(
                "settlement_return", key, {**fields, "exception": type(error).__name__}
            )
            raise
        sample(
            "settlement_return",
            key,
            {
                **(safe(lease_fields, self, operation) or {}),
                "succeeded": succeeded,
                "exception": None,
            },
        )
        return result

    return observed


def constructor_factory(role, entry):
    def factory(original):
        def observed(self, *args, **kwargs):
            rank = (
                kwargs.get("rank", args[4] if len(args) > 4 else None)
                if role == "worker"
                else None
            )
            safe(activate, role, rank, entry)
            try:
                return original(self, *args, **kwargs)
            except BaseException as error:
                safe(native_failure, "constructor", error)
                raise

        return observed

    return factory


def tag_queues(manager):
    for rank, queue in enumerate(manager.kv_outputs_queues):
        queue._k31_save_rank = rank


def start_aggregation_factory(original):
    def observed(self, func_name, args):
        value = runtime()
        if value is None or func_name != "async_proc_aggregation":
            return original(self, func_name, args)
        safe(tag_queues, self)
        if self._pending_kv_aggregation is not None:
            value.fail("multiple outstanding aggregations")
        entered = time.monotonic()
        try:
            result = original(self, func_name, args)
        except BaseException:
            value.fail("ambiguous aggregation broadcast")
            safe(value.publish)
            raise
        with value.lock:
            value.data["aggregation_count"] += 1
            value.data["previous"] = value.data["current"]
            value.data["current"] = {
                "ordinal": value.data["aggregation_count"],
                "entry_mono": entered,
                "native_started_at": result.started_at,
                "return_mono": time.monotonic(),
                "rank_mask": 0,
                "success_mask": 0,
                "receipts": {},
            }
        if value.data["operations"] or value.data["aggregation_count"] <= 2:
            safe(value.publish)
        return result

    return observed


def queue_get_factory(original):
    def observed(self):
        result = original(self)
        value = runtime()
        rank = getattr(self, "_k31_save_rank", None)
        if value is not None and value.data["role"] == "manager" and type(rank) is int:

            def capture():
                keys = selected_output(result)
                pending = value.data["current"]
                if pending is None or not 0 <= rank < 8:
                    value.fail("receipt without current broadcast")
                    return
                bit = 1 << rank
                if pending["rank_mask"] & bit:
                    value.fail("duplicate rank receipt")
                pending["rank_mask"] |= bit
                pending["receipts"][str(rank)] = {
                    "consumed_mono": time.monotonic(),
                    "keys": keys,
                }
                for channel, req, generation, succeeded in keys:
                    value.sample(
                        f"rank_{rank}_receipt",
                        (req, generation),
                        {
                            "snapshot": pending["ordinal"],
                            "channel": channel,
                            "succeeded": succeeded,
                            "rank": rank,
                        },
                    )

            safe(capture)
        return result

    return observed


def call_aggregation_factory(original):
    def observed(self, func_name, *args, **kwargs):
        value = runtime()
        if value is not None:
            value.data["consumed"] = None
        try:
            result = original(self, func_name, *args, **kwargs)
        except BaseException as error:
            safe(native_failure, "manager_aggregation", error)
            if value is not None:
                value.fail("aggregation exception")
                safe(value.publish)
            raise
        if value is not None and (
            value.data["operations"] or value.data["aggregation_count"] <= 2
        ):
            safe(value.publish)
        return result

    return observed


def aggregate_factory(original):
    def observed(self, worker_outputs):
        value = runtime()
        snapshot = None
        if value is not None:
            snapshot = safe(lambda: plain(value.data["current"]))
            value.data["consumed"] = snapshot
        with context(None, snapshot=None if snapshot is None else snapshot["ordinal"]):
            result = original(self, worker_outputs)
        return result

    return observed


def worker_aggregation_factory(original):
    def observed(self):
        value = runtime()
        if value is None:
            return original(self)
        with value.lock:
            value.data["aggregation_count"] += 1
            value.data["previous"] = value.data["current"]
            value.data["current"] = {
                "ordinal": value.data["aggregation_count"],
                "entry_mono": time.monotonic(),
                "return_mono": None,
                "keys": None,
                "exception": None,
            }
        if value.data["operations"] or value.data["aggregation_count"] <= 2:
            safe(value.publish)
        try:
            result = original(self)
        except BaseException as error:
            safe(native_failure, "worker_aggregation", error)
            value.data["current"]["exception"] = type(error).__name__
            safe(value.publish)
            raise
        value.data["current"].update(
            return_mono=time.monotonic(), keys=safe(selected_output, result)
        )
        if value.data["operations"] or value.data["aggregation_count"] <= 2:
            safe(value.publish)
        return result

    return observed


def completion_key(key):
    if type(key) is tuple and len(key) == 2 and key[0] == CHANNEL:
        return op_key(key[1])
    return None


def report_fields(group, key):
    values = group._reports.get(key, {})
    return {
        "rank_mask": sum(1 << rank for rank in values),
        "success_mask": sum(
            1 << rank for rank, success in values.items() if success is True
        ),
        "world_size": group._world_size,
        "tombstoned": key in group._tombstones,
        "snapshot": (getattr(TLS, "context", None) or {}).get("snapshot"),
    }


def report_factory(original):
    def observed(self, worker_idx, key, *, succeeded):
        selected = completion_key(key)
        before = safe(report_fields, self, key) if selected is not None else None
        result = original(self, worker_idx, key, succeeded=succeeded)
        if selected is not None:
            after = safe(report_fields, self, key) or {}
            sample(
                "tp_report",
                selected,
                {
                    **after,
                    "worker_idx": worker_idx,
                    "succeeded": succeeded,
                    "duplicate": bool(
                        (before or {}).get("rank_mask", 0) & (1 << worker_idx)
                    )
                    or (before or {}).get("tombstoned", False),
                },
            )
        return result

    return observed


def drain_factory(original):
    def observed(self):
        before = {}
        value = runtime()
        if value is not None:
            for key in self._reports:
                selected = completion_key(key)
                if selected is not None and safe(value.select, selected):
                    before[str(selected[1])] = safe(report_fields, self, key)
        result = original(self)
        for keys, succeeded in zip(result, (True, False)):
            for key in keys:
                selected = completion_key(key)
                if selected is not None:
                    sample(
                        "tp_quorum",
                        selected,
                        {
                            **(before.get(str(selected[1])) or {}),
                            "succeeded": succeeded,
                        },
                    )
        return result

    return observed


def process_completions_factory(original):
    def observed(self, output):
        value = runtime()
        snapshot = None if value is None else value.data["consumed"]
        with context(None, snapshot=None if snapshot is None else snapshot["ordinal"]):
            try:
                result = original(self, output)
            except BaseException as error:
                safe(native_failure, "scheduler_completions", error)
                raise
            finally:
                if value is not None and value.data["operations"]:
                    safe(value.publish)
        return result

    return observed


def patch(module):
    name = module.__name__
    expected = SOURCE_FILES.get(name)
    if expected is None:
        return
    if PLAN is not None and (
        Path(module.__file__).resolve() != expected.resolve()
        or digest(expected) != PLAN["native_sources"].get(str(expected))
    ):
        raise RuntimeError("native source binding differs: " + name)
    if name.endswith(".async_proc"):
        wrap(
            module.AsyncIOProcManager,
            "__init__",
            constructor_factory("manager", "manager"),
        )
        wrap(module.AsyncIOProc, "__init__", constructor_factory("worker", "worker"))
        wrap(
            module.AsyncIOProcManager,
            "_start_kv_aggregation",
            start_aggregation_factory,
        )
        wrap(
            module.AsyncIOProcManager,
            "call_func_with_aggregation",
            call_aggregation_factory,
        )
    elif name.endswith(".engine_core"):
        wrap(
            module.EngineCore, "__init__", constructor_factory("manager", "engine_core")
        )
    elif name.endswith(".model_runner"):
        wrap(module.ModelRunner, "async_proc_aggregation", worker_aggregation_factory)
    elif name.endswith(".native_state_worker"):
        cls = module.NativeStateLMCacheMPConnector
        wrap(cls, "_submit_native", submit_factory)
        wrap(cls, "_poll_native_save", poll_factory)
    elif name.endswith(".native_state_scheduler"):
        cls = module.NativeStateLMCacheMPConnectorScheduler
        wrap(cls, "_build_save_request", build_save_factory)
        wrap(cls, "_complete_native_save", settle_factory)
        wrap(cls, "process_completions", process_completions_factory)
    elif name == "atom.kv_transfer.offload.mp.scheduler":
        wrap(
            module.LMCacheMPConnectorScheduler,
            "build_connector_meta",
            build_meta_factory,
        )
    elif name.endswith(".aggregator"):
        wrap(module._TPCompletionGroup, "report", report_factory)
        wrap(module._TPCompletionGroup, "drain", drain_factory)
        wrap(module.KVOutputAggregator, "aggregate", aggregate_factory)
    elif name == "lmcache.v1.multiprocess.futures":
        wrap(module.MessagingFuture, "query", query_factory("raw_query"))
        wrap(module.MessagingFuture, "result", query_factory("raw_result"))
        wrap(module.DeviceMessagingFuture, "query", device_query_factory)
        wrap(module.DeviceMessagingFuture, "result", query_factory("device_result"))
    elif name.endswith(".multi_process_adapter"):
        wrap(module.AtomMPWorkerAdapter, "submit_store_request", adapter_factory)
    elif name.endswith(".event_ipc"):
        wrap(module.DefaultEventIPCBackend, "query_event", query_factory("event_query"))
    elif name.endswith(".timeline_semaphore_event_ipc"):
        wrap(
            module.TimelineSemaphoreEventIPCBackend,
            "query_event",
            query_factory("event_query"),
        )
    elif name == "queue":
        wrap(module.Queue, "get_nowait", queue_get_factory)
    for symbol, factory in (
        ("_terminal_future_result", terminal_factory),
        ("_enforce_transfer_deadline", deadline_factory),
        ("_transfer_operation_id", local_key_factory),
    ):
        if hasattr(module, symbol):
            wrap(module, symbol, factory)
    BOUND.add(name)
    value = runtime()
    if value is not None:
        safe(value.publish)


class NativeHookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in SOURCE_FILES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            original = spec.loader

            class Loader:
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    safe(patch, module)

            spec.loader = Loader()
        return spec


def install():
    global PLAN
    if "K31_OBSERVER_PLAN" in os.environ:
        raise RuntimeError("observer modes cannot share a service")
    PLAN = json.loads(Path(os.environ["K31_SAVE_OBSERVER_PLAN"]).read_text())
    if PLAN["mode"] != "save_completion_observer":
        raise RuntimeError("wrong observer mode")
    for name, expected in {**PLAN["sources"], **PLAN["native_sources"]}.items():
        if digest(name) != expected:
            raise RuntimeError("frozen source changed: " + name)
    sys.meta_path.insert(0, NativeHookFinder())
    for name in SOURCE_FILES:
        if name in sys.modules:
            safe(patch, sys.modules[name])


def reports(out, plan):
    result = {}
    for name in [
        "manager",
        *[f"rank-{rank}" for rank in plan["protocol"]["save_observer"]["ranks"]],
    ]:
        path = Path(out) / "save_observer" / f"{name}.json"
        try:
            raw = path.read_bytes()
            maximum = plan["protocol"]["save_observer"][
                "manager_bytes" if name == "manager" else "worker_bytes"
            ]
            if len(raw) > maximum:
                raise ValueError("payload cap")
            value = json.loads(raw)
            if any(
                value.get(k) != plan[v]
                for k, v in (
                    ("head", "head"),
                    ("config_sha256", "observer_config_sha256"),
                    ("sources", "sources"),
                    ("native_sources", "native_sources"),
                )
            ):
                raise ValueError("source/config identity mismatch")
            if value.get("activation_ordinal") != 0:
                raise ValueError("late activation")
            expected_role = "manager" if name == "manager" else "worker"
            expected_rank = None if name == "manager" else int(name.split("-")[1])
            if (
                value.get("role") != expected_role
                or value.get("rank") != expected_rank
                or type(value.get("pid")) is not int
            ):
                raise ValueError("role/rank identity mismatch")
            result[name] = value
        except (OSError, ValueError):
            result[name] = {"ready": False, "unavailable": ["missing/invalid receipt"]}
    pids = [v.get("pid") for v in result.values() if v.get("ready")]
    if len(pids) != len(set(pids)):
        for value in result.values():
            value["unavailable"] = ["duplicate role PID"]
            value["ready"] = False
    return result


def startup_ready(values):
    return len(values) == 9 and all(
        v.get("ready") and not v.get("unavailable") for v in values.values()
    )


def summarize(out, plan):
    values = reports(out, plan)
    reasons, generations = [], []
    try:
        status = json.loads((Path(out) / "status.json").read_text())
    except FileNotFoundError:
        status = {}
    except (OSError, ValueError):
        reasons.append("controller status unavailable")
        status = {}
    if type(status) is not dict:
        reasons.append("controller status invalid")
    elif status.get("save_observer_control_error"):
        reasons.append("stop-control publication failed")
    for name, value in values.items():
        if not value.get("ready") or value.get("unavailable") or value.get("dropped"):
            reasons.append(name + ": unavailable/dropped")
        generations.append(set(value.get("operations", {})))
    if not any(generations):
        reasons.append("selected request absent")
    if any(g != generations[0] for g in generations):
        reasons.append("generation sets differ")
    manager = values["manager"]
    joined = {}
    for generation in sorted(set.union(*generations)):
        stages = manager.get("operations", {}).get(generation, {}).get("stages", {})
        dispatch = stages.get("scheduler_dispatch", {}).get("value")
        item = {
            "dispatch": dispatch,
            "workers": {},
            "complete_chain": True,
            "missing": [],
        }
        for rank in plan["protocol"]["save_observer"]["ranks"]:
            worker = values[f"rank-{rank}"]
            seen = worker.get("operations", {}).get(generation, {}).get("stages", {})
            submit = seen.get("submit_enter", {}).get("value")
            terminal = seen.get("worker_terminal", {}).get("value")
            consumed = stages.get(f"rank_{rank}_receipt", {}).get("value")
            item["workers"][str(rank)] = {
                "submit": submit,
                "terminal": terminal,
                "consumed": consumed,
                "query": seen.get("device_query"),
                "poll": seen.get("poll_return"),
            }
            if (
                dispatch is None
                or submit is None
                or any(
                    dispatch.get(k) != submit.get(k)
                    for k in ("req_id", "start", "end", "boundary", "last_prefill")
                )
            ):
                item["missing"].append(f"rank-{rank}: semantic correspondence")
            if terminal is None or consumed is None:
                item["missing"].append(f"rank-{rank}: terminal/receipt")
            elif (
                terminal["aggregation_ordinal"] is None
                or terminal["aggregation_ordinal"] != consumed["snapshot"]
            ):
                item["missing"].append(f"rank-{rank}: ordinal join")
            if worker.get("aggregation_count", 0) > manager.get("aggregation_count", 0):
                item["missing"].append(f"rank-{rank}: execution without broadcast")
        quorum = stages.get("tp_quorum", {}).get("value")
        settlement = stages.get("settlement_return", {}).get("value")
        item.update(
            quorum=quorum, settlement=settlement, deadline=stages.get("deadline")
        )
        if quorum is None or quorum.get("rank_mask") != 255:
            item["missing"].append("all-rank quorum")
        if settlement is None or settlement.get("lease_present") is not False:
            item["missing"].append("native settlement")
        item["join_available"] = not reasons and not any(
            any(
                term in missing
                for term in (
                    "semantic correspondence",
                    "ordinal join",
                    "execution without broadcast",
                )
            )
            for missing in item["missing"]
        )
        item["complete_chain"] = not item["missing"] and item["join_available"]
        joined[generation] = item
    return {
        "mode": "save_completion_observer",
        "request_id": plan["protocol"]["save_observer"]["req_id"],
        "generations": joined,
        "join_available": not reasons
        and all(v["join_available"] for v in joined.values()),
        "complete_chains": bool(joined)
        and all(v["complete_chain"] for v in joined.values()),
        "incomplete": reasons,
        "roles": values,
        "performance_accepted": False,
        "limits": "Scalar CPU observations only. Missing/duplicate/late ordinals cannot be repaired by timestamps. Complete native False outcomes remain failures; client credits are not native settlement. Snapshots may omit an in-flight publication tail; no native/GPU duration or zero perturbation is claimed.",
    }


class SaveClock:
    """State for the existing supervisor's absolute limits, never a second runner."""

    def __init__(self, config, launched):
        self.config, self.launched = config, launched
        self.client_spawn = None
        self.reason = None
        self.phases = {}
        self.on_stop = None
        self.on_clock_change = None

    def deadline(self):
        stop = self.launched + self.config["workload_s"]
        return (
            stop
            if self.client_spawn is None
            else min(stop, self.client_spawn + self.config["client_s"])
        )

    def start_client(self, now):
        if self.reason is not None:
            return
        self.client_spawn = now
        if now >= self.deadline():
            self.latch("deadline", now, "client spawn at expired bound")
        elif self.on_clock_change is not None:
            self.on_clock_change()

    def latch(self, reason, now, detail=None):
        if self.reason is None:
            self.reason = {"reason": reason, "mono": now, "detail": plain(detail)}
            if self.on_stop is not None:
                self.on_stop()
        return self.reason

    def grace_end(self):
        return min(
            self.reason["mono"] + self.config["evidence_s"],
            self.launched + self.config["evidence_absolute_s"],
        )

    def anchors(self):
        return {
            "launch_mono": self.launched,
            "client_spawn_mono": self.client_spawn,
            "deadline_mono": self.deadline(),
            "phases": dict(self.phases),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.out / "plan.json").read_text())
    if plan["mode"] != "save_completion_observer":
        raise ValueError("not a save observation")
    report = summarize(args.out, plan)
    if args.report_out is not None:
        args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "roles"}, indent=2))


if __name__ == "__main__":
    main()
