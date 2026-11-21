"""Passive fixed-seam observer. Imports only stdlib until native worker setup."""

import faulthandler
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from numbers import Integral
from pathlib import Path

HERE = Path(__file__).resolve().parent
TLS = threading.local()
RUNTIME = None
HOOKS = []
NATIVE_SOURCES = {}
MODULES = {
    "atom.model_engine.model_runner",
    "atom.model_ops.attentions.aiter_mla",
    "atom.model_ops.indexer.indexer_forward",
    "atom.model_ops.attention_mla",
    "aiter.dist.parallel_state",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObserverCosts:
    """Fixed-size, inclusive observer scopes; no operands or per-call history."""

    NAMES = (
        "iteration_work",
        "query_sweep",
        "host_snapshot",
        "host_array_copy",
        "descriptor",
        "marker_initial_record",
        "marker_record",
        "initial_event_query",
        "pending_event_query",
        "json_encode",
        "file_write",
        "publication",
        "fr_startup_call",
        "fr_final_call",
        "fr_decode",
        "controller_phase_scan",
        "controller_snapshot_read",
        "client_ledger",
    )

    BYTE_CATEGORIES = frozenset(
        {
            "host_array_copy",
            "json_encode",
            "file_write",
            "fr_startup_call",
            "fr_final_call",
            "fr_decode",
            "controller_phase_scan",
            "controller_snapshot_read",
        }
    )

    def __init__(self):
        self.lock = threading.RLock()
        self.rows = {
            name: {
                "started": 0,
                "finished": 0,
                "failed": 0,
                "in_flight": 0,
                "max_in_flight": 0,
                "wall_sum_s": 0.0,
                "thread_cpu_sum_s": 0.0,
                "wall_max_s": 0.0,
                "thread_cpu_max_s": 0.0,
                "bytes_completed": 0 if name in self.BYTE_CATEGORIES else None,
                "bytes_unavailable_calls": 0 if name in self.BYTE_CATEGORIES else None,
                "true_results": 0,
                "false_results": 0,
            }
            for name in self.NAMES
        }

    @contextmanager
    def measure(self, name):
        row = self.rows[name]
        wall, cpu = time.monotonic(), time.thread_time()
        with self.lock:
            row["started"] += 1
            row["in_flight"] += 1
            row["max_in_flight"] = max(row["max_in_flight"], row["in_flight"])
        result, failed = {"bytes": None, "query_result": None}, True
        try:
            yield result
            failed = False
        finally:
            elapsed, used = time.monotonic() - wall, time.thread_time() - cpu
            with self.lock:
                row["finished"] += 1
                row["failed"] += int(failed)
                row["in_flight"] -= 1
                row["wall_sum_s"] += elapsed
                row["thread_cpu_sum_s"] += used
                row["wall_max_s"] = max(row["wall_max_s"], elapsed)
                row["thread_cpu_max_s"] = max(row["thread_cpu_max_s"], used)
                if name in self.BYTE_CATEGORIES:
                    if result["bytes"] is None:
                        row["bytes_unavailable_calls"] += 1
                    else:
                        row["bytes_completed"] += result["bytes"]
                row["true_results"] += int(result["query_result"] is True)
                row["false_results"] += int(result["query_result"] is False)

    def snapshot(self):
        with self.lock:
            return {
                "schema": 1,
                "available": True,
                "sampled_mono": time.monotonic(),
                "categories": {name: dict(row) for name, row in self.rows.items()},
                "limits": "Inclusive/nested scopes and concurrent calls overlap; do not sum them as native/GPU duration. Iteration_work includes query_sweep/queries, export/decode and publication but excludes the unchanged sleep. Failed counts mean exceptions escaping that scope, not every logical observer failure. Host_snapshot includes its encode/write/descriptor children; publication includes encode/write. Clock/aggregation overhead is not separately measured. Byte-null categories have no measured byte quantity; payload bytes are not GPU traffic. Bytes cover completed host-array copies, encodes/writes/FR returns/decodes and controller file reads; partial failed writes and unfinished calls are unavailable. This snapshot precedes its own encode/write completion; later work may be absent.",
            }


def measured(name):
    def decorate(method):
        @functools.wraps(method)
        def call(self, *args, **kwargs):
            costs = getattr(self, "costs", None)
            with costs.measure(name) if costs is not None else nullcontext():
                return method(self, *args, **kwargs)

        return call

    return decorate


def atomic(path, value, costs=None):
    path = Path(path)
    if not isinstance(value, bytes):
        with (
            costs.measure("json_encode")
            if costs is not None
            else nullcontext({"bytes": 0})
        ) as item:
            value = (json.dumps(value, allow_nan=False) + "\n").encode()
            item["bytes"] = len(value)
    with (
        costs.measure("file_write") if costs is not None else nullcontext({"bytes": 0})
    ) as item:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        written = tmp.write_bytes(value)
        tmp.replace(path)
        item["bytes"] = written


def host_int(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be a host integer")
    return int(value)


def host_counts(values, name):
    import numpy as np

    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise TypeError(f"{name} must be a one-dimensional host integer array")
        return values.tolist()
    if isinstance(values, list):
        return [host_int(value, name) for value in values]
    raise TypeError(f"{name} must be a native host list or ndarray")


def descriptor(tensor):
    # Only scalar values escape this function; no storage/operand is retained.
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "capacity_bytes": tensor.untyped_storage().nbytes(),
        "device": str(tensor.device),
        "allocation_generation": "unavailable",
    }


def stream_identity(torch):
    stream = torch.cuda.current_stream()
    return {
        "device": stream.device_index,
        "stream_id": stream.stream_id,
        "cuda_stream": stream.cuda_stream,
    }


def entry_key(entry):
    return (
        tuple(entry["process_group"]),
        entry["pg_id"],
        entry["record_id"],
        entry["collective_seq_id"],
    )


def selected_entries(trace, group):
    return [e for e in trace.get("entries", []) if e.get("process_group") == group]


def native_match(entry, receipt):
    # AITER supplies the native output as (world_size,) + input.shape.
    dtype = {
        "torch.uint8": "Byte",
        "torch.bfloat16": "BFloat16",
        "torch.float16": "Half",
        "torch.float32": "Float",
    }.get(receipt["dtype"])
    return (
        "allgather" in entry.get("profiling_name", "").lower().replace("_", "")
        and entry.get("input_sizes") == [receipt["shape"]]
        and entry.get("output_sizes") == [[8] + receipt["shape"]]
        and dtype is not None
        and entry.get("input_dtypes") == [dtype]
        and entry.get("output_dtypes") == [dtype]
    )


def calibrate(before, after, receipt, group):
    old = {entry_key(e) for e in selected_entries(before, group)}
    new = [e for e in selected_entries(after, group) if entry_key(e) not in old]
    if len(new) != 1 or not native_match(new[0], receipt):
        raise ValueError(
            "normal startup selected-PG association is ambiguous/unavailable"
        )
    span = receipt["after"] - receipt["before"]
    if span <= 0:
        raise ValueError("non-increasing selected-PG accessor")
    return {
        "offset": new[0]["collective_seq_id"] - receipt["before"],
        "span": span,
        "pg_id": new[0]["pg_id"],
    }


def associate(trace, group, calibration, receipts):
    entries = selected_entries(trace, group)
    joined = []
    for receipt in receipts:
        if (
            receipt.get("after", receipt["before"]) - receipt["before"]
            != calibration["span"]
        ):
            raise ValueError("target PG accessor interval differs/races")
        seq = receipt["before"] + calibration["offset"]
        found = [
            e
            for e in entries
            if e["collective_seq_id"] == seq
            and e["pg_id"] == calibration["pg_id"]
            and native_match(e, receipt)
        ]
        if len(found) != 1:
            raise ValueError("target collective missing, ambiguous, or ring-truncated")
        joined.append(
            {
                "semantic": receipt["semantic"],
                "collective_seq_id": seq,
                "record_id": found[0]["record_id"],
                "state": found[0]["state"],
            }
        )
    return joined


def frontier(markers):
    for i, marker in enumerate(markers):
        if not marker["recorded"] or marker["state"] != "completed":
            return i
    return len(markers)


class StopState:
    """Parent-only monotonic deadlines; worker queries/exporters cannot extend them."""

    def __init__(self, config, launched):
        self.config, self.launched = config, launched
        self.ready = None
        self.reason = None
        self.frontiers = {}
        self.on_stop = self.on_clock_change = None
        self.client_spawn = self.warmup = self.target_first = self.target_quorum = None
        self.target_identity = None
        self.phases = {}
        self.costs = ObserverCosts()

    def latch(self, reason, now, detail=None):
        if self.reason is None:
            self.reason = {"reason": reason, "mono": now, "detail": detail}
            if self.on_stop:
                self.on_stop()
        return self.reason

    def deadline(self):
        absolute = self.launched + self.config["workload_s"]
        if self.ready is None:
            return min(absolute, self.launched + self.config["startup_s"])
        if self.target_quorum is not None:
            return absolute
        ends = [absolute, self.ready + self.config["prefix_s"]]
        if self.warmup is not None:
            ends.append(self.warmup + self.config["target_arrival_s"])
        elif self.client_spawn is not None:
            ends.append(self.client_spawn + self.config["client_setup_s"])
        return min(ends)

    def anchors(self):
        return {
            "L": self.launched,
            "R": self.ready,
            "C": self.client_spawn,
            "W": self.warmup,
            "T": self.target_first,
            "quorum_observed_mono": self.target_quorum,
            "target_identity": self.target_identity,
            "deadline_mono": self.deadline(),
            "phase_receipts": self.phases,
        }

    def start_client(self, now):
        if self.reason:
            return self.reason
        if self.client_spawn is not None:
            raise ValueError("observer client already spawned; no second arm")
        if self.ready is None or now < self.ready:
            raise ValueError("client spawn requires sealed readiness")
        if now >= self.deadline():
            return self.latch("deadline", now, "before client spawn")
        self.client_spawn = now
        if self.on_clock_change:
            self.on_clock_change()

    def observe_warmup(self, receipt):
        if self.warmup is not None or self.reason:
            return
        now = receipt["observed_mono"]
        if self.client_spawn is not None and now < self.client_spawn:
            return self.latch(
                "observer-failure", now, "warmup observation precedes spawn"
            )
        if self.client_spawn is None or now >= self.deadline():
            return self.latch(
                "deadline", now, "warmup setup observation late/unavailable"
            )
        self.warmup = now
        self.phases["warmup"] = receipt
        if self.on_clock_change:
            self.on_clock_change()

    def entry_quorum(self, snapshots, now):
        if self.target_quorum is not None or self.warmup is None:
            return
        keys, recorded, streams = [], [], {}
        first_layer = self.config["layers"][0]
        expected_cache = (
            self.config["indexer_prefix_template"].format(layer=first_layer)
            + ".k_cache"
        )
        for rank in self.config["ranks"]:
            state = snapshots.get(rank, {})
            markers = state.get("markers", [])
            if not markers or markers[0].get("recorded") is not True:
                return
            marker = markers[0]
            headers = state.get("bracket", [])
            if not headers:
                raise ValueError("recorded entry without request header")
            header, semantic, stream = (
                headers[0],
                marker.get("semantic", {}),
                marker.get("stream", {}),
            )
            when = marker.get("recorded_mono")
            valid = (
                state.get("ready") is True
                and state.get("rank") == rank
                and header.get("context") == [self.config["contexts"][0]]
                and header.get("new") == [self.config["new_tokens"]]
                and header.get("cached")
                == [self.config["contexts"][0] - self.config["new_tokens"]]
                and len(header.get("request_ids", [])) == 1
                and header.get("total") == header.get("prefill") == 1
                and header.get("decode") == 0
                and header.get("dummy") is False
                and marker.get("boundary") == 0
                and marker.get("generation") == 1
                and marker.get("state") in ("recorded-pending", "completed")
                and semantic.get("request") == header["request_ids"][0]
                and semantic.get("forward") == header.get("forward")
                and semantic.get("context") == self.config["contexts"][0]
                and semantic.get("layer") == first_layer
                and semantic.get("cache") == expected_cache
                and semantic.get("layout") == "mfma"
                and stream.get("device") == rank
                and type(stream.get("stream_id")) is int
                and type(stream.get("cuda_stream")) is int
                and type(when) in (int, float)
                and math.isfinite(when)
                and self.client_spawn <= when <= now
            )
            if not valid:
                raise ValueError(f"rank {rank}: first-target entry identity invalid")
            if any(
                m.get("recorded")
                and (m.get("generation") != 1 or m.get("stream") != stream)
                for m in markers[:6]
            ):
                raise ValueError(
                    f"rank {rank}: first-helper generation/stream mismatch"
                )
            keys.append(
                {
                    "header": header,
                    "semantic": semantic,
                    "generation": marker["generation"],
                }
            )
            streams[rank] = dict(stream)
            recorded.append(when)
        if any(key != keys[0] for key in keys):
            raise ValueError("cross-rank first-target entry mismatch")
        self.target_first, self.target_quorum = min(recorded), now
        self.target_identity = dict(
            keys[0], per_rank_streams=streams, last_recorded_mono=max(recorded)
        )
        if self.on_clock_change:
            self.on_clock_change()

    def check(self, snapshots, now, phase=False):
        c = self.config
        if self.reason:
            return self.reason
        if now >= self.deadline():
            return self.latch("deadline", now)
        if phase:
            return self.latch(
                "phase-transition", now, "profiling overlap may already exist"
            )
        for rank, snapshot in snapshots.items():
            if snapshot.get("failure"):
                return self.latch("observer-failure", now, [rank, snapshot["failure"]])
            if now - snapshot["heartbeat"] > c["heartbeat_s"]:
                return self.latch(
                    "observer-failure",
                    now,
                    [rank, "query/export/worker heartbeat unavailable"],
                )
            markers = snapshot.get("markers", [])
            if not markers or not markers[0]["recorded"]:
                continue
            current = frontier(markers)
            previous, since = self.frontiers.get(rank, (0, markers[0]["recorded_mono"]))
            if current > previous:
                since = max(m["completed_mono"] for m in markers[:current])
            self.frontiers[rank] = current, since
            if not snapshot.get("complete") and now - since >= c["no_progress_s"]:
                return self.latch(
                    "no-progress", now, {"rank": rank, "frontier": current}
                )
        try:
            self.entry_quorum(snapshots, now)
        except (ValueError, TypeError, KeyError) as error:
            return self.latch("observer-failure", now, str(error))
        if len(snapshots) == len(c["ranks"]) and all(
            s.get("complete") for s in snapshots.values()
        ):
            signatures = [
                {
                    "bracket": s["bracket"],
                    "pg_order": [
                        [a["semantic"], a["collective_seq_id"]]
                        for a in s.get("associations", [])
                    ],
                }
                for s in snapshots.values()
            ]
            if any(s != signatures[0] for s in signatures):
                return self.latch(
                    "observer-failure",
                    now,
                    "cross-rank request/forward/catalog divergence",
                )
            return self.latch("full-bracket-completion", now)
        return None

    def grace_end(self):
        return min(
            self.reason["mono"] + self.config["evidence_s"],
            self.launched + self.config["evidence_absolute_s"],
        )


class Observer:
    def __init__(self, runner):
        import torch
        from aiter.dist.parallel_state import get_dcp_group as underlying
        from aiter.dist.parallel_state import get_tp_group

        from atom.distributed.dcp_utils import get_dcp_group

        self.torch = torch
        self.plan = json.loads(Path(os.environ["K31_OBSERVER_PLAN"]).read_text())
        self.config = self.plan["protocol"]["observer"]
        self.root = Path(self.plan["output"]) / "observer"
        self.rank = int(runner.rank)
        self.path = self.root / f"rank-{self.rank}"
        self.path.mkdir(parents=True, exist_ok=False)
        if (
            self.plan["mode"] != "warmup_observer"
            or self.rank not in self.config["ranks"]
        ):
            raise ValueError("unexpected observer mode/rank")
        for path in (HERE / "observe.py", HERE / "worker_bootstrap/sitecustomize.py"):
            if digest(path) != self.plan["sources"][str(path)]:
                raise ValueError("worker source hash differs from plan")
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()
        if config_hash != self.plan["observer_config_sha256"]:
            raise ValueError("worker observer config differs")
        self.state = {
            "rank": self.rank,
            "pid": os.getpid(),
            "head": self.plan["head"],
            "config_sha256": config_hash,
            "sources": self.plan["sources"],
            "worker_environment": {
                k: os.environ.get(k) for k in self.plan["environments"]["server"]
            },
            "hooks": list(HOOKS),
            "ready": False,
            "failure": None,
            "markers": [],
            "bracket": [],
            "complete": False,
        }
        self.lock = threading.RLock()
        self.costs = ObserverCosts()
        self.maps, self.receipts, self.calibrations = [], [], []
        self.state["gathers"] = self.receipts
        self.catalog, self.first_request, self.forward_number = {}, None, 0
        self.startup_exported = self.final_exported = False
        self.events = []
        self.initialized = []
        self.stack = (self.path / "python-stacks.txt").open("w")
        faulthandler.register(signal.SIGUSR2, file=self.stack, all_threads=True)
        self.state["stack_registered"] = True
        selected, tp, dcp = get_dcp_group(), get_tp_group(), underlying()
        pg = selected.device_group
        self.group = [str(pg.group_name), str(pg.group_desc)]
        self.state["group"] = {
            "torch_identity": self.group,
            "ranks": list(selected.ranks),
            "tp_ranks": list(tp.ranks),
            "dcp_ranks": list(dcp.ranks),
            "selected_is_tp": selected is tp,
            "underlying_device_group_alias": tp.device_group is dcp.device_group,
            "python_identity_diagnostic_only": id(pg),
            "coordinator": selected.unique_name,
        }
        if list(selected.ranks) != self.config["ranks"] or selected is not tp:
            raise ValueError("selected group/rank ordering changed")
        probe = self.dump()
        self.write(
            "probe.json",
            {
                "version": probe["version"],
                "group": self.state["group"],
                "accessor": pg._get_sequence_number_for_group(),
                "exporter_available": True,
                "file_routing": str(self.path),
            },
        )
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("activation during graph capture")
        count = self.config["pool_size"]
        if (
            count != len(self.config["layers"]) * 2 * 6
            or count > self.config["pool_ceiling"]
        ):
            raise ValueError("invalid bounded marker pool")
        self.state["initial_stream"] = stream_identity(torch)
        for _ in range(count):
            event = torch.cuda.Event(enable_timing=False)
            with self.costs.measure("marker_initial_record"):
                event.record(torch.cuda.current_stream())
            self.events.append(event)
            self.initialized.append(False)
        self.state["pool_generation"] = 0
        self.publish()
        threading.Thread(target=self.loop, daemon=True, name="k31-observer").start()

    def write(self, name, value):
        atomic(self.path / name, value, getattr(self, "costs", None))

    def dump(self, destination=None):
        costs = getattr(self, "costs", None)
        category = (
            "fr_final_call" if destination == "final-trace.json" else "fr_startup_call"
        )
        with costs.measure(category) if costs is not None else nullcontext({}) as item:
            raw = self.torch._C._distributed_c10d._dump_nccl_trace_json(
                includeCollectives=True, onlyActive=False
            )
            if isinstance(raw, bytes):
                item["bytes"] = len(raw)
        if not isinstance(raw, bytes) or len(raw) > self.config["export_bytes"]:
            raise ValueError("FlightRecorder format/size unavailable")
        with (
            costs.measure("fr_decode") if costs is not None else nullcontext({})
        ) as item:
            result = json.loads(raw)
            item["bytes"] = len(raw)
        if not all(k in result for k in ("version", "pg_config", "pg_status")):
            raise ValueError("FlightRecorder schema unavailable")
        if destination:
            self.write(destination, raw)
        return result

    def fail(self, message):
        with self.lock:
            if not self.state["failure"]:
                self.state["failure"] = str(message)

    @measured("publication")
    def publish(self):
        with self.lock:
            self.state["heartbeat"] = time.monotonic()
            self.state["observer_costs"] = (
                self.costs.snapshot()
                if hasattr(self, "costs")
                else {"available": False}
            )
            atomic(self.path / "state.json", self.state, getattr(self, "costs", None))

    def query(self, event, initial=False):
        costs = getattr(self, "costs", None)
        name = "initial_event_query" if initial else "pending_event_query"
        with costs.measure(name) if costs is not None else nullcontext({}) as item:
            value = event.query()
            item["query_result"] = value
            return value

    @measured("descriptor")
    def describe(self, tensor):
        return descriptor(tensor)

    @measured("query_sweep")
    def poll(self):
        for i, event in enumerate(self.events):
            if not self.initialized[i]:
                if self.query(event, initial=True):
                    self.initialized[i] = True
                continue
            with self.lock:
                marker = (
                    self.state["markers"][i] if i < len(self.state["markers"]) else None
                )
            if (
                marker is None
                or not marker["recorded"]
                or marker["state"] != "recorded-pending"
            ):
                continue
            try:
                complete = self.query(event)
            except Exception:
                with self.lock:
                    marker["state"] = "query-unavailable"
                raise
            if complete:
                with self.lock:
                    marker["state"] = "completed"
                    marker["completed_mono"] = time.monotonic()

    def loop(self):
        try:
            while True:
                costs = getattr(self, "costs", None)
                with (
                    costs.measure("iteration_work")
                    if costs is not None
                    else nullcontext()
                ):
                    self.poll()
                    with self.lock:
                        all_done = (
                            len(self.state["markers"]) == self.config["pool_size"]
                            and frontier(self.state["markers"])
                            == self.config["pool_size"]
                        )
                    trigger = (self.root / "trigger.json").exists()
                    if (trigger or all_done) and not self.final_exported:
                        self.final_exported = True
                        trace = self.dump("final-trace.json")
                        joined = []
                        try:
                            joined = associate(
                                trace, self.group, self.calibrations[0], self.receipts
                            )
                        except (ValueError, IndexError) as error:
                            self.state["association_unavailable"] = str(error)
                        entries = sorted(
                            selected_entries(trace, self.group),
                            key=lambda e: e["collective_seq_id"],
                        )
                        first = next(
                            (
                                i
                                for i, e in enumerate(entries)
                                if e["state"] != "completed"
                            ),
                            None,
                        )
                        self.state["communication_frontier"] = {
                            "first_unfinished": (
                                entries[first] if first is not None else None
                            ),
                            "preceding_completed": (
                                entries[first - 1]
                                if first is not None and first
                                else None
                            ),
                            "first_retained_record": (
                                entries[0]["record_id"] if entries else None
                            ),
                            "history_complete_before_ring_start": "unavailable",
                        }
                        with self.lock:
                            self.state["associations"] = joined
                            self.state["complete"] = bool(
                                all_done
                                and len(joined) == 96
                                and not self.state["failure"]
                            )
                    self.publish()
                    if trigger:
                        return
                time.sleep(0.1)
        except Exception as error:  # noqa: BLE001
            self.fail(f"observer query/export: {type(error).__name__}: {error}")
            self.publish()

    @measured("host_snapshot")
    def seal(self, runner):
        if self.state["worker_environment"] != self.plan["environments"]["server"]:
            raise ValueError("effective worker environment differs from server plan")
        if any(
            digest(p) != expected for p, expected in self.plan["native_sources"].items()
        ):
            raise ValueError("native seam source hash changed")
        self.state["native_sources"] = self.plan["native_sources"]
        if runner.config.dcp_config.interleave_size != 1 or runner.block_size != 64:
            raise ValueError("block/interleave contract differs")
        expected = {
            "ModelRunner._setup_device_and_distributed",
            "ModelRunner.forward",
            "ModelRunner.prepare_model",
            "ModelRunner._maybe_warmup",
            "AiterMLAMetadataBuilder._build_dcp_indexer_fp4_prefill_meta",
            "MLAAttention.process_weights_after_loading",
            "GroupCoordinator.all_gather",
            "atom.model_ops.indexer.indexer_forward._index_cache",
            "atom.model_ops.indexer.indexer_forward._dcp_stage_indexer_fp4_prefill",
        }
        if set(HOOKS) != expected:
            raise ValueError("worker hook registration incomplete")
        expected_catalog = {}
        for layer in self.config["layers"]:
            prefix = self.config["indexer_prefix_template"].format(layer=layer)
            expected_catalog[prefix] = {"layer": layer, "cache": prefix + ".k_cache"}
        context = runner.config.compilation_config.static_forward_context
        for prefix, item in context.items():
            if type(item).__name__ == "Indexer":
                if (
                    prefix not in expected_catalog
                    or not item._indexer_fp4
                    or item.k_cache.prefix != prefix + ".k_cache"
                ):
                    raise ValueError("unexpected target/draft indexer catalog")
                self.catalog[prefix] = expected_catalog[prefix]
        if self.catalog != expected_catalog:
            raise ValueError("runtime eligible indexer catalog differs")
        if len(self.calibrations) != 2 or self.calibrations[0] != self.calibrations[1]:
            raise ValueError("normal startup sequence calibration unavailable")
        self.poll()
        if not all(self.initialized):
            raise ValueError("lazy marker initialization not completed before warmup")
        trace = self.dump("startup-trace.json")
        if not selected_entries(trace, self.group):
            raise ValueError("selected-PG startup history missing")
        self.startup_exported = True
        with self.lock:
            self.state.update(
                catalog=self.catalog,
                calibration=self.calibrations,
                ready=True,
                pool_ready=True,
                hooks=list(HOOKS),
            )
        self.publish()

    @measured("host_snapshot")
    def prepare(self, runner, batch):
        self.forward_number += 1
        TLS.semantic = TLS.target = None
        if type(batch.is_dummy_run) is not bool:
            raise TypeError("is_dummy_run must be a native bool")
        if batch.is_dummy_run or self.torch.cuda.is_current_stream_capturing():
            return
        external_ids = getattr(batch, "external_request_ids", None)
        if not isinstance(batch.req_ids, list) or (
            external_ids is not None and not isinstance(external_ids, list)
        ):
            raise TypeError("request identities must be native host lists")
        semantic = {
            "forward": self.forward_number,
            "request_ids": list(batch.req_ids),
            "external_request_ids": external_ids,
            "new": host_counts(batch.num_scheduled_tokens, "num_scheduled_tokens"),
            "cached": host_counts(batch.num_cached_tokens, "num_cached_tokens"),
            "context": host_counts(batch.context_lens, "context_lens"),
            "dummy": batch.is_dummy_run,
            "total": host_int(batch.total_seqs_num, "total_seqs_num"),
            "prefill": host_int(batch.total_seqs_num_prefill, "total_seqs_num_prefill"),
            "decode": host_int(batch.total_seqs_num_decode, "total_seqs_num_decode"),
            "publication_epoch": host_int(runner.h2d_owner.epoch, "publication_epoch"),
            "allocation_generation": "unavailable",
            "input_fingerprint": "unavailable",
        }
        # Native batch fields are CPU arrays/scalars. Serialize now to prevent aliasing.
        costs = getattr(self, "costs", None)
        with (
            costs.measure("json_encode") if costs is not None else nullcontext({})
        ) as item:
            encoded = json.dumps(semantic)
            item["bytes"] = len(encoded.encode())
        semantic = json.loads(encoded)
        TLS.semantic, TLS.target = semantic, None
        if (
            semantic["dummy"]
            or semantic["total"] != 1
            or semantic["prefill"] != 1
            or semantic["decode"] != 0
            or len(semantic["request_ids"]) != 1
            or semantic["new"] != [self.config["new_tokens"]]
            or len(semantic["cached"]) != 1
            or len(semantic["context"]) != 1
            or semantic["cached"][0] + semantic["new"][0] != semantic["context"][0]
        ):
            return
        req, length = semantic["request_ids"][0], semantic["context"][0]
        if self.first_request is None and length == self.config["contexts"][0]:
            self.first_request = req
        if req == self.first_request and length in self.config["contexts"]:
            if length in [x["context"][0] for x in self.state["bracket"]]:
                raise ValueError("duplicate candidate forward")
            if length != self.config["contexts"][len(self.state["bracket"])]:
                raise ValueError("candidate forward order changed")
            TLS.target = semantic
            with self.lock:
                self.state["bracket"].append(semantic)

    @measured("host_snapshot")
    def metadata(self, builder, bs, lpad, cu_pad, g_cu, g_lens, block_tables):
        target = getattr(TLS, "target", None)
        if target is None:
            return
        import numpy as np

        if (
            bs != 1
            or g_lens.tolist() != target["context"]
            or builder.model_runner.block_size != 64
        ):
            raise ValueError("candidate metadata mismatch")
        arrays = {}
        for name, value in zip(
            ("block_tables", "g_lens", "lpad", "cu_pad", "g_cu"),
            (block_tables, g_lens, lpad, cu_pad, g_cu),
        ):
            if not isinstance(value, np.ndarray):
                raise TypeError("metadata witness is not host ndarray")
            costs = getattr(self, "costs", None)
            with (
                costs.measure("host_array_copy")
                if costs is not None
                else nullcontext({})
            ) as item:
                arrays[name] = value.copy(order="C")
                arrays[name].flags.writeable = False
                item["bytes"] = arrays[name].nbytes
        index = self.config["contexts"].index(target["context"][0])
        if int(cu_pad[bs]) != self.config["local_padded"][index]:
            raise ValueError("candidate local padded count changed")
        # No device readback: owned CPU maps are queued, never metadata/tensors.
        if len(self.maps) >= 2:
            raise ValueError("duplicate candidate metadata")
        self.maps.append(arrays)
        record = {
            "semantic": target,
            "arrays": {k: v.tolist() for k, v in arrays.items()},
            "producer_stream": stream_identity(self.torch),
            "array_layout": {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "strides": list(v.strides),
                    "bytes": v.nbytes,
                    "owned": v.flags.owndata,
                    "writeable": v.flags.writeable,
                }
                for k, v in arrays.items()
            },
        }
        self.write(f"host-map-{index}.json", record)

    @measured("host_snapshot")
    def published(self, metadata):
        if getattr(TLS, "target", None) is not None:
            record = {
                "semantic": TLS.target,
                "stream": stream_identity(self.torch),
                "source": "upload_numpy and decomposition use current stream",
                "fields": {
                    name: self.describe(getattr(metadata, name))
                    for name in (
                        "dcp_indexer_fp4_read_page",
                        "dcp_indexer_fp4_read_row",
                        "dcp_indexer_fp4_stage_page",
                        "dcp_indexer_fp4_stage_row",
                        "dcp_indexer_gather_index",
                        "dcp_indexer_fp4_block_tables",
                    )
                },
            }
            with self.lock:
                self.state.setdefault("metadata_publication", []).append(record)

    @measured("host_snapshot")
    def begin_call(self, prefix, cache, scale, metadata, layout):
        target = getattr(TLS, "target", None)
        if target is None or self.torch.cuda.is_current_stream_capturing():
            return None
        if not self.state["ready"] or prefix not in self.catalog:
            raise ValueError("unqualified helper/catalog")
        ordinal = len(self.state["markers"]) // 6
        index, layer_index = divmod(ordinal, len(self.config["layers"]))
        if (
            index >= 2
            or target["context"] != [self.config["contexts"][index]]
            or self.catalog[prefix]["layer"] != self.config["layers"][layer_index]
        ):
            raise ValueError("eligible helper order or coverage differs")
        key = {
            "request": target["request_ids"][0],
            "context": target["context"][0],
            "forward": target["forward"],
            **self.catalog[prefix],
            "layout": str(layout),
        }
        slots = [
            {
                "semantic": key,
                "boundary": i,
                "generation": ordinal + 1,
                "recorded": False,
                "state": "not-recorded",
                "stream": None,
            }
            for i in range(6)
        ]
        with self.lock:
            self.state["markers"].extend(slots)
            self.state.setdefault("operands", []).append(
                {
                    "semantic": key,
                    "cache": self.describe(cache),
                    "scale": self.describe(scale),
                    "read_page": self.describe(metadata.dcp_indexer_fp4_read_page),
                    "read_row": self.describe(metadata.dcp_indexer_fp4_read_row),
                    "gather_index": self.describe(metadata.dcp_indexer_gather_index),
                    "stage_page": self.describe(metadata.dcp_indexer_fp4_stage_page),
                    "stage_row": self.describe(metadata.dcp_indexer_fp4_stage_row),
                    "stage_pages": int(metadata.dcp_indexer_fp4_stage_pages),
                }
            )
        call = {"offset": ordinal * 6, "semantic": key, "gathers": 0}
        self.mark(call, 0)
        return call

    def mark(self, call, boundary):
        slot = call["offset"] + boundary
        if slot >= len(self.events) or not self.initialized[slot]:
            raise ValueError("marker pool unavailable/exhausted")
        marker = self.state["markers"][slot]
        if marker["recorded"]:
            raise ValueError("marker re-record attempted")
        stream = stream_identity(self.torch)
        if boundary and stream != self.state["markers"][call["offset"]]["stream"]:
            raise ValueError("helper stream identity changed")
        costs = getattr(self, "costs", None)
        with costs.measure("marker_record") if costs is not None else nullcontext():
            self.events[slot].record(self.torch.cuda.current_stream())
        with self.lock:
            marker.update(
                recorded=True,
                state="recorded-pending",
                stream=stream,
                recorded_mono=time.monotonic(),
            )


def safe(method, *args):
    try:
        return method(*args)
    except Exception as error:  # noqa: BLE001
        if RUNTIME:
            RUNTIME.fail(f"{method.__name__}: {type(error).__name__}: {error}")
        return None


def activate(runner):
    global RUNTIME
    try:
        if RUNTIME is not None:
            raise ValueError("duplicate worker activation")
        RUNTIME = Observer(runner)
    except Exception as error:  # noqa: BLE001
        plan = json.loads(Path(os.environ["K31_OBSERVER_PLAN"]).read_text())
        atomic(
            Path(plan["output"]) / "observer" / f"rank-{runner.rank}" / "state.json",
            {
                "rank": int(runner.rank),
                "pid": os.getpid(),
                "heartbeat": time.monotonic(),
                "failure": f"activation: {type(error).__name__}: {error}",
                "ready": False,
            },
        )


def wrap(owner, name, factory):
    original = getattr(owner, name)
    setattr(owner, name, functools.wraps(original)(factory(original)))
    HOOKS.append(f"{owner.__name__}.{name}")
    NATIVE_SOURCES[original.__code__.co_filename] = digest(
        original.__code__.co_filename
    )


def patch(module):
    if module.__name__ == "atom.model_engine.model_runner":
        cls = module.ModelRunner

        def setup(original):
            def hooked(self, *a, **kw):
                result = original(self, *a, **kw)
                activate(self)
                return result

            return hooked

        def forward(original):
            def hooked(self, *a, **kw):
                TLS.semantic = TLS.target = None
                try:
                    return original(self, *a, **kw)
                finally:
                    TLS.semantic = TLS.target = None

            return hooked

        def prepare(original):
            def hooked(self, batch, *a, **kw):
                if RUNTIME:
                    safe(RUNTIME.prepare, self, batch)
                return original(self, batch, *a, **kw)

            return hooked

        def seal(original):
            def hooked(self, *a, **kw):
                if RUNTIME and not RUNTIME.startup_exported:
                    safe(RUNTIME.seal, self)
                return original(self, *a, **kw)

            return hooked

        for name, factory in (
            ("_setup_device_and_distributed", setup),
            ("forward", forward),
            ("prepare_model", prepare),
            ("_maybe_warmup", seal),
        ):
            wrap(cls, name, factory)
    elif module.__name__ == "atom.model_ops.attentions.aiter_mla":

        def builder(original):
            def hooked(self, metadata, bs, lpad, cu_pad, g_cu, g_lens, block_tables):
                if RUNTIME:
                    safe(
                        RUNTIME.metadata,
                        self,
                        bs,
                        lpad,
                        cu_pad,
                        g_cu,
                        g_lens,
                        block_tables,
                    )
                result = original(
                    self, metadata, bs, lpad, cu_pad, g_cu, g_lens, block_tables
                )
                if RUNTIME:
                    safe(RUNTIME.published, metadata)
                return result

            return hooked

        wrap(
            module.AiterMLAMetadataBuilder,
            "_build_dcp_indexer_fp4_prefill_meta",
            builder,
        )
    elif module.__name__ == "atom.model_ops.attention_mla":

        def weights(original):
            def hooked(self, *a, **kw):
                TLS.loading = True
                try:
                    return original(self, *a, **kw)
                finally:
                    TLS.loading = False

            return hooked

        wrap(module.MLAAttention, "process_weights_after_loading", weights)
    elif module.__name__ == "atom.model_ops.indexer.indexer_forward":

        def cache(original):
            def hooked(indexer, *a, **kw):
                TLS.cache_prefix = indexer.prefix
                try:
                    return original(indexer, *a, **kw)
                finally:
                    TLS.cache_prefix = None

            return hooked

        def helper(original):
            def hooked(cache, scale, metadata, layout=module.PRESHUFFLE):
                call = (
                    safe(
                        RUNTIME.begin_call,
                        getattr(TLS, "cache_prefix", None),
                        cache,
                        scale,
                        metadata,
                        layout,
                    )
                    if RUNTIME
                    else None
                )
                TLS.call = call
                try:
                    result = original(cache, scale, metadata, layout)
                    if call:
                        if call["gathers"] != 2:
                            RUNTIME.fail(
                                "helper did not issue exactly data/scale gathers"
                            )
                        safe(RUNTIME.mark, call, 5)
                    return result
                finally:
                    TLS.call = None

            return hooked

        wrap(module, "_index_cache", cache)
        wrap(module, "_dcp_stage_indexer_fp4_prefill", helper)
    elif module.__name__ == "aiter.dist.parallel_state":

        def gather(original):
            def hooked(group, tensor, use_custom=False, dim=-1):
                runtime, call = RUNTIME, getattr(TLS, "call", None)
                startup = bool(
                    runtime
                    and getattr(TLS, "loading", False)
                    and len(runtime.calibrations) < 2
                )
                receipt = before_trace = None
                if runtime and (call or startup):

                    def before():
                        pg = group.device_group
                        if (
                            [str(pg.group_name), str(pg.group_desc)] != runtime.group
                            or use_custom
                            or dim != 0
                        ):
                            raise ValueError(
                                "observed gather group/backend/axis changed"
                            )
                        trace = runtime.dump() if startup else None
                        r = {
                            "before": pg._get_sequence_number_for_group(),
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "semantic": (
                                dict(call["semantic"], plane=call["gathers"])
                                if call
                                else {"startup": len(runtime.calibrations)}
                            ),
                        }
                        if call:
                            if call["gathers"] >= 2:
                                raise ValueError("extra eligible helper gather")
                            runtime.mark(call, 1 + call["gathers"] * 2)
                            with runtime.lock:
                                runtime.receipts.append(r)
                        return r, trace

                    costs = getattr(runtime, "costs", None)
                    with (
                        costs.measure("host_snapshot")
                        if costs is not None
                        else nullcontext()
                    ):
                        captured = safe(before)
                    if captured:
                        receipt, before_trace = captured
                try:
                    result = original(group, tensor, use_custom=use_custom, dim=dim)
                except Exception as error:
                    if receipt:
                        receipt["native_exception"] = type(error).__name__
                        runtime.fail(
                            "selected native gather raised " + type(error).__name__
                        )
                    raise
                if receipt:

                    def after():
                        receipt["after"] = (
                            group.device_group._get_sequence_number_for_group()
                        )
                        if call:
                            runtime.mark(call, 2 + call["gathers"] * 2)
                            call["gathers"] += 1
                        else:
                            calibration = calibrate(
                                before_trace, runtime.dump(), receipt, runtime.group
                            )
                            runtime.calibrations.append(calibration)
                            runtime.write(
                                f"calibration-{len(runtime.calibrations)}.json",
                                {"receipt": receipt, "mapping": calibration},
                            )

                    costs = getattr(runtime, "costs", None)
                    with (
                        costs.measure("host_snapshot")
                        if costs is not None
                        else nullcontext()
                    ):
                        safe(after)
                return result

            return hooked

        wrap(module.GroupCoordinator, "all_gather", gather)


class NativeHookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            original = spec.loader

            class Loader:
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    patch(module)

            spec.loader = Loader()
        return spec


def install():
    sys.meta_path.insert(0, NativeHookFinder())
