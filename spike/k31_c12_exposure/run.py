"""Execute only the reviewed, bounded exposure plan; never a candidate run."""

import argparse
import datetime
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from plan import PRIVATE, ROOT, WS, P, git, make_plan, sha

logger = logging.getLogger(__name__)
COMMON = WS / "harness/common.py"
LOCK = "/app/agent-team-scratch/g24-gpu.lock"
PAUSE = WS / "lead/GPU_PAUSE"
PHASE = re.compile(r"\bPhase \S+ \(profiling\) started\b")
OBSERVER_PHASE = re.compile(r"\bPhase (\S+) \((warmup|profiling)\) started\b")
RANK_EVENT = re.compile(r"Rank (\d+): profiler (started|stop completed)")
EXPORT = re.compile(r"Rank (\d+): trace exported to (\S+\.json\.gz)")
RANKS = {str(rank) for rank in range(8)}


class EvidenceDeadline(BaseException):
    """Bypass request retry handlers and end evidence collection immediately."""


def arm_evidence_deadline(observer):
    def expired(signum, frame):
        raise EvidenceDeadline()

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, max(0.001, observer.grace_end() - time.monotonic())
    )
    # Arm before any stop/receipt I/O. The callback runs once, at the first latch.
    process = getattr(observer, "client_process", None)
    observer.client_exit_at_stop = process.poll() if process is not None else None
    observer.client_censored = (
        process is not None and observer.client_exit_at_stop is None
    )
    if observer.client_censored:
        observer.client_stop_requested_mono = time.monotonic()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def save(path, value, costs=None):
    if costs is not None:
        from observe import atomic

        return atomic(path, value, costs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scan_observer_phases(observer, directory):
    with observer.costs.measure("controller_phase_scan") as cost:
        cost["bytes"] = 0
        for path in (directory / "logs/aiperf.log", directory / "console.log"):
            if not path.exists():
                continue
            raw = path.read_bytes()
            cost["bytes"] += len(raw)
            for line in raw.decode(errors="replace").splitlines():
                match = OBSERVER_PHASE.search(line)
                if not match or match[2] in observer.phases:
                    continue
                now, wall = time.monotonic(), time.time()
                stamp = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)", line)
                producer = (
                    datetime.datetime.fromisoformat(stamp[1]).timestamp()
                    if stamp
                    else None
                )
                receipt = {
                    "phase_id": match[1],
                    "source": str(path),
                    "observed_mono": now,
                    "observed_wall": wall,
                    "producer_timestamp_raw": stamp[1] if stamp else line[:12],
                    "producer_wall": producer,
                    "producer_clock_assumption": "naive timestamp in shared container local timezone",
                    "container_timezone": list(time.tzname),
                    "discovery_delay_wall_s": (
                        wall - producer if producer is not None else None
                    ),
                    "observed_after_stop": observer.reason is not None,
                }
                if observer.reason is None and now >= observer.deadline():
                    observer.latch(
                        "deadline", now, "phase observation at expired bound"
                    )
                if match[2] == "warmup" and observer.reason is None:
                    observer.observe_warmup(receipt)
                else:
                    observer.phases[match[2]] = receipt
                if match[2] == "profiling" and observer.reason is None:
                    observer.latch(
                        "phase-transition",
                        now,
                        "profiling traffic may precede this log",
                    )


def observer_client_ledger(directory):
    """Counts only available records; progress is never a terminal cancellation ledger."""
    from atom.benchmarks.results.records import from_aiperf

    result = {
        "terminal": None,
        "last_progress": {},
        "exports": None,
        "limitation": "Unknown terminal counts remain unknown; exported rows/progress do not prove all requests settled.",
    }
    logs = [directory / "logs/aiperf.log", directory / "console.log"]
    source = next((p for p in logs if p.exists()), None)
    if source:
        result["log_source"] = str(source)
        for line in source.read_text(errors="replace").splitlines():
            progress = re.search(
                r"Phase (warmup|profiling) progress \| returned=([\d,]+)/([\d,]+) \| sent=([\d,]+) \| in_flight=([\d,]+) \| errors=([\d,]+)",
                line,
            )
            if progress:
                result["last_progress"][progress[1]] = dict(
                    zip(
                        ("returned", "target", "sent", "in_flight", "errors"),
                        (int(v.replace(",", "")) for v in progress.groups()[1:]),
                    )
                )
            terminal = re.search(
                r"Phase (\S+) \((warmup|profiling)\) complete \| completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)",
                line,
            )
            if terminal:
                if result["terminal"] is None:
                    result["terminal"] = {}
                result["terminal"][terminal[2]] = dict(
                    zip(
                        ("completed", "cancelled", "errors"),
                        (int(v.replace(",", "")) for v in terminal.groups()[2:]),
                    )
                )
    path = directory / "profile_export.jsonl"
    if path.exists():
        counts = {
            phase: {
                status: 0 for status in ("success", "error", "cancelled", "unknown")
            }
            for phase in ("warmup", "profiling", "unknown")
        }
        result["exports"] = {
            "source": str(path),
            "rows": 0,
            "counts": counts,
            "parse_complete": True,
        }
        with path.open() as source:
            for index, line in enumerate(source):
                try:
                    row = from_aiperf(json.loads(line), index)
                except (ValueError, TypeError) as error:
                    result["exports"].update(
                        parse_complete=False,
                        unavailable_after_line=index + 1,
                        parse_error=type(error).__name__,
                    )
                    break
                phase = row["phase"] if row["phase"] in counts else "unknown"
                status = row["status"] if row["status"] in counts[phase] else "unknown"
                counts[phase][status] += 1
                result["exports"]["rows"] += 1
    return result


def observer_terminal_receipt(observer, state):
    process = getattr(observer, "client_process", None)
    result = {
        "spawn_attempted": observer.client_spawn is not None,
        "spawned": process is not None,
        "pid": getattr(observer, "client_pid", None),
        "spawn_return_mono": getattr(observer, "client_spawn_return", None),
        "exit_code_at_stop": getattr(observer, "client_exit_at_stop", None),
        "exit_code": process.poll() if process is not None else None,
        "censored": getattr(observer, "client_censored", False),
        "stop_request_mono": getattr(observer, "client_stop_requested_mono", None),
        "stop": observer.reason,
        "anchors": observer.anchors(),
        "phase_observed": "profiling" in observer.phases,
        "performance_accepted": False,
        "ledger": {
            "available": False,
            "terminal": None,
            "reason": "not collected within grace",
        },
    }
    state["observer_client"] = (
        result  # Preserve scalar process evidence before any I/O.
    )
    directory = getattr(observer, "client_directory", None)
    if directory is None:
        return
    try:
        scan_observer_phases(observer, directory)
        result["phase_observed"] = "profiling" in observer.phases
        result["anchors"] = observer.anchors()
        with observer.costs.measure("client_ledger"):
            result["ledger"] = observer_client_ledger(directory)
    except (OSError, ValueError, TypeError) as error:
        result["ledger"] = {
            "available": False,
            "terminal": None,
            "reason": type(error).__name__,
        }
    save(directory / "client-result.json", result, observer.costs)


def request(url, timeout=2, post=False):
    with urlopen(Request(url, data=b"" if post else None), timeout=timeout) as response:
        return response.read().decode()


def profile_window(url, out, server_log, trace_root):
    t0 = time.monotonic()
    cursor = server_log.stat().st_size
    before = {str(p.resolve()) for p in trace_root.rglob("*.json.gz")}
    evidence = {
        "start_request_wall": time.time(),
        "start_request_mono": t0,
        "started": {},
        "stopped": {},
        "exports": {},
        "errors": [],
        "before_files": sorted(before),
    }
    pending = b""
    with server_log.open("rb") as source:
        source.seek(max(0, cursor - 1))
        skip_partial = bool(cursor and source.read(1) != b"\n")

    def observe():
        nonlocal cursor, pending, skip_partial
        with server_log.open("rb") as source:
            source.seek(cursor)
            pending += source.read()
            cursor = source.tell()
        lines = pending.split(b"\n")
        pending = lines.pop()
        if skip_partial and lines:
            lines.pop(0)
            skip_partial = False
        for raw in lines:
            line = raw.decode(errors="replace")
            now = time.monotonic()
            match = RANK_EVENT.search(line)
            if match:
                rank, event = match.groups()
                field = "started" if event == "started" else "stopped"
                if field == "stopped" and rank not in evidence["started"]:
                    evidence["errors"].append(f"rank {rank}: stop without fresh start")
                if rank in evidence[field]:
                    evidence["errors"].append(f"rank {rank}: duplicate {field}")
                evidence[field][rank] = now
            match = EXPORT.search(line)
            if match:
                rank = match[1]
                if rank in evidence["exports"] or rank not in evidence["started"]:
                    evidence["errors"].append(
                        f"rank {rank}: unordered/duplicate export"
                    )
                evidence["exports"][rank] = match[2]
            if "profiler stop failed" in line or "failed to export trace" in line:
                evidence["errors"].append(line)

    try:
        evidence["start_response"] = json.loads(
            request(url + "/start_profile", P["profile_start_timeout_s"], True)
        )
        evidence["start_ack_mono"] = time.monotonic()
        end = time.monotonic() + P["profile_hold_s"]
        while time.monotonic() < end:
            observe()
            time.sleep(0.05)
    except (OSError, ValueError, HTTPException) as error:
        evidence["errors"].append(f"start: {error!r}")
    finally:
        evidence["stop_request_mono"] = time.monotonic()
        try:
            evidence["stop_response"] = json.loads(
                request(url + "/stop_profile", P["trace_export_timeout_s"], True)
            )
        except (OSError, ValueError, HTTPException) as error:
            evidence["errors"].append(f"stop: {error!r}")
        deadline = evidence["stop_request_mono"] + P["trace_export_timeout_s"]
        while time.monotonic() < deadline:
            observe()
            if set(evidence["stopped"]) == RANKS:
                break
            time.sleep(0.1)
        observe()
        evidence["upper_bound_s"] = {
            r: stopped - t0 for r, stopped in evidence["stopped"].items()
        }
        if set(evidence["stopped"]) != RANKS or set(evidence["exports"]) != RANKS:
            evidence["errors"].append(
                "all-rank stop/export completion was not observed"
            )
        if any(
            v > P["max_profile_upper_bound_s"]
            for v in evidence["upper_bound_s"].values()
        ):
            evidence["errors"].append("profile active-time upper bound exceeded")
        save(out, evidence)
    if evidence["errors"]:
        raise RuntimeError(f"profiler control/export failed; no further window: {out}")
    return evidence


def scan_save_phases(clock, directory):
    for path in (directory / "logs/aiperf.log", directory / "console.log"):
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = OBSERVER_PHASE.search(line)
            if match and match[2] not in clock.phases:
                stamp = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)", line)
                clock.phases[match[2]] = {
                    "phase_id": match[1],
                    "observed_mono": time.monotonic(),
                    "observed_wall": time.time(),
                    "producer_timestamp_raw": stamp[1] if stamp else None,
                    "producer_clock_assumption": "container local timezone",
                    "source": str(path),
                }


def save_completion_receipt(clock, state, collect=True):
    process = getattr(clock, "client_process", None)
    result = {
        "spawn_attempted": clock.client_spawn is not None,
        "spawned": process is not None,
        "pid": getattr(clock, "client_pid", None),
        "spawn_return_mono": getattr(clock, "client_spawn_return", None),
        "exit_code_at_stop": getattr(clock, "client_exit_at_stop", None),
        "exit_code": process.poll() if process is not None else None,
        "censored": getattr(clock, "client_censored", False),
        "anchors": clock.anchors(),
        "phase_observed": "profiling" in clock.phases,
        "instrumented": True,
        "performance_accepted": False,
        "ledger": {
            "available": False,
            "terminal": None,
            "reason": "not collected within grace",
        },
    }
    state["save_observer_client"] = result
    directory = getattr(clock, "client_directory", None)
    if directory is not None and collect:
        try:
            scan_save_phases(clock, directory)
            result["anchors"], result["phase_observed"] = (
                clock.anchors(),
                "profiling" in clock.phases,
            )
            result["ledger"] = observer_client_ledger(directory)
        except (OSError, ValueError, TypeError) as error:
            result["ledger"]["reason"] = type(error).__name__
        save(directory / "client-result.json", result)


def save_control(clock, out, frozen):
    deadline = (
        clock.grace_end()
        if clock.reason is not None
        else clock.launched + clock.config["evidence_absolute_s"]
    )
    path = out / "save-observer-control.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "head": frozen["head"],
                "evidence_end_mono": deadline,
                "stop": clock.reason,
                "launch_mono": clock.launched,
            }
        )
        + "\n"
    )
    temporary.replace(path)


def run_client(
    C, frozen, out, label, fd, track, check_deadline, observer=None, save_clock=None
):
    directory = out / label
    directory.mkdir()
    argv = frozen["client"] + ["--output-artifact-dir", str(directory)]
    env = frozen["environments"]["client"]
    save(directory / "launch.json", {"argv": argv, "environment": env})
    if observer:
        observer.client_directory = directory
        observer.start_client(time.monotonic())
        if observer.reason:
            return
    if save_clock is not None:
        save_clock.client_directory = directory
        save_clock.start_client(time.monotonic())
        if save_clock.reason:
            return
    try:
        process = C.spawn(
            argv,
            env=env,
            cwd=ROOT,
            log_path=directory / "console.log",
            lock_fd=fd,
            cpus="0-111",
        )
    except Exception as error:
        if observer:
            observer.latch("client-failure", time.monotonic(), type(error).__name__)
        if save_clock is not None:
            save_clock.latch("client-failure", time.monotonic(), type(error).__name__)
        raise
    if observer:
        observer.client_pid, observer.client_process = process.pid, process
        observer.client_spawn_return = time.monotonic()
    track(process)
    if save_clock is not None:
        save_clock.client_pid, save_clock.client_process = process.pid, process
        save_clock.client_spawn_return = time.monotonic()
        while process.poll() is None and save_clock.reason is None:
            scan_save_phases(save_clock, directory)
            check_deadline()
            time.sleep(0.1)
        save_clock.latch(
            "invocation-end" if process.poll() == 0 else "client-failure",
            time.monotonic(),
            {"exit_code": process.poll()},
        )
        return
    if observer:
        while process.poll() is None and observer.reason is None:
            scan_observer_phases(observer, directory)
            observer.check(observer_snapshots(out, frozen, observer), time.monotonic())
            if observer.reason is None:
                check_deadline()
                time.sleep(0.1)
        if observer.reason is None:
            observer.latch(
                "client-failure",
                time.monotonic(),
                {"exit_code": process.poll(), "coverage": "incomplete before stop"},
            )
        return
    deadline = time.monotonic() + P["client_timeout_s"]
    expired = threading.Event()

    def expire():
        expired.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            save(
                directory / "censored.json",
                {"deadline_mono": deadline, "observed_mono": time.monotonic()},
            )
        except OSError:
            logger.exception(
                "Client terminated at deadline; censor receipt write failed"
            )

    watchdog = threading.Timer(P["client_timeout_s"], expire)
    watchdog.daemon = True
    watchdog.start()
    windows = []
    try:
        result = _monitor_client(
            process, directory, label, out, windows, check_deadline
        )
    finally:
        watchdog.cancel()
        watchdog.join()
    result["censored"] = expired.is_set()
    save(directory / "client-result.json", result)
    if expired.is_set():
        raise TimeoutError(f"{label} censored by external client bound")
    if (
        process.returncode
        or not result["phase_observed"]
        or not (directory / "profile_export_aiperf.json").exists()
    ):
        raise RuntimeError(f"{label} produced no usable completed invocation")
    from analyze import request_ledger

    ledger = request_ledger(directory)
    save(directory / "ledger.json", ledger)
    if not ledger["usable_completed_requests"]:
        raise RuntimeError(f"{label} request ledger failed; no implicit retry")


def _monitor_client(process, directory, label, out, windows, check_deadline):
    phase, index = None, 0
    while process.poll() is None:
        check_deadline()
        logs = [directory / "logs/aiperf.log", directory / "console.log"]
        if phase is None:
            for log in logs:
                if log.exists() and PHASE.search(log.read_text(errors="replace")):
                    phase = time.monotonic()
                    save(
                        directory / "phase-start.json",
                        {
                            "observed_wall": time.time(),
                            "observed_mono": phase,
                            "source": str(log),
                        },
                    )
                    break
        if label == "diagnostic" and phase is not None and index < 2:
            scheduled = phase + P["profile_offsets_s"][index]
            if time.monotonic() >= scheduled:
                if time.monotonic() - scheduled > 1:
                    windows.append(
                        {"window": index, "state": "missed; not rescheduled"}
                    )
                else:
                    windows.append(
                        profile_window(
                            f"http://127.0.0.1:{P['port']}",
                            directory / f"window{index}.json",
                            out / "server.log",
                            out / "traces",
                        )
                    )
                index += 1
        time.sleep(0.1)
    return {
        "exit_code": process.returncode,
        "phase_observed": phase is not None,
        "windows": windows,
        "instrumented": label == "diagnostic",
    }


def capture_metadata(frozen, out):
    import aiter
    import aiter_worker_limits

    import atom
    from atom.benchmarks.results.metadata import capture_config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.backends import VLLM_CACHE_ROOT

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args, extra = parser.parse_known_args(frozen["server"][3:])
    if extra != ["--host", "127.0.0.1", "--server-port", str(P["port"])]:
        raise RuntimeError(f"unexpected serving CLI arguments: {extra}")
    kwargs = EngineArgs.from_cli_args(args)._get_engine_kwargs()
    if (
        kwargs["mark_trace"]
        or str(Path(VLLM_CACHE_ROOT) / "torch_compile_cache")
        != frozen["atom_compile_cache"]
    ):
        raise RuntimeError("profiler or ATOM compile-cache binding changed")
    save(out / "engine-kwargs.json", json.loads(json.dumps(kwargs, default=str)))
    env = frozen["environments"]["server"]
    config = capture_config(
        P["model"],
        frozen["server"],
        "agentic",
        P["concurrency"],
        {"argv": frozen["server"], "environment": env},
    )
    if frozen.get("mode") == "save_completion_observer":
        config["validity"]["instrumented"] = True
    save(out / "installed-metadata-config.json", config)
    actual = {}
    for name, module, root, commit in (
        ("atom", atom, ROOT, frozen["head"]),
        ("aiter", aiter, Path(P["aiter_root"]), P["aiter_sha"]),
        (
            "aiter_worker_limits",
            aiter_worker_limits,
            Path(P["aiter_root"]),
            P["aiter_sha"],
        ),
    ):
        path = Path(module.__file__).resolve()
        source_sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        actual[name] = {"path": str(path), "sha": source_sha, "dirty": dirty}
        if not path.is_relative_to(root.resolve()) or source_sha != commit or dirty:
            raise RuntimeError(f"runtime source mismatch: {name}: {actual[name]}")
    ck_sha = subprocess.check_output(
        [
            "git",
            "-C",
            str(Path(P["aiter_root"]) / "3rdparty/composable_kernel"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if ck_sha != P["ck_sha"]:
        raise RuntimeError("CK source revision changed")
    actual["ck_sha"] = ck_sha
    save(out / "runtime-imports.json", actual)
    config["software"]["aiter_sha"] = actual["aiter"]["sha"]
    config["software"]["aiter_sha_source"] = "verified_runtime_import_and_git"
    save(out / "base-config.json", config)


def metrics(out, label, state):
    for suffix, port in (("server", P["port"]), ("cache", P["cache_http_port"])):
        try:
            (out / f"{label}-{suffix}.prom").write_text(
                request(f"http://127.0.0.1:{port}/metrics")
            )
        except (OSError, HTTPException) as error:
            state["errors"].append(f"metrics {suffix}: {error!r}")


def observer_snapshots(out, frozen, observer=None, _cost=None):
    if observer is not None:
        with observer.costs.measure("controller_snapshot_read") as cost:
            cost["bytes"] = 0
            return observer_snapshots(out, frozen, _cost=cost)

    snapshots = {}
    for rank in frozen["protocol"]["observer"]["ranks"]:
        path = out / "observer" / f"rank-{rank}" / "state.json"
        if not path.exists():
            continue
        raw = path.read_bytes()
        if _cost is not None:
            _cost["bytes"] += len(raw)
        value = json.loads(raw)
        if value.get("ready") and (
            value.get("head") != frozen["head"]
            or value.get("config_sha256") != frozen["observer_config_sha256"]
            or value.get("sources") != frozen["sources"]
            or value.get("rank") != rank
        ):
            value["failure"] = "worker receipt identity differs from frozen plan"
        snapshots[rank] = value
    pids = [v["pid"] for v in snapshots.values()]
    if len(pids) != len(set(pids)):
        raise RuntimeError("multiple rank receipts from one worker")
    return snapshots


def execute(frozen, plan_sha):
    import psutil

    expected = make_plan(Path(frozen["output"]), frozen.get("mode", "exposure"))
    if frozen != expected or git("status", "--porcelain"):
        raise RuntimeError("source or frozen plan changed")
    if (
        os.environ.get("K31_TASK6_GPU_GRANT") != frozen["head"]
        or os.environ.get("K31_TASK6_PLAN_SHA") != plan_sha
    ):
        raise RuntimeError("no lease for this exact head and plan file")
    if os.environ.get("KFD_GUARD_LOG") or os.nice(0) != 0:
        raise RuntimeError("GPU work cannot inherit the CPU guard or nice19")
    ready = json.loads((PRIVATE / "runtime-ready.json").read_text())
    dataset = json.loads((PRIVATE / "dataset-ready.json").read_text())
    if not ready.get("ready") or not dataset.get("ready"):
        raise RuntimeError("runtime and cache closure is not ready")
    spec = importlib.util.spec_from_file_location("owned_lifecycle", COMMON)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    out = Path(frozen["output"])
    if out.exists():
        raise FileExistsError(out)
    fd = C.GpuLock(LOCK).acquire()
    out.mkdir()
    processes, seen, services = [], {}, []
    state = {
        "state": "STARTING",
        "head": frozen["head"],
        "started": time.time(),
        "errors": [],
        "admitted": False,
    }
    save(out / "plan.json", frozen)
    launched = time.monotonic()
    state["launch_mono"] = launched
    observer = None
    if frozen["mode"] == "warmup_observer":
        from observe import StopState

        observer = StopState(frozen["protocol"]["observer"], launched)
    save_clock = None
    if frozen["mode"] == "save_completion_observer":
        from save_observe import SaveClock

        save_clock = SaveClock(frozen["protocol"]["save_observer"], launched)
    clock = observer or save_clock
    limit = clock.config["workload_s"] if clock else P["workload_timeout_s"]
    deadline = launched + limit

    def track(process=None):
        if process is not None:
            processes.append(process)
            try:
                if process.poll() is None:
                    seen[process.pid] = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                pass
        for parent in processes:
            try:
                if parent.poll() is not None or psutil.Process(
                    parent.pid
                ).create_time() != seen.get(parent.pid):
                    continue
            except psutil.NoSuchProcess:
                continue
            candidates = {parent.pid, *C.descendants(parent.pid)}
            for item in psutil.process_iter(["pid", "create_time"]):
                try:
                    if os.getpgid(item.pid) == parent.pid:
                        candidates.add(item.pid)
                except (ProcessLookupError, PermissionError):
                    pass
            for pid in candidates:
                try:
                    seen[pid] = psutil.Process(pid).create_time()
                except psutil.NoSuchProcess:
                    pass
        live = []
        for pid, created in seen.items():
            try:
                proc = psutil.Process(pid)
                if (
                    proc.create_time() == created
                    and proc.status() != psutil.STATUS_ZOMBIE
                ):
                    live.append(proc)
            except psutil.NoSuchProcess:
                pass
        return live

    def check_deadline():
        track()
        exited = [
            {"pid": p.pid, "exit_code": p.returncode}
            for p in services
            if p.poll() is not None
        ]
        if exited:
            if clock:
                clock.latch("native-failure", time.monotonic(), exited)
            raise RuntimeError("a persistent server exited")
        if observer:
            observer.check(observer_snapshots(out, frozen, observer), time.monotonic())
            if observer.reason:
                raise RuntimeError(f"observer stopped: {observer.reason}")
        if save_clock is not None:
            from save_observe import reports

            failures = {
                name: v["native_failure"]
                for name, v in reports(out, frozen).items()
                if v.get("native_failure")
            }
            if failures:
                save_clock.latch("native-failure", time.monotonic(), failures)
                raise RuntimeError(
                    "native save observation reports an existing failure"
                )
            if time.monotonic() >= save_clock.deadline():
                save_clock.latch(
                    "deadline", time.monotonic(), "absolute client/workload bound"
                )
                raise TimeoutError("save diagnostic absolute bound exceeded")
        if time.monotonic() >= deadline:
            raise TimeoutError("whole workload bound exceeded")

    def interrupt(signum, frame):
        if clock:
            clock.latch(
                (
                    "deadline"
                    if signum == signal.SIGALRM
                    else ("signal" if save_clock else "observer-failure")
                ),
                time.monotonic(),
                f"signal {signum}",
            )
            raise KeyboardInterrupt(f"signal {signum}")
        for proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Cancellation must bypass request handlers that retry OSError.
        raise KeyboardInterrupt(f"signal {signum}")

    if clock:

        def on_stop():
            arm_evidence_deadline(clock)
            if save_clock is not None:
                try:
                    save_control(save_clock, out, frozen)
                except OSError as error:
                    state["save_observer_control_error"] = repr(error)

        clock.on_stop = on_stop
        clock.on_clock_change = lambda: signal.setitimer(
            signal.ITIMER_REAL, max(0.001, clock.deadline() - time.monotonic())
        )
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(
        signal.ITIMER_REAL,
        max(0.001, (clock.deadline() if clock else deadline) - time.monotonic()),
    )
    try:
        if save_clock is not None:
            save_control(save_clock, out, frozen)
        state["before"] = C.assert_gpus_idle()
        state["admitted"] = True
        if PAUSE.exists():
            raise RuntimeError("GPU_PAUSE exists")
        C.assert_ports_free(
            [
                P[k]
                for k in ("port", "cache_port", "cache_http_port", "cache_metrics_port")
            ]
        )
        metadata_env = dict(
            frozen["environments"]["metadata"], K31_TASK6_METADATA_LEASE=frozen["head"]
        )
        metadata = C.spawn(
            [
                "/opt/venv/bin/python",
                str(Path(__file__).resolve()),
                "--capture-metadata",
                str(out / "plan.json"),
            ],
            env=metadata_env,
            cwd=ROOT,
            log_path=out / "metadata.log",
            lock_fd=fd,
            cpus="0-111",
        )
        track(metadata)
        if metadata.wait(timeout=60):
            raise RuntimeError("metadata capture failed; see metadata.log")
        env = frozen["environments"]["cache"]
        cache = C.spawn(
            frozen["cache"],
            env=env,
            cwd=ROOT,
            log_path=out / "cache.log",
            lock_fd=fd,
            cpus="0-111",
        )
        track(cache)
        services.append(cache)
        ready_deadline = (
            launched + observer.config["startup_s"]
            if observer
            else time.monotonic() + P["readiness_timeout_s"]
        )
        while True:
            check_deadline()
            try:
                health = json.loads(
                    request(f"http://127.0.0.1:{P['cache_http_port']}/healthcheck")
                )
                if isinstance(health, dict) and health.get("status") == "healthy":
                    break
            except (OSError, HTTPException, ValueError):
                pass
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("cache readiness bound exceeded")
            time.sleep(0.2)
        server = C.spawn(
            frozen["server"],
            env=frozen["environments"]["server"],
            cwd=ROOT,
            log_path=out / "server.log",
            lock_fd=fd,
            cpus="0-111",
        )
        track(server)
        services.append(server)
        while True:
            check_deadline()
            if server.poll() is not None or cache.poll() is not None:
                raise RuntimeError("a server exited during readiness")
            try:
                models = json.loads(request(f"http://127.0.0.1:{P['port']}/v1/models"))
                if any(m["id"] == P["served_name"] for m in models["data"]):
                    snapshots = observer_snapshots(out, frozen) if observer else {}
                    save_ready = True
                    if save_clock is not None:
                        from save_observe import reports, startup_ready

                        save_ready = startup_ready(reports(out, frozen))
                    if save_ready and (
                        not observer
                        or (
                            len(snapshots) == 8
                            and all(v.get("ready") for v in snapshots.values())
                        )
                    ):
                        if observer and observer.check(snapshots, time.monotonic()):
                            raise RuntimeError("observer stopped at readiness boundary")
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("server readiness bound exceeded")
            time.sleep(1)
        state["ready_at"] = time.time()
        if observer:
            observer.ready = time.monotonic()
            state["observer_ready_mono"] = observer.ready
            signal.setitimer(
                signal.ITIMER_REAL, max(0.001, observer.deadline() - time.monotonic())
            )
        if observer:
            state["observer_clocks"] = observer.anchors()
            state["observer_costs"] = observer.costs.snapshot()
        save(out / "status.json", state, observer.costs if observer else None)
        for label in frozen["invocations"]:
            metrics(out, f"before-{label}", state)
            try:
                run_client(
                    C,
                    frozen,
                    out,
                    label,
                    fd,
                    track,
                    check_deadline,
                    observer,
                    save_clock,
                )
            finally:
                if clock is None or clock.reason is None:
                    metrics(out, f"after-{label}", state)
        state["state"] = (
            "COMPLETED"
            if not observer or observer.reason["reason"] == "full-bracket-completion"
            else "FAILED"
        )
        if save_clock is not None:
            state["state"] = (
                "COMPLETED"
                if save_clock.reason["reason"] == "invocation-end"
                else "FAILED"
            )
    except BaseException as error:
        state["state"] = "FAILED"
        state["errors"].append(repr(error))
        if clock:
            clock.latch("observer-failure", time.monotonic(), type(error).__name__)
        else:
            logger.exception("Bounded workload failed")
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if save_clock is not None:
                try:
                    save_clock.latch(
                        "observer-failure", time.monotonic(), state["errors"]
                    )
                    state.update(
                        mode="save_completion_observer",
                        performance_accepted=False,
                        save_observer_stop=save_clock.reason,
                        save_observer_clocks=save_clock.anchors(),
                    )
                    save_completion_receipt(
                        save_clock, state, time.monotonic() < save_clock.grace_end()
                    )
                    from save_observe import summarize

                    while time.monotonic() < save_clock.grace_end():
                        summary = summarize(out, frozen)
                        if state.get("save_observer_control_error"):
                            summary["join_available"] = False
                            summary["complete_chains"] = False
                            summary["incomplete"].append(
                                "stop-control publication failed"
                            )
                        state["save_observer_summary"] = {
                            "join_available": summary["join_available"],
                            "complete_chains": summary["complete_chains"],
                            "generations": list(summary["generations"]),
                            "incomplete": summary["incomplete"],
                        }
                        time.sleep(
                            max(0, min(0.1, save_clock.grace_end() - time.monotonic()))
                        )
                except EvidenceDeadline:
                    state["evidence_deadline_reached"] = True
                # Evidence failure must still reach owned cleanup.
                except Exception as error:  # noqa: BLE001
                    state["errors"].append(f"save evidence grace: {error!r}")
            if observer:
                try:
                    observer.latch(
                        "observer-failure", time.monotonic(), state["errors"]
                    )
                    state["observer_stop"] = observer.reason
                    state["observer_clocks"] = observer.anchors()
                    state["mode"] = "warmup_observer"
                    state["performance_accepted"] = False
                    process = getattr(observer, "client_process", None)
                    state["observer_client"] = {
                        "spawned": process is not None,
                        "pid": getattr(observer, "client_pid", None),
                        "exit_code_at_stop": getattr(
                            observer, "client_exit_at_stop", None
                        ),
                        "censored": getattr(observer, "client_censored", False),
                        "anchors": observer.anchors(),
                        "ledger": {
                            "available": False,
                            "terminal": None,
                            "reason": "grace unavailable",
                        },
                    }
                    # Keep the first-stop alarm armed across every evidence read/write.
                    if time.monotonic() < observer.grace_end():
                        save(
                            out / "observer/trigger.json",
                            observer.reason,
                            observer.costs,
                        )
                        observer_terminal_receipt(observer, state)
                        snapshots = observer_snapshots(out, frozen)
                        alive = {p.pid for p in track()}
                        for snap in snapshots.values():
                            if snap.get("stack_registered") and snap["pid"] in alive:
                                os.kill(snap["pid"], signal.SIGUSR2)
                        while time.monotonic() < observer.grace_end():
                            process = getattr(observer, "client_process", None)
                            terminal = state.get("observer_client", {})
                            if (
                                process is not None
                                and terminal.get("exit_code") is None
                                and process.poll() is not None
                            ):
                                observer_terminal_receipt(observer, state)
                            time.sleep(
                                max(
                                    0, min(0.1, observer.grace_end() - time.monotonic())
                                )
                            )
                except EvidenceDeadline:
                    state["evidence_deadline_reached"] = True
                except Exception as error:  # noqa: BLE001
                    state["errors"].append(f"evidence grace: {error!r}")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            track()
            for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
                for proc in track():
                    try:
                        proc.send_signal(sig)
                    except psutil.NoSuchProcess:
                        pass
                until = time.monotonic() + seconds
                while track() and time.monotonic() < until:
                    time.sleep(0.5)
            while True:
                if not state.get("admitted"):
                    state["no_owned_gpu_work_started"] = True
                    state["finished"] = time.time()
                    save(
                        out / "status.json", state, observer.costs if observer else None
                    )
                    os.close(fd)
                    break
                state["alive"] = [p.pid for p in track()]
                try:
                    released, state["after"] = C.wait_gpus_released(timeout=10)
                    released = (
                        released
                        and len(state["after"]["cards"]) == 8
                        and not state["alive"]
                    )
                except Exception as error:
                    logger.exception("Release check failed; retaining ownership lock")
                    released = False
                    state["cleanup_error"] = repr(error)
                state["gpu_released"] = released
                state["finished"] = time.time()
                if save_clock is not None:
                    process = getattr(save_clock, "client_process", None)
                    state.setdefault("save_observer_client", {})[
                        "exit_code_after_cleanup"
                    ] = (process.poll() if process is not None else None)
                if observer:
                    process = getattr(observer, "client_process", None)
                    state.setdefault("observer_client", {})[
                        "exit_code_after_cleanup"
                    ] = (process.poll() if process is not None else None)
                    state["observer_clocks"] = observer.anchors()
                    state["observer_costs"] = observer.costs.snapshot()
                try:
                    save(
                        out / "status.json", state, observer.costs if observer else None
                    )
                except OSError as error:
                    logger.exception(
                        "Cleanup receipt write failed; release still required"
                    )
                    state["state"] = "FAILED"
                    state["cleanup_write_error"] = repr(error)
                if released:
                    os.close(fd)
                    break
                logger.error("CLEANUP_FAILED_HOLDING_LOCK")
                time.sleep(5)
    print(
        json.dumps(
            {
                "head": frozen["head"],
                "state": state["state"],
                "gpu_released": state.get("gpu_released"),
                "artifacts": str(out),
            }
        )
    )
    if state["state"] != "COMPLETED":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-plan", type=Path)
    group.add_argument("--capture-metadata", type=Path)
    args = parser.parse_args()
    if args.capture_metadata:
        frozen = json.loads(args.capture_metadata.read_text())
        if os.environ.get("K31_TASK6_METADATA_LEASE") != frozen["head"]:
            raise RuntimeError("metadata device query requires the owned lease")
        capture_metadata(frozen, Path(frozen["output"]))
    else:
        execute(json.loads(args.run_plan.read_text()), sha(args.run_plan))


if __name__ == "__main__":
    main()
