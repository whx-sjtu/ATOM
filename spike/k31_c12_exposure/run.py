"""Execute only the reviewed, bounded exposure plan; never a candidate run."""

import argparse
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
    client_pid = getattr(observer, "client_pid", None)
    if client_pid is not None:
        try:
            os.killpg(client_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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


def run_client(C, frozen, out, label, fd, track, check_deadline, observer=None):
    directory = out / label
    directory.mkdir()
    argv = frozen["client"] + ["--output-artifact-dir", str(directory)]
    env = frozen["environments"]["client"]
    save(directory / "launch.json", {"argv": argv, "environment": env})
    process = C.spawn(
        argv,
        env=env,
        cwd=ROOT,
        log_path=directory / "console.log",
        lock_fd=fd,
        cpus="0-111",
    )
    track(process)
    if observer:
        observer.client_pid = process.pid
        phase = False
        while process.poll() is None and observer.reason is None:
            for path in (directory / "logs/aiperf.log", directory / "console.log"):
                phase = phase or (
                    path.exists()
                    and bool(PHASE.search(path.read_text(errors="replace")))
                )
            observer.check(observer_snapshots(out, frozen), time.monotonic(), phase)
            if observer.reason is None:
                check_deadline()
                time.sleep(0.1)
        if observer.reason is None:
            observer.latch(
                "observer-failure",
                time.monotonic(),
                "client exited before full bracket",
            )
        # The first-stop callback has already stopped submissions and armed grace.
        save(
            directory / "client-result.json",
            {
                "exit_code": process.poll(),
                "stop": observer.reason,
                "phase_observed": phase,
                "phase_overlap_possible": phase,
                "performance_accepted": False,
                "cancellation_ledger": "partial; preserve raw client artifacts and server request logs",
            },
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


def observer_snapshots(out, frozen):
    snapshots = {}
    for rank in frozen["protocol"]["observer"]["ranks"]:
        path = out / "observer" / f"rank-{rank}" / "state.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
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
    observer = None
    if frozen["mode"] == "warmup_observer":
        from observe import StopState

        observer = StopState(frozen["protocol"]["observer"], launched)
    limit = observer.config["workload_s"] if observer else P["workload_timeout_s"]
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
        if any(p.poll() is not None for p in services):
            raise RuntimeError("a persistent server exited")
        if observer:
            observer.check(observer_snapshots(out, frozen), time.monotonic())
            if observer.reason:
                raise RuntimeError(f"observer stopped: {observer.reason}")
        if time.monotonic() >= deadline:
            raise TimeoutError("whole workload bound exceeded")

    def interrupt(signum, frame):
        if observer:
            observer.latch(
                "deadline" if signum == signal.SIGALRM else "observer-failure",
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

    if observer:
        observer.on_stop = lambda: arm_evidence_deadline(observer)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(
        signal.ITIMER_REAL,
        max(0.001, observer.deadline() - time.monotonic()) if observer else limit,
    )
    try:
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
                    if not observer or (
                        len(snapshots) == 8
                        and all(v.get("ready") for v in snapshots.values())
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
        save(out / "status.json", state)
        for label in frozen["invocations"]:
            metrics(out, f"before-{label}", state)
            try:
                run_client(C, frozen, out, label, fd, track, check_deadline, observer)
            finally:
                if not observer or observer.reason is None:
                    metrics(out, f"after-{label}", state)
        state["state"] = "COMPLETED"
    except BaseException as error:
        if observer:
            observer.latch("observer-failure", time.monotonic(), type(error).__name__)
        else:
            logger.exception("Bounded workload failed")
        state["state"] = "FAILED"
        state["errors"].append(repr(error))
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if observer:
                try:
                    observer.latch(
                        "observer-failure", time.monotonic(), state["errors"]
                    )
                    state["observer_stop"] = observer.reason
                    state["mode"] = "warmup_observer"
                    state["performance_accepted"] = False
                    # Keep the first-stop alarm armed across every evidence read/write.
                    if time.monotonic() < observer.grace_end():
                        save(out / "observer/trigger.json", observer.reason)
                        snapshots = observer_snapshots(out, frozen)
                        alive = {p.pid for p in track()}
                        for snap in snapshots.values():
                            if snap.get("stack_registered") and snap["pid"] in alive:
                                os.kill(snap["pid"], signal.SIGUSR2)
                        while time.monotonic() < observer.grace_end():
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
                    save(out / "status.json", state)
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
                try:
                    save(out / "status.json", state)
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
