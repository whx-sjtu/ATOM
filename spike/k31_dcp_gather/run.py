"""Freeze or supervise the fixed gather experiment with owned cleanup."""

import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import probe

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIVATE = Path(P["private_root"])
logger = logging.getLogger(__name__)
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def common():
    if sha(P["common"]) != P["common_sha256"]:
        raise RuntimeError("resource helper bytes changed")
    spec = importlib.util.spec_from_file_location("gather_resources", P["common"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    env = {
        "PATH": "/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "PYTHONPATH": f"{ROOT}:{P['aiter_root']}",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "HOME": str(PRIVATE / "home"),
        "TMPDIR": str(PRIVATE / "tmp"),
        "XDG_CACHE_HOME": str(PRIVATE / "cache/xdg"),
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(P["master_port"]),
        "WORLD_SIZE": "8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARCH": "gfx950",
        "GPU_ARCHS": "gfx950",
        "FLYDSL_GPU_ARCH": "gfx950",
        "MAX_JOBS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "AITER_LOG_LEVEL": "WARNING",
        "AITER_META_DIR": P["aiter_root"],
        "AITER_ASM_DIR": str(Path(P["aiter_root"]) / "hsa"),
        "OPUS_GEN_CO_DIR": str(Path(P["aiter_root"]) / "csrc/opus_gemm/gen_co"),
    }
    for key, suffix in {
        "AITER_JIT_DIR": "aiter/jit",
        "AITER_CACHE_DIR": "aiter",
        "FLYDSL_RUNTIME_CACHE_DIR": "flydsl",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "AMD_COMGR_CACHE_DIR": "comgr",
        "TORCH_EXTENSIONS_DIR": "extensions",
    }.items():
        env[key] = str(PRIVATE / "cache" / suffix)
    return env


def make_plan(output):
    if (
        P["world_size"] != 8
        or P["global_keys"] != [393216, 401408]
        or P["repetitions"] != 3
    ):
        raise RuntimeError("matrix differs from the fixed experiment")
    bounds = (
        P["comm_timeout_s"],
        P["pair_timeout_s"],
        P["whole_timeout_s"],
        P["term_timeout_s"],
        P["kill_timeout_s"],
    )
    if bounds != (60, 30, 180, 15, 15):
        raise RuntimeError("execution bounds differ from the fixed experiment")
    if git("status", "--porcelain"):
        raise RuntimeError("commit the experiment before freezing it")
    allowed = {
        f"spike/k31_dcp_gather/{n}"
        for n in ("probe.py", "run.py", "protocol.json", "README.md")
    }
    if set(git("diff", "--name-only", P["atom_base"], "HEAD").splitlines()) - allowed:
        raise RuntimeError("branch changes inference or undeclared files")
    if git("rev-parse", "HEAD", root=P["aiter_root"]) != P["aiter_sha"] or git(
        "status", "--porcelain", "--untracked-files=no", root=P["aiter_root"]
    ):
        raise RuntimeError("AITER source changed")
    for filename, expected in P["source_sha256"].items():
        selected = Path(filename) if Path(filename).is_absolute() else ROOT / filename
        if sha(selected) != expected:
            raise RuntimeError(f"source bytes changed: {filename}")
    if sha(P["native_receipt"]) != P["native_receipt_sha256"]:
        raise RuntimeError("native-cache receipt changed")
    native = json.loads(Path(P["native_receipt"]).read_text())
    for item in native["files"]:
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError(f"native module changed: {item['path']}")
    common()
    output = output.resolve()
    if not output.is_relative_to(PRIVATE / "runs"):
        raise ValueError("output must be inside the private runs directory")
    return {
        "head": git("rev-parse", "HEAD"),
        "root": str(ROOT),
        "output": str(output),
        "protocol": P,
        "environment": environment(),
        "files": {str(p): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        "native_cache": native,
        "command": [
            P["runtime_python"],
            str(HERE / "probe.py"),
            "--worker",
            str(output / "plan.json"),
            "--rank",
            "RANK",
        ],
    }


def load_frozen(path, require_grant=False):
    frozen = json.loads(path.read_text())
    if frozen != make_plan(Path(frozen["output"])):
        raise RuntimeError("plan/source/environment changed")
    if require_grant and (
        os.environ.get("K31_TASK7_GPU_GRANT") != frozen["head"]
        or os.environ.get("K31_TASK7_PLAN_SHA") != sha(path)
    ):
        raise RuntimeError("no lease for this exact head and plan")
    return frozen


class Owned:
    """The reviewed PID/birth tracking pattern, limited to this child set."""

    def __init__(self, resources):
        import psutil

        self.psutil, self.resources = psutil, resources
        self.roots, self.seen = [], {}

    def add(self, child):
        self.roots.append(child)
        self.seen[child.pid] = self.psutil.Process(child.pid).create_time()
        self.live()

    def live(self):
        psutil = self.psutil
        for root in self.roots:
            try:
                if root.poll() is not None or psutil.Process(
                    root.pid
                ).create_time() != self.seen.get(root.pid):
                    continue
                candidates = {root.pid, *self.resources.descendants(root.pid)}
                for proc in psutil.process_iter(["pid", "create_time"]):
                    try:
                        if os.getpgid(proc.pid) == root.pid:
                            candidates.add(proc.pid)
                    except (ProcessLookupError, PermissionError):
                        pass
                for pid in candidates:
                    self.seen[pid] = psutil.Process(pid).create_time()
            except psutil.NoSuchProcess:
                pass
        live = []
        for pid, born in self.seen.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == born and proc.status() != psutil.STATUS_ZOMBIE:
                    live.append(proc)
            except psutil.NoSuchProcess:
                pass
        return live

    def stop(self):
        for sig, seconds in (
            (signal.SIGTERM, P["term_timeout_s"]),
            (signal.SIGKILL, P["kill_timeout_s"]),
        ):
            for proc in self.live():
                try:
                    proc.send_signal(sig)
                except self.psutil.NoSuchProcess:
                    pass
            until = time.monotonic() + seconds
            while self.live() and time.monotonic() < until:
                time.sleep(0.1)


def wait_for(out, owned, predicate, deadline, phase):
    while True:
        owned.live()
        states = [json.loads(p.read_text()) for p in sorted(out.glob("rank-*.json"))]
        errors = [s for s in states if s["phase"] == "error"]
        if errors:
            raise RuntimeError(f"rank failure: {errors[0]['error']}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{phase} deadline exceeded")
        if len(states) == 8 and predicate(states):
            return states
        if any(p.poll() is not None for p in owned.roots):
            raise RuntimeError("rank exited before the phase completed")
        time.sleep(0.01)


def execute(path):
    frozen = load_frozen(path, require_grant=True)
    if (
        os.environ.get("KFD_GUARD_LOG")
        or os.nice(0) != 0
        or not set(os.sched_getaffinity(0)).issubset(range(112))
    ):
        raise RuntimeError("GPU controller must not inherit the CPU guard/nice19")
    C = common()
    out = Path(frozen["output"])
    if out.exists():
        raise FileExistsError(out)
    fd = C.GpuLock().acquire()
    out.mkdir(parents=True)
    (out / "plan.json").write_bytes(path.read_bytes())
    owned = Owned(C)
    state = {"head": frozen["head"], "outcome": "INCONCLUSIVE", "phase": "admission"}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    try:
        state["before"] = C.assert_gpus_idle()
        C.assert_ports_free([P["master_port"]])
        for value in frozen["environment"].values():
            if value.startswith(str(PRIVATE)):
                Path(value).mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        whole = start + P["whole_timeout_s"]
        signal.setitimer(signal.ITIMER_REAL, P["whole_timeout_s"])
        state.update(phase="communicator", child_launch_monotonic=start)
        for rank in range(8):
            env = dict(
                frozen["environment"],
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                K31_TASK7_GPU_GRANT=frozen["head"],
                K31_TASK7_PLAN_SHA=sha(path),
            )
            command = frozen["command"][:-1] + [str(rank)]
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
            try:
                owned.add(
                    C.spawn(
                        command,
                        env=env,
                        cwd=ROOT,
                        log_path=out / f"rank-{rank}.log",
                        lock_fd=fd,
                        cpus="0-111",
                    )
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        wait_for(
            out,
            owned,
            lambda rows: all(r["phase"] == "ready" for r in rows),
            start + P["comm_timeout_s"],
            "communicator",
        )
        for index in range(6):
            state.update(phase="pair", pair=index)
            save(out / "controller.json", state)
            deadline = min(whole, time.monotonic() + P["pair_timeout_s"])
            (out / f"go-{index}").touch()
            wait_for(
                out,
                owned,
                lambda rows, index=index: all(
                    r["phase"] == "complete" and r["pair"] == index for r in rows
                ),
                deadline,
                f"pair {index}",
            )
        state["phase"] = "teardown"
        (out / "finish").touch()
        state["ranks"] = wait_for(
            out,
            owned,
            lambda rows: all(r["phase"] == "finished" for r in rows),
            whole,
            "teardown",
        )
        for child in owned.roots:
            if child.wait(timeout=max(0.01, whole - time.monotonic())) != 0:
                raise RuntimeError("rank failed after verification")
        if time.monotonic() >= whole:
            raise TimeoutError("whole execution deadline exceeded")
        state.update(
            outcome="PASS", phase="complete", elapsed_s=time.monotonic() - start
        )
    except BaseException as error:
        logger.exception("Gather execution failed")
        state["first_error"] = repr(error)
        if state["phase"] == "pair":
            state["outcome"] = "FAIL"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        try:
            (out / "abort").touch()
        except OSError as error:
            logger.exception("Abort marker failed; continuing owned cleanup")
            state["abort_error"] = repr(error)
        try:
            owned.stop()
        except Exception as error:
            logger.exception("Owned termination failed")
            state["cleanup_stop_error"] = repr(error)
        while True:
            try:
                alive = [p.pid for p in owned.live()]
                if owned.roots:
                    released, snapshot = C.wait_gpus_released(timeout=10)
                    kfd = snapshot["kfd_pids"]
                    released = (
                        released
                        and len(snapshot["cards"]) == 8
                        and not alive
                        and all(isinstance(pid, int) for pid in kfd)
                        and not set(kfd).intersection(owned.seen)
                    )
                    state["cleanup"] = {
                        "owned_births": owned.seen,
                        "alive": alive,
                        "snapshot": snapshot,
                        "released": released,
                    }
                else:
                    released = True
                    state["cleanup"] = {
                        "released": True,
                        "no_owned_gpu_work_started": True,
                    }
            except Exception as error:
                logger.exception("Release check failed; retaining lock")
                released = False
                state["cleanup_error"] = repr(error)
            try:
                save(out / "controller.json", state)
            except OSError as error:
                released = False
                print(f"Cleanup receipt failed: {error}", flush=True)
            if released:
                os.close(fd)
                break
            print("CLEANUP_FAILED_HOLDING_LOCK", flush=True)
            time.sleep(5)
    print(json.dumps({"outcome": state["outcome"], "artifacts": str(out)}))
    return 0 if state["outcome"] == "PASS" else 1


def self_check():
    import tempfile

    results = probe.self_check()
    C = common()
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        owned = Owned(C)
        for rank in range(8):
            save(out / f"rank-{rank}.json", {"phase": "ready", "rank": rank})
        wait_for(
            out,
            owned,
            lambda rows: all(r["phase"] == "ready" for r in rows),
            time.monotonic() + 1,
            "ready",
        )
        try:
            wait_for(out, owned, lambda rows: True, time.monotonic() - 1, "pair")
        except TimeoutError:
            results["parent_deadline_rejected"] = True
        else:
            raise AssertionError("expired deadline accepted")
        save(out / "rank-0.json", {"phase": "error", "error": "injected mismatch"})
        try:
            wait_for(out, owned, lambda rows: True, time.monotonic() + 1, "pair")
        except RuntimeError:
            results["rank_error_rejected"] = True
        else:
            raise AssertionError("rank error accepted")
        child = C.spawn(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=os.environ.copy(),
            cwd=out,
            log_path=out / "child.log",
            cpus="112-127",
        )
        try:
            owned.add(child)
            assert any(p.pid == child.pid for p in owned.live())
        finally:
            owned.stop()
        assert child.wait(timeout=1) != 0 and not owned.live()
        results["owned_cpu_child_released"] = True
    assert not any(name in sys.modules for name in ("torch", "aiter", "atom"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    elif args.plan and args.output:
        save(args.plan, make_plan(args.output))
        print(json.dumps({"plan": str(args.plan), "sha256": sha(args.plan)}))
    elif args.run:
        raise SystemExit(execute(args.run))
    else:
        parser.error("use --self-check, --plan FILE --output DIR, or --run FILE")


if __name__ == "__main__":
    main()
