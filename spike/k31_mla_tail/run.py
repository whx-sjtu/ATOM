"""Freeze a CPU-readable proposal, or run it only under an explicit GPU grant."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from adapter import AITER, DONOR, ROWS

ROOT = Path(__file__).resolve().parents[2]
WS = Path("/share_nfs/whx/work/k31_perf_opt/.opt-team")
PRIVATE = Path("/app/k31-agent-team-private/task-14")
RUNTIME = Path("/opt/venv")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def common():
    path = WS / "harness/common.py"
    spec = importlib.util.spec_from_file_location("tail_harness_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_files(root):
    return {
        str(p): sha(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    }


def runtime_files():
    site = RUNTIME / "lib/python3.12/site-packages"
    files = [
        RUNTIME / "bin/python",
        site / "torch/version.py",
        Path("/opt/rocm/llvm/bin/clang"),
        Path("/opt/rocm/llvm/bin/llvm-readobj"),
    ]
    for name in ("torch", "triton", "flydsl"):
        found = list(site.glob(name + "*.dist-info/METADATA"))
        assert found, f"missing {name} runtime metadata"
        files.extend(found)
    files.extend((site / "flydsl").rglob("*.so"))
    files.extend((site / "flydsl").rglob("*.py"))
    hip = site / "torch/lib/libamdhip64.so"
    files.append(
        (hip if hip.exists() else Path("/opt/rocm/lib/libamdhip64.so")).resolve(
            strict=True
        )
    )
    files.extend((site / "torch/lib").glob("libtorch_hip.so"))
    files.extend((site / "torch/lib").glob("libc10_hip.so"))
    return {str(p): sha(p) for p in sorted(set(files))}


def prepare(args):
    assert not git(
        "status", "--porcelain"
    ), "commit the spike before freezing a proposal"
    assert not args.out.exists(), "fresh output required"
    donor, aiter = PRIVATE / "donor-78e27", PRIVATE / "aiter-572232d"
    cache = PRIVATE / "cache" / args.out.name
    assert not cache.exists(), "fresh private cache required"
    files = tree_files(donor) | tree_files(aiter) | runtime_files() | DEPENDENCIES
    for name in ("adapter.py", "run.py", "README.md"):
        path = Path(__file__).with_name(name)
        files[str(path)] = sha(path)
    for path in (ROOT / "atom").rglob("*.py"):
        files[str(path)] = sha(path)
    files[str(WS / "harness/common.py")] = sha(WS / "harness/common.py")
    environment = {
        "PATH": "/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{ROOT}:{aiter}:{donor}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "ARCH": "gfx950",
        "GPU_ARCHS": "gfx950",
        "FLYDSL_GPU_ARCH": "gfx950",
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MAX_JOBS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "ATOM_MLA_ABSORB_BF16": "1",
        "ATOM_USE_TRITON_MXFP4_BMM": "0",
        "AITER_LOG_LEVEL": "WARNING",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TMPDIR": str(cache / "tmp"),
        **common().cache_env(cache),
    }
    if "LD_LIBRARY_PATH" in os.environ:
        environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    smoke = args.stage == "smoke"
    plan = {
        "head": git("rev-parse", "HEAD"),
        "base": git("merge-base", "HEAD", "origin/k31-perf-opt"),
        "atom": str(ROOT),
        "aiter": str(aiter),
        "aiter_head": AITER,
        "donor": str(donor),
        "donor_head": DONOR,
        "files": files,
        "environment": environment,
        "out": str(args.out.resolve()),
        "cache": str(cache),
        "stage": args.stage,
        "rows": [44] if smoke else list(ROWS),
        "pairs": 1 if smoke else 5,
        "calls": 3 if smoke else 32,
        "rotations": 2 if smoke else 32,
        "workload_s": 240 if smoke else 600,
        "cleanup_term_s": 10,
        "cleanup_kill_s": 30,
        "release_policy": "retain lock until verified; failed observations quarantine",
        "rendezvous": "file://" + str(cache / "tmp/gloo-init"),
        "scope": "deterministic interface fixture; no model, serving, quality or C12 claim",
        "gpu_lock": "/app/agent-team-scratch/g24-gpu.lock",
        "selected_bdfs": list(SELECTED_BDFS),
        "hip_runtime": next(
            p for p in files if Path(p).name.startswith("libamdhip64.so")
        ),
        "lease_required": True,
        "gpu_authorized_by_plan": False,
        "command": [
            "taskset",
            "-c",
            "0-111",
            str(RUNTIME / "bin/python"),
            str(Path(__file__).resolve()),
            "launch",
            "--plan",
            str(args.plan.resolve()),
        ],
    }
    assert not args.plan.exists(), "fresh plan required"
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    write(args.plan, plan)
    print(
        json.dumps(
            {k: plan[k] for k in ("head", "stage", "rows", "command", "scope")},
            indent=2,
        )
    )


def verify(plan):
    assert Path(plan["atom"]).resolve() == ROOT
    assert git("rev-parse", "HEAD") == plan["head"] and not git("status", "--porcelain")
    assert plan["donor_head"] == DONOR and plan["aiter_head"] == AITER
    for name, expected in plan["files"].items():
        assert sha(name) == expected, name


def resources(plan):
    directory = Path(plan["out"]) / "resources"
    directory.mkdir()
    result = []
    for p in sorted(Path(plan["cache"]).rglob("*")):
        if p.suffix not in (".hsaco", ".co", ".cubin"):
            continue
        proc = subprocess.run(
            ["/opt/rocm/llvm/bin/llvm-readobj", "--notes", str(p)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        name = sha(p)
        (directory / (name + ".txt")).write_text(proc.stdout + proc.stderr)
        fields = [
            line.strip()
            for line in proc.stdout.splitlines()
            if any(
                k in line
                for k in (
                    ".name:",
                    ".sgpr_count:",
                    ".vgpr_count:",
                    ".agpr_count:",
                    ".private_segment_fixed_size:",
                    ".group_segment_fixed_size:",
                    ".sgpr_spill_count:",
                    ".vgpr_spill_count:",
                )
            )
        ]
        result.append(
            {
                "path": str(p),
                "sha256": name,
                "readobj_exit": proc.returncode,
                "fields": fields,
            }
        )
    write(directory / "index.json", result)
    return bool(result) and all(r["readobj_exit"] == 0 and r["fields"] for r in result)


DEPENDENCIES = {
    str(
        PRIVATE / "dependencies/owned-run.py"
    ): "a102579f60f77f78faf47c3999cccc7d3b1b46dd7ed68f7ba40cf5fe583558ea",
    str(
        PRIVATE / "dependencies/owned-protocol.json"
    ): "d3345964118cecd1d786824a38220258dad654af12a0040c662d8a0422158ca3",
    str(
        WS / "harness/common.py"
    ): "1479cc118a3d667cc19c570dae34f779c6c7fcac5df8bd061c6537a9fbddd3ab",
}
SELECTED_BDFS = tuple(
    f"0000:{slot}:00.0" for slot in ("05", "15", "65", "75", "85", "95", "e5", "f5")
)


def resource_components():
    for path, expected in DEPENDENCIES.items():
        assert sha(path) == expected, path
    owned_path, protocol_path, _ = DEPENDENCIES
    node = next(
        n
        for n in ast.parse(Path(owned_path).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Owned"
    )
    protocol = json.loads(Path(protocol_path).read_text())
    protocol.update(term_timeout_s=10, kill_timeout_s=30)
    scope = {"os": os, "signal": signal, "time": time, "P": protocol}
    exec(  # noqa: S102 - unchanged Owned class from hash-verified source
        compile(ast.Module(body=[node], type_ignores=[]), owned_path, "exec"), scope
    )
    return common(), scope["Owned"]


def release_snapshot():
    used = {
        bdf: int(Path(f"/sys/bus/pci/devices/{bdf}/mem_info_vram_used").read_text())
        for bdf in SELECTED_BDFS
    }
    # KFD PIDs are global; this sample must drain without restarting SMI.
    kfd = sorted(int(p.name) for p in Path("/sys/class/kfd/kfd/proc").iterdir())
    return {"kfd_pids": kfd, "kfd_scope": "global", "source": "direct_sysfs"}, used


def selected_released(alive, snapshot, used):
    kfd = snapshot["kfd_pids"]
    return (
        not alive
        and set(used) == set(SELECTED_BDFS)
        and all(type(v) is int and 0 <= v < 2 * 1024**3 for v in used.values())
        and type(kfd) is list
        and not kfd
    )


def worker_guard(plan, path):
    assert os.environ.get("K31_TASK14_GPU_GRANT") == plan["head"]
    assert os.environ.get("K31_TASK14_PLAN_SHA") == sha(path)
    assert not os.environ.get("KFD_GUARD_LOG") and not os.environ.get("LD_PRELOAD")
    assert os.nice(0) == 0 and set(os.sched_getaffinity(0)).issubset(range(112))
    fd = int(os.environ["K31_TASK14_GPU_LOCK_FD"])
    assert Path(os.readlink(f"/proc/self/fd/{fd}")).resolve() == Path(plan["gpu_lock"])
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    assert not blocked.intersection((signal.SIGTERM, signal.SIGINT))


def launch(args, plan):
    assert os.environ.get("K31_TASK14_GPU_GRANT") == plan["head"]
    assert os.environ.get("K31_TASK14_PLAN_SHA") == sha(args.plan)
    assert not os.environ.get("KFD_GUARD_LOG") and not os.environ.get("LD_PRELOAD")
    assert os.nice(0) == 0 and set(os.sched_getaffinity(0)).issubset(range(112))
    assert plan["selected_bdfs"] == list(SELECTED_BDFS)
    assert plan["gpu_lock"] == "/app/agent-team-scratch/g24-gpu.lock"
    out = Path(plan["out"])
    assert not out.exists()
    harness, owned_class = resource_components()
    fd = harness.GpuLock(plan["gpu_lock"]).acquire()
    owned, children, stopped = owned_class(harness), [], []
    result = {"head": plan["head"], "complete": False, "phase": "admission"}
    previous = {
        sig: signal.signal(sig, lambda sig, frame: stopped.append(sig))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        out.mkdir(parents=True)
        write(out / "plan.json", plan)
        Path(plan["cache"], "tmp").mkdir(parents=True, exist_ok=False)
        result["admission"] = harness.assert_gpus_idle()
        if stopped:
            raise InterruptedError("cancelled before spawn")
        start = time.monotonic()
        result.update(phase="operator", start_monotonic=start)
        environment = dict(
            plan["environment"],
            K31_TASK14_GPU_GRANT=plan["head"],
            K31_TASK14_PLAN_SHA=sha(args.plan),
            K31_TASK14_GPU_LOCK_FD=str(fd),
        )
        # Pass the lease to every rank directly; multiprocessing.spawn drops arbitrary FDs.
        for rank in range(8):
            child = harness.spawn(
                [
                    str(RUNTIME / "bin/python"),
                    str(Path(__file__).resolve()),
                    "worker",
                    "--plan",
                    str(args.plan.resolve()),
                    "--rank",
                    str(rank),
                ],
                env=environment,
                cwd=ROOT,
                log_path=out / f"rank-{rank}.log",
                lock_fd=fd,
                cpus="0-111",
            )
            children.append(child)
            owned.add(child)
            if stopped:
                break
        while (
            any(p.poll() is None for p in children)
            and not stopped
            and all(p.poll() in (None, 0) for p in children)
            and time.monotonic() < start + plan["workload_s"]
        ):
            owned.live()
            time.sleep(0.05)
        result.update(
            exit_codes=[p.poll() for p in children],
            stop_signals=stopped,
            timed_out=any(p.poll() is None for p in children)
            and not stopped
            and time.monotonic() >= start + plan["workload_s"],
            elapsed_s=time.monotonic() - start,
        )
        if len(children) == 8 and all(p.poll() == 0 for p in children) and not stopped:
            ranks = [json.loads((out / f"rank-{r}.json").read_text()) for r in range(8)]
            assert {r["bdf"] for r in ranks} == set(SELECTED_BDFS)
            assert all(
                len(r["rows"]) == len(plan["rows"])
                and all(v["all_ranks_pass"] for v in r["rows"])
                for r in ranks
            )
            result["complete"] = True
    except BaseException as error:  # noqa: BLE001 - retain cleanup ownership
        result["error"] = repr(error)
    finally:
        result["phase"] = "cleanup"
        try:
            owned.stop()
        except Exception as error:  # noqa: BLE001 - quarantine on failed cleanup
            result["cleanup_stop_error"] = repr(error)
        while True:
            released = False
            try:
                alive = [p.pid for p in owned.live()]
                snapshot, used = release_snapshot()
                released = selected_released(alive, snapshot, used)
                result["final_exit_codes"] = [p.poll() for p in children]
                result["cleanup"] = {
                    "owned_births": owned.seen,
                    "alive": alive,
                    "snapshot": snapshot,
                    "selected_used_bytes": used,
                    "released": released,
                }
                harness.save_json(out / "launch-result.json", result)
            except Exception as error:  # noqa: BLE001 - quarantine
                released = False
                print(
                    json.dumps({"cleanup_error": repr(error), "retaining_lock": True}),
                    flush=True,
                )
            if released:
                os.close(fd)
                break
            print(
                json.dumps({"cleanup": "unverified", "retaining_lock": True}),
                flush=True,
            )
            time.sleep(5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result["resources_recorded"] = resources(plan) if children else False
    write(out / "launch-result.json", result)
    assert result["complete"] and result["resources_recorded"], result
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "check", "verify", "launch", "worker")
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--rank", type=int, choices=range(8))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=("smoke", "measure"), default="smoke")
    args = parser.parse_args()
    if args.mode == "prepare":
        assert args.plan and args.out
        prepare(args)
    elif args.mode == "check":
        from adapter import selfcheck

        print(json.dumps(selfcheck(ROOT, PRIVATE / "donor-78e27"), indent=2))
    else:
        plan = json.loads(args.plan.read_text())
        verify(plan)
        if args.mode == "verify":
            print(
                json.dumps(
                    {"verified": True, "head": plan["head"], "gpu_executed": False}
                )
            )
        elif args.mode == "launch":
            launch(args, plan)
        else:
            worker_guard(plan, args.plan)
            assert args.rank is not None
            from adapter import worker

            worker(args.rank, plan)


if __name__ == "__main__":
    main()
