"""Own one server load per width; measure only completed public HTTP requests."""

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx
import psutil

from spike.k31_draft_width.client import measure, save, sha256

ROOT = Path(__file__).resolve().parents[2]
UNSET = (
    "SPEC_DECODE_ACCEPTANCE_LENGTH",
    "ATOM_DSV41_BENCHMARK_SYNTHETIC",
    "AITER_AOT_IMPORT",
    "AITER_REBUILD",
)


def command(args, k):
    return [
        args.python,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        args.model,
        "--served-model-name",
        f"draft-width-{args.output.name}-k{k}",
        "--trust-remote-code",
        "-tp",
        "8",
        "--decode-context-parallel-size",
        "8",
        "--method",
        "dspark",
        "--draft-model",
        args.draft,
        "--num-speculative-tokens",
        str(k),
        "--kv_cache_dtype",
        "fp8",
        "--index-cache-dtype",
        "fp4",
        "--index-cache-layout",
        "mfma",
        "--block-size",
        "64",
        "--enable_prefix_caching",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "16384",
        "--max-model-len",
        "131072",
        "--gpu-memory-utilization",
        "0.88",
        "--level",
        "3",
        "--cudagraph-mode",
        "FULL",
        "--cudagraph-capture-sizes",
        "[1,2,3,4]",
        "--host",
        "127.0.0.1",
        "--server-port",
        str(args.port),
    ]


def environment(args, k, selection):
    cache = args.cache / f"k{k}"
    env = dict(os.environ)
    for key in UNSET:
        env.pop(key, None)
    env.update(
        PYTHONPATH=f"{ROOT}:{selection['aiter_root']}",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        AITER_LOG_LEVEL="WARNING",
        AITER_SITUV2_A4W4="1",
        ATOM_MONO_ENABLE="1",
        ATOM_ENABLE_REPLAYSSM="1",
        ATOM_GDN_SSM_DTYPE="fp32",
        ATOM_MLA_ABSORB_BF16="1",
        ATOM_REPLAYSSM_CACHE_LEN="16",
        ATOM_ENABLE_RELAXED_MTP="0",
        ATOM_SPEC_LOSSLESS_SAMPLING="0",
        HSA_ENABLE_IPC_MODE_LEGACY="1",
        MOONCAKE_DISABLE_HIP_DMABUF="1",
        HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        GPU_ARCHS="gfx950",
        ARCH="gfx950",
        AITER_JIT_DIR=selection["native_jit_dir"],
        AITER_CACHE_DIR=selection["native_cache_dir"],
        ATOM_MONO_COMPILE_INDEX=str(cache / "mono_index"),
        FLYDSL_RUNTIME_CACHE_DIR=str(cache / "flydsl"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        TMPDIR=str(cache / "tmp"),
    )
    return env


def vram():
    result = {
        str(p): int(p.read_text())
        for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")
    }
    if len(result) != 8:
        raise RuntimeError("expected eight GPU VRAM counters")
    return result


def birth(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def remember(proc, known):
    candidates = {proc.pid}
    if birth(proc.pid) == known.get(proc.pid):
        try:
            candidates.update(
                p.pid for p in psutil.Process(proc.pid).children(recursive=True)
            )
        except psutil.NoSuchProcess:
            pass
    for item in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(item.pid) == proc.pid:
                candidates.add(item.pid)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in candidates:
        ticks = birth(pid)
        if ticks is not None:
            known.setdefault(pid, ticks)
    return [
        pid for pid, ticks in known.items() if ticks is not None and birth(pid) == ticks
    ]


def cleanup(proc, known, output):
    old = {
        sig: signal.signal(sig, signal.SIG_IGN)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    for sig, budget in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        live = remember(proc, known)
        for pid in live:
            if birth(pid) == known[pid]:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + budget
        while remember(proc, known) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(0.2)
    # A failed release keeps the ownership lock; it never grants another job.
    while True:
        try:
            live, memory = remember(proc, known), vram()
            released = not live and all(n < 2 * 2**30 for n in memory.values())
            result = {
                "released": released,
                "owned": known,
                "remaining": live,
                "vram_bytes": memory,
            }
        except Exception as error:
            released, result = False, {"released": False, "error": str(error)}
        try:
            save(output / "cleanup.json", result)
        except OSError as error:
            print(
                f"cleanup evidence write failed; retaining lock: {error}",
                file=sys.stderr,
            )
            released = False
        if released:
            break
        time.sleep(5)
    for sig, handler in old.items():
        signal.signal(sig, handler)


async def wait_ready(proc, url, timeout):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during startup: {proc.returncode}")
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("model startup deadline exceeded")


def verify(args):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if head != args.head or dirty:
        raise RuntimeError("source is not the reviewed clean commit")
    data = json.loads(args.inputs.read_text())
    if Path(data["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("prepared inputs used another tokenizer source")
    cache = json.loads(args.cache_receipt.read_text())
    if (
        not cache.get("ready")
        or cache["atom_source"] != args.base
        or cache["settings"]
        != {"max_model_len": 131072, "block_size": 64, "replay_cap": 16}
    ):
        raise RuntimeError("current-source cache preparation is not complete")
    for path, expected in cache["artifacts"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"cache artifact changed: {path}")
    return data, cache


def run(args):
    if (
        os.environ.get("K31_SPIKE_GPU_GRANTED") != "1"
        or os.environ.get("KFD_GUARD_LOG")
        or os.getpriority(os.PRIO_PROCESS, 0) != 0
    ):
        raise RuntimeError(
            "GPU lease at normal priority is required; CPU guards cannot launch it"
        )
    inputs, cache = verify(args)
    args.output.mkdir(parents=True, exist_ok=False)
    selection = cache["aiter_selection"]
    fd = os.open(args.lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if args.pause.exists() or any(n >= 2 * 2**30 for n in vram().values()):
            raise RuntimeError("GPU admission refused; no retry")
        for k in (4, 3):
            output = args.output / f"k{k}"
            output.mkdir()
            env, cmd = environment(args, k, selection), command(args, k)
            for key in (
                "TMPDIR",
                "ATOM_MONO_COMPILE_INDEX",
                "FLYDSL_RUNTIME_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            ):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            with socket.socket() as port:
                port.bind(("127.0.0.1", args.port))
            recorded = {
                key: value
                for key, value in env.items()
                if key.startswith(
                    ("ATOM_", "AITER_", "FLYDSL_", "TRITON_", "TORCHINDUCTOR_")
                )
                or key
                in (
                    "PYTHONPATH",
                    "TMPDIR",
                    "HIP_VISIBLE_DEVICES",
                    "GPU_ARCHS",
                    "ARCH",
                    "HSA_ENABLE_IPC_MODE_LEGACY",
                    "MOONCAKE_DISABLE_HIP_DMABUF",
                )
            }
            save(
                output / "contract.json",
                {
                    "head": args.head,
                    "argv": cmd,
                    "environment": recorded,
                    "unset": list(UNSET),
                    "inputs_sha256": sha256(args.inputs),
                    "cache_receipt_sha256": sha256(args.cache_receipt),
                    "repeats": args.repeats,
                    "startup_timeout_s": args.startup_timeout,
                    "run_timeout_s": args.run_timeout,
                },
            )
            with (output / "server.log").open("wb") as log:
                proc = subprocess.Popen(
                    ["taskset", "-c", "0-111", *cmd],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(fd,),
                )
            known = {proc.pid: birth(proc.pid)}
            stop = threading.Event()

            def watch():
                deadline = time.monotonic() + args.run_timeout
                while not stop.wait(0.5):
                    remember(proc, known)
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        os.kill(os.getpid(), signal.SIGTERM)
                        return

            def cancel(signum, frame):
                raise RuntimeError(
                    f"cancelled or server/deadline failure: signal {signum}"
                )

            previous = {
                sig: signal.signal(sig, cancel)
                for sig in (signal.SIGTERM, signal.SIGINT)
            }
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            try:
                url = f"http://127.0.0.1:{args.port}"
                asyncio.run(wait_ready(proc, url, args.startup_timeout))
                memory = vram()
                if any(n < 2 * 2**30 for n in memory.values()):
                    raise RuntimeError(
                        "HTTP ready without expected all-rank model residency"
                    )
                save(
                    output / "ready.json",
                    {"epoch_ns": time.time_ns(), "vram_bytes": memory},
                )
                asyncio.run(
                    measure(
                        url,
                        cmd[cmd.index("--served-model-name") + 1],
                        inputs,
                        k,
                        args.repeats,
                        output / "measurement",
                    )
                )
            finally:
                stop.set()
                watcher.join(timeout=2)
                cleanup(proc, known, output)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "draft", "head", "base"):
        parser.add_argument("--" + name, required=True)
    for name in ("inputs", "cache", "cache-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", default="/opt/venv/bin/python")
    parser.add_argument("--port", type=int, default=8654)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--run-timeout", type=float, default=1500)
    parser.add_argument("--lock", default="/app/agent-team-scratch/g24-gpu.lock")
    parser.add_argument(
        "--pause",
        type=Path,
        default=Path("/share_nfs/whx/work/k31_perf_opt/.opt-team/lead/GPU_PAUSE"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"prepared_only": True, "commands": [command(args, k) for k in (4, 3)]}
            )
        )
        return
    run(args)


if __name__ == "__main__":
    main()
