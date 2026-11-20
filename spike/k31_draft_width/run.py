"""Own one server load per width; measure only completed public HTTP requests."""

import argparse
import asyncio
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from spike.k31_draft_width.client import (
    cache_changes,
    cache_inventory,
    measure,
    save,
    sha256,
    validate_inputs,
    validate_repeats,
)

ROOT = Path(__file__).resolve().parents[2]
UNSET = (
    "SPEC_DECODE_ACCEPTANCE_LENGTH",
    "ATOM_DSV41_BENCHMARK_SYNTHETIC",
    "AITER_AOT_IMPORT",
    "AITER_REBUILD",
    "ATOM_MONO_COMPILE_INDEX",
    "FLYDSL_RUNTIME_RUN_ONLY",
    "CK_DIR",
    "HIP_KITTENS_DIR",
    "OPUS_GEN_CO_DIR",
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
    for key in list(env):
        if key in UNSET or key.startswith(("ATOM_", "AITER_", "FLYDSL_")):
            env.pop(key)
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
        ATOM_MONO_POLL_SLEEP_UNITS="16",
        AITER_MAX_JOBS="16",
        MAX_JOBS="16",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        ATOM_ENABLE_RELAXED_MTP="0",
        ATOM_SPEC_LOSSLESS_SAMPLING="0",
        HSA_ENABLE_IPC_MODE_LEGACY="1",
        MOONCAKE_DISABLE_HIP_DMABUF="1",
        HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        GPU_ARCHS="gfx950",
        ARCH="gfx950",
        AITER_JIT_DIR=selection["native_jit_dir"],
        AITER_CACHE_DIR=selection["native_cache_dir"],
        AITER_META_DIR=selection["aiter_root"],
        AITER_ASM_DIR=str(Path(selection["aiter_root"]) / "hsa"),
        OPUS_GEN_CO_DIR=str(Path(selection["aiter_root"]) / "csrc/opus_gemm/gen_co"),
        HF_HOME=str(cache / "hf"),
        HF_MODULES_CACHE=str(cache / "hf_modules"),
        TIKTOKEN_CACHE_DIR=str(cache / "tiktoken"),
        XDG_CACHE_HOME=str(cache / "xdg"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        FLYDSL_RUNTIME_CACHE_DIR=str(cache / "flydsl"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        TORCHINDUCTOR_COMPILE_THREADS="1",
        HOME=str(cache / "runs" / args.output.name / "home"),
        TMPDIR=str(cache / "tmp"),
    )
    return env


def check_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


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


def process_table():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z":
                result[int(directory.name)] = (
                    fields[19],
                    int(fields[1]),
                    int(fields[2]),
                )
        except (OSError, IndexError, ValueError):
            continue
    return result


def remember(proc, known):
    table = process_table()
    anchors = {
        pid
        for pid, ticks in known.items()
        if ticks is not None and table.get(pid, (None,))[0] == ticks
    }
    candidates = set(anchors)
    # A numeric PGID can be reused once the original group dies. Only an
    # original pid/birth still in that group proves its present ownership.
    if any(table[pid][2] == proc.pid for pid in anchors):
        candidates.update(
            pid for pid, (_, _, pgid) in table.items() if pgid == proc.pid
        )
    while True:
        children = {
            pid for pid, (_, parent, _) in table.items() if parent in candidates
        } - candidates
        if not children:
            break
        candidates.update(children)
    for pid in candidates:
        known[pid] = table[pid][0]
    return sorted(
        pid
        for pid, ticks in known.items()
        if ticks is not None and table.get(pid, (None,))[0] == ticks
    )


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
        except Exception as error:  # noqa: BLE001 - telemetry failures retain ownership
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


def linked_receipt(reference):
    if sha256(reference["path"]) != reference["sha256"]:
        raise RuntimeError("linked receipt changed")
    return json.loads(Path(reference["path"]).read_text())


def verify(args):
    validate_repeats(args.repeats)
    if args.startup_timeout <= 0 or args.run_timeout <= args.startup_timeout:
        raise ValueError("run deadline must exceed the positive startup deadline")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if head != args.head or dirty:
        raise RuntimeError("source is not the reviewed clean commit")
    if (
        sha256(args.inputs) != args.inputs_sha
        or sha256(args.cache_receipt) != args.cache_sha
    ):
        raise RuntimeError("reviewed input/cache receipt changed")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", args.base, "HEAD"], cwd=ROOT, text=True
    ).splitlines()
    if any(not name.startswith("spike/k31_draft_width/") for name in changed):
        raise RuntimeError("spike includes production-source changes")
    pins = json.loads(args.model_pins.read_text())
    if sha256(args.model_pins) != args.model_pins_sha:
        raise RuntimeError("model identity receipt changed")
    for role, location in (("target", args.model), ("draft", args.draft)):
        record = pins[role]
        if (
            Path(location).resolve() != Path(record["path"]).resolve()
            or sha256(Path(location) / "config.json") != record["config_sha256"]
        ):
            raise RuntimeError(f"{role} model path/config differs from its receipt")
    if (
        sha256(Path(args.model) / "model.safetensors.index.json")
        != pins["target"]["index_sha256"]
    ):
        raise RuntimeError("target model index differs from its receipt")
    data = json.loads(args.inputs.read_text())
    validate_inputs(data, args.model)
    if Path(data["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("prepared inputs used another tokenizer source")
    cache = verify_cache(args)
    if sha256(args.assets_receipt) != args.assets_sha:
        raise RuntimeError("assembly asset receipt changed")
    assets = json.loads(args.assets_receipt.read_text())
    native_root = Path(cache["aiter_selection"]["aiter_root"]).resolve()
    expected = {
        "aiter_root": native_root,
        "asm_dir": native_root / "hsa",
        "opus_gen_co_dir": native_root / "csrc/opus_gemm/gen_co",
    }
    if (
        assets.get("ready") is not True
        or any(Path(assets[key]).resolve() != path for key, path in expected.items())
        or not assets["artifacts"]
    ):
        raise RuntimeError("assembly asset selection differs")
    headers = list(assets.get("generated_headers", []))
    if "base_receipt" in assets:
        headers += linked_receipt(assets["base_receipt"]).get("generated_headers", [])
    for header in headers:
        if sha256(header["path"]) != header["sha256"]:
            raise RuntimeError("assembly generated header changed")
    for path, digest in assets["artifacts"].items():
        if sha256(path) != digest:
            raise RuntimeError(f"assembly asset changed: {path}")
    return data, cache


def verify_cache(args):
    cache = json.loads(args.cache_receipt.read_text())
    expected_settings = {
        "max_model_len": 131072,
        "block_size": 64,
        "replay_cap": 16,
        "tp": 8,
        "dcp": 8,
        "poll_sleep_units": 16,
        "debug": False,
        "timeline": False,
        "arms": {
            f"k{k}": {
                "qlen": k + 1,
                "rows": [(k + 1) * c for c in (1, 2, 3, 4)],
                "state_len": k + 3,
                "replay_window": k + 1,
            }
            for k in (3, 4)
        },
    }
    if (
        cache.get("ready") is not True
        or cache["atom_source"] != args.base
        or any(
            cache["settings"].get(key) != value
            for key, value in expected_settings.items()
        )
    ):
        raise RuntimeError("current-source cache preparation is not complete")
    identity, selection = cache["identity"], cache["aiter_selection"]
    if (
        identity["aiter_selection"] != selection
        or identity["settings"] != cache["settings"]
    ):
        raise RuntimeError("cache identity differs from selected settings")
    sources = identity["production_sha256"]
    if not sources:
        raise RuntimeError("cache receipt has no source binding")
    for path, expected in sources.items():
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT / "atom") or sha256(resolved) != expected:
            raise RuntimeError(f"cache production source changed: {path}")
    model_config = identity["model_config"]
    if (
        Path(model_config["path"]).resolve()
        != (Path(args.model) / "config.json").resolve()
        or sha256(model_config["path"]) != model_config["sha256"]
    ):
        raise RuntimeError("cache model identity changed")
    artifacts = cache["artifacts"]
    for k in (3, 4):
        arm = (args.cache / f"k{k}" / "flydsl").resolve()
        if not any(Path(path).resolve().is_relative_to(arm) for path in artifacts):
            raise RuntimeError("selected cache has no prepared arm")
    for path, expected in artifacts.items():
        if (
            not Path(path).resolve().is_relative_to(args.cache.resolve())
            or sha256(path) != expected
        ):
            raise RuntimeError(f"cache artifact changed: {path}")
    native = linked_receipt(identity["native_receipt"])
    if (
        native.get("native_cpu_ready") is not True
        or native["identity"]["selection"] != selection
    ):
        raise RuntimeError("native selection is not prepared")
    source = native["identity"]["source"]
    if Path(source["source_root"]).resolve() != Path(selection["aiter_root"]).resolve():
        raise RuntimeError("native source root differs")
    for root, revision in (
        (source["source_root"], source["aiter_commit"]),
        (source["ck_root"], source["ck_commit"]),
    ):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
        ).strip()
        if head != revision or dirty:
            raise RuntimeError("native source revision changed")
    provenance, bound = native, dict(native["artifacts"])
    if "base_receipt" in native:
        provenance = linked_receipt(native["base_receipt"])
        for key in ("observed_receipt", "rope_fallback_receipt"):
            for module in linked_receipt(native[key]):
                for item in [
                    module["artifact"],
                    module["linked_artifact"],
                    module["ninja"],
                    *module["sources"],
                ]:
                    bound[item["path"]] = item["sha256"]
    build_source = provenance["identity"]["source"]
    if any(build_source[key] != source[key] for key in ("aiter_commit", "ck_commit")):
        raise RuntimeError("reused native source revision differs")
    if not native["artifacts"] or not provenance["compiled_inputs"]:
        raise RuntimeError("native receipt has no artifact/source binding")
    bound.update(provenance["compiled_inputs"])
    bound.update(
        {
            str(Path(build_source["source_root"]) / path): value
            for path, value in build_source["source_files_sha256"].items()
        }
    )
    for path, expected in bound.items():
        if sha256(path) != expected:
            raise RuntimeError(f"native artifact/source changed: {path}")
    return cache


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
            if args.pause.exists():
                raise RuntimeError("GPU pause requested; no next arm")
            output = args.output / f"k{k}"
            output.mkdir()
            env, cmd = environment(args, k, selection), command(args, k)
            for key in (
                "TMPDIR",
                "HOME",
                "FLYDSL_RUNTIME_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "HF_HOME",
                "HF_MODULES_CACHE",
                "TIKTOKEN_CACHE_DIR",
                "XDG_CACHE_HOME",
            ):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            check_port(args.port)
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
                    "HOME",
                    "HIP_VISIBLE_DEVICES",
                    "GPU_ARCHS",
                    "ARCH",
                    "HSA_ENABLE_IPC_MODE_LEGACY",
                    "MOONCAKE_DISABLE_HIP_DMABUF",
                    "OPUS_GEN_CO_DIR",
                )
            }
            save(
                output / "contract.json",
                {
                    "head": args.head,
                    "base": args.base,
                    "model_pins_sha256": sha256(args.model_pins),
                    "assets_receipt_sha256": sha256(args.assets_receipt),
                    "cache_identity": cache["identity_fingerprint"],
                    "removed_environment_prefixes": ["ATOM_", "AITER_", "FLYDSL_"],
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
            cache_roots = [
                Path(env[key])
                for key in (
                    "AITER_JIT_DIR",
                    "FLYDSL_RUNTIME_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            ] + [Path(env["HOME"]) / ".cache"]
            initial_cache = cache_inventory(cache_roots)
            save(
                output / "startup-cache-before.json",
                {
                    "epoch_ns": time.time_ns(),
                    "cache_roots": [str(path) for path in cache_roots],
                    "files": initial_cache,
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

            def watch(server=proc, ledger=known, stopped=stop):
                deadline = time.monotonic() + args.run_timeout
                while not stopped.wait(0.5):
                    remember(server, ledger)
                    if server.poll() is not None or time.monotonic() >= deadline:
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
                        cache_roots=cache_roots,
                        initial_cache=initial_cache,
                    )
                )
            finally:
                stop.set()
                watcher.join(timeout=2)
                cleanup(proc, known, output)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                save(
                    output / "cache-after-cleanup.json",
                    {
                        "epoch_ns": time.time_ns(),
                        "reached_http_ready": (output / "ready.json").exists(),
                        "reached_timed_phase": (
                            output / "measurement/startup.json"
                        ).exists(),
                        "changes": cache_changes(
                            initial_cache, cache_inventory(cache_roots)
                        ),
                    },
                )
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "draft",
        "head",
        "base",
        "inputs-sha",
        "cache-sha",
        "model-pins-sha",
        "assets-sha",
    ):
        parser.add_argument("--" + name, required=True)
    for name in (
        "inputs",
        "cache",
        "cache-receipt",
        "model-pins",
        "assets-receipt",
        "output",
    ):
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
