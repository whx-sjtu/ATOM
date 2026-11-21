"""Freeze and execute the one natural GSM8K profile under an explicit lease."""

import argparse
import ast
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

from verify import (
    HERE,
    P,
    bind_results,
    dependency,
    digest,
    first_valid,
    prepare_data,
    sha,
)

ROOT = HERE.parents[1]
PRIVATE = Path(P["private_root"])
STOP = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
logger = logging.getLogger(__name__)
PAUSE = Path("/share_nfs/whx/work/k31_perf_opt/.opt-team/lead/GPU_PAUSE")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def owned_type():
    record = P["dependencies"]["owned-run.py"]
    protocol = P["dependencies"]["owned-protocol.json"]
    if (
        sha(record["path"]) != record["sha256"]
        or sha(protocol["path"]) != protocol["sha256"]
    ):
        raise ValueError("Owned source/protocol changed")
    # Execute only the pinned class; importing the old controller also imports its probe.
    node = next(
        n
        for n in ast.parse(Path(record["path"]).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Owned"
    )
    scope = {
        "os": os,
        "signal": signal,
        "time": time,
        "P": json.loads(Path(protocol["path"]).read_text()),
    }
    exec(  # noqa: S102 - only the hash-verified Owned class declaration
        compile(ast.Module(body=[node], type_ignores=[]), record["path"], "exec"), scope
    )
    return scope["Owned"]


def argv(program, flags):
    result = list(program)
    for key, value in flags.items():
        result.append(key)
        if value is not None:
            result.append(
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
    return result


def environment(attempt):
    namespace = PRIVATE / "namespaces" / attempt
    env = dict(P["environment"])
    for key, suffix in {
        "HOME": "home",
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "cache/xdg",
        "HF_HOME": "hf",
        "HF_DATASETS_CACHE": "hf/datasets",
        "HF_MODULES_CACHE": "hf/modules",
        "TIKTOKEN_CACHE_DIR": "cache/tiktoken",
        "AITER_JIT_DIR": "cache/aiter/jit",
        "AITER_CACHE_DIR": "cache/aiter",
        "AITER_META_DIR": "aiter",
        "AITER_ASM_DIR": "aiter/hsa",
        "OPUS_GEN_CO_DIR": "aiter/csrc/opus_gemm/gen_co",
        "FLYDSL_RUNTIME_CACHE_DIR": "cache/flydsl",
        "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "AMD_COMGR_CACHE_DIR": "cache/comgr",
        "TORCH_EXTENSIONS_DIR": "cache/extensions",
    }.items():
        env[key] = str(namespace / suffix)
    env["AITER_ASM_DIR"] += "/"
    env["PYTHONPATH"] = f"{ROOT}:{namespace / 'aiter'}"
    return env


def commands(attempt):
    out = PRIVATE / "runs" / attempt
    ports = P["ports"]
    server = argv(
        [P["python"], "-m", "atom.entrypoints.openai_server"],
        {
            **P["server_flags"],
            "--model": P["model"],
            "--draft-model": P["draft"],
            "--server-port": ports["server"],
            "--request-log": str(out / "requests.jsonl"),
            "--kv-transfer-config": {
                "kv_connector": "lmcache_mp",
                "kv_role": "offload",
                "kv_connector_extra_config": {
                    "lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": ports["cache"],
                },
            },
        },
    )
    cache = argv(
        ["/opt/venv/bin/lmcache", "server"],
        {
            **P["cache_flags"],
            "--port": ports["cache"],
            "--http-port": ports["cache_http"],
            "--instance-id": f"task10-{attempt}",
        },
    )
    model_args = (
        f"model=kimi-k3.1-pro,base_url=http://127.0.0.1:{ports['server']}/v1/completions,"
        f"num_concurrent=8,max_retries=3,tokenized_requests=False,tokenizer={P['model']},"
        "trust_remote_code=True,max_gen_toks=256,timeout=300"
    )
    client = [
        "/opt/venv/bin/lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        "gsm8k",
        "--num_fewshot",
        "5",
        "--model_args",
        model_args,
        "--log_samples",
        "--output_path",
        str(out / "quality"),
        "--seed",
        ",".join(map(str, P["seeds"])),
    ]
    return {"server": server, "cache": cache, "client": client}


def archive(source, ref, destination):
    import tarfile

    destination.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        ["git", "-C", source, "archive", ref], stdout=subprocess.PIPE
    ) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as contents:
            contents.extractall(destination, filter="data")
        if process.wait() != 0:
            raise RuntimeError("source archive failed")


def check_cache_toolchain(cache):
    import importlib.metadata
    import re

    site = Path("/opt/venv/lib/python3.12/site-packages")
    toolchain = cache["identity"]["toolchain"]
    key = toolchain["compiler_key"]
    if not key.startswith("flydsl:" + importlib.metadata.version("flydsl") + "-"):
        raise ValueError("cache compiler version changed")
    matches = re.findall(r"(?:^|-)([^:]+\.py):([0-9a-f]{64})(?=-|$)", key)
    if not matches or any(
        sha(site / "flydsl" / name) != expected for name, expected in matches
    ):
        raise ValueError("cache compiler source changed")
    version = ast.parse((site / "torch/version.py").read_text())
    constants = {}
    for node in version.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            constants[target.id] = ast.literal_eval(node.value)
    if (
        constants["__version__"] != toolchain["torch_version"]
        or constants["hip"] != toolchain["torch_hip"]
    ):
        raise ValueError("cache Torch/ROCm identity changed")
    if any(
        P["environment"].get(k, "") != value
        for k, value in toolchain["key_environment"]
    ):
        raise ValueError("cache compiler environment changed")


def prepare(attempt):
    import importlib.metadata
    import re
    import shutil

    if not re.fullmatch(r"(author|independent)-[0-9]{3}", attempt):
        raise ValueError("invalid attempt name")
    namespace = PRIVATE / "namespaces" / attempt
    namespace.mkdir(parents=True, exist_ok=False)
    files, provenance = {}, {}
    for record in P["preparation_inputs"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("preparation input changed")
        files[record["path"]] = record["sha256"]
    cache = json.loads(
        Path(P["preparation_inputs"]["cache_receipt"]["path"]).read_text()
    )
    check_cache_toolchain(cache)
    if cache["atom_source"] != P["atom_base"]:
        raise ValueError("cache source is not the production base")
    for relative, expected in cache["identity"]["production_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"cache/source mismatch: {relative}")
        files[str(ROOT / relative)] = expected
    archive(P["aiter_source"], P["aiter_sha"], namespace / "aiter")
    archive(P["ck_source"], P["ck_sha"], namespace / "aiter/3rdparty/composable_kernel")
    for source, expected in cache["artifacts"].items():
        relative = Path(source).relative_to(Path(P["aiter_source"]).parent)
        if relative.parts[0] == "aiter-572232d":
            relative = Path("aiter", *relative.parts[1:])
        destination = namespace / relative
        if sha(source) != expected:
            raise ValueError(f"cache artifact changed: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        provenance[str(destination)] = {"source": source, "sha256": expected}
    for source in (namespace / "aiter").rglob("*"):
        if source.is_file():
            files[str(source)] = sha(source)
    for destination, record in provenance.items():
        files[destination] = record["sha256"]
    env = environment(attempt)
    for key, value in env.items():
        if key != "PYTHONPATH" and value.startswith(str(namespace)):
            Path(value).mkdir(parents=True, exist_ok=True)
    seed = Path(P["arrow_seed"])
    target = Path(env["HF_DATASETS_CACHE"]) / seed.relative_to(seed.parents[3])
    target.mkdir(parents=True, exist_ok=True)
    for filename, expected in P["arrow_sha256"].items():
        if sha(seed / filename) != expected:
            raise ValueError("Arrow seed changed")
        shutil.copyfile(seed / filename, target / filename)
        files[str(target / filename)] = expected
    pins = json.loads(Path(P["model_pins"]).read_text())
    for root, filename, expected in (
        (P["model"], "config.json", pins["target"]["config_sha256"]),
        (P["model"], "model.safetensors.index.json", pins["target"]["index_sha256"]),
        (P["draft"], "config.json", pins["draft"]["config_sha256"]),
    ):
        if sha(Path(root) / filename) != expected:
            raise ValueError("selected model identity changed")
    selected = {}
    for model in (P["model"], P["draft"]):
        for item in Path(model).iterdir():
            if item.is_file() and (
                item.suffix in (".json", ".py") or item.name == "tiktoken.model"
            ):
                files[str(item)] = selected[str(item)] = sha(item)
    site = Path("/opt/venv/lib/python3.12/site-packages")
    for package in (
        "lm_eval",
        "lmcache",
        "transformers",
        "datasets",
        "torch",
        "flydsl",
    ):
        for item in (site / package).rglob("*"):
            if item.is_file() and item.suffix in (".py", ".yaml", ".so"):
                files[str(item)] = sha(item)
    # The CPU guard is propagated only for preparation, never into a frozen GPU child.
    cpu_env = dict(
        env,
        **{
            k: os.environ[k] for k in ("LD_PRELOAD", "KFD_GUARD_LOG") if k in os.environ
        },
    )
    data = PRIVATE / "preparation" / f"{attempt}-data.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [P["python"], str(HERE / "run.py"), "--data", str(data)],
        cwd=ROOT,
        env=cpu_env,
        check=True,
        timeout=180,
    )
    files[str(data)] = sha(data)
    for item in Path(env["HF_MODULES_CACHE"]).rglob("*.py"):
        files[str(item)] = sha(item)
    receipt = {
        "files": files,
        "data": {"path": str(data), "sha256": sha(data)},
        "cache_provenance": provenance,
        "aiter_ref": P["aiter_sha"],
        "ck_ref": P["ck_sha"],
        "recorded_model_identity": json.loads(Path(P["model_pins"]).read_text()),
        "new_selected_model_hashes": selected,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "lm_eval",
                "torch",
                "transformers",
                "datasets",
                "lmcache",
                "flydsl",
            )
        },
    }
    for key in ("native_receipt", "assets_receipt", "runtime_imports", "consumer_plan"):
        reference = cache["identity"][key]
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("cache provenance receipt changed")
        receipt["files"][reference["path"]] = reference["sha256"]
    for name in receipt["versions"]:
        distribution = importlib.metadata.distribution(name)
        for item in distribution.files or []:
            if str(item).endswith((".dist-info/METADATA", ".dist-info/RECORD")):
                source = distribution.locate_file(item)
                receipt["files"][str(source)] = sha(source)
    for source in (P["python"], "/opt/venv/bin/lm_eval", "/opt/venv/bin/lmcache"):
        receipt["files"][source] = sha(source)
    receipt["cache_compatibility"] = {
        key: cache["identity"]["toolchain"][key]
        for key in (
            "compiler_key_sha256",
            "torch_version",
            "torch_hip",
            "target",
            "polling_cache_tag",
        )
    }
    save(PRIVATE / "preparation" / f"{attempt}.json", receipt)
    return {
        "attempt": attempt,
        "data_sha256": sha(data),
        "verified_seed_files": len(provenance),
    }


def make_plan(attempt):
    import re

    if not re.fullmatch(r"(author|independent)-[0-9]{3}", attempt):
        raise ValueError("invalid attempt name")
    allowed = {
        str((HERE / name).relative_to(ROOT))
        for name in (
            "run.py",
            "verify.py",
            "protocol.json",
            "test_contract.py",
            "README.md",
        )
    }
    if (
        git("status", "--porcelain")
        or set(git("diff", "--name-only", P["atom_base"], "HEAD").splitlines())
        - allowed
    ):
        raise ValueError("commit only the declared spike before freezing")
    receipt_path = PRIVATE / "preparation" / f"{attempt}.json"
    preparation = json.loads(receipt_path.read_text())
    check_cache_toolchain(
        json.loads(Path(P["preparation_inputs"]["cache_receipt"]["path"]).read_text())
    )
    for filename, expected in preparation["files"].items():
        if sha(filename) != expected:
            raise ValueError(f"preparation artifact changed: {filename}")
    for name in ("common.py", "gsm8k_compare.py"):
        dependency(name)
    owned_type()
    env = environment(attempt)
    if "SPEC_DECODE_ACCEPTANCE_LENGTH" in env or any(
        k.startswith(("K31_OBSERVER", "ATOM_PROFILER")) for k in env
    ):
        raise ValueError("non-natural environment")
    dependency("common.py").assert_ports_free(P["ports"].values())
    return {
        "head": git("rev-parse", "HEAD"),
        "base": P["atom_base"],
        "root": str(ROOT),
        "attempt": attempt,
        "output": str(PRIVATE / "runs" / attempt),
        "protocol": P,
        "files": {str(p): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        "preparation": {"path": str(receipt_path), "sha256": sha(receipt_path)},
        "data": preparation["data"],
        "environment": env,
        "commands": commands(attempt),
    }


def load_plan(path, grant=False):
    plan = json.loads(Path(path).read_text())
    if plan != make_plan(plan["attempt"]):
        raise ValueError("frozen source/command/environment changed")
    if grant and (
        os.environ.get("K31_TASK10_GPU_GRANT") != plan["head"]
        or os.environ.get("K31_TASK10_PLAN_SHA") != sha(path)
    ):
        raise ValueError("no root lease for this exact head and plan")
    return plan


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(
        Request(url, data=data, headers={"Content-Type": "application/json"}),
        timeout=10,
    ) as response:
        return response.read().decode()


def wait_until(owned, condition, deadline):
    while True:
        owned.live()
        if time.monotonic() >= deadline:
            raise TimeoutError("phase deadline")
        if any(child.poll() is not None for child in owned.roots):
            raise RuntimeError("owned service exited early")
        if condition():
            return
        time.sleep(0.25)


def launch(owned, resources, command, env, out, name, fd):
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, STOP)
    try:
        child = resources.spawn(
            command,
            env=env,
            cwd=ROOT,
            log_path=out / f"{name}.log",
            lock_fd=fd,
            cpus="0-111",
        )
        owned.add(child)
        return child
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def seal_registry(state, out):
    if state.get("registered"):
        registry_path = PRIVATE / "attempts.json"
        registry = json.loads(registry_path.read_text())
        entry = next(row for row in registry if row["attempt"] == state["attempt"])
        entry["status_sha256"] = sha(out / "controller.json")
        save(registry_path, registry)


def finish(owned, resources, fd, out, state):
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, STOP)
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig in STOP:
        signal.signal(sig, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    try:
        owned.stop()
    except Exception as error:
        logger.exception("Owned termination failed; retaining release responsibility")
        state.update(outcome="EXECUTION_INVALID", stop_error=repr(error))
    release_end = time.monotonic() + P["bounds"]["release"]
    try:
        state["cache_after"] = cache_inventory(state["attempt"])
        state["new_artifacts"] = {
            path: {
                "sha256": value,
                "head": state.get("head"),
                "aiter": P["aiter_sha"],
                "plan_sha256": state.get("plan_sha256"),
            }
            for path, value in state["cache_after"].items()
            if state.get("cache_before", {}).get(path) != value
        }
        state["artifacts"] = {
            str(p): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "controller.json" and p.suffix != ".tmp"
        }
    except Exception as error:
        logger.exception("Artifact inventory failed; continuing owned cleanup")
        state.update(outcome="EXECUTION_INVALID", inventory_error=repr(error))
    while True:
        try:
            alive = [proc.pid for proc in owned.live()]
            released, snapshot = resources.wait_gpus_released(
                timeout=(
                    min(5, release_end - time.monotonic())
                    if time.monotonic() < release_end
                    else 5
                )
            )
            released = bool(
                released
                and len(snapshot["cards"]) == 8
                and not alive
                and all(isinstance(pid, int) for pid in snapshot["kfd_pids"])
                and not set(snapshot["kfd_pids"]).intersection(owned.seen)
            )
            state["cleanup"] = {
                "released": released,
                "snapshot": snapshot,
                "alive": alive,
                "owned_births": owned.seen,
                "utc_unix": time.time(),
            }
            save(out / "controller.json", state)
            if released:
                seal_registry(state, out)
                os.close(fd)
                return
        except Exception as error:
            logger.exception("Cleanup proof failed; retaining lock")
            state.update(outcome="EXECUTION_INVALID", cleanup_error=repr(error))
            print(f"Cleanup evidence/release failed: {error!r}", flush=True)
        print(
            (
                "QUARANTINED_HOLDING_GPU_LOCK"
                if time.monotonic() >= release_end
                else "RELEASE_PENDING_HOLDING_GPU_LOCK"
            ),
            flush=True,
        )
        time.sleep(5)


def parsed_server_args(command):
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    args, extra = parser.parse_known_args(command[3:])
    expected = [
        "--host",
        "127.0.0.1",
        "--server-port",
        str(P["ports"]["server"]),
        "--request-log",
        command[command.index("--request-log") + 1],
    ]
    if extra != expected:
        raise ValueError("unexpected server arguments")
    result = EngineArgs.from_cli_args(args)
    if (
        result.spec_decode_acceptance_length is not None
        or result.spec_decode_acceptance_rate is not None
    ):
        raise ValueError("forced speculative acceptance")
    return result


def resolved_policy(command):
    import dataclasses

    from atom.entrypoints.openai.protocol import CompletionRequest
    from atom.sampling_params import SamplingParams

    engine = parsed_server_args(command)._get_engine_kwargs()
    spec = engine["speculative_config"]
    if (
        spec.synthetic_acceptance_length is not None
        or spec.synthetic_acceptance_rate is not None
    ):
        raise ValueError("forced speculative acceptance")
    request_model = CompletionRequest(
        model="kimi-k3.1-pro",
        prompt="policy receipt",
        max_tokens=256,
        temperature=0,
        stop=["Question:", "</s>", "<|im_end|>"],
    )
    sampling = SamplingParams(
        temperature=request_model.temperature,
        max_tokens=request_model.get_max_tokens(),
        stop_strings=request_model.stop,
        ignore_eos=request_model.ignore_eos,
    )
    return json.loads(
        json.dumps(
            {"engine": engine, "sampling": vars(sampling)},
            default=lambda v: (
                dataclasses.asdict(v) if dataclasses.is_dataclass(v) else str(v)
            ),
        )
    )


def cache_inventory(attempt):
    root = PRIVATE / "namespaces" / attempt
    return {
        str(p): sha(p)
        for directory in (
            root / "cache",
            root / "home",
            root / "hf",
            root / "aiter/hsa",
            root / "aiter/csrc/opus_gemm/gen_co",
        )
        for p in directory.rglob("*")
        if p.is_file()
    }


def process_receipt(owned, plan):
    rows = []
    private = PRIVATE / "namespaces" / plan["attempt"]
    for proc in owned.live():
        env = proc.environ()
        if "SPEC_DECODE_ACCEPTANCE_LENGTH" in env or any(
            env.get(k) != value for k, value in plan["environment"].items()
        ):
            raise ValueError(
                "owned child environment differs from the frozen allowlist"
            )
        modules = sorted(
            {
                m.path
                for m in proc.memory_maps(grouped=False)
                if "/module_" in m.path and m.path.endswith(".so")
            }
        )
        if any(not Path(name).is_relative_to(private) for name in modules):
            raise ValueError("a worker loaded a shared native module")
        rows.append(
            {
                "pid": proc.pid,
                "birth": proc.create_time(),
                "argv": proc.cmdline(),
                "environment": {k: env.get(k) for k in plan["environment"]},
                "native_modules": {name: sha(name) for name in modules},
            }
        )
    return rows


def runtime_check(path):
    import sys

    if "sitecustomize" in sys.modules:
        raise ValueError("an unqualified Python startup hook is active")
    plan = load_plan(path, grant=True)
    import aiter
    import aiter_worker_limits
    import torch

    import atom

    source = Path(plan["environment"]["AITER_META_DIR"])
    modules = {
        m.__name__: str(Path(m.__file__).resolve())
        for m in (atom, aiter, aiter_worker_limits)
    }
    if not Path(modules["atom"]).is_relative_to(ROOT) or any(
        not Path(modules[k]).is_relative_to(source)
        for k in ("aiter", "aiter_worker_limits")
    ):
        raise ValueError("runtime imports escaped the owned source")
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append(
            {k: getattr(prop, k) for k in ("name", "gcnArchName", "total_memory")}
        )
    expected = {k: v for k, v in P["hardware"].items() if k != "count"}
    if len(devices) != 8 or any(d != expected for d in devices):
        raise ValueError("hardware identity changed")
    receipt = {
        "resolved_policy": resolved_policy(plan["commands"]["server"]),
        "imports": modules,
        "devices": devices,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "environment": {k: os.environ.get(k) for k in plan["environment"]},
    }
    if (
        receipt["environment"] != plan["environment"]
        or os.environ.get("SPEC_DECODE_ACCEPTANCE_LENGTH") is not None
    ):
        raise ValueError("actual runtime environment differs")
    save(Path(plan["output"]) / "runtime.json", receipt)


def execute(path):
    start = time.monotonic()
    started_unix = time.time()
    whole = start + P["bounds"]["whole"]
    startup = start + P["bounds"]["startup"]
    plan = load_plan(path, grant=True)
    if (
        os.environ.get("KFD_GUARD_LOG")
        or os.nice(0) != 0
        or not set(os.sched_getaffinity(0)).issubset(range(112))
    ):
        raise ValueError(
            "GPU work requires normal-priority CPU0-111 without the CPU guard"
        )
    if PAUSE.exists():
        raise RuntimeError("GPU work is paused")
    resources = dependency("common.py")
    owned = owned_type()(resources)
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "attempt": plan["attempt"],
        "head": plan["head"],
        "plan_sha256": sha(path),
        "outcome": "EXECUTION_INVALID",
        "phase": "admission",
        "start_unix": time.time(),
    }
    fd = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in STOP:
        signal.signal(sig, interrupt)
    try:
        save(out / "plan.json", plan)
        expected = json.loads(Path(plan["data"]["path"]).read_text())
        if plan["attempt"].startswith("author-"):
            registry_path = PRIVATE / "attempts.json"
            registry = (
                json.loads(registry_path.read_text()) if registry_path.exists() else []
            )
            if first_valid(registry, expected):
                raise ValueError(
                    "canonical A0 already exists; another author score is not permitted"
                )
            if plan["attempt"] != f"author-{len(registry) + 1:03d}":
                raise ValueError("attempt order mismatch")
            registry.append(
                {
                    "attempt": plan["attempt"],
                    "status": str(out / "controller.json"),
                    "plan_sha256": sha(path),
                }
            )
            save(registry_path, registry)
            state["registered"] = True
        if plan["attempt"].startswith("independent-"):
            canonical = first_valid(
                json.loads((PRIVATE / "attempts.json").read_text()), expected
            )
            required = {
                "head": plan["head"],
                "profile_sha256": digest(plan["protocol"]),
                "data_sha256": plan["data"]["sha256"],
            }
            if canonical is None or any(
                canonical["evidence"][key] != value for key, value in required.items()
            ):
                raise ValueError(
                    "independent qualification requires matching A0 source/profile/data"
                )
        state["cache_before"] = cache_inventory(plan["attempt"])
        fd = resources.GpuLock().acquire()
        if PAUSE.exists():
            raise RuntimeError("GPU work paused before admission")
        state["lock_acquired_monotonic"] = time.monotonic()
        state["before"] = resources.assert_gpus_idle()
        resources.assert_ports_free(P["ports"].values())
        state.update(launch_monotonic=start, launch_unix=started_unix)
        if time.monotonic() >= startup:
            raise TimeoutError("startup deadline before child launch")
        signal.setitimer(signal.ITIMER_REAL, startup - time.monotonic())
        env = dict(
            plan["environment"],
            K31_TASK10_GPU_GRANT=plan["head"],
            K31_TASK10_PLAN_SHA=sha(path),
        )
        metadata = launch(
            owned,
            resources,
            [P["python"], str(HERE / "run.py"), "--runtime-check", str(path)],
            env,
            out,
            "runtime",
            fd,
        )
        while metadata.poll() is None:
            owned.live()
            if time.monotonic() >= startup:
                raise TimeoutError("runtime import/hardware deadline")
            time.sleep(0.1)
        if metadata.returncode != 0:
            raise RuntimeError("runtime import/hardware check failed")
        owned.roots.remove(metadata)
        for role in ("cache", "server"):
            launch(owned, resources, plan["commands"][role], env, out, role, fd)
        state["phase"] = "startup"
        url = f"http://127.0.0.1:{P['ports']['server']}"

        def ready():
            try:
                request(f"http://127.0.0.1:{P['ports']['cache_http']}/healthcheck")
                request(url + "/health")
                return True
            except OSError:
                return False

        wait_until(owned, ready, startup)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            bodies = [
                {
                    "model": "kimi-k3.1-pro",
                    "prompt": P["smoke"]["prompt"],
                    "max_tokens": 8,
                    "temperature": 0,
                }
                for _ in range(8)
            ]
            replies = list(
                pool.map(
                    lambda body: json.loads(request(url + "/v1/completions", body)),
                    bodies,
                )
            )
        if any(
            len(r.get("choices", [])) != 1
            or not isinstance(r["choices"][0].get("text"), str)
            for r in replies
        ):
            raise ValueError("smoke schema failed")
        if time.monotonic() >= startup:
            raise TimeoutError("startup/smoke exceeded its bound")
        snapshot = resources.gpu_snapshot()
        if len(snapshot["cards"]) != 8 or any(
            v["used_gib"] < 2 for v in snapshot["cards"].values()
        ):
            raise ValueError("eight-card service residency not established")
        state["processes_at_startup"] = process_receipt(owned, plan)
        state["startup"] = {
            "smoke": replies,
            "requested_concurrency": 8,
            "snapshot": snapshot,
            "server_info": json.loads(request(url + "/server_info")),
            "elapsed_s": time.monotonic() - start,
        }
        save(out / "controller.json", state)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, whole - time.monotonic()))
        state["phase"] = "scoring"
        client = launch(
            owned, resources, plan["commands"]["client"], env, out, "client", fd
        )
        state["client_start_monotonic"] = time.monotonic()
        client_end = min(whole, state["client_start_monotonic"] + P["bounds"]["client"])
        while client.poll() is None:
            owned.live()
            if time.monotonic() >= client_end or any(
                c.poll() is not None for c in owned.roots if c is not client
            ):
                raise RuntimeError("client deadline or service failure")
            time.sleep(0.25)
        state["client_end_monotonic"] = time.monotonic()
        state["client_rc"] = client.returncode
        if any(
            child.poll() is not None for child in owned.roots if child is not client
        ):
            raise RuntimeError("owned service exited before the final receipt")
        if client.returncode != 0:
            raise RuntimeError("scored client failed; partial evidence retained")
        state["evidence"] = bind_results(out, expected)
        state["processes_after_scoring"] = process_receipt(owned, plan)
        state["execution_elapsed_s"] = time.monotonic() - start
        if time.monotonic() >= whole:
            raise TimeoutError("whole execution deadline")
        state.update(outcome="COMPLETE", phase="cleanup")
    except BaseException as error:
        logger.exception("Baseline attempt failed; preserving outputs")
        state["error"] = repr(error)
        print(f"EXECUTION_INVALID: {error!r}", flush=True)
    finally:
        if fd is None:
            state["cleanup"] = {
                "released": True,
                "no_owned_gpu_work_started": True,
                "utc_unix": time.time(),
            }
            save(out / "controller.json", state)
            seal_registry(state, out)
        else:
            finish(owned, resources, fd, out, state)
    print(json.dumps(state))
    return 0 if state["outcome"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare")
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--runtime-check", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.prepare)))
    elif args.data:
        save(args.data, prepare_data())
    elif args.freeze:
        path = PRIVATE / "plans" / f"{args.freeze}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        save(path, make_plan(args.freeze))
        print(json.dumps({"plan": str(path), "sha256": sha(path)}))
    elif args.runtime_check:
        runtime_check(args.runtime_check)
    elif args.run:
        raise SystemExit(execute(args.run))
    else:
        parser.error(
            "use --data FILE, --freeze ATTEMPT, --run PLAN or --runtime-check PLAN"
        )


if __name__ == "__main__":
    main()
