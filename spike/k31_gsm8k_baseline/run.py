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
    for name, expected in P["startup_module"]["files"].items():
        if sha(name) != expected:
            raise ValueError(f"startup dependency changed: {name}")
        files[name] = expected
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
    extra = P["fmha_opus_preparation"]
    if sha(extra["path"]) != extra["sha256"]:
        raise ValueError("FMHA preparation receipt changed")
    ready = json.loads(Path(extra["path"]).read_text())
    detail_ref = ready["native_receipt"]
    if sha(detail_ref["path"]) != detail_ref["sha256"]:
        raise ValueError("FMHA native receipt changed")
    detail = json.loads(Path(detail_ref["path"]).read_text())
    if (
        detail["source_pins"]
        != {"atom": P["atom_base"], "aiter": P["aiter_sha"], "ck": P["ck_sha"]}
        or not ready["cpu_ready"]
        or detail["operator_executed"]
    ):
        raise ValueError("FMHA source/build qualification mismatch")
    for reference in (
        extra,
        detail_ref,
        detail["input_closure"],
        ready["include_order_validation"],
        ready["initial_strict_validation"],
        ready["build_log"],
        ready["load_validation_log"],
        ready["load_validation_exit"],
    ):
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("FMHA provenance input changed")
        files[reference["path"]] = reference["sha256"]
    build_source = Path(extra["path"]).parents[1] / "source/aiter"
    closure = json.loads(Path(detail["input_closure"]["path"]).read_text())
    for filename, record in closure.items():
        source = Path(filename)
        target = (
            namespace / "aiter" / source.relative_to(build_source)
            if source.is_relative_to(build_source)
            else source
        )
        if (
            record["sha256"] != record["expected"]
            or sha(source) != record["sha256"]
            or sha(target) != record["sha256"]
        ):
            raise ValueError(f"FMHA source/header closure changed: {filename}")
        files[str(target)] = record["sha256"]
    for filename, record in detail["compiler_runtime_binaries"].items():
        if sha(filename) != record["sha256"]:
            raise ValueError("FMHA compiler/runtime changed")
        files[filename] = record["sha256"]
    artifact = ready["artifact"]
    if sha(artifact["path"]) != artifact["sha256"]:
        raise ValueError("FMHA native artifact changed")
    destination = namespace / "cache/aiter/jit/module_fmha_fwd_bf16_opus.so"
    shutil.copyfile(artifact["path"], destination)
    files[str(destination)] = artifact["sha256"]
    provenance[str(destination)] = {
        "source": artifact["path"],
        "sha256": artifact["sha256"],
        "qualification": extra,
    }
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
    startup = data.with_name(f"{attempt}-startup.json")
    subprocess.run(
        [P["python"], str(HERE / "run.py"), "--startup-check", str(startup)],
        cwd=ROOT,
        env=cpu_env,
        check=True,
        timeout=60,
    )
    files[str(startup)] = sha(startup)
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
        "startup_qualification": {"path": str(startup), "sha256": sha(startup)},
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
    for name, expected in P["startup_module"]["files"].items():
        if sha(name) != expected:
            raise ValueError(f"startup dependency changed: {name}")
    for filename, expected in preparation["files"].items():
        if sha(filename) != expected:
            raise ValueError(f"preparation artifact changed: {filename}")
    startup = preparation["startup_qualification"]
    if sha(startup["path"]) != startup["sha256"]:
        raise ValueError("CPU startup qualification changed")
    witness = json.loads(Path(startup["path"]).read_text())
    if (
        not witness["qualified"]
        or witness["target_imports_present"]
        or witness["executable"] != P["python"]
        or not witness["cpu_guard_present"]
        or witness["nice"] != 19
        or not set(witness["cpus"]).issubset(range(112, 128))
        or witness["verifier_sha256"] != sha(HERE / "run.py")
        or witness["protocol_sha256"] != sha(HERE / "protocol.json")
    ):
        raise ValueError("CPU startup qualification is incomplete or stale")
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
        "startup_qualification": startup,
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


def request(url, body=None, *, deadline=None):
    timeout = 10 if deadline is None else deadline - time.monotonic()
    if timeout <= 0:
        raise TimeoutError("HTTP phase deadline exhausted")
    data = None if body is None else json.dumps(body).encode()
    with urlopen(
        Request(url, data=data, headers={"Content-Type": "application/json"}),
        timeout=timeout,
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


def smoke_batch(owned, url, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("smoke deadline exhausted before dispatch")
    bodies = [
        {
            "model": "kimi-k3.1-pro",
            "prompt": P["smoke"]["prompt"],
            "max_tokens": 8,
            "temperature": 0,
        }
        for _ in range(8)
    ]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    futures = []
    try:
        futures = [
            pool.submit(request, url + "/v1/completions", body, deadline=deadline)
            for body in bodies
        ]

        def completed():
            for future in futures:
                if future.done() and future.exception() is not None:
                    future.result()
            return all(future.done() for future in futures)

        wait_until(owned, completed, deadline)
        return [json.loads(future.result()) for future in futures]
    finally:
        # HTTP threads must not delay owned cleanup after a deadline or cancellation.
        pool.shutdown(wait=False, cancel_futures=True)


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
    cancel_alarm()
    for sig in STOP:
        signal.signal(sig, signal.SIG_IGN)
    arm_deadline(time.monotonic() + P["bounds"]["term"] + P["bounds"]["kill"])
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    try:
        owned.stop()
    except BaseException as error:
        logger.exception("Owned termination failed; retaining release responsibility")
        state.update(outcome="EXECUTION_INVALID", stop_error=repr(error))
    finally:
        cancel_alarm()
    release_end = time.monotonic() + P["bounds"]["release"]
    while True:
        durable = False
        now = time.monotonic()
        arm_deadline(release_end if now < release_end else now + 5)
        try:
            alive = [proc.pid for proc in owned.live()]
            released, snapshot = resources.wait_gpus_released(
                timeout=min(5, release_end - now) if now < release_end else 5
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
            save(
                out / "release.json",
                {
                    "head": state.get("head"),
                    "plan_sha256": state.get("plan_sha256"),
                    "cleanup": state["cleanup"],
                    "execution": {
                        key: state[key]
                        for key in (
                            "attempt",
                            "head",
                            "plan_sha256",
                            "outcome",
                            "client_rc",
                            "execution_elapsed_s",
                            "error",
                            "stop_error",
                            "cleanup_error",
                            "registered",
                            "cache_before_file",
                            "evidence",
                            "launch_monotonic",
                            "launch_unix",
                            "phase",
                            "start_unix",
                        )
                        if key in state
                    },
                },
            )
            durable = released
        except BaseException as error:
            logger.exception("Cleanup proof failed; retaining lock")
            state.update(outcome="EXECUTION_INVALID", cleanup_error=repr(error))
        finally:
            cancel_alarm()
        if durable:
            os.close(fd)
            break
        print(
            (
                "QUARANTINED_HOLDING_GPU_LOCK"
                if time.monotonic() >= release_end
                else "RELEASE_PENDING_HOLDING_GPU_LOCK"
            ),
            flush=True,
        )
        time.sleep(5)

    # Release is durable before any non-release filesystem work can block.
    os.sched_setaffinity(0, range(112, 128))
    os.nice(19 - os.nice(0))
    end = (
        state.get("launch_monotonic", time.monotonic())
        + P["bounds"]["whole"]
        + sum(P["bounds"][key] for key in ("term", "kill", "release"))
    )
    finalize_evidence(out, state, end)


def finalize_evidence(out, state, deadline):
    for sig in STOP:
        signal.signal(sig, interrupt)
    arm_deadline(deadline)
    outcome = state.get("execution_outcome", state["outcome"])
    state.update(execution_outcome=outcome, outcome="PENDING_VERIFICATION")
    marker = out / "pending-verification.json"
    sealed = False
    try:
        save(marker, {"head": state.get("head"), "execution_outcome": outcome})
        save(out / "controller.json", state)
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
            if p.is_file()
            and p.name not in ("controller.json", marker.name)
            and p.suffix != ".tmp"
        }
        state["outcome"] = outcome
        save(out / "controller.json", state)
        seal_registry(state, out)
        sealed = True
        marker.unlink()
        directory = os.open(out, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if sealed and not marker.exists():
            state["outcome"] = outcome
            logger.exception("Evidence and registry are durable; marker cleanup failed")
        else:
            state["outcome"] = "PENDING_VERIFICATION"
            logger.exception(
                "Evidence remains pending; recover this same attempt offline"
            )
        raise
    finally:
        cancel_alarm()


def recover_evidence(output):
    handlers = {sig: signal.getsignal(sig) for sig in STOP}
    deadline = time.monotonic() + P["bounds"]["release"]
    for sig in STOP:
        signal.signal(sig, interrupt)
    arm_deadline(deadline)
    try:
        if (
            not os.environ.get("KFD_GUARD_LOG")
            or os.nice(0) != 19
            or not set(os.sched_getaffinity(0)).issubset(range(112, 128))
        ):
            raise ValueError("offline evidence recovery requires the CPU guard")
        out = output.resolve()
        if not out.is_relative_to(PRIVATE / "runs"):
            raise ValueError("recovery must stay inside this task's runs")
        plan = json.loads((out / "plan.json").read_text())
        release = json.loads((out / "release.json").read_text())
        if (
            str(out) != plan["output"]
            or plan["head"] != git("rev-parse", "HEAD")
            or git("status", "--porcelain")
            or plan["protocol"] != P
            or release["head"] != plan["head"]
            or release["plan_sha256"] != sha(out / "plan.json")
            or not release["cleanup"]["released"]
        ):
            raise ValueError("recovery source/plan/release identity mismatch")
        already_sealed = False
        if plan["attempt"].startswith("author-"):
            registry = json.loads((PRIVATE / "attempts.json").read_text())
            entry = next(row for row in registry if row["attempt"] == plan["attempt"])
            if (
                (out / "controller.json").exists()
                and not (out / "pending-verification.json").exists()
                and entry.get("status_sha256") == sha(out / "controller.json")
            ):
                already_sealed = True
        state = (
            json.loads((out / "controller.json").read_text())
            if (out / "controller.json").exists()
            else {}
        )
        state.update(release["execution"], cleanup=release["cleanup"])
        state.pop("execution_outcome", None)
        before = state["cache_before_file"]
        if (
            sha(before["path"]) != before["sha256"]
            or sha(plan["data"]["path"]) != plan["data"]["sha256"]
        ):
            raise ValueError("recovery input identity mismatch")
        state["cache_before"] = json.loads(Path(before["path"]).read_text())
        if state["outcome"] == "COMPLETE":
            if state["client_rc"] != 0:
                raise ValueError("completed execution has a failed client")
            recovered = bind_results(
                out, json.loads(Path(plan["data"]["path"]).read_text())
            )
            if recovered != state["evidence"]:
                raise ValueError("scored evidence changed; preserve this attempt")
        if already_sealed:
            # Retry only the directory sync; the immutable answer/registry bytes stay put.
            directory = os.open(out, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        else:
            finalize_evidence(out, state, deadline)
        return {
            "attempt": state["attempt"],
            "outcome": state["outcome"],
            "gpu_work": False,
        }
    finally:
        cancel_alarm()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


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


def startup_check(receipt_path=None):
    import importlib.util
    import sys

    policy = P["startup_module"]
    hook = sys.modules.get("sitecustomize")
    spec = getattr(hook, "__spec__", None)
    observed = {
        "verifier_sha256": sha(HERE / "run.py"),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "cpu_guard_present": bool(os.environ.get("KFD_GUARD_LOG")),
        "nice": os.nice(0),
        "cpus": sorted(os.sched_getaffinity(0)),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "search_path": sys.path,
        "loaded": hook is not None,
        "file": getattr(hook, "__file__", None),
        "origin": getattr(spec, "origin", None),
        "cached": getattr(hook, "__cached__", None),
        "loader": type(getattr(spec, "loader", None)).__name__,
        "spec_cached": getattr(spec, "cached", None),
        "apport_loaded": "apport_python_hook" in sys.modules,
        "usercustomize_loaded": "usercustomize" in sys.modules,
        "default_excepthook": sys.excepthook is sys.__excepthook__,
        "trace_active": sys.gettrace() is not None,
        "profile_active": sys.getprofile() is not None,
        "target_imports_present": [
            name for name in ("atom", "aiter", "torch") if name in sys.modules
        ],
        "meta_path_modules": [
            getattr(f, "__module__", type(f).__module__) for f in sys.meta_path
        ],
    }
    errors = []
    for field, expected in (
        ("file", policy["origin"]),
        ("origin", policy["origin"]),
        ("cached", policy["cached"]),
        ("loader", "SourceFileLoader"),
    ):
        if observed[field] != expected:
            errors.append(f"unexpected startup {field}")
    for field, key in (("file", "source_sha256"), ("cached", "bytecode_sha256")):
        try:
            observed[field + "_resolved"] = str(Path(observed[field]).resolve())
            observed[key] = sha(observed[field])
            if observed[key] != policy[key]:
                errors.append(f"startup {key} changed")
        except (OSError, TypeError, RuntimeError) as error:
            observed[key + "_error"] = repr(error)
            errors.append(f"startup {field} unreadable")
    if observed.get("file_resolved") != policy["resolved_source"]:
        errors.append("unexpected resolved startup source")
    if (
        observed.get("cached_resolved") != policy["cached"]
        or observed["spec_cached"] != policy["cached"]
    ):
        errors.append("unexpected resolved startup bytecode")
    observed["runtime_files"] = {}
    for name, expected in policy["files"].items():
        try:
            observed["runtime_files"][name] = sha(name)
            if observed["runtime_files"][name] != expected:
                errors.append(f"startup dependency changed: {name}")
        except OSError as error:
            observed["runtime_files"][name] = repr(error)
            errors.append(f"startup dependency unreadable: {name}")
    observed["pth_files"] = sorted(
        str(p) for p in Path(policy["site_packages"]).glob("*.pth")
    )
    if observed["pth_files"] != sorted(
        name for name in policy["files"] if name.endswith(".pth")
    ):
        errors.append("startup .pth inventory changed")
    if set(observed["meta_path_modules"]) - set(policy["meta_path_modules"]):
        errors.append("unqualified import finder")
    if observed["apport_loaded"] or observed["usercustomize_loaded"]:
        errors.append("Apport or user startup hook is active")
    if (
        not observed["default_excepthook"]
        or observed["trace_active"]
        or observed["profile_active"]
    ):
        errors.append("exception/tracing hooks changed")
    if observed["target_imports_present"]:
        errors.append("target runtime imported before startup qualification")
    observed["resolvable"] = {}
    if not errors:
        for name in ("apport_python_hook", "usercustomize"):
            try:
                candidate = importlib.util.find_spec(name)
                observed["resolvable"][name] = (
                    None
                    if candidate is None
                    else {
                        "origin": candidate.origin,
                        "locations": list(candidate.submodule_search_locations or []),
                    }
                )
                if candidate is not None:
                    errors.append(
                        f"unqualified startup dependency is reachable: {name}"
                    )
            except (ImportError, ValueError, AttributeError) as error:
                observed["resolvable"][name] = repr(error)
                errors.append(f"startup resolution failed: {name}")
    observed.update(qualified=not errors, errors=errors)
    # The witness is captured even when an unexpected startup component is refused.
    print("STARTUP_PROVENANCE=" + json.dumps(observed, sort_keys=True), flush=True)
    if receipt_path is not None:
        save(receipt_path, observed)
    if errors:
        raise ValueError("unqualified Python startup: " + "; ".join(errors))
    return observed


def runtime_check(path):
    startup = startup_check()
    plan = load_plan(path, grant=True)
    save(Path(plan["output"]) / "startup-hook.json", startup)
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
        "startup_hook": startup,
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


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def arm_deadline(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("execution deadline exhausted")
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def cancel_alarm():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def execute(path):
    start = time.monotonic()
    handlers = {sig: signal.getsignal(sig) for sig in STOP}
    for sig in STOP:
        signal.signal(sig, interrupt)
    arm_deadline(start + min(P["bounds"]["startup"], P["bounds"]["whole"]))
    try:
        return execute_bound(path, start, time.time())
    finally:
        cancel_alarm()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def execute_bound(path, start, started_unix):
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
        save(out / "cache-before.json", state["cache_before"])
        state["cache_before_file"] = {
            "path": str(out / "cache-before.json"),
            "sha256": sha(out / "cache-before.json"),
        }
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
        replies = smoke_batch(owned, url, startup)
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
            cleanup_end = min(
                whole + sum(P["bounds"][k] for k in ("term", "kill", "release")),
                time.monotonic()
                + sum(P["bounds"][k] for k in ("term", "kill", "release")),
            )
            arm_deadline(cleanup_end)
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
    parser.add_argument("--finalize", type=Path)
    parser.add_argument("--runtime-check", type=Path)
    parser.add_argument("--startup-check", type=Path)
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
    elif args.startup_check:
        startup_check(args.startup_check)
        if (
            not os.environ.get("KFD_GUARD_LOG")
            or os.nice(0) != 19
            or not set(os.sched_getaffinity(0)).issubset(range(112, 128))
        ):
            raise ValueError("CPU startup qualification requires the CPU guard")
    elif args.runtime_check:
        runtime_check(args.runtime_check)
    elif args.finalize:
        print(json.dumps(recover_evidence(args.finalize)))
    elif args.run:
        raise SystemExit(execute(args.run))
    else:
        parser.error(
            "use --data FILE, --freeze ATTEMPT, --run PLAN or --runtime-check PLAN"
        )


if __name__ == "__main__":
    main()
