"""Freeze and execute the one natural GSM8K profile under an explicit lease."""

import argparse
import ast
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from verify import HERE, P, bind_results, dependency, first_valid, prepare_data, sha

ROOT = HERE.parents[1]
PRIVATE = Path(P["private_root"])
STOP = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
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
    exec(
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


def make_plan(attempt):
    import re

    if not re.fullmatch(r"(author|independent)-[0-9]{3}", attempt):
        raise ValueError("invalid attempt name")
    allowed = {str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file()}
    if (
        git("status", "--porcelain")
        or set(git("diff", "--name-only", P["atom_base"], "HEAD").splitlines())
        - allowed
    ):
        raise ValueError("commit only the declared spike before freezing")
    receipt_path = PRIVATE / "preparation" / f"{attempt}.json"
    preparation = json.loads(receipt_path.read_text())
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


def finish(owned, resources, fd, out, state):
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig in STOP:
        signal.signal(sig, signal.SIG_IGN)
    try:
        owned.stop()
    except Exception as error:
        state["stop_error"] = repr(error)
    release_end = time.monotonic() + P["bounds"]["release"]
    while True:
        try:
            alive = [proc.pid for proc in owned.live()]
            released, snapshot = resources.wait_gpus_released(
                timeout=min(5, max(0, release_end - time.monotonic()))
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
                os.close(fd)
                return
        except Exception as error:
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


def runtime_check(path):
    plan = load_plan(path, grant=True)
    import atom
    import aiter
    import aiter_worker_limits
    import torch

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
    fd = resources.GpuLock().acquire()

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
        state["before"] = resources.assert_gpus_idle()
        resources.assert_ports_free(P["ports"].values())
        start = time.monotonic()
        whole = start + P["bounds"]["whole"]
        startup = start + P["bounds"]["startup"]
        signal.setitimer(signal.ITIMER_REAL, P["bounds"]["startup"])
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
        if metadata.wait(timeout=max(0.01, startup - time.monotonic())) != 0:
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
        client_end = min(whole, time.monotonic() + P["bounds"]["client"])
        while client.poll() is None:
            owned.live()
            if time.monotonic() >= client_end or any(
                c.poll() is not None for c in owned.roots if c is not client
            ):
                raise RuntimeError("client deadline or service failure")
            time.sleep(0.25)
        state["client_rc"] = client.returncode
        if client.returncode != 0:
            raise RuntimeError("scored client failed; partial evidence retained")
        state["evidence"] = bind_results(out, expected)
        state.update(outcome="COMPLETE", phase="cleanup")
    except BaseException as error:
        state["error"] = repr(error)
        print(f"EXECUTION_INVALID: {error!r}", flush=True)
    finally:
        finish(owned, resources, fd, out, state)
    print(json.dumps(state))
    return 0 if state["outcome"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--runtime-check", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.data:
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
