"""Bounded actual-input capture/replay; prospective plans authorize no GPU work."""

import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import capture

ROOT, PRIVATE = capture.ROOT, capture.PRIVATE
WS = Path("/share_nfs/whx/work/k31_perf_opt/.opt-team")
RECIPE = (
    WS / "agent-team/evidence/task-6-save-completion-req9-0d914c12/frozen-plan.json"
)
sha = capture.sha
LEGACY = TAIL = None


def helpers():
    global LEGACY, TAIL
    if LEGACY is not None:
        return
    directory = PRIVATE / "deps/task14"
    for name, digest in {
        "adapter.py": "880154e1fb5a2134fcface30f0c499f5e5abeb8bcb636b9947a29b7e7fb5b560",
        "run.py": "018bfc4a01797f937bc3f03d9b82ca154c344dd3641df2ff789e766877d9f913",
    }.items():
        assert sha(directory / name) == digest
    sys.path.insert(0, str(directory))
    import adapter as legacy_adapter

    spec = importlib.util.spec_from_file_location(
        "_mla19_lifecycle", directory / "run.py"
    )
    LEGACY = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LEGACY)
    TAIL = legacy_adapter
    LEGACY.ROOT = ROOT
    LEGACY.DEPENDENCIES = {
        p.replace("/task-14/dependencies/", "/task-19/deps/owned/"): h
        for p, h in LEGACY.DEPENDENCIES.items()
    }


def binding_functions():
    path = PRIVATE / "deps/task15/replay.py"
    assert (
        sha(path) == "e2553d3b76ad4146da2b3742109d3ac1cab05971db98041746e1301a89b9347e"
    )
    tree = ast.parse(path.read_text())
    names = {"file_identity", "execution_binding", "verify_binding"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Constant)
                and item.value == "agent-team/work/15/cache-seed.json"
            ):
                item.value = "agent-team/work/19/cache-seed.json"
    scope = dict(globals())
    exec(  # noqa: S102 - pinned host-only identity functions
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope
    )
    return scope


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_plan(plan, path, worker=False):
    helpers()
    assert plan["head"] == LEGACY.git("rev-parse", "HEAD") and not LEGACY.git(
        "status", "--porcelain"
    )
    assert plan["mode"] in ("capture", "replay") and plan["rows"] == [capture.ROWS]
    assert plan["max_job_seconds"] == (900 if plan["mode"] == "capture" else 180)
    assert Path(plan["output"]).resolve().is_relative_to(PRIVATE / "runs")
    assert plan["selected_bdfs"] == list(LEGACY.SELECTED_BDFS)
    assert plan["protocol"] == {
        "pairs": 5,
        "aa_pairs": 5,
        "calls": 128,
        "input_sets": 1,
        "noise_fraction": 0.02,
        "reproduction": "fixed_author_observed_min_max",
    }
    binding_functions()["verify_binding"](plan["execution"])
    for name, expected in plan["sources"].items():
        assert sha(name) == expected, name
    if worker:
        assert all(os.environ.get(k) == v for k, v in plan["environment"].items())
        assert os.environ.get("K31_TASK19_GPU_GRANT") == plan["head"]
        assert os.environ.get("K31_TASK19_PLAN_SHA") == sha(path)
        assert not os.environ.get("KFD_GUARD_LOG") and not os.environ.get("LD_PRELOAD")
        assert os.nice(0) == 0 and set(os.sched_getaffinity(0)).issubset(range(112))
        fd = os.environ["K31_TASK19_GPU_LOCK_FD"]
        assert Path(os.readlink("/proc/self/fd/" + fd)).resolve() == Path(
            "/app/agent-team-scratch/g24-gpu.lock"
        )
        assert not signal.pthread_sigmask(signal.SIG_BLOCK, []).intersection(
            (signal.SIGTERM, signal.SIGINT)
        )


def make_plan(mode, output, path, snapshot=None):
    helpers()
    assert not LEGACY.git("status", "--porcelain") and not output.exists()
    base = json.loads(RECIPE.read_text())
    env = {
        k: v.replace("/app/k31-agent-team-private/task-6", str(PRIVATE)).replace(
            "/app/k31-opt-worktrees/agent-team-ATOM/6", str(ROOT)
        )
        for k, v in base["environment"].items()
        if not k.startswith("K31_")
    }
    env.update(
        PYTHONPATH=f"{Path(__file__).parent}:{ROOT}:{PRIVATE/'aiter-572232d'}:{PRIVATE/'donor-78e27'}",
        CX_FRONT_FLOOR_EXPECT_ATOM_ROOT=str(ROOT),
        ATOM_MONO_COMPILE_INDEX=str(PRIVATE / "cache/mono-index"),
        HF_HOME=str(PRIVATE / "home/hf-home"),
        HF_MODULES_CACHE=str(PRIVATE / "inputs/hf-modules"),
        TIKTOKEN_CACHE_DIR=str(PRIVATE / "inputs/tiktoken"),
        K31_MLA_TAIL_REAL_PLAN=str(path.resolve()),
    )
    argv = list(base["server"][3:])
    for flag in (
        "--served-model-name",
        "--host",
        "--server-port",
        "--torch-profiler-dir",
    ):
        index = argv.index(flag)
        del argv[index : index + 2]
    index = argv.index("--kv-transfer-config") + 1
    connector = json.loads(argv[index])
    connector["kv_connector_extra_config"]["lmcache.mp.port"] = 5599
    argv[index] = json.dumps(connector)
    argv += ["--port", "8019"]
    cache = [
        v.replace("5586", "5599").replace("8686", "8699").replace("9186", "9199")
        for v in base["cache"]
    ]
    cache[cache.index("--instance-id") + 1] = output.name
    binding = binding_functions()
    execution = binding["execution_binding"](base)
    for directory in (PRIVATE / "inputs", PRIVATE / "donor-78e27", PRIVATE / "deps"):
        execution["files"].update(
            {
                str(p): binding["file_identity"](p)
                for p in directory.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }
        )
    sources = {
        str(p): sha(p)
        for p in [
            *(ROOT / "atom").rglob("*.py"),
            *Path(__file__).parent.glob("*"),
            RECIPE,
            WS / "harness/common.py",
        ]
        if p.is_file()
    }
    prompt = "Explain in plain language why the sky looks blue during the day."
    plan = {
        "head": LEGACY.git("rev-parse", "HEAD"),
        "root": str(ROOT),
        "mode": mode,
        "output": str(output.resolve()),
        "environment": env,
        "model_args": argv,
        "cache": cache,
        "ports": [8019, 5599, 8699, 9199],
        "selected_bdfs": list(LEGACY.SELECTED_BDFS),
        "rows": [capture.ROWS],
        "max_job_seconds": 900 if mode == "capture" else 180,
        "selection": {
            "layer": 3,
            "requests": 1,
            "qlen": 4,
            "rows": 4,
            "ordinary_witness_ordinal": 1,
            "capture_ordinal": 2,
            "workspace_capacity_and_tile": "must be recorded and matched, not inferred from rows",
        },
        "request": {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "min_position": 1,
            "expected_cached_tokens": 0,
            "max_tokens": 16,
            "temperature": 0,
            "ignore_eos": True,
        },
        "treatment": "Fixed current Mono1/TP8/DCP8/K3/AL2.88 recipe; one actual request, not natural-quality/C12 evidence",
        "cache_policy": "Private compatible native/FlyDSL/Inductor/Triton bytes; higher-level torch graph indexes with original absolute paths are retained inactive; fresh graph compilation stays private.",
        "protocol": {
            "pairs": 5,
            "aa_pairs": 5,
            "calls": 128,
            "input_sets": 1,
            "noise_fraction": 0.02,
            "reproduction": "fixed_author_observed_min_max",
        },
        "sources": sources,
        "execution": execution,
        "hip_runtime": next(
            name
            for name in LEGACY.runtime_files()
            if Path(name).name.startswith("libamdhip64.so")
        ),
        "command": [
            "taskset",
            "-c",
            "0-111",
            "/opt/venv/bin/python",
            str(Path(__file__).resolve()),
            "--plan",
            str(path.resolve()),
        ],
        "gpu_authorized_by_plan": False,
    }
    if mode == "replay":
        if snapshot is None:
            raise ValueError("MISSING_NATIVE_INPUT: complete all-rank capture required")
        capture.initialize_snapshot()
        manifest = json.loads((snapshot / "manifest.json").read_text())
        assert manifest["complete"] and len(manifest["ranks"]) == 8
        producer = json.loads((snapshot.parent / "controller.json").read_text())
        assert producer["complete"] and producer["cleanup"]["released"]
        assert producer["mode"] == "capture" and producer["validated_results"]
        assert (
            producer["cleanup"]["alive"] == []
            and producer["cleanup"]["snapshot"]["kfd_pids"] == []
        )
        for rank in range(8):
            capture.SNAP.load_rank_snapshot(snapshot / f"rank{rank}.pt")
        plan["snapshot"] = str(snapshot.resolve())
        plan["sources"][str(snapshot / "manifest.json")] = sha(
            snapshot / "manifest.json"
        )
        plan["sources"][str(snapshot.parent / "controller.json")] = sha(
            snapshot.parent / "controller.json"
        )
        plan["sources"].update(
            {
                str(snapshot / f"rank{r}.pt"): sha(snapshot / f"rank{r}.pt")
                for r in range(8)
            }
        )
    return plan


def measure(graphs, stream):
    import torch
    import torch.distributed as dist

    result = []
    for pair in range(5):
        row = {}
        for name in (("native", "donor") if pair % 2 == 0 else ("donor", "native")):
            graphs[name].replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record(stream)
            graphs[name].replay()
            end.record(stream)
            torch.cuda.synchronize()
            value = torch.tensor(
                [start.elapsed_time(end) * 1000 / 128], dtype=torch.float64
            )
            values = [torch.empty_like(value) for _ in range(8)]
            dist.all_gather(values, value)
            times = [v.item() for v in values]
            assert all(
                0 < value < float("inf") for value in times
            ), "invalid GPU timing"
            row[name] = {
                "ranks_us": times,
                "max_us": max(times),
                "skew_us": max(times) - min(times),
            }
        result.append(row)
    return result


def device_identity(rank, plan):
    import torch

    assert torch.cuda.device_count() == 8
    assert torch.cuda.get_device_properties(rank).gcnArchName.split(":")[0] == "gfx950"
    loaded = {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libamdhip64.so" in line
    }
    assert loaded == {plan["hip_runtime"]}, loaded
    hip = ctypes.CDLL(plan["hip_runtime"])
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    buffer = ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(buffer, 32, rank) == 0
    bdf = buffer.value.decode().lower()
    assert bdf in plan["selected_bdfs"]
    return {
        "rank": rank,
        "bdf": bdf,
        "hip_runtime": plan["hip_runtime"],
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()[19],
    }


def compare_global(got, expected):
    import torch
    import torch.distributed as dist

    local = TAIL.compare(got, expected)
    local.update(
        error_max=(
            float((got.float() - expected.float()).abs().max())
            if local["finite"]
            else None
        ),
        reference_max=float(expected.float().abs().max()) if local["finite"] else None,
        equal=int((got.view(torch.int16) == expected.view(torch.int16)).sum()),
        count=got.numel(),
    )
    ranks = [None] * 8
    dist.all_gather_object(ranks, local)
    finite = all(v["finite"] for v in ranks)
    relative = (
        max(v["error_max"] for v in ranks)
        / max(max(v["reference_max"] for v in ranks), 1e-30)
        if finite
        else None
    )
    equal = sum(v["equal"] for v in ranks) / sum(v["count"] for v in ranks)
    return {
        "ranks": ranks,
        "finite": finite,
        "global_relative_max": relative,
        "bf16_equal": equal,
        "passed": finite and relative <= 2**-7 and equal >= 0.999,
    }


def empty_controls(item, graphs):
    import torch
    import torch.distributed as dist

    saved_owned, saved_raw = item["owned"].clone(), item["x"].clone()
    records = []
    for case in ("last_row", "all_rows"):
        item["owned"].copy_(saved_owned)
        item["x"].copy_(saved_raw)
        rows = slice(-1, None) if case == "last_row" else slice(None)
        item["owned"][rows] = 0
        item["x"][rows] = float("nan")
        for graph in graphs.values():
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
        passed = all(
            bool((output[rows] == 0).all())
            for output in (item["out"], item["native_out"])
        )
        records.append(
            {"case": case, "both_arms_exact_empty_zero": passed, "passed": passed}
        )
    item["owned"].copy_(saved_owned)
    item["x"].copy_(saved_raw)
    return records


def rank_worker(rank, plan):
    from datetime import timedelta
    from types import SimpleNamespace

    import aiter
    import amd_kernels
    import torch
    import torch.distributed as dist

    import atom

    helpers()
    capture.initialize_snapshot()
    snapshot = capture.SNAP.load_rank_snapshot(
        Path(plan["snapshot"]) / f"rank{rank}.pt"
    )
    torch.cuda.set_device(rank)
    assert (
        torch.cuda.device_count() == 8
        and torch.cuda.get_device_properties(rank).gcnArchName.split(":")[0] == "gfx950"
    )
    hardware = device_identity(rank, plan)
    for module, root in (
        (atom, ROOT),
        (aiter, PRIVATE / "aiter-572232d"),
        (amd_kernels, PRIVATE / "donor-78e27"),
    ):
        assert Path(module.__file__).resolve().is_relative_to(root)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(plan["output"]) / "rendezvous"),
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=90),
    )
    group = SimpleNamespace(
        world_size=8,
        rank_in_group=rank,
        cpu_group=dist.group.WORLD,
        device_group=dist.new_group(backend="nccl"),
    )
    device, stream = torch.device("cuda", rank), torch.cuda.Stream()
    with torch.cuda.stream(stream):
        load_start, load_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        load_start.record(stream)
        tensors = capture.SNAP.restore_tensors(snapshot, device)
        load_end.record(stream)
        torch.cuda.synchronize()
        input_load_us = load_start.elapsed_time(load_end) * 1000
        tail = TAIL.Tail(
            group, rank, device, snapshot["meta"]["key"]["workspace_capacity"]
        )
        from aiter.ops.flydsl.dcp_mailbox.common import token_tile

        assert token_tile(capture.ROWS) == snapshot["meta"]["key"]["token_tile"]
        workspace = capture.workspace_info(tail.ops._MAILBOX_WORKSPACES[tail.key])
        assert {
            k: v for k, v in workspace.items() if k not in ("base", "state_base")
        } == {
            k: v
            for k, v in snapshot["meta"]["workspace"].items()
            if k not in ("base", "state_base")
        }
        item = {
            "x": tensors["raw"],
            "lse": tensors["lse"],
            "weight": tensors["w_v"],
            "gate": tensors["gate"],
            "owned": tensors["owned"],
            "qsl": torch.arange(5, dtype=torch.int32, device=device),
            "out": torch.empty(4, 12, 128, dtype=torch.bfloat16, device=device),
        }
        graphs, proof = tail.capture([item], stream, 128)
        for graph in graphs.values():
            graph.replay()
            torch.cuda.synchronize()
        checks = {
            "native_reference": compare_global(
                item["native_out"], tensors["reference"]
            ),
            "donor_reference": compare_global(
                item["out"].view(4, 1536), tensors["reference"]
            ),
        }
        controls = tail.check([item], graphs) + empty_controls(item, graphs)
        passed = all(v["passed"] for v in checks.values()) and all(
            v["passed"] for v in controls
        )
        statuses = [None] * 8
        dist.all_gather_object(statuses, passed)
        result = {
            "rank": rank,
            "hardware": hardware,
            "workspace": workspace,
            "input_sha256": sha(Path(plan["snapshot"]) / f"rank{rank}.pt"),
            "checks": checks,
            "diagnostic_controls": controls,
            "graph_proof": proof,
            "capture_key": snapshot["meta"]["key"],
            "all_ranks_pass": all(statuses),
            "packing": "original stage arenas/strides restored once; no weight/gate repack; input loading and diagnostic controls excluded from timing",
            "copy_bytes": sum(v.numel() for v in snapshot["arenas"].values()),
        }
        result["input_materialization_gpu_span_us"] = input_load_us
        if all(statuses):
            result["aa_pairs"] = measure(
                {"native": graphs["native"], "donor": graphs["native"]}, stream
            )
            result["pairs"] = measure(graphs, stream)
            aa = [
                p["native"]["max_us"] / p["donor"]["max_us"] for p in result["aa_pairs"]
            ]
            ratios = [
                p["native"]["max_us"] / p["donor"]["max_us"] for p in result["pairs"]
            ]
            stable = all(abs(v - 1) <= 0.02 for v in aa)
            result["median_paired_ratio"] = statistics.median(ratios)
            result["verdict"] = (
                (
                    "advance"
                    if min(ratios) > 1.02
                    else "reject" if max(ratios) < 0.98 else "inconclusive"
                )
                if stable
                else "inconclusive_aa_noise"
            )
        else:
            result["verdict"] = "numerical_failure"
        save(Path(plan["output"]) / f"rank{rank}.json", result)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    assert passed and all(statuses), "numerical/control failure"


def validate_results(plan):
    output = Path(plan["output"])
    if plan["mode"] == "capture":
        capture.initialize_snapshot()
        manifest = json.loads((output / "data/manifest.json").read_text())
        assert manifest["complete"] and {r["rank"] for r in manifest["ranks"]} == set(
            range(8)
        )
        for rank in range(8):
            capture.SNAP.load_rank_snapshot(output / "data" / f"rank{rank}.pt")
            cleanup = json.loads(
                (output / "data" / f"rank{rank}.cleanup.json").read_text()
            )
            assert cleanup["disarmed"] and cleanup["snapshot_gpu_bytes_retained"] == 0
            assert cleanup["capture_buckets_removed"] == 1 and cleanup["hooks_restored"]
    else:
        ranks = [
            json.loads((output / f"rank{rank}.json").read_text()) for rank in range(8)
        ]
        assert {r["rank"] for r in ranks} == set(range(8))
        assert {r["hardware"]["bdf"] for r in ranks} == set(plan["selected_bdfs"])
        assert all(r["all_ranks_pass"] for r in ranks)
        assert all(len(r["pairs"]) == len(r["aa_pairs"]) == 5 for r in ranks)
    return True


def supervise(plan, path):
    check_plan(plan, path)
    assert os.environ.get("K31_TASK19_GPU_GRANT") == plan["head"]
    assert os.environ.get("K31_TASK19_PLAN_SHA") == sha(path)
    assert not os.environ.get("KFD_GUARD_LOG") and not os.environ.get("LD_PRELOAD")
    assert os.nice(0) == 0 and set(os.sched_getaffinity(0)).issubset(range(112))
    common, Owned = LEGACY.resource_components()
    owned, children, stopped = Owned(common), [], []
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", plan)
    result = {
        "head": plan["head"],
        "mode": plan["mode"],
        "plan_sha256": sha(path),
        "complete": False,
    }
    fd = common.GpuLock().acquire()
    handlers = {
        sig: signal.signal(sig, lambda sig, frame: stopped.append(sig))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        result["admission"] = common.assert_gpus_idle()
        common.assert_ports_free(plan["ports"])
        env = dict(
            plan["environment"],
            K31_TASK19_GPU_GRANT=plan["head"],
            K31_TASK19_PLAN_SHA=sha(path),
            K31_TASK19_GPU_LOCK_FD=str(fd),
        )
        start = time.monotonic()
        for rank in range(1 if plan["mode"] == "capture" else 8):
            script = (
                Path(__file__).with_name("capture.py")
                if plan["mode"] == "capture"
                else Path(__file__)
            )
            command = [
                "/opt/venv/bin/python",
                str(script),
                "--worker",
                "--plan",
                str(path),
            ]
            if plan["mode"] == "replay":
                command += ["--rank", str(rank)]
            child = common.spawn(
                command,
                env=env,
                cwd=ROOT,
                log_path=output / f"worker{rank}.log",
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
            and time.monotonic() < start + plan["max_job_seconds"]
        ):
            owned.live()
            time.sleep(0.05)
        result.update(
            exit_codes=[p.poll() for p in children],
            stop_signals=stopped,
            complete=len(children) == (1 if plan["mode"] == "capture" else 8)
            and all(p.poll() == 0 for p in children)
            and not stopped,
            elapsed_s=time.monotonic() - start,
        )
        if result["complete"]:
            validate_results(plan)
            result["validated_results"] = True
    except BaseException as error:  # noqa: BLE001 - cleanup remains owned
        result["error"] = repr(error)
        result["complete"] = False
    finally:
        try:
            owned.stop()
        except Exception as error:  # noqa: BLE001 - retain ownership on cleanup failure
            result["cleanup_stop_error"] = repr(error)
        while True:
            try:
                alive = [p.pid for p in owned.live()]
                snapshot, used = LEGACY.release_snapshot()
                released = LEGACY.selected_released(alive, snapshot, used)
                result["cleanup"] = {
                    "owned_births": owned.seen,
                    "alive": alive,
                    "snapshot": snapshot,
                    "selected_used_bytes": used,
                    "released": released,
                }
                common.save_json(output / "controller.json", result)
            except Exception as error:  # noqa: BLE001
                print(
                    json.dumps({"cleanup_error": repr(error), "retaining_lock": True}),
                    flush=True,
                )
                released = False
            if released:
                os.close(fd)
                break
            time.sleep(5)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert result["complete"], result


def ownership_child(directory):
    """CPU-only nested process shape used by the maintained ownership check."""
    assert os.environ.get("KFD_GUARD_LOG") and os.environ.get("LD_PRELOAD")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    save(
        Path(directory) / "nested.json",
        {
            "child": os.getpid(),
            "grandchild": process.pid,
            "blocked": list(signal.pthread_sigmask(signal.SIG_BLOCK, [])),
        },
    )
    process.wait()


def ownership_fixture(directory):
    import multiprocessing

    assert os.environ.get("KFD_GUARD_LOG") and os.environ.get("LD_PRELOAD")
    child = multiprocessing.get_context("spawn").Process(
        target=ownership_child, args=(directory,)
    )
    child.start()
    child.join()


def cpu_checks():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace as NS
    from unittest.mock import patch

    import psutil
    import torch

    helpers()
    result = {
        "capture": capture.cpu_checks(),
        "native_boundary": TAIL.selfcheck(ROOT, PRIVATE / "donor-78e27"),
    }
    import torch.distributed as dist

    item = {
        "owned": torch.ones(4, dtype=torch.int32),
        "x": torch.ones(4, 96, 512, dtype=torch.bfloat16),
        "out": torch.ones(4, 12, 128, dtype=torch.bfloat16),
        "native_out": torch.ones(4, 1536, dtype=torch.bfloat16),
    }

    def native():
        item["native_out"].fill_(1)
        item["native_out"][item["owned"] == 0] = 0

    def donor():
        item["out"].fill_(1)
        item["out"][item["owned"] == 0] = 0

    graphs = {"native": NS(replay=native), "donor": NS(replay=donor)}
    with (
        patch.object(dist, "barrier", lambda: None),
        patch.object(torch.cuda, "synchronize", lambda: None),
    ):
        assert all(r["passed"] for r in empty_controls(item, graphs))
        graphs["native"] = NS(replay=lambda: item["native_out"].fill_(1))
        assert all(not r["passed"] for r in empty_controls(item, graphs))
    assert bool((item["owned"] == 1).all()) and bool((item["x"] == 1).all())
    result["both_arms_empty_zero_negative_control"] = True
    binding = binding_functions()
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        runtime = directory / "binding.py"
        runtime.write_text("initial bytes")
        record = {
            "files": {str(runtime): binding["file_identity"](runtime)},
            "models": {},
        }
        binding["verify_binding"](record)
        runtime.write_text("changed bytes")
        try:
            binding["verify_binding"](record)
        except RuntimeError:
            result["runtime_mutation_refused"] = True
        else:
            raise AssertionError("changed runtime binding accepted")
        actual_common, Owned = LEGACY.resource_components()
        inherited_env = dict(os.environ)
        foreign = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=inherited_env,
            start_new_session=True,
        )
        lock_path = directory / "cpu-only-lock"
        descriptors, launched = [], []

        class CpuLock:
            def acquire(self):
                descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                descriptors.append(descriptor)
                return descriptor

        def spawn(command, **kwargs):
            child = actual_common.spawn(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--ownership-fixture",
                    str(directory),
                ],
                env=inherited_env,
                cwd=ROOT,
                log_path=directory / "child.log",
                lock_fd=kwargs["lock_fd"],
                cpus="112-127",
            )
            launched.append(child)
            return child

        cpu_common = NS(
            GpuLock=CpuLock,
            spawn=spawn,
            descendants=actual_common.descendants,
            assert_gpus_idle=lambda: {"cpu_stub": True},
            assert_ports_free=lambda ports: None,
            save_json=save,
        )
        path = directory / "plan.json"
        plan = {
            "head": "cpu-selfcheck",
            "mode": "capture",
            "output": str(directory / "output"),
            "ports": [],
            "environment": {},
            "max_job_seconds": 2,
        }
        save(path, plan)
        parent_env = {
            k: v
            for k, v in inherited_env.items()
            if k not in ("KFD_GUARD_LOG", "LD_PRELOAD")
        }
        parent_env.update(
            K31_TASK19_GPU_GRANT=plan["head"], K31_TASK19_PLAN_SHA=sha(path)
        )
        try:
            with (
                patch.dict(os.environ, parent_env, clear=True),
                patch("os.nice", lambda n: 0),
                patch("os.sched_getaffinity", lambda pid: {0}),
                patch.dict(globals(), check_plan=lambda *args: None),
                patch.object(
                    LEGACY, "resource_components", lambda: (cpu_common, Owned)
                ),
                patch.object(
                    LEGACY,
                    "release_snapshot",
                    lambda: ({"kfd_pids": []}, {b: 0 for b in LEGACY.SELECTED_BDFS}),
                ),
            ):
                try:
                    supervise(plan, path)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("deadline control passed as complete")
            controller = json.loads(
                (Path(plan["output"]) / "controller.json").read_text()
            )
            nested = json.loads((directory / "nested.json").read_text())
            assert not controller["complete"] and controller["cleanup"]["released"]
            seen = {int(pid) for pid in controller["cleanup"]["owned_births"]}
            assert {nested["child"], nested["grandchild"]} <= seen
            assert not nested["blocked"] and foreign.poll() is None
            for pid in seen:
                assert (
                    not psutil.pid_exists(pid)
                    or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                )
            try:
                os.fstat(descriptors[0])
            except OSError:
                pass
            else:
                raise AssertionError(
                    "ownership lock descriptor retained after proven release"
                )
            result["nested_owned_deadline"] = {
                "released": True,
                "owned_count": len(seen),
                "foreign_untouched": True,
                "child_signals_unblocked": True,
            }
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)
            for process in launched:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        assert LEGACY.selected_released(
            [], {"kfd_pids": []}, {b: 0 for b in LEGACY.SELECTED_BDFS}
        )
        assert not LEGACY.selected_released(
            [], {"kfd_pids": [123]}, {b: 0 for b in LEGACY.SELECTED_BDFS}
        )
        assert not LEGACY.selected_released(
            [], {"kfd_pids": []}, {b: 2 * 1024**3 for b in LEGACY.SELECTED_BDFS}
        )
        result["unknown_or_occupied_release_refused"] = True
    assert not torch.cuda.is_initialized()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", choices=("capture", "replay"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--ownership-fixture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, choices=range(8))
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(cpu_checks(), indent=2))
    elif args.ownership_fixture:
        ownership_fixture(args.ownership_fixture)
    elif args.write_plan:
        save(args.plan, make_plan(args.write_plan, args.out, args.plan, args.snapshot))
    else:
        assert args.plan is not None
        plan = json.loads(args.plan.read_text())
        if args.worker:
            check_plan(plan, args.plan, worker=True)
            assert args.rank is not None
            assert Path(
                os.readlink("/proc/self/fd/" + os.environ["K31_TASK19_GPU_LOCK_FD"])
            ).resolve() == Path("/app/agent-team-scratch/g24-gpu.lock")
            rank_worker(args.rank, plan)
        else:
            supervise(plan, args.plan)
