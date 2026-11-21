"""Record bounded gfx950 calibration with the pinned probe and AITER controls."""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import runpy
import signal
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PROBE_SHA = "6a63deee204d31a5e49ed1e1536b1a15c024725dac07ca57df8406b86256ec5f"
AITER_SHA = "572232d23661a094fbc4e98e879713f0e7d2d5f4"


def positive(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"expected finite positive measurement, got {value}")
    return value


def rates(flops, samples_us):
    values = [positive(float(value)) for value in samples_us]
    median = statistics.median(values)
    return {
        "samples_us": values,
        "median_us": median,
        "min_us": min(values),
        "max_us": max(values),
        "achieved_tflops": flops / median / 1e6,
    }


def check_ratio(error, limit):
    if not math.isfinite(error) or error > limit:
        raise AssertionError(f"mismatch ratio {error} exceeds {limit}")


def emit(row):
    print(json.dumps(row, allow_nan=False, default=str), flush=True)


@contextmanager
def clock_samples(bdf):
    import amdsmi

    stop, samples, errors = threading.Event(), [], []
    amdsmi.amdsmi_init()
    worker = None
    try:
        handle = next(
            h
            for h in amdsmi.amdsmi_get_processor_handles()
            if amdsmi.amdsmi_get_gpu_device_bdf(h).lower() == bdf
        )

        def sample():
            try:
                while not stop.is_set():
                    mhz = positive(
                        float(
                            amdsmi.amdsmi_get_clock_info(
                                handle, amdsmi.AmdSmiClkType.SYS
                            )["clk"]
                        )
                    )
                    samples.append({"monotonic_s": time.monotonic(), "sclk_mhz": mhz})
                    stop.wait(0.05)
            except Exception as exc:  # noqa: BLE001 - retain failure evidence
                errors.append(exc)

        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
        yield samples
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError("clock sampler did not stop")
        amdsmi.amdsmi_shut_down()
        emit({"clock_samples": samples, "clock_errors": [str(e) for e in errors]})
    if errors:
        raise RuntimeError("clock sampling failed") from errors[0]
    if not samples:
        raise RuntimeError("clock sampling returned no data")


def library_compute(base, torch, bdf, seconds):
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    windows = []
    start = time.monotonic()
    with clock_samples(bdf) as clocks:
        while time.monotonic() - start < seconds:
            begin = time.monotonic()
            us = positive(base.timed_us(lambda: torch.mm(a, b, out=out), 30))
            end = time.monotonic()
            if begin >= start + 3:
                windows.append({"start_s": begin, "end_s": end, "us": us})
    if not windows:
        raise RuntimeError("no post-warmup compute windows")
    accepted = [
        x["sclk_mhz"]
        for x in clocks
        if start + 3 <= x["monotonic_s"] <= windows[-1]["end_s"]
    ]
    if len(accepted) < 3:
        raise RuntimeError("too few sustained-clock samples")
    assert bool(torch.isfinite(out).all()), "library GEMM output is nonfinite"
    return dict(
        kind="library_bf16",
        shape=[8192] * 3,
        dtype="bfloat16",
        output_dtype="bfloat16",
        graph=False,
        bias=False,
        working_set_bytes=a.nbytes + b.nbytes + out.nbytes,
        timing="GPU events around 30 Python-issued calls; 5 warmups per window",
        warmup_s=3,
        windows=windows,
        clocks=clocks,
        sustained_sclk_mhz=statistics.median(accepted),
        theoretical_tflops=None,
        theoretical_reason="no verified gfx950 constant",
        resources="library kernel registers/LDS/spills unavailable",
        numerical_check="finite output; AITER control separately compares implementations",
        **rates(2 * 8192**3, [x["us"] for x in windows]),
    )


def memory_case(base, torch, nbytes, copy):
    n = nbytes // 2
    a = torch.randn(n, dtype=torch.float16, device="cuda")
    target = (
        torch.empty_like(a)
        if copy
        else torch.empty(
            (n + base.BLOCK - 1) // base.BLOCK, dtype=torch.float32, device="cuda"
        )
    )
    grid = ((n + base.BLOCK - 1) // base.BLOCK,)
    if copy:
        launch = lambda: base._copy[grid](a, target, n, base.BLOCK)
    else:
        launch = lambda: base._read[grid](a, n, target, base.BLOCK)
    compiled = launch()
    if copy:
        assert torch.equal(a, target), "copy differs from source"
    else:
        expected = a.view(-1, base.BLOCK).sum(dim=1, dtype=torch.float32)
        torch.testing.assert_close(target, expected, rtol=1e-4, atol=1e-3)
        del expected
    assert bool(torch.isfinite(target).all()), "memory output is nonfinite"
    us = positive(base.timed_us(launch))
    metadata = compiled.metadata
    moved = nbytes * (2 if copy else 1)
    return {
        "kind": "copy" if copy else "read",
        "input_bytes": nbytes,
        "working_set_bytes": a.nbytes + target.nbytes,
        "dtype": "float16",
        "graph": False,
        "calls": 30,
        "warmups": 5,
        "mean_us": us,
        "payload_tb_s": base.tbs(moved, us),
        "payload_bytes": moved,
        "reduction_output_bytes": 0 if copy else target.nbytes,
        "byte_boundary": "logical payload; read reduction stores excluded from rate",
        "numerical_check": "full output finite and matches torch reference",
        "resources": {
            "vgprs": getattr(compiled, "n_regs", None),
            "memory_spills": getattr(compiled, "n_spills", None),
            "lds_bytes": getattr(metadata, "shared", None),
            "warps": getattr(metadata, "num_warps", None),
            "sgprs": None,
            "occupancy": None,
        },
    }


def memory_roof(base, torch, bdf):
    rows = []
    with clock_samples(bdf) as clocks:
        for gib in (2, 8):
            for copy in (False, True):
                row = memory_case(base, torch, gib * 1024**3, copy)
                rows.append(row)
                emit(row)
    return {
        "rows": rows,
        "clocks": clocks,
        "clock_boundary": "includes allocations, correctness and timed memory calls",
    }


def cache_knee(base, torch, bdf):
    rows = []
    with clock_samples(bdf) as clocks:
        for mib in base.CACHE_SIZES_MIB:
            row = memory_case(base, torch, mib * 1024**2, False)
            row["mib"] = mib
            rows.append(row)
            emit(row)
            torch.cuda.empty_cache()
    eligible = {i for i, r in enumerate(rows) if r["mean_us"] >= base.MIN_US_FOR_KNEE}
    knee = base.cache_estimate(
        [r["mib"] for r in rows], [r["payload_tb_s"] for r in rows], eligible
    )
    return {
        "rows": rows,
        "clocks": clocks,
        "plateau_last_mib": knee,
        "interpretation": (
            "unresolved/no visible knee"
            if knee is None
            else "last sampled size near the visible plateau, not a physical capacity"
        ),
        "eligible_indices": sorted(eligible),
        "min_duration_us": base.MIN_US_FOR_KNEE,
        "limitation": "32 MiB minimum cannot resolve the device's 4 MiB L2",
        "clock_boundary": "includes allocations, correctness and timed cache calls",
    }


def tensor_inputs(torch, fn, args, kwargs):
    values = dict(kwargs)
    values.update({f"arg{i}": value for i, value in enumerate(args)})
    unwrapped = inspect.unwrap(fn)
    values.update(inspect.getclosurevars(unwrapped).nonlocals)
    return {
        key: {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "nbytes": value.nbytes,
        }
        for key, value in values.items()
        if isinstance(value, torch.Tensor)
    }


def aiter_control(kind, root, torch, bdf):
    import aiter
    import aiter.test_common as common

    assert Path(aiter.__file__).resolve().is_relative_to(root / "aiter")
    original_perf, original_check = common.perftest, common.checkAllclose
    original_benchmark = common.benchmark
    entries, checks, summaries = [], [], []
    signature = inspect.signature(original_check)

    def checked(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        error = original_check(*args, **kwargs)
        limit = call.arguments["tol_err_ratio"]
        check = {
            "msg": call.arguments["msg"],
            "ratio": float(error),
            "limit": limit,
            "rtol": call.arguments["rtol"],
            "atol": call.arguments["atol"],
            "catastrophic_check": call.arguments["catastrophic_check"],
        }
        checks.append(check)
        emit({"numerical_check": check})
        check_ratio(error, limit)
        return error

    def frozen_perf(*args, **kwargs):
        settings = inspect.signature(original_perf).bind(*args, **kwargs)
        settings.apply_defaults()
        settings.arguments.update(
            num_rotate_args=1, testGraph=False, use_cuda_event=False, needTrace=False
        )

        def decorate(fn):
            timed = original_perf(**settings.arguments)(fn)

            def run(*args, **kwargs):
                inputs = tensor_inputs(torch, fn, args, kwargs)
                start = time.monotonic()
                with clock_samples(bdf) as clocks:
                    output, us = timed(*args, **kwargs)
                row = {
                    "function": fn.__name__,
                    "mean_us": positive(float(us)),
                    "inputs": inputs,
                    "settings": dict(settings.arguments),
                    "start_s": start,
                    "end_s": time.monotonic(),
                    "clocks": clocks,
                    "resources": "registers/LDS/spills unavailable from perftest",
                    "timing": "AITER profiler kernel aggregate, not wall-clock latency",
                    "clock_boundary": "perftest warmup and profiler interval",
                }
                entries.append(row)
                emit(row)
                return output, us

            return run

        return decorate

    def recorded_benchmark():
        def decorate(fn):
            benchmark = original_benchmark()(fn)

            def run(*args, **kwargs):
                summary = benchmark(*args, **kwargs)
                summaries.append(summary)
                emit({"summary": summary})
                return summary

            return run

        return decorate

    common.benchmark = recorded_benchmark
    common.checkAllclose, common.perftest = checked, frozen_perf
    script = root / "op_tests" / f"test_gemm_{kind}.py"
    flags = (
        ["-d", "bf16", "-o", "bf16", "-mnk", "8192,8192,8192"]
        if kind == "a16w16"
        else ["-d", "bf16", "-mnk", "2048,8192,8192", "-mnk_splitk"]
    )
    saved_argv = sys.argv
    try:
        sys.argv = [str(script), *flags]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = saved_argv
        common.checkAllclose, common.perftest = original_check, original_perf
        common.benchmark = original_benchmark
    assert checks and entries, "control did not execute numerical checks and timing"
    return {
        "kind": kind,
        "argv": [str(script), *flags],
        "entries": entries,
        "checks": checks,
        "summaries": summaries,
        "theoretical_tflops": None,
        "working_set": "one input set per callable; allocations/workspace excluded from descriptors",
        "coverage": "all implementations in the pinned normal test, no sweep",
    }


DEPENDENCIES = {
    "/app/k31-agent-team-private/task-10/dependencies/owned-run.py": "a102579f60f77f78faf47c3999cccc7d3b1b46dd7ed68f7ba40cf5fe583558ea",
    "/app/k31-agent-team-private/task-10/dependencies/owned-protocol.json": "d3345964118cecd1d786824a38220258dad654af12a0040c662d8a0422158ca3",
    "/share_nfs/whx/work/k31_perf_opt/.opt-team/harness/common.py": "1479cc118a3d667cc19c570dae34f779c6c7fcac5df8bd061c6537a9fbddd3ab",
}


def resource_components():
    for filename, expected in DEPENDENCIES.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    owned_path, protocol_path, common_path = DEPENDENCIES
    node = next(
        n
        for n in ast.parse(Path(owned_path).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Owned"
    )
    protocol = json.loads(Path(protocol_path).read_text())
    protocol.update(term_timeout_s=3, kill_timeout_s=2)
    scope = {"os": os, "signal": signal, "time": time, "P": protocol}
    exec(  # noqa: S102 -- unchanged class from hash-verified source, same import contract
        compile(ast.Module(body=[node], type_ignores=[]), owned_path, "exec"), scope
    )
    spec = importlib.util.spec_from_file_location("hardware_resources", common_path)
    resources = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resources)
    return resources, scope["Owned"]


def selected_released(alive, seen, snapshot, used_bytes):
    kfd = snapshot["kfd_pids"]
    return (
        not alive
        and 0 <= used_bytes < 2 * 1024**3
        and all(isinstance(pid, int) for pid in kfd)
        and not set(kfd).intersection(seen)
    )


def execute_plan(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    root = Path(__file__).resolve().parents[2]
    assert os.environ.get("K31_TASK13_PLAN_SHA") == hashlib.sha256(raw).hexdigest()
    assert os.environ.get("K31_TASK13_GPU_GRANT") == plan["head"]
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        == plan["head"]
    )
    assert plan["bdf"] == "0000:75:00.0" and plan["execution_limit_s"] == 55
    assert not os.environ.get("KFD_GUARD_LOG") and not os.environ.get("LD_PRELOAD")
    assert os.nice(0) == 0 and set(os.sched_getaffinity(0)).issubset(range(112))
    for filename, expected in plan["files"].items():
        assert (
            hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
        ), filename
    output = Path(plan["output"])
    assert output.is_absolute() and output.is_relative_to(
        Path("/app/k31-agent-team-private/task-13/runs")
    )
    assert not output.exists()
    resources, owned_class = resource_components()
    fd = resources.GpuLock().acquire()
    owned = owned_class(resources)
    child, stopped = None, []
    state = {"head": plan["head"], "complete": False, "phase": "admission"}
    previous = {
        sig: signal.signal(sig, lambda sig, frame: stopped.append(sig))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        output.mkdir(parents=True)
        (output / "plan.json").write_bytes(raw)
        state["before"] = resources.assert_gpus_idle()
        start = time.monotonic()
        state.update(phase="measurement", start_monotonic_s=start)
        # Parent handlers only record cancellation; children retain normal TERM handling.
        child = resources.spawn(
            plan["command"],
            env=plan["environment"],
            cwd=root,
            log_path=output / "measurement.log",
            lock_fd=fd,
            cpus="0-111",
        )
        owned.add(child)
        while child.poll() is None and not stopped and time.monotonic() < start + 55:
            owned.live()
            time.sleep(0.05)
        state.update(
            exit_code=child.poll(),
            elapsed_s=time.monotonic() - start,
            stop_signals=stopped,
            timeout=child.poll() is None and not stopped,
        )
        if child.poll() == 0 and not stopped:
            state["complete"] = json.loads((output / "result.json").read_text())[
                "complete"
            ]
    except BaseException as error:  # noqa: BLE001 - retain failure evidence
        state["error"] = repr(error)
    finally:
        state["phase"] = "cleanup"
        try:
            owned.stop()
        except Exception as error:  # noqa: BLE001 - retain failure evidence
            state["cleanup_stop_error"] = repr(error)
        while True:
            released = False
            try:
                alive = [p.pid for p in owned.live()]
                snapshot = resources.gpu_snapshot() if child is not None else None
                used = (
                    int(
                        Path(
                            f"/sys/bus/pci/devices/{plan['bdf']}/mem_info_vram_used"
                        ).read_text()
                    )
                    if child is not None
                    else 0
                )
                released = child is None or selected_released(
                    alive, owned.seen, snapshot, used
                )
                state["cleanup"] = {
                    "owned_births": owned.seen,
                    "alive": alive,
                    "snapshot": snapshot,
                    "selected_bdf": plan["bdf"],
                    "selected_used_bytes": used,
                    "released": released,
                }
                resources.save_json(output / "controller.json", state)
            except Exception as error:  # noqa: BLE001 - retain failure evidence
                released = False
                emit({"cleanup_error": repr(error), "retaining_lock": True})
            if released:
                os.close(fd)
                break
            emit({"cleanup": "unverified", "retaining_lock": True})
            time.sleep(5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    emit(state)
    if not state["complete"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("roofline", "a16w16", "a4w4"), default="roofline")
    p.add_argument("--run-plan", type=Path)
    p.add_argument("--probe", type=Path)
    p.add_argument("--aiter", type=Path)
    p.add_argument("--bdf")
    p.add_argument("--what", default="compute,memory,cache")
    p.add_argument("--clock-seconds", type=float, default=12)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.run_plan:
        return execute_plan(args.run_plan)
    if not all((args.probe, args.aiter, args.bdf, args.out)):
        p.error("measurement requires --probe, --aiter, --bdf and --out")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert os.environ.get("K31_TASK13_GPU_GRANT") == head, "root GPU grant required"
    assert not args.out.exists(), "refusing to replace evidence"
    assert hashlib.sha256(args.probe.read_bytes()).hexdigest() == PROBE_SHA
    assert 6 <= args.clock_seconds <= 12
    what = args.what.split(",")
    assert (
        what
        and len(set(what)) == len(what)
        and set(what) <= {"compute", "memory", "cache"}
    )
    revision = subprocess.check_output(
        ["git", "-C", str(args.aiter), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == AITER_SHA, "AITER source revision differs"
    for change in ([], ["--cached"]):
        subprocess.run(
            ["git", "-C", str(args.aiter), "diff", "--exit-code", "--quiet", *change],
            check=True,
        )
    import torch
    import triton

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1, "expose only the reviewed device"
    prop = torch.cuda.get_device_properties(0)
    bdf = f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}.0"
    assert bdf == args.bdf and prop.gcnArchName.split(":")[0] == "gfx950"
    result = {
        "mode": args.mode,
        "bdf": bdf,
        "arch": prop.gcnArchName,
        "cu_count": prop.multi_processor_count,
        "l2_bytes": prop.L2_cache_size,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pinned_probe_sha256": PROBE_SHA,
        "declared_aiter_revision": AITER_SHA,
        "started_utc_s": time.time(),
        "complete": False,
    }
    emit(result)
    try:
        if args.mode == "roofline":
            spec = importlib.util.spec_from_file_location("roofline_base", args.probe)
            base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base)
            for name in what:
                if name == "compute":
                    row = library_compute(base, torch, bdf, args.clock_seconds)
                elif name == "memory":
                    row = memory_roof(base, torch, bdf)
                else:
                    row = cache_knee(base, torch, bdf)
                result[name] = row
                emit({name: row})
        else:
            result["control"] = aiter_control(
                args.mode, args.aiter.resolve(), torch, bdf
            )
        result["complete"] = True
    finally:
        result["finished_utc_s"] = time.time()
        args.out.write_text(
            json.dumps(result, indent=2, allow_nan=False, default=str) + "\n"
        )
    emit({"complete": True, "output": str(args.out)})


if __name__ == "__main__":
    main()
