"""Current whole-K2 replay using the existing restore and timing machinery."""

import argparse
import ast
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import capture

ROOT = capture.ROOT
HERE = Path(__file__).resolve().parent
WS = Path("/share_nfs/whx/work/k31_perf_opt/.opt-team")
PRIVATE = Path("/app/k31-agent-team-private/task-15")
RECIPE = (
    WS / "agent-team/evidence/task-6-save-completion-req9-0d914c12/frozen-plan.json"
)
TP, ROWS, NB, SLOT, TAG = (
    capture.TP,
    capture.ROWS,
    capture.NB,
    capture.SLOT,
    capture.TAG,
)
prior = timing = prepare_iteration = WEIGHTS = None
sha = capture.sha


def initialize_helpers():
    global prior, timing, prepare_iteration, WEIGHTS
    capture.initialize_snapshot()
    sys.path.append(str(capture.TOOLS))
    prior = capture.helper("front_floor_replay")
    timing = capture.helper("front_floor_timing_v2")
    prepare_iteration, WEIGHTS = prior.prepare_iteration, capture.SNAP.WEIGHTS


METHOD = "current_whole_k2_xready_ordinary_events"
CLOCK_MODE = "clock_boundary"
CLOCK_MARKER = PRIVATE / "cache/comgr/llvmcache.timestamp"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = set((ROOT / "atom").rglob("*.py"))
    paths.update(path for path in HERE.iterdir() if path.is_file())
    paths.update(
        capture.TOOLS / (name + ".py")
        for name in (
            "front_floor_capture",
            "front_floor_copy",
            "front_floor_snapshot",
            "front_floor_replay",
            "front_floor_replay_v2",
            "front_floor_timing_v2",
            "front_floor_event_probe",
            "front_floor_compile",
            "run_front_floor_v2",
        )
    )
    paths.update((WS / "harness/common.py", timing.HEADER, RECIPE))
    return {str(path): sha(path) for path in sorted(paths)}


def file_identity(path, *, content=True):
    path = Path(path)
    before = path.stat()
    result = {
        "resolved": str(path.resolve()),
        "stat": [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns],
    }
    if content:
        result["sha256"] = sha(path)
    elif path.suffix == ".safetensors":
        with path.open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            if not 0 < length <= 16 << 20:
                raise ValueError("invalid model shard header: " + str(path))
            result["header_sha256"] = hashlib.sha256(source.read(length)).hexdigest()
    after = path.stat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        raise RuntimeError("file changed during identity read: " + str(path))
    return result


def execution_binding(base, *, clock_only=False):
    """Bind installed runtime bytes without importing their executable modules."""
    import importlib.metadata

    aiter = PRIVATE / "aiter-572232d"
    head = subprocess.check_output(
        ["git", "-C", str(aiter), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != base["protocol"]["aiter_sha"]:
        raise RuntimeError("AITER source differs from the fixed recipe")
    site = Path("/opt/venv/lib/python3.12/site-packages")
    paths = {Path("/opt/venv/bin/python"), Path("/opt/venv/bin/lmcache")}
    versions = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        versions[distribution.metadata["Name"]] = distribution.version
        for entry in distribution.files or ():
            path = Path(distribution.locate_file(entry))
            if path.is_file() and path.suffix != ".pyc":
                paths.add(path)
    paths.update(site.glob("*.pth"))
    for directory in (aiter, PRIVATE / "cache", PRIVATE / "home/.cache"):
        paths.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.name != ".git"
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
            and not path.name.endswith((".lock", ".log"))
        )
    paths.update(
        Path(path)
        for path in (
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib/libhsa-runtime64.so",
            "/opt/rocm/lib/libamd_comgr.so",
            "/opt/rocm/llvm/bin/clang",
            "/opt/rocm/llvm/bin/ld.lld",
            "/opt/rocm/llvm/bin/llvm-objdump",
        )
    )
    if clock_only:
        paths.update(Path(name) for name in clock_protocol(PRIVATE)["profiler_files"])
    seed = WS / "agent-team/work/15/cache-seed.json"
    paths.add(seed)
    records, resolved = {}, {}
    for path in sorted(paths):
        real = path.resolve()
        if real not in resolved:
            resolved[real] = file_identity(path)
        records[str(path)] = resolved[real]
    models = {}
    for name in (() if clock_only else ("model", "draft")):
        directory = Path(base["protocol"][name])
        files = [
            p for p in directory.rglob("*") if p.is_file() and ".git" not in p.parts
        ]
        if not files:
            raise RuntimeError("model files unavailable: " + str(directory))
        models.update(
            {
                str(p): file_identity(
                    p, content=p.suffix not in (".safetensors", ".bin")
                )
                for p in files
            }
        )
    return {
        "versions": versions,
        "aiter_head": head,
        "aiter_submodules": subprocess.check_output(
            ["git", "-C", str(aiter), "submodule", "status"], text=True
        ),
        "files": records,
        "models": models,
        "model_scope": (
            "No model or snapshot binding/loading in clock-only mode."
            if clock_only
            else "Metadata/content and shard filesystem/header identity; actual selected materialized weight bytes are hashed in the real capture."
        ),
        "cache_scope": "Preexisting private artifacts are bound; runtime may append artifacts only within the declared private caches.",
    }


def clock_marker_policy():
    return {
        "path": str(CLOCK_MARKER),
        "vary": "st_mtime_ns",
        "size": 0,
        "sha256": EMPTY_SHA256,
    }


def verify_binding(binding, marker_policy=None):
    if marker_policy is not None and marker_policy != clock_marker_policy():
        raise RuntimeError("unsupported clock marker policy")
    checked, observations = {}, []
    for records in (binding["files"], binding["models"]):
        for name, expected in records.items():
            path = Path(name)
            key = (str(path.resolve()), "sha256" in expected)
            if key not in checked:
                checked[key] = file_identity(path, content=key[1])
            actual = checked[key]
            if (
                marker_policy is not None
                and records is binding["files"]
                and name == str(CLOCK_MARKER)
            ):
                if (
                    expected["resolved"] != str(CLOCK_MARKER)
                    or expected["stat"][2] != 0
                    or expected.get("sha256") != EMPTY_SHA256
                ):
                    raise RuntimeError("clock marker is not the bound exact empty file")
                observed = actual["stat"][3]
                observations.append(
                    {
                        "path": name,
                        "bound_mtime_ns": expected["stat"][3],
                        "observed_mtime_ns": observed,
                        "delta_mtime_ns": observed - expected["stat"][3],
                    }
                )
                actual = {**actual, "stat": [*actual["stat"][:3], expected["stat"][3]]}
            if actual != expected:
                raise RuntimeError("runtime/model/cache binding changed: " + name)
    if marker_policy is not None and len(observations) != 1:
        raise RuntimeError("clock marker missing from binding")
    return observations


def binding_check():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        for name in (
            "runtime.py",
            "hf-modules/tokenization_kimi.py",
            "tiktoken/encoding",
        ):
            path = Path(directory) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("initial runtime bytes")
            binding = {"files": {str(path): file_identity(path)}, "models": {}}
            verify_binding(binding)
            path.write_text("changed runtime bytes")
            try:
                verify_binding(binding)
            except RuntimeError as error:
                assert "binding changed" in str(error)
            else:
                raise AssertionError("modified runtime was accepted")
    return True


def verify_floor():
    native = ROOT / "atom/models/kimi_k3/mono/kernels/layer.py"
    return {
        "native_source": sha(native),
        "adapter": sha(HERE / "layer_floor.py"),
        "required_compile_proof": "direct/original-bindings/floor/failure/baseline-after source, IR and text",
    }


def compile_entries(key_args):
    import layer_floor

    from atom.models.kimi_k3.mono.kernels import layer
    from atom.mono.runtime.abi import KernelAbi
    from atom.mono.runtime.compile import compile_kernels, launcher_resources
    from atom.mono.runtime.step_begin import StepBeginBuild, _build

    key = layer.K2Build(**key_args)
    if (key.tokens, key.nblocks, key.reduce, key.diag, key.timeline) != (
        ROWS,
        NB,
        True,
        False,
        False,
    ):
        raise ValueError("unsupported frozen current key")
    keys = {0: key, 1: key}
    entries = {0: layer.build(key), 1: layer_floor.build(key)}
    step = _build(StepBeginBuild(TP, False))
    abi = KernelAbi(("s0", "n0", "s1", "n1", "peers", "fence_off", "rank", "state"))
    compile_kernels([(entries[arm], layer.ABI) for arm in (0, 1)] + [(step, abi)])
    return (
        keys,
        entries,
        {arm: launcher_resources(entry) for arm, entry in entries.items()},
    )


def graph_occupancy(structure):
    """HIP's resource occupancy limit, not achieved occupancy."""

    class Dim3(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint) for name in ("x", "y", "z")]

    class Params(ctypes.Structure):
        _fields_ = [
            ("block", Dim3),
            ("extra", ctypes.c_void_p),
            ("function", ctypes.c_void_p),
            ("grid", Dim3),
            ("args", ctypes.c_void_p),
            ("shared", ctypes.c_uint),
        ]

    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipGraphKernelNodeGetParams.argtypes = [ctypes.c_void_p, ctypes.POINTER(Params)]
    hip.hipGraphKernelNodeGetParams.restype = ctypes.c_int
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
    params = Params()
    error = hip.hipGraphKernelNodeGetParams(
        int(structure["nodes"][0]["handle"], 16), ctypes.byref(params)
    )
    if error or params.block.x * params.block.y * params.block.z != 512:
        raise RuntimeError("kernel-node parameters unavailable or unexpected")
    blocks = ctypes.c_int()
    error = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks), params.function, 512, params.shared
    )
    if error or blocks.value <= 0:
        raise RuntimeError("HIP module occupancy is unavailable")
    return {
        "active_blocks_per_CU_limit": blocks.value,
        "waves_per_block": 8,
        "dynamic_LDS_bytes": params.shared,
        "scope": "runtime resource limit, not achieved occupancy",
    }


def summarize(journal):
    answer = {
        "metric": "max-rank single-K2 graph event envelope",
        "delays": {},
        "rejected_pairs": sum(not p["accepted"] for p in journal),
    }
    for scale in (1, 2):
        selected = [p for p in journal if p["accepted"] and p["delay_scale"] == scale]
        pairs, noise = [], []
        for item in selected:
            left, right = item["trials"]
            if item["kind"] == "AA":
                noise.append(max(r["us"] for r in left) - max(r["us"] for r in right))
            else:
                a, b = (left, right) if item["arms"] == [0, 1] else (right, left)
                av, bv = max(r["us"] for r in a), max(r["us"] for r in b)
                pairs.append(
                    {
                        "pair": item["window"],
                        "order": item["kind"],
                        "A_max_rank_us": av,
                        "B_max_rank_us": bv,
                        "delta_us": av - bv,
                    }
                )
        if len(pairs) != 5 or len(noise) != 5:
            raise ValueError(
                "five alternating pairs and their A/A controls are required"
            )
        base = statistics.median(p["A_max_rank_us"] for p in pairs)
        delta = statistics.median(p["delta_us"] for p in pairs)
        spread = [min(p["delta_us"] for p in pairs), max(p["delta_us"] for p in pairs)]
        margin = max(map(abs, noise))
        reproduction = {
            name: [
                min(p[name] for p in pairs) - margin,
                max(p[name] for p in pairs) + margin,
            ]
            for name in ("A_max_rank_us", "B_max_rank_us", "delta_us")
        }
        answer["delays"][scale] = {
            "pairs": pairs,
            "median_A_us": base,
            "median_delta_us": delta,
            "relative_delta": delta / base,
            "delta_range_us": spread,
            "AA_delta_us": noise,
            "AA_max_abs_us": margin,
            "reproduction_ranges_us": reproduction,
            "above_2pct": delta > 0.02 * base and min(p["delta_us"] for p in pairs) > 0,
            "AA_within_2pct": max(map(abs, noise)) <= 0.02 * base,
        }
    d, d2 = answer["delays"][1], answer["delays"][2]
    answer["delay_sensitive"] = abs(d["median_delta_us"] - d2["median_delta_us"]) > max(
        d["AA_max_abs_us"], d2["AA_max_abs_us"], 0.02 * d["median_A_us"]
    )
    answer["worth_further_front_study"] = not answer["delay_sensitive"] and all(
        x["above_2pct"] and x["AA_within_2pct"] for x in (d, d2)
    )
    answer["limit"] = (
        "Diagnostic optimistic floor for this input/key only; no production gain, pure-wait attribution, C12 extrapolation or absolute ceiling."
    )
    return answer


def rank_main(rank, directory, outdir, port, windows, warmup):
    initialize_helpers()
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=TP,
        timeout=timedelta(seconds=30),
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from capture import load_rank_snapshot, restore_tensors

    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.models.kimi_k3.mono.kernels.stages import moe
    from atom.mono.runtime.compile import launcher_resources
    from atom.mono.runtime.lifecycle import owned_peer_buffer
    from atom.mono.runtime.mailboxes import StepMailboxes
    from atom.mono.runtime.step_begin import ARRIVED, EPOCH, FENCE_BYTES
    from atom.mono.runtime.uncached import DeviceBytes, uncached_bytes

    assert torch.cuda.get_device_properties(dev).multi_processor_count == 256
    snapshot = load_rank_snapshot(
        directory / f"rank{rank}.pt", map_location="cpu", verify=True
    )
    ts = restore_tensors(snapshot, dev)
    cpu = restore_tensors(snapshot, "cpu")
    key_args = {**snapshot["meta"]["key"], "diag": False, "timeline": False}
    keys, entries, _ = compile_entries(key_args)
    scratch_n = (k2.scratch_bytes(keys[0]) + 15) // 16 * 16
    assert k2.scratch_bytes(keys[0]) == k2.scratch_bytes(keys[1])
    scratch, queue = uncached_bytes(scratch_n, dev), uncached_bytes(
        moe.QUEUE_BYTES, dev
    )

    class Owner:
        pass

    owner = Owner()
    peer_n, plain_n = k2.peer_bytes(ROWS), k2.plain_bytes(keys[0])
    peers, cleanup = owned_peer_buffer(
        owner, peer_n, dist.group.WORLD, rank, TP, dev, plain_bytes=plain_n
    )
    raw = torch.as_tensor(
        DeviceBytes(peers.local, peer_n + FENCE_BYTES + plain_n), device=dev
    )
    fence = raw[peer_n : peer_n + FENCE_BYTES].view(torch.int64)
    mail = StepMailboxes(peers, scratch, queue)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    torch.cuda.set_stream(stream)
    output = torch.empty_like(ts["out"])
    frozen = {name: ts[name].clone() for name in ("core", "blocks", "prefix_before")}
    xoff, xn = k2.plain_layout(keys[0])["xrow"]
    xrow = peers.plain[xoff : xoff + xn].view(torch.bfloat16).view(ROWS, 7168)
    roff, rn = k2.scratch_layout(keys[0])["xrdy"]
    xrdy = scratch[roff : roff + rn].view(torch.int32).view(56, 2)
    ready_pairs = torch.tensor([[1, TAG]] * 56, dtype=torch.int32, device=dev)
    scalars = snapshot["meta"]["abi_scalars"]
    pointers = {
        name: ts[name].data_ptr()
        for name in (*WEIGHTS, "core", "blocks", "prefix_before")
    }
    args = {name: ts[name].data_ptr() for name in (*WEIGHTS, "core", "blocks")}
    args.update(
        prefix=ts["prefix_before"].data_ptr(),
        out=output.data_ptr(),
        blk_sm=scalars["blk_sm"],
        blk_sr=scalars["blk_sr"],
        scratch=scratch.data_ptr(),
        plain=peers.plain.data_ptr(),
        queue=queue.data_ptr() + moe.queue_offset(SLOT),
        peers=peers.addresses.data_ptr(),
        rank=rank,
        layer=SLOT,
        reset=0,
        diag=0,
        tl=0,
    )
    packed = k2.ABI.pack(args)
    epoch, journal, checks, graph_checks = 0, [], [], {}

    def rendezvous():
        stream.synchronize()
        dist.barrier()

    def prepare(arm):
        nonlocal epoch
        rendezvous()
        prepare_iteration(arm, ts, frozen, output, xrow, xrdy, ready_pairs, mail)
        epoch += 1
        rendezvous()

    def exact(name, actual, expected):
        a, b = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        ):
            different = (
                a.view(torch.uint8).flatten() != b.view(torch.uint8).flatten()
            ).nonzero()
            first = int(different[0]) if different.numel() else None
            raise AssertionError(f"rank{rank}: {name} differs at byte {first}")

    def check(label, immutable=False):
        rendezvous()
        problem = None
        try:
            for name, actual, expected in (
                ("out", output, cpu["out"]),
                ("prefix", ts["prefix_before"], cpu["prefix_after"]),
                ("x", xrow, cpu["x"]),
                ("core", ts["core"], cpu["core"]),
                ("blocks", ts["blocks"], cpu["blocks"]),
            ):
                exact(name, actual, expected)
            if not bool(torch.isfinite(output).all()):
                raise AssertionError("non-finite output")
            off, size = k2.scratch_layout(keys[0])["route"]
            route = scratch[off : off + size].view(torch.int32).view(ROWS, 32, 2)
            exact("route_pairs", route, cpu["route_pairs"])
            assert bool((route[:, :, 1] == TAG).all())
            assert (
                torch.unique(route[:, :16, 0]).numel()
                == torch.unique(cpu["route_ids"]).numel()
            )
            state = mail._state.cpu()
            expected = torch.zeros_like(state)
            expected[EPOCH] = epoch
            expected.view(torch.int32)[ARRIVED * 2 + 1 : ARRIVED * 2 + 256] = epoch
            exact("mail_state", state, expected)
            exact("fence", fence, torch.full_like(fence.cpu(), epoch))
            assert all(ts[name].data_ptr() == ptr for name, ptr in pointers.items())
            assert bool((queue.view(torch.int32) >= 0).all())
            if immutable:
                for name in WEIGHTS:
                    exact(name, ts[name], cpu[name])
        # Exchange every rank's validation failure before any rank exits.
        except Exception as error:  # noqa: BLE001
            problem = str(error)
        problems = [None] * TP
        dist.all_gather_object(problems, problem)
        if any(problems):
            save(
                outdir / f"failure-rank{rank}.json",
                {"label": label, "errors": problems},
            )
            raise AssertionError(problems)
        checks.append({"label": label, "epoch": epoch, "immutable_weights": immutable})

    try:
        timer = timing.Timer(rank, stream, rank)
        clock_check = timer.check_clock(outdir)
        rendezvous()
        raw.zero_()
        scratch.zero_()
        queue.zero_()
        mail._state.zero_()
        rendezvous()
        for index, arm in enumerate((0, 0, 1, 1)):
            prepare(arm)
            _run_compiled(entries[arm], *packed, stream)
            check(f"eager_{index}_arm{arm}", immutable=True)
        graphs = {}
        for arm in (0, 1):
            prepare(arm)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            graph.enable_debug_mode()
            with torch.cuda.graph(graph, stream=stream):
                _run_compiled(entries[arm], *packed, stream)
            dot = outdir / f"graph-rank{rank}-arm{arm}.dot"
            graph.debug_dump(str(dot))
            graph_checks[arm] = timing.verify_single_kernel_graph(graph)
            graph_checks[arm]["occupancy"] = graph_occupancy(graph_checks[arm])
            graph.instantiate()
            graphs[arm] = graph
        for arm in (0, 1):
            for index in range(warmup):
                prepare(arm)
                graphs[arm].replay()
                check(f"warm_{index}_arm{arm}")
        trial_number = 0
        for scale in (1, 2):
            for window in range(windows):
                for kind, arms in (
                    (("AB", (0, 1)) if window % 2 == 0 else ("BA", (1, 0))),
                    ("AA", (0, 0)),
                ):

                    def measure(arm, scale=scale):
                        nonlocal trial_number
                        prepare(arm)
                        row = timer.measure(graphs[arm], scale)
                        check(f"D{scale}_trial{trial_number}_arm{arm}")
                        trial_number += 1
                        reports = [None] * TP
                        dist.all_gather_object(reports, row)
                        return reports

                    timing.collect_pair(
                        arms,
                        measure,
                        journal,
                        delay_scale=scale,
                        window=window,
                        kind=kind,
                        world_size=TP,
                    )
                    if rank == 0:
                        save(outdir / "pair-journal.json", journal)
        check("final", immutable=True)
        report = {
            "state": "COMPLETE",
            "rank": rank,
            "method": METHOD,
            "checks": checks,
            "transport": {
                "class": type(peers).__module__ + "." + type(peers).__qualname__,
                "peer_addresses": peers.addresses.cpu().tolist(),
                "local_address": peers.local,
            },
            "graph_checks": graph_checks,
            "clock_check": clock_check,
            "resources": {arm: launcher_resources(e) for arm, e in entries.items()},
        }
        save(outdir / f"rank{rank}.json", report)
        if rank == 0:
            save(outdir / "summary.json", summarize(journal))

        rendezvous()
    except BaseException as error:
        save(
            outdir / f"failed-rank{rank}.json",
            {
                "state": (
                    "TIMING_INCONCLUSIVE"
                    if isinstance(error, timing.TimingInconclusive)
                    else "FAILED"
                ),
                "error": repr(error),
                "pair_journal": journal,
            },
        )
        raise
    finally:
        cleanup()
        dist.destroy_process_group()


def cpu_check(snapshot, out, compile_only=False):
    if "kfd-guard" not in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError("CPU preparation requires no_gpu.sh")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, replay; assert 'torch' not in sys.modules; assert replay.prior is None; assert replay.capture.SNAP is None",
        ],
        check=True,
    )
    initialize_helpers()
    result = {
        "bootstrap_imports_checked": True,
        "floor": verify_floor(),
        "restore": prior.host_check(),
        "timing": timing.host_check(),
    }
    journal = []
    a = [{"us": value} for value in [10, *[20] * 7]]
    b = [{"us": value} for value in [17, *[3] * 7]]
    for scale in (1, 2):
        for window in range(5):
            order = [0, 1] if window % 2 == 0 else [1, 0]
            for kind, arms in (
                ("AB" if order == [0, 1] else "BA", order),
                ("AA", [0, 0]),
            ):
                journal.append(
                    {
                        "accepted": True,
                        "delay_scale": scale,
                        "window": window,
                        "kind": kind,
                        "arms": arms,
                        "trials": [a if arm == 0 else b for arm in arms],
                    }
                )
    checked = summarize(journal)
    assert checked["delays"][1]["median_delta_us"] == 3
    assert not checked["delay_sensitive"]
    result["current_pair_reduction_checked"] = True
    result["ownership"] = ownership_check()
    result["binding_mutation_rejected"] = binding_check()
    if compile_only or snapshot:
        from atom.models.kimi_k3.mono.kernels import layer
        from atom.mono.runtime.compile import _code_object

        key = dataclasses.asdict(layer.K2Build(tokens=ROWS, nblocks=NB))
        if snapshot:
            _manifest, files = prior.snapshot_paths(snapshot)
            identities = []
            for rank, path in enumerate(files):
                value = capture.load_rank_snapshot(path)
                capture.validate_snapshot(value)
                ident = value["meta"]["identity"]
                if ident["rank"] != rank:
                    raise ValueError("rank identity differs")
                identities.append({k: v for k, v in ident.items() if k != "rank"})
                if value["meta"]["key"] != key:
                    raise ValueError("actual key differs from the declared current key")
            if any(item != identities[0] for item in identities):
                raise ValueError("not one matched invocation")
            result.update(
                snapshot=str(snapshot.resolve()),
                manifest_sha256=sha(snapshot / "manifest.json"),
                rank_sha256=[sha(path) for path in files],
                identity=identities[0],
            )
        _keys, entries, resources = compile_entries(key)
        out.mkdir(parents=True, exist_ok=False)
        compile_helper = capture.helper("front_floor_compile")
        artifacts = {}
        for arm, entry in entries.items():
            cached = list(entry._mem_cache.values())
            if len(cached) != 1:
                raise ValueError("one code object per arm required")
            binary = _code_object(cached[0]._ir_text)
            path = out / f"arm{arm}.hsaco"
            path.write_bytes(binary)
            (out / f"arm{arm}.text").write_bytes(compile_helper.elf_text(binary))
            (out / f"arm{arm}.mlir").write_text(cached[0]._ir_text)
            artifacts[arm] = {"binary": sha(path), "resources": resources[arm]}
        result.update(
            key=key,
            abi=list(layer.ABI.names),
            artifacts=artifacts,
            actual_snapshot_validated=bool(snapshot),
            gpu_validated=False,
        )
    else:
        out.mkdir(parents=True, exist_ok=False)
    result.update(
        state="SNAPSHOT_CPU_VALIDATED" if snapshot else "SOURCE_CPU_CHECKED",
        source_hashes=source_hashes(),
    )
    save(out / "ready.json", result)
    return result


def clock_protocol(out, plan_path=None):
    tool = Path("/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so")
    sdk = Path("/opt/rocm/lib/librocprofiler-sdk.so")
    rocpd = Path("/opt/rocm/lib/librocprofiler-sdk-rocpd.so.1")
    command = [
        "/opt/venv/bin/python",
        str(HERE / "replay.py"),
        "--clock-cohort",
        "traced",
        "--plan",
        str(plan_path or out / "frozen-plan.json"),
    ]
    return {
        "empty_marker_mtime_policy": clock_marker_policy(),
        "normal_timer_messages": {
            "origin": "simple_timer.cpp:55",
            "format": "{} :: {:12.6f} sec",
            "lifecycle_labels": [
                "[rocprofv3] " + label
                for label in (
                    "tool initialization",
                    "output generation",
                    "tool finalization",
                )
            ],
            "fixed_commands": [
                command,
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                ["git", "-C", str(ROOT), "status", "--porcelain"],
            ],
            "python": "/opt/venv/bin/python",
            "python_flags": ["-B", "-s"],
            "children": {
                "worker": {
                    "max": TP,
                    "caller": "popen_spawn_posix.Popen._launch",
                    "program": "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd={tracker_fd}, pipe_handle={pipe_handle})",
                    "suffix": ["--multiprocessing-fork"],
                },
                "tracker": {
                    "max": 1,
                    "caller": "resource_tracker.ResourceTracker.ensure_running",
                    "program": "from multiprocessing.resource_tracker import main;main({tracker_fd})",
                    "suffix": [],
                },
            },
            "meaning": "Source-proven elapsed summaries only; duplicates are allowed and are neither launch counts nor success/flush proof. Dynamic labels require the exact qualified pre-launch ledger.",
        },
        "body_ms": 0.5,
        "guard_ms": [1, 1, 2, 2],
        "body_replay_ordinals": [0, 1, 2, 3, 4],
        "module_api_count": 6,
        "dispatch_count": 10,
        "max_measured_rows": 64,
        "cohorts": ["untraced", "traced"],
        "pci_bdfs": [
            f"0000:{bus}:00.0"
            for bus in ("05", "15", "65", "75", "85", "95", "e5", "f5")
        ],
        "hsa_timestamp_frequency_hz": 10**9,
        "profiler_environment": {
            "ROCPROFILER_CI_STRICT_TIMESTAMPS": "1",
            "ROCPROFILER_CI_FREQ_SCALE_TIMESTAMPS": "0",
        },
        "profiler_tool_library": str(tool.resolve()),
        "profiler_preload_libraries": [str(tool.resolve()), str(sdk.resolve())],
        "profiler_loaded_libraries": [
            str(tool.resolve()),
            str(sdk.resolve()),
            str(rocpd.resolve()),
        ],
        "profiler_files": ["/opt/rocm/bin/rocprofv3", str(tool), str(sdk), str(rocpd)],
        "profiler_command": [
            "/opt/rocm/bin/rocprofv3",
            "--kernel-trace",
            "--hip-trace",
            "--output-format",
            "json",
            "--output-directory",
            str(out / "data/traced/profiler"),
            "--output-file",
            "clock-%pid%",
            "--log-level",
            "warning",
        ],
        "lineage": "One priming module dispatch, one capture-definition module API without dispatch, five graph body dispatches and four measured module guards. Join PID/native TID/raw graphExec/raw stream/API ordinal/internal correlation exactly.",
        "scope": "Required clock lineage and all visible loss/repair diagnostics; unrelated unmodelled records retained without a whole-export zero-loss claim.",
        "stop": "First hard content/tick/order/submission/trace/ownership/deadline error stops subsequent observations/cohorts. Finite old E/B gate failures remain failed observations. Seal/review/stop after one set, including success.",
        "interpretation": "B is first memrealtime through last loop read, excluding stores/return; K is dispatch processing/completion; E is unchanged graph event envelope. No tick alignment, overhead subtraction or physical attribution. Non-nesting, trace-only behavior, unmatched records and nonrecurrence remain unresolved.",
    }


def make_clock_plan(out, plan_path):
    if plan_path is None:
        raise ValueError("clock plan requires its exact future filename")
    base = json.loads(RECIPE.read_text())
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    environment = {
        key: value.replace("/app/k31-agent-team-private/task-6", str(PRIVATE))
        for key, value in base["environment"].items()
        if not key.startswith(
            (
                "K31_",
                "ROCP",
                "HSA_TOOLS",
                "LD_PRELOAD",
                "ATOM_PROFILER",
                "AIPERF",
                "LMCACHE",
            )
        )
    }
    environment.update(
        PYTHONPATH=f"{HERE}:{ROOT}:{PRIVATE / 'aiter-572232d'}",
        K31_K2_FRONT_WORKER_GRANT=head,
        ATOM_MONO_COMPILE_INDEX=str(PRIVATE / "cache/mono-index"),
        HF_MODULES_CACHE=str(PRIVATE / "cache/hf-modules"),
        TIKTOKEN_CACHE_DIR=str(PRIVATE / "cache/tiktoken"),
    )
    paths = set((ROOT / "atom").rglob("*.py"))
    paths.update(path for path in HERE.iterdir() if path.is_file())
    paths.update(
        capture.TOOLS / (name + ".py")
        for name in (
            "front_floor_timing_v2",
            "front_floor_event_probe",
            "run_front_floor_v2",
        )
    )
    paths.update(
        (
            WS / "harness/common.py",
            RECIPE,
            Path("/opt/rocm/include/hip/hip_runtime_api.h"),
        )
    )
    paths.update(
        Path("/usr/lib/python3.12/multiprocessing") / name
        for name in (
            "util.py",
            "spawn.py",
            "popen_spawn_posix.py",
            "resource_tracker.py",
        )
    )
    paths.update(
        (
            WS
            / "agent-team/evidence/task-15-independent-clock-outcome-419aa118/sdk-simple_timer.cpp",
            WS
            / "agent-team/work/15/clock-boundary-audit/source__lib__rocprofiler-sdk-tool__tool.cpp",
        )
    )
    return {
        "schema": 1,
        "mode": CLOCK_MODE,
        "head": head,
        "base": base["base"],
        "root": str(ROOT),
        "output": str(out.resolve()),
        "environment": environment,
        "ports": [29675],
        "gpu_ids": list(range(TP)),
        "cpus": "0-111",
        "max_job_seconds": 180,
        "gpu_authorized_by_plan": False,
        "clock_protocol": clock_protocol(out.resolve(), plan_path.resolve()),
        "sources": {str(path): sha(path) for path in sorted(paths)},
        "execution": execution_binding(base, clock_only=True),
    }


def make_plan(mode, out, snapshot=None, ready=None, plan_path=None):
    if mode == CLOCK_MODE:
        return make_clock_plan(out, plan_path)
    initialize_helpers()
    base = json.loads(RECIPE.read_text())
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    old_private = "/app/k31-agent-team-private/task-6"
    env = {
        key: value.replace(old_private, str(PRIVATE)).replace(
            "/app/k31-opt-worktrees/agent-team-ATOM/6", str(ROOT)
        )
        for key, value in base["environment"].items()
    }
    env.update(
        PYTHONPATH=f"{HERE}:{ROOT}:{PRIVATE / 'aiter-572232d'}",
        CX_FRONT_FLOOR_EXPECT_ATOM_ROOT=str(ROOT),
        ATOM_MONO_COMPILE_INDEX=str(PRIVATE / "cache/mono-index"),
        HF_MODULES_CACHE=str(PRIVATE / "cache/hf-modules"),
        TIKTOKEN_CACHE_DIR=str(PRIVATE / "cache/tiktoken"),
        K31_K2_FRONT_WORKER_GRANT=head,
    )
    for key in list(env):
        if key.startswith("K31_") and key != "K31_K2_FRONT_WORKER_GRANT":
            del env[key]
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
    connector["kv_connector_extra_config"]["lmcache.mp.port"] = 5595
    argv[index] = json.dumps(connector)
    argv += ["--port", "8015"]
    cache = [
        x.replace("5586", "5595").replace("8686", "8695").replace("9186", "9195")
        for x in base["cache"]
    ]
    cache[cache.index("--instance-id") + 1] = out.name
    plan = {
        "schema": 1,
        "mode": mode,
        "head": head,
        "base": base["base"],
        "root": str(ROOT),
        "output": str(out.resolve()),
        "environment": env,
        "model_args": argv,
        "cache": cache,
        "ports": [8015, 5595, 8695, 9195, 29675],
        "gpu_ids": list(range(TP)),
        "cpus": "0-111",
        "max_job_seconds": 900 if mode == "capture" else 180,
        "request": {
            "prompt": capture.PROMPT,
            "prompt_sha256": hashlib.sha256(capture.PROMPT.encode()).hexdigest(),
            "min_position": 1,
            "expected_cached_tokens": 0,
            "max_tokens": 16,
            "temperature": 0,
            "ignore_eos": True,
        },
        "selection": {
            "layer": capture.LAYER,
            "requests": 1,
            "qlen": ROWS,
            "rows": ROWS,
            "nblocks": NB,
            "rule": "ordinary real served-Mono witness first, next matching invocation captured once",
            "observed_yet": False,
            "groups": 1,
        },
        "treatment": "Current8a7 Mono=1, TP8/DCP8, K3/AL2.88 and original precision/state controls. A single fixed real request is capture stimulus, not C12 or #519 performance.",
        "departures": [
            "current Mono implementation instead of historical Mono0",
            "single fixed operator-capture request instead of a timed serving workload",
            "one named capture graph and capture runner entry, excluded from timing",
            "private paths/ports/service identity",
        ],
        "sources": source_hashes(),
        "execution": execution_binding(base),
        "warmup": 2,
        "pairs_per_delay": 5,
        "delays": [1, 2],
        "noise_fraction": 0.02,
        "gpu_authorized_by_plan": False,
    }
    if mode == "timing":
        if snapshot is None or ready is None:
            raise ValueError(
                "timing requires complete captured inputs and CPU qualification"
            )
        receipt = json.loads(ready.read_text())
        if receipt["state"] != "SNAPSHOT_CPU_VALIDATED":
            raise ValueError("current actual snapshot has not passed CPU preparation")
        plan.update(
            snapshot=str(snapshot.resolve()),
            ready=str(ready.resolve()),
            ready_sha256=sha(ready),
        )
        plan["sources"].update(receipt["source_hashes"])
        plan["sources"][str(snapshot / "manifest.json")] = receipt["manifest_sha256"]
        plan["sources"].update(
            {
                str(snapshot / f"rank{rank}.pt"): value
                for rank, value in enumerate(receipt["rank_sha256"])
            }
        )
    return plan


def check_plan(plan, path, worker=False):
    if (
        plan["gpu_ids"] != list(range(TP))
        or plan["environment"].get("ATOM_MONO_ENABLE") != "1"
    ):
        raise RuntimeError(
            "only the declared whole-node current-Mono treatment is allowed"
        )
    if not Path(plan["output"]).resolve().is_relative_to(PRIVATE / "runs"):
        raise RuntimeError("output must be in the task's private run directory")
    if plan["mode"] not in ("capture", "timing", CLOCK_MODE) or plan[
        "max_job_seconds"
    ] != (900 if plan["mode"] == "capture" else 180):
        raise RuntimeError("unexpected mode or bound")
    if (
        plan["head"]
        != subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    ):
        raise RuntimeError("source head differs")
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("source is not clean")
    if worker and any(
        os.environ.get(name) != value
        for name, value in plan["environment"].items()
        if not (plan["mode"] == CLOCK_MODE and name == "LD_LIBRARY_PATH")
    ):
        raise RuntimeError("worker environment differs from the frozen plan")
    if plan["mode"] == CLOCK_MODE:
        if plan["clock_protocol"] != clock_protocol(
            Path(plan["output"]), path.resolve()
        ):
            raise RuntimeError("clock protocol differs from fixed source")
        if (
            any(key in plan for key in ("snapshot", "model_args", "cache", "ready"))
            or plan["execution"]["models"]
        ):
            raise RuntimeError("clock-only plan contains model/snapshot work")
    marker_observations = verify_binding(
        plan["execution"],
        (
            plan["clock_protocol"]["empty_marker_mtime_policy"]
            if plan["mode"] == CLOCK_MODE
            else None
        ),
    )
    for name, expected in plan["sources"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError("frozen source/input changed: " + name)
    grant = "K31_K2_FRONT_WORKER_GRANT" if worker else "K31_TASK15_GPU_GRANT"
    if os.environ.get(grant) != plan["head"]:
        raise RuntimeError("no exact root grant")
    if not worker and os.environ.get("K31_TASK15_PLAN_SHA") != sha(path):
        raise RuntimeError("no exact plan grant")
    if os.environ.get("KFD_GUARD_LOG") or "kfd-guard" in os.environ.get(
        "LD_PRELOAD", ""
    ):
        raise RuntimeError("GPU job cannot run under the CPU guard")
    return marker_observations


def clock_remaining():
    remaining = (int(os.environ["K31_CLOCK_DEADLINE_NS"]) - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("absolute clock-diagnostic deadline exceeded")
    return remaining


def clock_controls(row, body_start, body_end, body_ticks):
    module = capture.helper("front_floor_timing_v2")
    body_us = (body_end - body_start) * 1000 / row["wall_khz"]
    result = {
        **row,
        "body_start_tick": body_start,
        "body_end_tick": body_end,
        "body_us": body_us,
    }
    gate = module.trial_gate([{**row, "rank": 0}], world_size=1)
    result["local_submission_gate"] = gate
    result["original_event_gate"] = row["us"] + 5 >= body_us and row[
        "us"
    ] <= body_us + max(50, body_us * 0.10)
    if (
        body_end - body_start < body_ticks
        or body_start < row["guard_end_tick"]
        or not gate["valid"]
    ):
        return result, "clock content/order/submission prerequisite failed"
    return result, None


def clock_identity(device, plan, cohort):
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    address = ctypes.create_string_buffer(32)
    hip_status = hip.hipDeviceGetPCIBusId(address, len(address), device)
    hsa = ctypes.CDLL("/opt/rocm/lib/libhsa-runtime64.so")
    hsa.hsa_system_get_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    hsa.hsa_system_get_info.restype = ctypes.c_int
    frequency = ctypes.c_uint64()
    hsa_status = hsa.hsa_system_get_info(3, ctypes.byref(frequency))
    loaded = {
        str(Path(line.split(None, 5)[5].strip()).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and "librocprofiler-sdk" in line
    }
    return {
        "pid": os.getpid(),
        "native_tid": threading.get_native_id(),
        "device": device,
        "pci_bus_id": address.value.decode("ascii", errors="backslashreplace"),
        "hip_pci_status": hip_status,
        "hsa_frequency_status": hsa_status,
        "hsa_timestamp_frequency_hz": frequency.value,
        "loaded_profiler_libraries": sorted(loaded),
        "profiler_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("ROCP", "HSA_TOOLS"))
            or key in ("LD_PRELOAD", "LD_LIBRARY_PATH")
        },
    }


def validate_clock_identity(identity, plan, cohort):
    protocol = plan["clock_protocol"]
    if (
        identity["hip_pci_status"]
        or identity["pci_bus_id"].lower() not in protocol["pci_bdfs"]
    ):
        raise RuntimeError(
            "HIP PCI device identity unavailable or outside the fixed set"
        )
    if (
        identity["hsa_frequency_status"]
        or identity["hsa_timestamp_frequency_hz"]
        != protocol["hsa_timestamp_frequency_hz"]
    ):
        raise RuntimeError("HSA timebase differs from the qualified mapping")
    loaded = set(identity["loaded_profiler_libraries"])
    expected = set(protocol["profiler_loaded_libraries"])
    env = identity["profiler_environment"]
    libraries = lambda value: {str(Path(p).resolve()) for p in value.split(":") if p}
    if cohort == "traced":
        if (
            loaded != expected
            or libraries(env.get("LD_PRELOAD", ""))
            != set(protocol["profiler_preload_libraries"])
            or libraries(env.get("ROCP_TOOL_LIBRARIES", ""))
            != {protocol["profiler_tool_library"]}
        ):
            raise RuntimeError(
                "profiler loader mapping differs from the declared cohort"
            )
        for key, value in protocol["profiler_environment"].items():
            if env.get(key) != value:
                raise RuntimeError("strict profiler timestamp environment differs")
        for key in (
            "ROCPROF_HIP_RUNTIME_API_TRACE",
            "ROCPROF_HIP_COMPILER_API_TRACE",
            "ROCPROF_KERNEL_TRACE",
        ):
            if env.get(key) != "1":
                raise RuntimeError("required profiler trace domain disabled")
        if any(
            value not in ("0", "")
            for key, value in env.items()
            if key.endswith("_TRACE")
            and key
            not in (
                "ROCPROF_HIP_RUNTIME_API_TRACE",
                "ROCPROF_HIP_COMPILER_API_TRACE",
                "ROCPROF_KERNEL_TRACE",
            )
        ):
            raise RuntimeError("unexpected profiler trace domain")
    elif loaded or any(value for key, value in env.items() if key != "LD_LIBRARY_PATH"):
        raise RuntimeError("untraced cohort inherited profiling settings")
    if libraries(env.get("LD_LIBRARY_PATH", "")) != libraries(
        plan["environment"]["LD_LIBRARY_PATH"]
    ):
        raise RuntimeError("runtime library search path differs")


def clock_mapping(records, plan, expected=None):
    actual = [record["pci_bus_id"].lower() for record in records]
    if (
        len(actual) != TP
        or set(actual) != set(plan["clock_protocol"]["pci_bdfs"])
        or (expected is not None and actual != expected)
    ):
        raise RuntimeError("clock cohort ordinal/BDF bijection differs")
    return actual


def clock_child_record(table, role, descriptors, argv, passfds):
    expected_keys = (
        {"tracker_fd", "pipe_handle"} if role == "worker" else {"tracker_fd"}
    )
    if role not in table["children"] or set(descriptors) != expected_keys:
        raise RuntimeError("unknown clock child role/descriptors")
    for fd in [*descriptors.values(), *passfds]:
        if trace_uint(fd, True) >= 2**31:
            raise RuntimeError("invalid child descriptor")
    if not set(descriptors.values()) <= set(passfds):
        raise RuntimeError("clock child descriptors not passed")
    rule = table["children"][role]
    expected = [
        table["python"],
        *table["python_flags"],
        "-c",
        rule["program"].format(**descriptors),
        *rule["suffix"],
    ]
    argv = list(map(os.fsdecode, argv))
    if argv != expected:
        raise RuntimeError("unqualified clock child argv")
    return {
        "role": role,
        "caller": rule["caller"],
        "descriptors": descriptors,
        "passfds": list(passfds),
        "argv": argv,
        "label": "[rocprofv3] '" + " ".join(argv) + "'",
    }


def clock_timer_labels(plan, complete=False):
    table = plan["clock_protocol"]["normal_timer_messages"]
    records = load_clock_json(Path(plan["output"]) / "data/traced/command-ledger.json")
    counts = {role: 0 for role in table["children"]}
    labels = [
        *table["lifecycle_labels"],
        *("[rocprofv3] '" + " ".join(cmd) + "'" for cmd in table["fixed_commands"]),
    ]
    if type(records) is not list:
        raise RuntimeError("invalid clock command ledger")
    for record in records:
        checked = clock_child_record(
            table,
            record["role"],
            record["descriptors"],
            record["argv"],
            record["passfds"],
        )
        if checked != record:
            raise RuntimeError("clock command ledger identity differs")
        counts[record["role"]] += 1
        labels.append(record["label"])
    for role, count in counts.items():
        maximum = table["children"][role]["max"]
        if count > maximum or complete and count != maximum:
            raise RuntimeError("clock child launch count differs")
    return labels


def clock_spawn(plan, mp, worker_plan):
    from multiprocessing import popen_spawn_posix, resource_tracker, util

    table = plan["clock_protocol"]["normal_timer_messages"]
    path = Path(plan["output"]) / "data/traced/command-ledger.json"
    if (
        util._args_from_interpreter_flags() != table["python_flags"]
        or load_clock_json(path) != []
    ):
        raise RuntimeError("clock spawn flags/initial ledger differs")
    original, records = util.spawnv_passfds, []
    worker_code = popen_spawn_posix.Popen._launch.__code__
    tracker_code = resource_tracker.ResourceTracker.ensure_running.__code__

    def launch(executable, argv, passfds):
        frame = sys._getframe(1)
        local = frame.f_locals
        if frame.f_code is worker_code:
            role, descriptors = "worker", {
                "tracker_fd": local["tracker_fd"],
                "pipe_handle": local["child_r"],
            }
            expected_fds = local["self"]._fds
        elif frame.f_code is tracker_code:
            role, descriptors = "tracker", {"tracker_fd": local["r"]}
            expected_fds = local["fds_to_pass"]
        else:
            raise RuntimeError("unknown clock spawn caller")
        if os.fsdecode(executable) != table["python"] or list(passfds) != list(
            expected_fds
        ):
            raise RuntimeError("clock spawn executable/passfds differs")
        record = clock_child_record(table, role, descriptors, argv, passfds)
        records.append(record)
        if (
            sum(item["role"] == role for item in records)
            > table["children"][role]["max"]
        ):
            raise RuntimeError("extra clock child launch")
        temporary = path.with_suffix(".tmp")
        save(temporary, records)
        temporary.replace(path)
        return original(executable, argv, passfds)

    util.spawnv_passfds = launch
    try:
        mp.spawn(clock_rank_main, args=(worker_plan, "traced"), nprocs=TP, join=True)
    finally:
        changed = util.spawnv_passfds is not launch
        util.spawnv_passfds = original
        if changed or util.spawnv_passfds is not original:
            raise RuntimeError("clock spawn hook restoration failed")
    clock_timer_labels(plan, complete=True)


def clock_live_check(plan, cohort, record=False):
    clock_remaining()
    if cohort == "traced":
        directory = Path(plan["output"]) / "data" / cohort
        clock_trace_logs(
            directory / "worker.log",
            list((directory / "profiler").glob("clock-*_results.json")),
            live=True,
            timer_labels=clock_timer_labels(plan),
            classification=directory / "log-classification.json" if record else None,
        )
    clock_remaining()


def clock_observations(timer, graph, body, report, plan, cohort, dist, out):
    for ordinal in range(1, 5):
        error = None
        try:
            clock_live_check(plan, cohort)
        except Exception as failure:  # noqa: BLE001
            error = repr(failure)
        errors = [None] * TP
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("clock pre-observation failure: " + repr(errors))
        record = {
            "body_replay_ordinal": ordinal,
            "delay_scale": 1 if ordinal <= 2 else 2,
            "trial": (ordinal - 1) % 2,
        }
        try:
            row = timer.measure(graph, record["delay_scale"])
            start, end = map(int, body.cpu().tolist())
            checked, problem = clock_controls(row, start, end, report["body_ticks"])
            record.update(checked)
            if problem:
                raise RuntimeError(problem)
            clock_live_check(plan, cohort)
        except Exception as failure:  # noqa: BLE001
            error = repr(failure)
            record["hard_failure"] = error
            record = {
                key: (
                    repr(value)
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in record.items()
            }
        report["rows"].append(record)
        save(out / f"rank{report['rank']}.json", report)
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(
                "clock cohort stopped after hard failure: " + repr(errors)
            )


def clock_rank_main(rank, plan, cohort):
    import torch
    import torch.distributed as dist

    clock_remaining()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)
    if torch.cuda.device_count() != TP:
        raise RuntimeError("clock diagnostic requires all eight existing mappings")
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:29675",
        rank=rank,
        world_size=TP,
        timeout=timedelta(seconds=min(30, clock_remaining())),
    )
    sys.path.append(str(capture.TOOLS))
    module = capture.helper("front_floor_timing_v2")
    out = Path(plan["output"]) / "data" / cohort
    report = {
        "state": "STARTED",
        "scope": CLOCK_MODE,
        "rank": rank,
        "cohort": cohort,
        "rows": [],
    }
    stream = torch.cuda.Stream(device=torch.device("cuda", rank))
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    try:
        report.update(clock_identity(rank, plan, cohort))
        validate_clock_identity(report, plan, cohort)
        identities = [None] * TP
        dist.all_gather_object(identities, report)
        expected = None
        if cohort == "traced":
            expected = json.loads(
                (Path(plan["output"]) / "data/untraced/mapping.json").read_text()
            )
        clock_mapping(identities, plan, expected)
        clock_live_check(plan, cohort)
        timer = module.Timer(rank, stream, rank)
        body = torch.empty_like(timer.guard)
        ticks = max(1, math.ceil(timer.rate * 0.5))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            timer.run(timer.delay, body.data_ptr(), ticks, stream)
        graph.debug_dump(str(out / f"clock-rank{rank}.dot"))
        report["graph"] = module.verify_single_kernel_graph(graph)
        graph.instantiate()
        report.update(
            graph_exec=int(graph.raw_cuda_graph_exec()),
            stream_handle=int(stream.cuda_stream),
            body_ticks=ticks,
            wall_khz=timer.rate,
            graph_dot_sha256=sha(out / f"clock-rank{rank}.dot"),
        )
        graph.replay()
        stream.synchronize()
        report["body_warmup_ordinal"] = 0
        dist.barrier()
        clock_observations(timer, graph, body, report, plan, cohort, dist, out)
        report["state"] = "COMPLETE"
        report["original_event_gate_failures"] = sum(
            not row["original_event_gate"] for row in report["rows"]
        )
        report["meaning"] = (
            "Fixed observation set complete; failed old event gates stay failed and never authorize K2."
        )
        clock_remaining()
    except BaseException as failure:
        report.update(state="FAILED", error=repr(failure))
        raise
    finally:
        save(out / f"rank{rank}.json", report)
        dist.destroy_process_group()


def trace_uint(value, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value < 2**64:
        raise ValueError("invalid integral trace identity/timestamp")
    return value


def trace_pointer(value):
    if type(value) is not str or not re.fullmatch(r"0|0x[0-9a-f]+", value):
        raise ValueError("unsupported raw pointer representation")
    return trace_uint(int(value, 16), True)


def clock_trace_logs(
    path, allowed_outputs=(), live=False, timer_labels=(), classification=None
):
    text = Path(path).read_text(errors="strict")
    lines = text.splitlines()
    if live and text and not text.endswith("\n"):
        lines = lines[:-1]
    result = {"state": "CHECKED", "complete_lines": len(lines), "classifications": []}
    try:
        for number, raw in enumerate(lines, 1):
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            if re.search(
                r"dropping record|dropped.{0,20}record|lost.{0,20}record|swapping the values|cannot offload buffer|cannot read tmp file|buffer is not empty after offload",
                line,
                re.IGNORECASE,
            ):
                raise RuntimeError("profiler loss/repair diagnostic: " + line)
            normal = re.fullmatch(
                r"W[0-9]{8} [0-2][0-9]:[0-5][0-9]:[0-5][0-9]\.[0-9]{6} ([1-9][0-9]*) simple_timer\.cpp:55\] (.+) :: ( *[0-9]+\.[0-9]{6}) sec",
                raw,
            )
            if normal and normal[2] in timer_labels:
                datetime.strptime(raw[1:25], "%Y%m%d %H:%M:%S.%f")
                seconds = float(normal[3])
                trace_uint(int(normal[1]))
                if (
                    math.isfinite(seconds)
                    and seconds >= 0
                    and f"{seconds:12.6f}" == normal[3]
                ):
                    result["classifications"].append(
                        {
                            "line": number,
                            "raw": raw,
                            "class": "source_normal_timer_summary",
                            "label": normal[2],
                            "seconds": seconds,
                        }
                    )
                    continue
            if re.match(r"[WEF][0-9]{4,8}\s", line) or re.search(
                r"(?<![A-Za-z])(?:WARNING|ERROR|FATAL)(?![A-Za-z])", line, re.IGNORECASE
            ):
                if re.match(
                    r"E[0-9]{4,8} .*output_stream\.cpp:[0-9]+\] Opened result file: ",
                    line,
                ) and any(
                    line.endswith(
                        (
                            "Opened result file: " + str(p),
                            'Opened result file: "' + str(p) + '"',
                        )
                    )
                    for p in allowed_outputs
                ):
                    result["classifications"].append(
                        {
                            "line": number,
                            "raw": raw,
                            "class": "expected_result_file_notice",
                        }
                    )
                    continue
                raise RuntimeError("unqualified profiler diagnostic: " + line)
    except Exception as error:
        result.update(state="REFUSED", line=number, error=repr(error))
        raise
    finally:
        if classification is not None:
            save(classification, result)
    return result


def trace_operation(document, record):
    kind, operation = trace_uint(record["kind"], True), trace_uint(
        record["operation"], True
    )
    names = document["strings"]["buffer_records"]
    return names[kind]["kind"], names[kind]["operations"][operation]


def trace_args(record):
    entries = record["args"]
    if type(entries) is not list or len({x["name"] for x in entries}) != len(entries):
        raise ValueError("missing/duplicate API arguments")
    if any(type(x[k]) is not str for x in entries for k in ("type", "name", "value")):
        raise ValueError("invalid API argument schema")
    return {x["name"]: x["value"] for x in entries}


def trace_inventory(document, report):
    if trace_uint(document["metadata"]["pid"]) != report["pid"]:
        raise ValueError("trace PID differs")
    apis = document["buffer_records"]["hip_api"]
    kernels = document["buffer_records"]["kernel_dispatch"]
    api_ids, dispatch_ids = [], []
    for record in [*apis, *kernels]:
        start = trace_uint(record["start_timestamp"])
        end = trace_uint(record["end_timestamp"])
        if end < start:
            raise ValueError("reversed trace timestamp")
        trace_uint(record["thread_id"])
        trace_uint(record["correlation_id"]["internal"])
        trace_uint(record["stream_id"]["handle"], True)
    for record in apis:
        api_ids.append(record["correlation_id"]["internal"])
    for record in kernels:
        dispatch_ids.append(trace_uint(record["dispatch_info"]["dispatch_id"]))
    if len(set(api_ids)) != len(api_ids) or len(set(dispatch_ids)) != len(dispatch_ids):
        raise ValueError("duplicate API/dispatch identity")
    graph_calls, module_calls = [], []
    for api in apis:
        kind, operation = trace_operation(document, api)
        if operation.startswith(("hipGraphLaunch", "hipModuleLaunchKernel")):
            if kind != "HIP_RUNTIME_API_EXT" or operation not in (
                "hipGraphLaunch",
                "hipModuleLaunchKernel",
            ):
                raise ValueError("unexpected required API domain/variant")
            args = trace_args(api)
            if (
                api["thread_id"] != report["native_tid"]
                or trace_uint(api["retval"]["uint64_t_retval"], True) != 0
            ):
                raise ValueError("required API thread/result differs")
            if trace_pointer(args["stream"]) != trace_uint(report["stream_handle"]):
                raise ValueError("required raw stream handle differs")
            if operation == "hipGraphLaunch":
                if set(args) != {"graphExec", "stream"} or trace_pointer(
                    args["graphExec"]
                ) != trace_uint(report["graph_exec"]):
                    raise ValueError("graph handle/argument schema differs")
                graph_calls.append(api)
            else:
                required = {
                    "func",
                    "gridDimX",
                    "gridDimY",
                    "gridDimZ",
                    "blockDimX",
                    "blockDimY",
                    "blockDimZ",
                    "sharedMemBytes",
                    "stream",
                    "kernelParams",
                    "extra",
                }
                if set(args) != required or not trace_pointer(args["func"]):
                    raise ValueError("module function/argument schema differs")
                for name in (
                    "gridDimX",
                    "gridDimY",
                    "gridDimZ",
                    "blockDimY",
                    "blockDimZ",
                ):
                    if args[name] != "1":
                        raise ValueError("module launch geometry differs")
                if args["blockDimX"] != "64" or args["sharedMemBytes"] != "0":
                    raise ValueError("module launch block/shared memory differs")
                module_calls.append(api)
    relevant = []
    for dispatch in kernels:
        info = dispatch["dispatch_info"]
        kid = trace_uint(info["kernel_id"])
        symbol = document["kernel_symbols"][kid]
        if trace_uint(symbol["kernel_id"]) != kid or not symbol["kernel_name"]:
            raise ValueError("kernel symbol entity differs")
        if "delay_0" in (symbol["kernel_name"], symbol["formatted_kernel_name"]):
            if (
                symbol["kernel_name"] != "delay_0"
                or symbol["formatted_kernel_name"] != "delay_0"
            ):
                raise ValueError("clock raw/formatted kernel symbol differs")
            if (
                trace_operation(document, dispatch)
                != ("KERNEL_DISPATCH", "KERNEL_DISPATCH_COMPLETE")
                or dispatch["operation"] != 2
            ):
                raise ValueError("not a completed kernel dispatch")
            if (
                dispatch["thread_id"] != report["native_tid"]
                or dispatch["end_timestamp"] <= dispatch["start_timestamp"]
            ):
                raise ValueError("clock thread/timestamp differs")
            for geometry in ("workgroup_size", "grid_size"):
                for dimension in ("x", "y", "z"):
                    trace_uint(info[geometry][dimension])
            if info["workgroup_size"] != {"x": 64, "y": 1, "z": 1} or info[
                "grid_size"
            ] != {"x": 64, "y": 1, "z": 1}:
                raise ValueError("clock dispatch geometry differs")
            agent_id = trace_uint(info["agent_id"]["handle"])
            agents = [
                agent
                for agent in document["agents"]
                if agent["id"]["handle"] == agent_id
            ]
            if len(agents) != 1:
                raise ValueError("unknown/duplicate dispatch agent")
            agent = agents[0]
            location = trace_uint(agent["location_id"], True)
            domain = trace_uint(agent["domain"], True)
            bdf = f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7:x}"
            if (
                location > 65535
                or domain > 65535
                or bdf != report["pci_bus_id"].lower()
            ):
                raise ValueError("dispatch physical device differs")
            trace_uint(info["queue_id"]["handle"])
            relevant.append(dispatch)
    if len(graph_calls) != 5 or len(module_calls) != 6 or len(relevant) != 10:
        raise ValueError("required five-body/five-guard/six-module ledger incomplete")
    if (
        len({kernel["dispatch_info"]["queue_id"]["handle"] for kernel in relevant}) != 1
        or len({kernel["dispatch_info"]["kernel_id"] for kernel in relevant}) != 1
    ):
        raise ValueError("clock dispatches changed queue/kernel")
    if len({trace_pointer(trace_args(api)["func"]) for api in module_calls}) != 1:
        raise ValueError("module function identity changed")
    if [row["body_replay_ordinal"] for row in report["rows"]] != [1, 2, 3, 4] or report[
        "body_warmup_ordinal"
    ] != 0:
        raise ValueError("worker body ordinal ledger differs")
    cid = lambda record: record["correlation_id"]["internal"]
    graphs, modules = sorted(graph_calls, key=cid), sorted(module_calls, key=cid)
    ledger = [modules[0], modules[1], graphs[0]]
    for guard, body in zip(modules[2:], graphs[1:]):
        ledger.extend((guard, body))
    if ledger != sorted(ledger, key=cid) or any(
        a["end_timestamp"] > b["start_timestamp"] for a, b in pairwise(ledger)
    ):
        raise ValueError("required API call ordinal/order differs")
    matched = {}
    for api in ledger:
        children = [kernel for kernel in kernels if cid(kernel) == cid(api)]
        if api is modules[1]:
            if children:
                raise ValueError("capture-definition API unexpectedly dispatched")
            continue
        if len(children) != 1 or children[0] not in relevant:
            raise ValueError("missing/duplicate/extra required correlation dispatch")
        kernel = children[0]
        if api["stream_id"]["handle"] != kernel["stream_id"]["handle"]:
            raise ValueError("API/dispatch SDK stream differs")
        matched[cid(api)] = kernel
    ordered = list(matched.values())
    if len({cid(kernel) for kernel in relevant}) != 10 or ordered != sorted(
        relevant, key=lambda k: k["dispatch_info"]["dispatch_id"]
    ):
        raise ValueError("clock dispatch correlation/order differs")
    return graphs, modules, matched, relevant


def load_clock_json(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate trace JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite trace JSON value: " + value)

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant
    )


def validate_clock_trace(plan, reports, directory):
    """Check complete required lineage; row B/K/E analysis stays after release."""
    by_pid = {trace_uint(row["pid"]): row for row in reports}
    if len(by_pid) != TP:
        raise ValueError("missing/duplicate traced worker PID")
    documents, seen = {}, set()
    files = sorted((Path(directory) / "profiler").glob("*_results.json"))
    for path in files:
        if path.stat().st_size > 128 << 20:
            raise ValueError("unexpected trace export size")
        processes = load_clock_json(path)["rocprofiler-sdk-tool"]
        if type(processes) is not list or len(processes) != 1:
            raise ValueError("unexpected trace process schema")
        document = processes[0]
        pid = trace_uint(document["metadata"]["pid"])
        if path.name != f"clock-{pid}_results.json" or pid in seen:
            raise ValueError("trace output/PID identity differs")
        seen.add(pid)
        if pid in by_pid:
            validate_clock_identity(by_pid[pid], plan, "traced")
            trace_inventory(document, by_pid[pid])
            documents[pid] = document
        elif document["buffer_records"]["kernel_dispatch"]:
            raise ValueError("unmodelled GPU work in a non-worker trace")
    clock_trace_logs(
        Path(directory) / "worker.log",
        files,
        timer_labels=clock_timer_labels(plan, complete=True),
        classification=Path(directory) / "log-classification.json",
    )
    if set(documents) != set(by_pid):
        raise ValueError("missing worker trace export")
    return documents


def join_clock_trace(document, report):
    graphs, modules, matched, relevant = trace_inventory(document, report)
    rows = []
    for sample, api in zip(report["rows"], graphs[1:]):
        dispatch = matched[api["correlation_id"]["internal"]]
        k_ns = dispatch["end_timestamp"] - dispatch["start_timestamp"]
        k_us = k_ns / 1000
        rows.append(
            {
                **sample,
                "graph_api": api,
                "dispatch": dispatch,
                "K_ns": k_ns,
                "K_us": k_us,
                "K_minus_B_us": k_us - sample["body_us"],
                "E_minus_K_us": sample["us"] - k_us,
                "nested_durations": sample["body_us"] <= k_us <= sample["us"],
            }
        )
    return {
        "rows": rows,
        "warmup_dispatch": matched[graphs[0]["correlation_id"]["internal"]],
        "capture_definition_api": modules[1],
        "guard_dispatches": [
            matched[api["correlation_id"]["internal"]]
            for api in [modules[0], *modules[2:]]
        ],
        "unmodelled_dispatches": [
            kernel
            for kernel in document["buffer_records"]["kernel_dispatch"]
            if kernel not in relevant
        ],
        "completeness_scope": "Independent required clock lineage and visible loss diagnostics; no whole-export zero-loss certificate.",
    }


def analyze_clock(plan, path, target):
    out = Path(plan["output"])
    contract = load_clock_json(out / "contract.json")
    if (
        contract.get("mode") != CLOCK_MODE
        or contract.get("head") != plan["head"]
        or contract.get("clock_plan_sha256") != sha(path)
        or contract.get("state") != "COMPLETE"
        or contract.get("cleanup_status") != "released"
        or not contract.get("gpu_released")
        or not contract.get("stopped")
        or contract.get("alive") != []
        or contract["after"].get("owned_identity_or_group_unknown")
        or not whole_node_idle(contract["after"])
    ):
        raise RuntimeError(
            "offline analysis requires exact-plan completed owned release"
        )
    if contract["workload_elapsed_seconds"] >= 180:
        raise RuntimeError("late set cannot be analyzed as completed")
    records = [load_clock_json(out / "data" / f"rank{rank}.json") for rank in range(TP)]
    untraced = [record["cohorts"]["untraced"] for record in records]
    traced = [record["cohorts"]["traced"] for record in records]
    clock_mapping(traced, plan, clock_mapping(untraced, plan))
    documents = validate_clock_trace(plan, traced, out / "data/traced")
    result = {
        "state": "CLOCK_BOUNDARIES_JOINED_AFTER_RELEASE",
        "plan_sha256": sha(path),
        "contract_sha256": sha(out / "contract.json"),
        "inputs": {
            str(p.relative_to(out)): sha(p)
            for p in sorted((out / "data").rglob("*"))
            if p.is_file()
        },
        "untraced": untraced,
        "traced": [
            join_clock_trace(documents[record["pid"]], record) for record in traced
        ],
        "interpretation": plan["clock_protocol"]["interpretation"],
        "K2_executed": False,
        "next": "Seal/review/stop; no method adoption or automatic continuation.",
    }
    with Path(target).open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    return result


def clock_driver(plan, path):
    """Run the two fixed cohorts; full B/K/E joining is an offline operation."""
    out = Path(plan["output"]) / "data"
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for cohort in ("untraced", "traced"):
        clock_remaining()
        directory = out / cohort
        directory.mkdir()
        command = [
            "/opt/venv/bin/python",
            str(HERE / "replay.py"),
            "--clock-cohort",
            cohort,
            "--plan",
            str(path),
        ]
        if cohort == "traced":
            save(directory / "command-ledger.json", [])
            command = [*plan["clock_protocol"]["profiler_command"], "--", *command]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ROCP", "HSA_TOOLS")) and key != "LD_PRELOAD"
        }
        if cohort == "traced":
            environment.update(plan["clock_protocol"]["profiler_environment"])
        clock_remaining()
        with (directory / "worker.log").open("x") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
            save(
                directory / "launch.json",
                {
                    "pid": process.pid,
                    "command": command,
                    "started_ns": time.monotonic_ns(),
                },
            )
            while True:
                clock_live_check(plan, cohort, record=True)
                if process.poll() is not None:
                    break
                time.sleep(min(0.1, clock_remaining()))
            clock_remaining()
            if process.returncode:
                raise RuntimeError(f"{cohort} diagnostic exited {process.returncode}")
        records = [
            json.loads((directory / f"rank{rank}.json").read_text())
            for rank in range(TP)
        ]
        if any(
            r["state"] != "COMPLETE" or r["rank"] != rank or len(r["rows"]) != 4
            for rank, r in enumerate(records)
        ):
            raise RuntimeError("incomplete clock observation set")
        for record in records:
            validate_clock_identity(record, plan, cohort)
        expected = (
            None
            if cohort == "untraced"
            else [record["pci_bus_id"].lower() for record in reports["untraced"]]
        )
        mapping = clock_mapping(records, plan, expected)
        save(directory / "mapping.json", mapping)
        reports[cohort] = records
        if cohort == "traced":
            # Strict schema/loss checks run here; same-invocation joining stays offline.
            validate_clock_trace(plan, records, directory)
        clock_remaining()
    for rank in range(TP):
        save(
            out / f"rank{rank}.json",
            {
                "state": "COMPLETE",
                "rank": rank,
                "scope": CLOCK_MODE,
                "cohorts": {name: records[rank] for name, records in reports.items()},
            },
        )
    clock_remaining()
    save(
        out / "clock-set.json",
        {
            "state": "OBSERVATION_SET_COMPLETE",
            "rows": 64,
            "K2_executed": False,
            "elapsed_seconds": (
                time.monotonic_ns() - int(os.environ["K31_CLOCK_STARTED_NS"])
            )
            / 1e9,
        },
    )
    clock_remaining()


def whole_node_idle(snapshot):
    if type(snapshot) is not dict:
        return False
    cards = snapshot.get("cards")
    if snapshot.get("kfd_pids") != [] or type(cards) is not dict:
        return False
    if set(cards) != {f"card{rank}" for rank in range(TP)}:
        return False
    try:
        return all(
            type(card[name]) in (int, float)
            and math.isfinite(card[name])
            and card[name] >= 0
            and (name != "used_gib" or card[name] < 2)
            for card in cards.values()
            for name in ("used_gib", "free_gib")
        ) and all(card["used_gib"] + card["free_gib"] > 0 for card in cards.values())
    except (KeyError, TypeError):
        return False


def ownership_boundary(common, namespace):
    from types import SimpleNamespace

    import psutil

    proxy = SimpleNamespace(**vars(common))

    def remember():
        live = {}
        namespace["identity_error"] = None
        for pid, birth in list(namespace["seen"].items()):
            try:
                proc = psutil.Process(pid)
                if (
                    birth is not None
                    and proc.create_time() == birth
                    and proc.status() != psutil.STATUS_ZOMBIE
                ):
                    live[pid] = proc
                    for child in proc.children(recursive=True):
                        namespace["seen"].setdefault(child.pid, child.create_time())
                        if (
                            child.create_time() == namespace["seen"][child.pid]
                            and child.status() != psutil.STATUS_ZOMBIE
                        ):
                            live[child.pid] = child
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as error:
                namespace["identity_error"] = repr(error)
        return list(live.values())

    def unknown_group():
        known = {(p.pid, p.create_time()) for p in remember()}
        if namespace["identity_error"] or any(
            v is None for v in namespace["seen"].values()
        ):
            return True
        try:
            for proc in psutil.process_iter():
                try:
                    if (
                        os.getpgid(proc.pid) == namespace["proc"].pid
                        and proc.status() != psutil.STATUS_ZOMBIE
                        and (proc.pid, proc.create_time()) not in known
                    ):
                        return True
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass
        except (PermissionError, psutil.AccessDenied):
            return True
        return False

    def spawn(*args, **kwargs):
        proc = common.spawn(*args, **kwargs)
        try:
            namespace["seen"][proc.pid] = psutil.Process(proc.pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            namespace["seen"][proc.pid] = None
        return proc

    def signal_owned(proc, sig):
        try:
            proc.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            namespace["identity_error"] = repr(error)

    def stop(_proc, _name, _log, term_timeout):
        for sig, seconds in ((15, term_timeout), (9, 30)):
            for proc in remember():
                signal_owned(proc, sig)
            until = namespace["time"].monotonic() + seconds
            while remember() and namespace["time"].monotonic() < until:
                namespace["time"].sleep(0.5)

    def idle():
        snapshot = common.assert_gpus_idle()
        if not whole_node_idle(snapshot):
            raise RuntimeError("whole-node KFD/all-eight-card admission unavailable")
        return snapshot

    def released(timeout):
        done, snapshot = common.wait_gpus_released(timeout=timeout)
        unknown = unknown_group()
        return done and whole_node_idle(snapshot) and not unknown, {
            **snapshot,
            "owned_identity_or_group_unknown": unknown,
        }

    proxy.spawn, proxy.stop_tree = spawn, stop
    proxy.assert_gpus_idle, proxy.wait_gpus_released = idle, released
    namespace["owned_remember"], namespace["signal_owned"] = remember, signal_owned
    return proxy


def supervise(plan, path):
    """Reuse the pinned V2 ownership body; only plan plumbing is adapted."""
    marker_observations = check_plan(plan, path)
    if Path(plan["output"]).exists():
        raise FileExistsError(plan["output"])
    if os.nice(0) != 0:
        raise RuntimeError("GPU supervision requires normal priority")
    from types import SimpleNamespace

    original = capture.TOOLS / "run_front_floor_v2.py"
    tree = ast.parse(original.read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    start = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Import) and n.names[0].name == "psutil"
    )
    body = main.body[start:]
    changes = {
        "environment": 0,
        "ports": 0,
        "grant": 0,
        "remember": 0,
        "signal": 0,
        "clock_start": 0,
        "clock_poll": 0,
        "clock_end": 0,
    }

    class Plumbing(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "remember":
                node.body = ast.parse("return owned_remember()").body
                changes["remember"] += 1
                return node
            return self.generic_visit(node)

        def visit_If(self, node):
            if (
                plan["mode"] == CLOCK_MODE
                and ast.unparse(node.test) == "proc.poll() is not None"
            ):
                changes["clock_poll"] += 1
                return [
                    *ast.parse("check_absolute_clock()").body,
                    self.generic_visit(node),
                ]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if plan["mode"] == CLOCK_MODE:
                target = ast.unparse(node.targets[0])
                if (
                    target == "proc"
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "C.spawn"
                ):
                    changes["clock_start"] += 1
                    return [
                        *ast.parse("start_absolute_clock(env)").body,
                        self.generic_visit(node),
                    ]
                if (
                    target == "state['state']"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == "COMPLETE"
                ):
                    changes["clock_end"] += 1
                    return [
                        *ast.parse("check_absolute_clock()").body,
                        self.generic_visit(node),
                    ]
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "env":
                node.value = ast.parse('plan["environment"].copy()', mode="eval").body
                changes["environment"] += 1
            if (
                isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "env"
                and isinstance(node.targets[0].slice, ast.Name)
                and node.targets[0].slice.id == "grant_env"
            ):
                node.value = ast.parse('plan["head"]', mode="eval").body
                changes["grant"] += 1
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "send_signal":
                changes["signal"] += 1
                return ast.Call(
                    func=ast.Name(id="signal_owned", ctx=ast.Load()),
                    args=[node.func.value, *node.args],
                    keywords=[],
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "assert_ports_free"
            ):
                node.args = [ast.parse('plan["ports"]', mode="eval").body]
                changes["ports"] += 1
            return self.generic_visit(node)

    module = ast.Module(body=body, type_ignores=[])
    module = ast.fix_missing_locations(Plumbing().visit(module))
    if changes != {
        "environment": 1,
        "ports": 1,
        "grant": 1,
        "remember": 1,
        "signal": 1,
        "clock_start": int(plan["mode"] == CLOCK_MODE),
        "clock_poll": int(plan["mode"] == CLOCK_MODE),
        "clock_end": int(plan["mode"] == CLOCK_MODE),
    }:
        raise RuntimeError("pinned supervisor shape changed")
    sys.path.insert(0, str(WS / "harness"))
    import common as C

    script = HERE / ("capture.py" if plan["mode"] == "capture" else "replay.py")
    command = [C.PYTHON, str(script), "--worker", "--plan", str(path)]
    env = dict(plan["environment"], K31_K2_FRONT_PLAN=str(path))
    actual = dict(plan, environment=env, command=command)
    if plan["mode"] == CLOCK_MODE:
        actual["clock_plan_sha256"] = sha(path)
        actual["clock_marker_observations"] = marker_observations
    namespace = {
        "C": C,
        "os": os,
        "sys": sys,
        "json": json,
        "Path": Path,
        "signal": __import__("signal"),
        "time": __import__("time"),
        "args": SimpleNamespace(probe=False),
        "out": Path(plan["output"]),
        "plan": actual,
        "command": command,
        "cpus": "0-111",
        "bound": plan["max_job_seconds"],
        "grant_env": "K31_K2_FRONT_WORKER_GRANT",
        "WS": WS,
        "HERE": HERE,
        "TREE": ROOT,
    }

    def start_absolute_clock(env):
        started = time.monotonic_ns()
        deadline = started + plan["max_job_seconds"] * 10**9
        namespace["clock_started_ns"], namespace["clock_deadline_ns"] = (
            started,
            deadline,
        )
        env.update(
            K31_CLOCK_STARTED_NS=str(started), K31_CLOCK_DEADLINE_NS=str(deadline)
        )
        namespace["state"].update(
            workload_started_ns=started, absolute_deadline_ns=deadline
        )

    def check_absolute_clock():
        now = time.monotonic_ns()
        namespace["state"]["workload_elapsed_seconds"] = (
            now - namespace["clock_started_ns"]
        ) / 1e9
        if now >= namespace["clock_deadline_ns"]:
            raise TimeoutError(
                "absolute diagnostic bound exceeded; late completion is failure"
            )

    namespace.update(
        start_absolute_clock=start_absolute_clock,
        check_absolute_clock=check_absolute_clock,
    )
    namespace["C"] = ownership_boundary(C, namespace)
    # Only the source-hash-bound local supervisor AST is executable here.
    exec(compile(module, str(original), "exec"), namespace)  # noqa: S102


def ownership_check():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    sys.path.insert(0, str(WS / "harness"))
    import common as original_common

    class Held(BaseException):
        pass

    assert not whole_node_idle(None)
    good = {
        "kfd_pids": [],
        "cards": {
            f"card{rank}": {"used_gib": 0.277, "free_gib": 287.707} for rank in range(8)
        },
    }
    cases = {
        "valid": good,
        "missing_card": {**good, "cards": dict(list(good["cards"].items())[:-1])},
        "global_kfd": {**good, "kfd_pids": [987654321]},
        "kfd_error": {**good, "kfd_pids": ["OSError: unavailable"]},
        "pid_reuse": good,
        "malformed_cards": {**good, "cards": {"card0": None}},
        "nonfinite_memory": {
            **good,
            "cards": {
                **good["cards"],
                "card0": {"used_gib": float("nan"), "free_gib": 1},
            },
        },
    }
    outcomes = {}
    for case, snapshot in cases.items():
        with TemporaryDirectory() as directory:
            out = Path(directory) / "result"
            fd = os.open(Path(directory) / "lock", os.O_CREAT | os.O_RDWR, 0o600)
            calls, births = [], []

            def birth(births=births, case=case):
                births.append(1)
                return 2.0 if case == "pid_reuse" and len(births) > 1 else 1.0

            process = SimpleNamespace(
                pid=19,
                create_time=birth,
                children=lambda recursive: [],
                status=lambda case=case: "running" if case == "pid_reuse" else "zombie",
                send_signal=lambda sig, calls=calls: calls.append(sig),
            )
            popen = SimpleNamespace(pid=19, returncode=0, poll=lambda: 0)

            def spawn(*args, out=out, popen=popen, **kwargs):
                (out / "data").mkdir()
                for rank in range(8):
                    save(
                        out / "data" / f"rank{rank}.json",
                        {"state": "COMPLETE", "rank": rank},
                    )
                return popen

            common = SimpleNamespace(
                PYTHON="cpu-fixture",
                GpuLock=lambda fd=fd: SimpleNamespace(acquire=lambda: fd),
                save_json=original_common.save_json,
                now_iso=lambda: "cpu-fixture",
                spawn=spawn,
                assert_gpus_idle=lambda: good,
                assert_ports_free=lambda ports: None,
                wait_gpus_released=lambda timeout, snapshot=snapshot: (True, snapshot),
            )
            psutil = SimpleNamespace(
                Process=lambda pid, process=process: process,
                NoSuchProcess=ProcessLookupError,
                AccessDenied=PermissionError,
                STATUS_ZOMBIE="zombie",
                process_iter=lambda case=case, process=process: (
                    [process] if case == "pid_reuse" else []
                ),
            )
            plan = {
                "head": "cpu",
                "mode": "capture",
                "output": str(out),
                "environment": {},
                "gpu_ids": list(range(8)),
                "ports": [],
                "max_job_seconds": 1,
            }
            held = False
            try:
                with (
                    patch.dict(sys.modules, {"common": common, "psutil": psutil}),
                    patch.object(
                        sys.modules[__name__], "check_plan", lambda *a, **k: None
                    ),
                    patch("os.nice", return_value=0),
                    patch("os.getpgid", return_value=19),
                    patch("signal.signal"),
                    patch("time.monotonic", return_value=0),
                    patch("time.sleep", side_effect=Held),
                ):
                    try:
                        save(Path(directory) / "plan.json", plan)
                        supervise(plan, Path(directory) / "plan.json")
                    except Held:
                        held = True
                assert not calls
                state = json.loads((out / "contract.json").read_text())
                assert held is (case != "valid")
                if held:
                    os.fstat(fd)
                    assert not state["gpu_released"]
                else:
                    assert state["gpu_released"] and state["alive"] == []
                outcomes[case] = "retained" if held else "released"
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
    signals, alive = [], [True]
    owned = SimpleNamespace(
        pid=23,
        create_time=lambda: 3.0,
        children=lambda recursive: [],
        status=lambda: "running" if alive[0] else "zombie",
        send_signal=lambda sig: (signals.append(sig), alive.__setitem__(0, False)),
    )
    fake_ps = SimpleNamespace(
        Process=lambda pid: owned,
        process_iter=list,
        NoSuchProcess=ProcessLookupError,
        AccessDenied=PermissionError,
        STATUS_ZOMBIE="zombie",
    )
    namespace = {"seen": {23: 3.0}, "proc": owned}
    with patch.dict(sys.modules, {"psutil": fake_ps}):
        proxy = ownership_boundary(
            SimpleNamespace(wait_gpus_released=lambda timeout: (True, good)), namespace
        )
        assert namespace["owned_remember"]() == [owned]
        namespace["signal_owned"](owned, 15)
        assert signals == [15] and not namespace["owned_remember"]()
        assert proxy.wait_gpus_released(1)[0]
    sys.path.insert(0, str(WS / "harness"))
    import common

    events = []
    smi = {
        f"card{r}": {
            "VRAM Total Memory (B)": str(288 * 2**30),
            "VRAM Total Used Memory (B)": "0",
        }
        for r in range(8)
    }
    with (
        patch.object(
            common.os, "listdir", side_effect=lambda path: events.append("kfd") or []
        ),
        patch.object(
            common.subprocess,
            "check_output",
            side_effect=lambda *a, **k: events.append("smi") or json.dumps(smi),
        ),
    ):
        assert whole_node_idle(common.gpu_snapshot())
    assert events == ["kfd", "smi"]
    return {
        "cases": outcomes,
        "owned_signal_and_exit": signals == [15],
        "KFD_before_SMI": True,
        "no_PID_namespace_intersection": True,
    }


def clock_trace_check():
    from copy import deepcopy
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    report = {
        "pid": 101,
        "native_tid": 201,
        "graph_exec": 0x123456789ABCDEF0,
        "stream_handle": 0x2000,
        "pci_bus_id": "0000:05:00.0",
        "body_warmup_ordinal": 0,
        "body_ticks": 50_000,
        "rank": 0,
        "rows": [
            {"body_replay_ordinal": i, "body_us": 500.0, "us": 514.0}
            for i in range(1, 5)
        ],
    }
    document = {
        "metadata": {"pid": 101},
        "strings": {
            "buffer_records": [
                {
                    "kind": "HIP_RUNTIME_API_EXT",
                    "operations": ["hipGraphLaunch", "hipModuleLaunchKernel"],
                },
                {
                    "kind": "KERNEL_DISPATCH",
                    "operations": ["NONE", "ENQUEUE", "KERNEL_DISPATCH_COMPLETE"],
                },
            ]
        },
        "kernel_symbols": [
            {"kernel_id": 0, "kernel_name": "", "formatted_kernel_name": "0"},
            {
                "kernel_id": 1,
                "kernel_name": "delay_0",
                "formatted_kernel_name": "delay_0",
            },
        ],
        "agents": [{"id": {"handle": 9}, "domain": 0, "location_id": 0x500}],
        "buffer_records": {"hip_api": [], "kernel_dispatch": []},
    }
    module_args = {
        "func": "0x3000",
        "gridDimX": "1",
        "gridDimY": "1",
        "gridDimZ": "1",
        "blockDimX": "64",
        "blockDimY": "1",
        "blockDimZ": "1",
        "sharedMemBytes": "0",
        "stream": "0x2000",
        "kernelParams": "0x4000",
        "extra": "0",
    }
    for ordinal, operation in enumerate([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 1):
        args = (
            module_args
            if operation
            else {"graphExec": hex(report["graph_exec"]), "stream": "0x2000"}
        )
        api = {
            "kind": 0,
            "operation": operation,
            "thread_id": 201,
            "start_timestamp": ordinal * 1_000_000,
            "end_timestamp": ordinal * 1_000_000 + 100,
            "correlation_id": {"internal": ordinal, "ancestor": 0, "external": 0},
            "stream_id": {"handle": 7},
            "retval": {"uint64_t_retval": 0},
            "args": [
                {"type": "RTTI", "name": key, "value": value}
                for key, value in args.items()
            ],
        }
        document["buffer_records"]["hip_api"].append(api)
        if ordinal != 2:
            document["buffer_records"]["kernel_dispatch"].append(
                {
                    "kind": 1,
                    "operation": 2,
                    "thread_id": 201,
                    "start_timestamp": ordinal * 1_000_000 + 100,
                    "end_timestamp": ordinal * 1_000_000 + 501_100,
                    "correlation_id": {
                        "internal": ordinal,
                        "ancestor": 0,
                        "external": 0,
                    },
                    "stream_id": {"handle": 7},
                    "dispatch_info": {
                        "kernel_id": 1,
                        "dispatch_id": ordinal,
                        "agent_id": {"handle": 9},
                        "queue_id": {"handle": 10},
                        "workgroup_size": {"x": 64, "y": 1, "z": 1},
                        "grid_size": {"x": 64, "y": 1, "z": 1},
                    },
                }
            )
    joined = join_clock_trace(document, report)
    assert len(joined["rows"]) == 4 and joined["rows"][0]["K_minus_B_us"] == 1
    assert joined["rows"][0]["E_minus_K_us"] == 13
    assert trace_pointer("0x123456789abcdef0") == report["graph_exec"]
    cases = []
    changes = [
        ("pid", ("metadata", "pid"), 102),
        ("tid", ("buffer_records", "hip_api", 2, "thread_id"), 202),
        ("graph", ("buffer_records", "hip_api", 2, "args", 0, "value"), "0x9999"),
        ("raw_stream", ("buffer_records", "hip_api", 2, "args", 1, "value"), "0x9999"),
        ("float_pointer", ("buffer_records", "hip_api", 2, "args", 0, "value"), 1.0),
        ("retval", ("buffer_records", "hip_api", 2, "retval", "uint64_t_retval"), 1),
        (
            "bool_retval",
            ("buffer_records", "hip_api", 2, "retval", "uint64_t_retval"),
            False,
        ),
        ("invalid_time", ("buffer_records", "hip_api", 2, "end_timestamp"), 1),
        ("float_time", ("buffer_records", "hip_api", 2, "end_timestamp"), 1.5),
        (
            "wrong_correlation",
            ("buffer_records", "kernel_dispatch", 1, "correlation_id", "internal"),
            99,
        ),
        (
            "capture_dispatch",
            ("buffer_records", "kernel_dispatch", 1, "correlation_id", "internal"),
            2,
        ),
        (
            "duplicate_correlation",
            ("buffer_records", "kernel_dispatch", 1, "correlation_id", "internal"),
            1,
        ),
        (
            "duplicate_dispatch",
            ("buffer_records", "kernel_dispatch", 1, "dispatch_info", "dispatch_id"),
            1,
        ),
        (
            "agent",
            (
                "buffer_records",
                "kernel_dispatch",
                1,
                "dispatch_info",
                "agent_id",
                "handle",
            ),
            19,
        ),
        (
            "queue",
            (
                "buffer_records",
                "kernel_dispatch",
                1,
                "dispatch_info",
                "queue_id",
                "handle",
            ),
            19,
        ),
        (
            "sdk_stream",
            ("buffer_records", "kernel_dispatch", 1, "stream_id", "handle"),
            19,
        ),
        (
            "sdk_stream_type",
            ("buffer_records", "kernel_dispatch", 1, "stream_id", "handle"),
            "7",
        ),
        ("bdf", ("agents", 0, "location_id"), 0x1500),
        ("symbol_hole", ("kernel_symbols", 1, "kernel_id"), 0),
        (
            "bool_geometry",
            (
                "buffer_records",
                "kernel_dispatch",
                1,
                "dispatch_info",
                "workgroup_size",
                "y",
            ),
            True,
        ),
        (
            "float_geometry",
            ("buffer_records", "kernel_dispatch", 1, "dispatch_info", "grid_size", "x"),
            64.0,
        ),
        (
            "raw_symbol_tamper",
            ("kernel_symbols", 1, "kernel_name"),
            "unexpected_actual_kernel",
        ),
        (
            "formatted_symbol_tamper",
            ("kernel_symbols", 1, "formatted_kernel_name"),
            "unexpected_label",
        ),
        (
            "module_function",
            ("buffer_records", "hip_api", 1, "args", 0, "value"),
            "0x3333",
        ),
    ]
    for name, path, value in changes:
        changed = deepcopy(document)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        cases.append((name, changed, report))
    for name, field, index, add in (
        ("missing_dispatch", "kernel_dispatch", 1, False),
        ("extra_dispatch", "kernel_dispatch", 1, True),
        ("missing_graph_call", "hip_api", 2, False),
        ("missing_capture_call", "hip_api", 1, False),
        ("extra_graph_call", "hip_api", 2, True),
    ):
        changed = deepcopy(document)
        records = changed["buffer_records"][field]
        if add:
            records.append(deepcopy(records[index]))
        else:
            records.pop(index)
        cases.append((name, changed, report))
    changed = deepcopy(document)
    for api in changed["buffer_records"]["hip_api"]:
        if api["operation"] == 1:
            api["args"][0]["value"] = "0x10000000000000000"
    cases.append(("all_module_pointer_overflow", changed, report))
    assert trace_pointer("0xffffffffffffffff") == 2**64 - 1
    assert trace_pointer("0") == 0
    changed_report = deepcopy(report)
    changed_report["rows"][2]["body_replay_ordinal"] = 4
    cases.append(("ordinal_gap", document, changed_report))
    rejected = []
    for name, changed, record in cases:
        try:
            join_clock_trace(changed, record)
        except (ValueError, KeyError, IndexError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError("invalid trace accepted: " + name)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "fixture"
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":'):
            path.write_text(text)
            try:
                load_clock_json(path)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed/nonfinite/duplicate JSON accepted")
        for text in (
            "W1122 warning",
            "E1122 error",
            "Dropped 1 record",
            "swapping the values",
            "cannot offload buffer",
            "[rocprofiler-sdk] [warning] visible diagnostic",
            "error: unrelated domain",
        ):
            path.write_text(text + "\n")
            try:
                clock_trace_logs(path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("loss/error diagnostic accepted")
        path.write_text("UserWarning: DEBUG: graph dump\n")
        clock_trace_logs(path)
        path.write_text(
            "E1122 12:00:00 output_stream.cpp:123] Opened result file: /expected\n"
        )
        clock_trace_logs(path, ["/expected"])
        measure_calls, check_calls = [], []
        row = {
            "rank": 0,
            "clock_id": "cpu",
            "t0_ns": 0,
            "t1_ns": 40_000,
            "wall_khz": 100_000,
            "requested_ticks": 100_000,
            "observed_ticks": 100_000,
            "guard_start_tick": 0,
            "guard_end_tick": 100_000,
            "us": 570.0,
        }
        for case in (
            "event_mismatch",
            "content",
            "order",
            "submission",
            "trace_after",
            "trace_before",
            "deadline",
        ):
            measure_calls.clear()
            check_calls.clear()
            sample = deepcopy(report)
            sample["rows"] = []

            def measure(graph, scale, case=case):
                measure_calls.append(scale)
                return {
                    **row,
                    "t1_ns": 900_000 if case == "submission" else row["t1_ns"],
                }

            def live(*args, case=case):
                check_calls.append(True)
                if (
                    case in ("trace_before", "deadline")
                    or case == "trace_after"
                    and len(check_calls) == 2
                ):
                    raise RuntimeError(case)

            ticks = (
                [100_740, 150_739]
                if case == "content"
                else [99_999, 150_740] if case == "order" else [100_740, 150_740]
            )
            body = SimpleNamespace(
                cpu=lambda ticks=ticks: SimpleNamespace(tolist=lambda: ticks)
            )
            dist = SimpleNamespace(
                all_gather_object=lambda results, error: results.__setitem__(
                    slice(None), [error] * TP
                )
            )
            with patch.object(
                sys.modules[__name__], "clock_live_check", side_effect=live
            ):
                try:
                    clock_observations(
                        SimpleNamespace(measure=measure),
                        None,
                        body,
                        sample,
                        {},
                        "traced",
                        dist,
                        Path(directory),
                    )
                except RuntimeError:
                    assert case != "event_mismatch"
                else:
                    assert case == "event_mismatch" and all(
                        not r["original_event_gate"] for r in sample["rows"]
                    )
            assert len(measure_calls) == (
                4
                if case == "event_mismatch"
                else 0 if case in ("trace_before", "deadline") else 1
            )
    driver_cases = clock_driver_check(document, report)
    return {
        "rejected_trace_cases": rejected,
        "driver_and_release_cases": driver_cases,
        "JSON_log_and_observation_stop_negatives": True,
        "finite_event_mismatch_finishes_fixed_set": True,
        "GPU_executed": False,
    }


def clock_driver_check(document, report):
    from copy import deepcopy
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    outcomes = {}
    for case in (
        "complete",
        "before_spawn",
        "late_spawn",
        "late_validation",
        "hard_cohort",
    ):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            plan = {
                "mode": CLOCK_MODE,
                "head": "cpu",
                "output": str(out),
                "environment": {"LD_LIBRARY_PATH": "/opt/rocm/lib"},
                "clock_protocol": clock_protocol(out),
            }
            path = out / "plan.json"
            save(path, plan)
            now, launched = [0], []

            def check(case=case, now=now):
                if case == "before_spawn" or now[0] >= 180:
                    raise TimeoutError("fixture deadline")
                return 180 - now[0]

            def popen(
                command,
                case=case,
                now=now,
                launched=launched,
                plan=plan,
                out=out,
                **kwargs,
            ):
                cohort = command[command.index("--clock-cohort") + 1]
                launched.append(cohort)
                if cohort == "traced":
                    table = plan["clock_protocol"]["normal_timer_messages"]
                    commands = []
                    for ordinal in range(TP + 1):
                        role = "tracker" if ordinal == 0 else "worker"
                        descriptors = (
                            {"tracker_fd": 19}
                            if ordinal == 0
                            else {"tracker_fd": 20, "pipe_handle": 30 + ordinal}
                        )
                        rule = table["children"][role]
                        argv = [
                            table["python"],
                            *table["python_flags"],
                            "-c",
                            rule["program"].format(**descriptors),
                            *rule["suffix"],
                        ]
                        commands.append(
                            clock_child_record(
                                table,
                                role,
                                descriptors,
                                argv,
                                list(descriptors.values()),
                            )
                        )
                    save(out / "data/traced/command-ledger.json", commands)
                if case == "late_spawn":
                    now[0] = 181
                protocol = plan["clock_protocol"]
                for rank, bdf in enumerate(protocol["pci_bdfs"]):
                    record = deepcopy(report)
                    record.update(
                        state="COMPLETE",
                        rank=rank,
                        cohort=cohort,
                        pid=101 + rank,
                        native_tid=201,
                        pci_bus_id=bdf,
                        hip_pci_status=0,
                        hsa_frequency_status=0,
                        hsa_timestamp_frequency_hz=10**9,
                        loaded_profiler_libraries=[],
                        profiler_environment={"LD_LIBRARY_PATH": "/opt/rocm/lib"},
                    )
                    if cohort == "traced":
                        record["loaded_profiler_libraries"] = protocol[
                            "profiler_loaded_libraries"
                        ]
                        record["profiler_environment"].update(
                            protocol["profiler_environment"],
                            LD_PRELOAD=":".join(protocol["profiler_preload_libraries"]),
                            ROCP_TOOL_LIBRARIES=protocol["profiler_tool_library"],
                            ROCPROF_HIP_RUNTIME_API_TRACE="1",
                            ROCPROF_HIP_COMPILER_API_TRACE="1",
                            ROCPROF_KERNEL_TRACE="1",
                        )
                        trace = deepcopy(document)
                        trace["metadata"]["pid"] = record["pid"]
                        trace["agents"][0]["location_id"] = (
                            int(bdf.split(":")[1], 16) << 8
                        )
                        prof = out / "data/traced/profiler"
                        prof.mkdir(exist_ok=True)
                        save(
                            prof / f"clock-{record['pid']}_results.json",
                            {"rocprofiler-sdk-tool": [trace]},
                        )
                    save(out / "data" / cohort / f"rank{rank}.json", record)
                return SimpleNamespace(
                    pid=99, returncode=1 if case == "hard_cohort" else 0, poll=lambda: 0
                )

            original_validate = validate_clock_trace

            def validate(
                *args, original_validate=original_validate, case=case, now=now
            ):
                result = original_validate(*args)
                if case == "late_validation":
                    now[0] = 181
                return result

            with (
                patch.object(
                    sys.modules[__name__], "clock_remaining", side_effect=check
                ),
                patch.object(
                    sys.modules[__name__], "validate_clock_trace", side_effect=validate
                ),
                patch("subprocess.Popen", side_effect=popen),
                patch.dict(os.environ, {"K31_CLOCK_STARTED_NS": "0"}),
                patch("time.monotonic_ns", side_effect=lambda now=now: now[0] * 10**9),
            ):
                try:
                    clock_driver(plan, path)
                except (RuntimeError, TimeoutError):
                    assert case != "complete"
                    outcomes[case] = "REJECTED"
                else:
                    assert case == "complete"
                    outcomes[case] = "COMPLETE"
            assert len(launched) == (
                0
                if case == "before_spawn"
                else 1 if case in ("late_spawn", "hard_cohort") else 2
            )
            if case != "complete":
                continue
            released = {
                "mode": CLOCK_MODE,
                "head": "cpu",
                "clock_plan_sha256": sha(path),
                "state": "COMPLETE",
                "cleanup_status": "released",
                "gpu_released": True,
                "stopped": True,
                "alive": [],
                "workload_elapsed_seconds": 10,
                "after": {
                    "owned_identity_or_group_unknown": False,
                    "kfd_pids": [],
                    "cards": {
                        f"card{r}": {"used_gib": 0.277, "free_gib": 287.707}
                        for r in range(8)
                    },
                },
            }
            for field, bad in (
                ("gpu_released", False),
                ("alive", [99]),
                ("clock_plan_sha256", "wrong"),
                ("workload_elapsed_seconds", 180),
            ):
                save(out / "contract.json", {**released, field: bad})
                try:
                    analyze_clock(plan, path, out / "analysis.json")
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("analysis preceded qualified release")
            save(out / "contract.json", released)
            analyzed = analyze_clock(plan, path, out / "analysis.json")
            assert sum(len(record["rows"]) for record in analyzed["traced"]) == 32
            bad_mapping = [deepcopy(report)] * 8
            try:
                clock_mapping(bad_mapping, plan)
            except RuntimeError:
                pass
            else:
                raise AssertionError("non-bijective BDF mapping accepted")
            original = [
                {"pci_bus_id": bdf} for bdf in plan["clock_protocol"]["pci_bdfs"]
            ]
            try:
                clock_mapping(original[::-1], plan, clock_mapping(original, plan))
            except RuntimeError:
                pass
            else:
                raise AssertionError("cross-cohort ordinal mapping changed")
    return outcomes


def clock_pre_setup_check_function(name):
    source = subprocess.check_output(
        [
            "git",
            "show",
            "419aa118c82f82ebc93b620dbd2a8e02e1d1b832:spike/k31_k2_front/replay.py",
        ],
        text=True,
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = dict(globals())
    exec(  # noqa: S102
        compile(
            ast.Module(body=[function], type_ignores=[]), "qualified-pre-setup", "exec"
        ),
        namespace,
    )
    return namespace[name]


def clock_log_check():
    from multiprocessing import get_context, util
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    with TemporaryDirectory() as directory:
        out = Path(directory)
        traced = out / "data/traced"
        traced.mkdir(parents=True)
        ledger = traced / "command-ledger.json"
        save(ledger, [])
        plan = {"output": str(out), "clock_protocol": clock_protocol(out)}
        table = plan["clock_protocol"]["normal_timer_messages"]
        table["python"] = sys.executable
        table["python_flags"] = util._args_from_interpreter_flags()
        original = util.spawnv_passfds

        def spawn_cpu(function, args, nprocs, join):
            assert (
                function is clock_rank_main
                and args == ({}, "traced")
                and nprocs == TP
                and join
            )
            processes = [
                get_context("spawn").Process(target=time.sleep, args=(0.01,))
                for _ in range(TP)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
                assert process.exitcode == 0

        clock_spawn(plan, SimpleNamespace(spawn=spawn_cpu), {})
        assert util.spawnv_passfds is original
        records = load_clock_json(ledger)
        labels = clock_timer_labels(plan, complete=True)
        assert len(records) == TP + 1
        log = traced / "worker.log"
        prefix = "W20261122 00:41:29.331333 123 simple_timer.cpp:55] "
        normal = prefix + labels[0] + " ::     0.001101 sec"
        log.write_text(
            "\n".join(
                prefix + label + f" :: {seconds:12.6f} sec"
                for label in labels
                for seconds in (0, 1.25)
            )
            + "\n"
        )
        result = clock_trace_logs(
            log, timer_labels=labels, classification=traced / "classification.json"
        )
        assert len(result["classifications"]) == len(labels) * 2
        try:
            clock_pre_setup_check_function("clock_trace_logs")(log)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pre-fix parser did not reject the normal summaries")
        refused = []
        for bad in (
            normal.replace("tool initialization", "tool detachment"),
            prefix + "[rocprofv3] 'unregistered command' ::     0.000000 sec",
            normal.replace("simple_timer.cpp:55", "other.cpp:55"),
            normal.replace("0.001101", "nan"),
            normal.replace("0.001101", "inf"),
            normal.replace("0.001101", "-0.001101"),
            normal.replace(" ::     ", " :: "),
            normal.replace("20261122", "20261322"),
            normal + " appended diagnostic",
            normal + "\nW20261122 00:41:29.331333 123 unrelated.cpp:1] unknown warning",
            normal + "\nERROR: failed export",
            normal + "\nFATAL: failed export",
            normal + "\nDropping record",
            normal + "\nswapping the values",
            normal + "\ncannot offload buffer",
        ):
            log.write_text(bad + "\n")
            try:
                clock_trace_logs(log, timer_labels=labels)
            except (RuntimeError, ValueError):
                refused.append(bad)
            else:
                raise AssertionError("unqualified normal summary accepted")
        for bad in (
            records[:-1],
            [*records, records[-1]],
            [{**records[0], "label": "forged"}],
            [{**records[0], "argv": ["unknown"]}],
            [{**records[0], "passfds": []}],
            [{**records[0], "descriptors": {"tracker_fd": True}}],
        ):
            save(ledger, bad)
            try:
                clock_timer_labels(plan, complete=True)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("invalid command ledger accepted")
        for malformed in (None, "{", '{"launches":[],"launches":[]}'):
            if malformed is None:
                ledger.unlink()
            else:
                ledger.write_text(malformed)
            try:
                clock_timer_labels(plan)
            except (ValueError, RuntimeError, FileNotFoundError):
                pass
            else:
                raise AssertionError("missing/corrupt command ledger accepted")
        save(ledger, [])
        table["python_flags"].append("-I")
        try:
            clock_spawn(plan, SimpleNamespace(spawn=spawn_cpu), {})
        except RuntimeError:
            assert util.spawnv_passfds is original
        else:
            raise AssertionError("unknown interpreter flags accepted")
        table["python_flags"].pop()
        for failure in ("unknown_caller", "corrupt_hook", "spawn_failure"):
            save(ledger, [])

            def failed_spawn(*args, failure=failure, **kwargs):
                if failure == "unknown_caller":
                    util.spawnv_passfds(b"unknown", [], [])
                elif failure == "corrupt_hook":
                    util.spawnv_passfds = None
                else:
                    raise OSError("controlled launch failure")

            try:
                clock_spawn(plan, SimpleNamespace(spawn=failed_spawn), {})
            except (RuntimeError, OSError):
                assert util.spawnv_passfds is original
            else:
                raise AssertionError("invalid launch/hook accepted")
    return {
        "exact_CPU_prelaunch_forwarding": True,
        "dynamic_children": TP + 1,
        "prelaunch_records": records,
        "normal_summary_classes_checked": True,
        "pre_fix_parser_refused_normal_summaries": True,
        "diagnostic_refusals": len(refused),
        "ledger_and_hook_refusals": True,
        "GPU_executed": False,
    }


def clock_marker_check():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    assert (
        str(CLOCK_MARKER)
        == "/app/k31-agent-team-private/task-15/cache/comgr/llvmcache.timestamp"
    )
    rejected = []
    old_verify = clock_pre_setup_check_function("verify_binding")
    for case in (
        "mtime_only",
        "missing",
        "missing_record",
        "replaced",
        "retargeted",
        "nonempty",
        "other_file",
        "model",
        "wrong_policy",
        "read_mtime_race",
        "read_identity_race",
        "read_ctime_race",
    ):
        with TemporaryDirectory() as directory:
            marker, other, model = [
                Path(directory) / name
                for name in ("llvmcache.timestamp", "payload", "model.bin")
            ]
            marker.write_bytes(b"")
            other.write_bytes(b"payload")
            model.write_bytes(b"weight")
            binding = {
                "files": {str(p): file_identity(p) for p in (marker, other)},
                "models": {str(model): file_identity(model)},
            }
            with patch.object(sys.modules[__name__], "CLOCK_MARKER", marker):
                policy = clock_marker_policy()
                assert verify_binding(binding, policy)[0]["delta_mtime_ns"] == 0
                if case in ("mtime_only", "other_file", "model"):
                    target = {
                        "mtime_only": marker,
                        "other_file": other,
                        "model": model,
                    }[case]
                    status = target.stat()
                    os.utime(target, ns=(status.st_atime_ns, status.st_mtime_ns + 100))
                elif case == "missing":
                    marker.unlink()
                elif case == "missing_record":
                    del binding["files"][str(marker)]
                elif case == "replaced":
                    replacement = marker.with_suffix(".replacement")
                    replacement.write_bytes(b"")
                    replacement.replace(marker)
                elif case == "retargeted":
                    target = marker.with_suffix(".target")
                    os.link(marker, target)
                    marker.unlink()
                    marker.symlink_to(target)
                elif case == "nonempty":
                    marker.write_bytes(b"changed")
                elif case == "wrong_policy":
                    policy = {**policy, "path": str(other)}
                original_sha = sha

                def racing_sha(
                    path, case=case, marker=marker, original_sha=original_sha
                ):
                    value = original_sha(path)
                    if Path(path) == marker:
                        status = marker.stat()
                        if case == "read_mtime_race":
                            os.utime(
                                marker,
                                ns=(status.st_atime_ns, status.st_mtime_ns + 100),
                            )
                        elif case == "read_identity_race":
                            replacement = marker.with_suffix(".replacement")
                            replacement.write_bytes(b"")
                            replacement.replace(marker)
                        elif case == "read_ctime_race":
                            marker.chmod(0o600)
                    return value

                with patch.object(sys.modules[__name__], "sha", side_effect=racing_sha):
                    try:
                        observed = verify_binding(binding, policy)
                    except (RuntimeError, FileNotFoundError):
                        assert case != "mtime_only"
                        rejected.append(case)
                    else:
                        assert (
                            case == "mtime_only"
                            and observed[0]["delta_mtime_ns"] == 100
                        )
                        for verify in (old_verify, verify_binding):
                            try:
                                verify(binding)
                            except RuntimeError:
                                pass
                            else:
                                raise AssertionError(
                                    "strict old/ordinary binding accepted marker mtime change"
                                )
    return {
        "mtime_only_clock_positive": True,
        "ordinary_binding_stays_strict": True,
        "pre_fix_binding_refused_mtime": True,
        "refused": rejected,
        "real_cache_modified": False,
    }


def clock_source_check():
    if "kfd-guard" not in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError("clock CPU checks require no_gpu.sh")
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    row = {
        "rank": 0,
        "clock_id": "cpu",
        "t0_ns": 0,
        "t1_ns": 40_000,
        "wall_khz": 100_000,
        "requested_ticks": 100_000,
        "observed_ticks": 100_000,
        "guard_start_tick": 0,
        "guard_end_tick": 100_000,
        "us": 514.0,
    }
    good, error = clock_controls(row, 100_740, 150_740, 50_000)
    assert error is None and good["original_event_gate"]
    mismatch, error = clock_controls({**row, "us": 570.0}, 100_740, 150_740, 50_000)
    assert error is None and not mismatch["original_event_gate"]
    for changed, start, end in (
        (row, 100_740, 150_739),
        (row, 99_999, 149_999),
        ({**row, "t1_ns": 900_000}, 100_740, 150_740),
    ):
        assert clock_controls(changed, start, end, 50_000)[1]
    fixed = clock_protocol(Path("/cpu-fixture"))
    rocpd = Path("/opt/rocm/lib/librocprofiler-sdk-rocpd.so.1")
    dynamic = subprocess.check_output(
        ["readelf", "-d", fixed["profiler_tool_library"]], text=True
    )
    assert "Shared library: [librocprofiler-sdk-rocpd.so.1]" in dynamic
    assert str(rocpd) in fixed["profiler_files"]
    assert set(fixed["profiler_loaded_libraries"]) == set(
        fixed["profiler_preload_libraries"]
    ) | {str(rocpd.resolve())}
    identity = {
        "hip_pci_status": 0,
        "pci_bus_id": "0000:05:00.0",
        "hsa_frequency_status": 0,
        "hsa_timestamp_frequency_hz": 10**9,
        "loaded_profiler_libraries": fixed["profiler_loaded_libraries"],
        "profiler_environment": {
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "LD_PRELOAD": ":".join(fixed["profiler_preload_libraries"]),
            "ROCP_TOOL_LIBRARIES": fixed["profiler_tool_library"],
            **fixed["profiler_environment"],
            "ROCPROF_HIP_RUNTIME_API_TRACE": "1",
            "ROCPROF_HIP_COMPILER_API_TRACE": "1",
            "ROCPROF_KERNEL_TRACE": "1",
        },
    }
    protocol = {
        "environment": {"LD_LIBRARY_PATH": "/opt/rocm/lib"},
        "clock_protocol": fixed,
    }
    validate_clock_identity(identity, protocol, "traced")
    old = {
        **protocol,
        "clock_protocol": {
            **fixed,
            "profiler_loaded_libraries": fixed["profiler_preload_libraries"],
        },
    }
    try:
        validate_clock_identity(identity, old, "traced")
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "old two-library map accepted the required rocpd dependency"
        )
    rejected_identity = []
    for name, changed in (
        ("frequency", {"hsa_timestamp_frequency_hz": 10**8}),
        ("frequency_status", {"hsa_frequency_status": 1}),
        (
            "missing_rocpd",
            {"loaded_profiler_libraries": fixed["profiler_preload_libraries"]},
        ),
        (
            "unexpected_mapping",
            {
                "loaded_profiler_libraries": [
                    *fixed["profiler_loaded_libraries"],
                    "/unexpected",
                ]
            },
        ),
        (
            "extra_explicit_preload",
            {
                "profiler_environment": {
                    **identity["profiler_environment"],
                    "LD_PRELOAD": ":".join(fixed["profiler_loaded_libraries"]),
                }
            },
        ),
        ("missing_environment", {"profiler_environment": {}}),
        ("hip_status", {"hip_pci_status": 1}),
        ("wrong_bdf", {"pci_bus_id": "0000:ff:00.0"}),
    ):
        try:
            validate_clock_identity({**identity, **changed}, protocol, "traced")
        except RuntimeError:
            rejected_identity.append(name)
        else:
            raise AssertionError("invalid trace identity accepted: " + name)
    loader_check = {
        "DT_NEEDED_rocpd": True,
        "rocpd": file_identity(rocpd),
        "explicit_preloads": fixed["profiler_preload_libraries"],
        "required_filtered_maps": fixed["profiler_loaded_libraries"],
        "old_two_library_expectation_rejected": True,
        "negative_cases": rejected_identity,
        "libraries_loaded": False,
    }
    sys.path.insert(0, str(WS / "harness"))
    import common as original_common

    outcomes = {}
    for case in (
        "on_time",
        "late_exit",
        "late_validation",
        "negative_without_new_gate",
    ):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "result"
            fd = os.open(Path(directory) / "lock", os.O_CREAT | os.O_RDWR, 0o600)
            now = [0]
            process = SimpleNamespace(
                pid=19,
                create_time=lambda: 1.0,
                status=lambda: "zombie",
                children=lambda recursive: [],
            )
            popen = SimpleNamespace(pid=19, returncode=0, poll=lambda: 0)

            def spawn(*args, out=out, now=now, popen=popen, case=case, **kwargs):
                (out / "data").mkdir()
                for rank in range(8):
                    save(
                        out / "data" / f"rank{rank}.json",
                        {"state": "COMPLETE", "rank": rank},
                    )
                now[0] = (
                    181 if case in ("late_exit", "negative_without_new_gate") else 10
                ) * 10**9
                return popen

            original_loads = json.loads

            def loads(
                value,
                *args,
                now=now,
                case=case,
                original_loads=original_loads,
                **kwargs,
            ):
                result = original_loads(value, *args, **kwargs)
                if (
                    case == "late_validation"
                    and isinstance(result, dict)
                    and result.get("rank") == 7
                ):
                    now[0] = 181 * 10**9
                return result

            good_snapshot = {
                "kfd_pids": [],
                "cards": {
                    f"card{r}": {"used_gib": 0.277, "free_gib": 287.707}
                    for r in range(8)
                },
            }
            common = SimpleNamespace(
                PYTHON="cpu-fixture",
                GpuLock=lambda fd=fd: SimpleNamespace(acquire=lambda: fd),
                save_json=original_common.save_json,
                now_iso=lambda: "cpu-fixture",
                spawn=spawn,
                assert_gpus_idle=lambda good_snapshot=good_snapshot: good_snapshot,
                assert_ports_free=lambda ports: None,
                wait_gpus_released=lambda timeout, good_snapshot=good_snapshot: (
                    True,
                    good_snapshot,
                ),
            )
            psutil = SimpleNamespace(
                Process=lambda pid, process=process: process,
                process_iter=list,
                NoSuchProcess=ProcessLookupError,
                AccessDenied=PermissionError,
                STATUS_ZOMBIE="zombie",
            )
            plan = {
                "head": "cpu",
                "mode": "clock_boundary",
                "output": str(out),
                "environment": {},
                "gpu_ids": list(range(8)),
                "ports": [],
                "max_job_seconds": 180,
            }
            rejected = False
            try:
                with (
                    patch.dict(sys.modules, {"common": common, "psutil": psutil}),
                    patch.object(
                        sys.modules[__name__], "check_plan", lambda *a, **k: None
                    ),
                    patch.object(
                        sys.modules[__name__],
                        "CLOCK_MODE",
                        (
                            "disabled"
                            if case == "negative_without_new_gate"
                            else "clock_boundary"
                        ),
                    ),
                    patch("json.loads", side_effect=loads),
                    patch("os.nice", return_value=0),
                    patch("signal.signal"),
                    patch("time.monotonic", return_value=0),
                    patch("time.monotonic_ns", side_effect=lambda now=now: now[0]),
                ):
                    try:
                        save(Path(directory) / "plan.json", plan)
                        supervise(plan, Path(directory) / "plan.json")
                    except TimeoutError:
                        rejected = True
                state = json.loads((out / "contract.json").read_text())
                assert rejected == (case in ("late_exit", "late_validation"))
                assert state["gpu_released"] and state["alive"] == []
                outcomes[case] = state["state"]
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
    return {
        "state": "CLOCK_DRIVER_CPU_CHECKED",
        "clock_marker_policy_checked": clock_marker_check(),
        "clock_normal_logs_checked": clock_log_check(),
        "old_event_failure_retained": True,
        "hard_content_order_submission_failures_checked": True,
        "timebase_and_loader_failures_checked": True,
        "static_loader_dependency_check": loader_check,
        "absolute_deadline_cases": outcomes,
        "trace_schema_and_stops": clock_trace_check(),
        "GPU_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source-check", action="store_true")
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--cpu-check", action="store_true")
    action.add_argument("--write-plan", choices=("capture", "timing", CLOCK_MODE))
    action.add_argument("--clock-cohort", choices=("untraced", "traced"))
    action.add_argument("--clock-source-check", action="store_true")
    action.add_argument("--analyze-clock", action="store_true")
    action.add_argument("--run-plan", type=Path)
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--ready", type=Path)
    args = parser.parse_args()
    if args.source_check or args.compile_only or args.cpu_check:
        print(
            json.dumps(
                cpu_check(
                    args.snapshot if args.cpu_check else None,
                    args.out,
                    args.compile_only,
                ),
                indent=2,
            )
        )
    elif args.write_plan:
        plan = make_plan(
            args.write_plan, args.out, args.snapshot, args.ready, args.plan
        )
        if args.plan:
            with args.plan.open("x") as target:
                json.dump(plan, target, indent=2)
        print(json.dumps(plan, indent=2))
    elif args.run_plan:
        supervise(json.loads(args.run_plan.read_text()), args.run_plan)
    elif args.analyze_clock:
        analyze_clock(load_clock_json(args.plan), args.plan, args.out)
    elif args.clock_source_check:
        print(json.dumps(clock_source_check(), indent=2))
    elif args.clock_cohort:
        plan = json.loads(args.plan.read_text())
        marker_observations = check_plan(plan, args.plan, worker=True)
        if plan["mode"] != CLOCK_MODE:
            raise ValueError("wrong clock cohort mode")
        save(
            Path(plan["output"]) / "data" / args.clock_cohort / "binding.json",
            marker_observations,
        )
        clock_remaining()
        import torch.multiprocessing as mp

        worker_plan = {
            key: plan[key]
            for key in ("mode", "output", "environment", "clock_protocol")
        }
        if args.clock_cohort == "traced":
            clock_spawn(plan, mp, worker_plan)
        else:
            mp.spawn(
                clock_rank_main,
                args=(worker_plan, args.clock_cohort),
                nprocs=TP,
                join=True,
            )
        clock_remaining()
    else:
        plan = json.loads(args.plan.read_text())
        marker_observations = check_plan(plan, args.plan, worker=True)
        if plan["mode"] == CLOCK_MODE:
            save(Path(plan["output"]) / "driver-binding.json", marker_observations)
            clock_driver(plan, args.plan)
            return
        if plan["mode"] != "timing":
            raise ValueError("wrong worker mode")
        out = Path(plan["output"]) / "data"
        out.mkdir(parents=True, exist_ok=False)
        import torch.multiprocessing as mp

        mp.spawn(
            rank_main,
            args=(Path(plan["snapshot"]), out, 29675, 5, 2),
            nprocs=TP,
            join=True,
        )


if __name__ == "__main__":
    main()
