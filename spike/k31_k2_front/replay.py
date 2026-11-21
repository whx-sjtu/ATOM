"""Current whole-K2 replay using the existing restore and timing machinery."""

import argparse
import ast
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import timedelta
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


def execution_binding(base):
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
    seed = WS / "agent-team/work/15/cache-seed.json"
    paths.add(seed)
    records, resolved = {}, {}
    for path in sorted(paths):
        real = path.resolve()
        if real not in resolved:
            resolved[real] = file_identity(path)
        records[str(path)] = resolved[real]
    models = {}
    for name in ("model", "draft"):
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
        "model_scope": "Metadata/content and shard filesystem/header identity; actual selected materialized weight bytes are hashed in the real capture.",
        "cache_scope": "Preexisting private artifacts are bound; runtime may append artifacts only within the declared private caches.",
    }


def verify_binding(binding):
    checked = {}
    for records in (binding["files"], binding["models"]):
        for name, expected in records.items():
            path = Path(name)
            key = (str(path.resolve()), "sha256" in expected)
            if key not in checked:
                checked[key] = file_identity(path, content=key[1])
            if checked[key] != expected:
                raise RuntimeError("runtime/model/cache binding changed: " + name)


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


def make_plan(mode, out, snapshot=None, ready=None):
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
    if plan["mode"] not in ("capture", "timing") or plan["max_job_seconds"] != (
        900 if plan["mode"] == "capture" else 180
    ):
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
        os.environ.get(name) != value for name, value in plan["environment"].items()
    ):
        raise RuntimeError("worker environment differs from the frozen plan")
    verify_binding(plan["execution"])
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
    check_plan(plan, path)
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
    changes = {"environment": 0, "ports": 0, "grant": 0, "remember": 0, "signal": 0}

    class Plumbing(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "remember":
                node.body = ast.parse("return owned_remember()").body
                changes["remember"] += 1
                return node
            return self.generic_visit(node)

        def visit_Assign(self, node):
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
    }:
        raise RuntimeError("pinned supervisor shape changed")
    sys.path.insert(0, str(WS / "harness"))
    import common as C

    script = HERE / ("capture.py" if plan["mode"] == "capture" else "replay.py")
    command = [C.PYTHON, str(script), "--worker", "--plan", str(path)]
    env = dict(plan["environment"], K31_K2_FRONT_PLAN=str(path))
    actual = dict(plan, environment=env, command=command)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source-check", action="store_true")
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--cpu-check", action="store_true")
    action.add_argument("--write-plan", choices=("capture", "timing"))
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
        plan = make_plan(args.write_plan, args.out, args.snapshot, args.ready)
        if args.plan:
            with args.plan.open("x") as target:
                json.dump(plan, target, indent=2)
        print(json.dumps(plan, indent=2))
    elif args.run_plan:
        supervise(json.loads(args.run_plan.read_text()), args.run_plan)
    else:
        plan = json.loads(args.plan.read_text())
        check_plan(plan, args.plan, worker=True)
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
