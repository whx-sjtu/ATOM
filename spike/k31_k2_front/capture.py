"""One-shot, same-invocation current K2 capture, excluded from replay timing."""

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path("/app/k31-opt-worktrees/CX-front-floor/ATOM/tools")
ROWS, NB, LAYER, TP = 4, 2, 17, 8
SLOT, TAG = 128 + LAYER, 129 + LAYER
PROMPT = "Explain in plain language why the sky looks blue during the day."
_CAPTURE = None
_LIVE = False
_WITNESS = None
WEIGHT_LAYOUT = {
    "gate_up_interleaved": False,
    "fp8_activations": False,
    "w13_is_shuffled": True,
    "w2_is_shuffled": True,
    "triton_swizzle": False,
    "mega_owns_weights": False,
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def helper(name, module_name=None):
    module_name = module_name or name
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = TOOLS / (name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SNAP = None


def views(snapshot):
    import torch

    result = {}
    for name, desc in snapshot["tensors"].items():
        dtype = SNAP.dtype_from_name(desc["dtype"])
        result[name] = torch.empty(0, dtype=dtype).set_(
            snapshot["arenas"][desc["arena"]].untyped_storage(),
            desc["offset_bytes"] // desc["itemsize"],
            tuple(desc["shape"]),
            tuple(desc["stride"]),
        )
    return result


def add_route_views(layout):
    pair = layout["tensors"]["route_pairs"]
    if pair["dtype"] != "int32" or pair["shape"] != [ROWS, 32, 2]:
        raise ValueError("current route pair layout differs")
    for name, dtype, offset in (
        ("route_ids", "int32", 0),
        ("route_weights", "float32", 128),
    ):
        layout["tensors"][name] = {
            **pair,
            "dtype": dtype,
            "shape": [ROWS, 16],
            "stride": [64, 2],
            "offset_bytes": pair["offset_bytes"] + offset,
            "logical_nbytes": ROWS * 16 * 4,
            "source_storage_offset_bytes": pair["source_storage_offset_bytes"] + offset,
        }


def validate_snapshot(snapshot, *, allow_test_fixture=False):
    import torch

    meta = snapshot["meta"]
    ident, key, abi = meta["identity"], meta["key"], meta["abi_scalars"]
    if snapshot.get("schema") != SNAP.SCHEMA:
        raise ValueError("snapshot schema differs")
    if ident.get("fixture_kind") != "natural_k2" or allow_test_fixture:
        raise ValueError("only a real current invocation is a replay input")
    if (
        ident["tp"],
        ident["model_layer"],
        ident["rows"],
        abi["layer"],
        abi["reset"],
    ) != (TP, LAYER, ROWS, SLOT, 0):
        raise ValueError("current identity or ABI differs")
    if not 0 <= ident["rank"] < TP or ident["epoch"] <= 0:
        raise ValueError("rank or epoch invalid")
    if (
        key["tokens"] != ROWS
        or key["nblocks"] != NB
        or not key["reduce"]
        or key["diag"]
        or key["timeline"]
        or "sgu_pair" in key
        or "x_ready_floor" in key
    ):
        raise ValueError("snapshot is not the declared native current key")
    if not meta.get("capture_done") or not ident.get("arm_id"):
        raise ValueError("incomplete capture")
    if not ident.get("ordinary_served_witness"):
        raise ValueError("no prior real served-Mono witness")
    if set(SNAP.REQUIRED) - set(snapshot["tensors"]):
        raise ValueError("missing actual input, weight or output")
    total = 0
    for name, blob in snapshot["arenas"].items():
        if (
            blob.device.type != "cpu"
            or blob.dtype != torch.uint8
            or blob.ndim != 1
            or not blob.is_contiguous()
        ):
            raise ValueError("arena must be contiguous CPU bytes")
        total += blob.numel()
        if SNAP.arena_hash(blob) != snapshot["arena_hashes"][name]:
            raise ValueError("arena hash differs")
    if total > SNAP.MAX_BYTES:
        raise ValueError("snapshot exceeds the existing per-rank cap")
    for name, desc in snapshot["tensors"].items():
        itemsize = torch.empty(
            (), dtype=SNAP.dtype_from_name(desc["dtype"])
        ).element_size()
        if desc["itemsize"] != itemsize or desc["offset_bytes"] % itemsize:
            raise ValueError("dtype/offset mismatch")
        ranges = SNAP.used_ranges(
            desc["shape"], desc["stride"], itemsize, desc["offset_bytes"]
        )
        if max(end for _, end in ranges) > snapshot["arenas"][desc["arena"]].numel():
            raise ValueError("view escapes arena")
        if (
            SNAP.tensor_hash(snapshot["arenas"][desc["arena"]], desc)
            != snapshot["tensor_hashes"][name]
        ):
            raise ValueError("tensor hash differs")
    ts = views(snapshot)
    for name, shape in (
        ("core", (ROWS, 1536)),
        ("prefix_before", (ROWS, 7168)),
        ("prefix_after", (ROWS, 7168)),
        ("x", (ROWS, 7168)),
        ("out", (ROWS, 7168)),
        ("blocks", (ROWS, NB, 7168)),
    ):
        if (
            ts[name].dtype != torch.bfloat16
            or tuple(ts[name].shape) != shape
            or not bool(torch.isfinite(ts[name]).all())
        ):
            raise ValueError("invalid current activation: " + name)
    if ts["blocks"].stride() != (abi["blk_sm"], abi["blk_sr"], 1):
        raise ValueError("blocks strides differ from the native ABI")
    for name, width in (
        ("core", 1536),
        ("prefix_before", 7168),
        ("prefix_after", 7168),
        ("x", 7168),
        ("out", 7168),
    ):
        if ts[name].stride() != (width, 1):
            raise ValueError("activation ABI stride differs")
    pairs, ids, weights = ts["route_pairs"], ts["route_ids"], ts["route_weights"]
    if (
        pairs.dtype != torch.int32
        or pairs.shape != (ROWS, 32, 2)
        or not bool((pairs[:, :, 1] == TAG).all())
    ):
        raise ValueError("route tag/layout differs")
    if not bool(((ids >= 0) & (ids < 896)).all()) or any(
        len(set(row)) != 16 for row in ids.tolist()
    ):
        raise ValueError("invalid ordered route IDs")
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("invalid ordered route weights")
    raw_sizes = {
        "w13": 896 * 2 * 384 * 3584 // 2,
        "w2": 896 * 3584 * 384 // 2,
        "w13s": 896 * 2 * 384 * 112,
        "w2s": 896 * 3584 * 16,
    }
    for name, size in raw_sizes.items():
        desc = snapshot["tensors"][name]
        if desc["logical_nbytes"] != size or SNAP.used_ranges(
            desc["shape"], desc["stride"], desc["itemsize"], desc["offset_bytes"]
        ) != [(desc["offset_bytes"], desc["offset_bytes"] + size)]:
            raise ValueError("incomplete packed weight plane: " + name)
    for name, shape in {
        "w_o": (7168, 1536),
        "w_gate": (896, 7168),
        "w_ld": (3584, 7168),
        "w_up": (896, 3584),
        "w_sgu": (1536, 7168),
        "w_sd": (7168, 768),
        "ares_nw": (7168,),
        "ares_qk": (7168,),
        "in_nw": (7168,),
        "ln_w": (3584,),
    }.items():
        if (
            ts[name].dtype != torch.bfloat16
            or ts[name].shape != shape
            or not ts[name].is_contiguous()
        ):
            raise ValueError("materialized weight differs: " + name)
    if (
        ts["bias"].dtype != torch.float32
        or ts["bias"].shape != (896,)
        or not ts["bias"].is_contiguous()
    ):
        raise ValueError("bias binding differs")
    wanted = WEIGHT_LAYOUT
    if meta.get("weight_layout") != wanted:
        raise ValueError("materialization layout was not qualified")
    positions = ts["positions"].reshape(-1)
    if (
        positions.numel() != ROWS
        or int(positions.min()) < 1
        or not bool((positions[1:] - positions[:-1] == 1).all())
    ):
        raise ValueError("not the declared real single-request verify")
    return {
        "valid": True,
        "rank": ident["rank"],
        "epoch": ident["epoch"],
        "bytes": total,
        "U": len(set(ids.reshape(-1).tolist())),
        "test_fixture": False,
    }


def initialize_snapshot():
    global SNAP, load_rank_snapshot, restore_tensors, finalize_manifest
    if SNAP is None:
        SNAP = helper("front_floor_snapshot", "_front_floor_snapshot_v1")
        SNAP.add_route_views = add_route_views
        SNAP.validate_snapshot = validate_snapshot
        load_rank_snapshot = SNAP.load_rank_snapshot
        restore_tensors = SNAP.restore_tensors
        finalize_manifest = SNAP.finalize_manifest


def key_meta(runner, build, layer, core, prefix, blocks):
    key = build.k2_keys[layer.layer_id]
    if (
        layer.layer_id != LAYER
        or layer.kda is None
        or layer.moe is None
        or layer.moe.write_idx != -1
    ):
        raise RuntimeError("selected current layer is not a non-writing KDA+K2 layer")
    if (
        key.tokens != ROWS
        or key.nblocks != NB
        or key.diag
        or key.timeline
        or not key.reduce
    ):
        raise RuntimeError("actual current key differs from the predeclared shape")
    ids, positions, qlen = _CAPTURE._TLS.forward
    if qlen != ROWS or ids.numel() != ROWS or positions.numel() != ROWS:
        raise RuntimeError("not the current K3 verify key")
    w = layer.moe
    tensors = {
        "core": core,
        "prefix_before": prefix,
        "blocks": blocks[:, :NB, :],
        "input_ids": ids,
        "positions": positions,
        "w_o": w.w_o,
        "ares_nw": w.seam.nw,
        "ares_qk": w.seam.qk,
        "in_nw": w.seam.out_nw,
        "w_gate": w.w_gate,
        "bias": w.bias,
        "w_ld": w.w_ld,
        "w_sgu": w.w_sgu,
        "w_sd": w.w_sd,
        "w13": w.w13,
        "w13s": w.w13s,
        "w2": w.w2,
        "w2s": w.w2s,
        "ln_w": w.ln_w,
        "w_up": w.w_up,
    }
    expert = runner.inner.layers[LAYER].block_sparse_moe.experts
    layout = {
        "gate_up_interleaved": bool(
            getattr(expert.quant_method, "is_guinterleave", True)
        ),
        "fp8_activations": bool(getattr(expert.quant_method, "fp8_activations", False)),
        "w13_is_shuffled": bool(getattr(expert.w13_weight, "is_shuffled", False)),
        "w2_is_shuffled": bool(getattr(expert.w2_weight, "is_shuffled", False)),
        "triton_swizzle": getattr(expert, "w13_swizzle_layout", None) is not None,
        "mega_owns_weights": getattr(expert, "_mega_w1", None) is not None,
    }
    wanted = WEIGHT_LAYOUT
    if layout != wanted or runner.tp != TP:
        raise RuntimeError("current materialized weight/TP treatment differs")
    return (
        key,
        tensors,
        {
            "key": dataclasses.asdict(key),
            "identity": {
                "fixture_kind": "natural_k2",
                "rank": int(runner.rank),
                "tp": TP,
                "model_layer": LAYER,
                "rows": ROWS,
            },
            "abi_scalars": {
                "blk_sm": blocks.stride(0),
                "blk_sr": blocks.stride(1),
                "layer": SLOT,
                "reset": 0,
            },
            "weight_layout": layout,
            "block_read_scope": "all declared read blocks with original row/block strides",
            "prefix_rule": "non-writing layer still updates prefix; entry and exit are distinct",
        },
    )


def exit_tensors(runner, key, prefix, output):
    import torch

    from atom.models.kimi_k3.mono.kernels import layer

    xoff, xn = layer.plain_layout(key)["xrow"]
    roff, rn = layer.scratch_layout(key)["route"]
    return {
        "prefix_after": prefix,
        "out": output,
        "x": runner.plain[xoff : xoff + xn].view(torch.bfloat16).reshape(ROWS, 7168),
        "route_pairs": runner.scratch[roff : roff + rn]
        .view(torch.int32)
        .reshape(ROWS, 32, 2),
    }


def install(plan):
    global _CAPTURE
    initialize_snapshot()
    import torch

    from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
    from atom.utils.forward_context import get_forward_context, request_bucket
    from atom.utils.region_graph import ReqGraph

    cap = _CAPTURE = helper("front_floor_capture")
    cap._key_meta, cap._exit_tensors, cap.validate_snapshot = (
        key_meta,
        exit_tensors,
        validate_snapshot,
    )
    cap.RUN_ROOT = Path(plan["output"]) / "data"
    cap.CAPTURE_PHASE = "k31_current_k2_front_capture"
    cap._source_info()
    original_forward, original_k2 = K3MonoDecodeRunner.forward, K3MonoDecodeRunner._k2
    original_record, original_replay = ReqGraph.record, ReqGraph.replay
    target = request_bucket(1, ROWS)

    @functools.wraps(original_forward)
    def forward(self, ids, positions, qlen):
        previous = getattr(cap._TLS, "forward", None)
        cap._TLS.forward = (ids, positions, qlen)
        try:
            return original_forward(self, ids, positions, qlen)
        finally:
            cap._TLS.forward = previous

    @functools.wraps(original_k2)
    def k2(self, build, layer, core, prefix, blocks):
        mode = getattr(cap._TLS, "mode", None)
        if mode not in ("prepare", "capture") or layer.layer_id != LAYER:
            return original_k2(self, build, layer, core, prefix, blocks)
        key, tensors, meta = key_meta(self, build, layer, core, prefix, blocks)
        entry, aliases = SNAP.describe_stage(tensors, "entry")
        if mode == "capture":
            cap._CONTROLLER.before(entry, meta)
        output = original_k2(self, build, layer, core, prefix, blocks)
        ending, _ = SNAP.describe_stage(
            exit_tensors(self, key, prefix, output), "exit", aliases
        )
        add_route_views(ending)
        if mode == "capture":
            cap._CONTROLLER.after(ending)
        else:
            cap._TLS.plans.append((self, key, meta, entry, ending))
        return output

    @functools.wraps(original_record)
    def record(self, fn, args=(), kwargs=None, key=None, phase=None):
        result = original_record(
            self, fn, args=args, kwargs=kwargs, key=key, phase=phase
        )
        if phase is not None or key != target or cap._CONTROLLER is not None:
            return result
        cap._TLS.plans = []
        with cap._mode("prepare"):
            fn(*args, **(kwargs or {}))
        if not cap._TLS.plans:
            return result
        if len(cap._TLS.plans) != 1:
            raise RuntimeError("capture graph did not contain exactly one selected K2")
        runner, selected, meta, entry, ending = cap._TLS.plans.pop()
        c = cap._Capture(runner, selected, dict(meta), entry, ending)
        c.meta_at_prepare = meta
        c.graph, c.graph_key = self, key
        c.allocate()
        cap._CONTROLLER = c
        with cap._mode("capture"):
            original_record(
                self, fn, args=args, kwargs=kwargs, key=key, phase=cap.CAPTURE_PHASE
            )
        c.recorded = True
        return result

    @functools.wraps(original_replay)
    def replay(self, args=(), kwargs=None, key=None, phase=None):
        global _WITNESS
        c = cap._CONTROLLER
        if not _LIVE or c is None or c.closed or self is not c.graph or key != target:
            return original_replay(self, args=args, kwargs=kwargs, key=key, phase=phase)
        ctx = get_forward_context().context
        if (
            phase is not None
            or ctx is None
            or getattr(ctx, "is_dummy", False)
            or ctx.scheduled_bs != 1
            or ctx.scheduled_tokens != ROWS
        ):
            raise RuntimeError("selected graph is not the declared real invocation")
        c.replay_ordinal += 1
        witness = {
            "key": list(key),
            "replay_ordinal": c.replay_ordinal,
            "rows": ROWS,
            "requests": 1,
            "runner": type(c.runner).__name__,
            "key_fields": c.meta["key"],
        }
        if _WITNESS is None:
            result = original_replay(
                self, args=args, kwargs=kwargs, key=key, phase=phase
            )
            torch.cuda.synchronize()
            _WITNESS = witness
            c.prepare(directory=plan["output"] + "/data", capture_id="capture")
            c.arm_id, c.request_hint = "real-verify-after-witness", plan["request"]
            flags = [0] * cap._backend().WORDS
            flags[cap._backend().ARMED] = 1
            c.state.copy_(
                torch.tensor(flags, dtype=torch.int64, device=c.runner.device)
            )
            c.pending = True
            return result
        if not c.pending:
            return original_replay(self, args=args, kwargs=kwargs, key=key, phase=phase)
        c.meta["identity"]["ordinary_served_witness"] = _WITNESS
        c.selected = witness
        c.pending = False
        return original_replay(
            self, args=args, kwargs=kwargs, key=key, phase=cap.CAPTURE_PHASE
        )

    K3MonoDecodeRunner.forward, K3MonoDecodeRunner._k2 = forward, k2
    ReqGraph.record, ReqGraph.replay = record, replay
    cap._HOOKS = [
        (K3MonoDecodeRunner, "forward", forward, original_forward),
        (K3MonoDecodeRunner, "_k2", k2, original_k2),
        (ReqGraph, "record", record, original_record),
        (ReqGraph, "replay", replay, original_replay),
    ]


def runner(rank, config):
    """Config.runner_qualname entry; ordinary serving never selects this factory."""
    from atom.model_engine.model_runner import ModelRunner

    plan = json.loads(Path(os.environ["K31_K2_FRONT_PLAN"]).read_text())
    install(plan)

    class CaptureRunner(ModelRunner):
        def capture_cudagraph(self):
            global _LIVE
            result = super().capture_cudagraph()
            if _CAPTURE._CONTROLLER is None:
                raise RuntimeError("declared current Mono/key is not exposed")
            _LIVE = True
            return result

        def forward(self, batch):
            result = super().forward(batch)
            c = _CAPTURE._CONTROLLER
            if c is not None and c.selected is not None and c.receipt is None:
                receipt = c.dump()
                (Path(plan["output"]) / "data" / f"rank{rank}.json").write_text(
                    json.dumps(
                        {
                            "state": "COMPLETE",
                            "rank": rank,
                            "capture": receipt,
                            "served_witness": _WITNESS,
                        }
                    )
                )
                c.disarm()
            return result

    return CaptureRunner(rank, config)


def worker(plan, path):
    import subprocess
    from urllib.request import urlopen

    import replay

    replay.check_plan(plan, path, worker=True)
    initialize_snapshot()
    import torch

    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import LLMEngine
    from atom.sampling_params import SamplingParams

    data = Path(plan["output"]) / "data"
    data.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cache_log = (Path(plan["output"]) / "cache.log").open("w")
    cache = subprocess.Popen(
        plan["cache"],
        stdout=cache_log,
        stderr=subprocess.STDOUT,
        env=plan["environment"],
    )
    engine = None
    try:
        end = time.monotonic() + 60
        while True:
            if cache.poll() is not None:
                raise RuntimeError("cache service exited")
            try:
                with urlopen(
                    "http://127.0.0.1:8695/healthcheck", timeout=2
                ) as response:
                    if json.load(response).get("status") == "healthy":
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= end:
                raise TimeoutError("cache readiness exceeded 60 seconds")
            time.sleep(0.2)
        parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
        args = EngineArgs.from_cli_args(parser.parse_args(plan["model_args"]))
        kwargs = args._get_engine_kwargs()
        kwargs["runner_qualname"] = "spike.k31_k2_front.capture.runner"
        engine = LLMEngine(args.model, **kwargs)
        request = plan["request"]
        outputs = engine.generate(
            [request["prompt"]],
            SamplingParams(
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
                ignore_eos=request["ignore_eos"],
            ),
            request_ids=["k31-k2-front-capture"],
        )
        if len(outputs) != 1:
            raise RuntimeError("the fixed real request did not complete")
        evidence = {
            "natural_probe_passed": True,
            "excluded_from_timing": True,
            "request": request,
            "output_sha256": hashlib.sha256(
                json.dumps(outputs, sort_keys=True).encode()
            ).hexdigest(),
            "recipe_sources": plan["sources"],
            "treatment": plan["treatment"],
            "scope": "one real request under current declared Mono treatment; not natural quality or C12 performance",
        }
        result = finalize_manifest(data, evidence)
        (data / "request.json").write_text(json.dumps(evidence, indent=2))
        print(
            json.dumps(
                {
                    "state": "CAPTURE_COMPLETE",
                    "manifest": str(data / "manifest.json"),
                    "rank_files": len(result["ranks"]),
                }
            )
        )
    finally:
        if engine is not None:
            engine.close()
        cache.terminate()
        try:
            cache.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cache.kill()
            cache.wait(timeout=15)
        cache_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    worker(json.loads(args.plan.read_text()), args.plan)
