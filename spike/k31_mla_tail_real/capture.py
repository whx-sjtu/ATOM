"""One actual all-rank native MLA tuple through a named one-shot graph."""

import argparse
import ast
import functools
import hashlib
import importlib.util
import json
import os
import sys
import time  # noqa: F401 - global consumed by the pinned model driver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path("/app/k31-agent-team-private/task-19")
ROWS, LAYER, TP = 4, 3, 8
CAP = SNAP = ROUTE = None
PINS = {
    "snapshot/front_floor_capture.py": "739032996c1a121c67455b18ac57466a31e9b27cf5239ab6071dd15568127613",
    "snapshot/front_floor_snapshot.py": "bb4a7370f5a7fe095c1cda886491e2934191d7bb183c0319a1a85e9f2c37734e",
    "snapshot/front_floor_copy.py": "6afe5eed4475489802e7617cef9029106639a02b0683acf521020e4bb71f7a4f",
    "task15/capture.py": "48c1996bb90d7072e096da0a5692b3c8b67419ea24531baa8b472192e9314709",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def initialize_snapshot():
    global CAP, SNAP
    if CAP is not None:
        return
    for name, expected in PINS.items():
        assert sha(PRIVATE / "deps" / name) == expected
    assert (
        "_front_floor_snapshot_v1" not in sys.modules
    ), "shared capture namespace is not qualified"
    path = PRIVATE / "deps/snapshot/front_floor_capture.py"
    spec = importlib.util.spec_from_file_location("_mla19_capture_tools", path)
    CAP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CAP)
    SNAP = CAP._module
    SNAP.SCHEMA = "k31.mla_tail.actual.v1"
    SNAP.MAX_BYTES = 32 << 20
    SNAP.WEIGHTS = ("w_v",)
    SNAP.REQUIRED = (
        "raw",
        "lse",
        "owned",
        "w_v",
        "input_ids",
        "positions",
        "gate",
        "reference",
    )
    SNAP.validate_snapshot = CAP.validate_snapshot = validate_snapshot
    CAP._source_info = functools.partial(source_info, CAP._source_info)


def source_info(original):
    result = original()
    assert Path(result["aiter_root"]).resolve() == PRIVATE / "aiter-572232d"
    return result


def workspace_info(workspace):
    from aiter.ops.flydsl.dcp_mailbox.combine import peer_layout
    from aiter.ops.flydsl.dcp_mailbox.common import epoch_offset

    layout = peer_layout(TP, workspace.max_tokens, 12, 128)
    return {
        "layout": {name: list(span) for name, span in layout.items()},
        "epoch_offset": epoch_offset(layout),
        "bytes": workspace.peers.bytes.numel(),
        "base": workspace.peers.bytes.data_ptr(),
        "state_base": workspace.state.data_ptr(),
        "debug": workspace.debug,
    }


def views(snapshot):
    return SNAP.restore_tensors(snapshot, device="cpu")


def validate_snapshot(snapshot, **kwargs):
    import torch

    ident, key = snapshot["meta"]["identity"], snapshot["meta"]["key"]
    assert snapshot["schema"] == SNAP.SCHEMA and ident["fixture_kind"] == "native_mla"
    assert not kwargs.get("allow_test_fixture") and not ident.get("test_fixture")
    assert (ident["tp"], ident["model_layer"], ident["rows"]) == (TP, LAYER, ROWS)
    assert (
        0 <= ident["rank"] < TP
        and ident["epoch"] > 0
        and snapshot["meta"]["capture_done"]
    )
    assert ident["ordinary_served_witness"]["replay_ordinal"] == 1
    assert ident["selected_graph"]["replay_ordinal"] == 2
    assert snapshot["meta"]["abi_scalars"] == {
        "decode": True,
        "lse_base_e": True,
        "quantized_gate": False,
    }
    assert snapshot["meta"]["captured_route"] == {"mailbox_calls": 1, **key}
    assert set(snapshot["tensors"]) == set(SNAP.REQUIRED)
    assert sum(a.numel() for a in snapshot["arenas"].values()) <= 32 << 20
    for name, blob in snapshot["arenas"].items():
        assert (
            blob.device.type == "cpu"
            and blob.dtype == torch.uint8
            and blob.is_contiguous()
        )
        assert SNAP.arena_hash(blob) == snapshot["arena_hashes"][name]
    for name, desc in snapshot["tensors"].items():
        assert (
            SNAP.tensor_hash(snapshot["arenas"][desc["arena"]], desc)
            == snapshot["tensor_hashes"][name]
        )
    tensors = views(snapshot)
    shapes = {
        "raw": (ROWS, 96, 512),
        "lse": (ROWS, 96),
        "owned": (ROWS,),
        "w_v": (96, 128, 512),
        "gate": (ROWS, 1536),
        "reference": (ROWS, 1536),
    }
    for name, shape in shapes.items():
        assert tuple(tensors[name].shape) == shape, name
        dtype = (
            torch.float32
            if name == "lse"
            else torch.int32 if name == "owned" else torch.bfloat16
        )
        assert tensors[name].dtype == dtype, name
    assert bool((tensors["owned"] >= 0).all()) and tensors["owned"].is_contiguous()
    assert tensors["w_v"].is_contiguous() and tensors["gate"].is_contiguous()
    assert tensors["raw"].stride(-1) == 1 and all(
        st % 2 == 0 for st in tensors["raw"].stride()[:2]
    )
    assert all(
        bool(torch.isfinite(tensors[name]).all())
        for name in ("w_v", "gate", "reference")
    )
    live = tensors["owned"] > 0
    assert bool(torch.isfinite(tensors["raw"][live]).all()) and bool(
        torch.isfinite(tensors["lse"][live]).all()
    )
    assert ROWS <= key["workspace_capacity"] <= (1 << 21) // (TP * 12 * 128)
    assert key["token_tile"] == 1
    workspace = snapshot["meta"]["workspace"]
    slots = TP * key["workspace_capacity"] * 12
    assert workspace["layout"] == {
        "data": [0, slots * 256],
        "lse": [slots * 256, slots * 4],
    }
    assert workspace["epoch_offset"] == 2 * slots * 260
    assert workspace["bytes"] == workspace["epoch_offset"] + TP * 4
    assert (
        not workspace["debug"] and workspace["base"] > 0 and workspace["state_base"] > 0
    )
    assert key["raw_stride"] == list(tensors["raw"].stride())
    assert key["lse_stride"] == list(tensors["lse"].stride())
    assert tensors["input_ids"].numel() == tensors["positions"].numel() == ROWS
    assert tensors["input_ids"].dtype == tensors["positions"].dtype == torch.int64
    assert tensors["input_ids"].reshape(-1).tolist() == ident["selected_input_ids"]
    assert tensors["positions"].reshape(-1).tolist() == ident["selected_positions"]
    assert bool(
        (
            tensors["positions"].reshape(-1)[1:] - tensors["positions"].reshape(-1)[:-1]
            == 1
        ).all()
    )
    assert int(tensors["positions"].min()) >= 1
    gate_alias = snapshot["tensors"]["gate"]["source_alias_group"]
    assert all(
        snapshot["tensors"][name]["source_alias_group"] != gate_alias
        for name in ("raw", "lse", "owned", "w_v")
    ), "temporal gate/input alias needs separate producer accounting"
    return {
        "valid": True,
        "rank": ident["rank"],
        "epoch": ident["epoch"],
        "test_fixture": False,
        "workspace_capacity": key["workspace_capacity"],
        "token_tile": key["token_tile"],
    }


def invalid_context(ctx):
    # Class defaults cannot certify missing per-invocation identity.
    fields = getattr(ctx, "__dict__", {})
    return (
        fields.get("is_dummy_run") is not False
        or fields.get("scheduled_bs") != 1
        or fields.get("scheduled_tokens") != ROWS
    )


def inherited(name, namespace, *, nested=False):
    """Reuse the pinned routing or driver body with explicit local bindings."""
    path = PRIVATE / "deps/task15/capture.py"
    assert sha(path) == PINS["task15/capture.py"]
    tree = ast.parse(path.read_text())
    parent = (
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "install"
        )
        if nested
        else tree
    )
    node = next(
        n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    substitutions = {
        "spike.k31_k2_front.capture.runner": "spike.k31_mla_tail_real.capture.runner",
        "http://127.0.0.1:8695/healthcheck": "http://127.0.0.1:8699/healthcheck",
        "k31-k2-front-capture": "k31-mla-tail-real",
        "capture graph did not contain exactly one selected K2": "capture graph did not contain exactly one selected MLA",
    }
    context_bindings = 0
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Constant)
            and type(item.value) is str
            and item.value in substitutions
        ):
            item.value = substitutions[item.value]
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "getattr"
            and len(item.args) == 3
            and isinstance(item.args[1], ast.Constant)
            and item.args[1].value == "is_dummy"
        ):
            item.func.id = "invalid_context"
            item.args = item.args[:1]
            context_bindings += 1
    if name == "replay":
        assert nested and context_bindings == 1, "current Context guard was not rebound"
    exec(  # noqa: S102 - pinned body and explicit local bindings
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def finalize_manifest(directory, evidence):
    weights = {
        SNAP.load_rank_snapshot(Path(directory) / f"rank{rank}.pt", verify=False)[
            "tensor_hashes"
        ]["w_v"]
        for rank in range(TP)
    }
    assert len(weights) == 1, "DCP replicated weights differ across ranks"
    result = SNAP.finalize_manifest(directory, evidence)
    ranks = [
        SNAP.load_rank_snapshot(Path(directory) / f"rank{rank}.pt")
        for rank in range(TP)
    ]
    assert {s["meta"]["hardware"]["bdf"] for s in ranks} == set(
        evidence["selected_bdfs"]
    )
    result["weights"] = "actual materialized BF16 W_V_dcp on all eight ranks"
    Path(directory, "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def install(plan):
    global ROUTE
    initialize_snapshot()
    import torch
    from aiter.ops.flydsl.dcp_mailbox.common import token_tile

    from atom.model_ops import dcp_ops
    from atom.model_ops.attention_mla import MLAAttention
    from atom.models.kimi_k3 import model
    from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
    from atom.utils.forward_context import get_forward_context, request_bucket
    from atom.utils.region_graph import ReqGraph

    CAP.RUN_ROOT = Path(plan["output"]) / "data"
    CAP.CAPTURE_PHASE = "k31_actual_mla_capture"
    CAP._source_info()
    forward0, core0 = K3MonoDecodeRunner.forward, model.KimiFullAttention._gated_core
    tail0, gate0 = (
        MLAAttention._dcp_project_merge_out,
        model.fused_sigmoid_mul_maybe_quant,
    )

    @functools.wraps(forward0)
    def forward(self, ids, positions, qlen):
        CAP._TLS.forward, CAP._TLS.runner = (ids, positions, qlen), self
        try:
            return forward0(self, ids, positions, qlen)
        finally:
            CAP._TLS.forward = CAP._TLS.runner = None

    @functools.wraps(core0)
    def core(self, *args, **kwargs):
        CAP._TLS.selected = self.layer_num == LAYER and getattr(
            CAP._TLS, "mode", None
        ) in ("prepare", "capture")
        try:
            return core0(self, *args, **kwargs)
        finally:
            CAP._TLS.selected = False
            CAP._TLS.entry = None

    @functools.wraps(tail0)
    def tail(
        self, out, lse, ctx=None, merge_in_fp32=False, owned_counts=None, decode=False
    ):
        selected = getattr(CAP._TLS, "selected", False)
        if selected:
            owner = CAP._TLS.runner
            ids, positions, qlen = CAP._TLS.forward
            assert decode and not merge_in_fp32 and self.pbm_enabled and qlen == ROWS
            assert self.dcp_world_size == TP and self.dcp_comm_backend == "mailbox"
            assert (
                isinstance(self.o_proj, torch.nn.Identity)
                and self.W_V_dcp_scale is None
            )
            assert (
                self.W_V_dcp.dtype == out.dtype == torch.bfloat16
                and owned_counts is not None
            )
            workspace_key = dcp_ops._mailbox_key(self.dcp_group, 12, 128)
            workspace = dcp_ops._MAILBOX_WORKSPACES[workspace_key]
            key = {
                "workspace_capacity": workspace.max_tokens,
                "token_tile": token_tile(out.shape[0]),
                "raw_stride": list(out.stride()),
                "lse_stride": list(lse.stride()),
            }
            meta = {
                "key": key,
                "identity": {
                    "fixture_kind": "native_mla",
                    "rank": owner.rank,
                    "tp": TP,
                    "model_layer": LAYER,
                    "rows": ROWS,
                },
                "abi_scalars": {
                    "decode": True,
                    "lse_base_e": True,
                    "quantized_gate": False,
                },
            }
            entry, aliases = SNAP.describe_stage(
                {
                    "raw": out,
                    "lse": lse,
                    "owned": owned_counts,
                    "w_v": self.W_V_dcp,
                    "input_ids": ids,
                    "positions": positions,
                },
                "entry",
            )
            CAP._TLS.entry = (owner, None, meta, entry, aliases)
            if CAP._TLS.mode == "capture":
                CAP._CONTROLLER.before(entry, meta)
                CAP._CONTROLLER.entry = entry
                CAP._CONTROLLER.meta["workspace"] = workspace_info(workspace)
            before = dcp_ops._MAILBOX_LAUNCHES.get(workspace_key, 0)
        result = tail0(
            self,
            out,
            lse,
            ctx=ctx,
            merge_in_fp32=merge_in_fp32,
            owned_counts=owned_counts,
            decode=decode,
        )
        if selected and torch.cuda.is_current_stream_capturing():
            count = dcp_ops._MAILBOX_LAUNCHES.get(workspace_key, 0) - before
            assert count == 1
            CAP._CONTROLLER.meta["captured_route"] = {"mailbox_calls": count, **key}
        return result

    @functools.wraps(gate0)
    def gate(attn_output, gate_value, quant=False, **kwargs):
        result, scale = gate0(attn_output, gate_value, quant=quant, **kwargs)
        if getattr(CAP._TLS, "selected", False):
            assert not quant and scale is None
            owner, key, meta, entry, aliases = CAP._TLS.entry
            ending, _ = SNAP.describe_stage(
                {"gate": gate_value, "reference": result}, "exit", aliases
            )
            if CAP._TLS.mode == "capture":
                CAP._CONTROLLER.after(ending)
                CAP._CONTROLLER.exit = ending
            else:
                CAP._TLS.plans.append((owner, key, meta, entry, ending))
        return result, scale

    ROUTE = dict(
        globals(),
        cap=CAP,
        torch=torch,
        SNAP=SNAP,
        plan=plan,
        _LIVE=False,
        _WITNESS=None,
        target=request_bucket(1, ROWS),
        get_forward_context=get_forward_context,
        original_record=ReqGraph.record,
        original_replay=ReqGraph.replay,
    )
    record = inherited("record", ROUTE, nested=True)
    route_replay = inherited("replay", ROUTE, nested=True)

    @functools.wraps(route_replay)
    def replay(self, args=(), kwargs=None, key=None, phase=None):
        controller = CAP._CONTROLLER
        if (
            ROUTE["_LIVE"]
            and controller is not None
            and controller.pending
            and self is controller.graph
            and key == ROUTE["target"]
            and phase is None
        ):
            if invalid_context(ROUTE["get_forward_context"]().context):
                raise RuntimeError("selected graph is not the declared real invocation")
            buffers = controller.model_runner.forward_vars
            ids, positions = (
                buffers["input_ids"].gpu[:ROWS],
                buffers["positions"].gpu[:ROWS],
            )
            for name, tensor in (("input_ids", ids), ("positions", positions)):
                desc = controller.entry["tensors"][name]
                arena = controller.entry["arenas"][desc["arena"]]
                assert (
                    tensor.data_ptr() == arena["source_base"] + desc["offset_bytes"]
                ), "graph identity address changed"
            controller.meta["identity"]["selected_input_ids"] = (
                ids.detach().cpu().reshape(-1).tolist()
            )
            controller.meta["identity"]["selected_positions"] = (
                positions.detach().cpu().reshape(-1).tolist()
            )
            controller.meta["identity_copy_scope"] = (
                "two tiny D2H copies before the selected replay; capture-only, excluded from timing"
            )
        return route_replay(self, args=args, kwargs=kwargs, key=key, phase=phase)

    hooks = [
        (K3MonoDecodeRunner, "forward", forward),
        (model.KimiFullAttention, "_gated_core", core),
        (MLAAttention, "_dcp_project_merge_out", tail),
        (model, "fused_sigmoid_mul_maybe_quant", gate),
        (ReqGraph, "record", record),
        (ReqGraph, "replay", replay),
    ]
    CAP._HOOKS = [
        (owner, name, wrapper, getattr(owner, name)) for owner, name, wrapper in hooks
    ]
    for owner, name, wrapper in hooks:
        setattr(owner, name, wrapper)


def runner(rank, config):
    from atom.model_engine.model_runner import ModelRunner

    plan = json.loads(Path(os.environ["K31_MLA_TAIL_REAL_PLAN"]).read_text())
    install(plan)

    class CaptureRunner(ModelRunner):
        def capture_cudagraph(self):
            result = super().capture_cudagraph()
            if CAP._CONTROLLER is None or "captured_route" not in CAP._CONTROLLER.meta:
                raise RuntimeError(
                    "MISSING_NATIVE_INPUT: declared current Mono/key was not exposed"
                )
            CAP._CONTROLLER.meta["cost"][
                "scope"
            ] = "one actual MLA invocation; capture/dump excluded from timing"
            import replay

            CAP._CONTROLLER.model_runner = self
            CAP._CONTROLLER.meta["hardware"] = replay.device_identity(rank, plan)
            ROUTE["_LIVE"] = True
            return result

        def forward(self, batch):
            result = super().forward(batch)
            controller = CAP._CONTROLLER
            if (
                controller is not None
                and controller.selected is not None
                and controller.receipt is None
            ):
                receipt = controller.dump()
                Path(plan["output"], "data", f"rank{rank}.json").write_text(
                    json.dumps({"state": "COMPLETE", "rank": rank, "capture": receipt})
                )
                cleanup = controller.disarm()
                Path(plan["output"], "data", f"rank{rank}.cleanup.json").write_text(
                    json.dumps(cleanup)
                )
            return result

    return CaptureRunner(rank, config)


def worker(plan, path):
    initialize_snapshot()
    original_finalize = finalize_manifest

    def finalize(directory, evidence):
        return original_finalize(
            directory, {**evidence, "selected_bdfs": plan["selected_bdfs"]}
        )

    return inherited("worker", dict(globals(), finalize_manifest=finalize))(plan, path)


def cpu_checks():
    """Controlled CPU tensors and native wrapper bodies; never real model evidence."""
    import copy
    from dataclasses import dataclass
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace as NS
    from unittest.mock import patch

    import torch

    initialize_snapshot()
    raw = torch.zeros(ROWS, 96, 514, dtype=torch.bfloat16)[:, :, 1:513]
    ids_positions = torch.arange(8, dtype=torch.int64)
    tensors = {
        "raw": raw,
        "lse": torch.zeros(96, ROWS).t(),
        "owned": torch.ones(ROWS, dtype=torch.int32),
        "w_v": torch.ones(96, 128, 512, dtype=torch.bfloat16),
        "input_ids": ids_positions[:4],
        "positions": ids_positions[4:],
        "gate": torch.ones(ROWS, 1536, dtype=torch.bfloat16),
        "reference": torch.zeros(ROWS, 1536, dtype=torch.bfloat16),
    }
    entry, aliases = SNAP.describe_stage(
        {k: v for k, v in tensors.items() if k not in ("gate", "reference")}, "entry"
    )
    ending, _ = SNAP.describe_stage(
        {k: tensors[k] for k in ("gate", "reference")}, "exit", aliases
    )
    arenas = {}
    for layout in (entry, ending):
        for name, arena in layout["arenas"].items():
            first = next(k for k, d in layout["tensors"].items() if d["arena"] == name)
            source = torch.empty(0, dtype=torch.uint8).set_(
                tensors[first].untyped_storage(),
                0,
                (tensors[first].untyped_storage().nbytes(),),
                (1,),
            )
            blob = torch.full((arena["nbytes"],), 165, dtype=torch.uint8)
            for offset, size in arena["copy_spans"]:
                begin = arena["source_min_offset_bytes"] + offset
                blob[offset : offset + size].copy_(source[begin : begin + size])
            arenas[name] = blob
    key = {
        "workspace_capacity": 96,
        "token_tile": 1,
        "raw_stride": list(raw.stride()),
        "lse_stride": list(tensors["lse"].stride()),
    }
    slots = TP * 96 * 12
    workspace = {
        "layout": {"data": [0, slots * 256], "lse": [slots * 256, slots * 4]},
        "epoch_offset": 2 * slots * 260,
        "bytes": 2 * slots * 260 + TP * 4,
        "base": 4096,
        "state_base": 8192,
        "debug": False,
    }
    meta = {
        "key": key,
        "workspace": workspace,
        "capture_done": True,
        "captured_route": {"mailbox_calls": 1, **key},
        "abi_scalars": {"decode": True, "lse_base_e": True, "quantized_gate": False},
        "identity": {
            "fixture_kind": "native_mla",
            "tp": TP,
            "model_layer": LAYER,
            "rows": ROWS,
            "rank": 0,
            "epoch": 1,
            "ordinary_served_witness": {"replay_ordinal": 1},
            "selected_graph": {"replay_ordinal": 2},
            "selected_input_ids": tensors["input_ids"].tolist(),
            "selected_positions": tensors["positions"].tolist(),
        },
    }
    snapshot = SNAP.seal_snapshot(
        {
            "schema": SNAP.SCHEMA,
            "meta": meta,
            "arenas": arenas,
            "tensors": {**entry["tensors"], **ending["tensors"]},
            "arena_layouts": {**entry["arenas"], **ending["arenas"]},
        }
    )
    assert validate_snapshot(snapshot)["valid"]
    restored = views(snapshot)
    for name, tensor in tensors.items():
        assert (
            torch.equal(restored[name], tensor)
            and restored[name].stride() == tensor.stride()
        )
    assert (
        restored["input_ids"].untyped_storage().data_ptr()
        == restored["positions"].untyped_storage().data_ptr()
    )
    failures = []
    for name, edit in (
        (
            "stale_identity",
            lambda s: s["meta"]["identity"].update(selected_input_ids=[9] * 4),
        ),
        (
            "wrong_ordinal",
            lambda s: s["meta"]["identity"]["selected_graph"].update(replay_ordinal=1),
        ),
        ("missing_rank_field", lambda s: s["meta"]["identity"].pop("rank")),
        ("wrong_tile", lambda s: s["meta"]["key"].update(token_tile=2)),
        ("wrong_mailbox_layout", lambda s: s["meta"]["workspace"].update(bytes=1)),
        (
            "temporal_gate_alias",
            lambda s: s["tensors"]["gate"].update(
                source_alias_group=s["tensors"]["raw"]["source_alias_group"]
            ),
        ),
        ("mutated_arena", lambda s: s["arenas"][next(iter(s["arenas"]))].zero_()),
        ("missing_tensor", lambda s: s["tensors"].pop("owned")),
    ):
        candidate = copy.deepcopy(snapshot)
        edit(candidate)
        try:
            validate_snapshot(candidate)
        except (AssertionError, KeyError, ValueError):
            failures.append(name)
        else:
            raise AssertionError("accepted negative control: " + name)
    ops = NS(
        _MAILBOX_WORKSPACES={"workspace": NS(max_tokens=96)},
        _MAILBOX_LAUNCHES={},
        _mailbox_key=lambda *a: "workspace",
    )
    owner = NS(rank=0, device="cpu")
    attention = NS(
        pbm_enabled=True,
        dcp_world_size=8,
        dcp_comm_backend="mailbox",
        dcp_group=object(),
        o_proj=torch.nn.Identity(),
        W_V_dcp=tensors["w_v"],
        W_V_dcp_scale=None,
    )
    scope = dict(
        globals(),
        torch=torch,
        dcp_ops=ops,
        token_tile=lambda rows: 1,
        workspace_info=lambda ws: workspace,
    )

    def original_tail(*args, **kwargs):
        if CAP._TLS.mode == "capture":
            ops._MAILBOX_LAUNCHES["workspace"] = (
                ops._MAILBOX_LAUNCHES.get("workspace", 0) + 1
            )
        return tensors["reference"]

    def original_core(*args, **kwargs):
        output = scope["tail"](
            attention, raw, tensors["lse"], owned_counts=tensors["owned"], decode=True
        )
        return scope["gate"](output, tensors["gate"])

    scope.update(
        tail0=original_tail,
        gate0=lambda out, gate, **kw: (out, None),
        core0=original_core,
        forward0=lambda *args: scope["core"](NS(layer_num=LAYER)),
    )
    install_node = next(
        n
        for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "install"
    )
    nodes = {n.name: n for n in install_node.body if isinstance(n, ast.FunctionDef)}
    for name in ("forward", "core", "tail", "gate"):
        exec(  # noqa: S102 - current wrapper body under controlled CPU bindings
            compile(ast.Module(body=[nodes[name]], type_ignores=[]), __file__, "exec"),
            scope,
        )
    CAP._TLS.plans = []
    with patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: CAP._TLS.mode == "capture"
    ):
        with CAP._mode("prepare"):
            scope["forward"](owner, tensors["input_ids"], tensors["positions"], ROWS)
        assert len(CAP._TLS.plans) == 1 and CAP._TLS.forward is None
        _, _, planned_meta, planned_entry, planned_exit = CAP._TLS.plans.pop()
        controller = CAP._Capture(
            owner, None, dict(planned_meta), planned_entry, planned_exit
        )
        controller.meta_at_prepare = planned_meta

        def check_entry(layout, meta):
            assert meta == planned_meta
            assert SNAP.layout_signature(layout) == SNAP.layout_signature(planned_entry)

        def check_exit(layout):
            assert SNAP.layout_signature(layout) == SNAP.layout_signature(planned_exit)

        controller.before, controller.after = check_entry, check_exit
        CAP._CONTROLLER = controller
        with CAP._mode("capture"):
            scope["forward"](owner, tensors["input_ids"], tensors["positions"], ROWS)
        assert controller.meta["captured_route"]["mailbox_calls"] == 1
        assert CAP._TLS.entry is None and CAP._TLS.runner is None
        for tree in (controller.entry, controller.exit):
            assert "Tensor" not in str(tree)
    with TemporaryDirectory() as directory:
        graph, calls = object(), []
        context_path = ROOT / "atom/utils/forward_context.py"
        context_class = next(
            n
            for n in ast.parse(context_path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "Context"
        )
        context_scope = {"torch": torch, "dataclass": dataclass, "__name__": __name__}
        context_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                context_class,
            ],
            type_ignores=[],
        )
        exec(  # noqa: S102 - actual Context class under CPU torch
            compile(
                ast.fix_missing_locations(context_module), str(context_path), "exec"
            ),
            context_scope,
        )
        current_context = context_scope["Context"]
        context = NS(
            context=current_context(
                tensors["positions"], scheduled_bs=1, scheduled_tokens=ROWS
            )
        )
        original_replay = lambda *a, **kw: calls.append(kw.get("phase"))
        route = dict(
            globals(),
            cap=CAP,
            torch=torch,
            plan={"output": directory, "request": {}},
            _LIVE=True,
            _WITNESS=None,
            target=(1, 4),
            get_forward_context=lambda: context,
            original_replay=original_replay,
        )
        controller.graph, controller.recorded = graph, True
        controller.state = torch.zeros(8, dtype=torch.int64)
        controller.model_runner = NS(
            forward_vars={k: NS(gpu=tensors[k]) for k in ("input_ids", "positions")}
        )
        scope.update(ROUTE=route, route_replay=inherited("replay", route, nested=True))
        exec(  # noqa: S102 - current wrapper body under controlled CPU bindings
            compile(
                ast.Module(body=[nodes["replay"]], type_ignores=[]), __file__, "exec"
            ),
            scope,
        )
        with (
            patch.object(CAP, "RUN_ROOT", Path(directory) / "data"),
            patch.object(CAP, "_backend", lambda: NS(WORDS=8, ARMED=0)),
            patch.object(torch.cuda, "synchronize", lambda: None),
        ):
            context_rejections = []
            for case in (
                "dummy_run",
                "missing_dummy_marker",
                "unknown_dummy_marker",
                "missing_scheduled_bs",
                "missing_scheduled_tokens",
                "missing_context",
            ):
                context.context = current_context(
                    tensors["positions"],
                    is_dummy_run=case == "dummy_run",
                    scheduled_bs=1,
                    scheduled_tokens=ROWS,
                )
                assert "is_dummy_run" in vars(context.context) and not hasattr(
                    context.context, "is_dummy"
                )
                if case == "missing_dummy_marker":
                    del context.context.is_dummy_run
                elif case == "unknown_dummy_marker":
                    context.context.is_dummy_run = None
                elif case.startswith("missing_scheduled_"):
                    delattr(context.context, case.removeprefix("missing_"))
                elif case == "missing_context":
                    context.context = None
                try:
                    scope["replay"](graph, key=(1, 4))
                except RuntimeError:
                    context_rejections.append(case)
                else:
                    raise AssertionError(
                        f"{case} accepted: ordinal={controller.replay_ordinal}, pending={controller.pending}, witness={route['_WITNESS']}"
                    )
                assert (
                    not calls
                    and controller.replay_ordinal == 0
                    and not controller.pending
                )
                assert (
                    route["_WITNESS"] is None
                    and controller.selected is None
                    and not bool(controller.state.any())
                )
            context.context = current_context(
                tensors["positions"],
                is_dummy_run=False,
                scheduled_bs=1,
                scheduled_tokens=ROWS,
            )
            scope["replay"](graph, key=(1, 4))
            assert controller.pending and route["_WITNESS"]["replay_ordinal"] == 1
            identity_before = dict(controller.meta["identity"])
            state_before = controller.state.clone()
            context.context.is_dummy_run = True
            try:
                scope["replay"](graph, key=(1, 4))
            except RuntimeError:
                context_rejections.append("armed_dummy_run")
            else:
                raise AssertionError("armed capture accepted a dummy context")
            assert controller.pending and controller.replay_ordinal == 1
            assert controller.selected is None and calls == [None]
            assert controller.meta["identity"] == identity_before
            assert torch.equal(controller.state, state_before)
            context.context.is_dummy_run = False
            tensors["input_ids"].add_(10)
            tensors["positions"].add_(4)
            scope["replay"](graph, key=(1, 4))
            assert controller.selected["replay_ordinal"] == 2 and not controller.pending
            assert (
                controller.meta["identity"]["selected_input_ids"]
                == tensors["input_ids"].tolist()
            )
            scope["replay"](graph, key=(1, 4))
            assert calls == [None, CAP.CAPTURE_PHASE, None]
    CAP._CONTROLLER = None
    assert not torch.cuda.is_initialized()
    return {
        "controlled_cpu_only": True,
        "snapshot_negative_controls": failures,
        "actual_context_rejections": context_rejections,
        "actual_wrapper_bodies": [
            "raw entry",
            "gate exit",
            "mailbox count",
            "live graph-owned identity",
            "ordinary then one-shot then ordinary",
        ],
        "gpu_model_or_real_input_evidence": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    worker(json.loads(args.plan.read_text()), args.plan)
