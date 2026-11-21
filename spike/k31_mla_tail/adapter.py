"""Matched native and donor operator boundaries; no model frontend replacement."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROWS = (8, 16, 20, 36, 40, 44, 48)
DONOR = "78e27b0e75abb7e58a5b8d7bd6834300dd1ac90e"
AITER = "572232d23661a094fbc4e98e879713f0e7d2d5f4"
PACKAGE = "amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512"


def compare(got, expected):
    import torch

    assert got.shape == expected.shape and got.dtype == expected.dtype == torch.bfloat16
    assert (
        got.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()
    ), "arms share output storage"
    finite = bool(torch.isfinite(got).all() and torch.isfinite(expected).all())
    relative = (
        float(
            (got.float() - expected.float()).abs().max()
            / expected.float().abs().max().clamp_min(1e-30)
        )
        if finite
        else None
    )
    equality = float(
        (got.view(torch.int16) == expected.view(torch.int16)).float().mean()
    )
    return {
        "finite": finite,
        "relative_max": relative,
        "bf16_equal": equality,
        "passed": finite and relative <= 2**-7 and equality >= 0.999,
    }


def selfcheck(root, donor):
    """CPU boundary checks; native GPU helpers remain outside this evidence."""
    import torch

    path = Path(root) / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dcp_project_merge_out"
    )
    method.decorator_list = []
    namespace = {}
    exec(  # noqa: S102 - execute the current native dispatch body in the CPU check
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    seen = []
    owned = torch.tensor([3, 0, 7], dtype=torch.int32)
    x = torch.ones(3, 96, 512, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        pbm_enabled=True,
        W_V_dcp=object(),
        W_V_dcp_scale=None,
        num_heads=12,
        v_head_dim=128,
        o_proj=torch.nn.Identity(),
        _dcp_fused_quant_dtype=lambda: None,
    )
    layer._v_up_proj = lambda *a, **k: torch.ones(3, 96, 128, dtype=torch.bfloat16)

    def merge(o, lse, **kwargs):
        seen.append(kwargs)
        return o[:, :12].contiguous()

    layer._dcp_merge = merge
    result = namespace["_dcp_project_merge_out"](
        layer, x, torch.zeros(3, 96), owned_counts=owned, decode=True
    )
    assert result.shape == (3, 1536) and seen[0]["decode"] is True
    assert seen[0]["owned_counts"] is owned
    refpath = Path(donor) / (PACKAGE.replace(".", "/") + "/reference.py")
    spec = importlib.util.spec_from_file_location("tail_reference_cpu", refpath)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert torch.equal(
        reference.row_mask(3, owned, torch.arange(4, dtype=torch.int32), "cpu"),
        owned > 0,
    )
    assert compare(result, result.clone())["passed"]
    bad = result.clone()
    bad[0, 0] = 100
    assert not compare(bad, result)["passed"]
    bad[0, 0] = float("nan")
    assert not compare(bad, result)["passed"]
    try:
        compare(result, result)
    except AssertionError:
        pass
    else:
        raise AssertionError("alias control did not fail")
    assert not torch.cuda.is_initialized()
    return {
        "passed": True,
        "checks": [
            "actual native method decode/owned forwarding",
            "Identity projection boundary",
            "donor row mask correspondence",
            "numeric corruption, NaN and alias rejection",
        ],
        "limits": "CPU helpers controlled; no GPU, native graph or performance qualification",
    }


class Tail:
    def __init__(self, group, rank, device, max_rows):
        import torch
        from amd_kernels import warm_up
        from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import (
            DcpVupCombine,
            dcp8_lse_combine_vup,
        )

        from atom.model_ops import dcp_ops
        from atom.model_ops.attention_mla import (
            MLAAttention,
            is_rocm_aiter_fp4bmm_enabled,
        )
        from atom.model_ops.triton_fused_sigmoid_mul_quant import (
            fused_sigmoid_mul_maybe_quant,
        )

        assert not is_rocm_aiter_fp4bmm_enabled()
        self.torch, self.ops, self.gate = torch, dcp_ops, fused_sigmoid_mul_maybe_quant
        self.rank, self.device, self.group = rank, device, group
        self.native = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(self.native)
        for key, value in {
            "pbm_enabled": True,
            "dcp_world_size": 8,
            "num_heads": 12,
            "v_head_dim": 128,
            "kv_lora_rank": 512,
            "W_V_scale": None,
            "W_V_dcp_scale": None,
            "dcp_comm_backend": "mailbox",
            "dcp_group": group,
            "o_proj": torch.nn.Identity(),
        }.items():
            setattr(self.native, key, value)
        assert dcp_ops.prepare_dcp_mailbox(group, 12, 128, max_rows)
        warm_up.set_auto(False)  # Benchmark prewarms exactly its selected variants.
        warm_up.set_workers(1)
        self.comm = DcpVupCombine(rank, group.cpu_group, device, max_tokens=max_rows)
        self.donor = dcp8_lse_combine_vup
        self.key = dcp_ops._mailbox_key(group, 12, 128)

    def calls(self):
        return self.ops._MAILBOX_LAUNCHES.get(self.key, 0)

    def fixtures(self, rows, rotations):
        t = self.torch
        generator = t.Generator(device=self.device).manual_seed(
            14000 + self.rank + rows
        )
        weights = t.Generator(device=self.device).manual_seed(14000 + rows)
        data = []
        for _ in range(rotations):
            x = (
                t.randn(rows, 96, 512, device=self.device, generator=generator) * 0.5
            ).bfloat16()
            lse = t.randn(rows, 96, device=self.device, generator=generator) * 2 + 3
            weight = (
                t.randn(96, 128, 512, device=self.device, generator=weights) * 0.05
            ).bfloat16()
            gate = t.randn(
                rows, 1536, device=self.device, generator=generator
            ).bfloat16()
            owned = t.full((rows,), 37, dtype=t.int32, device=self.device)
            qsl = t.arange(rows + 1, dtype=t.int32, device=self.device)
            data.append(
                {
                    "x": x,
                    "lse": lse,
                    "weight": weight,
                    "gate": gate,
                    "owned": owned,
                    "qsl": qsl,
                    "out": t.empty(rows, 12, 128, dtype=t.bfloat16, device=self.device),
                }
            )
        return data

    def native_call(self, item):
        self.native.W_V_dcp = item["weight"]
        raw = self.native._dcp_project_merge_out(
            item["x"], item["lse"], owned_counts=item["owned"], decode=True
        )
        result, scale = self.gate(raw, item["gate"], quant=False)
        assert scale is None
        item["native_out"] = result
        return result

    def donor_call(self, item):
        return self.donor(
            self.comm,
            item["x"],
            item["lse"],
            item["weight"],
            item["out"],
            item["owned"],
            item["qsl"],
            gate=item["gate"],
        ).view(-1, 1536)

    def capture(self, items, stream, calls):
        import torch.distributed as dist

        before = self.calls()
        self.native_call(items[0])
        assert self.calls() == before, "eager path unexpectedly used mailbox"
        for item in items:
            self.donor_call(item)
            self.native_call(item)
        self.torch.cuda.synchronize()
        graphs = {}
        for name, call in (("native", self.native_call), ("donor", self.donor_call)):
            dist.barrier(group=self.group.cpu_group)
            graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(graph, stream=stream):
                for i in range(calls):
                    call(items[i % len(items)])
            graphs[name] = graph
        assert self.calls() - before == calls, "captured native mailbox did not execute"
        self.ops.check_dcp_mailbox_launches()
        return graphs, {
            "decode": True,
            "eager_mailbox_calls": 0,
            "captured_mailbox_calls": calls,
            "heads_per_rank": 12,
        }

    def check(self, items, graphs):
        import torch.distributed as dist

        t, first = self.torch, items[0]
        saved = {k: first[k].clone() for k in ("x", "lse", "owned", "gate", "weight")}
        records = []
        for case in ("ordinary", "partial_empty", "invalid_lse", "all_empty"):
            for key, value in saved.items():
                first[key].copy_(value)
            if case == "partial_empty":
                first["owned"][-1] = 0  # padding, empty on every rank
                if self.rank % 2 == 0:
                    first["owned"][0] = 0
            if case == "all_empty":
                first["owned"].zero_()
            if case == "invalid_lse":
                first["lse"][0, 0] = float("nan")
                first["lse"][0, 1] = float("inf")
            first["x"][first["owned"] == 0] = float("nan")
            inputs = {k: first[k].clone() for k in saved}
            for epoch in range(3):
                for name in ("native", "donor"):
                    dist.barrier(group=self.group.cpu_group)
                    graphs[name].replay()
                    t.cuda.synchronize()
                for item in items:
                    output = item["out"].view(-1, 1536)
                    assert all(
                        output.data_ptr() != item[k].data_ptr()
                        for k in ("x", "lse", "weight", "gate", "owned")
                    )
                    record = compare(output, item["native_out"])
                    if item is first and case in ("partial_empty", "all_empty"):
                        empty = output[-1] if case == "partial_empty" else output
                        record["empty_zero"] = bool((empty == 0).all())
                        record["passed"] &= record["empty_zero"]
                    records.append({"case": case, "epoch": epoch, **record})
                self.donor_call(first)  # Alternate mailbox parity between graph epochs.
                t.cuda.synchronize()
            for key, expected in inputs.items():
                t.testing.assert_close(
                    first[key], expected, rtol=0, atol=0, equal_nan=True
                )
        for key, value in saved.items():
            first[key].copy_(value)
        return records


def worker(rank, plan):
    import ctypes
    import datetime
    import os
    import statistics

    import aiter
    import amd_kernels
    import torch
    import torch.distributed as dist

    import atom

    for module, key in ((atom, "atom"), (aiter, "aiter"), (amd_kernels, "donor")):
        assert Path(module.__file__).resolve().is_relative_to(Path(plan[key]))
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() == 8
    assert torch.cuda.get_device_properties(rank).gcnArchName.split(":")[0] == "gfx950"
    device = torch.device("cuda", rank)
    from amd_kernels.common.hip_runtime import load_hip_runtime

    hip = load_hip_runtime()
    loaded_hip = {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libamdhip64.so" in line
    }
    assert loaded_hip == {plan["hip_runtime"]}, loaded_hip
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    bdf_buffer = ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(bdf_buffer, 32, rank) == 0
    bdf = bdf_buffer.value.decode().lower()
    assert bdf in plan["selected_bdfs"]
    dist.init_process_group(
        "gloo",
        init_method=plan["rendezvous"],
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=120),
    )
    group = SimpleNamespace(
        world_size=8,
        rank_in_group=rank,
        cpu_group=dist.group.WORLD,
        device_group=dist.new_group(backend="nccl"),
    )
    stream = torch.cuda.Stream(device=device)
    output = {
        "rank": rank,
        "bdf": bdf,
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()[19],
        "device": str(torch.cuda.get_device_properties(device)),
        "scope": "deterministic operator fixture; no representative model claim",
        "rows": [],
    }
    with torch.cuda.stream(stream):
        tail = Tail(group, rank, device, max(plan["rows"]))
        for rows in plan["rows"]:
            items = tail.fixtures(rows, plan["rotations"])
            graphs, proof = tail.capture(items, stream, plan["calls"])
            proof["layouts"] = {
                k: {
                    "shape": list(v.shape),
                    "stride": list(v.stride()),
                    "dtype": str(v.dtype),
                }
                for k, v in items[0].items()
            }
            checks = tail.check(items, graphs)
            all_pass = [None] * 8
            dist.all_gather_object(all_pass, all(r["passed"] for r in checks))
            record = {
                "tokens": rows,
                "graph_proof": proof,
                "correctness": checks,
                "all_ranks_pass": all(all_pass),
                "pairs": [],
            }
            if record["all_ranks_pass"]:
                for pair in range(plan["pairs"]):
                    times = {}
                    for name in (
                        ("native", "donor") if pair % 2 == 0 else ("donor", "native")
                    ):
                        graphs[name].replay()
                        torch.cuda.synchronize()
                        dist.barrier()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        graphs[name].replay()
                        end.record(stream)
                        torch.cuda.synchronize()
                        local = torch.tensor(
                            [start.elapsed_time(end) * 1000 / plan["calls"]],
                            dtype=torch.float64,
                        )
                        values = [torch.empty_like(local) for _ in range(8)]
                        dist.all_gather(values, local)
                        ranks = [float(v.item()) for v in values]
                        times[name] = {
                            "ranks_us": ranks,
                            "max_us": max(ranks),
                            "skew_us": max(ranks) - min(ranks),
                        }
                    record["pairs"].append(times)
                ratio = [
                    p["native"]["max_us"] / p["donor"]["max_us"]
                    for p in record["pairs"]
                ]
                record["speedup"] = statistics.median(ratio)
                record["verdict"] = (
                    (
                        "advance"
                        if min(ratio) > 1.02
                        else "reject" if max(ratio) < 0.98 else "inconclusive"
                    )
                    if plan["pairs"] == 5
                    else "smoke_only"
                )
            else:
                record["verdict"] = "correctness_failed"
            record["sent_bytes_per_rank"] = 7 * rows * 12 * 260
            record["rotating_input_bytes"] = sum(
                sum(
                    v.numel() * v.element_size()
                    for k, v in item.items()
                    if k in ("x", "weight", "lse", "gate")
                )
                for item in items
            )
            output["rows"].append(record)
            Path(plan["out"], f"rank-{rank}.json").write_text(
                json.dumps(output, indent=2) + "\n"
            )
            del graphs, items
            if not record["all_ranks_pass"]:
                break
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    assert all(r["all_ranks_pass"] for r in output["rows"]), "numerical gate failed"
