"""CPU state tests and a guarded, model-free native eight-worker startup probe."""

import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import observe as O
from plan import P, make_plan


def marker(state="recorded-pending", recorded=True, now=100):
    return {
        "recorded": recorded,
        "state": state,
        "recorded_mono": now,
        "completed_mono": now,
        "generation": 1,
    }


def test_plan_isolates_shim(tmp_path):
    plan = make_plan(tmp_path, "warmup_observer")
    assert plan["invocations"] == ["warmup_observer"]
    assert str(HERE / "worker_bootstrap/sitecustomize.py") in plan["sources"]
    for role, env in plan["environments"].items():
        assert ("worker_bootstrap" in env["PYTHONPATH"]) == (role == "server")
        assert ("K31_OBSERVER_PLAN" in env) == (role == "server")
    assert make_plan(tmp_path)["invocations"] == ["u1", "u2", "diagnostic"]


def test_deadlines_and_independent_first_unfinished():
    stop = O.StopState(P["observer"], 0)
    stop.ready = 300
    assert stop.check({}, 535) is None  # No candidate: no 30s readiness timer.
    states = {
        0: {
            "heartbeat": 535,
            "markers": [marker(now=520), marker("completed", now=534)],
        },
        1: {"heartbeat": 535, "markers": [marker("completed", now=534)]},
    }
    assert stop.check(states, 535) is None
    for s in states.values():
        s["heartbeat"] = 551
    assert stop.check(states, 551)["reason"] == "no-progress"
    assert stop.reason["detail"]["rank"] == 0
    old = copy.deepcopy(stop.reason)
    assert stop.latch("full-bracket-completion", 552) == old
    assert stop.grace_end() == 566
    cap = O.StopState(P["observer"], 0)
    cap.ready = 899
    assert cap.check({}, 1319)["reason"] == "deadline"
    startup = O.StopState(P["observer"], 0)
    assert startup.check({}, 900)["reason"] == "deadline"


def test_unrecorded_query_true_is_not_completion():
    assert O.frontier([marker("completed", recorded=False)]) == 0
    stop = O.StopState(P["observer"], 0)
    stop.ready = 0
    assert (
        stop.check(
            {0: {"heartbeat": 100, "markers": [marker("completed", False)]}}, 100
        )
        is None
    )
    assert (
        stop.check({0: {"heartbeat": 100, "markers": []}}, 106)["reason"]
        == "observer-failure"
    )


def test_full_bracket_requires_all_ranks_and_same_identity():
    stop = O.StopState(P["observer"], 0)
    stop.ready = 0
    states = {
        rank: {"heartbeat": 20, "complete": True, "bracket": ["request-9", 1, 2]}
        for rank in range(8)
    }
    assert stop.check(dict(list(states.items())[:7]), 20) is None
    states[7]["bracket"] = ["other-request", 1, 2]
    assert stop.check(states, 20)["reason"] == "observer-failure"
    stop = O.StopState(P["observer"], 0)
    states[7]["bracket"] = states[0]["bracket"]
    assert stop.check(states, 20)["reason"] == "full-bracket-completion"


def trace_entry(seq=19, record=50):
    return {
        "process_group": ["2", "tp"],
        "pg_id": 2,
        "record_id": record,
        "collective_seq_id": seq,
        "profiling_name": "nccl:_allgather_base",
        "input_sizes": [[2, 3]],
        "output_sizes": [[8, 2, 3]],
        "input_dtypes": ["BFloat16"],
        "output_dtypes": ["BFloat16"],
        "state": "completed",
    }


def test_sequence_calibration_does_not_assume_offset():
    receipt = {
        "before": 23,
        "after": 24,
        "shape": [2, 3],
        "dtype": "torch.bfloat16",
        "semantic": "W_K",
    }
    trace = {"entries": [trace_entry()]}
    calibration = O.calibrate({}, trace, receipt, ["2", "tp"])
    assert calibration == {"offset": -4, "span": 1, "pg_id": 2}
    assert (
        O.associate(trace, ["2", "tp"], calibration, [receipt])[0]["collective_seq_id"]
        == 19
    )
    with pytest.raises(ValueError):
        O.calibrate(
            {}, {"entries": [trace_entry(), trace_entry(20, 51)]}, receipt, ["2", "tp"]
        )
    with pytest.raises(ValueError):
        O.associate({}, ["2", "tp"], calibration, [receipt])
    with pytest.raises(ValueError):
        O.associate(trace, ["2", "tp"], calibration, [dict(receipt, after=25)])


def test_passive_import_and_capture_no_marker(monkeypatch):
    assert O.RUNTIME is None
    runtime = object.__new__(O.Observer)
    runtime.torch = NS(cuda=NS(is_current_stream_capturing=lambda: True))
    monkeypatch.setattr(O.TLS, "target", {"context": [393216]}, raising=False)
    assert runtime.begin_call(None, None, None, None, None) is None


def test_pool_recording_and_generations():
    runtime = object.__new__(O.Observer)
    runtime.torch = NS(
        cuda=NS(current_stream=lambda: NS(device_index=0, stream_id=7, cuda_stream=5))
    )
    runtime.events = [NS(record=lambda stream: None)]
    runtime.initialized = [True]
    runtime.lock = __import__("threading").RLock()
    runtime.state = {"markers": [marker("not-recorded", False)]}
    call = {"offset": 0}
    runtime.mark(call, 0)
    assert runtime.state["markers"][0]["stream"]["stream_id"] == 7
    assert runtime.state["markers"][0]["state"] == "recorded-pending"
    with pytest.raises(ValueError):
        runtime.mark(call, 0)
    with pytest.raises(ValueError):
        runtime.mark({"offset": 1}, 0)


def fake_runtime(tmp_path):
    import threading

    import numpy as np

    runtime = object.__new__(O.Observer)
    runtime.path, runtime.config = tmp_path, P["observer"]
    runtime.first_request, runtime.forward_number = None, 0
    runtime.maps, runtime.state, runtime.lock = [], {"bracket": []}, threading.RLock()
    runtime.torch = NS(
        cuda=NS(current_stream=lambda: NS(device_index=0, stream_id=7, cuda_stream=0))
    )
    batch = NS(
        req_ids=[17],
        external_request_ids=None,
        num_scheduled_tokens=np.array([8192]),
        num_cached_tokens=np.array([385024]),
        context_lens=np.array([393216]),
        is_dummy_run=False,
        total_seqs_num=1,
        total_seqs_num_prefill=1,
        total_seqs_num_decode=0,
    )
    return runtime, batch, NS(h2d_owner=NS(epoch=2), block_size=64)


def test_whole_batch_selection_and_owned_host_copies(tmp_path):
    import numpy as np

    runtime, batch, runner = fake_runtime(tmp_path)
    batch.total_seqs_num_decode = 1
    runtime.prepare(runner, batch)
    assert O.TLS.target is None
    batch.total_seqs_num_decode = 0
    runtime.prepare(runner, batch)
    table = np.zeros((1, 1000), dtype=np.int32)
    runtime.metadata(
        NS(model_runner=runner),
        1,
        np.array([49152]),
        np.array([0, 49152]),
        np.array([0, 393216]),
        np.array([393216]),
        table,
    )
    table[0, 0] = 99
    assert runtime.maps[0]["block_tables"][0, 0] == 0
    assert runtime.maps[0]["block_tables"].flags.owndata
    assert not runtime.maps[0]["block_tables"].flags.writeable
    batch.context_lens[:] = 401408
    batch.num_cached_tokens[:] = 393216
    batch.req_ids = [18]
    runtime.prepare(runner, batch)
    assert O.TLS.target is None
    batch.req_ids = [17]
    runtime.prepare(runner, batch)
    assert [b["context"][0] for b in runtime.state["bracket"]] == [393216, 401408]


def test_unrecorded_events_are_never_queried_and_query_failure_is_explicit():
    import threading

    runtime = object.__new__(O.Observer)
    runtime.lock = threading.RLock()
    runtime.initialized = [True]
    runtime.state = {"markers": [marker("not-recorded", False)]}
    runtime.events = [
        NS(query=lambda: (_ for _ in ()).throw(RuntimeError("query unavailable")))
    ]
    runtime.poll()  # Would raise if it queried the unrecorded slot.
    runtime.state["markers"][0] = marker()
    with pytest.raises(RuntimeError):
        runtime.poll()
    assert runtime.state["markers"][0]["state"] == "query-unavailable"


def test_phase_stop_and_evidence_grace_cap():
    stop = O.StopState(P["observer"], 0)
    stop.ready = 899
    assert stop.check({}, 1310, phase=True)["reason"] == "phase-transition"
    assert stop.grace_end() == 1325
    stop = O.StopState(P["observer"], 0)
    stop.latch("observer-failure", 1330, "exporter hung")
    assert stop.grace_end() == 1335


def test_shim_in_nonworker_imports_no_gpu_stack(tmp_path):
    import subprocess

    env = dict(
        os.environ,
        PYTHONPATH=str(HERE / "worker_bootstrap"),
        K31_OBSERVER_PLAN=str(tmp_path / "absent-plan.json"),
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,k31_observe; assert k31_observe.RUNTIME is None; assert 'torch' not in sys.modules; assert 'aiter' not in sys.modules",
        ],
        env=env,
        check=True,
        timeout=10,
    )


def test_full_24_layer_two_forward_pool_and_stream_identity(tmp_path):
    import torch

    runtime, _batch, _runner = fake_runtime(tmp_path)
    runtime.torch.cuda.is_current_stream_capturing = lambda: False
    runtime.events = [
        NS(record=lambda stream: None, query=lambda: True) for _ in range(288)
    ]
    runtime.initialized = [True] * 288
    runtime.catalog = {
        f"model.layers.{layer}.self_attn.indexer": {
            "layer": layer,
            "cache": f"model.layers.{layer}.self_attn.indexer.k_cache",
        }
        for layer in P["observer"]["layers"]
    }
    runtime.state.update(ready=True, markers=[])
    tensor = torch.zeros(2, 2)
    metadata = NS(
        **{
            f"dcp_indexer_{name}": tensor
            for name in (
                "fp4_read_page",
                "fp4_read_row",
                "gather_index",
                "fp4_stage_page",
                "fp4_stage_row",
            )
        },
        dcp_indexer_fp4_stage_pages=2,
    )
    for context in P["observer"]["contexts"]:
        O.TLS.target = {"request_ids": [17], "context": [context], "forward": context}
        for prefix in runtime.catalog:
            call = runtime.begin_call(prefix, tensor, tensor, metadata, "mfma")
            for boundary in range(1, 6):
                runtime.mark(call, boundary)
            runtime.poll()
    assert O.frontier(runtime.state["markers"]) == 288
    assert runtime.state["markers"][-1]["semantic"]["layer"] == 92
    assert runtime.state["markers"][-1]["generation"] == 48
    with pytest.raises(ValueError):
        runtime.begin_call(prefix, tensor, tensor, metadata, "mfma")
    runtime.state["markers"][1]["recorded"] = False
    runtime.torch.cuda.current_stream = lambda: NS(
        device_index=0, stream_id=8, cuda_stream=0
    )
    with pytest.raises(ValueError, match="stream identity"):
        runtime.mark({"offset": 0}, 1)


def test_gather_wrapper_preserves_native_exception_and_entry_receipt(monkeypatch):
    import threading
    import types

    class GroupCoordinator:
        def all_gather(self, tensor, use_custom=False, dim=-1):
            raise LookupError("native error")

    module = types.ModuleType("aiter.dist.parallel_state")
    module.GroupCoordinator = GroupCoordinator
    runtime = NS(
        group=["2", "tp"],
        lock=threading.RLock(),
        receipts=[],
        mark=lambda *a: None,
        fail=lambda message: None,
    )
    monkeypatch.setattr(O, "RUNTIME", runtime)
    monkeypatch.setattr(O, "HOOKS", [])
    monkeypatch.setattr(O.TLS, "loading", False, raising=False)
    monkeypatch.setattr(
        O.TLS, "call", {"semantic": {"layer": 3}, "gathers": 0}, raising=False
    )
    O.patch(module)
    group = GroupCoordinator()
    group.device_group = NS(
        group_name="2", group_desc="tp", _get_sequence_number_for_group=lambda: 100
    )
    with pytest.raises(LookupError, match="native error"):
        group.all_gather(NS(shape=(49152, 4, 16), dtype="torch.uint8"), dim=0)
    assert runtime.receipts[0]["before"] == 100
    assert runtime.receipts[0]["native_exception"] == "LookupError"
    assert "after" not in runtime.receipts[0]


# The probe below is opt-in and runs as a separate guarded process, never during
# test collection. No install()/patch() calls: sitecustomize performs registration.
class ModelBoundary(Exception):
    pass


class QueueBoundary:
    def __init__(self, *a, **kw):
        pass

    def export_handle(self):
        return 0

    @classmethod
    def create_from_handle(cls, *a):
        return cls()


class EventBoundary:
    def __init__(self, **kw):
        assert kw == {"enable_timing": False}

    def record(self, stream):
        pass

    def query(self):
        return True


def setup_cpu_route():
    """Keep native startup/imports; replace only external kernels, IO and GPU/model work."""
    sys.path.insert(0, str(ROOT / "tests"))
    from aiter_stub import atom_under_stubs

    context = atom_under_stubs()
    load = context.__enter__()
    # External stub finder normally wins. Let the shim see actual AITER's class file.
    external = next(
        f for f in sys.meta_path if type(f).__name__ == "_ExternalModuleFinder"
    )
    sys.meta_path.remove(external)
    sys.meta_path.append(external)
    aiter = importlib.import_module("aiter")
    aiter.__path__ = []
    dist = importlib.import_module("aiter.dist")
    dist.__path__ = [str(Path(P["aiter_root"]) / "aiter/dist")]
    aiter.torch_compile_guard = lambda *a, **kw: lambda fn: fn
    ps = importlib.import_module("aiter.dist.parallel_state")
    model = load("atom.model_engine.model_runner")
    load("atom.model_ops.attentions.aiter_mla")
    load("atom.model_ops.indexer.indexer_forward")
    async_proc = load("atom.model_engine.async_proc")
    import torch

    # Only the existing native setup call invokes this controlled device boundary.
    def device_boundary(config, rank):
        pg = NS(
            group_name="cpu-wiring-only",
            group_desc="controlled",
            _get_sequence_number_for_group=lambda: 0,
        )
        group = object.__new__(ps.GroupCoordinator)
        group.ranks, group.device_group, group.unique_name = (
            list(range(8)),
            pg,
            "tp:cpu-fixture",
        )
        ps._TP, ps._DCP = group, NS(ranks=list(range(8)), device_group=object())
        return torch.device("cpu")

    model.init_model_env = device_boundary
    model.tokenIDProcessor = lambda *a: None
    model.sampler_from_config = lambda *a: None
    model.get_model = lambda *a: (_ for _ in ()).throw(ModelBoundary())
    torch._C._distributed_c10d._dump_nccl_trace_json = (
        lambda **kw: b'{"version":"2.10","pg_config":{},"pg_status":{},"nccl_comm_state":{}}'
    )
    torch.cuda.Event = EventBoundary
    torch.cuda.current_stream = lambda: NS(device_index=0, stream_id=7, cuda_stream=0)
    torch.cuda.is_current_stream_capturing = lambda: False
    async_proc.MessageQueue = QueueBoundary
    async_proc.numa_bind_to_node = lambda *a: None

    def wait_for_stop(self):
        while not (Path(os.environ["K31_OBSERVER_PLAN"]).parent / "cpu-stop").exists():
            time.sleep(0.05)

    async_proc.AsyncIOProc.busy_loop = wait_for_stop

    class NativeRunner(model.ModelRunner):
        def __init__(self, rank, config):
            try:
                super().__init__(rank, config)
            except ModelBoundary:
                import k31_observe

                runtime = k31_observe.RUNTIME
                assert runtime is not None and len(k31_observe.HOOKS) == 9
                runtime.state["cpu_wiring_only"] = True
                runtime.state["native_route"] = [
                    "AsyncIOProcManager",
                    "AsyncIOProc",
                    "ModelRunner.__init__",
                    "ModelRunner._setup_device_and_distributed",
                ]
                runtime.state["native_sources"] = {
                    str(p): O.digest(p)
                    for p in (
                        ROOT / "atom/model_engine/async_proc.py",
                        ROOT / "atom/model_engine/model_runner.py",
                        Path(ps.__file__),
                    )
                }
                runtime.publish()
                assert (
                    not torch.cuda.is_initialized()
                    and not torch.distributed.is_initialized()
                )

    globals()["PreflightRunner"] = NativeRunner
    return async_proc, context


def cpu_preflight():
    assert "KFD_GUARD_LOG" in os.environ, "CPU preflight requires no_gpu.sh"
    import k31_observe

    assert k31_observe.RUNTIME is None
    config = NS(
        hf_config=NS(model_type="deepseek_v2", architectures=["DeepseekV2ForCausalLM"]),
        kv_cache_block_size=64,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        tensor_parallel_size=8,
        tp_world_size=8,
        torch_dtype=__import__("torch").bfloat16,
        parallel_config=NS(
            data_parallel_rank_local=0,
            data_parallel_rank=0,
            data_parallel_size=1,
            pipeline_parallel_rank=0,
            data_parallel_master_ip="127.0.0.1",
        ),
        pcp_group_size=1,
        tp_rank_start=0,
        torch_profiler_dir=None,
        pipeline_parallel_size=1,
        speculative_config=None,
        max_num_batched_tokens=8,
        max_num_seqs=1,
        max_model_len=8,
        decode_context_parallel_size=8,
    )
    manager = CPU_ASYNC.AsyncIOProcManager(
        lambda: None, 8, "test_observe.PreflightRunner", config
    )
    out = Path(os.environ["K31_OBSERVER_PLAN"]).parent
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            paths = list((out / "observer").glob("rank-*/state.json"))
            if len(paths) == 8 and all(
                json.loads(p.read_text()).get("cpu_wiring_only") for p in paths
            ):
                values = [json.loads(p.read_text()) for p in paths]
                assert len({v["pid"] for v in values}) == 8
                assert {v["rank"] for v in values} == set(range(8))
                assert k31_observe.RUNTIME is None
                O.atomic(
                    out / "cpu-preflight.json",
                    {
                        "passed": True,
                        "gpu_readiness": False,
                        "parent_pid": os.getpid(),
                        "workers": values,
                        "controlled_boundaries": [
                            "external kernel imports",
                            "device/PG/Event/exporter",
                            "model construction",
                            "IO queues",
                        ],
                    },
                )
                return
            time.sleep(0.2)
        raise TimeoutError("eight native worker receipts missing")
    finally:
        manager.keep_monitoring = False
        (out / "cpu-stop").touch()
        manager.exit()


if os.environ.get("K31_CPU_ROUTE") == "1":
    sys.modules["test_observe"] = sys.modules[__name__]
    CPU_ASYNC, CPU_CONTEXT = setup_cpu_route()
    if __name__ == "__main__":
        try:
            cpu_preflight()
        finally:
            CPU_CONTEXT.__exit__(None, None, None)
