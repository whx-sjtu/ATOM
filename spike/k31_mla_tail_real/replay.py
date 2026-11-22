"""Bounded actual-input capture/replay; prospective plans authorize no GPU work."""

import argparse
import ast
import ctypes
import functools
import hashlib
import importlib.util
import json
import os
import signal
import stat
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
COMGR_PROOF = (
    WS / "agent-team/evidence/task-19-comgr-metadata-preflight-02/manifest.json"
)
COMGR_PROOF_SHA = "cbbd57601473bf94fd6bb82ba27d2d09c100cceebe53ad7d0adcc00b7dd64168"


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


def pruning_marker_identity(original, path, *, content=True):
    marker = PRIVATE / "cache/comgr/llvmcache.timestamp"
    if Path(path) != marker:
        return original(path, content=content)
    before = marker.lstat()
    assert content and stat.S_ISREG(before.st_mode) and not before.st_mode & 0o111
    assert marker.resolve() == marker and marker.read_bytes() == b""
    after = marker.lstat()
    stable = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mode)
    assert stable(before) == stable(after) and before.st_size == 0
    return {
        "resolved": str(marker),
        "stat": [before.st_dev, before.st_ino, 0, None],
        "sha256": hashlib.sha256(b"").hexdigest(),
        "mode": stat.S_IMODE(before.st_mode),
        "mtime_policy": "mutable COMGR pruning-interval marker only",
    }


def marker_policy_sources(execution):
    assert sha(COMGR_PROOF) == COMGR_PROOF_SHA
    files = {str(COMGR_PROOF): COMGR_PROOF_SHA}
    files.update(
        {
            str(COMGR_PROOF.parent / name): digest
            for name, digest in json.loads(COMGR_PROOF.read_text()).items()
        }
    )
    for name, digest in files.items():
        assert sha(name) == digest
    proof = json.loads((COMGR_PROOF.parent / "source-bindings.json").read_text())
    assert (
        execution["files"]["/opt/rocm/lib/libamd_comgr.so"]["sha256"]
        == proof["comgr_sha256"]
    )
    return files


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
    scope["file_identity"] = functools.partial(
        pruning_marker_identity, scope["file_identity"]
    )
    return scope


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def qualify_preserved(plan_path, receipts, report_path):
    """Qualify unchanged payload bytes; failed producer admission stays refused."""
    import torch

    helpers()
    capture.initialize_snapshot()
    assert os.environ.get("KFD_GUARD_LOG") and os.environ.get("LD_PRELOAD")
    assert not LEGACY.git("status", "--porcelain") and not report_path.exists()
    assert report_path.resolve().is_relative_to(WS / "agent-team/evidence")
    plan = json.loads(plan_path.read_text())
    assert Path(plan["root"]).resolve() == ROOT
    output = Path(plan["output"])
    controller_path, stop_path, release_path = (
        output / "controller.json",
        receipts / "root-stop-request.json",
        receipts / "root-release-verification.json",
    )
    controller, stop, release = (
        json.loads(p.read_text()) for p in (controller_path, stop_path, release_path)
    )
    assert (
        plan["mode"] == controller["mode"] == "capture"
        and controller["head"] == plan["head"]
    )
    assert controller["plan_sha256"] == sha(plan_path) and not controller["complete"]
    assert (
        controller["stop_signals"] == [signal.SIGTERM] and stop["signal"] == "SIGTERM"
    )
    assert stop["no_foreign_process_targeted"] and stop["no_retry_authorized"]
    assert release["controller_sha256"] == sha(controller_path) and release[
        "stop_request_sha256"
    ] == sha(stop_path)
    assert (
        release["controller_absent"]
        and release["lock_acquired"]
        and not release["global_kfd"]
    )
    assert not release["capture_qualified"] and not release["further_GPU_grant"]
    used = {
        f"0000:{slot}:00.0": value for slot, value in release["bdf_vram_bytes"].items()
    }
    assert LEGACY.selected_released(
        controller["cleanup"]["alive"], controller["cleanup"]["snapshot"], used
    )
    assert controller["cleanup"]["released"]
    for name, expected in plan["sources"].items():
        source = Path(name)
        if source.is_relative_to(ROOT / "spike/k31_mla_tail_real"):
            original = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    plan["head"] + ":" + str(source.relative_to(ROOT)),
                ]
            )
            assert hashlib.sha256(original).hexdigest() == expected
        else:
            assert sha(source) == expected, name
    ranks, matches, weights, bdfs, owned_counts = [], [], set(), set(), []
    for rank in range(8):
        path = output / "data" / f"rank{rank}.invalid.pt"
        before = sha(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validation = capture.validate_snapshot(payload)
        meta, identity = payload["meta"], payload["meta"]["identity"]
        assert (
            identity["rank"] == rank
            and meta["provenance"]["atom_commit"] == plan["head"]
        )
        assert identity["request_hint"] == plan["request"]
        tensors = capture.views(payload)
        matched = {
            k: identity[k]
            for k in ("epoch", "arm_id", "ordinary_served_witness", "selected_graph")
        }
        matched.update(
            key=meta["key"],
            abi=meta["abi_scalars"],
            ids=tensors["input_ids"].tolist(),
            positions=tensors["positions"].tolist(),
        )
        matches.append(matched)
        weights.add(payload["tensor_hashes"]["w_v"])
        bdfs.add(meta["hardware"]["bdf"])
        owned_counts.append(tensors["owned"].tolist())
        assert sha(path) == before
        ranks.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": before,
                "bytes": path.stat().st_size,
                "validation": validation,
                "identity": identity,
                "key": meta["key"],
                "workspace": meta["workspace"],
                "hardware": meta["hardware"],
                "tensor_hashes": payload["tensor_hashes"],
            }
        )
    assert (
        all(m == matches[0] for m in matches)
        and len(weights) == 1
        and bdfs == set(plan["selected_bdfs"])
    )
    assert [sum(row[i] for row in owned_counts) for i in range(capture.ROWS)] == [
        p + 1 for p in matches[0]["positions"]
    ]
    required = [output / "data/manifest.json", output / "data/request.json"]
    required += [
        output / "data" / f"rank{r}.{suffix}.json"
        for r in range(8)
        for suffix in ("receipt", "cleanup")
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert len(missing) == len(required), "unexpected producer completion evidence"
    report = {
        "state": "PAYLOAD_INTEGRITY_ONLY",
        "capture_remains_failed": True,
        "replay_admitted": False,
        "validator_head": LEGACY.git("rev-parse", "HEAD"),
        "validator_sha256": sha(Path(capture.__file__)),
        "qualifier_sha256": sha(Path(__file__)),
        "original_capture_head": plan["head"],
        "cpu_runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "executable": sys.executable,
        },
        "raw_files_unchanged": True,
        "bindings": {
            str(p): sha(p)
            for p in (plan_path, controller_path, stop_path, release_path)
        },
        "original_sources_verified": len(plan["sources"]),
        "missing_producer_evidence": missing,
        "matched_identity": matches[0],
        "ranks": ranks,
        "numerical_timing_or_quality_qualified": False,
    }
    assert not torch.cuda.is_initialized()
    save(report_path, report)
    return {
        "report": str(report_path),
        "sha256": sha(report_path),
        "state": report["state"],
        "ranks": len(ranks),
    }


RECOVERY_RUN = PRIVATE / "runs/5880d8e4-capture-01"
RECOVERY_EVIDENCE = {
    str(
        WS
        / "agent-team/evidence/task-19-payload-integrity-3d2deed6-01/payload-integrity.json"
    ): "01b4695af4ef37955758c13f4c0d286ecb29a7dce9f66bddc9aed05b1cd69322",
    str(
        WS / "agent-team/evidence/review-19-payload-integrity-3d2deed6-01/manifest.json"
    ): "4a3b4014c3a9697544048ebb12f54ae5bff3473541e661b5cf653b95d6fe8b00",
    str(
        WS / "agent-team/evidence/task-19-recovery-ruling-7071087912/comment.json"
    ): "034a72ef4e2a4f88192485cc0d42961104b8058fea57b33de098d7480cd43e2a",
}


def recovery_record():
    """This exact ruled tuple only; reuse its independently qualified proof."""
    helpers()
    capture.initialize_snapshot()
    bindings = dict(RECOVERY_EVIDENCE)
    for name, expected in bindings.items():
        assert sha(name) == expected, name
    integrity_path, review_path, ruling_path = map(Path, RECOVERY_EVIDENCE)
    integrity = json.loads(integrity_path.read_text())
    review_manifest = json.loads(review_path.read_text())
    bindings.update(
        {
            str(review_path.parent / name): value
            for name, value in review_manifest.items()
        }
    )
    bindings.update(integrity["bindings"])
    assert (
        integrity["state"] == "PAYLOAD_INTEGRITY_ONLY"
        and not integrity["replay_admitted"]
    )
    assert (
        integrity["capture_remains_failed"]
        and integrity["validator_head"] == "3d2deed6065ab1a44aaab899b1b73cebf5b0c4f8"
    )
    assert (
        integrity["original_capture_head"] == "5880d8e4ba8d9563ce7df51f78e136dd9cf42b35"
    )
    assert sha(Path(capture.__file__)) == integrity["validator_sha256"]
    ruling = json.loads(ruling_path.read_text())
    assert ruling["id"] == 7071087912
    for name, key in (
        ("capture.py", "validator_sha256"),
        ("replay.py", "qualifier_sha256"),
    ):
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                integrity["validator_head"] + ":spike/k31_mla_tail_real/" + name,
            ]
        )
        assert hashlib.sha256(source).hexdigest() == integrity[key]
    capture_plan = (
        WS / "agent-team/evidence/task-19-preparation-5880d8e4-01/capture-plan.json"
    )
    assert (
        bindings[str(capture_plan)]
        == sha(capture_plan)
        == "78a5c221b2fc10f2c57e9ca4684db3518697b529225ef36bef1d23cb4fadfdcb"
    )
    original = json.loads(capture_plan.read_text())
    assert Path(original["output"]) == RECOVERY_RUN
    import io
    import tarfile

    original_root = Path(original["root"])
    historical = {
        str(Path(name).relative_to(original_root)): expected
        for name, expected in original["sources"].items()
        if Path(name).is_relative_to(original_root)
    }
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "archive",
            "--format=tar",
            original["head"],
            *historical,
        ]
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as sources:
        for name, expected in historical.items():
            member = sources.getmember(name)
            assert member.isfile()
            assert (
                hashlib.sha256(sources.extractfile(member).read()).hexdigest()
                == expected
            )
    for name, expected in original["sources"].items():
        if not Path(name).is_relative_to(original_root):
            assert sha(name) == expected, name
    ranks = integrity["ranks"]
    assert len(ranks) == 8 and [r["rank"] for r in ranks] == list(range(8))
    for rank, item in enumerate(ranks):
        assert Path(item["path"]) == RECOVERY_RUN / "data" / f"rank{rank}.invalid.pt"
        bindings[item["path"]] = item["sha256"]
    for name, expected in bindings.items():
        assert sha(name) == expected, name
    assert all(
        not Path(name).exists() for name in integrity["missing_producer_evidence"]
    )
    proof = json.loads((review_path.parent / "lifetime-audit.json").read_text())
    for name, expected in proof[
        "pinned_sources_verified_against_original_plan"
    ].items():
        assert sha(name) == expected
        bindings[name] = expected
    return {
        "state": "RECOVERED_OPERATOR_INPUT_ADMISSION",
        "ruling": ruling["html_url"],
        "producer_success": False,
        "original_capture_head": integrity["original_capture_head"],
        "executed_capture_plan_sha256": bindings[str(capture_plan)],
        "integrity_report": str(integrity_path),
        "integrity_replay_admitted": False,
        "proof_scope": {
            key: proof[key]
            for key in (
                "device_completion_proof",
                "serialization_order",
                "one_shot_source_proof",
                "retained_limits",
            )
        },
        "missing_evidence": [
            "full original drained state vector was not serialized",
            *integrity["missing_producer_evidence"],
        ],
        "bindings": bindings,
        "matched_identity": integrity["matched_identity"],
        "ranks": ranks,
        "GPU_grant_or_numerical_timing_quality_result": False,
    }


def check_recovery(path):
    path = Path(path)
    assert not path.resolve().is_relative_to(RECOVERY_RUN)
    record = json.loads(path.read_text())
    assert record == recovery_record(), "altered or wrong-run recovery record"
    return record


def recovery_checks():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    expected = recovery_record()
    failures = []
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "recovery.json"
        save(path, expected)
        assert check_recovery(path) == expected
        for field, value in (
            ("ruling", "wrong ruling"),
            ("original_capture_head", "wrong run"),
            ("producer_success", True),
            ("ranks", expected["ranks"][:-1]),
            ("ranks", expected["ranks"][::-1]),
            ("matched_identity", {}),
            ("bindings", {}),
            ("missing_evidence", []),
        ):
            save(path, {**expected, field: value})
            try:
                check_recovery(path)
            except AssertionError:
                failures.append(field)
            else:
                raise AssertionError("recovery mutation accepted: " + field)
        save(path, expected)
        original_sha = sha
        for target in (
            expected["ranks"][0]["path"],
            next(
                name
                for name in expected["bindings"]
                if name.endswith("capture-plan.json")
            ),
            next(
                name
                for name in expected["bindings"]
                if name.endswith("root-release-verification.json")
            ),
        ):
            with patch.dict(
                globals(),
                sha=lambda name, target=target: (
                    "0" * 64 if str(name) == target else original_sha(name)
                ),
            ):
                try:
                    check_recovery(path)
                except AssertionError:
                    failures.append(target)
                else:
                    raise AssertionError("changed raw/release evidence accepted")
        import copy

        import torch

        recovered_plan = {"recovery_record": str(path)}
        for rank in range(8):
            payload, actual_path = load_replay_input(recovered_plan, rank)
            assert payload["meta"]["identity"]["rank"] == rank
            assert str(actual_path) == expected["ranks"][rank]["path"]
        wrong = copy.deepcopy(payload)
        wrong["meta"]["identity"]["rank"] = 0
        with patch.object(torch, "load", return_value=wrong):
            try:
                load_replay_input(recovered_plan, 7)
            except AssertionError:
                failures.append("loaded_rank_identity")
            else:
                raise AssertionError("consumer accepted a different rank")
        with patch.object(
            capture.SNAP, "load_rank_snapshot", return_value="normal"
        ) as normal:
            loaded, actual_path = load_replay_input({"snapshot": temporary}, 0)
            assert loaded == "normal" and actual_path == Path(temporary) / "rank0.pt"
            normal.assert_called_once_with(actual_path)
        release_path = next(
            name
            for name in expected["bindings"]
            if name.endswith("root-release-verification.json")
        )

        def missing_release(name):
            if str(name) == release_path:
                raise FileNotFoundError(name)
            return original_sha(name)

        with patch.dict(globals(), sha=missing_release):
            try:
                check_recovery(path)
            except FileNotFoundError:
                failures.append("missing_release")
            else:
                raise AssertionError("missing release evidence accepted")
        path.unlink()
        try:
            check_recovery(path)
        except FileNotFoundError:
            failures.append("missing_record")
        else:
            raise AssertionError("missing recovery record accepted")
    return {
        "exact_record_passed": True,
        "refusals": failures,
        "original_integrity_replay_admitted": False,
    }


def validation_command(path):
    return [
        "flock",
        "-x",
        str(WS / "locks/cpu-compile.lock"),
        str(WS / "harness/no_gpu.sh"),
        "/opt/venv/bin/python",
        "-B",
        str(Path(__file__).resolve()),
        "--validate-results",
        "--plan",
        str(path.resolve()),
    ]


def check_plan(plan, path, worker=False):
    helpers()
    assert plan["head"] == LEGACY.git("rev-parse", "HEAD") and not LEGACY.git(
        "status", "--porcelain"
    )
    assert plan["mode"] in ("capture", "replay", "native_setup") and plan["rows"] == [
        capture.ROWS
    ]
    assert plan["max_job_seconds"] == (900 if plan["mode"] == "capture" else 180)
    assert Path(plan["output"]).resolve().is_relative_to(PRIVATE / "runs")
    assert plan["selected_bdfs"] == list(LEGACY.SELECTED_BDFS)
    assert plan["validation_command"] == validation_command(path)
    assert plan["protocol"] == (
        setup_protocol()
        if plan["mode"] == "native_setup"
        else {
            "pairs": 5,
            "aa_pairs": 5,
            "calls": 128,
            "input_sets": 1,
            "noise_fraction": 0.02,
            "reproduction": "fixed_author_observed_min_max",
        }
    )
    if plan["mode"] == "native_setup":
        assert plan["host_clock"] == host_clock_identity() and "recovery_record" in plan
    binding_functions()["verify_binding"](plan["execution"])
    marker_policy = plan["execution"]["cache_pruning_marker"]
    assert (
        marker_policy["path"] == str(PRIVATE / "cache/comgr/llvmcache.timestamp")
        and marker_policy["mutable_field"] == "mtime_ns"
    )
    if "recovery_record" in plan:
        assert plan["mode"] in ("replay", "native_setup") and "snapshot" not in plan
        check_recovery(plan["recovery_record"])
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


def make_plan(mode, output, path, snapshot=None, recovery=None):
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
            WS / "harness/no_gpu.sh",
            WS / "harness/nogpu/kfd_guard.c",
        ]
        if p.is_file()
    }
    sources.update(marker_policy_sources(execution))
    marker = PRIVATE / "cache/comgr/llvmcache.timestamp"
    execution["cache_pruning_marker"] = {
        "path": str(marker),
        "observed_mtime_ns": marker.stat().st_mtime_ns,
        "mutable_field": "mtime_ns",
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
        "cache_policy": "Private compiled artifacts/code/models remain byte/stat bound. Only the exact empty regular COMGR pruning marker permits mtime drift; other identity fields remain bound. Higher-level graph indexes with old absolute paths stay inactive; new artifacts remain private.",
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
        "validation_command": validation_command(path),
        "gpu_authorized_by_plan": False,
    }
    if recovery is not None:
        assert mode in ("replay", "native_setup") and snapshot is None
        record = check_recovery(recovery)
        plan["recovery_record"] = str(recovery.resolve())
        plan["sources"].update(record["bindings"])
        plan["sources"][str(recovery.resolve())] = sha(recovery)
    elif mode == "native_setup":
        raise ValueError("native setup diagnostic requires the exact recovery record")
    elif mode == "replay":
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
    if mode == "native_setup":
        plan["protocol"] = setup_protocol()
        plan["host_clock"] = host_clock_identity()
        for name, digest in SETUP_EVIDENCE.items():
            assert sha(name) == digest, name
        plan["sources"].update(SETUP_EVIDENCE)
        plan["treatment"] = (
            "Finite native-only F/P event-setup diagnostic; old ADVANCE/INCONCLUSIVE and ranges unchanged; no performance acceptance or timer adoption"
        )
    return plan


SETUP_EVIDENCE = {
    "/share_nfs/whx/work/k31_perf_opt/.opt-team/agent-team/work/root-19-noise-published-copy-7073444630/ruling.md": "3b2607c55a1eeef1334bf1f98f9077bd90a0db47477a36fa54dafc8c051579b3",
    "/share_nfs/whx/work/k31_perf_opt/.opt-team/agent-team/evidence/task-19-author-replay-661bbb29-01/reproduction-intervals.json": "1fcb193fa6b0fbbec1b58decfb9379f500eabb64e0adcb175a66b0cb5e74249c",
    "/share_nfs/whx/work/k31_perf_opt/.opt-team/agent-team/evidence/review-19-reproduction-outcome-661bbb29-01/verdict.json": "c20f2bf8ac10b2a5a0139eb7e1a1a5accec54358ace8826b014e2301a6a387f5",
}

EVENT_FLAGS = {
    "enable_timing": True,
    "blocking": False,
    "interprocess": False,
    "external": False,
}


def host_clock_identity():
    import socket

    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "time_namespace": os.readlink("/proc/self/ns/time"),
        "clock": "CLOCK_MONOTONIC",
    }


def setup_protocol():
    return {
        "kind": "native_only_event_setup_diagnostic",
        "calls": 128,
        "windows": [["F", "P"], ["P", "F"], ["F", "P"], ["P", "F"], ["F", "P"]],
        "labels_even": ["native", "donor"],
        "labels_odd": ["donor", "native"],
        "label_targets": {"native": "native", "donor": "native"},
        "timed_replays_per_rank": 20,
        "warmups_per_timed_replay": 1,
        "event_flags": EVENT_FLAGS,
        "stream": "one dedicated per-rank measured stream for both settings",
        "duration_gather": "one existing default-Gloo all_gather per observation, after final sync; raw batch_ms and batch_ms*1000/128",
        "publishing": "buffer locally; write after the fixed set or a handled failure, never inside a timing window",
        "P_pairs": 1,
        "P_priming": "two records and device sync before first diagnostic warmup/barrier",
        "F_pairs": 10,
        "F_setup": "fresh pair after warmup/device sync/default Gloo barrier",
        "F_retirement": "clear each completed F pair after duration gather, observed separately before the next warmup",
        "P_retirement": "retain through the entire fixed set; release at measurement-function exit",
        "observer": "buffered CLOCK_MONOTONIC call entry/return; no GPU-start or peer-wait inference",
        "noise_fraction": 0.02,
        "interruption": "SIGTERM/SIGINT raise and save buffered partial records outside normal timing; incomplete returns remain absent; parent ownership/release unchanged",
        "preparation": {
            "materialization_event_pairs": 1,
            "native_eager_calls": 2,
            "native_capture_nodes": 128,
            "control_graph_replays": 4,
            "diagnostic_warmups": 20,
            "diagnostic_timed_replays": 20,
            "post_diagnostic_graph_replays": 0,
            "between_sample_workspace_resets": 0,
        },
        "interpretation": "setup sensitivity only when setting differences have corresponding host components; nonrecurrence/mixed results unresolved; no old-result repair or timer adoption",
    }


def setup_order():
    return [
        (window, setting, label)
        for window, settings in enumerate(setup_protocol()["windows"])
        for setting in settings
        for label in (("native", "donor") if window % 2 == 0 else ("donor", "native"))
    ]


def observe_call(spans, name, function, *args, **kwargs):
    span = [None, None]
    spans[name] = span
    span[0] = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    value = function(*args, **kwargs)
    span[1] = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return value


def setup_measure(graph, stream, result):
    import torch
    import torch.distributed as dist

    def events():
        return [torch.cuda.Event(**EVENT_FLAGS), torch.cuda.Event(**EVENT_FLAGS)]

    priming = result["P_initial"] = {
        "phase": "before_first_diagnostic_warmup",
        "spans": {},
        "pair": "P0",
    }
    primed = observe_call(priming["spans"], "create", events)
    observe_call(priming["spans"], "start_record", primed[0].record, stream)
    observe_call(priming["spans"], "end_record", primed[1].record, stream)
    observe_call(priming["spans"], "synchronize", torch.cuda.synchronize)
    result["samples"], result["pairs"] = [], []
    fresh = 0
    for ordinal, (window, setting, label) in enumerate(setup_order(), 1):
        sample = {
            "ordinal": ordinal,
            "window": window,
            "setting": setting,
            "label": label,
            "spans": {},
            "complete": False,
        }
        result["samples"].append(sample)
        spans = sample["spans"]
        observe_call(spans, "warmup_submit", graph.replay)
        observe_call(spans, "warmup_synchronize", torch.cuda.synchronize)
        observe_call(spans, "barrier", dist.barrier)
        if setting == "F":
            pair = observe_call(spans, "create", events)
            fresh += 1
            sample["pair"] = "F" + str(fresh)
        else:
            pair = primed
            sample["pair"] = "P0"
        observe_call(spans, "start_record", pair[0].record, stream)
        observe_call(spans, "graph_submit", graph.replay)
        observe_call(spans, "end_record", pair[1].record, stream)
        observe_call(spans, "final_synchronize", torch.cuda.synchronize)
        sample["local_batch_ms"] = pair[0].elapsed_time(pair[1])
        sample["local_us"] = sample["local_batch_ms"] * 1000 / 128
        local = torch.tensor(
            [sample["local_batch_ms"], sample["local_us"]],
            dtype=torch.float64,
            device="cpu",
        )
        gathered = [torch.empty_like(local) for _ in range(8)]
        dist.all_gather(gathered, local)
        batch_times, durations = zip(
            *(entry.tolist() for entry in gathered), strict=True
        )
        sample.update(batch_ms=list(batch_times), ranks_us=list(durations))
        assert all(0 < v < float("inf") for v in durations)
        sample.update(
            max_us=max(durations),
            duration_spread_us=max(durations) - min(durations),
        )
        if setting == "F":
            observe_call(spans, "retire", pair.clear)
        sample["complete"] = True
    for window, settings in enumerate(setup_protocol()["windows"]):
        for setting in settings:
            pair = {
                s["label"]: s
                for s in result["samples"]
                if s["window"] == window and s["setting"] == setting
            }
            ratio = pair["native"]["max_us"] / pair["donor"]["max_us"]
            result["pairs"].append(
                {
                    "window": window,
                    "setting": setting,
                    "ratio": ratio,
                    "within_2pct": abs(ratio - 1) <= 0.02,
                }
            )
    return result


def validate_setup_observations(result):
    def ordered(spans, names):
        assert set(spans) == set(names)
        previous = 0
        for name in names:
            entry, returned = spans[name]
            assert (
                type(entry) is int
                and type(returned) is int
                and previous <= entry <= returned
            )
            previous = returned
        return previous

    initial = result["P_initial"]
    assert (
        initial["phase"] == "before_first_diagnostic_warmup" and initial["pair"] == "P0"
    )
    primed_at = ordered(
        initial["spans"], ("create", "start_record", "end_record", "synchronize")
    )
    assert len(result["samples"]) == 20 and len(result["pairs"]) == 10
    previous, fresh = primed_at, 0
    for ordinal, (sample, expected) in enumerate(
        zip(result["samples"], setup_order(), strict=True), 1
    ):
        assert (
            sample["window"],
            sample["setting"],
            sample["label"],
        ) == expected and sample["ordinal"] == ordinal
        assert sample["complete"] and previous <= sample["spans"]["warmup_submit"][0]
        names = ["warmup_submit", "warmup_synchronize", "barrier"]
        if sample["setting"] == "F":
            fresh += 1
            assert sample["pair"] == "F" + str(fresh)
            names.append("create")
        else:
            assert sample["pair"] == "P0"
        names += ["start_record", "graph_submit", "end_record", "final_synchronize"]
        if sample["setting"] == "F":
            names.append("retire")
        previous = ordered(sample["spans"], names)
        assert 0 < sample["local_batch_ms"] < float("inf")
        assert sample["local_us"] == sample["local_batch_ms"] * 1000 / 128
        assert len(sample["batch_ms"]) == len(sample["ranks_us"]) == 8
        assert all(0 < value < float("inf") for value in sample["batch_ms"])
        assert [value * 1000 / 128 for value in sample["batch_ms"]] == sample[
            "ranks_us"
        ]
        assert sample["max_us"] == max(sample["ranks_us"])
        assert sample["duration_spread_us"] == max(sample["ranks_us"]) - min(
            sample["ranks_us"]
        )
    expected_pairs = [
        (w, s)
        for w, settings in enumerate(setup_protocol()["windows"])
        for s in settings
    ]
    for pair, key in zip(result["pairs"], expected_pairs, strict=True):
        assert (pair["window"], pair["setting"]) == key
        samples = {
            s["label"]: s
            for s in result["samples"]
            if (s["window"], s["setting"]) == key
        }
        ratio = samples["native"]["max_us"] / samples["donor"]["max_us"]
        assert pair["ratio"] == ratio and pair["within_2pct"] == (
            abs(ratio - 1) <= 0.02
        )
    return True


def setup_host_summary(ranks):
    def joined(spans):
        result = {}
        assert all(set(s) == set(spans[0]) for s in spans)
        for name in spans[0]:
            entries = [s[name][0] for s in spans]
            returns = [s[name][1] for s in spans]
            result[name] = {
                "entry_spread_ns": max(entries) - min(entries),
                "return_spread_ns": max(returns) - min(returns),
                "call_duration_ns_per_rank": [
                    end - start for start, end in zip(entries, returns, strict=True)
                ],
            }
        return result

    samples = []
    for ordinal in range(20):
        peers = [row["samples"][ordinal] for row in ranks]
        record = {
            name: peers[0][name]
            for name in (
                "ordinal",
                "window",
                "setting",
                "label",
                "max_us",
                "duration_spread_us",
            )
        }
        record["host_calls"] = joined([p["spans"] for p in peers])
        record["host_gaps_ns_per_rank"] = {
            "barrier_return_to_start_record_entry": [
                p["spans"]["start_record"][0] - p["spans"]["barrier"][1] for p in peers
            ],
            "start_record_return_to_graph_submit_entry": [
                p["spans"]["graph_submit"][0] - p["spans"]["start_record"][1]
                for p in peers
            ],
            "graph_submit_return_to_end_record_entry": [
                p["spans"]["end_record"][0] - p["spans"]["graph_submit"][1]
                for p in peers
            ],
        }
        samples.append(record)
    return {
        "state": "SETUP_DIAGNOSTIC_OBSERVATIONS_ONLY",
        "host": ranks[0]["host"],
        "rank_identities": [
            {"rank": r["rank"], "hardware": r["hardware"], "stream": r["stream"]}
            for r in ranks
        ],
        "initial_P_host_calls": joined([r["P_initial"]["spans"] for r in ranks]),
        "samples": samples,
        "pairs": ranks[0]["pairs"],
        "noise_counts": {
            s: sum(not p["within_2pct"] for p in ranks[0]["pairs"] if p["setting"] == s)
            for s in ("F", "P")
        },
        "scope": "Host call entry/return and duration envelopes only; not GPU starts, peer waiting or clock evidence. No overhead subtraction, old-verdict repair, timer adoption or next job.",
    }


def native_diagnostic_graph(group, rank, device, item, stream, capacity):
    import torch
    import torch.distributed as dist

    from atom.model_ops import dcp_ops
    from atom.model_ops.attention_mla import MLAAttention, is_rocm_aiter_fp4bmm_enabled
    from atom.model_ops.triton_fused_sigmoid_mul_quant import (
        fused_sigmoid_mul_maybe_quant,
    )

    assert not is_rocm_aiter_fp4bmm_enabled()
    tail = TAIL.Tail.__new__(TAIL.Tail)
    tail.torch, tail.ops, tail.gate = torch, dcp_ops, fused_sigmoid_mul_maybe_quant
    tail.rank, tail.device, tail.group = rank, device, group
    tail.native = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(tail.native)
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
        setattr(tail.native, key, value)
    assert dcp_ops.prepare_dcp_mailbox(group, 12, 128, capacity)
    tail.key = dcp_ops._mailbox_key(group, 12, 128)
    before = tail.calls()
    for _ in range(2):
        tail.native_call(item)
    assert tail.calls() == before
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(128):
            tail.native_call(item)
    assert tail.calls() - before == 128
    dcp_ops.check_dcp_mailbox_launches()
    return tail, graph


def save_setup_result(path, result):
    import math

    def invalid_values(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"invalid_nonfinite_float": repr(value)}
        if isinstance(value, dict):
            return {key: invalid_values(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [invalid_values(item) for item in value]
        return value

    save(path, result if result["complete"] else invalid_values(result))


def setup_worker(rank, plan, snapshot, input_path, device, stream, group, hardware):
    import torch
    import torch.distributed as dist
    from aiter.ops.flydsl.dcp_mailbox.common import token_tile

    host = host_clock_identity()
    assert host == plan["host_clock"]
    result = {
        "rank": rank,
        "hardware": hardware,
        "host": host,
        "stream": int(stream.cuda_stream),
        "input_sha256": sha(input_path),
        "protocol": setup_protocol(),
        "complete": False,
        "old_author_and_independent_results_unchanged": True,
        "diagnostic_not_method_adoption": True,
    }

    def interrupted(sig, frame):
        result["stop_signal"] = sig
        raise InterruptedError("native setup diagnostic interrupted")

    handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with torch.cuda.stream(stream):
            load_start, load_end = torch.cuda.Event(**EVENT_FLAGS), torch.cuda.Event(
                **EVENT_FLAGS
            )
            load_start.record(stream)
            tensors = capture.SNAP.restore_tensors(snapshot, device)
            load_end.record(stream)
            torch.cuda.synchronize()
            result["materialization_us"] = load_start.elapsed_time(load_end) * 1000
            result["copy_bytes"] = sum(v.numel() for v in snapshot["arenas"].values())
            item = {
                "x": tensors["raw"],
                "lse": tensors["lse"],
                "owned": tensors["owned"],
                "weight": tensors["w_v"],
                "gate": tensors["gate"],
            }
            saved = {k: value.clone() for k, value in item.items()}
            tail, graph = native_diagnostic_graph(
                group,
                rank,
                device,
                item,
                stream,
                snapshot["meta"]["key"]["workspace_capacity"],
            )
            workspace = capture.workspace_info(tail.ops._MAILBOX_WORKSPACES[tail.key])
            assert token_tile(capture.ROWS) == snapshot["meta"]["key"]["token_tile"]
            assert {
                k: v for k, v in workspace.items() if k not in ("base", "state_base")
            } == {
                k: v
                for k, v in snapshot["meta"]["workspace"].items()
                if k not in ("base", "state_base")
            }
            result["workspace"], result["capture_key"] = (
                workspace,
                snapshot["meta"]["key"],
            )
            result["graph_proof"] = {
                "decode": True,
                "captured_native_nodes": 128,
                "eager_mailbox_calls": 0,
                "donor_graphs_or_calls": 0,
            }
            graph.replay()
            torch.cuda.synchronize()
            result["native_before"] = compare_global(
                item["native_out"], tensors["reference"]
            )
            empty = []
            for case in ("last_row", "all_rows"):
                item["owned"].copy_(saved["owned"])
                item["x"].copy_(saved["x"])
                rows = slice(-1, None) if case == "last_row" else slice(None)
                item["owned"][rows] = 0
                item["x"][rows] = float("nan")
                graph.replay()
                torch.cuda.synchronize()
                empty.append(
                    {
                        "case": case,
                        "passed": bool(
                            (item["native_out"][rows] == 0).all()
                            and torch.isfinite(item["native_out"]).all()
                        ),
                    }
                )
            for key, value in saved.items():
                item[key].copy_(value)
            graph.replay()
            torch.cuda.synchronize()
            result["native_restored"] = compare_global(
                item["native_out"], tensors["reference"]
            )
            statuses = [None] * 8
            dist.all_gather_object(
                statuses,
                result["native_before"]["passed"]
                and result["native_restored"]["passed"]
                and all(e["passed"] for e in empty),
            )
            result["empty_controls"] = empty
            assert all(statuses), "native preparation correctness failed"
            setup_measure(graph, stream, result)
            result["native_after"] = compare_global(
                item["native_out"], tensors["reference"]
            )
            readonly = all(
                torch.equal(item[key], value) for key, value in saved.items()
            )
            dist.all_gather_object(
                statuses, readonly and result["native_after"]["passed"]
            )
            result["inputs_unchanged"] = readonly
            assert all(statuses), "native diagnostic correctness/lifetime failed"
            validate_setup_observations(result)
            result["complete"] = True
            result["verdict"] = "SETUP_DIAGNOSTIC_OBSERVATIONS_ONLY"
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        result["error"] = repr(error)
        result["complete"] = False
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        save_setup_result(Path(plan["output"]) / f"rank{rank}.json", result)


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


def load_replay_input(plan, rank):
    import torch

    capture.initialize_snapshot()
    if "recovery_record" in plan:
        record = check_recovery(plan["recovery_record"])
        input_path = Path(record["ranks"][rank]["path"])
        snapshot = torch.load(input_path, map_location="cpu", weights_only=True)
        capture.validate_snapshot(snapshot)
        entry = record["ranks"][rank]
        assert snapshot["meta"]["identity"] == entry["identity"]
        assert (
            snapshot["meta"]["key"] == entry["key"]
            and snapshot["tensor_hashes"] == entry["tensor_hashes"]
        )
    else:
        input_path = Path(plan["snapshot"]) / f"rank{rank}.pt"
        snapshot = capture.SNAP.load_rank_snapshot(input_path)
    return snapshot, input_path


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
    snapshot, input_path = load_replay_input(plan, rank)
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
    if plan["mode"] == "native_setup":
        return setup_worker(
            rank, plan, snapshot, input_path, device, stream, group, hardware
        )
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
            "input_sha256": sha(input_path),
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
        if plan["mode"] == "native_setup":
            admitted = json.loads(Path(plan["recovery_record"]).read_text())
            for rank, row in enumerate(ranks):
                assert row["rank"] == row["hardware"]["rank"] == rank
                expected = admitted["ranks"][rank]
                assert (
                    row["input_sha256"] == expected["sha256"]
                    and row["capture_key"] == expected["key"]
                )
                assert {
                    k: v
                    for k, v in row["workspace"].items()
                    if k not in ("base", "state_base")
                } == {
                    k: v
                    for k, v in expected["workspace"].items()
                    if k not in ("base", "state_base")
                }
                assert (
                    row["complete"] and "error" not in row and row["inputs_unchanged"]
                )
                assert (
                    row["host"] == plan["host_clock"]
                    and row["protocol"] == setup_protocol()
                )
                assert all(
                    row[name]["passed"]
                    for name in ("native_before", "native_restored", "native_after")
                )
                assert len(row["empty_controls"]) == 2 and all(
                    c["passed"] for c in row["empty_controls"]
                )
                assert row["graph_proof"] == {
                    "decode": True,
                    "captured_native_nodes": 128,
                    "eager_mailbox_calls": 0,
                    "donor_graphs_or_calls": 0,
                }
                validate_setup_observations(row)
                for sample in row["samples"]:
                    assert sample["local_batch_ms"] == sample["batch_ms"][rank]
                    assert sample["local_us"] == sample["ranks_us"][rank]
            common = lambda row: [
                {
                    k: v
                    for k, v in sample.items()
                    if k not in ("spans", "local_batch_ms", "local_us")
                }
                for sample in row["samples"]
            ]
            assert all(
                common(row) == common(ranks[0]) and row["pairs"] == ranks[0]["pairs"]
                for row in ranks
            )
            save(output / "setup-summary.json", setup_host_summary(ranks))
        else:
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
    start, validation = None, None
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
        deadline = start + plan["max_job_seconds"]
        expected_workers = 1 if plan["mode"] == "capture" else 8
        for rank in range(expected_workers):
            if stopped or time.monotonic() >= deadline:
                break
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
            if plan["mode"] != "capture":
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
        for phase in ("work", "validation"):
            if phase == "validation":
                if stopped or time.monotonic() >= deadline:
                    raise TimeoutError("work deadline reached before validation")
                validation = common.spawn(
                    plan["validation_command"],
                    env={**env, "KFD_GUARD_LOG": str(output / "validation-kfd.log")},
                    cwd=ROOT,
                    log_path=output / "validation.log",
                    lock_fd=fd,
                    cpus="112-127",
                )
                owned.add(validation)
            active = children if phase == "work" else [validation]
            while (
                any(p.poll() is None for p in active)
                and not stopped
                and all(p.poll() in (None, 0) for p in active)
                and time.monotonic() < deadline
            ):
                owned.live()
                time.sleep(0.05)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{phase} exceeded the absolute work deadline")
            if (
                stopped
                or len(children) != expected_workers
                or any(p.poll() != 0 for p in active)
            ):
                raise RuntimeError(f"{phase} cancelled or exited unsuccessfully")
        result["validated_results"] = True
        result["complete"] = True
    except BaseException as error:  # noqa: BLE001 - cleanup remains owned
        result["error"] = repr(error)
        result["complete"] = False
    finally:
        result.update(
            exit_codes=[p.poll() for p in children],
            validation_exit_code=None if validation is None else validation.poll(),
            stop_signals=stopped,
            elapsed_s=None if start is None else time.monotonic() - start,
        )
        if start is not None and result["elapsed_s"] >= plan["max_job_seconds"]:
            result["complete"] = False
            result.setdefault("error", "absolute work deadline exceeded")
        cleanup_started = time.monotonic()
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
                result["cleanup_elapsed_s"] = time.monotonic() - cleanup_started
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


def deadline_checks():
    """Real CPU children expose late terminal observations and slow validation."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace as NS
    from unittest.mock import patch

    helpers()
    actual, Owned = LEGACY.resource_components()
    records = []
    inherited_env = dict(os.environ)
    original_sleep = time.sleep
    for case, limit, worker_seconds, validation_seconds in (
        ("late_child", 0.06, 0.08, 0),
        ("late_validation", 0.15, 0.01, 0.30),
        ("within_deadline", 1, 0.01, 0.01),
    ):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            descriptors = []

            class CpuLock:
                def acquire(self, directory=directory, descriptors=descriptors):
                    descriptor = os.open(
                        directory / "cpu.lock", os.O_CREAT | os.O_RDWR, 0o600
                    )
                    descriptors.append(descriptor)
                    return descriptor

            def spawn(
                command,
                directory=directory,
                validation_seconds=validation_seconds,
                worker_seconds=worker_seconds,
                **kwargs,
            ):
                seconds = (
                    validation_seconds
                    if "--validate-results" in command
                    else worker_seconds
                )
                return actual.spawn(
                    ["/bin/sleep", str(seconds)],
                    env=inherited_env,
                    cwd=ROOT,
                    log_path=directory / "child.log",
                    lock_fd=kwargs["lock_fd"],
                    cpus="112-127",
                )

            common = NS(
                GpuLock=CpuLock,
                spawn=spawn,
                descendants=actual.descendants,
                assert_gpus_idle=lambda: {"cpu_stub": True},
                assert_ports_free=lambda ports: None,
                save_json=save,
            )
            plan = {
                "head": "cpu-deadline",
                "mode": "capture",
                "output": str(directory / "out"),
                "ports": [],
                "environment": {},
                "max_job_seconds": limit,
                "validation_command": ["--validate-results"],
            }
            path = directory / "plan.json"
            save(path, plan)
            env = {
                k: v
                for k, v in inherited_env.items()
                if k not in ("LD_PRELOAD", "KFD_GUARD_LOG")
            }
            env.update(K31_TASK19_GPU_GRANT=plan["head"], K31_TASK19_PLAN_SHA=sha(path))

            def validate(_plan, seconds=validation_seconds):
                original_sleep(seconds)
                return True

            def scheduling_delay(seconds, case=case):
                original_sleep(
                    0.20 if case == "late_child" and seconds == 0.05 else seconds
                )

            started = time.monotonic()
            with (
                patch.dict(os.environ, env, clear=True),
                patch("os.nice", lambda n: 0),
                patch("os.sched_getaffinity", lambda pid: {0}),
                patch.dict(
                    globals(), check_plan=lambda *args: None, validate_results=validate
                ),
                patch.object(time, "sleep", scheduling_delay),
                patch.object(
                    LEGACY, "resource_components", lambda common=common: (common, Owned)
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
            result = json.loads((directory / "out/controller.json").read_text())
            assert result["cleanup"]["released"] and not result["cleanup"]["alive"]
            try:
                os.fstat(descriptors[0])
            except OSError:
                pass
            else:
                raise AssertionError("deadline control leaked its ownership lock")
            records.append(
                {
                    "case": case,
                    "declared_bound_s": limit,
                    "complete": result["complete"],
                    "reported_elapsed_s": result["elapsed_s"],
                    "wall_s": time.monotonic() - started,
                    "owned_release": result["cleanup"]["released"],
                }
            )
    assert all(
        not r["complete"] and r["reported_elapsed_s"] >= r["declared_bound_s"]
        for r in records[:2]
    ), records
    assert (
        records[2]["complete"]
        and records[2]["reported_elapsed_s"] < records[2]["declared_bound_s"]
    ), records
    return records


def setup_cpu_checks():
    import copy
    import weakref
    from unittest.mock import patch

    import torch
    import torch.distributed as dist

    helpers()
    seen, events, created_after_barriers = [], [], []
    state = {
        "inside": False,
        "barriers": 0,
        "graphs": 0,
        "syncs": 0,
        "elapsed": 0,
        "gathers": 0,
        "clock": 1_000_000,
    }
    stream = object()

    def now(clock):
        assert clock == time.CLOCK_MONOTONIC
        state["clock"] += 10
        return state["clock"]

    class Event:
        def __init__(self, **flags):
            assert flags == EVENT_FLAGS and not state["inside"]
            self.index = len(events)
            self.ready, self.recorded = False, False
            events.append(weakref.ref(self))
            created_after_barriers.append(state["barriers"])
            seen.append("create")

        def __del__(self):
            seen.append("destroy")

        def record(self, given_stream):
            assert given_stream is stream and (not self.recorded or self.ready)
            self.ready, self.recorded = False, True
            state["inside"] = self.index % 2 == 0
            seen.append("start_record" if self.index % 2 == 0 else "end_record")

        def elapsed_time(self, other):
            assert self.ready and other.ready and not state["inside"]
            value = 0.3072 if state["elapsed"] == 0 else 0.256
            state["elapsed"] += 1
            seen.append("elapsed")
            return value

    class Graph:
        def replay(self):
            if state["graphs"] == 0:
                assert len(events) == 2 and all(
                    ref().ready and ref().recorded for ref in events
                ), "P was not primed before first warmup"
            state["graphs"] += 1
            seen.append("timed_graph" if state["inside"] else "warmup_graph")

    def synchronize():
        assert not state["inside"]
        for ref in events:
            event = ref()
            if event is not None and event.recorded:
                event.ready = True
        state["syncs"] += 1
        seen.append("synchronize")

    def barrier():
        assert not state["inside"]
        state["barriers"] += 1
        seen.append("barrier")

    def gather(receivers, value):
        assert not state["inside"] and len(receivers) == 8
        for rank, receiver in enumerate(receivers):
            ms = value[0].item() + rank * 0.00001
            receiver.copy_(torch.tensor([ms, ms * 1000 / 128], dtype=torch.float64))
        state["gathers"] += 1
        seen.append("gather")

    with (
        patch.object(torch.cuda, "Event", Event),
        patch.object(torch.cuda, "synchronize", synchronize),
        patch.object(dist, "barrier", barrier),
        patch.object(dist, "all_gather", gather),
        patch.object(time, "clock_gettime_ns", now),
    ):
        result = setup_measure(Graph(), stream, {})
    assert EVENT_FLAGS == {
        "enable_timing": True,
        "blocking": False,
        "interprocess": False,
        "external": False,
    }
    assert [
        [pair["setting"] for pair in result["pairs"] if pair["window"] == window]
        for window in range(5)
    ] == [["F", "P"], ["P", "F"], ["F", "P"], ["P", "F"], ["F", "P"]]
    for setting in ("F", "P"):
        assert [
            sample["label"]
            for sample in result["samples"]
            if sample["setting"] == setting
        ] == [
            "native",
            "donor",
            "donor",
            "native",
            "native",
            "donor",
            "donor",
            "native",
            "native",
            "donor",
        ]
    assert seen[:5] == ["create", "create", "start_record", "end_record", "synchronize"]
    assert len(events) == 22 and state["graphs"] == 40 and state["syncs"] == 41
    assert state["barriers"] == state["gathers"] == state["elapsed"] == 20
    assert sum(not pair["within_2pct"] for pair in result["pairs"]) == 1
    fresh_ordinals = [
        i for i, (_, setting, _) in enumerate(setup_order(), 1) if setting == "F"
    ]
    assert created_after_barriers[2::2] == fresh_ordinals
    expected_trace = list(seen[:5])
    for _, setting, _ in setup_order():
        expected_trace += ["warmup_graph", "synchronize", "barrier"]
        if setting == "F":
            expected_trace += ["create", "create"]
        expected_trace += [
            "start_record",
            "timed_graph",
            "end_record",
            "synchronize",
            "elapsed",
            "gather",
        ]
        if setting == "F":
            expected_trace += ["destroy", "destroy"]
    expected_trace += ["destroy", "destroy"]
    assert (
        seen == expected_trace
    ), "F retirement was not after its gather and before the next warmup"
    assert all(ref() is None for ref in events), "event references retained"
    assert validate_setup_observations(result)
    failures = []
    for name, edit in (
        ("missing_observation", lambda r: r["samples"].pop()),
        ("wrong_setting", lambda r: r["samples"][0].update(setting="P")),
        (
            "late_P_prime",
            lambda r: r["P_initial"]["spans"]["synchronize"].__setitem__(
                1, r["samples"][0]["spans"]["warmup_submit"][0] + 1
            ),
        ),
        ("missing_F_setup", lambda r: r["samples"][0]["spans"].pop("create")),
        ("invented_P_setup", lambda r: r["samples"][2]["spans"].update(create=[1, 2])),
        ("missing_F_retirement", lambda r: r["samples"][0]["spans"].pop("retire")),
        ("local_conversion", lambda r: r["samples"][0].update(local_us=99)),
        (
            "raw_batch_conversion",
            lambda r: r["samples"][0]["ranks_us"].__setitem__(0, 99),
        ),
        ("noise_relabel", lambda r: r["pairs"][0].update(within_2pct=True)),
        (
            "incomplete_call",
            lambda r: r["samples"][0]["spans"]["graph_submit"].__setitem__(1, None),
        ),
    ):
        bad = copy.deepcopy(result)
        edit(bad)
        try:
            validate_setup_observations(bad)
        except (AssertionError, KeyError, TypeError, ValueError):
            failures.append(name)
        else:
            raise AssertionError("diagnostic evidence mutation accepted: " + name)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        out = Path(directory)
        host = host_clock_identity()
        admitted = {
            "ranks": [
                {
                    "sha256": f"controlled-{rank}",
                    "key": {"capacity": 96},
                    "workspace": {"bytes": 4792352},
                }
                for rank in range(8)
            ]
        }
        save(out / "admission.json", admitted)
        plan = {
            "mode": "native_setup",
            "output": directory,
            "selected_bdfs": list(LEGACY.SELECTED_BDFS),
            "host_clock": host,
            "recovery_record": str(out / "admission.json"),
        }
        for rank in range(8):
            row = copy.deepcopy(result)
            row.update(
                rank=rank,
                hardware={"bdf": LEGACY.SELECTED_BDFS[rank], "rank": rank},
                stream=rank + 1,
                host=host,
                protocol=setup_protocol(),
                complete=True,
                inputs_unchanged=True,
                input_sha256=admitted["ranks"][rank]["sha256"],
                capture_key={"capacity": 96},
                workspace={"bytes": 4792352},
                native_before={"passed": True},
                native_restored={"passed": True},
                native_after={"passed": True},
                empty_controls=[{"passed": True}, {"passed": True}],
                graph_proof={
                    "decode": True,
                    "captured_native_nodes": 128,
                    "eager_mailbox_calls": 0,
                    "donor_graphs_or_calls": 0,
                },
            )
            for spans in [
                row["P_initial"]["spans"],
                *[sample["spans"] for sample in row["samples"]],
            ]:
                for span in spans.values():
                    span[0] += rank * 100
                    span[1] += rank * 100
            for sample in row["samples"]:
                sample["local_batch_ms"] = sample["batch_ms"][rank]
                sample["local_us"] = sample["ranks_us"][rank]
            save(out / f"rank{rank}.json", row)
        assert validate_results(plan)
        joined = json.loads((out / "setup-summary.json").read_text())
        assert (
            joined["samples"][0]["host_calls"]["graph_submit"]["entry_spread_ns"] == 700
        )
        assert (
            joined["samples"][0]["host_calls"]["graph_submit"][
                "call_duration_ns_per_rank"
            ]
            == [10] * 8
        )
        assert joined["noise_counts"] == {"F": 1, "P": 0}
        assert [r["rank"] for r in joined["rank_identities"]] == list(range(8))
        original = json.loads((out / "rank0.json").read_text())
        for name, edit in (
            ("failed_native_before", lambda r: r["native_before"].update(passed=False)),
            ("failed_native_after", lambda r: r["native_after"].update(passed=False)),
            ("changed_inputs", lambda r: r.update(inputs_unchanged=False)),
            (
                "failed_empty_zero",
                lambda r: r["empty_controls"][0].update(passed=False),
            ),
            ("wrong_clock_domain", lambda r: r["host"].update(boot_id="wrong")),
            ("wrong_input_hash", lambda r: r.update(input_sha256="wrong")),
            ("wrong_workspace", lambda r: r["workspace"].update(bytes=1)),
            ("wrong_hardware_rank", lambda r: r["hardware"].update(rank=1)),
            ("wrong_hardware_bdf", lambda r: r["hardware"].update(bdf="wrong")),
            (
                "wrong_local_rank_value",
                lambda r: r["samples"][0].update(
                    local_batch_ms=r["samples"][0]["batch_ms"][1],
                    local_us=r["samples"][0]["ranks_us"][1],
                ),
            ),
        ):
            bad = copy.deepcopy(original)
            edit(bad)
            save(out / "rank0.json", bad)
            try:
                validate_results(plan)
            except AssertionError:
                failures.append(name)
            else:
                raise AssertionError("diagnostic result mutation accepted: " + name)
        save(out / "rank0.json", original)
        peer = json.loads((out / "rank1.json").read_text())
        save(out / "rank0.json", peer)
        save(out / "rank1.json", original)
        try:
            validate_results(plan)
        except AssertionError:
            failures.append("swapped_rank_files")
        else:
            raise AssertionError("swapped rank files accepted")
        save(out / "rank0.json", original)
        save(out / "rank1.json", peer)
        assert validate_results(plan)
    events.clear()
    seen.clear()
    state.update(
        inside=False,
        barriers=0,
        graphs=0,
        syncs=0,
        elapsed=0,
        gathers=0,
        clock=1_000_000,
    )
    original_replay = Graph.replay

    def fail_submission(graph):
        original_replay(graph)
        if state["graphs"] == 4:
            raise RuntimeError("controlled submission failure")

    partial = {}
    with (
        patch.object(torch.cuda, "Event", Event),
        patch.object(torch.cuda, "synchronize", synchronize),
        patch.object(dist, "barrier", barrier),
        patch.object(dist, "all_gather", gather),
        patch.object(time, "clock_gettime_ns", now),
        patch.object(Graph, "replay", fail_submission),
    ):
        try:
            setup_measure(Graph(), stream, partial)
        except RuntimeError as error:
            assert str(error) == "controlled submission failure"
        else:
            raise AssertionError("hard submission failure did not stop")
    assert len(partial["samples"]) == 2 and partial["samples"][0]["complete"]
    assert (
        not partial["samples"][1]["complete"]
        and partial["samples"][1]["spans"]["graph_submit"][1] is None
    )
    assert state["graphs"] == 4 and state["gathers"] == 1
    for case in ("gather_failure", "gather_interruption", "invalid_gather"):
        events.clear()
        seen.clear()
        state.update(
            inside=False,
            barriers=0,
            graphs=0,
            syncs=0,
            elapsed=0,
            gathers=0,
            clock=1_000_000,
        )
        partial = {}

        def failed_gather(receivers, value, case=case):
            if case == "gather_failure":
                raise RuntimeError(case)
            if case == "gather_interruption":
                raise InterruptedError(case)
            gather(receivers, value)
            for receiver, invalid in zip(
                receivers[:3], (float("inf"), -float("inf"), float("nan")), strict=True
            ):
                receiver.fill_(invalid)

        with (
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "synchronize", synchronize),
            patch.object(dist, "barrier", barrier),
            patch.object(dist, "all_gather", failed_gather),
            patch.object(time, "clock_gettime_ns", now),
        ):
            try:
                setup_measure(Graph(), stream, partial)
            except (RuntimeError, InterruptedError, AssertionError):
                pass
            else:
                raise AssertionError("gather failure did not stop")
        assert state["graphs"] == 2 and len(partial["samples"]) == 1
        sample = partial["samples"][0]
        assert (
            sample["local_batch_ms"] == 0.3072 and sample["local_us"] == 2.4
        ), "returned local timing lost"
        assert not sample["complete"] and "max_us" not in sample
        if case == "invalid_gather":
            assert sample["batch_ms"][0] == sample["ranks_us"][0] == float("inf")
        else:
            assert "batch_ms" not in sample and "ranks_us" not in sample
        partial["complete"] = False
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.json"
            save_setup_result(path, partial)
            restored = json.loads(path.read_text())
            saved_sample = restored["samples"][0]
            assert not restored["complete"] and not saved_sample["complete"]
            assert saved_sample["local_batch_ms"] == 0.3072
            assert saved_sample["local_us"] == 2.4
            if case == "invalid_gather":
                tags = [
                    {"invalid_nonfinite_float": value}
                    for value in ("inf", "-inf", "nan")
                ]
                assert (
                    saved_sample["batch_ms"][:3] == saved_sample["ranks_us"][:3] == tags
                )
                partial["complete"] = True
                try:
                    save_setup_result(path, partial)
                except ValueError:
                    pass
                else:
                    raise AssertionError("complete nonfinite result was saved")
            else:
                assert restored == partial
        failures.append(case)
    old = ast.parse((PRIVATE / "deps/task14/adapter.py").read_text())
    new = ast.parse(Path(__file__).read_text())

    def native_fields(tree):
        return next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            and any(
                isinstance(key, ast.Constant) and key.value == "pbm_enabled"
                for key in n.keys
            )
        )

    assert ast.dump(native_fields(old), include_attributes=False) == ast.dump(
        native_fields(
            next(
                n
                for n in new.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "native_diagnostic_graph"
            )
        ),
        include_attributes=False,
    )
    assert not torch.cuda.is_initialized()
    return {
        "controlled_cpu_only": True,
        "timed_observations": 20,
        "warmups": 20,
        "primed_event_pairs": 1,
        "fresh_event_pairs": 10,
        "noise_retained_without_repeat": True,
        "hard_failure_preserves_partial_and_stops": True,
        "F_retirement_before_next_warmup": True,
        "returned_local_timing_preserved_on_gather_failure": True,
        "partial_failure_json_write_readback": True,
        "file_logical_hardware_rank_join_checked": True,
        "native_constructor_fields_match_pinned_adapter": True,
        "refusals": failures,
        "event_order": "F/P,P/F,F/P,P/F,F/P; original within-pair alternation",
    }


def cache_marker_checks():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    failures = []
    functions = binding_functions()
    with (
        TemporaryDirectory() as directory,
        patch.dict(globals(), PRIVATE=Path(directory)),
    ):
        marker = PRIVATE / "cache/comgr/llvmcache.timestamp"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"")
        compiled = marker.with_name("llvmcache-compiled-control")
        compiled.write_bytes(b"compiled bytes control")
        binding = {
            "files": {
                str(p): functions["file_identity"](p) for p in (marker, compiled)
            },
            "models": {},
        }

        def touched():
            info = marker.stat()
            os.utime(marker, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))

        def refuses(name):
            try:
                functions["verify_binding"](binding)
            except (AssertionError, RuntimeError, FileNotFoundError):
                failures.append(name)
            else:
                raise AssertionError("cache identity change accepted: " + name)

        touched()
        functions["verify_binding"](binding)
        original_read = Path.read_bytes

        def read_and_touch(path):
            value = original_read(path)
            if path == marker:
                touched()
            return value

        with patch.object(Path, "read_bytes", read_and_touch):
            functions["verify_binding"](binding)
        compiled.write_bytes(b"changed compiled bytes")
        refuses("compiled_bytes")
        compiled.write_bytes(b"compiled bytes control")
        os.utime(
            compiled,
            ns=(
                compiled.stat().st_atime_ns,
                binding["files"][str(compiled)]["stat"][3] + 2_000_000_000,
            ),
        )
        refuses("compiled_mtime")
        binding["files"][str(compiled)] = functions["file_identity"](compiled)
        marker.write_bytes(b"not bookkeeping")
        refuses("marker_content")
        marker.write_bytes(b"")
        os.chmod(marker, 0o600)
        refuses("marker_mode")
        os.chmod(marker, 0o755)
        refuses("marker_executable")
        os.chmod(marker, 0o644)
        replacement = marker.with_name("replacement")
        replacement.write_bytes(b"")
        replacement.replace(marker)
        refuses("marker_inode")
        marker.unlink()
        refuses("missing_marker")
        marker.symlink_to(compiled)
        refuses("marker_symlink")
    return {
        "marker_mtime_and_concurrent_touch_pass": True,
        "refusals": failures,
        "actual_cache_mutated": False,
    }


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
    result["absolute_deadline"] = deadline_checks()
    result["recovered_input_admission"] = recovery_checks()
    result["cache_pruning_marker"] = cache_marker_checks()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", choices=("capture", "replay", "native_setup"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--check-native-setup", action="store_true")
    parser.add_argument("--qualify-preserved", action="store_true")
    parser.add_argument("--root-receipts", type=Path)
    parser.add_argument("--ownership-fixture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--write-recovery", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--validate-results", action="store_true")
    parser.add_argument("--rank", type=int, choices=range(8))
    args = parser.parse_args()
    if args.check_native_setup:
        print(json.dumps(setup_cpu_checks(), indent=2))
    elif args.selfcheck:
        print(json.dumps(cpu_checks(), indent=2))
    elif args.qualify_preserved:
        print(
            json.dumps(
                qualify_preserved(args.plan, args.root_receipts, args.out), indent=2
            )
        )
    elif args.write_recovery:
        assert not args.out.exists() and not args.out.resolve().is_relative_to(
            RECOVERY_RUN
        )
        save(args.out, recovery_record())
    elif args.ownership_fixture:
        ownership_fixture(args.ownership_fixture)
    elif args.write_plan:
        save(
            args.plan,
            make_plan(
                args.write_plan, args.out, args.plan, args.snapshot, args.recovery
            ),
        )
    else:
        assert args.plan is not None
        plan = json.loads(args.plan.read_text())
        if args.validate_results:
            assert (
                not args.worker
                and os.environ.get("KFD_GUARD_LOG")
                and os.environ.get("LD_PRELOAD")
            )
            check_plan(plan, args.plan)
            validate_results(plan)
        elif args.worker:
            check_plan(plan, args.plan, worker=True)
            assert args.rank is not None
            assert Path(
                os.readlink("/proc/self/fd/" + os.environ["K31_TASK19_GPU_LOCK_FD"])
            ).resolve() == Path("/app/agent-team-scratch/g24-gpu.lock")
            rank_worker(args.rank, plan)
        else:
            supervise(plan, args.plan)
