"""Exercise the experiment through ATOM's actual schema and HTTP handler."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from atom.entrypoints.openai import api_server
from spike.k31_draft_width.client import complete, payload, summarize


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(tokens=[11, 12], reason="max_tokens", calls=[])

    def preprocess(prompt, sampling, stream_callback=None, **kwargs):
        state.calls.append((prompt, sampling))
        stream_callback(
            SimpleNamespace(
                output_tokens=state.tokens,
                logprobs=None,
                finished=True,
                finish_reason=state.reason,
                num_cached_tokens=0,
            )
        )
        return SimpleNamespace(
            id=7,
            prompt_token_ids=list(prompt),
            num_prompt_tokens=len(prompt),
            max_tokens=sampling.max_tokens,
        )

    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=SimpleNamespace(add_request=lambda seqs: None),
    )
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setattr(api_server, "model_name", "width")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda *a: None
    )
    monkeypatch.setattr(api_server, "delivered_text", lambda ids: "x")
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)
    return state


def send(body):
    async def run():
        ledger = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=api_server.app, raise_app_exceptions=False
            ),
            base_url="http://test",
        ) as client:
            row = await complete(client, body, {"fixture": 0}, ledger)
        assert ledger == [row]
        return row

    return asyncio.run(run())


def test_real_callback_count_not_text_or_requested_cap(server):
    row = send(payload("width", [101, 102, 103, 104], limit=2, echo_ids=True))
    assert row["success"] and row["output_tokens"] == 2
    assert row["usage"]["prompt_tokens"] == 4
    assert row["confirmed_prompt_sha256"]
    prompt, params = server.calls[0]
    assert prompt == [101, 102, 103, 104]
    assert params.temperature == 0 and params.ignore_eos and params.max_tokens == 2
    assert row["actual_internal_batch"] is None


def test_short_http_success_preserves_count_but_is_not_usable(server):
    row = send(payload("width", [1, 2], limit=128))
    assert row["http_status"] == 200 and row["output_tokens"] == 2
    assert not row["success"]
    with pytest.raises(ValueError, match="incomplete request set"):
        summarize([row])


def test_aborted_200_stop_is_not_a_completed_sample(server):
    server.tokens, server.reason = [], "aborted"
    row = send(payload("width", [1, 2], limit=128))
    assert row["http_status"] == 200
    assert row["output_tokens"] == 0 and not row["success"]


@pytest.mark.parametrize(
    "wrong",
    [{"input_ids": [1, 2]}, {"prompt_token_ids": [-1]}, {"prompt_token_ids": [[1, 2]]}],
)
def test_real_http_schema_rejects_wrong_inputs(server, wrong):
    body = payload("width", [1, 2], limit=2)
    body.pop("prompt_token_ids")
    body.update(wrong)
    row = send(body)
    assert row["http_status"] in (400, 422)
    assert not row["success"] and not server.calls


def test_missing_usage_is_not_retokenized(server, monkeypatch):
    original = api_server.build_completion_response

    def malformed(*args, **kwargs):
        return original(*args, **kwargs).model_copy(
            update={"usage": {"prompt_tokens": 2}}
        )

    monkeypatch.setattr(api_server, "build_completion_response", malformed)
    row = send(payload("width", [1, 2], limit=2))
    assert row["http_status"] == 200 and row["output_tokens"] is None
    assert not row["success"]


def test_summary_uses_matching_wall_and_genuine_tokens():
    rows = [
        {
            "success": True,
            "output_tokens": 2,
            "wall_ns": 10_000_000,
            "start_ns": 100,
            "end_ns": 10_000_100,
            "usage": {"prompt_tokens": 5},
        },
        {
            "success": True,
            "output_tokens": 4,
            "wall_ns": 20_000_000,
            "start_ns": 100,
            "end_ns": 20_000_100,
            "usage": {"prompt_tokens": 7},
        },
    ]
    result = summarize(rows)
    assert result["request_wall_ms_per_token"] == 5
    assert result["output_tps"] == 300
    assert result["total_tps_per_gpu"] == 112.5
    assert result["official_result"] is False


def test_real_server_parser_keeps_natural_acceptance(monkeypatch):
    import sys
    from pathlib import Path

    from spike.k31_draft_width.run import command

    original = api_server.FlexibleArgumentParser.parse_args

    class Parsed(Exception):
        pass

    seen = []

    def stop_after_parse(parser, *args, **kwargs):
        parsed = original(parser, *args, **kwargs)
        seen.append(parsed)
        raise Parsed

    monkeypatch.setattr(
        api_server.FlexibleArgumentParser, "parse_args", stop_after_parse
    )
    args = SimpleNamespace(
        python="python", model="model", draft="draft", output=Path("test"), port=8654
    )
    for k in (3, 4):
        argv = command(args, k)
        monkeypatch.setattr(sys, "argv", ["server", *argv[3:]])
        with pytest.raises(Parsed):
            api_server.main()
        parsed = seen[-1]
        assert parsed.num_speculative_tokens == k
        assert parsed.spec_decode_acceptance_length is None
        assert parsed.spec_decode_acceptance_rate is None
        assert parsed.server_port == 8654


def test_child_environment_removes_inherited_synthetic_settings(monkeypatch, tmp_path):
    from spike.k31_draft_width.run import environment

    monkeypatch.setenv("SPEC_DECODE_ACCEPTANCE_LENGTH", "3.19")
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", "1")
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    monkeypatch.setenv("AITER_AOT_IMPORT", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    monkeypatch.setenv("ATOM_MONO_DEBUG", "1")
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "1")
    monkeypatch.setenv("OPUS_GEN_CO_DIR", "/other/source")
    monkeypatch.setenv("AITER_META_DIR", "/other/source")
    env = environment(
        SimpleNamespace(cache=tmp_path, output=tmp_path / "run-test"),
        3,
        {
            "aiter_root": "/private/aiter",
            "native_jit_dir": "/private/jit",
            "native_cache_dir": "/private/cache",
        },
    )
    assert "SPEC_DECODE_ACCEPTANCE_LENGTH" not in env
    assert "AITER_AOT_IMPORT" not in env
    assert "FLYDSL_RUNTIME_RUN_ONLY" not in env
    assert "ATOM_MONO_DEBUG" not in env
    assert "FLYDSL_KTRACE_ENABLE" not in env
    assert env["AITER_ASM_DIR"] == "/private/aiter/hsa"
    assert env["AITER_META_DIR"] == "/private/aiter"
    assert env["OPUS_GEN_CO_DIR"] == "/private/aiter/csrc/opus_gemm/gen_co"
    assert env["ATOM_MONO_POLL_SLEEP_UNITS"] == "16"
    assert env["ATOM_ENABLE_RELAXED_MTP"] == "0"
    assert env["ATOM_SPEC_LOSSLESS_SAMPLING"] == "0"
    assert env["ATOM_MLA_ABSORB_BF16"] == "1"


@pytest.fixture
def good_verify(monkeypatch, tmp_path):
    import json

    from spike.k31_draft_width import client, run

    model, draft = tmp_path / "model", tmp_path / "draft"
    for path in (model, draft):
        path.mkdir()
        (path / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    definition = {
        "source_sha256": "source",
        "requests": [
            {
                "context": 1,
                "offset": 0,
                "length": 2,
                "sha256": client.token_hash([1, 2]),
            }
        ],
    }
    (tmp_path / "fixtures.json").write_text(json.dumps(definition))
    monkeypatch.setattr(client, "HERE", tmp_path)
    inputs = {
        "model": str(model),
        "source_sha256": "source",
        "definition_sha256": client.sha256(tmp_path / "fixtures.json"),
        "requests": [{"fixture": 0, **definition["requests"][0], "token_ids": [1, 2]}],
    }
    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps(inputs))
    pins = {
        "target": {
            "path": str(model),
            "config_sha256": client.sha256(model / "config.json"),
            "index_sha256": client.sha256(model / "model.safetensors.index.json"),
        },
        "draft": {
            "path": str(draft),
            "config_sha256": client.sha256(draft / "config.json"),
        },
    }
    pins_file = tmp_path / "pins.json"
    pins_file.write_text(json.dumps(pins))
    cache_file = tmp_path / "cache.json"
    binding = tmp_path / "atom" / "binding"
    binding.parent.mkdir()
    binding.write_text("same source")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    selection = {
        "aiter_root": str(tmp_path),
        "native_jit_dir": str(tmp_path),
        "native_cache_dir": str(tmp_path),
    }
    native_file = tmp_path / "native.json"
    native_file.write_text(
        json.dumps(
            {
                "native_cpu_ready": True,
                "identity": {
                    "selection": selection,
                    "source": {
                        "source_root": str(tmp_path),
                        "aiter_commit": "head",
                        "ck_root": str(tmp_path),
                        "ck_commit": "head",
                        "source_files_sha256": {"atom/binding": client.sha256(binding)},
                    },
                },
                "artifacts": {str(binding): client.sha256(binding)},
                "compiled_inputs": {str(binding): client.sha256(binding)},
            }
        )
    )
    artifacts = {}
    for k in (3, 4):
        artifact = tmp_path / f"k{k}" / "flydsl" / "key"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("cached")
        artifacts[str(artifact)] = client.sha256(artifact)
    settings = {
        "max_model_len": 131072,
        "block_size": 64,
        "replay_cap": 16,
        "tp": 8,
        "dcp": 8,
        "poll_sleep_units": 16,
        "debug": False,
        "timeline": False,
        "arms": {
            "k3": {
                "qlen": 4,
                "rows": [4, 8, 12, 16],
                "state_len": 6,
                "replay_window": 4,
            },
            "k4": {
                "qlen": 5,
                "rows": [5, 10, 15, 20],
                "state_len": 7,
                "replay_window": 5,
            },
        },
    }
    cache_file.write_text(
        json.dumps(
            {
                "ready": True,
                "atom_source": "base",
                "settings": settings,
                "aiter_selection": selection,
                "artifacts": artifacts,
                "identity": {
                    "aiter_selection": selection,
                    "settings": settings,
                    "production_sha256": {"atom/binding": client.sha256(binding)},
                    "model_config": {
                        "path": str(model / "config.json"),
                        "sha256": client.sha256(model / "config.json"),
                    },
                    "native_receipt": {
                        "path": str(native_file),
                        "sha256": client.sha256(native_file),
                    },
                },
            }
        )
    )
    asm_file = tmp_path / "hsa" / "kernel.co"
    asm_file.parent.mkdir()
    asm_file.write_bytes(b"ELF-asset")
    assets_file = tmp_path / "assets.json"
    assets_file.write_text(
        json.dumps(
            {
                "ready": True,
                "aiter_root": str(tmp_path),
                "asm_dir": str(tmp_path / "hsa"),
                "opus_gen_co_dir": str(tmp_path / "csrc/opus_gemm/gen_co"),
                "artifacts": {str(asm_file): client.sha256(asm_file)},
            }
        )
    )
    args = SimpleNamespace(
        assets_receipt=assets_file,
        assets_sha=client.sha256(assets_file),
        repeats=2,
        startup_timeout=1,
        run_timeout=2,
        head="head",
        base="base",
        model=str(model),
        draft=str(draft),
        inputs=inputs_file,
        inputs_sha=client.sha256(inputs_file),
        model_pins=pins_file,
        model_pins_sha=client.sha256(pins_file),
        cache=tmp_path,
        cache_receipt=cache_file,
        cache_sha=client.sha256(cache_file),
    )

    def git(argv, **kwargs):
        if "rev-parse" in argv:
            return "head\n"
        if "status" in argv:
            return ""
        return "spike/k31_draft_width/client.py\n"

    monkeypatch.setattr(run.subprocess, "check_output", git)
    assert run.verify(args)[0] == inputs
    return args, inputs


def test_run_verify_rejects_changed_tokens_after_good_receipt(good_verify):
    import json

    from spike.k31_draft_width import client, run

    args, inputs = good_verify
    inputs["requests"][0]["token_ids"][0] = 99
    args.inputs.write_text(json.dumps(inputs))
    args.inputs_sha = client.sha256(args.inputs)
    with pytest.raises(ValueError, match="frozen prompt tokens"):
        run.verify(args)


def test_run_verify_rejects_missing_fixture_after_good_receipt(good_verify):
    import json

    from spike.k31_draft_width import client, run

    args, inputs = good_verify
    inputs["requests"] = []
    args.inputs.write_text(json.dumps(inputs))
    args.inputs_sha = client.sha256(args.inputs)
    with pytest.raises(ValueError, match="frozen request set"):
        run.verify(args)


@pytest.mark.parametrize("repeats", [0, 1, -1])
def test_run_rejects_no_spread_before_other_work(good_verify, repeats):
    from spike.k31_draft_width import run

    args, _ = good_verify
    args.repeats = repeats
    with pytest.raises(ValueError, match="at least two"):
        run.verify(args)


def test_reused_pgid_is_not_owned(monkeypatch):
    from spike.k31_draft_width import run

    known = {12345: "original"}
    monkeypatch.setattr(
        run,
        "process_table",
        lambda: {12345: ("new", 1, 12345), 12346: ("foreign", 12345, 12345)},
    )
    assert run.remember(SimpleNamespace(pid=12345), known) == []
    assert 12346 not in known


def test_original_group_member_proves_orphan_ownership(monkeypatch):
    from spike.k31_draft_width import run

    known = {12345: "parent", 12346: "original-child"}
    monkeypatch.setattr(
        run,
        "process_table",
        lambda: {12346: ("original-child", 1, 12345), 12347: ("new-child", 1, 12345)},
    )
    assert run.remember(SimpleNamespace(pid=12345), known) == [12346, 12347]


def test_new_owned_child_reusing_pid_gets_new_birth(monkeypatch):
    from spike.k31_draft_width import run

    known = {12345: "parent", 12346: "old-child"}
    monkeypatch.setattr(
        run,
        "process_table",
        lambda: {12345: ("parent", 1, 12345), 12346: ("new-child", 12345, 12345)},
    )
    assert run.remember(SimpleNamespace(pid=12345), known) == [12345, 12346]
    assert known[12346] == "new-child"


@pytest.mark.parametrize("changed", ["atom/binding", "k3/flydsl/key", "native.json"])
def test_verify_rejects_changed_bound_artifact(good_verify, changed):
    from spike.k31_draft_width import run

    args, _ = good_verify
    (args.cache / changed).write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        run.verify(args)


def test_verify_rejects_unprepared_cache_directory(good_verify):
    from spike.k31_draft_width import run

    args, _ = good_verify
    args.cache = args.cache / "other-cache"
    with pytest.raises(RuntimeError, match="no prepared arm"):
        run.verify(args)


@pytest.mark.parametrize("failure", ["telemetry", "evidence", "residency"])
def test_cleanup_waits_for_confirmed_release(monkeypatch, tmp_path, failure):
    from spike.k31_draft_width import run

    records, sleeps = [], []
    memory_calls = 0

    def memory():
        nonlocal memory_calls
        memory_calls += 1
        if memory_calls == 1 and failure == "telemetry":
            raise OSError("unavailable")
        return {"gpu": 3 * 2**30 if memory_calls == 1 and failure == "residency" else 0}

    def save(path, result):
        records.append(result)
        if len(records) == 1 and failure == "evidence":
            raise OSError("full")

    monkeypatch.setattr(run, "remember", lambda *a: [])
    monkeypatch.setattr(run, "vram", memory)
    monkeypatch.setattr(run, "save", save)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(run.signal, "signal", lambda *a: run.signal.SIG_DFL)
    run.cleanup(SimpleNamespace(pid=12345), {}, tmp_path)
    assert len(records) == 2 and records[-1]["released"]
    assert sleeps == [5]


def test_cleanup_rechecks_birth_before_signal(monkeypatch, tmp_path):
    from spike.k31_draft_width import run

    calls = 0

    def remember(*args):
        nonlocal calls
        calls += 1
        return [12345] if calls == 1 else []

    killed = []
    monkeypatch.setattr(run, "remember", remember)
    monkeypatch.setattr(run, "birth", lambda pid: "reused")
    monkeypatch.setattr(run.os, "kill", lambda *args: killed.append(args))
    monkeypatch.setattr(run, "vram", lambda: {"gpu": 0})
    monkeypatch.setattr(run.signal, "signal", lambda *a: run.signal.SIG_DFL)
    run.cleanup(SimpleNamespace(pid=12345), {12345: "original"}, tmp_path)
    assert not killed


def test_verify_rejects_changed_assembly_selection(good_verify):
    import json

    from spike.k31_draft_width import client, run

    args, _ = good_verify
    assets = json.loads(args.assets_receipt.read_text())
    assets["asm_dir"] += "/gfx950"
    args.assets_receipt.write_text(json.dumps(assets))
    args.assets_sha = client.sha256(args.assets_receipt)
    with pytest.raises(RuntimeError, match="assembly asset selection differs"):
        run.verify(args)


def test_verify_rejects_changed_assembly_bytes(good_verify):
    from spike.k31_draft_width import run

    args, _ = good_verify
    (args.cache / "hsa/kernel.co").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="assembly asset changed"):
        run.verify(args)


@pytest.mark.parametrize("k", [3, 4])
def test_real_cache_resolver_uses_private_child_home(monkeypatch, tmp_path, k):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from spike.k31_draft_width.run import environment

    polluted = tmp_path / "inherited-shared"
    for key in (
        "HOME",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "ATOM_CACHE_ROOT",
    ):
        monkeypatch.setenv(key, str(polluted))
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "32")
    args = SimpleNamespace(
        cache=tmp_path / "private-cache", output=tmp_path / "run-one"
    )
    env = environment(
        args,
        k,
        {
            "aiter_root": str(tmp_path / "aiter"),
            "native_jit_dir": str(tmp_path / "native/jit"),
            "native_cache_dir": str(tmp_path / "native"),
        },
    )
    code = """import json
from atom.utils.backends import VLLM_CACHE_ROOT
from torch._inductor.config import compile_threads
print(json.dumps({"cache_root": VLLM_CACHE_ROOT, "compile_threads": compile_threads}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    expected = args.cache / f"k{k}" / "runs/run-one/home/.cache/atom"
    assert Path(observed["cache_root"]) == expected
    assert observed["compile_threads"] == 1
    assert os.environ["HOME"] == str(polluted)
    assert not polluted.exists()


@pytest.mark.parametrize("write_request", [1, 5])
def test_startup_cache_writes_are_recorded_and_timed_writes_rejected(
    good_verify, tmp_path, write_request
):
    import json

    from spike.k31_draft_width.client import measure

    _, inputs = good_verify
    cache = tmp_path / "runtime-jit"
    cache.mkdir()
    output = tmp_path / "measurement"
    requests = 0

    def respond(request):
        nonlocal requests
        if request.url.path.startswith("/debug/"):
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        requests += 1
        if requests == write_request:
            (cache / "new_kernel.so").write_bytes(b"compiled")
        return httpx.Response(
            200,
            json={
                "id": "test",
                "prompt_token_ids": body["prompt_token_ids"],
                "usage": {
                    "prompt_tokens": len(body["prompt_token_ids"]),
                    "completion_tokens": body["max_tokens"],
                },
                "choices": [{"finish_reason": "length"}],
            },
        )

    async def experiment():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://test"
        ) as client:
            return await measure(
                "http://test",
                "test",
                inputs,
                3,
                2,
                output,
                client=client,
                cache_roots=[cache],
                initial_cache={},
            )

    if write_request == 5:
        with pytest.raises(RuntimeError, match="compiler cache changed"):
            asyncio.run(experiment())
        assert not (output / "summary.json").exists()
        assert json.loads((output / "timed-cache-changes.json").read_text())
    else:
        asyncio.run(experiment())
        startup = json.loads((output / "startup.json").read_text())
        assert str(cache / "new_kernel.so") in startup["untimed_cache_changes"]
        assert json.loads((output / "summary.json").read_text())["complete"]


def test_startup_failure_keeps_before_and_after_cache_evidence(
    good_verify, monkeypatch, tmp_path
):
    import json

    from spike.k31_draft_width import run

    args, inputs = good_verify
    args.cache, args.output = tmp_path / "runtime", tmp_path / "failed-run"
    args.python, args.port = "unused-python", 0
    args.pause, args.lock = tmp_path / "pause", str(tmp_path / "gpu.lock")
    native = tmp_path / "native"
    cache = {
        "identity_fingerprint": "test",
        "aiter_selection": {
            "aiter_root": str(tmp_path / "aiter"),
            "native_cache_dir": str(native),
            "native_jit_dir": str(native),
        },
    }
    monkeypatch.setenv("K31_SPIKE_GPU_GRANTED", "1")
    monkeypatch.delenv("KFD_GUARD_LOG", raising=False)
    monkeypatch.setattr(run.os, "getpriority", lambda *args: 0)
    monkeypatch.setattr(run, "verify", lambda args: (inputs, cache))
    monkeypatch.setattr(run, "vram", lambda: {str(i): 0 for i in range(8)})
    # No process with this out-of-range PID exists; cleanup must own no real PID.
    monkeypatch.setattr(
        run.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(pid=2**30, poll=lambda: None),
    )

    async def fail_startup(*args):
        assert (tmp_path / "failed-run/k4/startup-cache-before.json").exists()
        native.mkdir()
        (native / "new_kernel.so").write_bytes(b"new")
        raise RuntimeError("startup failure")

    monkeypatch.setattr(run, "wait_ready", fail_startup)
    with pytest.raises(RuntimeError, match="startup failure"):
        run.run(args)
    arm = tmp_path / "failed-run/k4"
    after = json.loads((arm / "cache-after-cleanup.json").read_text())
    assert not after["reached_http_ready"] and not after["reached_timed_phase"]
    assert after["changes"][str(native / "new_kernel.so")]["before"] is None
    assert json.loads((arm / "cleanup.json").read_text())["released"]
    assert not (tmp_path / "failed-run/k3").exists()
