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
    env = environment(
        SimpleNamespace(cache=tmp_path),
        3,
        {
            "aiter_root": "/private/aiter",
            "native_jit_dir": "/private/jit",
            "native_cache_dir": "/private/cache",
        },
    )
    assert "SPEC_DECODE_ACCEPTANCE_LENGTH" not in env
    assert "AITER_AOT_IMPORT" not in env
    assert env["ATOM_ENABLE_RELAXED_MTP"] == "0"
    assert env["ATOM_SPEC_LOSSLESS_SAMPLING"] == "0"
    assert env["ATOM_MLA_ABSORB_BF16"] == "1"
