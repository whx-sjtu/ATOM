"""CPU contracts for real consumers, complete evidence and owned cleanup."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import run
import verify


def test_actual_consumers_and_offline_data(tmp_path):
    code = r"""
import asyncio, copy, json, os, sys
from unittest.mock import patch
from lm_eval._cli import HarnessCLI
from lm_eval.models.openai_completions import LocalCompletionsAPI
from atom.entrypoints.openai.protocol import CompletionRequest
import run, verify
attempt = os.environ['K31_TASK10_CPU_ATTEMPT']
commands = run.commands(attempt)
sys.argv = commands['client'][:]
args = HarnessCLI().parse_args()
assert args.seed == [0, 1234, 1234, 1234] and args.limit is None
assert args.num_fewshot == 5 and args.tasks == ['gsm8k']
with patch('lm_eval.utils.check_remote_tokenizer_support', return_value=False):
    model = LocalCompletionsAPI.create_from_arg_obj(args.model_args)
assert (model.timeout, model.max_retries, model._concurrent, model._max_gen_toks) == (300, 3, 8, 256)
expected = json.loads((run.PRIVATE / 'preparation' / f'{attempt}-data.json').read_text())
assert verify.prepare_data() == expected
payload = model._create_payload('test', generate=True, gen_kwargs=copy.deepcopy(expected['config']['generation_kwargs']))
request = CompletionRequest(**payload)
assert request.get_max_tokens() == 256 and request.temperature == 0 and not request.ignore_eos
assert request.stop == ['Question:', '</s>', '<|im_end|>']
policy = run.parsed_server_args(commands['server'])
assert policy.spec_decode_acceptance_length is None and policy.spec_decode_acceptance_rate is None
assert policy.tensor_parallel_size == 8 and policy.decode_context_parallel_size == 8
assert policy.cudagraph_mode == 'FULL' and policy.max_num_seqs == 24
from aiohttp import ClientResponseError, web
async def exercise():
    seen = []
    async def handle(request):
        seen.append(await request.json())
        return web.json_response({'choices':[{'index':0,'text':''}]}, status=200 if len(seen) == 3 else 500)
    app = web.Application()
    app.router.add_post('/v1/completions', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    model.base_url = f'http://127.0.0.1:{port}/v1/completions'
    try:
        result = await model.get_batched_requests(['test'], [None], gen_kwargs=expected['config']['generation_kwargs'])
        assert result == [['']] and len(seen) == 3
        assert all(item['max_tokens'] == 256 and item['temperature'] == 0 for item in seen)
        try:
            await model.get_batched_requests(['test'], [None], gen_kwargs=expected['config']['generation_kwargs'])
        except ClientResponseError:
            assert len(seen) == 6
        else:
            raise AssertionError('terminal API failure was swallowed')
    finally:
        await runner.cleanup()
asyncio.run(exercise())
print('ACTUAL_CONSUMERS_PASS')
"""
    attempt = os.environ.get("K31_TASK10_CPU_ATTEMPT", "author-001")
    env = dict(run.environment(attempt), K31_TASK10_CPU_ATTEMPT=attempt)
    env.update(
        {k: os.environ[k] for k in ("LD_PRELOAD", "KFD_GUARD_LOG") if k in os.environ}
    )
    env["PYTHONPATH"] = f"{run.HERE}:{env['PYTHONPATH']}"
    result = subprocess.run(
        [run.P["python"], "-B", "-c", code],
        cwd=run.ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    (tmp_path / "consumers.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTUAL_CONSUMERS_PASS" in result.stdout


@pytest.fixture
def evidence(tmp_path):
    attempt = os.environ.get("K31_TASK10_CPU_ATTEMPT", "author-001")
    expected = json.loads(
        (run.PRIVATE / "preparation" / f"{attempt}-data.json").read_text()
    )
    expected["docs"] = {}
    rows = []
    for index in range(run.P["samples"]):
        doc = {"question": str(index), "answer": "#### 1"}
        prompt = "Question: " + str(index) + "\nAnswer:"
        identity = {
            "doc_hash": hashlib.sha256(
                json.dumps(doc, indent=2, ensure_ascii=False).encode()
            ).hexdigest(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "target_hash": hashlib.sha256(doc["answer"].encode()).hexdigest(),
        }
        expected["docs"][str(index)] = identity
        for filt in run.P["filters"]:
            rows.append(
                dict(
                    identity,
                    doc_id=index,
                    filter=filt,
                    doc=doc,
                    target=doc["answer"],
                    arguments={
                        "gen_args_0": {
                            "arg_0": prompt,
                            "arg_1": expected["config"]["generation_kwargs"],
                        }
                    },
                    resps=[[""]],
                    filtered_resps=["[invalid]"],
                    exact_match=0,
                )
            )
    result = {
        "configs": {"gsm8k": expected["config"]},
        "versions": {"gsm8k": 3.0},
        "n-samples": {"gsm8k": {"effective": 1319}},
        "results": {"gsm8k": {"exact_match," + f: 0 for f in run.P["filters"]}},
    }

    def seal():
        run.save(tmp_path / "results.json", result)
        (tmp_path / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return {
            "output": str(tmp_path),
            "result": {
                "path": str(tmp_path / "results.json"),
                "sha256": verify.sha(tmp_path / "results.json"),
            },
            "samples": {
                "path": str(tmp_path / "samples.jsonl"),
                "sha256": verify.sha(tmp_path / "samples.jsonl"),
            },
            "head": "fixed",
            "profile_sha256": "fixed",
            "data_sha256": "fixed",
        }

    return expected, rows, result, seal


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "document", "prompt", "identity", "task"]
)
def test_bad_evidence_rejected(evidence, fault):
    expected, rows, result, seal = evidence
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "document":
        rows[0]["doc"]["question"] = "different"
    elif fault == "prompt":
        rows[0]["arguments"]["gen_args_0"]["arg_0"] = "different"
    elif fault == "identity":
        rows[0]["doc_hash"] = "different"
    else:
        result["versions"]["gsm8k"] = 4
    with pytest.raises(ValueError):
        verify.validate(seal(), expected)


def test_low_score_is_complete_and_first_valid_is_sticky(evidence, tmp_path):
    expected, _, _, seal = evidence
    manifest = seal()
    _, score = verify.validate(manifest, expected)
    assert (
        score["quality_floor"] == "FAIL" and score["correct"]["flexible-extract"] == 0
    )
    registry = []
    for index, outcome in enumerate(("EXECUTION_INVALID", "COMPLETE", "COMPLETE"), 1):
        status = tmp_path / f"status-{index}.json"
        run.save(
            status,
            {
                "plan_sha256": str(index),
                "cleanup": {"released": True},
                "outcome": outcome,
                "evidence": manifest,
            },
        )
        registry.append(
            {
                "attempt": f"author-{index:03d}",
                "status": str(status),
                "status_sha256": verify.sha(status),
                "plan_sha256": str(index),
            }
        )
    assert verify.first_valid(registry, expected)["attempt"] == "author-002"
    qualified = verify.qualify(manifest, manifest, expected)
    assert qualified["paired"]["paired_questions"] == 1319
    assert qualified["paired"]["by_filter"]["flexible-extract"]["mcnemar_p"] == 1
    changed = dict(manifest, head="different")
    with pytest.raises(ValueError, match="source/profile/data"):
        verify.qualify(manifest, changed, expected)


@pytest.mark.parametrize("fault", ["status_write", "missing_cards", "owned_kfd"])
def test_cleanup_keeps_lock_until_durable_release(monkeypatch, tmp_path, fault):
    state = {"attempt": "independent-001", "outcome": "COMPLETE"}
    closed, checks = [], []
    real_save = run.save

    def save(path, value):
        if fault == "status_write" and len(checks) == 1:
            raise OSError("injected disk error")
        real_save(path, value)

    def release(**kwargs):
        assert not closed
        checks.append(1)
        assert kwargs["timeout"] > 0
        cards = (
            {}
            if fault == "missing_cards" and len(checks) == 1
            else {str(i): {"used_gib": 0} for i in range(8)}
        )
        return True, {
            "cards": cards,
            "kfd_pids": [42] if fault == "owned_kfd" and len(checks) == 1 else [],
        }

    monkeypatch.setattr(run, "save", save)
    real_close = os.close
    monkeypatch.setattr(
        run.os, "close", lambda fd: closed.append(fd) if fd == 99 else real_close(fd)
    )
    monkeypatch.setattr(run.signal, "signal", lambda *args: None)
    monkeypatch.setattr(run.signal, "setitimer", lambda *args: None)
    ticks = iter([0.0, 65.0])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks, 65.0))

    def sleep(seconds):
        assert len(checks) < 3

    monkeypatch.setattr(run.time, "sleep", sleep)
    run.finish(
        SimpleNamespace(stop=lambda: None, live=list, seen={42: 1}),
        SimpleNamespace(wait_gpus_released=release),
        99,
        tmp_path,
        state,
    )
    assert len(checks) == 2 and closed == [99]
    assert json.loads((tmp_path / "controller.json").read_text())["cleanup"]["released"]
    assert (state["outcome"] == "EXECUTION_INVALID") == (fault == "status_write")


def test_owned_pid_birth_cancel_and_early_exit(tmp_path):
    common = verify.dependency("common.py")
    owned = run.owned_type()(common)

    def spawn(command, **kwargs):
        kwargs["cpus"] = "112-127"
        return common.spawn(command, **kwargs)

    resources = SimpleNamespace(spawn=spawn)
    child = run.launch(
        owned,
        resources,
        [sys.executable, "-c", "import time;time.sleep(60)"],
        os.environ.copy(),
        tmp_path,
        "child",
        None,
    )
    born = owned.seen[child.pid]
    try:
        owned.seen[child.pid] = born - 1
        assert not owned.live()
        owned.stop()
        assert child.poll() is None
        owned.seen[child.pid] = born
        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt("cancellation")
            finally:
                owned.stop()
        assert child.wait(timeout=1) != 0 and not owned.live()
        with pytest.raises(RuntimeError, match="exited early"):
            run.wait_until(owned, lambda: True, time.monotonic() + 1)
    finally:
        owned.seen[child.pid] = born
        owned.stop()
