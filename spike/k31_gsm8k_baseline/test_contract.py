"""CPU contracts for real consumers, complete evidence and owned cleanup."""

import copy
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
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
    monkeypatch.setattr(run, "cache_inventory", lambda attempt: {})
    state = {"attempt": "independent-001", "outcome": "COMPLETE"}
    closed, checks = [], []
    clock = {"now": 0.0}
    real_save = run.save

    def save(path, value):
        if fault == "status_write" and len(checks) == 1:
            raise OSError("injected disk error")
        real_save(path, value)

    def release(**kwargs):
        assert not closed
        checks.append(1)
        clock["now"] = 65.0
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
    monkeypatch.setattr(run.time, "monotonic", lambda: clock["now"])

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


def test_owned_pid_birth_cancel_and_early_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "cache_inventory", lambda attempt: {})
    common = verify.dependency("common.py")
    owned = run.owned_type()(common)

    def spawn(command, **kwargs):
        kwargs["cpus"] = "112-127"
        return common.spawn(command, **kwargs)

    resources = SimpleNamespace(spawn=spawn)
    fd = common.GpuLock(path=str(tmp_path / "cpu.lock")).acquire()
    release = SimpleNamespace(
        wait_gpus_released=lambda **kwargs: (
            True,
            {"cards": {str(i): {"used_gib": 0} for i in range(8)}, "kfd_pids": []},
        )
    )
    monkeypatch.setattr(run.signal, "signal", lambda *args: None)
    monkeypatch.setattr(run.signal, "setitimer", lambda *args: None)
    child = run.launch(
        owned,
        resources,
        [sys.executable, "-c", "import time;time.sleep(60)"],
        os.environ.copy(),
        tmp_path,
        "child",
        fd,
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
                run.finish(
                    owned,
                    release,
                    fd,
                    tmp_path,
                    {"attempt": "independent-001", "outcome": "EXECUTION_INVALID"},
                )
        assert child.wait(timeout=1) != 0 and not owned.live()
        assert json.loads((tmp_path / "controller.json").read_text())["cleanup"][
            "released"
        ]
        with pytest.raises(OSError):
            os.fstat(fd)
        with pytest.raises(RuntimeError, match="exited early"):
            run.wait_until(owned, lambda: True, time.monotonic() + 1)
    finally:
        owned.seen[child.pid] = born
        owned.stop()


def test_blocked_preflight_is_interrupted(monkeypatch):
    completed = []
    monkeypatch.setitem(run.P["bounds"], "startup", 0.03)

    def blocked(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        time.sleep(1)
        completed.append(True)

    monkeypatch.setattr(run, "load_plan", blocked)
    with pytest.raises(KeyboardInterrupt):
        run.execute("never-read-cpu-fixture")
    assert not completed and signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize("blocked", ["query", "receipt"])
def test_release_io_timeout_preserves_lock_until_retry(tmp_path, monkeypatch, blocked):
    handlers = {sig: signal.getsignal(sig) for sig in run.STOP}
    fd = os.open(tmp_path / "release-lock", os.O_CREAT | os.O_RDWR, 0o600)
    query_calls, receipt_calls, inventoried = [], [], []
    real_sleep, real_save = time.sleep, run.save
    monkeypatch.setitem(run.P["bounds"], "release", 0.03)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)

    def release(**kwargs):
        os.fstat(fd)
        query_calls.append(1)
        if blocked == "query" and len(query_calls) == 1:
            real_sleep(1)
        return True, {
            "cards": {str(i): {"used_gib": 0} for i in range(8)},
            "kfd_pids": [],
        }

    def save(path, state):
        if path.name == "release.json":
            receipt_calls.append(1)
            if blocked == "receipt" and len(receipt_calls) == 1:
                real_sleep(1)
        real_save(path, state)

    def inventory(attempt):
        with pytest.raises(OSError):
            os.fstat(fd)
        assert json.loads((tmp_path / "release.json").read_text())["cleanup"][
            "released"
        ]
        inventoried.append(True)
        return {}

    monkeypatch.setattr(run, "save", save)
    monkeypatch.setattr(run, "cache_inventory", inventory)
    state = {"attempt": "independent-001", "outcome": "COMPLETE"}
    try:
        run.finish(
            SimpleNamespace(stop=lambda: None, live=list, seen={}),
            SimpleNamespace(wait_gpus_released=release),
            fd,
            tmp_path,
            state,
        )
        assert len(query_calls) == 2 and inventoried == [True]
        assert state["outcome"] == "EXECUTION_INVALID"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.mark.parametrize(
    "failure", ["registry", "registry_directory", "marker_directory"]
)
def test_registry_failure_is_pending_and_recovers_same_low_attempt(
    evidence, tmp_path, monkeypatch, failure
):
    expected, _, _, seal = evidence
    original = seal()
    monkeypatch.setattr(run, "PRIVATE", tmp_path)
    monkeypatch.setattr(run, "cache_inventory", lambda attempt: {})
    monkeypatch.setattr(
        run, "git", lambda *args: "fixture-head" if args[0] == "rev-parse" else ""
    )
    out = tmp_path / "runs/author-001"
    quality = out / "quality"
    quality.mkdir(parents=True)
    (quality / "results_fixture.json").write_bytes(
        Path(original["result"]["path"]).read_bytes()
    )
    (quality / "samples_gsm8k_fixture.jsonl").write_bytes(
        Path(original["samples"]["path"]).read_bytes()
    )
    run.save(tmp_path / "data.json", expected)
    plan = {
        "attempt": "author-001",
        "output": str(out),
        "head": "fixture-head",
        "protocol": run.P,
        "data": {
            "path": str(tmp_path / "data.json"),
            "sha256": verify.sha(tmp_path / "data.json"),
        },
    }
    run.save(out / "plan.json", plan)
    run.save(out / "cache-before.json", {})
    state = {
        "attempt": "author-001",
        "head": "fixture-head",
        "plan_sha256": verify.sha(out / "plan.json"),
        "outcome": "COMPLETE",
        "registered": True,
        "client_rc": 0,
        "evidence": verify.bind_results(out, expected),
        "cache_before_file": {
            "path": str(out / "cache-before.json"),
            "sha256": verify.sha(out / "cache-before.json"),
        },
        "cleanup": {"released": True},
    }
    run.save(
        out / "release.json",
        {
            "head": state["head"],
            "plan_sha256": state["plan_sha256"],
            "cleanup": state["cleanup"],
            "execution": state.copy(),
        },
    )
    registry = [
        {
            "attempt": "author-001",
            "status": str(out / "controller.json"),
            "plan_sha256": state["plan_sha256"],
        }
    ]
    run.save(tmp_path / "attempts.json", registry)
    real_seal = run.seal_registry

    def fail_seal(*args):
        assert json.loads((out / "release.json").read_text())["cleanup"]["released"]
        raise OSError("injected post-release registry failure")

    real_fsync = os.fsync
    marker = out / "pending-verification.json"

    def fail_sync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            if failure == "marker_directory" and not marker.exists():
                raise OSError("marker cleanup fsync failure")
            if (
                failure == "registry_directory"
                and info.st_ino == tmp_path.stat().st_ino
                and "status_sha256"
                in json.loads((tmp_path / "attempts.json").read_text())[0]
            ):
                raise OSError("critical registry fsync failure")
        return real_fsync(fd)

    if failure == "registry":
        monkeypatch.setattr(run, "seal_registry", fail_seal)
    else:
        monkeypatch.setattr(run.os, "fsync", fail_sync)
    handlers = {sig: signal.getsignal(sig) for sig in run.STOP}
    try:
        with pytest.raises(OSError):
            run.finalize_evidence(out, state, time.monotonic() + 10)
        actual_registry = json.loads((tmp_path / "attempts.json").read_text())
        if failure == "marker_directory":
            assert state["outcome"] == "COMPLETE"
            assert (
                verify.first_valid(actual_registry, expected)["attempt"] == "author-001"
            )
        else:
            assert state["outcome"] == "PENDING_VERIFICATION"
            with pytest.raises(ValueError, match="pending verification"):
                verify.first_valid(actual_registry, expected)
        monkeypatch.setattr(run, "seal_registry", real_seal)
        monkeypatch.setattr(run.os, "fsync", real_fsync)
        result = run.recover_evidence(out)
        assert result == {
            "attempt": "author-001",
            "outcome": "COMPLETE",
            "gpu_work": False,
        }
        chosen = verify.first_valid(
            json.loads((tmp_path / "attempts.json").read_text()), expected
        )
        assert (
            chosen["attempt"] == "author-001"
            and chosen["score"]["quality_floor"] == "FAIL"
        )
        assert chosen["score"]["correct"]["flexible-extract"] == 0
        preserved = {
            name: verify.sha(out / name) for name in ("controller.json", "release.json")
        }
        assert run.recover_evidence(out) == result
        assert preserved == {name: verify.sha(out / name) for name in preserved}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def test_pinned_startup_helper_accepts_stock_and_records_all_refusals(tmp_path):
    code = r"""
import importlib.util, json, sys, types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import run
out = Path(sys.argv[1])
policy = run.P['startup_module']
original = sys.modules['sitecustomize']
real_sha = run.sha
accepted = run.startup_check(out/'allowed.json')
assert accepted['qualified'] and not accepted['target_imports_present']
assert accepted['source_sha256'] == policy['source_sha256']
assert accepted['bytecode_sha256'] == policy['bytecode_sha256']
assert accepted['resolvable'] == {'apport_python_hook':None, 'usercustomize':None}
legacy = out/'legacy.py'
legacy.write_bytes(Path(policy['origin']).read_bytes())
replacement = types.ModuleType('sitecustomize')
replacement.__file__ = str(legacy)
replacement.__cached__ = str(legacy.with_suffix('.pyc'))
replacement.__spec__ = importlib.util.spec_from_file_location('sitecustomize', legacy)
for name in ('apport_python_hook','usercustomize'):
    directory = out/name
    directory.mkdir()
    (directory/(name+'.py')).write_text("raise AssertionError('negative fixture must never execute')\n")
class UnknownFinder:
    def find_spec(self, *args):
        raise AssertionError('unqualified finder must not run')
cases = ('source_bytes','bytecode_bytes','wrong_origin','legacy_hook','missing_hook',
         'user_loaded','apport_loaded','apport_reachable','user_reachable',
         'exception_hook','trace_hook','profile_hook','dependency_bytes','unknown_finder')
for case in cases:
    with ExitStack() as stack:
        if case in ('source_bytes','bytecode_bytes','dependency_bytes'):
            target = {'source_bytes':policy['resolved_source'], 'bytecode_bytes':policy['cached'], 'dependency_bytes':'/usr/lib/python3.12/site.py'}[case]
            stack.enter_context(patch.object(run, 'sha', lambda path: '0'*64 if str(Path(path).resolve()) == target else real_sha(path)))
        elif case == 'wrong_origin':
            stack.enter_context(patch.object(original.__spec__, 'origin', str(legacy)))
        elif case == 'legacy_hook':
            stack.enter_context(patch.dict(sys.modules, {'sitecustomize':replacement}))
        elif case == 'missing_hook':
            stack.enter_context(patch.dict(sys.modules))
            del sys.modules['sitecustomize']
        elif case in ('user_loaded','apport_loaded'):
            name = 'usercustomize' if case == 'user_loaded' else 'apport_python_hook'
            stack.enter_context(patch.dict(sys.modules, {name:types.ModuleType(name)}))
        elif case in ('apport_reachable','user_reachable'):
            name = 'apport_python_hook' if case == 'apport_reachable' else 'usercustomize'
            stack.enter_context(patch.object(sys, 'path', [str(out/name), *sys.path]))
        elif case == 'exception_hook':
            stack.enter_context(patch.object(sys, 'excepthook', lambda *args: None))
        elif case in ('trace_hook','profile_hook'):
            stack.enter_context(patch.object(sys, 'gettrace' if case == 'trace_hook' else 'getprofile', lambda: object()))
        else:
            stack.enter_context(patch.object(sys, 'meta_path', [UnknownFinder(), *sys.meta_path]))
        path = out/(case+'.json')
        try:
            run.startup_check(path)
        except ValueError:
            receipt = json.loads(path.read_text())
            assert not receipt['qualified'] and receipt['errors'] and 'origin' in receipt
        else:
            raise AssertionError('unqualified startup accepted: '+case)
assert sys.modules['sitecustomize'] is original
assert not any(name in sys.modules for name in ('apport_python_hook','usercustomize','atom','aiter','torch'))
print('PINNED_STARTUP_BOUNDARIES_PASS:'+str(len(cases)))
"""
    env = dict(run.environment(os.environ.get("K31_TASK10_CPU_ATTEMPT", "author-001")))
    env.update(
        {k: os.environ[k] for k in ("LD_PRELOAD", "KFD_GUARD_LOG") if k in os.environ}
    )
    env["PYTHONPATH"] = f"{run.HERE}:{env['PYTHONPATH']}"
    result = subprocess.run(
        [run.P["python"], "-B", "-c", code, str(tmp_path)],
        cwd=run.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (tmp_path / "startup-boundaries.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PINNED_STARTUP_BOUNDARIES_PASS:14" in result.stdout


@pytest.fixture
def smoke_service(tmp_path):
    common = verify.dependency("common.py")
    owned = run.owned_type()(common)
    server = tmp_path / "http_fixture.py"
    server.write_text(r"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
root = Path(sys.argv[1])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'healthy')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with (root/'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(body)+'\n')
        trickle = (root/'trickle').exists()
        if not trickle:
            time.sleep(float((root/'delay').read_text()))
        reply = json.dumps({'choices':[{'text':'ready'}]}).encode()
        try:
            self.send_response(200); self.send_header('Content-Length',str(len(reply)))
            self.end_headers()
            if trickle:
                for byte in reply:
                    self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(0.1)
            else:
                self.wfile.write(reply)
        except (BrokenPipeError,ConnectionResetError):
            pass
    def log_message(self,*args):
        pass
server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
(root/'port').write_text(str(server.server_port))
server.serve_forever()
""")
    (tmp_path / "delay").write_text("12")
    fd = common.GpuLock(path=str(tmp_path / "cpu-resource.lock")).acquire()
    child = common.spawn(
        [sys.executable, str(server), str(tmp_path)],
        env=os.environ.copy(),
        cwd=tmp_path,
        log_path=tmp_path / "server.log",
        lock_fd=fd,
        cpus="112-127",
    )
    owned.add(child)
    run.wait_until(owned, lambda: (tmp_path / "port").exists(), time.monotonic() + 5)
    url = "http://127.0.0.1:" + (tmp_path / "port").read_text()
    try:
        yield owned, child, fd, url, tmp_path
    finally:
        owned.stop()
        child.wait(timeout=1)
        try:
            os.close(fd)
        except OSError:
            pass


def test_real_smoke_can_finish_after_health_probe_timeout(smoke_service):
    owned, _child, _fd, url, path = smoke_service
    assert run.request(url + "/health") == "healthy"
    started = time.monotonic()
    replies = run.smoke_batch(owned, url, started + 15)
    assert len(replies) == 8 and time.monotonic() - started >= 10
    requests = [
        json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 8
    assert all(
        row
        == {
            "model": "kimi-k3.1-pro",
            "prompt": run.P["smoke"]["prompt"],
            "max_tokens": 8,
            "temperature": 0,
        }
        for row in requests
    )


@pytest.mark.parametrize("failure", ["deadline", "cancel"])
def test_smoke_does_not_wait_for_workers_before_owned_cleanup(
    smoke_service, monkeypatch, failure
):
    owned, child, _fd, url, path = smoke_service
    (path / "delay").write_text("5")
    if failure == "deadline":
        (path / "trickle").touch()
    if failure == "cancel":

        def cancel(*args):
            while not (path / "requests.jsonl").exists():
                time.sleep(0.01)
            raise KeyboardInterrupt("cancelled smoke")

        monkeypatch.setattr(run, "wait_until", cancel)
    started = time.monotonic()
    expected = KeyboardInterrupt if failure == "cancel" else TimeoutError
    try:
        with pytest.raises(expected):
            run.smoke_batch(
                owned, url, started + (0.3 if failure == "deadline" else 10)
            )
        assert time.monotonic() - started < 1
    finally:
        owned.stop()
    assert child.wait(timeout=1) != 0 and not owned.live()
    assert time.monotonic() - started < 2
