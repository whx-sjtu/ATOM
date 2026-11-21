"""Build the exact bounded exposure plan without starting a service."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WS = Path("/share_nfs/whx/work/k31_perf_opt/.opt-team")
P = json.loads((HERE / "protocol.json").read_text())
PRIVATE = Path(P["private_root"])
CLIENT = Path(P["client_root"])


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _cli(program, flags):
    argv = list(program)
    for flag, value in flags.items():
        argv.append(flag)
        if value is not None:
            argv.append(
                json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            )
    return argv


def commands(out):
    url = f"http://127.0.0.1:{P['port']}"
    kv = {
        "kv_connector": "lmcache_mp",
        "kv_role": "offload",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": P["cache_port"],
        },
    }
    server = _cli(
        ["/opt/venv/bin/python", "-m", "atom.entrypoints.openai_server"],
        {
            "--model": P["model"],
            "--served-model-name": P["served_name"],
            "--host": "127.0.0.1",
            "--server-port": P["port"],
            "--trust-remote-code": None,
            "-tp": 8,
            "--decode-context-parallel-size": 8,
            "--kv_cache_dtype": "fp8",
            "--index-cache-dtype": "fp4",
            "--index-cache-layout": "mfma",
            "--block-size": P["block_size"],
            "--enable_prefix_caching": None,
            "--max-model-len": P["max_model_len"],
            "--max-num-seqs": P["max_num_seqs"],
            "--max-num-batched-tokens": P["max_num_batched_tokens"],
            "--gpu-memory-utilization": P["gpu_memory_utilization"],
            "--state-checkpoint-interval-tokens": 8192,
            "--level": 3,
            "--cudagraph-mode": "FULL",
            "--cudagraph-capture-sizes": P["graph_batches"],
            "--method": "dspark",
            "--draft-model": P["draft"],
            "--num-speculative-tokens": 3,
            "--spec-decode-acceptance-length": 2.88,
            "--kv-transfer-config": kv,
            "--torch-profiler-dir": str(out / "traces"),
        },
    )
    cache = _cli(
        ["/opt/venv/bin/lmcache", "server"],
        {
            "--host": "127.0.0.1",
            "--port": P["cache_port"],
            "--http-port": P["cache_http_port"],
            "--prometheus-port": P["cache_metrics_port"],
            "--instance-id": out.name,
            "--chunk-size": P["chunk_size"],
            "--null-block-id": -1,
            "--separate-object-groups": None,
            "--supported-transfer-mode": "lmcache_driven",
            "--l1-size-gb": P["cpu_cache_gb"],
            "--l1-use-lazy": None,
            "--eviction-policy": "LRU",
            "--max-gpu-workers": 1,
            "--enable-extra-logging": None,
            "--extra-logging-interval": 10,
        },
    )
    client = _cli(
        [str(CLIENT / "venv/bin/agentx"), "profile"],
        {
            "--unsafe-override": None,
            "--scenario": "agentx",
            "--url": url,
            "--endpoint": "/v1/chat/completions",
            "--endpoint-type": "chat",
            "--streaming": None,
            "--model": P["served_name"],
            "--concurrency": 12,
            "--benchmark-duration": 120,
            "--stats-interval": 30,
            "--random-seed": 42,
            "--failed-request-threshold": "0.10",
            "--trajectory-start-min-ratio": "0.25",
            "--trajectory-start-max-ratio": "0.75",
            "--warmup-requests-per-lane": 10,
            "--trace-idle-gap-cap-seconds": 300,
            "--agentic-warmup-grace-period": 1800,
            "--use-server-token-count": None,
            "--no-gpu-telemetry": None,
            "--tokenizer": P["model"],
            "--tokenizer-trust-remote-code": None,
            "--num-dataset-entries": 393,
            "--slice-duration": "1.0",
            "--public-dataset": "semianalysis_cc_traces_weka_062126",
            "--server-metrics": url + "/metrics",
        },
    )
    return server, cache, client


def environment():
    env = {
        k: os.environ[k]
        for k in ("LD_LIBRARY_PATH", "LANG", "LC_ALL")
        if k in os.environ
    }
    env.update(
        {
            "PATH": "/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": f"{ROOT}:{P['aiter_root']}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "HOME": str(PRIVATE / "home"),
            "TMPDIR": str(PRIVATE / "tmp"),
            "AITER_JIT_DIR": str(PRIVATE / "cache/aiter/jit"),
            "AITER_CACHE_DIR": str(PRIVATE / "cache/aiter"),
            "FLYDSL_RUNTIME_CACHE_DIR": str(PRIVATE / "cache/flydsl"),
            "TORCHINDUCTOR_CACHE_DIR": str(PRIVATE / "cache/inductor"),
            "TRITON_CACHE_DIR": str(PRIVATE / "cache/triton"),
            "AMD_COMGR_CACHE_DIR": str(PRIVATE / "cache/comgr"),
            "TORCH_EXTENSIONS_DIR": str(PRIVATE / "cache/extensions"),
            "XDG_CACHE_HOME": str(PRIVATE / "cache/xdg"),
            "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "ARCH": "gfx950",
            "GPU_ARCHS": "gfx950",
            "FLYDSL_GPU_ARCH": "gfx950",
            "MAX_JOBS": "1",
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HOME": str(CLIENT / "hf-home"),
            "HF_MODULES_CACHE": str(CLIENT / "hf-modules"),
            "TIKTOKEN_CACHE_DIR": str(CLIENT / "cache/tiktoken"),
            "HF_DATASETS_CACHE": str(PRIVATE / "hf-datasets"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ATOM_MONO_ENABLE": "1",
            "ATOM_MONO_POLL_SLEEP_UNITS": "16",
            "ATOM_MLA_ABSORB_BF16": "1",
            "AITER_META_DIR": P["aiter_root"],
            "AITER_ASM_DIR": str(Path(P["aiter_root"]) / "hsa"),
            "OPUS_GEN_CO_DIR": str(Path(P["aiter_root"]) / "csrc/opus_gemm/gen_co"),
            "ATOM_ENABLE_REPLAYSSM": "1",
            "ATOM_GDN_SSM_DTYPE": "fp32",
            "AITER_SITUV2_A4W4": "1",
            "AITER_LOG_LEVEL": "WARNING",
            "ATOM_STATE_CHECKPOINT_DEMAND": "1",
            "LMCACHE_CHUNK_SIZE": str(P["chunk_size"]),
            "ATOM_PROFILER_RECORD_SHAPES": "1",
            "ATOM_PROFILER_WITH_STACK": "0",
            "ATOM_PROFILER_PROFILE_MEMORY": "0",
            "ATOM_PROFILER_MORE": "0",
            "ATOM_ENABLE_DETAILED_ANNOTATION": "1",
            "ATOM_PROFILER_TIMEOUT": "120",
            "AIPERF_BENCHMARK_DURATION": "120",
            "AIPERF_UNSAFE_OVERRIDE": "1",
            "AIPERF_PUBLIC_DATASET": "semianalysis_cc_traces_weka_062126",
            "AIPERF_WARMUP_REQUESTS_PER_LANE": "10",
            "AIPERF_DATASET_MMAP_CACHE_ENABLED": "false",
            "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES": "0",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
            "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
            "ATOM_HARNESS_VERSION": "1.0.13",
            "GITHUB_REPOSITORY": "whx-sjtu/ATOM",
            "BENCHMARK_HARDWARE": "mi355x",
            "GPU_NAME": "AMD Instinct MI355X",
            "BENCHMARK_PRECISION": "native-bf16-a4w4-fp8-kv-fp4-index",
            "ATOM_IMAGE_DIGEST": "rocm/atom-dev@sha256:ea74fa0b05d8c7a062ae235b901cb2b1bc0c169b0a5edd7ca07bfa3b967ca462",
            "DOCKER_IMAGE": "rocm/atom-dev:latest",
        }
    )
    revision = Path(P["model"]) / ".cache/huggingface/download/config.json.metadata"
    if revision.exists():
        env["BENCHMARK_MODEL_REVISION"] = revision.read_text().splitlines()[0]
    info = CLIENT / "venv/lib/python3.12/site-packages/aiperf/_build_info.py"
    match = info.exists() and re.search(
        r'COMMIT_SHA = "([0-9a-f]{40})"', info.read_text()
    )
    if match:
        env["ATOM_HARNESS_SHA"] = match[1]
    return env


def make_plan(out, mode="exposure"):
    if mode not in ("exposure", "warmup_observer"):
        raise ValueError(mode)
    server, cache, client = commands(out)
    external = [
        WS / "harness/common.py",
        PRIVATE / "runtime-ready.json",
        PRIVATE / "runtime-imports.json",
        PRIVATE / "dataset-ready.json",
    ]
    env = environment()
    roles = {role: dict(env) for role in ("server", "cache", "client", "metadata")}
    config = P["observer"] if mode == "warmup_observer" else None
    if config:
        roles["server"].update(
            {
                "PYTHONPATH": f"{HERE / 'worker_bootstrap'}:{env['PYTHONPATH']}",
                "K31_OBSERVER_PLAN": str(out / "plan.json"),
                "TORCH_NCCL_TRACE_BUFFER_SIZE": str(config["trace_buffer_size"]),
                "TORCH_FR_BUFFER_SIZE": str(config["trace_buffer_size"]),
                "TORCH_NCCL_ENABLE_TIMING": "0",
            }
        )
    return {
        "mode": mode,
        "native_sources": {
            str(p): sha(p)
            for p in (
                ROOT / "atom/model_engine/async_proc.py",
                ROOT / "atom/model_engine/model_runner.py",
                ROOT / "atom/model_ops/attentions/aiter_mla.py",
                ROOT / "atom/model_ops/attention_mla.py",
                ROOT / "atom/model_ops/indexer/indexer_forward.py",
                Path(P["aiter_root"]) / "aiter/dist/parallel_state.py",
            )
        },
        "observer_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "environments": roles,
        "head": git("rev-parse", "HEAD"),
        "base": P["base"],
        "output": str(out),
        "protocol": P,
        "server": server,
        "cache": cache,
        "client": client,
        "environment": env,
        "sources": {
            str(p): sha(p)
            for p in sorted(HERE.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
        "atom_compile_cache": str(PRIVATE / "home/.cache/atom/torch_compile_cache"),
        "invocations": ["warmup_observer"] if config else ["u1", "u2", "diagnostic"],
        "official_submission_eligible": False,
        "gpu_authorized_by_plan": False,
        "readiness": {str(p): sha(p) if p.exists() else None for p in external},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument(
        "--mode", choices=("exposure", "warmup_observer"), default="exposure"
    )
    args = parser.parse_args()
    plan = make_plan(args.out.resolve(), args.mode)
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
