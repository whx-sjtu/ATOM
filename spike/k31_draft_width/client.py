"""Completed HTTP request wall per returned token, with natural speculation."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(str(ids).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cache_inventory(roots):
    files = {}
    for root in roots:
        for path in Path(root).rglob("*"):
            if (
                path.is_file()
                and path.name not in ("lock", ".ninja_lock")
                and not (path.name.startswith("lock_") or path.name.endswith(".lock"))
            ):
                stat = path.stat()
                files[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return files


def cache_changes(before, after):
    return {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }


def validate_inputs(data, model=None):
    definition = json.loads((HERE / "fixtures.json").read_text())
    if (
        data.get("definition_sha256") != sha256(HERE / "fixtures.json")
        or data.get("source_sha256") != definition["source_sha256"]
    ):
        raise ValueError("input definition/source identity differs")
    requests = data.get("requests", [])
    if len(requests) != len(definition["requests"]):
        raise ValueError("frozen request set differs")
    if model is not None and Path(data["model"]).resolve() != Path(model).resolve():
        raise ValueError("input tokenizer source differs")
    for index, (item, expected) in enumerate(zip(requests, definition["requests"])):
        ids = item.get("token_ids", [])
        if item.get("fixture") != index or any(
            item.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("frozen request metadata differs")
        if (
            len(ids) != expected["length"]
            or any(type(x) is not int or x < 0 for x in ids)
            or token_hash(ids) != expected["sha256"]
        ):
            raise ValueError("frozen prompt tokens differ")


def validate_repeats(repeats):
    if type(repeats) is not int or repeats < 2:
        raise ValueError("at least two measured repeats are required")


def prepare_inputs(model, source, output):
    from transformers import AutoTokenizer

    definition = json.loads((HERE / "fixtures.json").read_text())
    if sha256(source) != definition["source_sha256"]:
        raise ValueError("source text differs from the frozen fixture")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokens = tokenizer.encode(Path(source).read_text(), add_special_tokens=False)
    requests = []
    for index, item in enumerate(definition["requests"]):
        ids = tokens[item["offset"] : item["offset"] + item["length"]]
        if len(ids) != item["length"] or token_hash(ids) != item["sha256"]:
            raise ValueError(f"fixture {index} token identity differs")
        requests.append({"fixture": index, **item, "token_ids": ids})
    save(
        output,
        {
            "model": str(Path(model).resolve()),
            "source_sha256": sha256(source),
            "definition_sha256": sha256(HERE / "fixtures.json"),
            "requests": requests,
        },
    )


def payload(model, ids, limit=128, echo_ids=False):
    return {
        "model": model,
        "prompt_token_ids": ids,
        "n": 1,
        "temperature": 0.0,
        "max_tokens": limit,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": echo_ids,
    }


async def complete(client, body, identity, ledger):
    record = {
        **identity,
        "success": False,
        "cancelled": False,
        "error": None,
        "output_tokens": None,
        "requested_limit": body["max_tokens"],
        "actual_internal_batch": None,
    }
    start = time.perf_counter_ns()
    record["start_ns"] = start
    record["start_epoch_ns"] = time.time_ns()
    try:
        response = await client.post("/v1/completions", json=body)
        record["http_status"] = response.status_code
        response.raise_for_status()
        result = response.json()
        usage = result.get("usage", {})
        count = usage.get("completion_tokens")
        record.update(response_id=result.get("id"), usage=usage, output_tokens=count)
        prompt_count = usage.get("prompt_tokens")
        if type(prompt_count) is not int or prompt_count != len(
            body["prompt_token_ids"]
        ):
            raise ValueError("response prompt token count differs from the request")
        if type(count) is not int or count <= 0:
            raise ValueError("missing or invalid response completion_tokens")
        if len(result.get("choices", [])) != 1:
            raise ValueError("expected exactly one completed choice")
        reason = result["choices"][0].get("finish_reason")
        record["finish_reason"] = reason
        # 'stop' can represent an aborted request, so it does not certify this
        # experiment's ignore_eos output cap even when HTTP returned 200.
        if reason != "length" or count != body["max_tokens"]:
            raise ValueError("response did not complete the declared output cap")
        if body["return_token_ids"]:
            returned = result.get("prompt_token_ids")
            if returned != body["prompt_token_ids"]:
                raise ValueError("server prompt IDs differ from the frozen input")
            record["confirmed_prompt_sha256"] = token_hash(returned)
        record["success"] = True
    except asyncio.CancelledError:
        record.update(cancelled=True, error="cancelled")
        raise
    except Exception as error:  # noqa: BLE001 - preserve request failure evidence
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["end_ns"] = time.perf_counter_ns()
        record["wall_ns"] = record["end_ns"] - start
        ledger.append(record)
    return record


def summarize(records, gpu_count=8):
    if not records or any(not row["success"] for row in records):
        raise ValueError("incomplete request set; no survivor-only performance result")
    tokens = sum(row["output_tokens"] for row in records)
    span = (
        max(row["end_ns"] for row in records) - min(row["start_ns"] for row in records)
    ) / 1e9
    inputs = sum(row["usage"]["prompt_tokens"] for row in records)
    return {
        "completed_requests": len(records),
        "output_tokens": tokens,
        "input_tokens": inputs,
        "request_wall_ms_per_token": sum(row["wall_ns"] for row in records)
        / 1e6
        / tokens,
        "completion_span_s": span,
        "output_tps": tokens / span,
        "total_tps_per_gpu": (inputs + tokens) / span / gpu_count,
        "actual_internal_batch": None,
        "official_result": False,
    }


async def observe(client):
    result = {}
    for name in ("cache_stats", "mtp_stats"):
        response = await client.get(f"/debug/{name}")
        response.raise_for_status()
        result[name] = response.json()
    return result


async def measure(
    url,
    model,
    inputs,
    k,
    repeats,
    output,
    client=None,
    cache_roots=(),
    initial_cache=None,
):
    validate_inputs(inputs)
    validate_repeats(repeats)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    owned = client is None
    client = client or httpx.AsyncClient(
        base_url=url, timeout=httpx.Timeout(180), trust_env=False
    )
    ledger, rounds, measured_cache = [], [], None
    stages = [("prime", 4, 0, 8)] + [("warm", c, 0, 128) for c in (1, 2, 4)]
    stages += [("measure", c, r, 128) for c in (1, 2, 4) for r in range(repeats)]
    try:
        for stage, concurrency, repeat, limit in stages:
            if stage == "measure" and measured_cache is None and cache_roots:
                measured_cache = cache_inventory(cache_roots)
                save(
                    output / "startup.json",
                    {
                        "warmup_finished_epoch_ns": time.time_ns(),
                        "cache_roots": [str(path) for path in cache_roots],
                        "cache_after_warmup": measured_cache,
                        "untimed_cache_changes": cache_changes(
                            initial_cache or {}, measured_cache
                        ),
                    },
                )
            before = await observe(client)
            semaphore = asyncio.Semaphore(concurrency)

            async def one(
                item, sem=semaphore, phase=stage, c=concurrency, r=repeat, cap=limit
            ):
                async with sem:
                    identity = {
                        "k": k,
                        "stage": phase,
                        "client_concurrency": c,
                        "repeat": r,
                        "fixture": item["fixture"],
                        "prompt_sha256": item["sha256"],
                    }
                    return await complete(
                        client,
                        payload(model, item["token_ids"], cap, phase == "prime"),
                        identity,
                        ledger,
                    )

            records = await asyncio.gather(*(one(item) for item in inputs["requests"]))
            save(output / "requests.json", ledger)
            if stage == "measure" and measured_cache is not None:
                changes = cache_changes(measured_cache, cache_inventory(cache_roots))
                if changes:
                    save(output / "timed-cache-changes.json", changes)
                    raise RuntimeError("compiler cache changed during timed requests")
            summary = summarize(records)
            rounds.append(
                {
                    "stage": stage,
                    "client_concurrency": concurrency,
                    "repeat": repeat,
                    **summary,
                    "observations_before": before,
                    "observations_after": await observe(client),
                }
            )
            save(output / "rounds.json", rounds)
        table = []
        for c in (1, 2, 4):
            costs = [
                r["request_wall_ms_per_token"]
                for r in rounds
                if r["stage"] == "measure" and r["client_concurrency"] == c
            ]
            table.append(
                {
                    "k": k,
                    "client_concurrency": c,
                    "repeats": len(costs),
                    "median_ms_per_token": statistics.median(costs),
                    "min_ms_per_token": min(costs),
                    "max_ms_per_token": max(costs),
                    "round_ms_per_token": costs,
                }
            )
        save(output / "summary.json", {"complete": True, "table": table})
        return table
    finally:
        save(output / "requests.json", ledger)
        if owned:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("measure")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--inputs", required=True)
    run.add_argument("--k", type=int, choices=(3, 4), required=True)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_inputs(args.model, args.source, args.output)
        return
    if os.environ.get("K31_SPIKE_GPU_GRANTED") != "1" or args.repeats < 2:
        raise RuntimeError("reviewed GPU lease and at least two repeats required")
    inputs = json.loads(Path(args.inputs).read_text())
    for item in inputs["requests"]:
        if token_hash(item["token_ids"]) != item["sha256"]:
            raise ValueError("input artifact changed")
    asyncio.run(
        measure(args.base_url, args.model, inputs, args.k, args.repeats, args.output)
    )


if __name__ == "__main__":
    main()
