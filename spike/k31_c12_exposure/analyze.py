"""Preserve canonical metrics, classify invalid completions, and qualify traces."""

import argparse
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

from plan import PRIVATE, ROOT

RANKS = {str(rank) for rank in range(8)}
SENT = re.compile(
    r"Phase (\S+) \(profiling\) sending complete \| sent=([\d,]+), completed=([\d,]+), in_flight=([\d,]+)"
)
DONE = re.compile(
    r"Phase (\S+) \(profiling\) complete \| completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)"
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def request_ledger(directory):
    from atom.benchmarks.results.records import from_aiperf

    log = directory / "logs/aiperf.log"
    if not log.exists():
        log = directory / "console.log"
    text = log.read_text(errors="replace")
    sent, done = SENT.findall(text), DONE.findall(text)
    records = directory / "profile_export.jsonl"
    result = {
        "ledger_source": str(log),
        "sent": None,
        "completed": None,
        "cancelled": None,
        "errors": None,
        "zero_or_missing_output_success": [],
        "profile_success_records": 0,
        "profile_output_tokens": 0,
        "usable_completed_requests": False,
    }
    if sent and done and sent[-1][0] == done[-1][0]:
        result["sent"] = int(sent[-1][1].replace(",", ""))
        result.update(
            dict(
                zip(
                    ("completed", "cancelled", "errors"),
                    (int(x.replace(",", "")) for x in done[-1][1:]),
                )
            )
        )
    if not records.exists():
        result["missing"] = "raw AIPerf request JSONL"
        return result
    with records.open() as source:
        for index, line in enumerate(source):
            row = from_aiperf(json.loads(line), index)
            if row.get("status") != "success":
                continue
            if row.get("output_tokens", 0) <= 0:
                result["zero_or_missing_output_success"].append(row["request_id"])
            if row.get("phase") == "profiling":
                result["profile_success_records"] += 1
                result["profile_output_tokens"] += row.get("output_tokens", 0)
    result["usable_completed_requests"] = bool(
        sent
        and done
        and result["profile_success_records"] > 0
        and result["completed"] == result["profile_success_records"]
        and result["errors"] == 0
        and not result["zero_or_missing_output_success"]
        and result["sent"] == result["completed"] + result["cancelled"]
    )
    result["scope"] = (
        "Raw final ledger including grace. HTTP 200 and zero output is not a usable completion."
    )
    return result


def union_us(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        total += stop - start if end is None or start > end else max(0.0, stop - end)
        end = stop if end is None else max(end, stop)
    return total


def row_fields(name):
    match = re.match(
        r"((?:dummy_)?(?:prefill|decode|eager_decode))\[bs=(\d+)(?:/(\d+))?(?: tok=(\d+)(?:/(\d+))?)?",
        name,
    )
    if not match:
        return None
    kind, batch, running, tokens, rows = match.groups()
    return {
        "kind": kind,
        "scheduled_batch": int(batch),
        "running_batch": int(running or batch),
        "scheduled_tokens": int(tokens) if tokens else None,
        "running_rows": int(rows or tokens) if tokens else None,
    }


def qualify_window(record, trace_root):
    reasons = []
    if record.get("errors"):
        reasons.append("control/export error")
    if (
        set(record.get("started", {})) != RANKS
        or set(record.get("stopped", {})) != RANKS
    ):
        reasons.append("all-rank lifecycle unproven")
    upper = record.get("upper_bound_s", {})
    if set(upper) != RANKS or any(not 0 <= value <= 10 for value in upper.values()):
        reasons.append("all-rank active-time upper bound unproven")
    if any(
        record["stopped"][r] < start or start < record["start_request_mono"]
        for r, start in record.get("started", {}).items()
        if r in record.get("stopped", {})
    ):
        reasons.append("unordered lifecycle")
    if set(record.get("exports", {})) != RANKS:
        reasons.append("all-rank exports missing")
    traces = {}
    for rank, filename in record.get("exports", {}).items():
        path = Path(filename).resolve()
        try:
            if not path.is_relative_to(trace_root.resolve() / f"rank_{rank}") or str(
                path
            ) in record.get("before_files", []):
                raise ValueError("not a fresh export from this rank")
            with gzip.open(path, "rb") as source:
                body = source.read(512 * 2**20 + 1)
            if len(body) > 512 * 2**20:
                raise ValueError(
                    "trace exceeds bounded parser capacity; raw file retained"
                )
            events = json.loads(body)["traceEvents"]
            kernels = [
                e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"
            ]
            if not kernels:
                raise ValueError("no complete GPU kernel events")
            families = {}
            for event in kernels:
                name = event["name"]
                family = next(
                    (
                        label
                        for stem, label in (
                            ("k3_mono_k2", "mono_k2"),
                            ("k3_mono_kda_pre", "mono_k1"),
                            ("mono_dsa_indexer_", "mono_indexer"),
                        )
                        if stem in name
                    ),
                    None,
                )
                if family is None:
                    family = (
                        "communication"
                        if re.search(
                            r"nccl|rccl|all_reduce|all_gather|reduce_scatter",
                            name,
                            re.IGNORECASE,
                        )
                        else "other"
                    )
                families.setdefault(family, []).append(
                    (event["ts"], event["ts"] + event["dur"])
                )
            labels = [
                e
                for e in events
                if e.get("ph") == "X"
                and (
                    e.get("name", "").startswith("ATOM::")
                    or row_fields(e.get("name", "")) is not None
                )
            ]
            traces[rank] = {
                "path": filename,
                "kernel_count": len(kernels),
                "kernel_union_us": union_us(
                    [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
                ),
                "family_union_us_nonadditive": {
                    k: union_us(v) for k, v in families.items()
                },
                "host_scopes": [
                    {
                        "name": e["name"],
                        "ts": e["ts"],
                        "dur": e["dur"],
                        "rows": row_fields(e["name"]),
                    }
                    for e in labels
                ],
                "mono_kernel_rows": sorted(
                    {
                        int(m[1])
                        for e in kernels
                        if (
                            m := re.search(r"k3_mono_(?:k2|kda_pre)_s(\d+)_", e["name"])
                        )
                    }
                ),
                "complete_mla_tail_gpu_cost": None,
                "embedded_k2_communication_cost": None,
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append(f"rank {rank}: {error}")
    return {
        "qualified": not reasons,
        "reasons": reasons,
        "ranks": traces,
        "scope": "Trace-visible intervals only. Families can overlap; host scopes are not GPU wall time.",
    }


def build_cases(out):
    from atom.benchmarks.results.bundle import build_bundle
    from atom.benchmarks.results.metadata import capture_hf_dataset

    base = read(out / "base-config.json")
    for label in ("u1", "u2", "diagnostic"):
        directory = out / label
        export = directory / "profile_export_aiperf.json"
        if not export.exists():
            continue
        ledger = request_ledger(directory)
        write(directory / "ledger.json", ledger)
        config = json.loads(json.dumps(base))
        config["validity"]["instrumented"] = label == "diagnostic"
        identity = capture_hf_dataset(PRIVATE / "hf-datasets", export)
        write(directory / "dataset-identity.json", identity)
        bundle = directory / "bundle"
        client_path = directory / "client-result.json"
        client = read(client_path) if client_path.exists() else {}
        if not bundle.exists():
            build_bundle(
                config,
                directory / "profile_export.jsonl",
                bundle,
                record_format="aiperf",
                raw_dir=directory,
                logs={
                    "client": directory / "logs/aiperf.log",
                    "server": out / "server.log",
                },
                exit_code=client.get("exit_code", 1),
                client_launch=directory / "launch.json",
            )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atom.benchmarks.results",
                "verify",
                str(bundle),
                "--require-full",
            ],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        write(
            directory / "verification.json",
            {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


def summarize(out):
    result = {
        "head": read(out / "plan.json")["head"],
        "official_submission_eligible": False,
        "unprofiled": {},
        "diagnostics": [],
        "unknown": ["full-workload cost distribution", "strict B300 headroom"],
    }
    for label in ("u1", "u2"):
        path = out / label / "bundle/summary.json"
        if path.exists():
            data = read(path)
            item = {
                "request_metrics": data.get("request_metrics"),
                "accounting": data.get("accounting"),
                "window": data.get("window"),
                "capabilities": data.get("capabilities"),
            }
            for field, filename in (
                ("validation", path.with_name("validation.json")),
                ("verification", out / label / "verification.json"),
                ("ledger", out / label / "ledger.json"),
                ("client_result", out / label / "client-result.json"),
            ):
                item[field] = read(filename) if filename.exists() else {}
            item["censored"] = (out / label / "censored.json").exists() or item[
                "client_result"
            ].get("censored", True)
            item["range_eligible"] = bool(
                item["ledger"].get("usable_completed_requests")
                and item["client_result"].get("exit_code") == 0
                and item["client_result"].get("phase_observed") is True
                and not item["censored"]
                and item["verification"].get("exit_code") == 0
                and item["validation"].get("complete") is True
                and item["validation"].get("exit_code") == 0
                and not data["accounting"]["counts"].get("invalid")
                and data["accounting"]["counts"].get("profiled_success")
                == item["ledger"].get("profile_success_records")
            )
            result["unprofiled"][label] = item
        else:
            result["unprofiled"][label] = {"missing": "canonical bundle"}
    for path in sorted((out / "diagnostic").glob("window*.json")):
        result["diagnostics"].append(qualify_window(read(path), out / "traces"))
    qualified = [d for d in result["diagnostics"] if d["qualified"]]
    result["conclusion"] = (
        "Missing observation: complete native MLA tail and embedded K2 communication boundaries. Trace family intervals and host scopes cannot establish either optimization benefit."
        if qualified
        else "Missing observation: a qualified all-rank diagnostic window with complete trace exports within the active-time bound."
    )
    result["reproduction_range"] = {}
    for name, path in (
        ("total_throughput_per_gpu_tps", ("throughput", "per_gpu", "total_tput_tps")),
        (
            "full_response_interactivity_p90_tps",
            ("latency", "full_response_intvty", "p90"),
        ),
    ):
        values = []
        for item in result["unprofiled"].values():
            if item.get("range_eligible"):
                value = item.get("request_metrics", {})
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, (int, float)):
                    values.append(value)
        result["reproduction_range"][name] = (
            {"min": min(values), "max": max(values), "samples": values}
            if len(values) == 2
            else None
        )

    write(out / "exposure.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--build-bundles", action="store_true")
    args = parser.parse_args()
    if args.build_bundles:
        build_cases(args.out)
    print(json.dumps(summarize(args.out)))
