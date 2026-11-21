"""Validate complete manifest-bound GSM8K evidence before paired comparison."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def dependency(name):
    record = P["dependencies"][name]
    if sha(record["path"]) != record["sha256"]:
        raise ValueError(f"dependency changed: {name}")
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), record["path"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_data():
    import importlib.metadata

    from lm_eval.tasks import TaskManager
    from lm_eval.utils import hash_string

    if importlib.metadata.version("lm_eval") != "0.4.13":
        raise ValueError("unqualified lm_eval version")
    task = TaskManager().load("gsm8k")["tasks"]["gsm8k"]
    if {k: len(v) for k, v in task.dataset.items()} != {"train": 7473, "test": 1319}:
        raise ValueError("wrong dataset split sizes")
    content = {}
    for split, data in task.dataset.items():
        rows = list(data)
        if any(
            set(d) != {"question", "answer"}
            or any(type(v) is not str for v in d.values())
            for d in rows
        ):
            raise ValueError("wrong dataset schema")
        content[split] = digest(rows)
    task.set_fewshot_seed(P["seeds"][3])
    task.build_all_requests()
    docs = {}
    for request in task.instances:
        key = str(request.doc_id)
        if key in docs:
            raise ValueError("duplicate task request")
        docs[key] = {
            "doc_hash": hash_string(
                json.dumps(request.doc, indent=2, ensure_ascii=False)
            ),
            "prompt_hash": hash_string(request.arguments[0]),
            "target_hash": hash_string(str(task.doc_to_target(request.doc))),
        }
    if set(docs) != {str(i) for i in range(P["samples"])}:
        raise ValueError("task request coverage changed")
    config = task.dump_config()
    return {
        "task": "gsm8k",
        "lm_eval": "0.4.13",
        "config": config,
        "config_sha256": digest(config),
        "docs": docs,
        "content_sha256": content,
        "content_algorithm": "SHA256 UTF-8 json.dumps(list(split), sort_keys=True, ensure_ascii=False); not an Arrow-file hash or the historical content algorithm",
    }


def checked_file(record, root):
    path = Path(record["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or sha(path) != record["sha256"]:
        raise ValueError("evidence file escaped its attempt or changed")
    return path


def validate(manifest, expected):
    result = json.loads(
        checked_file(manifest["result"], manifest["output"]).read_text()
    )
    config = result["configs"]["gsm8k"]
    if (
        digest(config) != expected["config_sha256"]
        or result["versions"]["gsm8k"] != 3.0
    ):
        raise ValueError("task identity mismatch")
    if result["n-samples"]["gsm8k"]["effective"] != P["samples"]:
        raise ValueError("incomplete result count")
    rows = {}
    for line in (
        checked_file(manifest["samples"], manifest["output"]).read_text().splitlines()
    ):
        row = json.loads(line)
        doc_id, filt = row["doc_id"], row["filter"]
        key = (doc_id, filt)
        if (
            type(doc_id) is not int
            or str(doc_id) not in expected["docs"]
            or filt not in P["filters"]
            or key in rows
        ):
            raise ValueError("duplicate or unexpected sample")
        identity = expected["docs"][str(doc_id)]
        raw_identity = {
            "doc_hash": hashlib.sha256(
                json.dumps(row["doc"], indent=2, ensure_ascii=False).encode()
            ).hexdigest(),
            "prompt_hash": hashlib.sha256(
                row["arguments"]["gen_args_0"]["arg_0"].encode()
            ).hexdigest(),
            "target_hash": hashlib.sha256(str(row["target"]).encode()).hexdigest(),
        }
        if raw_identity != identity:
            raise ValueError("sample content disagrees with its identity")
        if (
            row["arguments"]["gen_args_0"]["arg_1"]
            != expected["config"]["generation_kwargs"]
        ):
            raise ValueError("sample generation configuration mismatch")
        if any(row.get(k) != v for k, v in identity.items()):
            raise ValueError("document/prompt/target identity mismatch")
        # Empty, wrong and cap-limited model answers remain scored evidence.
        if (
            row["exact_match"] not in (0, 1)
            or not isinstance(row["resps"], list)
            or not isinstance(row["filtered_resps"], list)
        ):
            raise ValueError("malformed scored response")
        if row.get("task", "gsm8k") != "gsm8k":
            raise ValueError("wrong task")
        rows[key] = row
    if set(rows) != {(i, f) for i in range(P["samples"]) for f in P["filters"]}:
        raise ValueError("missing document/filter evidence")
    counts = {
        f: sum(row["exact_match"] for (i, g), row in rows.items() if g == f)
        for f in P["filters"]
    }
    for filt, count in counts.items():
        if (
            abs(
                result["results"]["gsm8k"]["exact_match," + filt] - count / P["samples"]
            )
            > 1e-12
        ):
            raise ValueError("result/sample score mismatch")
    return rows, {
        "correct": counts,
        "denominator": P["samples"],
        "accuracy": {k: v / P["samples"] for k, v in counts.items()},
        "quality_floor": (
            "PASS" if counts["flexible-extract"] >= P["floor_correct"] else "FAIL"
        ),
    }


def bind_results(output, expected):
    output = Path(output)
    results = list((output / "quality").rglob("results_*.json"))
    samples = list((output / "quality").rglob("samples_gsm8k_*.jsonl"))
    if len(results) != 1 or len(samples) != 1:
        raise ValueError(
            "expected exactly one result/sample set in the fresh namespace"
        )
    manifest = {
        "output": str(output),
        **{
            k: {"path": str(v), "sha256": sha(v)}
            for k, v in (("result", results[0]), ("samples", samples[0]))
        },
    }
    _, score = validate(manifest, expected)
    plan = json.loads((output / "plan.json").read_text())
    return dict(
        manifest,
        score=score,
        head=plan["head"],
        profile_sha256=digest(plan["protocol"]),
        data_sha256=plan["data"]["sha256"],
    )


def first_valid(registry, expected):
    selected = None
    for number, entry in enumerate(registry, 1):
        if entry["attempt"] != f"author-{number:03d}":
            raise ValueError("author attempt history is not contiguous")
        if sha(entry["status"]) != entry["status_sha256"]:
            raise ValueError("attempt status was not sealed")
        status = json.loads(Path(entry["status"]).read_text())
        if (
            status["plan_sha256"] != entry["plan_sha256"]
            or not status["cleanup"]["released"]
        ):
            raise ValueError("attempt identity/release is not established")
        if status["outcome"] == "COMPLETE":
            _, score = validate(status["evidence"], expected)
            if selected is None:
                selected = {
                    "attempt": entry["attempt"],
                    "evidence": status["evidence"],
                    "score": score,
                }
        elif status["outcome"] != "EXECUTION_INVALID":
            raise ValueError("unclassified attempt")
    return selected


def qualify(author, independent, expected):
    if any(
        author[k] != independent[k] for k in ("head", "profile_sha256", "data_sha256")
    ):
        raise ValueError("qualification source/profile/data mismatch")
    a, score_a = validate(author, expected)
    b, score_b = validate(independent, expected)
    helper = dependency("gsm8k_compare.py")
    paired = helper.compare(a, b, show=0)
    flex = paired["by_filter"]["flexible-extract"]
    assert flex["mcnemar_p"] == helper.mcnemar_exact(
        flex["base_only"], flex["candidate_only"]
    )
    established = (
        flex["mcnemar_p"] >= P["alpha"]
        and score_a["quality_floor"] == score_b["quality_floor"]
    )
    return {
        "author": score_a,
        "independent": score_b,
        "paired": paired,
        "repeatability": "NOT_REJECTED" if established else "UNESTABLISHED",
        "interpretation": "Nonsignificance does not establish equivalence; the independent score never replaces A0.",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--select", type=Path)
    parser.add_argument("--qualify", type=Path, nargs=2)
    args = parser.parse_args()
    expected = json.loads(args.data.read_text())
    if args.select:
        result = first_valid(json.loads(args.select.read_text()), expected)
    elif args.qualify:
        statuses = [json.loads(p.read_text()) for p in args.qualify]
        if any(
            s["outcome"] != "COMPLETE" or not s["cleanup"]["released"] for s in statuses
        ):
            raise ValueError("qualification requires two complete released attempts")
        result = qualify(*(s["evidence"] for s in statuses), expected)
    else:
        parser.error(
            "use --select REGISTRY or --qualify AUTHOR_STATUS INDEPENDENT_STATUS"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
