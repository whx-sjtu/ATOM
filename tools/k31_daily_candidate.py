"""Compose detached ATOM/AITER candidates without running models.

JSON inputs declare source roots/base commits/inference_paths, components, flags,
base_flags, dependencies, comparison_controls, recipe, evidence and a hash-bound
harness reference. compose preserves conflicts for manual resolution; freeze
records a clean resolved result. reproduce materializes the exact recorded heads.
A source manifest may be prospective; require_measurable never grants a lease.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def reference(ref):
    path = Path(ref["path"])
    if not path.is_absolute() or file_hash(path) != ref["sha256"]:
        raise ValueError("missing or changed evidence reference")
    return ref


def harness(ref):
    reference(ref)
    spec = importlib.util.spec_from_file_location("candidate_harness", ref["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.git_info


def git(root, *args, env=None):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command refused")
    return result.stdout.strip()


def commit(root, value):
    if (
        not re.fullmatch("[0-9a-f]{40}", value)
        or git(root, "rev-parse", value + "^{commit}") != value
    ):
        raise ValueError("an exact available commit is required")
    return value


def clean(root, inspect):
    info = inspect(root)
    head = git(root, "rev-parse", "HEAD")
    if (
        info["commit"] != head
        or info["dirty_files"]
        or git(root, "status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("dirty or inconsistent source identity")
    return info


def closure(root, revision, paths):
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("explicit inference paths are required")
    for path in paths:
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or str(parsed) != path
            or path == "."
        ):
            raise ValueError("invalid inference path")
    entries = git(root, "ls-tree", "-rz", "--full-tree", revision).split("\0")
    selected = {}
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        if any(name == path or name.startswith(path + "/") for path in paths):
            selected[name] = metadata
    if any(
        not any(name == path or name.startswith(path + "/") for name in selected)
        for path in paths
    ):
        raise ValueError("inference path is absent from source")
    return {"paths": sorted(paths), "tree_fingerprint": fingerprint(selected)}


def evidence(record, subject, required=False):
    if record.get("status") not in ("pending", "passed", "failed"):
        raise ValueError("explicit evidence status is required")
    if record.get("reference") is not None:
        reference(record["reference"])
    if record["status"] == "passed":
        if record.get("subject") != subject or not record.get("reference"):
            raise ValueError("evidence does not bind the exact subject")
    elif required:
        raise ValueError("required evidence is not passed")


def selected(request):
    chosen = []
    identifiers = []
    for item in request["components"]:
        identifiers.append(item["id"])
        if (
            item["kind"] not in ("inference", "operator", "diagnostic", "tool")
            or type(item["implemented"]) is not bool
        ):
            raise ValueError("unknown component classification")
        ready = item["kind"] == "inference" and item["implemented"] is True
        if type(item["include"]) is not bool:
            raise ValueError("inclusion must be explicit")
        if item["include"]:
            if not ready:
                raise ValueError("only implemented inference changes can be included")
            for key in ("source_review", "numerical"):
                evidence(item["evidence"][key], item["commit"], required=True)
            if any(
                key not in request["flags"]
                or fingerprint(request["flags"][key]) != fingerprint(value)
                for key, value in item["flags"].items()
            ):
                raise ValueError("component flags conflict with candidate flags")
            if (
                item["repo"] == "aiter"
                and not {"atom", "abi", "runtime"} <= item["requires"].keys()
            ):
                raise ValueError(
                    "AITER changes require exact ATOM, ABI and runtime constraints"
                )
            chosen.append(item)
        elif not item.get("reason") or not item.get("omission_evidence"):
            raise ValueError("every omission needs a reason and evidence")
        else:
            reference(item["omission_evidence"])
        if (
            not item["include"]
            and ready
            and all(
                item["evidence"].get(key, {}).get("status") == "passed"
                for key in ("source_review", "numerical")
            )
        ):
            if item.get("omission") != "incompatible" or not item.get(
                "omission_evidence"
            ):
                raise ValueError(
                    "ready inference changes cannot be omitted for pending noise/quality"
                )
            reference(item["omission_evidence"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate roster identity")
    return chosen


def inputs(request, inspect):
    if set(request["sources"]) != {"atom", "aiter"}:
        raise ValueError("exact ATOM and AITER source records are required")
    reference(request["recipe"]["reference"])
    if request["recipe"]["qualification"] not in ("pending", "qualified"):
        raise ValueError("unknown recipe qualification")
    if (
        not request["comparison_controls"]
        or not {"abi", "runtime"} <= request["dependencies"].keys()
    ):
        raise ValueError("fixed comparison, ABI and runtime identities are required")
    if set(request["evidence"]) != {
        "source_review",
        "numerical",
        "smoke",
        "quality",
        "performance",
    }:
        raise ValueError("combined evidence and quality status must be explicit")
    if request["recipe"]["qualification"] == "qualified":
        value = request["recipe"].get("fingerprint") or ""
        if not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("canonical recipe fingerprint is missing")
        evidence(request["recipe"]["review"], value, required=True)
    info = {}
    for name, source in request["sources"].items():
        root = Path(source["root"]).resolve()
        if root != Path(git(root, "rev-parse", "--show-toplevel")).resolve():
            raise ValueError("source root must be the Git worktree toplevel")
        remote = git(root, "remote", "get-url", "origin").removesuffix(".git")
        expected = "ATOM" if name == "atom" else "aiter"
        if remote not in (
            f"https://github.com/whx-sjtu/{expected}",
            f"git@github.com:whx-sjtu/{expected}",
        ):
            raise ValueError("candidate sources must use the declared fork")
        info[name] = clean(root, inspect)
        commit(root, source["base"])
        closure(root, source["base"], source["inference_paths"])
    for item in request["components"]:
        if (
            item.get("commit") is None
            and item["implemented"] is False
            and item["include"] is False
        ):
            continue
        commit(request["sources"][item["repo"]]["root"], item["commit"])
    return info


def destination_directory(request, destination):
    destination = Path(destination).resolve()
    if any(
        destination.is_relative_to(Path(item["root"]).resolve())
        for item in request["sources"].values()
    ):
        raise ValueError("candidate output cannot be inside an original source")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def prepare(request, destination, inspect):
    chosen = selected(request)
    before = inputs(request, inspect)
    destination = destination_directory(request, destination)
    for name, source in request["sources"].items():
        git(
            source["root"],
            "worktree",
            "add",
            "--detach",
            str(destination / name),
            source["base"],
        )
    for name, source in request["sources"].items():
        target = destination / name
        for item in (item for item in chosen if item["repo"] == name):
            epoch = max(
                int(git(target, "show", "-s", "--format=%ct", revision))
                for revision in ("HEAD", item["commit"])
            )
            env = dict(
                os.environ,
                GIT_AUTHOR_NAME="ATOM candidate",
                GIT_COMMITTER_NAME="ATOM candidate",
                GIT_AUTHOR_EMAIL="candidate@localhost",
                GIT_COMMITTER_EMAIL="candidate@localhost",
                GIT_AUTHOR_DATE=f"{epoch} +0000",
                GIT_COMMITTER_DATE=f"{epoch} +0000",
            )
            git(
                target,
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "merge",
                "--no-ff",
                "-m",
                f"Compose {name} {item['commit']}",
                item["commit"],
                env=env,
            )
    if before != inputs(request, inspect):
        raise ValueError("original source changed during composition")


def freeze(request, directory, inspect):
    chosen = selected(request)
    source_info = inputs(request, inspect)
    directory = Path(directory)
    results, inference, base_inference = {}, {}, {}
    for name, source in request["sources"].items():
        root = directory / name
        info = clean(root, inspect)
        if info["branch"] != "HEAD":
            raise ValueError("only detached candidate trees may be frozen")
        head = info["commit"]
        for ancestor in [
            source["base"],
            *(item["commit"] for item in chosen if item["repo"] == name),
        ]:
            git(root, "merge-base", "--is-ancestor", ancestor, head)
        components = [item["commit"] for item in chosen if item["repo"] == name]
        results[name] = {
            "base": source["base"],
            "components": components,
            "commit": head,
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "submodules": info["submodules"],
            "composition_commits": git(
                root, "rev-list", head, "--not", source["base"], *components
            ).splitlines(),
            "changed_paths": git(
                root, "diff", "--name-only", source["base"], head
            ).splitlines(),
        }
        inference[name] = closure(root, head, source["inference_paths"])
        base_inference[name] = closure(root, source["base"], source["inference_paths"])
    for item in chosen:
        for name, expected in item["requires"].items():
            if name in results:
                commit(directory / name, expected)
                git(
                    directory / name,
                    "merge-base",
                    "--is-ancestor",
                    expected,
                    results[name]["commit"],
                )
            elif name not in request["dependencies"] or fingerprint(
                request["dependencies"][name]
            ) != fingerprint(expected):
                raise ValueError("incompatible dependency constraint")
    identity = {
        "inference": inference,
        "flags": request["flags"],
        "dependencies": request["dependencies"],
        "comparison_controls": request["comparison_controls"],
    }
    candidate_id = fingerprint(identity)
    for record in request["evidence"].values():
        evidence(record, candidate_id)
    manifest = {
        "schema": 1,
        "candidate_id": candidate_id,
        "identity": identity,
        "comparison_fingerprint": fingerprint(request["comparison_controls"]),
        "unchanged": inference == base_inference
        and fingerprint(request["flags"]) == fingerprint(request["base_flags"]),
        "results": results,
        "roster": request["components"],
        "recipe": request["recipe"],
        "evidence": request["evidence"],
        "request": request,
        "provenance": {
            "sources": source_info,
            "harness": request["harness"],
            "tool_sha256": file_hash(__file__),
        },
    }
    manifest["bundle_fingerprint"] = fingerprint(manifest)
    return manifest


def write_manifest(path, manifest):
    with Path(path).open("x") as output:
        json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    Path(path).chmod(0o444)


def compose(request, destination, inspect):
    prepare(request, destination, inspect)
    manifest = freeze(request, destination, inspect)
    write_manifest(Path(destination) / "manifest.json", manifest)
    return manifest


def verify_manifest(manifest):
    if (
        manifest["schema"] != 1
        or fingerprint(
            {
                key: value
                for key, value in manifest.items()
                if key != "bundle_fingerprint"
            }
        )
        != manifest["bundle_fingerprint"]
        or fingerprint(manifest["identity"]) != manifest["candidate_id"]
    ):
        raise ValueError("manifest fingerprint differs")


def reproduce(manifest, destination, inspect):
    verify_manifest(manifest)
    request = manifest["request"]
    selected(request)
    inputs(request, inspect)
    destination = destination_directory(request, destination)
    for name, result in manifest["results"].items():
        root = destination / name
        git(
            request["sources"][name]["root"],
            "worktree",
            "add",
            "--detach",
            str(root),
            result["commit"],
        )
        if (
            git(root, "rev-parse", "HEAD^{tree}") != result["tree"]
            or closure(
                root, result["commit"], request["sources"][name]["inference_paths"]
            )
            != manifest["identity"]["inference"][name]
        ):
            raise ValueError("reproduced source differs")
        clean(root, inspect)
    write_manifest(destination / "manifest.json", manifest)
    return manifest


def require_measurable(manifest):
    verify_manifest(manifest)
    recipe = manifest["recipe"]
    reference(recipe["reference"])
    if recipe["qualification"] != "qualified" or not re.fullmatch(
        "[0-9a-f]{64}", recipe.get("fingerprint") or ""
    ):
        raise ValueError("daily recipe qualification is pending")
    evidence(recipe["review"], recipe["fingerprint"], required=True)
    for key in ("source_review", "numerical", "smoke"):
        evidence(manifest["evidence"][key], manifest["candidate_id"], required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compose", "freeze", "reproduce", "ready"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    if args.action == "ready":
        require_measurable(value)
        return
    if args.output is None:
        parser.error("output directory is required")
    request = value["request"] if args.action == "reproduce" else value
    inspect = harness(request["harness"])
    if args.action == "compose":
        compose(request, args.output, inspect)
    elif args.action == "reproduce":
        reproduce(value, args.output, inspect)
    else:
        write_manifest(
            args.output / "manifest.json", freeze(request, args.output, inspect)
        )


if __name__ == "__main__":
    main()
