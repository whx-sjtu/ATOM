import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def save_commit(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    git(root, "add", ".")
    git(root, "commit", "-qm", text)
    return git(root, "rev-parse", "HEAD")


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def request_spec(tmp_path):
    sources = {}
    for name, remote in (("atom", "ATOM"), ("aiter", "aiter")):
        root = tmp_path / name
        root.mkdir()
        git(root, "init", "-q", "-b", "main")
        git(root, "config", "user.name", "Test")
        git(root, "config", "user.email", "test@example.invalid")
        git(
            root, "remote", "add", "origin", f"https://github.com/whx-sjtu/{remote}.git"
        )
        base = save_commit(root, name + "/core.py", "base")
        sources[name] = {"root": str(root), "base": base, "inference_paths": [name]}
    proof = tmp_path / "evidence.json"
    proof.write_text('{"fixture": "explicit external evidence"}')
    common = tmp_path / "common.py"
    common.write_text(
        "import subprocess\n"
        "def git_info(root):\n"
        " def git(*args):\n"
        "  return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()\n"
        " return {'root':str(root),'commit':git('rev-parse','HEAD'),"
        "'branch':git('rev-parse','--abbrev-ref','HEAD'),"
        "'dirty_files':git('status','--porcelain','--untracked-files=no').splitlines(),"
        "'submodules':git('submodule','status')}\n"
    )
    return {
        "sources": sources,
        "components": [],
        "flags": {},
        "base_flags": {},
        "dependencies": {"abi": "fixture-abi", "runtime": "fixture-runtime"},
        "comparison_controls": {"concurrency": 12, "duration_s": 3600},
        "recipe": {
            "reference": ref(proof),
            "qualification": "pending",
            "fingerprint": None,
        },
        "evidence": {
            key: {"status": "pending"}
            for key in ("source_review", "numerical", "smoke", "quality", "performance")
        },
        "harness": ref(common),
    }


@pytest.fixture
def candidate():
    return importlib.import_module("tools.k31_daily_candidate")


def component(spec, name, text):
    root = Path(spec["sources"][name]["root"])
    git(root, "checkout", "-qb", text)
    revision = save_commit(root, name + "/core.py", text)
    git(root, "checkout", "-q", "main")
    proof = {
        "status": "passed",
        "subject": revision,
        "reference": spec["recipe"]["reference"],
    }
    return {
        "id": text,
        "repo": name,
        "commit": revision,
        "include": True,
        "kind": "inference",
        "implemented": True,
        "flags": {},
        "requires": {},
        "evidence": {"source_review": proof, "numerical": proof},
    }


def test_cli_empty_candidate(request_spec, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(request_spec))
    tool = Path(__file__).resolve().parents[1] / "tools/k31_daily_candidate.py"
    result = subprocess.run(
        [sys.executable, str(tool), "compose", str(spec), str(tmp_path / "result")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "result/manifest.json").read_text())
    assert manifest["unchanged"] and not manifest["roster"]


def test_exact_reproduction_and_pending_admission(candidate, request_spec, tmp_path):
    inspect = candidate.harness(request_spec["harness"])
    output = tmp_path / "first"
    before = {
        name: inspect(source["root"])
        for name, source in request_spec["sources"].items()
    }
    manifest = candidate.compose(request_spec, output, inspect)
    candidate.reproduce(manifest, tmp_path / "second", inspect)
    assert (output / "manifest.json").read_bytes() == (
        tmp_path / "second/manifest.json"
    ).read_bytes()
    assert before == {
        name: inspect(source["root"])
        for name, source in request_spec["sources"].items()
    }
    with pytest.raises(ValueError, match="recipe"):
        candidate.require_measurable(manifest)
    with pytest.raises(FileExistsError):
        candidate.write_manifest(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="inside an original"):
        candidate.reproduce(
            manifest, Path(request_spec["sources"]["atom"]["root"]) / "bad", inspect
        )


def test_unmerged_component_status_and_tool_identity(candidate, request_spec, tmp_path):
    item = component(request_spec, "atom", "feature")
    request_spec["components"] = [item]
    inspect = candidate.harness(request_spec["harness"])
    output = tmp_path / "candidate"
    first = candidate.compose(request_spec, output, inspect)
    assert (output / "atom/atom/core.py").read_text() == "feature"
    assert not first["unchanged"] and first["results"]["atom"]["components"] == [
        item["commit"]
    ]
    assert (
        git(Path(request_spec["sources"]["atom"]["root"]), "rev-parse", "HEAD")
        == request_spec["sources"]["atom"]["base"]
    )
    updated = copy.deepcopy(request_spec)
    for key in ("source_review", "numerical", "smoke"):
        updated["evidence"][key] = {
            "status": "passed",
            "subject": first["candidate_id"],
            "reference": updated["recipe"]["reference"],
        }
    updated["recipe"].update(
        qualification="qualified",
        fingerprint="a" * 64,
        review={
            "status": "passed",
            "subject": "a" * 64,
            "reference": updated["recipe"]["reference"],
        },
    )
    second = candidate.freeze(updated, output, inspect)
    assert second["candidate_id"] == first["candidate_id"]
    assert second["bundle_fingerprint"] != first["bundle_fingerprint"]
    candidate.require_measurable(second)
    assert second["evidence"]["quality"]["status"] == "pending"
    second["evidence"]["smoke"]["status"] = "failed"
    with pytest.raises(ValueError, match="fingerprint"):
        candidate.require_measurable(second)


def test_tool_only_base_change_not_a_new_candidate(candidate, request_spec, tmp_path):
    inspect = candidate.harness(request_spec["harness"])
    first = candidate.compose(request_spec, tmp_path / "first", inspect)
    source = request_spec["sources"]["atom"]
    source["base"] = save_commit(
        Path(source["root"]), "tools/controller.py", "tool-only"
    )
    second = candidate.compose(request_spec, tmp_path / "second", inspect)
    assert first["candidate_id"] == second["candidate_id"]
    assert first["results"]["atom"]["commit"] != second["results"]["atom"]["commit"]
    assert first["bundle_fingerprint"] != second["bundle_fingerprint"]
    request_spec["comparison_controls"]["concurrency"] = 16
    third = candidate.compose(request_spec, tmp_path / "third", inspect)
    assert third["candidate_id"] != second["candidate_id"]
    assert third["comparison_fingerprint"] != second["comparison_fingerprint"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_review",
        "wrong_subject",
        "incompatible",
        "wrong_flags",
        "missing_null_flag",
        "wrong_flag_type",
        "omitted_ready",
    ],
)
def test_inference_refusals(candidate, request_spec, tmp_path, failure):
    item = component(request_spec, "atom", failure)
    request_spec["components"] = [item]
    if failure == "missing_review":
        item["evidence"]["source_review"] = {"status": "pending"}
    elif failure == "wrong_subject":
        item["evidence"]["numerical"] = {
            **item["evidence"]["numerical"],
            "subject": "0" * 40,
        }
    elif failure == "incompatible":
        item["requires"] = {"abi": "other"}
    elif failure == "wrong_flags":
        item["flags"] = {"ENABLE": "1"}
    elif failure == "missing_null_flag":
        item["flags"] = {"--enable": None}
    elif failure == "wrong_flag_type":
        item["flags"] = {"enable": 1}
        request_spec["flags"] = {"enable": True}
    else:
        item.update(
            include=False,
            reason="microbenchmark noise investigation pending",
            omission_evidence=request_spec["recipe"]["reference"],
        )
    with pytest.raises(ValueError):
        candidate.compose(
            request_spec,
            tmp_path / "result",
            candidate.harness(request_spec["harness"]),
        )


def test_aiter_constraints_and_non_inference_roster(candidate, request_spec, tmp_path):
    item = component(request_spec, "aiter", "aiter-feature")
    request_spec["components"] = [item]
    inspect = candidate.harness(request_spec["harness"])
    with pytest.raises(ValueError, match="AITER"):
        candidate.compose(request_spec, tmp_path / "invalid", inspect)
    item["requires"] = {
        **request_spec["dependencies"],
        "atom": request_spec["sources"]["atom"]["base"],
    }
    candidate.compose(request_spec, tmp_path / "valid", inspect)
    item.update(
        include=False,
        kind="operator",
        reason="standalone operator, no inference wiring",
        omission_evidence=request_spec["recipe"]["reference"],
    )
    omitted = candidate.compose(request_spec, tmp_path / "omitted", inspect)
    assert omitted["unchanged"] and omitted["roster"][0]["reason"]
    item.update(implemented=False, commit=None, reason="proposal not implemented")
    unimplemented = candidate.compose(request_spec, tmp_path / "unimplemented", inspect)
    assert unimplemented["unchanged"]
    item["include"] = True
    with pytest.raises(ValueError, match="implemented inference"):
        candidate.compose(request_spec, tmp_path / "bad", inspect)


@pytest.mark.parametrize("untracked", [False, True])
def test_dirty_original_refused(candidate, request_spec, tmp_path, untracked):
    root = Path(request_spec["sources"]["atom"]["root"])
    (root / ("unknown.py" if untracked else "atom/core.py")).write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        candidate.compose(
            request_spec,
            tmp_path / "result",
            candidate.harness(request_spec["harness"]),
        )


def test_conflict_resolution_stays_isolated(candidate, request_spec, tmp_path):
    request_spec["components"] = [
        component(request_spec, "atom", name) for name in ("left", "right")
    ]
    inspect = candidate.harness(request_spec["harness"])
    original = request_spec["sources"]["atom"]["base"]
    output = tmp_path / "conflicted"
    with pytest.raises(ValueError):
        candidate.compose(request_spec, output, inspect)
    assert (
        git(Path(request_spec["sources"]["atom"]["root"]), "rev-parse", "HEAD")
        == original
    )
    save_commit(output / "atom", "atom/core.py", "left plus right")
    manifest = candidate.freeze(request_spec, output, inspect)
    assert manifest["results"]["atom"]["composition_commits"]
    assert manifest["evidence"]["source_review"]["status"] == "pending"
    candidate.write_manifest(output / "manifest.json", manifest)
    candidate.reproduce(manifest, tmp_path / "resolved-copy", inspect)
    assert (
        tmp_path / "resolved-copy/atom/atom/core.py"
    ).read_text() == "left plus right"


def test_changed_evidence_refused(candidate, request_spec, tmp_path):
    Path(request_spec["recipe"]["reference"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="evidence reference"):
        candidate.compose(
            request_spec,
            tmp_path / "result",
            candidate.harness(request_spec["harness"]),
        )
