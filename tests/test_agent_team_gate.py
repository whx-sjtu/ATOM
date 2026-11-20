import csv
import hashlib
import importlib.util
import io
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agent_team_gate", ROOT / "tools/agent_team/gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def project(tmp_path):
    # Real repository lint configuration, with a small source tree to inspect.
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    (tmp_path / "probe.py").write_text("import os\n")
    return tmp_path


def baseline_csv(findings):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(gate.FIELDS)
    writer.writerows((*key, count) for key, count in sorted(findings.items()))
    return buffer.getvalue()


def test_existing_ruff_rules_reject_new_undefined_name(project):
    approved = gate.ruff_findings(project)
    assert any(key[1] == "F401" for key in approved)
    gate.compare_findings(approved, gate.parse_baseline(baseline_csv(approved)))
    with (project / "probe.py").open("a") as stream:
        stream.write("\n\ndef value():\n    return task_two_missing_name\n")
    actual = gate.ruff_findings(project)
    assert any(key[1] == "F821" for key in actual)
    with pytest.raises(gate.GateError):
        gate.compare_findings(actual, approved)


def test_existing_diagnostics_survive_line_insertions(project):
    before = gate.ruff_findings(project)
    path = project / "probe.py"
    path.write_text("# Unrelated introductory comment.\n" + path.read_text())
    assert gate.ruff_findings(project) == before


def test_resolved_allowances_must_be_removed(project):
    approved = gate.ruff_findings(project)
    (project / "probe.py").write_text("value = 1\n")
    with pytest.raises(gate.GateError):
        gate.compare_findings(gate.ruff_findings(project), approved)


def test_duplicate_diagnostic_growth_is_rejected(project):
    approved = gate.ruff_findings(project)
    actual = approved.copy()
    actual[next(iter(actual))] += 1
    with pytest.raises(gate.GateError):
        gate.compare_findings(actual, approved)


@pytest.mark.parametrize(
    "text",
    [
        "unexpected,header\n",
        "path,code,message,source,count\nprobe.py,F821,message,source\n",
        "path,code,message,source,count\n../probe.py,F821,message,source,1\n",
        "path,code,message,source,count\nprobe.py,F821,message,source,0\n",
        (
            "path,code,message,source,count\nprobe.py,F821,message,source,1\n"
            "probe.py,F821,message,source,1\n"
        ),
        'path,code,message,source,count\n"unterminated',
    ],
)
def test_malformed_baseline_is_rejected(text):
    with pytest.raises((gate.GateError, csv.Error)):
        gate.parse_baseline(text)


def test_allowance_changes_require_review(tmp_path):
    path = tmp_path / "baseline.csv"
    original = gate.BASELINE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == gate.APPROVED_BASELINE_SHA256
    path.write_bytes(original + b"probe.py,F821,message,source,1\n")
    with pytest.raises(gate.GateError, match="allowance changed"):
        gate.load_baseline(path)


@pytest.mark.parametrize("version", ["ruff 0.17.5\n", "ruff invalid\n"])
def test_unapproved_ruff_version_fails(monkeypatch, tmp_path, version):
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, version, ""),
    )
    with pytest.raises(gate.GateError, match="required"):
        gate.ruff_findings(tmp_path)


@pytest.mark.parametrize(
    "status, output", [(2, "[]"), (0, "bad JSON"), (0, "{}"), (1, "[]")]
)
def test_ruff_tool_errors_fail(monkeypatch, tmp_path, status, output):
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, f"ruff {gate.RUFF_VERSION}\n", ""),
            subprocess.CompletedProcess([], status, output, "tool diagnostic"),
        ]
    )
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(gate.GateError):
        gate.ruff_findings(tmp_path)


def test_malformed_diagnostic_fails(tmp_path):
    with pytest.raises(gate.GateError):
        gate.finding_identities(tmp_path, [{"filename": "/outside.py"}])


def test_gate_runs_existing_cpu_entrypoint(monkeypatch, tmp_path):
    stages = []
    monkeypatch.setenv("UNIT_TEST_REPORT", str(tmp_path / "report.xml"))
    monkeypatch.setattr(gate, "load_baseline", Counter)
    monkeypatch.setattr(gate, "ruff_findings", lambda root: Counter())
    monkeypatch.setattr(
        gate, "run_stage", lambda name, command, root, env: stages.append(command)
    )
    assert gate.main() == 0
    assert stages[-1] == ["bash", ".github/scripts/run_unit_tests.sh"]
    assert "--check" in stages[0]


def test_failed_stage_and_missing_tool_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 4),
    )
    with pytest.raises(gate.GateError, match="failed"):
        gate.run_stage("CPU tests", ["bash"], tmp_path, {})
    monkeypatch.setattr(gate, "load_baseline", lambda: (_ for _ in ()).throw(OSError()))
    assert gate.main() == 1
