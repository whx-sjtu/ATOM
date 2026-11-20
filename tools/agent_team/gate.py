"""Run CPU tests, formatting and the reviewed Ruff diagnostic allowance."""

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("ruff_baseline.csv")
BASELINE_SOURCE = "cbf1617468110a81e720a00021c5d3c5e14ce232"
RUFF_VERSION = "0.17.6"
APPROVED_BASELINE_SHA256 = (
    "b0a2a3f320de5f5b7fe479e9f4edfda464ef1cf9bc85bf81219da2c21e8c7883"
)
FIELDS = ("path", "code", "message", "source", "count")


class GateError(Exception):
    """A failed check, malformed evidence or unavailable tool."""


def parse_baseline(text):
    rows = csv.reader(io.StringIO(text), strict=True)
    if next(rows, None) != list(FIELDS):
        raise GateError("Invalid Ruff baseline header")
    findings = Counter()
    for row in rows:
        if len(row) != len(FIELDS):
            raise GateError("Invalid Ruff baseline row")
        path, code, message, source, count = row
        relative = PurePosixPath(path)
        if (
            not path
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in path
            or relative.as_posix() != path
            or re.fullmatch(r"[A-Z]+[0-9]{3,4}", code) is None
            or not message
            or not source
            or re.fullmatch(r"[1-9][0-9]*", count) is None
        ):
            raise GateError("Malformed Ruff baseline finding")
        identity = (path, code, message, source)
        if identity in findings:
            raise GateError("Duplicate Ruff baseline finding")
        findings[identity] = int(count)
    return findings


def load_baseline(path=BASELINE):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != APPROVED_BASELINE_SHA256:
        raise GateError(
            "Ruff allowance changed; review the CSV and its approved fingerprint"
        )
    return parse_baseline(data.decode("utf-8"))


def finding_identities(root, diagnostics):
    if not isinstance(diagnostics, list):
        raise GateError("Ruff JSON must be a diagnostic list")
    findings = Counter()
    lines = {}
    for item in diagnostics:
        if not isinstance(item, dict):
            raise GateError("Malformed Ruff diagnostic")
        try:
            path = Path(item["filename"]).resolve()
            relative = path.relative_to(root.resolve()).as_posix()
            code, message = item["code"], item["message"]
            row = item["location"]["row"]
            if (
                not isinstance(code, str)
                or re.fullmatch(r"[A-Z]+[0-9]{3,4}", code) is None
                or not isinstance(message, str)
                or not message
                or type(row) is not int
                or row < 1
                or item.get("cell") is not None
            ):
                raise GateError("Malformed or unsupported Ruff diagnostic")
            if path not in lines:
                lines[path] = path.read_text(encoding="utf-8").splitlines()
            source = lines[path][row - 1]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise GateError(
                "Malformed Ruff diagnostic or path outside repository"
            ) from exc
        findings[(relative, code, message, source)] += 1
    return findings


def ruff_findings(root):
    version = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode or version.stdout.strip() != f"ruff {RUFF_VERSION}":
        raise GateError(
            f"Ruff {RUFF_VERSION} required: {version.stdout}{version.stderr}"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format=json",
            ".",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise GateError(f"Ruff tool failed ({result.returncode}): {result.stderr}")
    try:
        diagnostics = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GateError("Ruff returned malformed JSON") from exc
    findings = finding_identities(root, diagnostics)
    if bool(findings) != bool(result.returncode):
        raise GateError("Ruff exit status disagrees with its diagnostics")
    return findings


def compare_findings(actual, approved):
    unexpected = actual - approved
    resolved = approved - actual
    for title, findings in (
        ("New Ruff findings", unexpected),
        ("Resolved allowances to remove", resolved),
    ):
        for (path, code, message, source), count in sorted(findings.items()):
            print(f"{title}: {path}: {code}: {message} ({count})\n    {source}")
    if unexpected or resolved:
        raise GateError("Ruff differs from the reviewed baseline")


def run_stage(name, command, root, env):
    print(f"Running {name}", flush=True)
    result = subprocess.run(command, cwd=root, env=env, check=False)
    if result.returncode:
        raise GateError(f"{name} failed ({result.returncode})")


def main():
    try:
        approved = load_baseline()
        env = dict(os.environ)
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        if not env.get("UNIT_TEST_REPORT"):
            env["UNIT_TEST_REPORT"] = str(
                Path(tempfile.mkdtemp(prefix="atom-agent-team-")) / "unit-report.xml"
            )
        print(f"Ruff {RUFF_VERSION}; allowance source {BASELINE_SOURCE}", flush=True)
        run_stage("Black", [sys.executable, "-m", "black", "--check", "."], ROOT, env)
        actual = ruff_findings(ROOT)
        compare_findings(actual, approved)
        print(
            f"Ruff baseline matched: {sum(actual.values())} existing diagnostics; "
            "no new or resolved allowances",
            flush=True,
        )
        run_stage(
            "CPU tests",
            ["bash", ".github/scripts/run_unit_tests.sh"],
            ROOT,
            env,
        )
        print(f"CPU report: {env['UNIT_TEST_REPORT']}", flush=True)
        print("Gate passed: CPU tests, Black and reviewed Ruff baseline", flush=True)
        return 0
    except (GateError, OSError, UnicodeError, csv.Error) as exc:
        print(f"Gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
