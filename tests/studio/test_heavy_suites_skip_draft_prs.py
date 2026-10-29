# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows, macOS and Tauri jobs skip draft pull requests, and marking a PR ready runs them.

Draft pushes were taking the scarcest runners for work nobody had asked to be tested yet.
Each such job carries GATE in its `if`, and its workflow lists ready_for_review so the
run that tests the PR starts when it leaves draft. Checked across every workflow, so a new
Windows or macOS job cannot be added without the gate.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

GATE = "github.event_name != 'pull_request' || !github.event.pull_request.draft"
# Runs only when a label is added, which is already an explicit request for the run.
EXEMPT = frozenset({"windows-installer-differential-ci.yml"})
# Every job, Linux build included: the suite is the Tauri shell, not an OS.
WHOLE = frozenset({"studio-tauri-smoke.yml"})
_HEAVY = re.compile(r"windows|macos", re.I)


def _documents() -> dict[str, dict]:
    out = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        document = yaml.safe_load(path.read_text(encoding = "utf-8"))
        if isinstance(document, dict):
            out[path.name] = document
    return out


def _pull_request(document: dict):
    triggers = document.get(True) or document.get("on") or {}
    if not isinstance(triggers, dict) or "pull_request" not in triggers:
        return None
    return triggers["pull_request"] or {}


def _labels(job: dict) -> list[str] | None:
    """The runner labels a job can land on, or None when its matrix is built at run time."""
    runs_on = str(job.get("runs-on"))
    found = re.search(r"matrix\.([\w-]+)", runs_on)
    if not found:
        return [runs_on]
    matrix = (job.get("strategy") or {}).get("matrix")
    if not isinstance(matrix, dict):
        return None
    values = matrix.get(found.group(1))
    if isinstance(values, list):
        return [str(value) for value in values]
    include = matrix.get("include")
    if isinstance(include, list) and all(
        isinstance(i, dict) and found.group(1) in i for i in include
    ):
        return [str(i[found.group(1)]) for i in include]
    return None


def _heavy(name: str, job: dict) -> bool:
    labels = _labels(job)
    if labels is None:
        # A matrix from a selector job: the job is named for the OS it runs on.
        return bool(re.match(r"(macos|windows)\b", name))
    return all(_HEAVY.search(label) for label in labels)


def _gated_jobs() -> dict[str, list[str]]:
    out = {}
    for filename, document in _documents().items():
        if filename in EXEMPT or _pull_request(document) is None:
            continue
        jobs = {n: j for n, j in (document.get("jobs") or {}).items() if isinstance(j, dict)}
        names = {
            n
            for n, j in jobs.items()
            if filename in WHOLE
            or (
                _heavy(n, j)
                and str(j.get("if", "")).strip() != "github.event_name != 'pull_request'"
            )
        }
        if names:
            out[filename] = sorted(names)
    return out


def test_the_scan_found_the_heavy_suites():
    """A rule that matched nothing would pass every check below."""
    gated = _gated_jobs()
    for filename in (
        "studio-windows-ui-smoke.yml",
        "studio-mac-ui-smoke.yml",
        "studio-tauri-smoke.yml",
    ):
        assert (
            filename in gated
        ), f"{filename} has no Windows/macOS job by this scan; the scan is wrong"


def test_windows_macos_and_tauri_jobs_skip_drafts():
    documents = _documents()
    missing = [
        f"{filename}::{name}"
        for filename, names in _gated_jobs().items()
        for name in names
        if GATE not in str(documents[filename]["jobs"][name].get("if", ""))
    ]
    assert not missing, (
        f"{missing} run on draft pull requests. Add `{GATE}` to the job's `if` "
        f"(AND it with any condition already there)."
    )


def test_marking_a_pull_request_ready_starts_the_skipped_jobs():
    documents = _documents()
    missing = []
    for filename in _gated_jobs():
        pull_request = _pull_request(documents[filename])
        types = pull_request.get("types") if isinstance(pull_request, dict) else None
        if not types or "ready_for_review" not in types:
            missing.append(filename)
    assert not missing, (
        f"{missing} skip draft pull requests but do not list ready_for_review, so leaving "
        f"draft starts nothing and the jobs only run on the next push."
    )


def test_ready_for_review_kept_the_default_events():
    """Listing types replaces the defaults, so opened, synchronize and reopened must stay."""
    documents = _documents()
    dropped = {}
    for filename in _gated_jobs():
        types = set(_pull_request(documents[filename]).get("types") or ())
        lost = {"opened", "synchronize", "reopened"} - types
        if lost:
            dropped[filename] = sorted(lost)
    assert not dropped, f"these workflows stopped running on {dropped}"
