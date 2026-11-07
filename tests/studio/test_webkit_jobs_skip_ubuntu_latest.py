# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""No workflow job that drives Playwright WebKit runs on ubuntu-latest.

ubuntu-latest now resolves to the ubuntu-26.04 image, where Playwright's WebKit exits at launch (TargetClosedError)
even after install-deps, and every Linux job that probes or drives it goes red. Those jobs pin ubuntu-24.04; a matrix
leg named ubuntu-latest keeps its check name and maps to ubuntu-24.04 in runs-on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
MAPPED = "${{ matrix.os == 'ubuntu-latest' && 'ubuntu-24.04' || matrix.os }}"


def _matrix_oses(job: dict) -> list[str]:
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    oses = list(matrix.get("os") or [])
    oses += [row["os"] for row in matrix.get("include") or [] if "os" in row]
    return oses


def runners(job: dict) -> list[str]:
    """The images a job lands on, resolving the two runs-on spellings used here."""
    runs_on = job.get("runs-on")
    if not isinstance(runs_on, str):
        return [json.dumps(runs_on)]
    if "matrix.os" not in runs_on:
        return [runs_on]
    oses = _matrix_oses(job)
    if runs_on.replace(" ", "") == "${{matrix.os}}":
        return oses
    if runs_on == MAPPED:
        return ["ubuntu-24.04" if os == "ubuntu-latest" else os for os in oses]
    pytest.fail(f"teach this guard the runs-on expression {runs_on!r}")


def webkit_jobs() -> list[tuple[str, str, dict]]:
    """Jobs that drive Playwright with WebKit as an engine; a file name such as test_webkit_*.py is not one."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        for name, job in (doc.get("jobs") or {}).items():
            text = json.dumps(job.get("steps") or []) + json.dumps(job.get("strategy") or {})
            if "playwright" in text and re.search(r"\bwebkit\b", text):
                found.append((path.name, name, job))
    return found


def test_the_guard_sees_the_webkit_jobs():
    names = {(wf, job) for wf, job, _ in webkit_jobs()}
    assert {
        ("studio-ui-smoke.yml", "ui-smoke"),
        ("studio-ui-smoke.yml", "ui-indicator"),
        ("studio-composer-compatibility.yml", "composer"),
        ("studio-inline-images.yml", "images"),
    } <= names, names


def test_no_webkit_job_runs_on_ubuntu_latest():
    offenders = [
        f"{wf}::{job}" for wf, job, spec in webkit_jobs() if "ubuntu-latest" in runners(spec)
    ]
    assert not offenders, (
        "these jobs drive Playwright WebKit on ubuntu-latest, whose ubuntu-26.04 image WebKit does not start on: "
        f"{offenders}. Pin ubuntu-24.04 (or map a matrix leg with {MAPPED})."
    )


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ({"runs-on": "ubuntu-latest"}, ["ubuntu-latest"]),
        ({"runs-on": "ubuntu-24.04"}, ["ubuntu-24.04"]),
        (
            {
                "runs-on": "${{ matrix.os }}",
                "strategy": {"matrix": {"os": ["ubuntu-latest", "macos-latest"]}},
            },
            ["ubuntu-latest", "macos-latest"],
        ),
        (
            {
                "runs-on": MAPPED,
                "strategy": {
                    "matrix": {"include": [{"os": "ubuntu-latest"}, {"os": "windows-latest"}]}
                },
            },
            ["ubuntu-24.04", "windows-latest"],
        ),
    ],
)
def test_runners_resolves_each_spelling(job, expected):
    assert runners(job) == expected
