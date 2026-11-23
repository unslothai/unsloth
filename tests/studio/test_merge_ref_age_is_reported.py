# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The merge-ref-age notice says a PR's merge ref may predate main's fix; it must never fail a job."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
ACTION = ROOT / ".github" / "actions" / "merge-ref-age" / "action.yml"

_USES = "./.github/actions/merge-ref-age"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]


def _jobs_that_run_pytest() -> list[str]:
    names = []
    for name, job in _jobs().items():
        if any("pytest" in str(step.get("run", "")) for step in job.get("steps", [])):
            names.append(name)
    return names


def test_some_job_here_runs_pytest():
    """Rule 1 below is vacuous if nothing matches it, which is how such rules die."""
    assert _jobs_that_run_pytest(), (
        f"{WORKFLOW.name} no longer has a job that runs pytest, so either the suites "
        f"moved and this guard should move with them, or it is reading the wrong file"
    )


@pytest.mark.parametrize("job_name", _jobs_that_run_pytest())
def test_every_pytest_job_reports_the_age_of_its_merge_ref(job_name):
    steps = _jobs()[job_name]["steps"]
    reporting = [step for step in steps if step.get("uses") == _USES]
    assert reporting, (
        f"job {job_name!r} in {WORKFLOW.name} runs pytest but does not use {_USES}. A "
        f"failure in it cannot tell the reader that the base branch moved since the "
        f"merge ref was built, which is how a fix that already landed gets debugged "
        f"a second time"
    )
    for step in reporting:
        condition = str(step.get("if", ""))
        assert "failure()" in condition, (
            f"job {job_name!r} runs {_USES} under {condition!r}. It is meant to speak "
            f"only when the job has already failed; on a green job the notice is noise "
            f"and the API call is waste"
        )


def test_reporting_the_age_cannot_fail_the_job():
    """It runs on the failure path, where a second error would bury the first one."""
    action = ACTION.read_text(encoding = "utf-8")
    body = yaml.safe_load(action)
    scripts = [str(step.get("run", "")) for step in body["runs"]["steps"]]
    assert any(scripts), f"{ACTION} has no script left to check"

    for script in scripts:
        code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))
        bad = re.findall(r"^\s*exit\s+(?!0\b)\S+", code, re.M)
        assert not bad, (
            f"{ACTION} exits non-zero ({bad}). It runs only when the job is already "
            f"failing, so a non-zero exit here adds a second error on top of the real "
            f"one and points the reader at the wrong thing"
        )
        assert "::error" not in code, (
            f"{ACTION} emits ::error::. Advisory only: a stale merge ref does not make "
            f"the failure untrue, and annotating it as an error says it does"
        )
