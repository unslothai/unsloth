# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failing pytest job says how old its merge ref is, and saying so cannot fail the job.

A `pull_request` run tests `refs/pull/N/merge`, a merge of the branch into the base as
the base stood when the run was created. GitHub does not rebuild that ref because the
base later moved, and a re-run replays the same merge commit, so a fix that lands on
main after a run starts is invisible to that run for as long as the branch sits still.

That cost real time: #11372 added `encoding = "utf-8"` to three test files and merged at
14:35 UTC, and runs created at 13:39, 14:26 and 14:29 on #11375, #11357 and #11373 went
on failing `tests/test_source_read_encoding.py` on exactly those three lines. All three
branches had never touched the files.

`.github/actions/merge-ref-age` prints a notice that says so. What this file keeps is
the two properties that make it worth having, because both are easy to lose:

  1. Every job here that runs pytest carries it, and carries it gated on `failure()`.
     A job added later that does not is a job where the same confusion returns.

  2. It cannot fail a job. It runs exactly when a job is already failing, so a bug in
     it would land as a second, misleading error on top of the real one. The action's
     script therefore never exits non-zero and never emits `::error::`.

It deliberately does NOT check the wording of the notice. The message is prose, prose
guards have to parse English to be right, and the value here is that the step exists and
is harmless, not that any particular sentence survives.
"""

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
        # Comments explain the exits; the code is what runs.
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
