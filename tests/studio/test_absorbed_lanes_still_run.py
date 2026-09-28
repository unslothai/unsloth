# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Work absorbed onto a shared runner still runs, and still fails its job when it fails.

Two workflows each held their own runner slot on every commit for a few seconds of
read-only checking:

    Unsloth load-orchestrator CI :: test    ~33 s, its own slot
    Lockfile supply-chain audit :: audit     ~6 s, its own slot

Both now run as background lanes inside `Lint CI`, which has no path filter and was
already going to occupy a runner on every commit. Absorbing narrower-triggered work into
an unfiltered job can only reduce the slots a commit takes, and running the lanes in the
background rather than as extra steps means they overlap the ~65 s of lint instead of
being appended to it.

Backgrounding is what makes this worth guarding. Three things go silently wrong with it,
and none of them turns a job red on its own:

  * the lane is launched but never collected, so a failure is invisible and the absorbed
    job has effectively been deleted rather than moved;
  * the lane never starts, and the collect step reads a missing exit status as success;
  * the launch blocks on the lane's output instead of returning, so the overlap the whole
    design buys quietly disappears and the job just gets slower.

The payloads live in `.github/scripts/lane-*.sh` so the standalone workflows and the Lint
CI lanes cannot drift apart. That single-definition property is asserted here too, since
the obvious "fix" when a lane breaks is to inline it back into the workflow.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
SCRIPTS = REPO / ".github" / "scripts"

LINT_CI = WORKFLOWS / "lint-ci.yml"

# lane name -> (shared script, the workflow that keeps a standalone copy of the job)
ABSORBED = {
    "load-orchestrator": ("lane-load-orchestrator.sh", "studio-load-orchestrator-ci.yml"),
    "lockfile-audit": ("lane-lockfile-audit.sh", "lockfile-audit.yml"),
}


def _lint_steps():
    doc = yaml.safe_load(LINT_CI.read_text(encoding = "utf-8"))
    return doc["jobs"]["source-lint"]["steps"]


def _step(fragment: str) -> dict:
    for step in _lint_steps():
        if fragment.lower() in (step.get("name") or "").lower():
            return step
    raise AssertionError(f"no step in lint-ci.yml source-lint is named like {fragment!r}")


@pytest.mark.parametrize("lane", sorted(ABSORBED))
def test_each_absorbed_lane_is_launched(lane):
    run = str(_step("Start the absorbed").get("run", ""))
    script = ABSORBED[lane][0]
    assert f"lane {lane} " in run, f"the {lane} lane is no longer launched by Lint CI"
    assert script in run, (
        f"the {lane} lane no longer runs {script}. The point of the shared script is that "
        f"the standalone workflow and this lane cannot diverge."
    )


@pytest.mark.parametrize("lane", sorted(ABSORBED))
def test_each_absorbed_lane_is_collected(lane):
    """A launched-but-uncollected lane is a deleted test that looks like a passing one."""
    run = str(_step("Collect the absorbed").get("run", ""))
    assert lane in run, (
        f"Lint CI launches the {lane} lane but never collects it, so nothing reads its "
        f"exit status and a failure leaves the job green"
    )


def test_the_collect_step_runs_even_when_a_lint_step_failed():
    """Without always(), a lint failure hides every lane result behind it."""
    step = _step("Collect the absorbed")
    assert "always()" in str(step.get("if", "")), (
        "the collect step is not if: always(), so a lint failure above it skips the lane "
        "results entirely and a lane regression surfaces only once the lint is fixed"
    )


def test_a_failing_lane_fails_the_job():
    run = str(_step("Collect the absorbed").get("run", ""))
    assert re.search(r"exit\s+\"?\$", run), (
        "the collect step never propagates a non-zero lane status, so the absorbed suites "
        "run but cannot fail anything"
    )


def test_a_lane_that_never_finishes_is_a_failure_not_a_pass():
    """The sharp edge of a sentinel file: absence must not read as success."""
    run = str(_step("Collect the absorbed").get("run", ""))
    assert "::error::" in run and "never finished" in run, (
        "the collect step does not treat a missing exit-status file as a failure. A lane "
        "that was killed, or never started, would then be indistinguishable from one that "
        "passed."
    )


def test_the_launch_detaches_from_the_steps_output():
    """Measured, not theoretical: without this the launch blocks for the lane's duration.

    A background child inherits the step's stdout and stderr pipes, and the step is not
    considered finished while a writer still holds them. Locally, launching a 4 s lane
    took 4.0 s before the redirect and 0.0 s after. The lanes would still run and still be
    collected, so nothing would go red -- the job would just quietly stop overlapping them
    and get slower, which is the entire benefit gone.
    """
    run = str(_step("Start the absorbed").get("run", ""))
    assert re.search(r"\)\s*<\s*/dev/null\s*>\s*/dev/null\s*2>&1\s*&", run), (
        "the background lanes are not detached from the step's stdout/stderr, so the "
        "launch step blocks until they finish and the overlap is lost"
    )


@pytest.mark.parametrize("lane", sorted(ABSORBED))
def test_the_payload_has_exactly_one_definition(lane):
    """The standalone workflow must call the same script, not a copy of its commands."""
    script, workflow = ABSORBED[lane]
    assert (SCRIPTS / script).exists(), f"{script} is gone; the lane and the workflow will drift"
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding = "utf-8"))
    runs = "\n".join(
        str(step.get("run", ""))
        for job in doc["jobs"].values()
        for step in job.get("steps") or []
        if isinstance(step, dict)
    )
    assert script in runs, (
        f"{workflow} no longer calls {script}, so it has its own copy of the commands and "
        f"the two definitions can diverge without anything noticing"
    )


@pytest.mark.parametrize("lane", sorted(ABSORBED))
def test_the_absorbed_workflow_no_longer_takes_a_slot_per_commit(lane):
    """Absorbing without removing the original trigger doubles the work instead of moving it."""
    workflow = ABSORBED[lane][1]
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding = "utf-8"))
    on = doc.get(True) if True in doc else doc.get("on")
    still_per_commit = sorted(k for k in on if k in ("pull_request", "push"))
    assert not still_per_commit, (
        f"{workflow} still triggers on {still_per_commit} while Lint CI also runs its "
        f"work, so the commit now pays for both. Absorbing is only a saving if the "
        f"original per-commit trigger goes."
    )


def test_the_nightly_lockfile_audit_survived():
    """The schedule is a different check from the per-commit one and must not be lost.

    A commit-triggered audit reads the lockfiles against advisories known at commit time.
    The nightly one re-reads the same lockfiles against advisories published since, which
    no commit run can do. Removing the per-commit trigger must not take that with it.
    """
    doc = yaml.safe_load((WORKFLOWS / "lockfile-audit.yml").read_text(encoding = "utf-8"))
    on = doc.get(True) if True in doc else doc.get("on")
    assert on.get("schedule"), (
        "lockfile-audit.yml lost its nightly schedule. That run catches advisories "
        "published after the last commit, which the Lint CI lane cannot."
    )
