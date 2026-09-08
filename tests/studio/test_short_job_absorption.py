# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two ubuntu-latest jobs stopped taking a runner slot. Neither may stop being checked.

Measured over 400 completed main push runs, both were pure slot overhead: seconds of
execution behind a queue of about three hours.

    Security audit :: pytest tests/security         72 s exec, 11096 s queue
    Unsloth export capability :: capability (ubuntu-latest)
                                                    67 s exec, 10642 s queue

They were dealt with differently, because they are different problems.

The security suite MOVED, into `Workflow trigger lint`. What makes that safe is that the
host's harden-runner block is identical to the one the job carried, so the suite runs
under exactly the isolation it had. The obvious alternative host, `Lint CI`, is where the
lockfile and load-orchestrator lanes went, and it is wrong for this one: Lint CI installs
shellcheck from apt, so its runner has to permit escalation and an apt mirror. Absorbing a
security gate there would weaken it. That is the property asserted here, not the fact of
the move.

The capability leg was DELETED, because it was already duplicated: Backend CI runs
`pytest tests/` without ignoring `tests/test_export_capability.py`, so the file executes
there on ubuntu-latest either way. That justification is only true while the ignore list
stays as it is, and adding one line to it would silently delete the coverage rather than
turn anything red. So the ignore list is what gets asserted.

Both are the same failure shape: a change that looks unrelated makes a check stop running
without failing anything.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

TRIGGER_LINT = WORKFLOWS / "workflow-trigger-lint.yml"
SECURITY_AUDIT = WORKFLOWS / "security-audit.yml"
CAPABILITY = WORKFLOWS / "studio-export-capability-ci.yml"
BACKEND_CI = WORKFLOWS / "studio-backend-ci.yml"


def _doc(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding = "utf-8")) or {}


def _steps(path: Path):
    """(job name, step dict) for every step in the workflow."""
    for job_name, job in (_doc(path).get("jobs") or {}).items():
        if isinstance(job, dict):
            for step in job.get("steps") or []:
                if isinstance(step, dict):
                    yield job_name, step


def _harden_runner(path: Path, job_name: str) -> dict:
    for name, step in _steps(path):
        if name == job_name and "harden-runner" in str(step.get("uses", "")):
            return step.get("with") or {}
    return {}


def _endpoints(config: dict) -> set[str]:
    return set(str(config.get("allowed-endpoints", "")).split())


def _runs_security_suite(path: Path) -> list[str]:
    return [
        f"{job}::{step.get('name', '<unnamed>')}"
        for job, step in _steps(path)
        if "tests/security" in str(step.get("run", ""))
    ]


def test_the_security_suite_still_runs_somewhere() -> None:
    """A move that drops the step is indistinguishable from a move that lands it."""
    assert _runs_security_suite(TRIGGER_LINT), (
        "no step in workflow-trigger-lint.yml runs `pytest tests/security`. It was moved "
        "there off its own runner; if it moved again, assert it at its new home rather "
        "than deleting this."
    )


def test_the_security_suite_is_not_left_running_twice() -> None:
    """Absorbing work is only a saving if the original stops taking a slot."""
    assert not _runs_security_suite(SECURITY_AUDIT), (
        "security-audit.yml runs `pytest tests/security` again. It runs in "
        "workflow-trigger-lint.yml now, so this is a second runner for the same work."
    )


def test_the_absorbing_job_did_not_weaken_the_isolation() -> None:
    """The whole reason this host was chosen over Lint CI.

    The suite carried `egress-policy: block`, `disable-sudo: true` and six endpoints.
    Landing it on a runner that allows more than that is a downgrade wearing the costume
    of a cleanup, and nothing else would report it.
    """
    original = {
        "api.github.com:443",
        "github.com:443",
        "codeload.github.com:443",
        "objects.githubusercontent.com:443",
        "pypi.org:443",
        "files.pythonhosted.org:443",
    }
    config = _harden_runner(TRIGGER_LINT, "workflow-trigger-lint")
    assert config, "the absorbing job has no harden-runner step at all"
    assert config.get("egress-policy") == "block", config.get("egress-policy")
    assert config.get("disable-sudo") is True, config.get("disable-sudo")
    widened = sorted(_endpoints(config) - original)
    assert not widened, (
        f"the job now reaches endpoints the security suite's own runner did not: {widened}. "
        "Moving the suite here is only free while the policy matches; if this host needs "
        "a wider allowlist, the suite has to move somewhere else, not travel with it."
    )


def test_the_absorbing_job_stays_unfiltered() -> None:
    """It is also the host that makes the suite run MORE often, not less.

    security-audit.yml's pull_request is path-filtered. This workflow's is not, and must
    never be -- see its header and scripts/lint_workflow_triggers.py.
    """
    triggers = _doc(TRIGGER_LINT).get(True) or _doc(TRIGGER_LINT).get("on") or {}
    pull_request = triggers.get("pull_request")
    assert not isinstance(pull_request, dict) or not (
        set(pull_request) & {"paths", "paths-ignore"}
    ), (
        "workflow-trigger-lint.yml's pull_request gained a paths filter. That is forbidden "
        "on its own terms, and it would also make the absorbed security suite run less "
        "often than it did in security-audit.yml."
    )


def test_the_capability_job_kept_the_leg_nothing_else_covers() -> None:
    """Windows is the only platform running this file; ubuntu was the duplicate."""
    matrix = _doc(CAPABILITY)["jobs"]["capability"]["strategy"]["matrix"]["os"]
    assert "windows-latest" in matrix, (
        f"the export capability job no longer runs on Windows: {matrix}. Nothing else in "
        "CI runs tests/test_export_capability.py there, and the per-OS import probe is "
        "what the job exists for."
    )


def test_backend_ci_still_runs_the_file_the_ubuntu_leg_used_to() -> None:
    """The ubuntu leg was dropped BECAUSE Backend CI covers it. Keep that true.

    One line added to that job's --ignore list would remove the coverage on every
    platform at once, and no test would fail.
    """
    text = BACKEND_CI.read_text(encoding = "utf-8")
    assert "test_export_capability" not in text, (
        "studio-backend-ci.yml now names tests/test_export_capability.py, which almost "
        "certainly means it is being ignored or deselected. The ubuntu leg of "
        "studio-export-capability-ci.yml was removed because this job runs that file; "
        "restore the leg, or the file stops being checked on Linux entirely."
    )
    assert "pytest tests/" in text, (
        "studio-backend-ci.yml no longer runs the whole tests/ tree, so it can no longer "
        "be relied on to cover tests/test_export_capability.py"
    )
