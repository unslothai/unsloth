# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Workflow-reading guards must run in workflow-trigger-lint, the one job a path filter cannot skip."""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "studio"
LINT = REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"

# Modules that read a workflow file but cannot run in that job; growing it needs a written reason.
EXEMPT = {
    # Imports PIL, which that lint job deliberately does not install.
    "test_tauri_branding_contract.py",
    "test_update_release_notes.py",
    # Imports numpy/torch through the studio backend; runs in the Studio backend job instead.
    "test_mlx_context_platform_matrix.py",
}


def _guard_step() -> dict:
    doc = yaml.safe_load(LINT.read_text(encoding = "utf-8"))
    for step in doc["jobs"]["workflow-trigger-lint"]["steps"]:
        if "pytest" in str(step.get("run", "")):
            return step
    raise AssertionError("workflow-trigger-lint.yml no longer runs pytest at all")


def _modules_run_by_the_lint_job() -> set:
    doc = yaml.safe_load(LINT.read_text(encoding = "utf-8"))
    runs = "\n".join(
        str(step.get("run", "")) for step in doc["jobs"]["workflow-trigger-lint"]["steps"]
    )
    return set(re.findall(r"tests/studio/(test_[\w]+\.py)", runs))


def _modules_that_read_a_workflow() -> set:
    found = set()
    for path in sorted(TESTS.glob("test_*.py")):
        src = path.read_text(encoding = "utf-8", errors = "replace")
        if ".github/workflows" in src or re.search(r'"\.github"\s*/\s*"workflows"', src):
            found.add(path.name)
    return found


def test_the_scan_finds_the_guards_it_claims_to():
    """A scan that matched nothing would pass the check below on an empty set."""
    found = _modules_that_read_a_workflow()
    assert len(found) >= 10, f"only found {len(found)} workflow-reading guards; scan is wrong"
    for expected in ("test_playwright_suites_run_in_ci.py", "test_backend_ci_matrix.py"):
        assert expected in found, f"{expected} reads a workflow but the scan missed it"


def test_every_workflow_reading_guard_runs_in_the_unfiltered_job():
    uncovered = sorted(_modules_that_read_a_workflow() - _modules_run_by_the_lint_job() - EXEMPT)
    assert not uncovered, (
        f"these guards read a workflow file but are not run by workflow-trigger-lint, the "
        f"only job with no paths filter: {uncovered}. A PR that edits only workflow files "
        f"-- which is exactly the change each of them exists to reject -- never collects "
        f"them, so they cannot block it. Add them to the guard step, or to EXEMPT with a "
        f"reason."
    )


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_the_exemptions_still_exist_and_are_still_needed(name):
    """An exemption that outlives its file, or its reason, quietly shrinks the check."""
    assert (TESTS / name).is_file(), f"EXEMPT names {name}, which no longer exists"
    assert name not in _modules_run_by_the_lint_job(), (
        f"{name} is exempted from the unfiltered job but that job now runs it. Remove it "
        f"from EXEMPT so the check keeps covering it."
    )


def test_the_guards_run_in_one_pytest_invocation():
    """One pytest invocation: a step per module pays this repo's expensive conftest import every time."""
    doc = yaml.safe_load(LINT.read_text(encoding = "utf-8"))
    steps = doc["jobs"]["workflow-trigger-lint"]["steps"]
    invocations = [s for s in steps if "-m pytest" in str(s.get("run", ""))]
    assert len(invocations) == 1, (
        f"workflow-trigger-lint runs pytest in {len(invocations)} separate steps. Add the "
        f"module to the existing invocation instead: one step per module costs about 15s "
        f"of interpreter and conftest startup each."
    )


def test_the_job_that_runs_them_has_no_paths_filter():
    """The entire premise. If this job gains a filter, every guard above loses its point."""
    doc = yaml.safe_load(LINT.read_text(encoding = "utf-8"))
    on = doc.get(True) if True in doc else doc.get("on")
    for trigger in ("pull_request", "push"):
        config = on.get(trigger)
        if not isinstance(config, dict):
            continue
        assert not config.get("paths") and not config.get("paths-ignore"), (
            f"workflow-trigger-lint now filters its {trigger} trigger on paths. It is the "
            f"only job a workflow-only PR is guaranteed to start, and every guard it runs "
            f"depends on that."
        )
