# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A test that reads a workflow file must run on a PR that only edits workflow files.

`workflow-trigger-lint.yml` is the one job in this repo with no `paths:` filter, which
makes it the only job a workflow-only PR is guaranteed to start. Every other job filters on
source paths: Backend CI matches `studio/**`, `tests/**`, `scripts/**` and its own YAML, but
not arbitrary workflow files.

So a guard that reads `.github/workflows` and is not run by that job has a specific,
silent hole: the edit it exists to reject is by definition a workflow-only edit, and on
such a PR the guard is never collected. It is collected later, by Backend CI's unfiltered
push on main -- after the change has merged. The guard still works; it just stops being
able to block anything.

That went unnoticed for ten modules at once, which is why this is asserted from the
workflow rather than maintained by hand. Three separate review rounds reported one instance
each of it before the pattern was recognised.

The list is also deliberately ONE pytest invocation rather than one step per module. At 17
modules that is 53.9s against 300.8s measured locally, because this repo's conftest is
expensive to import and a step per module pays it every time.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "studio"
LINT = REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"

# Modules that read a workflow file but cannot run in that job, each with the reason.
# Shrinking this is the point; growing it needs a reason written here.
EXEMPT = {
    # Imports PIL, which that job does not install (it installs pyyaml, pytest, pytest-xdist and vermin only,
    # deliberately, so the lint stays seconds rather than minutes).
    "test_tauri_branding_contract.py",
    # Imports a local `utils` helper that resolves only under the full test environment.
    "test_update_release_notes.py",
    # Spoofs the hardware stack, so it imports numpy/torch through the studio backend.
    # It runs in the Studio backend job, which installs them; this one deliberately does not.
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
    """Test modules that reach into `.github/workflows`.

    Matched on the directory path as it appears in the source, either as a literal or as
    the `"workflows"` component of a `Path` join, which are the only two forms in use.
    """
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
    """One step per module pays this repo's conftest import cost once per module.

    Measured over these 17 modules: 53.9s as a single invocation, 300.8s as one each. The
    shape regresses naturally, because the obvious way to add a guard is to add a step.
    """
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
