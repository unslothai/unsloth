# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scoped `pull_request` trigger must be scoped the same way on `push`.

Sixteen workflows carefully scoped their `pull_request` trigger and then left
`push: branches: [main]` completely unfiltered. The effect is invisible on a pull request
and only appears after merge: a commit that changes nothing those workflows can observe
still starts every one of them. Measured over a 1.5 hour window, this repo's 300 most
recent runs contained 125 queued, and 70% of a job's wall clock was spent waiting for a
runner rather than running (`unsloth-zoo`, sharing the same account, measured a 143 minute
median queue against 5.7 minutes of work). Post-merge work that cannot change a verdict is
therefore paid for twice: once as the slots it holds, and again as the queue every pull
request behind it waits in.

`tests/studio/test_macos_slots_per_commit.py` already pins this for the macOS workflows,
where the five-concurrent-job account-wide cap made it urgent first. This module asks the
same question of every workflow, on the same reasoning it gives: the two lists are one
question asked twice -- "could this commit break this workflow" -- so drift is always a
bug, and a push list narrower than the pull_request list is the dangerous direction,
because it silently stops testing after merge something that was tested before it.

The prerequisite, and why this file is not just a preference: scoping a push trigger is
only safe if the `pull_request` list is COMPLETE. While push was unfiltered, every commit
re-ran everything after merge, so a helper the workflow executes but never listed was a
gap nobody could observe. Narrowing push is exactly what turns that gap into a real hole.
`studio-export-capability-ci.yml` had one -- it is `uses:` on pip-cache-restore and
pip-cache-save and listed neither -- and both were added in the same change that scoped its
push trigger. `test_local_actions_are_in_path_filters.py` and
`test_macos_slots_per_commit.py::test_every_helper_a_workflow_executes_is_in_its_trigger`
are the guards for that half of the invariant.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Workflows whose push trigger is deliberately broader than their pull_request trigger.
# Each needs the reason written here; this set shrinking is good and it growing needs an
# argument, not a convenience.
DELIBERATELY_UNFILTERED_ON_PUSH = {
    # Every job here is a function of the dependency manifests, so a PR touching none of
    # them cannot change a verdict -- but a FLOATING pin can, with no diff at all: a
    # version range resolving to a new release, or a freshly published advisory against an
    # unchanged tree. The workflow's own comment records that the daily cron and every
    # push to main are left unfiltered for exactly that reason, so scoping push here would
    # narrow a deliberate safety net rather than remove waste.
    "security-audit.yml",
}


def _on(doc):
    """The `on:` mapping, which PyYAML parses as the boolean True."""
    return doc.get(True) if True in doc else doc.get("on")


def _targets_main(push) -> bool:
    branches = push.get("branches") or []
    return "main" in branches if isinstance(branches, list) else branches == "main"


def _scoped_on_pull_request():
    """(name, pr_paths, push) for workflows with a scoped PR trigger and a push to main."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        if not isinstance(doc, dict):
            continue
        on = _on(doc)
        if not isinstance(on, dict):
            continue
        pull_request = on.get("pull_request")
        push = on.get("push")
        if not isinstance(pull_request, dict) or not isinstance(push, dict):
            continue
        pr_paths = pull_request.get("paths")
        if not pr_paths or not _targets_main(push):
            continue
        yield path.name, pr_paths, push


def test_the_scan_finds_the_workflows_it_claims_to():
    """A scan that matched nothing would pass every check below on an empty set."""
    names = {name for name, _, _ in _scoped_on_pull_request()}
    assert len(names) >= 15, f"only found {len(names)} scoped-on-PR workflows; the scan is wrong"
    for expected in (
        "studio-backend-ci.yml",
        "studio-windows-ui-smoke.yml",
        "mlx-ci.yml",
    ):
        assert (
            expected in names
        ), f"{expected} scopes its pull_request trigger but the scan missed it"


def test_every_exemption_still_exists_and_still_needs_one():
    """A stale exemption is a filter nobody is asking for any more."""
    names = {name for name, _, _ in _scoped_on_pull_request()}
    for name in DELIBERATELY_UNFILTERED_ON_PUSH:
        assert (WORKFLOWS / name).exists(), f"{name} is exempted here but no longer exists"
        assert name in names, (
            f"{name} is exempted here but no longer has a scoped pull_request trigger plus a "
            f"push to main, so the exemption describes nothing. Remove it."
        )
        doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
        push = _on(doc).get("push")
        assert not push.get("paths") and not push.get("paths-ignore"), (
            f"{name} now scopes its push trigger, so it does not need to be exempted. "
            f"Remove it from DELIBERATELY_UNFILTERED_ON_PUSH."
        )


def test_no_scoped_workflow_runs_unfiltered_on_every_push_to_main():
    offenders = [
        name
        for name, _, push in _scoped_on_pull_request()
        if name not in DELIBERATELY_UNFILTERED_ON_PUSH
        and not push.get("paths")
        and not push.get("paths-ignore")
    ]
    assert not offenders, (
        f"these workflows scope their pull_request trigger but run on EVERY commit to main: "
        f"{offenders}. A merge that cannot change what they test still holds their runner "
        f"slots, and the queue that produces is what every pull request behind them waits "
        f"in. Mirror the pull_request paths onto push, as mlx-ci.yml does. If the broader "
        f"push trigger is deliberate, add it to DELIBERATELY_UNFILTERED_ON_PUSH with the "
        f"reason."
    )


@pytest.mark.parametrize("name", sorted(n for n, _, _ in _scoped_on_pull_request()))
def test_the_push_filter_matches_the_pull_request_filter(name):
    """Narrower on push than on PR drops the post-merge backstop; broader wastes slots."""
    if name in DELIBERATELY_UNFILTERED_ON_PUSH:
        pytest.skip(f"{name} is deliberately unfiltered on push")
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = _on(doc)
    pr_paths = (on.get("pull_request") or {}).get("paths")
    push_paths = (on.get("push") or {}).get("paths")
    assert push_paths, f"{name} no longer scopes its push trigger"
    assert sorted(pr_paths) == sorted(push_paths), (
        f"{name}: the push and pull_request path filters have drifted apart.\n"
        f"  only on pull_request: {sorted(set(pr_paths) - set(push_paths))}\n"
        f"  only on push:         {sorted(set(push_paths) - set(pr_paths))}"
    )
