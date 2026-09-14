# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A path-filtered workflow must list the local actions it `uses:`.

Factoring a workflow's setup into `.github/actions/<name>/` moves real logic outside every
path in that workflow's `paths:` filter. The workflow then does not run when only the action
changes, so a PR can break the install preamble of a job and watch the job not run. The
failure is silent in the worst way: the checks that would have caught it are simply absent,
and the PR is green because nothing looked.

`.github/scripts/retry-with-apt-lock.sh` is already carried in two filters for this reason.
This makes that a rule rather than a thing someone remembered once.

A workflow with no `paths:` filter runs on everything and is not at risk, so it is skipped.
"""

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Actions already unlisted across 18 workflows when this guard was written. Recorded rather
# than fixed here so the guard can block NEW ones today; adding each of these to the filters
# of every workflow that uses it is a separate change, and one that needs its own thought
# about which of them can actually change a job's behaviour rather than only its cache.
_PRE_EXISTING = {
    ".github/actions/pip-cache-restore",
    ".github/actions/pip-cache-save",
    ".github/actions/uv-cache-restore",
    ".github/actions/uv-cache-save",
    ".github/actions/frontend-dist-restore",
    ".github/actions/frontend-dist-save",
    ".github/actions/install-unsloth-local",
}


def _triggers(doc: dict) -> dict:
    # `on:` is the YAML 1.1 boolean True once parsed, unless it was quoted.
    return doc.get(True) or doc.get("on") or {}


def _local_actions(node) -> set:
    """Every `uses: ./path` in the workflow, at any depth."""
    found = set()
    if isinstance(node, dict):
        uses = node.get("uses")
        if isinstance(uses, str) and uses.startswith("./"):
            found.add(uses[2:].rstrip("/"))
        for value in node.values():
            found |= _local_actions(value)
    elif isinstance(node, list):
        for value in node:
            found |= _local_actions(value)
    return found


def _covers(pattern: str, path: str) -> bool:
    """Does a `paths:` entry select `path`? Only the `**` suffix form needs handling."""
    pattern = pattern.strip("'\"")
    if pattern.endswith("/**"):
        return path == pattern[:-3] or path.startswith(pattern[:-3] + "/")
    return pattern == path or path.startswith(pattern.rstrip("/") + "/")


def _workflows() -> list:
    return sorted(_WORKFLOWS.glob("*.yml"))


@pytest.mark.parametrize("workflow", _workflows(), ids = lambda p: p.name)
def test_a_path_filtered_workflow_lists_the_actions_it_uses(workflow):
    doc = yaml.safe_load(workflow.read_text(encoding = "utf-8"))
    if not isinstance(doc, dict):
        pytest.skip("not a workflow mapping")
    actions = _local_actions(doc.get("jobs") or {})
    if not actions:
        pytest.skip("uses no local action")

    for event, spec in _triggers(doc).items():
        paths = (spec or {}).get("paths") if isinstance(spec, dict) else None
        if not paths:
            continue  # unfiltered: it already runs on any change
        missing = sorted(
            a for a in actions - _PRE_EXISTING if not any(_covers(p, a) for p in paths)
        )
        assert not missing, (
            f"{workflow.name} `{event}` is path-filtered but does not list {missing}, so a "
            f"change to only that action runs none of the jobs that depend on it"
        )


def test_the_pre_existing_list_does_not_outlive_the_problem():
    """An entry that is now listed everywhere it is used must leave the waiver, or the
    waiver quietly re-permits a regression someone already paid to fix."""
    still_unlisted = set()
    for workflow in _workflows():
        doc = yaml.safe_load(workflow.read_text(encoding = "utf-8"))
        if not isinstance(doc, dict):
            continue
        actions = _local_actions(doc.get("jobs") or {})
        for spec in _triggers(doc).values():
            paths = (spec or {}).get("paths") if isinstance(spec, dict) else None
            if not paths:
                continue
            still_unlisted |= {a for a in actions if not any(_covers(p, a) for p in paths)}
    settled = sorted(_PRE_EXISTING - still_unlisted)
    assert not settled, f"these are now listed everywhere and must leave _PRE_EXISTING: {settled}"


def test_the_guard_sees_a_workflow_that_uses_a_local_action():
    """Vacuity: if nothing in the tree uses a local action, everything above skips."""
    users = [
        w
        for w in _workflows()
        if _local_actions(yaml.safe_load(w.read_text(encoding = "utf-8")) or {})
    ]
    assert users, "no workflow uses a local action, so this guard checks nothing"
