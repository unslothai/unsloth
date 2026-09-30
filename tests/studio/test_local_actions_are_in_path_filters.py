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

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Local actions a path-filtered workflow may use without listing. Empty: the pip cache pair,
# the last entry here, is now listed by every workflow that uses it. Growing this again needs
# a reason written next to the entry, and the test below removes an entry that is no longer
# needed.
_PRE_EXISTING: set = set()


def _triggers(doc: dict) -> dict:
    # `on:` is the YAML 1.1 boolean True once parsed, unless it was quoted.
    return doc.get(True) or doc.get("on") or {}


def _resolve(used: str) -> str:
    """The action's real location in this repo, given a `uses:` path.

    Some workflows check the repo into a subdirectory and so say
    `uses: ./unsloth/.github/actions/pip-cache-restore`. No such directory exists here; the
    action being referenced is the repo-root one. Taking the literal string would put a path
    under `unsloth/` into the check, where a `unsloth/**` filter entry satisfies it and the
    guard passes on a workflow that would still skip. So leading segments are dropped until
    the result is a directory that actually holds an action.
    """
    parts = used.split("/")
    for start in range(len(parts)):
        candidate = "/".join(parts[start:])
        if (REPO_ROOT / candidate / "action.yml").is_file():
            return candidate
    return used


def _local_actions(node) -> set:
    """Every `uses: ./path` in the workflow, at any depth."""
    found = set()
    if isinstance(node, dict):
        uses = node.get("uses")
        if isinstance(uses, str) and uses.startswith("./"):
            found.add(_resolve(uses[2:].rstrip("/")))
        for value in node.values():
            found |= _local_actions(value)
    elif isinstance(node, list):
        for value in node:
            found |= _local_actions(value)
    return found


def _matches(pattern: str, path: str) -> bool:
    """One `paths:` glob against one file path, with GitHub's wildcard semantics.

    `path` is a FILE inside the action, not the action directory, because that is what
    GitHub matches a filter against. The distinction decides real cases: a bare
    `.github/actions/foo` entry selects only a file literally at that path, so it does NOT
    cover `.github/actions/foo/action.yml` and the workflow still skips.
    """
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, path) is not None


def _selects(paths: list, path: str) -> bool:
    """Does the whole `paths:` list select `path`?

    The list is ordered and `!` negates, with the LAST matching pattern deciding, so the
    entries cannot be tested independently: `.github/actions/**` followed by
    `!.github/actions/foo/**` does not select foo. This repo already uses negation
    (startup-profile-ci.yml excludes `!studio/backend/tests/**`), so reading `!` as a
    literal character would quietly pass a workflow that really does skip.
    """
    selected = False
    for entry in paths:
        entry = str(entry).strip("'\"")
        negated = entry.startswith("!")
        if _matches(entry[1:] if negated else entry, path):
            selected = not negated
    return selected


def _action_files(action: str, root: Path = REPO_ROOT) -> list:
    """Every file GitHub could report as changed for an edit to this action.

    The whole directory, not just the manifest. A composite action that grows a helper
    script keeps working with a filter naming only `action.yml`, and a helper-only change
    would then skip every workflow that uses it, which is the exact failure this guard
    exists to prevent. Listing the directory means such a change has to be covered too,
    by a `/**` entry or by naming the file.
    """
    directory = root / action
    if not directory.is_dir():
        return [f"{action}/action.yml"]
    return sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    )


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
            a
            for a in actions - _PRE_EXISTING
            if not all(_selects(paths, f) for f in _action_files(a))
        )
        assert not missing, (
            f"{workflow.name} `{event}` is path-filtered but does not list {missing}, so a "
            f"change to only that action runs none of the jobs that depend on it"
        )


def test_an_ordered_negation_is_not_read_as_a_literal_bang():
    """`paths:` is ordered and `!` negates, with the last match deciding.

    Testing the entries independently would accept `.github/actions/**` followed by
    `!.github/actions/foo/**` as covering foo, while GitHub skips the workflow on a
    foo-only change. This repo already negates in startup-profile-ci.yml, so the case is
    reachable rather than theoretical.
    """
    action = ".github/actions/foo/action.yml"
    assert _selects([".github/actions/**"], action)
    assert not _selects([".github/actions/**", "!.github/actions/foo/**"], action)
    assert _selects(["!.github/actions/foo/**", ".github/actions/**"], action)
    # The negation this repo actually carries, and a sibling it must not touch.
    assert not _selects(
        ["studio/backend/**", "!studio/backend/tests/**"], "studio/backend/tests/x.py"
    )
    assert _selects(["studio/backend/**", "!studio/backend/tests/**"], "studio/backend/main.py")


def test_a_helper_file_beside_the_manifest_is_checked_too(tmp_path):
    """Every action here is a lone action.yml today, so this is the case the tree cannot
    show: an action that grows a helper script.

    A filter naming only `<action>/action.yml` still selects the manifest, so checking the
    manifest alone would pass while a helper-only change skipped every consuming workflow.
    """
    action = ".github/actions/grown"
    (tmp_path / action).mkdir(parents = True)
    (tmp_path / action / "action.yml").write_text("name: grown\n")
    (tmp_path / action / "run.sh").write_text("echo hi\n")

    files = _action_files(action, root = tmp_path)
    assert files == [f"{action}/action.yml", f"{action}/run.sh"]

    manifest_only = [f"{action}/action.yml"]
    assert _selects(manifest_only, files[0])
    assert not all(
        _selects(manifest_only, f) for f in files
    ), "naming only the manifest must not satisfy the guard once the action has a helper"
    assert all(_selects([f"{action}/**"], f) for f in files)


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
            still_unlisted |= {
                a for a in actions if not all(_selects(paths, f) for f in _action_files(a))
            }
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
