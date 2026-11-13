# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paths-filtered workflow must list the local actions it uses, or an action-only change skips it."""

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Growing this needs a reason next to the entry; the test below prunes stale ones.
_PRE_EXISTING: set = set()


def _triggers(doc: dict) -> dict:
    # `on:` is the YAML 1.1 boolean True once parsed, unless quoted.
    return doc.get(True) or doc.get("on") or {}


def _resolve(used: str) -> str:
    """Drops leading segments of a uses: path until a real action.yml is found, e.g. a nested checkout."""
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
    """A bare action-directory entry matches only that literal path, not files inside it."""
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, path) is not None


def _selects(paths: list, path: str) -> bool:
    """In ordered paths the last matching entry decides, and '!' negates, so entries are not independent."""
    selected = False
    for entry in paths:
        entry = str(entry).strip("'\"")
        negated = entry.startswith("!")
        if _matches(entry[1:] if negated else entry, path):
            selected = not negated
    return selected


def _action_files(action: str, root: Path = REPO_ROOT) -> list:
    """The whole action directory, not just action.yml, since a helper-only change must still trigger."""
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
            continue
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
    """Ordered negation: '.github/actions/**' then '!.github/actions/foo/**' does not select foo."""
    action = ".github/actions/foo/action.yml"
    assert _selects([".github/actions/**"], action)
    assert not _selects([".github/actions/**", "!.github/actions/foo/**"], action)
    assert _selects(["!.github/actions/foo/**", ".github/actions/**"], action)
    assert not _selects(
        ["studio/backend/**", "!studio/backend/tests/**"], "studio/backend/tests/x.py"
    )
    assert _selects(["studio/backend/**", "!studio/backend/tests/**"], "studio/backend/main.py")


def test_a_helper_file_beside_the_manifest_is_checked_too(tmp_path):
    """A helper script beside action.yml must also be selected, or helper-only edits skip consumers."""
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
