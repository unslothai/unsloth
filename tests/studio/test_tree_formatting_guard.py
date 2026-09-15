# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""scripts/check_tree_is_formatted.py reads the hook, and Lint CI runs it on main.

The checker is only worth having if it selects the same files the hook does. If the
two drift, it reports a fixed point over a set nobody formats, which is a green that
means less than no check at all. So these read `.pre-commit-config.yaml` rather than
restating it, and fail when the hook grows a shape the checker cannot follow.

The wiring is guarded for the same reason: a checker no job runs is not a check.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_tree_is_formatted.py"
CONFIG = REPO / ".pre-commit-config.yaml"
LINT_CI = REPO / ".github" / "workflows" / "lint-ci.yml"
HOOK_ID = "ruff-format-with-kwargs"

sys.path.insert(0, str(REPO / "scripts"))

import check_tree_is_formatted as checker  # noqa: E402


def _hook() -> dict:
    doc = yaml.safe_load(CONFIG.read_text(encoding = "utf-8"))
    for repo in doc["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == HOOK_ID:
                return hook
    pytest.fail(f"no {HOOK_ID} hook in .pre-commit-config.yaml")


def test_the_checker_reads_the_hooks_own_exclude():
    """Parsed out of the config, not restated. A second copy is a second thing to forget."""
    assert checker.hook_exclude(CONFIG.read_text(encoding = "utf-8")).pattern == _hook()["exclude"]


def test_the_checker_reads_the_hooks_own_pin():
    """The ruff CI installs has to be the ruff the repo was formatted with."""
    printed = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-pin"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    pinned = [d for d in _hook()["additional_dependencies"] if d.startswith("ruff==")]
    assert pinned == [f"ruff=={printed}"]


def test_the_checker_selects_exactly_the_hooks_files():
    """`types: [python]` minus `exclude:`, over tracked files.

    Compared against git rather than against a second regex here, so a file the hook
    would newly cover cannot pass by matching a copy of the old rule.
    """
    selected = set(checker.selected_files(REPO, CONFIG.read_text(encoding = "utf-8")))
    exclude = re.compile(_hook()["exclude"])
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.splitlines()
    assert selected == {p for p in tracked if not exclude.search(p)}
    assert selected, "the hook selects nothing, so the check would be vacuous"
    # The exclusions are load-bearing, not decorative: ruff itself skips these.
    assert any(exclude.search(p) for p in tracked), "nothing is excluded; is the pattern live?"


def test_a_hook_shape_the_checker_cannot_follow_is_refused_not_guessed():
    """Silently checking the old file set would be the one unacceptable outcome."""
    config = CONFIG.read_text(encoding = "utf-8")
    widened = config.replace("types: [python]", "types: [python, pyi]", 1)
    assert widened != config, "the hook no longer declares `types: [python]`"
    with pytest.raises(checker.CannotTell):
        checker.selected_files(REPO, widened)

    without = re.sub(r"^\s*exclude: '\(chat_templates.*\n", "", config, count = 1, flags = re.M)
    assert without != config
    with pytest.raises(checker.CannotTell):
        checker.selected_files(REPO, without)

    renamed = config.replace(f"id: {HOOK_ID}", "id: something-else", 1)
    with pytest.raises(checker.CannotTell):
        checker.selected_files(REPO, renamed)


def test_lint_ci_runs_it_on_main_and_only_on_main():
    """A checker no job runs is not a check.

    On a pull request this would duplicate pre-commit.ci and go red for a base branched
    before a formatting fix landed, which is the noise #11019 was cleaning up after. The
    signal that was missing is after the merge.
    """
    doc = yaml.safe_load(LINT_CI.read_text(encoding = "utf-8"))
    steps = doc["jobs"]["source-lint"]["steps"]
    running = [s for s in steps if SCRIPT.name in (s.get("run") or "")]
    assert len(running) == 1, f"expected one step running {SCRIPT.name}, found {len(running)}"
    condition = running[0].get("if", "")
    assert "refs/heads/main" in condition and "push" in condition, condition
    # PyYAML reads the `on:` key as the boolean True (YAML 1.1), as the other workflow
    # guards here do.
    triggers = doc.get(True) if True in doc else doc.get("on")
    assert "main" in str(triggers["push"]["branches"]), "Lint CI does not run on main at all"
