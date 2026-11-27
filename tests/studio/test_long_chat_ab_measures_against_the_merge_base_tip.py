# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The long-chat A/B must compare the PR merge against the base tip it was merged onto.

``github.event.pull_request.base.sha`` does not move when main advances. A PR opened before a
chat regression landed on main was measured against that older base while its merge ref already
carried the regression, so the check charged the PR with it (#19169 measured against cf00c93,
merged onto 7d4eb4aa, and inherited #19172's stream_scroll long tasks).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "studio-frontend-ci.yml"
STEP = "Long-chat cost against the base build"


def _step() -> dict:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    steps = [step for step in jobs["long-chat-ab"]["steps"] if step.get("name") == STEP]
    assert len(steps) == 1, steps
    return steps[0]


def _resolution_lines() -> str:
    """The step's script up to (not including) the fetch of the base it resolved."""
    script = _step()["run"]
    return script[: script.index("git fetch")]


def test_the_event_base_is_only_a_fallback():
    step = _step()
    assert step["env"].get("EVENT_BASE_SHA") == "${{ github.event.pull_request.base.sha }}"
    assert "BASE_SHA" not in step["env"], "BASE_SHA must be resolved from the checkout"
    assert '--base-ref "$BASE_SHA"' in step["run"]


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return subprocess.run(
        ["git", *args], cwd = cwd, env = env, check = True, capture_output = True, text = True
    ).stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding = "utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _resolve(checkout: Path, event_base: str) -> str:
    script = _resolution_lines() + 'printf "%s" "$BASE_SHA"\n'
    return (
        subprocess.run(
            ["bash", "-e", "-c", script],
            cwd = checkout,
            env = {**os.environ, "EVENT_BASE_SHA": event_base},
            check = True,
            capture_output = True,
            text = True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )


@pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("git")),
    reason = "runs the step's bash against a real git checkout",
)
def test_a_depth_one_merge_checkout_resolves_the_tip_it_merged_onto(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    stale_base = _commit(origin, "base")
    _git(origin, "checkout", "-q", "-b", "pr")
    _commit(origin, "pr-change")
    _git(origin, "checkout", "-q", "main")
    current_tip = _commit(origin, "landed-on-main-later")
    _git(origin, "checkout", "-q", "-b", "merge")
    _git(origin, "merge", "-q", "--no-ff", "-m", "Merge pr into main", "pr")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "--depth=1", "--branch", "merge", origin.as_uri(), str(checkout))
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "true"

    assert _resolve(checkout, stale_base) == current_tip


@pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("git")),
    reason = "runs the step's bash against a real git checkout",
)
def test_a_checkout_that_is_not_a_merge_keeps_the_event_base(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    event_base = _commit(repo, "base")
    _commit(repo, "head")
    assert _resolve(repo, event_base) == event_base
