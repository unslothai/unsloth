# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
The uv download cache must stay a download cache, and must never reach a
cold-install lane.

`Install Unsloth (--local, --no-torch)` is the single largest cost in CI: 92s
median across 39 job runs in one sample, more total time than any test, and all
of it uv re-downloading the same wheels because its cache is per-runner.

Caching that is safe *because of what is cached*. uv's cache is content-addressed
by URL and hash, so a stale entry cannot serve wrong content -- the worst it can
do is miss. That property is the whole justification, and it is exactly what a
later edit could take away by pointing the same cache config at the venv, or at
`~/.unsloth`, where an editable overlay, a moving `unsloth-zoo @ git+main` and
absolute paths in console scripts all live. These tests pin the distinction.

The second invariant is the one with teeth. `clean-machine-install-ci.yml` and
`desktop-app-clean-machine-ci.yml` exist to prove the installer works on a
machine with nothing on it; both set their own `UV_CACHE_DIR` and delete it
before running. If either ever adopted this action, the composite writes
`UV_CACHE_DIR` to `$GITHUB_ENV`, which outranks a job-level `env:` for every
later step -- so a warm cache would silently replace the cold machine those
workflows are named after, and they would still go green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION = REPO_ROOT / ".github" / "actions" / "install-unsloth-local" / "action.yml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Named, not detected: a lane whose whole point is a cold machine should have to
# be removed from this list deliberately, in a diff someone reads.
COLD_INSTALL_WORKFLOWS = (
    "clean-machine-install-ci.yml",
    "desktop-app-clean-machine-ci.yml",
    "interrupted-install-ci.yml",
)


def _steps() -> list[dict]:
    return yaml.safe_load(ACTION.read_text(encoding = "utf-8"))["runs"]["steps"]


def _index_of(predicate) -> int:
    for i, step in enumerate(_steps()):
        if predicate(step):
            return i
    return -1


def test_the_cache_holds_uvs_downloads_and_not_the_venv() -> None:
    """
    A venv cache would have to reason about the editable overlay, a moving
    unsloth-zoo pin, and absolute paths in console scripts. A download cache
    reasons about none of that, which is why this one is safe at all.
    """
    for step in _steps():
        if "cache" not in str(step.get("uses", "")):
            continue
        path = str((step.get("with") or {}).get("path", ""))
        assert ".uv-cache" in path, f"cache step points at {path!r}, not uv's download cache"
        for forbidden in (".unsloth", "site-packages", "unsloth_studio", "venv"):
            assert forbidden not in path, (
                f"cache step points at {path!r}, which is an INSTALL, not a download "
                f"cache. A restored install can be wrong; a restored download cannot."
            )


def test_uv_cache_dir_is_set_before_the_install_runs() -> None:
    """Set afterwards it configures nothing, and the step would still look right."""
    setter = _index_of(lambda s: "UV_CACHE_DIR" in str(s.get("run", "")))
    install = _index_of(lambda s: "install.sh --local --no-torch" in str(s.get("run", "")))
    assert setter != -1, "the action no longer points UV_CACHE_DIR anywhere"
    assert install != -1, "the action no longer runs the local install"
    assert setter < install, (
        "UV_CACHE_DIR is set after the install, so the install used uv's default "
        "cache and the restored one was never read"
    )


def test_the_restore_happens_before_the_install_too() -> None:
    restore = _index_of(lambda s: "cache/restore" in str(s.get("uses", "")))
    install = _index_of(lambda s: "install.sh --local --no-torch" in str(s.get("run", "")))
    assert restore != -1 and restore < install


def test_a_near_miss_still_supplies_most_wheels() -> None:
    """
    restore-keys is what makes this worth having on a PR whose requirements moved
    by one line. It is correct here precisely because the entry is content-
    addressed; the same fallback on a venv cache would be a bug.
    """
    restore = next(s for s in _steps() if "cache/restore" in str(s.get("uses", "")))
    assert (restore.get("with") or {}).get("restore-keys"), (
        "no restore-keys, so any change to requirements or pyproject drops the "
        "cache to zero instead of to almost-full"
    )


def test_the_cache_is_saved_on_main_only() -> None:
    """
    A PR-scoped entry can only be restored by re-runs of that same PR, while every
    PR can restore from the default branch. Saving on PRs spends a budget measured
    at 99.3% full once already, and evicts main's copy -- the one everyone reads.
    """
    saves = [s for s in _steps() if "cache/save" in str(s.get("uses", ""))]
    assert saves, "the cache is never saved, so it can never be restored either"
    for step in saves:
        condition = str(step.get("if", ""))
        assert (
            "refs/heads/main" in condition
        ), f"a cache/save step is not gated on main: if: {condition!r}"


@pytest.mark.parametrize("name", COLD_INSTALL_WORKFLOWS)
def test_cold_install_lanes_never_adopt_this_action(name: str) -> None:
    """
    These prove the installer works on a machine with nothing on it. The composite
    writes UV_CACHE_DIR to $GITHUB_ENV, which outranks a job-level `env:` for every
    later step, so adopting it would hand a cold lane a warm cache and the lane
    would still report success.
    """
    path = WORKFLOWS / name
    if not path.exists():
        pytest.skip(f"{name} no longer exists")
    assert "install-unsloth-local" not in path.read_text(encoding = "utf-8"), (
        f"{name} uses install-unsloth-local, which warms uv's cache. A cached "
        f"cold-install test proves nothing and still goes green."
    )


def test_the_action_is_actually_used() -> None:
    """Otherwise every assertion above guards something nothing runs."""
    users = [
        p.name
        for p in WORKFLOWS.glob("*.yml")
        if "install-unsloth-local" in p.read_text(encoding = "utf-8")
    ]
    assert len(users) >= 5, f"only {len(users)} workflows use the action: {users}"
