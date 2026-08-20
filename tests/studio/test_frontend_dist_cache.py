# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The frontend dist cache and setup.sh's rebuild check must read the same inputs.

Measured over 13 distinct Linux jobs on main, the frontend build is a median 36s of a
74s `install-unsloth-local`, 49% of it, and 468s per commit producing byte-identical
output. The cache exists to stop paying that 13 times.

What makes it safe is not the cache action, it is the agreement between two places:

    studio/setup.sh          rebuilds when anything under frontend/ (maxdepth 1, minus
                             bun.lock), frontend/src or frontend/public is NEWER than
                             frontend/dist
    the action's cache key   hashes exactly those three path groups

A hit therefore means the build inputs are byte-identical, which is strictly stronger
than the mtime test it rides on. Break the agreement and nothing goes red: the cache
keeps hitting and quietly starts serving a dist built from inputs the key no longer
covers, and every job downstream tests a stale bundle that passes. That is the whole
reason this file exists, and it is why it asserts against setup.sh's own source rather
than a list written down here.

Three subtler failure modes are pinned too, each of which looks like success:

  * `restore-keys` on this cache. A near-miss download cache still supplies most of the
    wheels; a near-miss dist is a bundle built from different source. Wrong, not partial.
  * A restore with no `touch`. actions/cache restores through tar, which preserves the
    original mtimes, so the restored dist is older than the checkout that just wrote
    every source file and setup.sh rebuilds anyway. The cache would cost a download,
    save nothing, and report a hit.
  * An empty `hashFiles`. It returns "" when a glob matches nothing, collapsing every
    commit onto one key and serving an arbitrary dist.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ACTION = REPO / ".github" / "actions" / "install-unsloth-local" / "action.yml"
SETUP_SH = REPO / "studio" / "setup.sh"


def _steps() -> list[dict]:
    doc = yaml.safe_load(ACTION.read_text(encoding = "utf-8")) or {}
    return [s for s in (doc.get("runs") or {}).get("steps") or [] if isinstance(s, dict)]


def _step(fragment: str) -> dict | None:
    for step in _steps():
        if fragment.lower() in str(step.get("name", "")).lower():
            return step
    return None


def _restore_step() -> dict:
    step = _step("Restore the built frontend")
    assert step is not None, (
        "install-unsloth-local no longer restores a built frontend. If the cache was "
        "removed on purpose, delete this file; if it was renamed, retarget it."
    )
    return step


def _key() -> str:
    return str((_restore_step().get("with") or {}).get("key", ""))


def _key_globs() -> set[str]:
    """The paths hashFiles() reads, normalised so a trailing /** does not matter."""
    inner = re.search(r"hashFiles\((.*?)\)", _key())
    assert inner, f"the dist cache key does not call hashFiles: {_key()!r}"
    return {
        g.strip().strip("'\"").removesuffix("/**").rstrip("/*").rstrip("/")
        for g in inner.group(1).split(",")
    }


def _staleness_inputs() -> set[str]:
    """The paths setup.sh's rebuild check compares against frontend/dist.

    Read out of setup.sh rather than hardcoded: a list written here would agree with
    itself forever while setup.sh moved.
    """
    text = SETUP_SH.read_text(encoding = "utf-8")
    block = re.search(
        r"Detect whether frontend needs building(.*?)end packaged/Tauri guard", text, re.S
    )
    assert block, "could not find the frontend staleness check in studio/setup.sh"
    body = block.group(1)
    found = set()
    for m in re.finditer(r'"\$SCRIPT_DIR/(frontend[^"]*)"', body):
        path = m.group(1)
        if path.endswith("/dist"):
            continue
        found.add("studio/" + path)
    return found


def test_the_key_covers_every_path_the_rebuild_check_reads() -> None:
    missing = sorted(_staleness_inputs() - _key_globs())
    assert not missing, (
        f"studio/setup.sh decides to rebuild the frontend by looking at {missing}, and the "
        f"dist cache key does not hash them. A change to those files would not change the "
        f"key, so the cache would hit and serve a dist built from different source, and "
        f"nothing would go red. Key: {_key()!r}"
    )


def test_the_key_does_not_hash_paths_the_rebuild_check_ignores() -> None:
    """Not a style rule: an over-broad key silently destroys the hit rate.

    bun.lock is the deliberate exception. setup.sh must exclude it because the install
    regenerates it and it would self-trigger every run; the cache has no such problem,
    and a lockfile change means different dependencies and so a different bundle. It is
    covered by the `studio/frontend/*` glob, which is why that glob is allowed to be
    broader than the check's maxdepth-1 scan rather than being narrowed to match it.
    """
    extra = sorted(_key_globs() - _staleness_inputs())
    assert extra == [], (
        f"the dist cache key hashes {extra}, which setup.sh's rebuild check does not "
        f"read. Every unrelated edit to those paths would miss the cache for no reason. "
        f"If the extra path genuinely affects the built bundle, say so where the key is "
        f"defined and widen this test deliberately."
    )


def test_the_dist_cache_has_no_restore_keys() -> None:
    with_ = _restore_step().get("with") or {}
    assert "restore-keys" not in with_, (
        "the frontend dist cache has restore-keys. A prefix hit would serve a bundle "
        "built from DIFFERENT source, which is wrong rather than partial. The uv "
        "download cache in the same action does want them, and that contrast is the "
        "point: a near-miss download still supplies most of the wheels."
    )


def test_a_restored_dist_is_made_newer_than_the_checkout() -> None:
    step = _step("outrank its sources")
    assert step is not None, (
        "nothing touches the restored dist. actions/cache restores through tar, which "
        "preserves the original mtimes, so setup.sh's `find -newer dist` sees the whole "
        "freshly checked-out tree as newer and rebuilds anyway. The cache would report a "
        "hit, cost a download and save nothing."
    )
    body = str(step.get("run", ""))
    assert re.search(r"^\s*touch studio/frontend/dist\s*$", body, re.M), (
        f"the step meant to make the restored dist outrank its sources does not touch "
        f"studio/frontend/dist: {body!r}"
    )
    assert str(step.get("if", "")).strip() == "steps.fe-dist.outputs.cache-hit == 'true'", (
        "the touch must be gated on a cache hit, or a miss would touch a dist that was "
        "never restored and suppress the build that has to happen"
    )


def test_a_degenerate_key_is_refused() -> None:
    step = _step("hashes nothing")
    assert step is not None, (
        'nothing refuses an empty hashFiles result. It returns "" when a glob matches '
        "no file, which collapses every commit onto one key and serves an arbitrary "
        "dist, with the restore succeeding and the build skipped."
    )
    assert "exit 1" in str(step.get("run", "")), "the degenerate-key check does not fail the job"


def test_the_dist_cache_is_saved_on_main_only() -> None:
    step = _step("Save the built frontend")
    assert step is not None, "the dist cache is restored but never saved, so it can only ever miss"
    cond = str(step.get("if", ""))
    assert "refs/heads/main" in cond, (
        f"the dist cache is saved off main: {cond!r}. A PR-scoped entry can only be "
        f"restored by re-runs of that same PR while still counting against the shared "
        f"budget, evicting the copy every PR can read."
    )


def test_the_guard_is_reading_real_files() -> None:
    """Every assertion above passes vacuously if these two files stop being found."""
    assert ACTION.is_file(), ACTION
    assert SETUP_SH.is_file(), SETUP_SH
    assert len(_staleness_inputs()) >= 2, _staleness_inputs()
    assert len(_key_globs()) >= 2, _key_globs()
