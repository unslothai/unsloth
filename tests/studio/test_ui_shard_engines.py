# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Each Chat UI shard installs exactly the engines its steps drive, checked in both directions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-ui-smoke.yml"

ENGINES = ("chromium", "firefox", "webkit")


def _doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _shards() -> list[dict]:
    include = _doc()["jobs"]["ui-smoke"]["strategy"]["matrix"]["include"]
    assert include, "the ui-smoke matrix has no include list; this guard checks nothing"
    return include


def _engines_driven_by(shard: str) -> set[str]:
    """Excludes the install step, which would make every shard trivially consistent with itself."""
    driven: set[str] = set()
    for step in _doc()["jobs"]["ui-smoke"]["steps"]:
        run = step.get("run") or ""
        name = step.get("name") or ""
        if not run:
            continue
        if "playwright install" in run or "probe " in run:
            continue
        cond = str(step.get("if") or "")
        others = re.findall(r"matrix\.shard\s*==\s*'([a-z]+)'", cond)
        if others and shard not in others:
            continue
        for engine in ENGINES:
            if re.search(rf"\b{engine}\b", run) or re.search(rf"\b{engine}\b", name):
                driven.add(engine)
    return driven


def test_every_shard_declares_engines_and_a_key() -> None:
    for cell in _shards():
        assert cell.get("engines"), f"shard {cell.get('shard')!r} declares no engines"
        assert cell.get("engine_key"), (
            f"shard {cell.get('shard')!r} has no engine_key, so its browser cache would "
            f"share a key with a shard that installs a different engine set"
        )


def test_the_engine_key_distinguishes_the_engine_set() -> None:
    """Different engine sets need different cache keys, or a smaller saved set restores as a cache hit."""
    by_key: dict[str, set[str]] = {}
    for cell in _shards():
        by_key.setdefault(cell["engine_key"], set()).add(cell["engines"])
    for key, sets in by_key.items():
        assert len(sets) == 1, (
            f"engine_key {key!r} is used for different engine sets {sorted(sets)}; the "
            f"browser cache would serve one shard's browsers to another"
        )


@pytest.mark.parametrize("cell", _shards(), ids = lambda c: c["shard"])
def test_shard_installs_every_engine_it_drives(cell: dict) -> None:
    installed = set(cell["engines"].split())
    driven = _engines_driven_by(cell["shard"])
    missing = driven - installed
    assert not missing, (
        f"shard {cell['shard']!r} drives {sorted(missing)} but installs only "
        f"{sorted(installed)}. Playwright will fail to launch mid-suite, minutes after "
        f"the install step went green. Add the engine to this shard's `engines`."
    )


@pytest.mark.parametrize("cell", _shards(), ids = lambda c: c["shard"])
def test_shard_installs_nothing_it_never_drives(cell: dict) -> None:
    installed = set(cell["engines"].split())
    driven = _engines_driven_by(cell["shard"])
    # chromium is the default engine for every suite, so it is installed whether named or not.
    extra = installed - driven - {"chromium"}
    assert not extra, (
        f"shard {cell['shard']!r} installs {sorted(extra)} but no step drives them. "
        f"webkit alone is 181 packages and 102 MB of apt on a mirror that has already "
        f"timed this job out; drop it or point at the step that needs it."
    )


def test_the_detector_sees_the_cross_browser_steps() -> None:
    """Otherwise both directions above pass by finding nothing driven anywhere."""
    chat = _engines_driven_by("chat")
    assert {"firefox", "webkit"} <= chat, (
        f"the chat shard drives {sorted(chat)}; it runs Cross-browser permission "
        f"controls, so the step scan is not seeing engine names any more"
    )
    extra = _engines_driven_by("extra")
    assert "webkit" not in extra, (
        f"the extra shard now appears to drive {sorted(extra)}; if that is real the "
        f"matrix needs updating, and if it is not the scan is over-matching"
    )
