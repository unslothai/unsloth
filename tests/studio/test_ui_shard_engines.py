# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
Each Chat UI shard must install exactly the browser engines its own steps drive.

The four shards used to install all three engines each. Two of them -- `extra` and
`picker` -- never open anything but chromium, and installing webkit's system
libraries for them is an apt transaction of 181 packages and 102 MB:

    0 upgraded, 181 newly installed, 0 to remove
    Need to get 102 MB/114 MB of archives
    Get:2 .../noble/universe amd64 fonts-wqy-zenhei all 0.9.45-8 [7472 kB]
    -> 4m51s later: attempt 2/2 did not finish within 300s

Fonts, X fonts and a soundfont, fetched so that two shards which never launch webkit
could time out fetching them. That is what this file exists to stop coming back.

It is enforced in BOTH directions, and the second one is the dangerous one:

  * installing an engine no step drives is waste, and waste on a degraded mirror is
    an outage;
  * driving an engine the shard did not install is a broken run. Playwright reports
    it as a launch failure deep inside a suite, minutes after the install step went
    green, which reads as a flaky test rather than a missing package.

Derived from the workflow, never from a hardcoded list: the whole point is that
adding a webkit step to `picker` fails HERE, at the edit, rather than in CI.
"""

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
    """Engines the steps that RUN for ``shard`` actually name.

    The install step itself is excluded: it names engines because it installs them,
    so counting it would make every shard trivially consistent with itself.
    """
    driven: set[str] = set()
    for step in _doc()["jobs"]["ui-smoke"]["steps"]:
        run = step.get("run") or ""
        name = step.get("name") or ""
        # Only steps that RUN something can drive a browser.
        if not run:
            continue
        if "playwright install" in run or "probe " in run:
            continue
        cond = str(step.get("if") or "")
        # A step gated on another shard does not run for this one. Anything ungated runs for every shard.
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
    """
    The cache holds the downloaded browsers. Two shards installing different engine
    sets under one key means the smaller set gets saved, the larger one restores it,
    reports a hit, skips the download, and then cannot launch what it did not get.
    """
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
    # chromium is the default engine for every suite here, so it is legitimately installed whether or not a step names
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
    picker = _engines_driven_by("picker")
    assert "webkit" not in picker, (
        f"the picker shard now appears to drive {sorted(picker)}; if that is real the "
        f"matrix needs updating, and if it is not the scan is over-matching"
    )
