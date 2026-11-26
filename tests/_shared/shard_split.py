# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Runs one part of a shard's tests: ``UNSLOTH_TEST_SPLIT=k/n`` keeps part k of n and deselects the rest.

distributed-cpu-ci.yml gives a slow shard several matrix entries with the same selection and ``split: 1/n`` ..
``split: n/n``, so a shard grows by runners rather than past its time cap. Every process (each xdist worker
collects on its own) computes the same parts from the collected ids alone:

- an ``xdist_group`` stays whole, because its tests share a module-scoped run or must meet on one worker;
- every other test is its own unit;
- each unit weighs its measured seconds from shard_durations.json. A test missing there weighs its file's
  ``"*"`` (the median of that file's fast tests), or 2 s in a file with no entry;
- units go heaviest first, ties by id, each to the part with the least weight so far.

The parts are disjoint and together are every collected test (tests/test_shard_split.py checks both). The weights
only balance the parts: a stale or missing entry makes one part longer, never drops or repeats a test.
Refresh them from junit XML of a full run: ``python tests/_shared/shard_split.py --update a.xml b.xml ...``."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

ENV = "UNSLOTH_TEST_SPLIT"
DURATIONS = Path(__file__).with_name("shard_durations.json")
DEFAULT_SECONDS = 2
LISTED_FROM_SECONDS = 5  # faster tests are covered by their file's "*"


def parse(spec: str | None) -> tuple[int, int] | None:
    """(k, n) from "k/n" with 1 <= k <= n; None when unset or empty."""
    if not spec or not spec.strip():
        return None
    try:
        k, n = (int(part) for part in spec.strip().split("/"))
    except ValueError:
        raise ValueError(f"{ENV}={spec!r}: expected k/n, for example 2/3") from None
    if not 1 <= k <= n:
        raise ValueError(f"{ENV}={spec!r}: k must be between 1 and n")
    return k, n


def _durations() -> dict:
    try:
        return json.loads(DURATIONS.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}


def _weight(nodeid: str, durations: dict) -> float:
    path, _, name = nodeid.split("@", 1)[0].partition("::")
    per_file = durations.get(path) or {}
    return float(per_file.get(name, per_file.get("*", DEFAULT_SECONDS)))


def _unit(item: Any) -> str:
    marker = item.get_closest_marker("xdist_group")
    if marker is not None:
        name = marker.args[0] if marker.args else marker.kwargs.get("name", "")
        return f"group:{name}"
    return "test:" + item.nodeid.split("@", 1)[0]  # loadgroup may already have tagged the id


def parts(
    items: list,
    n: int,
    durations: dict | None = None,
) -> list[int]:
    """The part (1..n) of each item, in the items' order."""
    durations = _durations() if durations is None else durations
    units: dict[str, list[int]] = {}
    weights: dict[str, float] = {}
    for index, item in enumerate(items):
        unit = _unit(item)
        units.setdefault(unit, []).append(index)
        weights[unit] = weights.get(unit, 0.0) + _weight(item.nodeid, durations)
    load = [0.0] * n
    part_of = [0] * len(items)
    for unit in sorted(units, key = lambda u: (-weights[u], u)):
        target = min(range(n), key = lambda p: (load[p], p))
        load[target] += weights[unit]
        for index in units[unit]:
            part_of[index] = target + 1
    return part_of


def select(config: Any, items: list) -> None:
    """Keeps part k of n of ``items`` in place; the rest are reported deselected."""
    spec = parse(os.environ.get(ENV))
    if spec is None:
        return
    k, n = spec
    keep, drop = [], []
    for item, part in zip(items, parts(items, n)):
        (keep if part == k else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items = drop)
    items[:] = keep


def update(junit_files: list[str]) -> dict:
    """Weights from junit XML (setup + call + teardown per test): the median of every sample, whole seconds."""
    import xml.etree.ElementTree as ET

    samples: dict[str, dict[str, list[float]]] = {}
    for junit in junit_files:
        for case in ET.parse(junit).getroot().iter("testcase"):
            path = case.get("classname", "").replace(".", "/") + ".py"
            name = case.get("name", "").split("@", 1)[0]
            samples.setdefault(path, {}).setdefault(name, []).append(float(case.get("time") or 0))
    out = {}
    for path in sorted(samples):
        seconds = {name: max(1, round(statistics.median(v))) for name, v in samples[path].items()}
        fast = [s for s in seconds.values() if s < LISTED_FROM_SECONDS]
        out[path] = {"*": round(statistics.median(fast)) if fast else DEFAULT_SECONDS}
        out[path].update({k: v for k, v in sorted(seconds.items()) if v >= LISTED_FROM_SECONDS})
    return out


def write(durations: dict, path: Path = DURATIONS) -> None:
    """One line per test file, so a refresh diffs per file."""
    files = sorted(durations)
    lines = ["{"]
    for index, name in enumerate(files):
        tail = "," if index < len(files) - 1 else ""
        lines.append(
            json.dumps(name) + ":" + json.dumps(durations[name], separators = (",", ":")) + tail
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding = "utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--update":
        write(update(sys.argv[2:]))
    else:
        raise SystemExit("usage: shard_split.py --update junit.xml [junit.xml ...]")
