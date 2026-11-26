# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Runs one part of a shard's tests: ``UNSLOTH_TEST_SPLIT=k/n`` keeps part k of n and deselects the rest.

distributed-cpu-ci.yml gives a slow shard several matrix entries with the same selection and ``split: 1/n`` ..
``split: n/n``, so a shard grows by runners rather than past its time cap. Every process (each xdist worker
collects on its own) computes the same parts from the collected ids alone:

- an ``xdist_group`` stays whole, because its tests share a module-scoped run or must meet on one worker;
- every other test is its own unit;
- units are dealt in sorted id order to the part with the fewest tests so far, so the parametrizations of one
  slow test spread over the parts.

The parts are disjoint and together are every collected test (tests/test_shard_split.py checks both)."""

from __future__ import annotations

import os
from typing import Any

ENV = "UNSLOTH_TEST_SPLIT"


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


def _unit(item: Any) -> str:
    marker = item.get_closest_marker("xdist_group")
    if marker is not None:
        name = marker.args[0] if marker.args else marker.kwargs.get("name", "")
        return f"group:{name}"
    return "test:" + item.nodeid.split("@", 1)[0]  # loadgroup may already have tagged the id


def parts(items: list, n: int) -> list[int]:
    """The part (1..n) of each item, in the items' order."""
    units: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        units.setdefault(_unit(item), []).append(index)
    load = [0] * n
    part_of = [0] * len(items)
    # Groups first (largest first), then single tests, each in sorted order: the deal depends on ids only.
    order = sorted(units, key = lambda u: (not u.startswith("group:"), -len(units[u]), u))
    for unit in order:
        target = min(range(n), key = lambda p: (load[p], p))
        load[target] += len(units[unit])
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
