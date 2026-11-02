# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch.compile evidence for the parallel capability registry: graph breaks, recompiles and graphs compiled.

    before = compile_counters.snapshot()
    trainer.train()
    used = compile_counters.since(before)   # {"graph_breaks": 0, "recompiles": 0, "unique_graphs": 8, ...}

A composed layout (tp, cp, pp, FSDP2) is compile-clean when `used` has no more graph breaks and recompiles than
the same script in one process (`not_above`)."""

from __future__ import annotations

from collections import Counter

__all__ = ["snapshot", "since", "not_above", "COUNTED"]

COUNTED = ("graph_breaks", "recompiles", "unique_graphs", "frames")


def snapshot() -> dict:
    try:
        from torch._dynamo import utils
    except Exception:
        return {k: 0 for k in COUNTED} | {"graph_break_reasons": {}}
    counters = utils.counters
    reasons = {str(k).split("\n", 1)[0]: int(v) for k, v in counters.get("graph_break", {}).items()}
    failures = getattr(utils, "guard_failures", {}) or {}
    return {
        "graph_breaks": sum(reasons.values()),
        "recompiles": sum(len(v) for v in failures.values()),
        "unique_graphs": int(counters.get("stats", {}).get("unique_graphs", 0)),
        "frames": int(counters.get("frames", {}).get("ok", 0)),
        "graph_break_reasons": reasons,
        "recompile_reasons": [
            f"{getattr(code, 'co_name', code)}: {str(r).splitlines()[0] if str(r) else ''}"
            for code, v in failures.items()
            for r in v
        ],
    }


def since(before: dict) -> dict:
    now = snapshot()
    out = {k: now[k] - before.get(k, 0) for k in COUNTED}
    old = before.get("graph_break_reasons", {})
    out["graph_break_reasons"] = {
        k: v - old.get(k, 0) for k, v in now["graph_break_reasons"].items() if v - old.get(k, 0)
    }
    seen = Counter(before.get("recompile_reasons", []))
    new = []
    for reason in now["recompile_reasons"]:
        if seen[reason]:
            seen[reason] -= 1
        else:
            new.append(reason)
    out["recompile_reasons"] = new[:50]
    return out


def not_above(composed: dict, single: dict) -> list:
    """The counters where the composed run exceeds the one-process run (empty: compile-clean)."""
    return [
        f"{k} {composed.get(k, 0)} > {single.get(k, 0)}"
        for k in ("graph_breaks", "recompiles")
        if composed.get(k, 0) > single.get(k, 0)
    ]
