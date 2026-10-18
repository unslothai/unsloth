# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rules every payload reader must apply; a flat null shows repeatability, never comparability."""

from __future__ import annotations

from typing import Any, Iterable

UNCOMPARABLE_ACROSS_ARMS: dict[str, str] = {
    "census_peak": (
        "chosen by a max() over per-action censuses that race the action's own teardown, so the "
        "moment it describes differs between arms. Measured swing on a null control, same bundle "
        "both sides: 70.1% within one arm."
    ),
}


def completed_cell_ids(records: Iterable[dict]) -> set[str]:
    """Window rows exist for unfinished cells too, so window readers must intersect with this set."""
    return {
        r.get("cell_id")
        for r in records
        if r.get("row_type") == "cell" and r.get("completed") and r.get("cell_id")
    }


def aborted_cell_ids(records: Iterable[dict]) -> set[str]:
    """Cell ids explicitly marked as not having finished, from the terminal `cell_aborted` row."""
    return {
        r.get("cell_id")
        for r in records
        if r.get("row_type") == "cell_aborted" and r.get("cell_id")
    }


def windows_of_completed_cells(records: list[dict]) -> list[dict]:
    """Reduces to the latest attempt first, since a resumed retry reuses the cell id of the dead attempt."""
    from .from_payload import latest_attempt_rows

    records = list(latest_attempt_rows(records))
    done = completed_cell_ids(records)
    return [r for r in records if r.get("row_type") == "window" and r.get("cell_id") in done]


def censored_metrics(records: Iterable[dict]) -> dict[str, set[str]]:
    """Censored and discarded-action timings are absent, and pooling the rest is survivorship bias."""
    out: dict[str, set[str]] = {}
    for r in records:
        if r.get("row_type") != "action":
            continue
        action = r.get("action")
        cell = r.get("cell_id")
        expect = r.get("expect") or {}
        for key, value in expect.items():
            if key.endswith("_censored") and value:
                metric = f"{action}.{key[:-len('_censored')]}_ms"
                out.setdefault(metric, set()).add(cell)
        if r.get("ran") and r.get("expect_ok") is False:
            for key in r.get("timings") or {}:
                out.setdefault(f"{action}.{key}", set()).add(cell)
    return out


def measured_cells(records: Iterable[dict], metric: str) -> set[str]:
    """Completed cells where the action ran, its own assertion did not fail, and the timing is real."""
    action, _, key = metric.partition(".")
    done = completed_cell_ids(records)
    out: set[str] = set()
    for r in records:
        if r.get("row_type") != "action" or r.get("action") != action:
            continue
        if r.get("cell_id") not in done or not r.get("ran") or r.get("expect_ok") is False:
            continue
        value = (r.get("timings") or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.add(r.get("cell_id"))
    return out


def refuse_partial_censoring(records: list[dict], metric: str) -> str | None:
    """Checked per cell, not per rung name, since the surviving repetitions select against the effect."""
    censored = censored_metrics(records).get(metric, set())
    done = completed_cell_ids(records)
    censored = {c for c in censored if c in done}
    if not censored:
        return None
    measured = measured_cells(records, metric)
    if not measured:
        # Censored everywhere: nothing to bias, so do not refuse.
        return None
    rungs_censored = sorted({c.split(".", 1)[0] for c in censored if c})
    rungs_measured = sorted({c.split(".", 1)[0] for c in measured if c})
    where = (
        f"at {rungs_censored} and measured at {rungs_measured}"
        if set(rungs_censored) != set(rungs_measured)
        else f"on {len(censored)} of {len(censored) + len(measured)} cells within {rungs_censored}"
    )
    return (
        f"{metric} is censored {where}. Pooling what is left reports the cells that could answer "
        f"as if they were the whole ladder, and the ones that could not are the slow ones."
    )


def refuse_uncomparable(metric: str) -> str | None:
    """Why `metric` must not be differenced between arms, or None if it may be."""
    for key, why in UNCOMPARABLE_ACROSS_ARMS.items():
        if metric == key or metric.startswith(key + "."):
            return f"{metric} is not comparable across arms: {why}"
    return None


def settled(action_row: dict) -> bool:
    """Did this action's census come from a DOM that had stopped changing?"""
    return bool((action_row.get("expect") or {}).get("settled"))


def comparability_key(run_meta: dict) -> str:
    """Keyed on the computed corpus hash, not the harness commit: a commit only claims provenance."""
    import hashlib
    import json

    # Hashed over `comparability_fields` so the key and its explanation cannot disagree.
    blob = json.dumps(comparability_fields(run_meta), sort_keys = True, default = str).encode()
    return "cmp:" + hashlib.sha256(blob).hexdigest()[:10]


def run_metas(records: Iterable[dict]) -> list[dict]:
    """EVERY `run_meta` row in the payload, in the order written. There is rarely only one."""
    return [r for r in records if r.get("row_type") == "run_meta"]


def merged_run_meta(records: Iterable[dict]) -> tuple[dict | None, list[str]]:
    """Merges all run_meta headers, since --resume appends one; disagreeing fields mean no single key."""
    metas = run_metas(records)
    if not metas:
        return None, []
    merged = dict(metas[0])
    rungs: list = []
    for m in metas:
        for rung in m.get("rungs") or []:
            if rung not in rungs:
                rungs.append(rung)
    merged["rungs"] = rungs
    base = comparability_fields(metas[0])
    conflicts: list[str] = []
    for m in metas[1:]:
        other = comparability_fields(m)
        for key in sorted(base):
            if key == "rungs":
                continue
            if other.get(key) != base.get(key):
                line = f"{key}: {base.get(key)!r} != {other.get(key)!r}"
                if line not in conflicts:
                    conflicts.append(line)
    return merged, conflicts


def comparability_fields(run_meta: dict) -> dict:
    """The fields the key is computed over, so a mismatch can be explained rather than asserted."""
    platform = run_meta.get("platform") or {}
    return {
        "corpus_hash": run_meta.get("corpus_hash"),
        "tier": run_meta.get("tier"),
        "rungs": run_meta.get("rungs"),
        "engine": platform.get("engine"),
        "tool_version": run_meta.get("tool_version"),
        "instrument_level": run_meta.get("instrument_level"),
        "cadence": run_meta.get("cadence"),
        "stream_tail_chars": run_meta.get("stream_tail_chars"),
        "corpus_dollars": run_meta.get("corpus_dollars"),
        # These change what is measured and `--resume` refuses to toggle them; an injected-cost arm
        # must never compare against a clean one. `bool` maps pre-field payloads to False.
        "click_probe": bool(run_meta.get("click_probe")),
        "probe_init_script": run_meta.get("probe_init_script"),
        "inject_stream_cost_ms": run_meta.get("inject_stream_cost_ms"),
        # The host matters: webkit is the default on both Darwin and Linux, so engine alone misses it.
        "system": platform.get("system"),
        "machine": platform.get("machine"),
        # `machine()` is only the architecture; `node()` separates two Linux x86_64 hosts. Pre-field
        # payloads carry None and are not comparable, which is correct.
        "node": platform.get("node"),
        # Headed and headless use different Chromium binaries since Playwright 1.57; `bool` maps
        # pre-field payloads to the headless default.
        "headed": bool(run_meta.get("headed")),
    }


def explain_incomparable(a: dict, b: dict) -> list[str]:
    """Which comparability fields differ between two `run_meta` rows."""
    fa, fb = comparability_fields(a), comparability_fields(b)
    return [
        f"{k}: {fa[k]!r} != {fb[k]!r}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)
    ]
