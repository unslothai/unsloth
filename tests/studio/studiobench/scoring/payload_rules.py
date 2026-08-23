# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The rules every reader of a payload has to apply, in one place so nobody reimplements them.

Four numbers were published from this harness and acted on before being withdrawn. All four came
from the same mistake in different clothes: MEASURING AT A MOMENT WHOSE MEANING IS NOT STABLE
ACROSS THE THINGS BEING COMPARED.

    census_peak                     an action chosen by max(), racing its own teardown, so a
                                    different action on each arm
    orphaned window rows            windows outlive their cell, so an unfinished film contributes
                                    whichever phase it reached
    highlight_spans_while_open      read on the frame a state attribute flipped, which is a
                                    different distance from "mounted" on each arm
    reasoning_toggle.open_ms        terminated on that same attribute flip, quantised to a paint
                                    and censored at a timeout

`floor_table.cell_metrics` already applied the completed-cell rule; nothing else did, and the one
consumer that did not apply it produced a headline regression that did not exist. Hence this
module: the rules are importable, not folklore.

THE COMPANION RULE TO THE NULL CONTROL, which is what makes three of these survivable.

    A null control cannot detect a bias it shares.

A null runs one build against itself. Any skew that is symmetric between the two sides cancels
exactly, so the null reads flat and certifies the metric. `highlight_spans_while_open` has a null
of 0.0% at 100K and 500K, exact to the span -- and was still 41% wrong across two real arms,
because the timing skew it carried was symmetric within the null and asymmetric between the arms.
A flat null says "this measure is repeatable", never "this measure is comparable".
"""

from __future__ import annotations

from typing import Any, Iterable

#: Metrics that must never be differenced between two arms, and why.
UNCOMPARABLE_ACROSS_ARMS: dict[str, str] = {
    "census_peak": (
        "chosen by a max() over per-action censuses that race the action's own teardown, so the "
        "moment it describes differs between arms. Measured swing on a null control, same bundle "
        "both sides: 70.1% within one arm."
    ),
}


def completed_cell_ids(records: Iterable[dict]) -> set[str]:
    """Cell ids whose film ran to the end.

    A `window` row is written while the film runs; the `cell` row is written when it ends. So an
    in-flight or aborted cell leaves a complete-looking set of window rows with no completed cell
    owning them. Any analysis that reads window rows directly must intersect with this set.
    """
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
    """Every `window` row that belongs to a cell that finished. The only windows worth pooling."""
    done = completed_cell_ids(records)
    return [r for r in records if r.get("row_type") == "window" and r.get("cell_id") in done]


def censored_metrics(records: Iterable[dict]) -> dict[str, set[str]]:
    """{metric: {cell_id, ...}} for timings that were censored rather than measured.

    A censored timing is ABSENT from `timings`, and absence is invisible once the values are
    pooled: the cells that survive are the fast ones, and their mean is survivorship bias wearing
    a number. `reasoning_toggle.open_ms` is censored on every cell above the 100K rung, so a row
    labelled `open_ms` on a 100K/500K/1M ladder is silently a 100K-only figure.
    """
    out: dict[str, set[str]] = {}
    for r in records:
        if r.get("row_type") != "action":
            continue
        expect = r.get("expect") or {}
        for key, value in expect.items():
            if key.endswith("_censored") and value:
                metric = f"{r.get('action')}.{key[:-len('_censored')]}_ms"
                out.setdefault(metric, set()).add(r.get("cell_id"))
    return out


def refuse_partial_censoring(records: list[dict], metric: str) -> str | None:
    """Why `metric` must not be pooled across this payload's rungs, or None if it is safe.

    Pooling a metric that is censored at some rungs and measured at others compares the rungs that
    could answer against the rungs that could not, under one label.
    """
    censored = censored_metrics(records).get(metric, set())
    if not censored:
        return None
    done = completed_cell_ids(records)
    rungs_censored = {c.split(".", 1)[0] for c in censored if c}
    rungs_all = {c.split(".", 1)[0] for c in done if c}
    if rungs_censored and rungs_censored != rungs_all:
        return (
            f"{metric} is censored at {sorted(rungs_censored)} and measured at "
            f"{sorted(rungs_all - rungs_censored)}. Pooling those under one label reports the "
            f"rungs that could answer as if they were the whole ladder."
        )
    return None


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
    """A short token over everything that must match for two payloads to be comparable.

    KEYED ON THE COMPUTED CORPUS HASH, NOT ON THE HARNESS COMMIT. Two trees can both call
    themselves studiobench and generate different corpora: one tree here ran a corpus hash that
    was neither the branch's pre-fix hash nor its post-fix one, silently and self-consistently,
    and the only reason anyone noticed is that the hash was printed and compared. A commit is a
    claim about provenance; the computed hash is the thing itself.

    `floor_table.load` already refuses to POOL across tiers and corpora. This exists for the case
    that guard cannot reach: a comparison made in prose between two separately published runs.
    Quote the token beside any number and the comparison becomes checkable after the fact.
    """
    import hashlib
    import json

    platform = run_meta.get("platform") or {}
    fields = {
        "corpus_hash": run_meta.get("corpus_hash"),
        "tier": run_meta.get("tier"),
        "rungs": run_meta.get("rungs"),
        "engine": platform.get("engine"),
        "tool_version": run_meta.get("tool_version"),
        "instrument_level": run_meta.get("instrument_level"),
        "cadence": run_meta.get("cadence"),
        "stream_tail_chars": run_meta.get("stream_tail_chars"),
        "corpus_dollars": run_meta.get("corpus_dollars"),
    }
    blob = json.dumps(fields, sort_keys = True, default = str).encode()
    return "cmp:" + hashlib.sha256(blob).hexdigest()[:10]


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
    }


def explain_incomparable(a: dict, b: dict) -> list[str]:
    """Which comparability fields differ between two `run_meta` rows."""
    fa, fb = comparability_fields(a), comparability_fields(b)
    return [
        f"{k}: {fa[k]!r} != {fb[k]!r}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)
    ]
