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
    """Every `window` row that belongs to a cell that finished. The only windows worth pooling.

    REDUCED TO THE LATEST ATTEMPT FIRST, because a cell id is not unique across attempts. `--resume`
    re-runs a cell that died under the SAME deterministic id in the SAME file under a new session,
    so a completed retry puts that id in `done` and matching on the id alone hands back the dead
    attempt's windows as well -- precisely the unfinished film this helper exists to exclude.

    Measured on a payload written through the real Recorder, one aborted attempt at 28.7 fps and
    its completed retry at 46.7 fps: the helper returned all 13 windows, and pushing its own output
    through `_frame_measures` gave `max_frame_ms` 34.84 against the truth of 21.41, and a
    `jank_index` of 2.719 against 0.000 -- a jank score invented entirely by the run that crashed.
    `floor_table.cell_metrics` got this right on the same records, so the importable rule was wrong
    where the older ad-hoc guard was right, which is the wrong way round for a module whose claim is
    that the rules are not folklore.

    `aborted_cell_ids` IS NOT A MITIGATION FOR THIS, and must not be subtracted from `done`: on that
    payload the same cell id is in both sets, so `done - aborted` is empty and a reader who tries it
    loses the good reading instead of the bad one.
    """
    from .from_payload import latest_attempt_rows

    records = list(latest_attempt_rows(records))
    done = completed_cell_ids(records)
    return [r for r in records if r.get("row_type") == "window" and r.get("cell_id") in done]


def censored_metrics(records: Iterable[dict]) -> dict[str, set[str]]:
    """{metric: {cell_id, ...}} for timings that were censored rather than measured.

    A censored timing is ABSENT from `timings`, and absence is invisible once the values are
    pooled: the cells that survive are the fast ones, and their mean is survivorship bias wearing
    a number. `reasoning_toggle.open_ms` is censored on every cell above the 100K rung, so a row
    labelled `open_ms` on a 100K/500K/1M ladder is silently a 100K-only figure.

    A TIMING IS ALSO UNAVAILABLE WHEN ITS ACTION WAS DISCARDED, which is the case that reaches
    furthest. `_action_timings` drops every timing of an action whose `expect_ok` is False, so a
    measurement that succeeded on its own terms still contributes nothing -- and nothing marked it.

    `reasoning_toggle` is where this bites. `ok` is one conjunction over four clauses, so a
    censored `open_ms` above 100K makes the whole action fail, and `close_ms` is thrown away with
    it while `close_censored` stays False. On a 100K/500K ladder the close row was then pooled from
    the 100K cells alone and printed under a bare metric name with no refusal, exactly the
    survivorship bias the open row is marked for. The rule is not specific to that pair: an action
    that was discarded contributes nothing, so every timing it carries is censored at that cell.
    """
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
        # The row still CARRIES the timings the scoring layer refuses to read, so the names are
        # known here even though the values will never be pooled.
        if r.get("ran") and r.get("expect_ok") is False:
            for key in r.get("timings") or {}:
                out.setdefault(f"{action}.{key}", set()).add(cell)
    return out


def measured_cells(records: Iterable[dict], metric: str) -> set[str]:
    """Completed cells on which `metric` actually produced a number.

    Resolved the way `floor_table._action_timings` resolves it -- the action ran, its own assertion
    did not fail, and the timing is a real number -- so this is the set the pooled mean is actually
    built from rather than a second opinion about it.
    """
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
    """Why `metric` must not be pooled across this payload, or None if it is safe.

    PER CELL, NOT PER RUNG NAME. Censoring is decided cell by cell against a fixed budget, on a
    metric whose own spread is 33 to 42%, so a rung sitting near that budget censors SOME of its
    repetitions and not others -- which is the expected shape near the boundary, not a contrived
    one. Comparing sets of rung names cannot see it: every rung appears in both sets, the refusal
    stays silent, and `paired()` quietly keeps only the repetitions where both arms answered.

    Measured on one rung with three of four treatment repetitions censored, the surviving pair
    reported +10.0% on n=1 with a full SLOWER verdict, counted as a metric that cleared all three
    gates. The true paired delta over all four repetitions was +35.7%. The censored repetitions
    were the slow ones, which is exactly why they were censored, so what survives is not a sample
    of the effect but a selection against it.

    This module's own docstring already stated the rule in cell terms -- "the cells that survive
    are the fast ones" -- and only the implementation reduced it to rung names.
    """
    censored = censored_metrics(records).get(metric, set())
    done = completed_cell_ids(records)
    censored = {c for c in censored if c in done}
    if not censored:
        return None
    measured = measured_cells(records, metric)
    if not measured:
        # Censored everywhere it was attempted. Nothing survived to be biased, and there is no
        # pooled number to refuse -- refusing here would fire on every payload that simply cannot
        # answer, which is a different and much noisier rule than the one being enforced.
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

    # COMPUTED OVER `comparability_fields`, not over a second copy of the same dict. The two were
    # written out separately and had to be kept in step by hand, which is the drift this module
    # exists to argue against: a field added to one and forgotten in the other would make the key
    # and its own explanation disagree about what the key covers.
    blob = json.dumps(comparability_fields(run_meta), sort_keys = True, default = str).encode()
    return "cmp:" + hashlib.sha256(blob).hexdigest()[:10]


def run_metas(records: Iterable[dict]) -> list[dict]:
    """EVERY `run_meta` row in the payload, in the order written. There is rarely only one."""
    return [r for r in records if r.get("row_type") == "run_meta"]


def merged_run_meta(records: Iterable[dict]) -> tuple[dict | None, list[str]]:
    """One `run_meta` describing the WHOLE payload, plus the disagreements that forbid one.

    A payload is append-only and `--resume` writes a SECOND header behind the first, so reading
    only the first describes the run that started the file rather than the cells now in it.
    `floor_table` has been bitten by this twice and says so where it was fixed: "Reading only the
    first is what let a fast-tier film and a standard-tier film sit in one file and pass the
    refusal below." This is the same mistake in the one reader that was not converted.

    `rungs` is UNIONED rather than compared, because extending the ladder is explicitly legitimate.
    `IDENTITY_AXES` leaves `rungs` and `reps` out on purpose: "Resuming with more repetitions or
    another rung is a legitimate continuation -- it ADDS cells rather than reinterpreting the ones
    already recorded." A resumed payload really does describe the union, so the union is what a
    later reader has to compare against.

    Every OTHER key field is a conflict when the headers disagree, and three of them can differ
    without `--resume` objecting: `engine` (`--engine` is free on resume), `tool_version` (a module
    constant, so pulling a harness upgrade mid-campaign changes it), and anything the identity
    check does not cover. A file whose own headers disagree on those is not one measurement, and no
    single key can honestly stand for it.
    """
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
        # THE THREE THAT CHANGE WHAT IS MEASURED, not merely what is measured on. The harness
        # already treats all three as identity axes that `--resume` refuses to toggle, and the
        # `run_meta` emission says why in its own words -- so a key that ignored them would call
        # two payloads comparable that the tool itself refuses to continue as one run.
        #
        # `inject_stream_cost_ms` is the sharpest: an arm running it is not a measurement of the
        # build at all, because the harness put the slowdown there on purpose. Nothing in the
        # scoring path refuses an injected payload, so `--compare` blessing it against a clean run
        # was a live route to publishing a difference the harness itself created.
        #
        # `click_probe` is normalised through `bool` so a payload written before the field existed
        # reads as False rather than None; the other two are `None` in the ordinary case already.
        "click_probe": bool(run_meta.get("click_probe")),
        "probe_init_script": run_meta.get("probe_init_script"),
        "inject_stream_cost_ms": run_meta.get("inject_stream_cost_ms"),
        # THE HOST, which the engine alone does not stand in for. `run_meta` records `system` and
        # `machine` and the key took neither, so two payloads from different hardware and operating
        # systems hashed the same and `--compare` blessed them.
        #
        # It needs no unusual invocation: `browser.default_engine()` returns webkit on Darwin AND
        # on Linux, so a tester's Mac payload and the Linux dev box's payload, both with default
        # settings, differ in `system`, `machine` and `engine_note` and were declared comparable.
        # Only Windows was caught, and only incidentally, because its default engine differs.
        #
        # Nothing else in the tool refuses cross-host pooling: `floor_table.load` checks tier,
        # corpus and probe only, and `platform` is not an identity axis. The sole existing
        # statement is prose in `report/render.py` -- "machine-local; does not travel between
        # machines" -- so the authors knew the hazard and the one guard meant to police a prose
        # comparison was the place it was missing.
        "system": platform.get("system"),
        "machine": platform.get("machine"),
        # AND WHICH PHYSICAL MACHINE, because the two fields above cannot say. `platform.machine()`
        # is the ARCHITECTURE, not a machine identifier: on two ordinary Linux x86_64 hosts it
        # returns `x86_64` on both while `system` returns `Linux` on both, so the pair that was
        # added here to stand for the host caught only the cross-OS case -- a tester's Mac against
        # the Linux dev box -- and left the commonest one, a dev box against a CI runner or a
        # second dev box, hashing identically. `platform.node()` is the host's network name and is
        # the field that separates them.
        #
        # This is the axis with the least slack in it: `floor_table.render` refuses a floor whose
        # comparability fields differ from the payload's, in the words "a floor is the scatter of
        # THIS measurement on THIS machine", and that refusal is computed from this dict. Without
        # the host in it a null control measured on one machine certifies a result measured on
        # another, which is the one comparison the report text says never travels.
        #
        # A payload recorded before this field existed carries None and is therefore not comparable
        # with one that carries a host. That is the honest reading rather than a cost: such a
        # payload does not record which machine produced it, so nothing can show it was this one.
        "node": platform.get("node"),
        # WHICH BROWSER BINARY DREW THE FRAMES, which `engine` does not settle. Since Playwright
        # 1.57 headed and headless default to different executables -- `chrome` against
        # `chrome-headless-shell` -- and headless falls back to software rendering for
        # GPU-accelerated work while its compositor keeps its own pacing. For a tool whose output
        # is frames, jank and time-to-settle, those are two different renderers under one engine
        # name. Normalised through `bool` so a payload written before the field existed reads as
        # the headless default rather than as None.
        "headed": bool(run_meta.get("headed")),
    }


def explain_incomparable(a: dict, b: dict) -> list[str]:
    """Which comparability fields differ between two `run_meta` rows."""
    fa, fb = comparability_fields(a), comparability_fields(b)
    return [
        f"{k}: {fa[k]!r} != {fb[k]!r}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)
    ]
