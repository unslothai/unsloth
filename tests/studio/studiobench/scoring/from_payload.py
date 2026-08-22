# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a recorded payload into the `{rung: {metric_key: Measure}}` the scoring layer consumes.

This is the seam between the two halves of the tool, and it was the one piece neither half owned:
the session layer emits rows shaped around what it observed, the scoring layer consumes readings
shaped around what it scores, and nothing converted one into the other. Until this existed the
ladder, the A/B and the report were all unreachable from a real run.

Two rules it exists to enforce:

  A MISSING READING IS NOT A GOOD READING.
      An action that did not run, an action that ran without the timing key, and an action that
      was never in the scene are three different facts and produce three different notes. None of
      them produces a number, and none of them produces a zero. This matters most for the actions
      that legitimately do not run at small rungs -- `scroll_during_generation` reports "the thread
      is shorter than the viewport" at 1K -- because scoring those as instant would make a small
      thread look like a fast one.

  THE QUANTITY SCORED IS NAMED, NOT ASSUMED.
      Every Measure carries the payload key it came from, because the anchor names and the
      recorded names are not always the same quantity (see SCROLL_SETTLE_NOTE below).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .anchors import METRIC_BY_KEY
from .frames import compute_frame_stats
from .schema import Measure

# ── where each scored metric actually lives in the payload ──────────────────────────────────

# (action name, timing key). Anything not listed here comes from the window frame recorder.
ACTION_SOURCES: Mapping[str, tuple[str, str]] = {
    "keystroke_p95_ms": ("keystroke", "p95_ms"),
    "menu_open_ms": ("message_menu", "open_ms"),
    "scroll_settle_ms": ("scroll_after", "gesture_ms"),
}

# The one mapping that is NOT an identity of meaning, so it is stated rather than buried. The
# anchor was written for settle time (how long after the gesture the thread stops moving); the
# scene records the gesture itself and the per-step cost, and never measures settle separately.
# `gesture_ms` is the closest recorded quantity and is on the same scale, but a reader comparing
# this column against the anchor's 100 ms / 3000 ms rationale is comparing against a slightly
# different thing, and should be told so in the cell rather than in a changelog.
SCROLL_SETTLE_NOTE = (
    "recorded as scroll_after.gesture_ms; the scene does not measure settle separately, so this "
    "is gesture duration, not post-gesture settle"
)

FRAME_METRICS: tuple[str, ...] = ("time_in_jank_pct", "jank_index", "max_frame_ms")

# Windows that are deliberately quiet. Pooling these into the frame metrics would dilute every
# jank share with idle time and make a bad build look average.
IDLE_WINDOW_KINDS: frozenset[str] = frozenset({"idle"})


# The row types keyed by `cell_id`, and so the ones a superseded attempt can leak through.
ATTEMPT_ROW_TYPES: frozenset[str] = frozenset({"cell", "action", "window"})


def latest_attempt_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Drop the rows of a SUPERSEDED attempt at a cell, keeping every other row untouched.

    `--resume` appends to the payload it is continuing and re-runs the cells that DID NOT
    complete, and `make_cell_id` is deterministic: the retry of `r10K.base.rep0` is written under
    the same `cell_id` as the attempt that died. Nothing downstream keys on the attempt, so both
    were read as one cell. Two ways that produced a wrong number, neither of them visible:

      THE DEAD ATTEMPT'S FRAMES BECAME THE RETRY'S. `_frame_measures` pools every window row
      carrying the cell id, so a 100 ms frame from the run that crashed stayed the RETRY's
      `max_frame_ms`, and its gaps stayed in the retry's jank distribution.

      THE RETRY DID NOT COUNT. `measures_from_records` keeps the FIRST cell row per rung and
      `report.build._completion_by_rung` keeps a failure over a success, so a rung whose only
      failure had already been re-run successfully still scored zero as INCOMPLETE.

    An attempt is `(cell_id, session_id)` and the LAST one in file order wins, which is the one
    the resumed run just wrote. Rows without a session id are kept: a payload from before the
    recorder stamped them cannot be split into attempts, and dropping it would lose the run.

    THE LATEST ATTEMPT IS THE LAST ONE THAT WROTE ANYTHING, not the last one that FINISHED. Keying
    this on cell rows alone made an attempt invisible unless it reached its terminal row, and
    `CellRunner.run` writes that in a `finally` -- which a SIGKILL, an OOM kill or a lost machine
    never reaches, while the Recorder has already flushed and fsynced every action and window row
    before it. So a resume hard-killed inside a cell left the older, completed attempt named as
    the latest, and `__main__._resume_set` skipped it. Combined with a resume that had already
    repaired an earlier pair, every cell then read as complete across two sessions, the next
    `--resume` ran nothing at all and exited 0 over a stale table.

    Any attempt-keyed row is evidence that an attempt happened, so all three types set it. This is
    the same set the filter below applies to, which is the point: a row type that can leak from a
    superseded attempt is a row type that can prove a newer one exists.
    """
    latest: dict[str, Any] = {}
    for row in records:
        if row.get("row_type") in ATTEMPT_ROW_TYPES and row.get("cell_id") is not None:
            latest[str(row.get("cell_id"))] = row.get("session_id")

    out: list[Mapping[str, Any]] = []
    for row in records:
        if row.get("row_type") in ATTEMPT_ROW_TYPES:
            keep = latest.get(str(row.get("cell_id")))
            if keep is not None and row.get("session_id") not in (None, keep):
                continue
        out.append(row)
    return out


def _cell_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in records if r.get("row_type") == "cell"]


def _actions_for(
    records: Sequence[Mapping[str, Any]], cell_id: str
) -> dict[str, Mapping[str, Any]]:
    """Index the standalone `action` rows for one cell by action name.

    The `actions` list embedded in the cell row is not used: it carries the timings but drops the
    names, so it can only be decoded positionally and only if nothing was skipped.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for r in records:
        if r.get("row_type") == "action" and r.get("cell_id") == cell_id:
            name = r.get("action")
            if name:
                out[str(name)] = r
    return out


def _action_measure(metric_key: str, actions: Mapping[str, Mapping[str, Any]]) -> Measure:
    action_name, timing_key = ACTION_SOURCES[metric_key]
    unit = METRIC_BY_KEY[metric_key].unit
    note = SCROLL_SETTLE_NOTE if metric_key == "scroll_settle_ms" else f"{action_name}.{timing_key}"

    row = actions.get(action_name)
    if row is None:
        return Measure.not_attempted(unit, f"{action_name} is not in this scene")
    if not row.get("ran"):
        reason = row.get("reason") or "no reason recorded"
        # Attempted-and-did-not-run, which is a fact about the run, not an absent instrument.
        return Measure.failed(unit, f"{action_name} did not run: {reason}")
    if row.get("expect_ok") is False:
        # RAN IS NOT DID WHAT IT CLAIMED. An action carries its own assertion -- the composer's
        # value grew by the characters that were typed, the menu that was opened has items, the
        # panes that were expanded are open -- and when that assertion fails the timing describes
        # something other than the action. `report/payload.py` already lists these cells under
        # EXCLUDED CELLS with "its timings exist and must not be quoted"; scoring them anyway let
        # the same number be excluded in the report and load-bearing in the headline.
        reason = row.get("reason") or "no reason recorded"
        return Measure.failed(unit, f"{action_name} ran but its own assertion failed: {reason}")

    value = (row.get("timings") or {}).get(timing_key)
    if value is None:
        return Measure.failed(unit, f"{action_name} ran but recorded no {timing_key}")
    return Measure.read(float(value), unit, note = note)


def _frame_measures(windows: Sequence[Mapping[str, Any]]) -> dict[str, Measure]:
    """Pool the active windows of one cell into the three frame metrics.

    Pooled rather than averaged per window: `time_in_jank_pct` is a share of wall time and
    `jank_index` is a sum normalised by wall time, so both are defined over the concatenated
    distribution. Averaging per-window figures would weight a 2 s window equally with a 30 s one.
    """
    unit_by_key = {k: METRIC_BY_KEY[k].unit for k in FRAME_METRICS}

    deltas: list[float] = []
    window_ms = 0.0
    truncated = 0
    attempted_any = False
    max_frame: float | None = None

    for w in windows:
        frames = (w.get("instruments") or {}).get("frames")
        if not isinstance(frames, Mapping):
            continue
        if not frames.get("frames_attempted"):
            continue
        attempted_any = True
        mx = frames.get("max_frame_ms")
        if mx is not None:
            max_frame = float(mx) if max_frame is None else max(max_frame, float(mx))
        if frames.get("frame_gaps_truncated"):
            truncated += 1
            continue
        gaps = frames.get("frame_gaps_ms")
        if not gaps:
            continue
        deltas.extend(float(g) for g in gaps)
        window_ms += float(w.get("duration_ms") or 0.0)

    if not attempted_any:
        reason = "no window in this cell had the frame recorder installed"
        return {k: Measure.not_attempted(unit_by_key[k], reason) for k in FRAME_METRICS}

    out: dict[str, Measure] = {}
    out["max_frame_ms"] = (
        Measure.read(max_frame, "ms", note = "worst frame across the cell's active windows")
        if max_frame is not None
        else Measure.failed("ms", "the recorder ran but observed no frames")
    )

    if not deltas or window_ms <= 0:
        reason = (
            f"{truncated} window(s) exceeded the per-window gap cap, so their distribution was "
            "not exported"
            if truncated
            else "the recorder ran but exported no per-frame deltas"
        )
        for k in ("time_in_jank_pct", "jank_index"):
            out[k] = Measure.failed(unit_by_key[k], reason)
        return out

    stats = compute_frame_stats(deltas, window_ms)
    out["time_in_jank_pct"] = stats.time_in_jank_pct
    out["jank_index"] = stats.jank_index
    return out


def measures_from_records(
    records: Sequence[Mapping[str, Any]], metric_keys: Iterable[str] | None = None
) -> dict[int, dict[str, Measure]]:
    """Build `{rung_tokens: {metric_key: Measure}}` from one run's payload rows.

    A cell that did not complete still contributes its readings. Dropping it would be the same
    mistake as scoring an incomplete rung as NaN: the fact that a build died at 500K is the most
    important thing the run has to say, and it cannot say it if the rung disappears.
    """
    keys = list(metric_keys) if metric_keys is not None else list(METRIC_BY_KEY)
    by_rung: dict[int, dict[str, Measure]] = {}

    for cell in _cell_rows(records):
        cell_id = cell.get("cell_id")
        tokens = cell.get("target_tokens")
        if cell_id is None or tokens is None:
            continue
        rung = int(tokens)

        actions = _actions_for(records, str(cell_id))
        windows = [
            w
            for w in records
            if w.get("row_type") == "window"
            and w.get("cell_id") == cell_id
            and str(w.get("kind") or "") not in IDLE_WINDOW_KINDS
        ]
        frames = _frame_measures(windows)

        readings: dict[str, Measure] = {}
        for key in keys:
            if key in ACTION_SOURCES:
                readings[key] = _action_measure(key, actions)
            elif key in frames:
                readings[key] = frames[key]
            else:
                readings[key] = Measure.not_attempted(
                    METRIC_BY_KEY[key].unit, f"no source is wired for {key}"
                )

        # Repetitions of the same rung: keep the first and let the caller ask for reps
        # explicitly. Silently averaging reps here would hide a bimodal rung.
        by_rung.setdefault(rung, readings)

    return by_rung


def measures_by_cell(
    records: Sequence[Mapping[str, Any]], metric_keys: Iterable[str] | None = None
) -> dict[tuple[int, int], dict[str, Measure]]:
    """`{(rung_tokens, rep): {metric_key: Measure}}` -- one entry per CELL, not per rung.

    `measures_from_records` collapses repetitions because a score is per rung. An A/B must not:
    every repetition is an independent paired observation, and with them collapsed a run with
    `--reps 4` produces one pair per metric, the bootstrap reports "too few pairs", and the
    confidence interval that decides whether a difference is real never has anything to work with.
    """
    keys = list(metric_keys) if metric_keys is not None else list(METRIC_BY_KEY)
    out: dict[tuple[int, int], dict[str, Measure]] = {}

    for cell in _cell_rows(records):
        cell_id = cell.get("cell_id")
        tokens = cell.get("target_tokens")
        if cell_id is None or tokens is None:
            continue
        rep = int((cell.get("cell") or {}).get("rep") or 0)
        single = measures_from_records(
            [cell]
            + [
                r
                for r in records
                if r.get("row_type") in {"action", "window"} and r.get("cell_id") == cell_id
            ],
            keys,
        )
        for readings in single.values():
            out[(int(tokens), rep)] = readings
    return out
