# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam from payload rows to scoring Measures; a missing reading is never a good one, nor a zero."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .anchors import METRIC_BY_KEY
from .frames import compute_frame_stats
from .schema import Measure


# (action name, timing key); anything not listed comes from the window frame recorder.

ACTION_SOURCES: Mapping[str, tuple[str, str]] = {
    "keystroke_p95_ms": ("keystroke", "p95_ms"),
    "menu_open_ms": ("message_menu", "open_ms"),
    "scroll_settle_ms": ("scroll_after", "gesture_ms"),
}

# Scene never measures settle time, so `gesture_ms` stands in for the anchor's settle metric.
SCROLL_SETTLE_NOTE = (
    "recorded as scroll_after.gesture_ms; the scene does not measure settle separately, so this "
    "is gesture duration, not post-gesture settle"
)

FRAME_METRICS: tuple[str, ...] = ("time_in_jank_pct", "jank_index", "max_frame_ms")

# Streaming phase, normalised per character. Taken from `stream_cost` (SSE traffic), never
# from the window kind: `_gap_window` labels every inter-slot gap `stream:gapN`.
STREAM_METRICS: tuple[str, ...] = (
    "stream_delta_cost_ms_per_kchar",
    "stream_cost_ms_per_kchar",
    "stream_busy_pct",
    "stream_jank_index",
    "stream_time_in_jank_pct",
    "stream_max_frame_ms",
)

# Below this the per-char ratio is dominated by whatever else shared the window.
MIN_STREAM_CHARS_PER_WINDOW = 100

# `clocks_agree` is null on headless engines, so `timer_clock_ratio` gates the clamp instead.
MAX_TIMER_CLOCK_RATIO = 1.2

# `idle` would dilute jank shares; `setup` is Playwright's actionability script (~11 s at 500K).
UNSCORED_WINDOW_KINDS: frozenset[str] = frozenset({"idle", "setup"})

# Kinds where no scripted action runs: `gap` (inter-slot wait) and `stream` (`stream:drain`).
UNAIDED_WINDOW_KINDS: frozenset[str] = frozenset({"gap", "stream"})


ATTEMPT_ROW_TYPES: frozenset[str] = frozenset({"cell", "action", "window"})


def latest_attempt_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Latest attempt is the last session to write any row; a killed run never writes its terminal row."""
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
    """The embedded actions list drops names, so only the standalone action rows can be matched by name."""
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
        return Measure.failed(unit, f"{action_name} did not run: {reason}")
    if row.get("expect_ok") is False:
        # A failed own-assertion means the timing describes something else; the report excludes these.
        reason = row.get("reason") or "no reason recorded"
        return Measure.failed(unit, f"{action_name} ran but its own assertion failed: {reason}")

    value = (row.get("timings") or {}).get(timing_key)
    if value is None:
        return Measure.failed(unit, f"{action_name} ran but recorded no {timing_key}")
    return Measure.read(float(value), unit, note = note)


def _frame_measures(windows: Sequence[Mapping[str, Any]]) -> dict[str, Measure]:
    """Pooled, not averaged per window: both are shares of wall time, so windows must not weigh equally."""
    unit_by_key = {k: METRIC_BY_KEY[k].unit for k in FRAME_METRICS}

    deltas: list[float] = []
    window_ms = 0.0
    truncated = 0
    frameless = 0
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
            frameless += 1
            continue
        deltas.extend(float(g) for g in gaps)
        window_ms += float(w.get("duration_ms") or 0.0)

    if not attempted_any:
        reason = "no window in this cell had the frame recorder installed"
        return {k: Measure.not_attempted(unit_by_key[k], reason) for k in FRAME_METRICS}

    if frameless:
        # An unscheduled-rAF window poisons the pool instead of being skipped, or a freeze scores clean.
        reason = (
            f"{frameless} window(s) recorded no frames at all (rAF may be unscheduled), so the "
            "pooled frame metrics would describe only the windows that were measured"
        )
        return {k: Measure.failed(unit_by_key[k], reason) for k in FRAME_METRICS}

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


def _stream_windows(windows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict]:
    """Needs SSE traffic and reply growth together; either alone admits idle tails or a rebuilt thread."""
    picked: list[Mapping[str, Any]] = []
    rejected: dict[str, int] = {}

    def reject(why: str) -> None:
        rejected[why] = rejected.get(why, 0) + 1

    for w in windows:
        inst = w.get("instruments") or {}
        sc = inst.get("stream_cost")
        if not isinstance(sc, Mapping) or not sc.get("stream_cost_attempted"):
            reject("the stream_cost instrument did not run in this window")
            continue
        if not sc.get("streaming_observed"):
            reject("no SSE traffic reached the page during this window")
            continue
        delta = sc.get("reply_chars_delta")
        if delta is None:
            reject(
                str(sc.get("reply_chars_delta_reason") or "the reply's growth was not measurable")
            )
            continue
        # Unparsed or unterminated SSE frames lose characters, inflating per-char cost.
        # `is False` so payloads recorded before the flag existed are still admitted.
        if sc.get("reply_chars_scoreable") is False:
            reject(
                str(
                    sc.get("reply_chars_unscoreable_reason")
                    or "the instrument marked this window's wire character count unscoreable"
                )
            )
            continue
        if int(delta) < MIN_STREAM_CHARS_PER_WINDOW:
            reject(f"the reply grew by fewer than {MIN_STREAM_CHARS_PER_WINDOW} characters")
            continue
        frames = inst.get("frames")
        if isinstance(frames, Mapping):
            if frames.get("clocks_agree") is False:
                reject("the window's clocks disagreed, so it is not scoreable")
                continue
            ratio = frames.get("timer_clock_ratio")
            if isinstance(ratio, (int, float)) and float(ratio) > MAX_TIMER_CLOCK_RATIO:
                reject("more timer ticks than the calibrated clamp allows, so the clamp is wrong")
                continue
        picked.append(w)
    return picked, rejected


def _unaided(windows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Only gap and stream:drain windows are unaided; action windows charge their own work to the stream."""
    return [w for w in windows if str(w.get("kind") or "") in UNAIDED_WINDOW_KINDS]


def _stream_measures(windows: Sequence[Mapping[str, Any]]) -> dict[str, Measure]:
    """Two numerators: targeted delta-task cost (sharper) and broad blocked time (honest total, noisier)."""
    unit_by_key = {
        "stream_delta_cost_ms_per_kchar": "ms/kchar",
        "stream_cost_ms_per_kchar": "ms/kchar",
        "stream_busy_pct": "%",
        "stream_jank_index": "ms",
        "stream_time_in_jank_pct": "%",
        "stream_max_frame_ms": "ms",
    }
    picked, rejected = _stream_windows(windows)
    if not picked:
        why = (
            "; ".join(f"{n} window(s): {r}" for r, n in sorted(rejected.items()))
            or "this cell recorded no windows"
        )
        return {
            k: Measure.not_attempted(u, f"no window in this cell carried streaming ({why})")
            for k, u in unit_by_key.items()
        }

    # A recorder crash after a qualifying window truncates the cell; poison it rather than score
    # the partial integral.
    crashed = sorted(
        {
            str(sc.get("unavailable"))
            for w in _unaided(windows)
            if isinstance((sc := (w.get("instruments") or {}).get("stream_cost")), Mapping)
            and not sc.get("stream_cost_attempted")
            and sc.get("unavailable")
        }
    )
    if crashed:
        reason = (
            "the stream_cost recorder stopped partway through this cell "
            f"({'; '.join(crashed)}), so the streaming metrics would describe only the part of "
            "the reply that streamed before it went away"
        )
        return {k: Measure.failed(u, reason) for k, u in unit_by_key.items()}

    unaided = _unaided(picked)
    chars = 0
    delta_task_ms = 0.0
    for w in unaided:
        sc = (w.get("instruments") or {}).get("stream_cost") or {}
        chars += int(sc.get("reply_chars_delta") or 0)
        delta_task_ms += float(sc.get("delta_task_ms") or 0.0)

    unaided_chars = 0
    blocked_ms = 0.0
    blocked_reason: str | None = None
    streaming_ms = 0.0
    deltas: list[float] = []
    window_ms = 0.0
    max_frame: float | None = None
    frameless = 0

    for w in unaided:
        inst = w.get("instruments") or {}
        sc = inst.get("stream_cost") or {}
        unaided_chars += int(sc.get("reply_chars_delta") or 0)
        streaming_ms += float(sc.get("streaming_ms") or 0.0)
        blocked = sc.get("stream_blocked_ms")
        if blocked is None:
            blocked_reason = str(
                sc.get("stream_blocked_ms_reason") or "blocked time was not measurable"
            )
        else:
            blocked_ms += float(blocked)

        frames = inst.get("frames")
        if not isinstance(frames, Mapping) or not frames.get("frames_attempted"):
            continue
        mx = frames.get("max_frame_ms")
        if mx is not None:
            max_frame = float(mx) if max_frame is None else max(max_frame, float(mx))
        if frames.get("frame_gaps_truncated"):
            continue
        gaps = frames.get("frame_gaps_ms")
        if not gaps:
            frameless += 1
            continue
        deltas.extend(float(g) for g in gaps)
        window_ms += float(w.get("duration_ms") or 0.0)

    out: dict[str, Measure] = {}
    note = f"{len(unaided)} unaided streaming window(s), {chars} streamed characters"
    unaided_note = (
        f"{len(unaided)} unaided streaming window(s), {unaided_chars} streamed characters"
    )

    out["stream_delta_cost_ms_per_kchar"] = (
        Measure.read(1000.0 * delta_task_ms / chars, "ms/kchar", note = note)
        if chars > 0
        else Measure.failed("ms/kchar", "the streaming windows recorded no streamed characters")
    )
    if blocked_reason:
        out["stream_cost_ms_per_kchar"] = Measure.failed("ms/kchar", blocked_reason)
    elif unaided_chars <= 0:
        out["stream_cost_ms_per_kchar"] = Measure.failed(
            "ms/kchar",
            "no window streamed without a scripted action running in it, so there is no "
            "unaided streaming cost to divide",
        )
    else:
        out["stream_cost_ms_per_kchar"] = Measure.read(
            1000.0 * blocked_ms / unaided_chars, "ms/kchar", note = unaided_note
        )

    out["stream_busy_pct"] = (
        Measure.failed("%", blocked_reason)
        if blocked_reason
        else (
            Measure.read(100.0 * blocked_ms / streaming_ms, "%", note = unaided_note)
            if streaming_ms > 0
            else Measure.failed(
                "%", "the instrument observed no unaided streaming time in this cell"
            )
        )
    )
    if frameless:
        # Same rule as `_frame_measures`: an unscheduled rAF loop poisons the three frame metrics only.
        reason = (
            f"{frameless} unaided streaming window(s) recorded no frames at all (rAF may be "
            "unscheduled), so the pooled streaming frame metrics would describe only the windows "
            "that were measured"
        )
        out["stream_max_frame_ms"] = Measure.failed("ms", reason)
        out["stream_time_in_jank_pct"] = Measure.failed("%", reason)
        out["stream_jank_index"] = Measure.failed("ms", reason)
        return out

    out["stream_max_frame_ms"] = (
        Measure.read(max_frame, "ms", note = "worst frame inside the UNAIDED streaming windows")
        if max_frame is not None
        else Measure.failed("ms", "the frame recorder observed no frames streaming unaided")
    )

    if deltas and window_ms > 0:
        stats = compute_frame_stats(deltas, window_ms)
        out["stream_time_in_jank_pct"] = stats.time_in_jank_pct
        out["stream_jank_index"] = stats.jank_index
    else:
        reason = "the unaided streaming windows exported no per-frame deltas"
        out["stream_time_in_jank_pct"] = Measure.failed("%", reason)
        out["stream_jank_index"] = Measure.failed("ms", reason)
    return out


def measures_from_records(
    records: Sequence[Mapping[str, Any]], metric_keys: Iterable[str] | None = None
) -> dict[int, dict[str, Measure]]:
    """Incomplete cells still contribute readings: dropping a rung that died would hide the failure."""
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
            and str(w.get("kind") or "") not in UNSCORED_WINDOW_KINDS
        ]
        frames = _frame_measures(windows)
        # Not in METRIC_BY_KEY on purpose: anchors are hashed into `weights_id`, so adding one would
        # make every existing run incomparable.
        stream = _stream_measures(windows)

        readings: dict[str, Measure] = {}
        for key in keys:
            if key in ACTION_SOURCES:
                readings[key] = _action_measure(key, actions)
            elif key in frames:
                readings[key] = frames[key]
            elif key in stream:
                readings[key] = stream[key]
            else:
                readings[key] = Measure.not_attempted(
                    METRIC_BY_KEY[key].unit, f"no source is wired for {key}"
                )

        # Keep the first rep; averaging here would hide a bimodal rung.
        by_rung.setdefault(rung, readings)

    return by_rung


def measures_by_cell(
    records: Sequence[Mapping[str, Any]], metric_keys: Iterable[str] | None = None
) -> dict[tuple[int, int], dict[str, Measure]]:
    """Keyed per cell, not per rung, so every repetition stays an independent pair for the A/B bootstrap."""
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


def probe_scripts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reads every run_meta, since --resume appends a second header that may carry a probe."""
    found: list[str] = []
    for row in records:
        script = ""
        if row.get("row_type") == "run_meta":
            script = str(row.get("probe_init_script") or "")
        elif row.get("row_type") == "gate" and row.get("name") == "probe_free":
            if not row.get("passed"):
                detail = row.get("detail")
                detail = detail if isinstance(detail, Mapping) else {}
                script = str(detail.get("probe_init_script") or "an unnamed probe")
        if script and script not in found:
            found.append(script)
    return found


def refuse_if_probed(records: Sequence[Mapping[str, Any]], where: str) -> None:
    """Called from every scoring entry point, so no report can be produced from a probed payload."""
    scripts = probe_scripts(records)
    if not scripts:
        return
    raise SystemExit(
        f"refusing to score {where}: it was recorded with an external init script "
        f"installed ({', '.join(scripts)}). A probe samples the DOM and forces layout "
        f"on its own schedule, so these timings are a measurement of the page and the "
        f"instrument together. Re-run with SBENCH_EXTRA_INIT_SCRIPT unset."
    )
