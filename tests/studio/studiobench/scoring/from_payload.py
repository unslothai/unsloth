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

# ── the streaming phase, separated out and normalised per character ─────────────────────────
#
# The three metrics above are not blind to the stream. `_frame_measures` pools every window whose
# kind is not `idle`, and the streaming windows are in that pool. What they cannot do is separate
# it: one 57.3 s film collapses eighteen action windows and the streaming stretch into a single
# number, and the action windows dominate. On a measured 100K null control, `reasoning_toggle`
# alone contributed 2,865 ms of blocked time at 99.3% busy with a 1,866 ms worst frame while the
# streaming stretch beside it ran at 3.6% busy with a 100 ms worst frame.
#
# THE WINDOW KIND CANNOT BE USED TO SEPARATE THEM, and the name is what misleads.
# `SceneRunner._gap_window` opens every inter-slot gap as `kind = "stream"`, so on the standard
# film eighteen windows are named `stream:gapN` and only the first four contain streaming;
# `stream:drain` is opened after the film has finished and measured 7 ms on that same cell. The
# phase is therefore taken from the `stream_cost` instrument, which detects it from the SSE
# traffic, and never from the label.
STREAM_METRICS: tuple[str, ...] = (
    "stream_delta_cost_ms_per_kchar",
    "stream_cost_ms_per_kchar",
    "stream_busy_pct",
    "stream_jank_index",
    "stream_time_in_jank_pct",
    "stream_max_frame_ms",
)

# A window has to carry at least this much streamed text before its cost is divided by it. Below
# it the denominator is small enough that the ratio is dominated by whatever else shared the
# window: on a measured cell an `action:send_turn` window grew the reply by 13 characters while
# accumulating 475 ms of blocked time, which as a rate is 36,500 ms per thousand characters and is
# a statement about opening a menu, not about streaming.
MIN_STREAM_CHARS_PER_WINDOW = 100

# More timer ticks than the clamp says are possible means the clamp is wrong, and every blocked
# figure derived from it is a subtraction against the wrong floor. `clocks_agree` would be the
# gate, but on a headless engine it is null by design (see instruments/pagejs.py: rAF has no vsync
# to be checked against and the screencast is rate-limited), so `timer_clock_ratio` is the sound
# availability signal and this is the bound it has to respect.
MAX_TIMER_CLOCK_RATIO = 1.2

# Windows that are not part of the film, and whose frames therefore say nothing about the build.
#
# `idle` is deliberately quiet: pooling it into the frame metrics would dilute every jank share
# with idle time and make a bad build look average.
#
# `setup` is the opposite and is excluded for the opposite reason. The only one is the composer
# click that starts the film, and most of it is Playwright's injected actionability script --
# selector resolution, visibility, stability and the `elementsFromPoint` hit test -- running on
# the page's own main thread, where it blocks frames indistinguishably from app work. At 500K
# that window alone runs about 11 s against a `max_frame_ms` anchor whose worst case is 2,000 ms,
# so pooling it would peg all three frame metrics on every run, including runs that never asked
# for the click probe.
UNSCORED_WINDOW_KINDS: frozenset[str] = frozenset({"idle", "setup"})

# The window kinds in which NO SCRIPTED ACTION IS RUNNING. `gap` is the scheduler's inter-slot
# wait; `stream` is `stream:drain`, the window the session layer opens after the film to wait the
# reply out. Both are quiet by construction, which is the property `_unaided` needs -- see there
# for why the streaming numbers are taken only from these, and for what the `stream` half is worth.
# `action` is excluded, and `idle` never reaches here (UNSCORED_WINDOW_KINDS strips it first).
UNAIDED_WINDOW_KINDS: frozenset[str] = frozenset({"gap", "stream"})


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
        # ONE WINDOW THAT SAW NOTHING POISONS THE POOL, IT DOES NOT DROP OUT OF IT.
        #
        # A window whose recorder was installed and exported no deltas at all is the rAF-
        # unscheduled trap, and `compute_frame_stats` already refuses to score it: a single such
        # window reads `Measure.failed` on every metric with `no_frames_recorded` set, never zero
        # jank. Pooled, the same window was skipped by the `continue` above -- it contributed no
        # deltas, no wall time and (its `max_frame_ms` being null) no worst frame -- so the
        # REMAINING windows answered for the whole cell and a complete freeze during one action
        # came back as clean numbers: a 4 s frozen window beside a smooth one scored 0.0% time in
        # jank and a 40 ms worst frame, byte-identical to the cell without the freeze in it.
        # An unmeasured window is not an absent one, so the cell's frame metrics fail here for
        # the same reason and in the same shape as the single-window path.
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
    """The windows that carried streaming, and why each rejected one was rejected.

    A window qualifies when the `stream_cost` instrument SAW SSE traffic in it and the reply it was
    feeding grew by a usable amount. Both halves are needed. Traffic alone admits the tail window
    in which the stream ended after 400 ms and the remaining twelve seconds were idle; growth alone
    admits `thread_reopen`, which rebuilds the whole thread and grows the character count by tens
    of thousands without a byte of it having been streamed.
    """
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
        # THE INSTRUMENT'S OWN VERDICT ON ITS DENOMINATOR, and it is consulted here because this is
        # the only place that can act on it. `instruments/streamcost.py` marks a window unscoreable
        # when an SSE frame failed to parse inside it or an unterminated frame was still buffered
        # at its close -- and per the HTML standard an event is dispatched only at the blank line
        # that terminates it, so those characters have not been counted and cannot be recovered.
        # The delta is therefore short by an unknown amount, and every cost-per-character divided
        # by it comes out inflated. Publishing the flag and then summing the delta anyway left the
        # official metrics derived from a denominator the instrument had already disowned.
        #
        # `is False` and not falsiness: a payload recorded before the flag existed carries no key
        # at all, and those windows are admitted exactly as they were rather than voided wholesale.
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
    """Of the streaming windows, the ones with no scripted action running in them.

    EVERY streaming quantity is taken from these, including the targeted numerator, and that is a
    correction the measurements forced rather than a position held from the start.

    The first version of this metric fed `delta_task_ms` from every streaming window, on the
    reasoning that a task chain an SSE chunk started is attributable to the stream wherever it
    happened. That reasoning is wrong, and a standard-tier 10K null shows exactly how. The chain
    is measured from the chunk to the moment the event loop next reaches a macrotask, so ANY work
    that lands in between is charged to it. Three of the film's slots run during generation on
    purpose, and in `action:keystroke` on one cell the chain cost 23.77 ms per burst against 1.69
    ms in the gap windows either side of it. That is typing, billed to the stream.

    The window-wide quantities -- blocked time, the frame distribution, the worst frame -- fail
    the same way and more obviously, because they charge the whole window. On a fast-tier 100K
    null, admitting the action windows put a 1,738 ms worst frame into `stream_max_frame_ms` when
    the unaided stretch beside it peaked at 286 ms: a scroll, reported as a streaming stall.

    So the streaming phase, for scoring, is the quiet stretches where the stream is doing its work
    unaided. Measured against the alternative on the same payload this also has the narrower null
    floor (32.9% against 36.0%), which is the weaker argument of the two but points the same way.

    Restricting FURTHER -- to the opening turn only -- was tried and is much worse (101.5%),
    because fewer windows average less and one outlier then owns the cell. More streaming windows
    is better as long as every one of them is unaided.

    UNAIDED IS NOT THE SAME PREDICATE AS `kind == "gap"`, which is what this used to test. The
    session layer opens one more quiet window that the scheduler does not: `stream:drain`, with
    `kind = "stream"`, held open after the film to wait the reply out. On the default fixture it
    carries nothing -- the tail is pinned at 6,000 characters, drains in 14 to 18 s against a
    243 s standard film, and the measured drain window was 7 ms long -- so `_stream_windows`
    rejects it for having seen no SSE traffic and the distinction never showed. It shows the
    moment `--stream-tail-chars` is used, which is the one supported way to make the reply long:
    at 96,000 characters the reply streams for 291 s at field cadence, so roughly 48 s of it lands
    AFTER the last slot has closed, in the drain window, with nothing scripted running in it. That
    stretch is unaided streaming by every part of the definition above, and dropping it dropped
    the characters and the cost of the LAST fifth of the reply -- the part streamed into the
    largest thread, so the most expensive part -- out of every streaming metric. Nor is
    `stream_max_frame_ms` a ratio that might absorb it: a worst frame in that stretch was simply
    never seen.

    The kind filter is still what does the work, because it is the only thing that separates a
    quiet window from an action window. It now names both quiet kinds instead of one.
    """
    return [w for w in windows if str(w.get("kind") or "") in UNAIDED_WINDOW_KINDS]


def _stream_measures(windows: Sequence[Mapping[str, Any]]) -> dict[str, Measure]:
    """The streaming phase alone, integrated, and divided by the characters it streamed.

    TWO NUMERATORS, deliberately, because they fail in opposite directions and a reader should be
    able to see both:

      `stream_delta_cost_ms_per_kchar` is TARGETED. It sums only the main-thread task chains that
      SSE chunks start, so it excludes the background churn -- async highlighting, GC, the app's
      own timers -- that a whole-window figure charges to the stream. It is the sharper of the two
      and the one a change to the delta path should move.

      `stream_cost_ms_per_kchar` is BROAD. It sums blocked time over the streaming stretch, so it
      catches stream-driven cost that lands outside the delta's own task chain, which is most of
      the asynchronous work. It is the honest total and the noisier of the two.

    Both are `null` with a reason rather than zero when the timer clamp was never established:
    blocked time is a subtraction against an idle floor, and without a floor the quantity does not
    exist.
    """
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

    # A CELL WHOSE RECORDER DIED IS A TRUNCATED CELL, not a short one. `rejected` above is read
    # only when NOTHING qualified, so once one window has qualified every later rejection is
    # discarded -- including the one that says the page went away. The streaming metrics are
    # integrals divided by the characters that were streamed, and both halves are then missing the
    # same unmeasured stretch, so the result is not a wide error bar, it is a number computed over
    # whatever ran before the crash and reported as if it described the cell.
    #
    # Measured on the payload corpus: 138 unaided windows record `unavailable` while a qualifying
    # window precedes them, carrying `TargetClosedError: Page.evaluate: Target page, context or
    # browser has been closed` and `Error: Page.evaluate: Target crashed`. They are all AFTER the
    # last qualifying window, which is not evidence that they are ordinary end-of-stream windows:
    # a crash is trailing by construction, because nothing can qualify once the page is gone. They
    # are also long, a median of 11.0 s and up to 36.0 s, so what went unmeasured is a real
    # fraction of the reply and not a rounding edge.
    #
    # `unavailable` was consumed nowhere in this module before this, so the crash signal the
    # instrument already emits was being dropped in full. Poisoning here costs 12 cells of 1,461
    # that currently publish, and every one of the 12 is a genuine page crash.
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

    # Every streaming quantity comes from the UNAIDED windows. See _unaided for why the targeted
    # numerator is not exempt from that, which is the one thing here that measurement overturned.
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
        # THE SAME RULE AS `_frame_measures`, ON THE SAME SHAPE OF WINDOW, AND FOR THE SAME REASON.
        #
        # `instruments/frames.js` emits `frames_attempted: true` with `frame_gaps_ms: []` and a
        # null `max_frame_ms` whenever the rAF loop was never scheduled in the window, and rAF
        # going unscheduled is a rendering fact, not a quiet one: on a headless engine the loop
        # runs off the rendering pipeline, so a renderer that stalls stops delivering callbacks
        # while SSE keeps arriving. Skipped rather than refused, such a window contributes no
        # deltas, no wall time and no worst frame, so the REMAINING unaided windows answer for the
        # whole streaming stretch: one frozen window beside one smooth one reported a 16.7 ms worst
        # frame and 0.0% time in jank, byte-identical to the cell without the freeze in it.
        # An unmeasured window is not an absent one. Only the three FRAME metrics are poisoned --
        # the cost, busy and character figures come from `stream_cost`, which measured this window
        # perfectly well.
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
            and str(w.get("kind") or "") not in UNSCORED_WINDOW_KINDS
        ]
        frames = _frame_measures(windows)
        # The streaming phase is NOT in METRIC_BY_KEY, and that is deliberate rather than an
        # omission. The anchor table is hashed into `weights_id`, and a report that compares two
        # runs with different `weights_id` values is refused; adding a seventh weighted metric
        # would change every existing run's composite score and make it incomparable with every
        # run this tool has already taken. So these are scored through the per-metric floor table,
        # which judges a metric on its own floor, and the composite score is left alone. Giving
        # them anchors and a weight is a separate decision that should be argued in anchors.py.
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


# ── was there an instrument in the shot ──────────────────────────────


def probe_scripts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every external init script this payload records, in order, without duplicates.

    EVERY `run_meta`, not the first one. `--resume` continues an interrupted run by APPENDING to
    the existing payload, so a file can carry a clean `run_meta` at the top and a second one
    further down with a probe named in it, above the cells that were re-recorded under that
    probe. Returning on the first row reads such a file as clean and scores perturbed cells.

    The failed `probe_free` gate is read as well as the metadata field. Two independent records of
    one fact, so a payload written by a version that emits only one of them is still refused.
    """
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
    """Raise rather than score a payload that was recorded with a probe in the page.

    Called from every scoring entry point rather than from one of them. A refusal that only
    `floor_table` performs still lets the run print an `ab.md` at the end and `--report` produce a
    score from the same file afterwards, and those are the two tables somebody actually reads.
    """
    scripts = probe_scripts(records)
    if not scripts:
        return
    raise SystemExit(
        f"refusing to score {where}: it was recorded with an external init script "
        f"installed ({', '.join(scripts)}). A probe samples the DOM and forces layout "
        f"on its own schedule, so these timings are a measurement of the page and the "
        f"instrument together. Re-run with SBENCH_EXTRA_INIT_SCRIPT unset."
    )
