# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streaming-phase metrics, and the ways they are supposed to REFUSE to produce a number.

Half of this file tests the failure direction on purpose. A gate that only ever passes is worse
than no gate, because it gets cited: the harness has three separate occasions on record where code
that could never fire was reported as "no effect", and every one of them would have been caught by
a test that asserted a rejection rather than an acceptance.

The rejections that matter here, each of which corresponds to a real shape in a real payload:

  * a window named `stream:gapN` with no SSE traffic in it. Eighteen of these exist per standard
    film and only the first four carry streaming, so a metric that trusts the label measures
    mostly post-stream idle.
  * `action:send_turn`, which grows the reply by about a dozen characters while spending hundreds
    of milliseconds opening things. Divided out, that is tens of thousands of ms per thousand
    characters, and it is a statement about a button.
  * `action:thread_reopen`, which rebuilds the thread so the last assistant message is REPLACED.
    Its character count can fall, and a signed delta would subtract from the denominator.
  * a cell whose timer clamp was never established, where blocked time is a subtraction against a
    floor that does not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.scoring.from_payload import (  # noqa: E402
    MIN_STREAM_CHARS_PER_WINDOW,
    STREAM_METRICS,
    _stream_measures,
    _stream_windows,
)


def _window(
    name: str,
    *,
    kind: str = "gap",
    duration_ms: float = 10_000.0,
    streaming_observed: bool = True,
    streaming_ms: float = 9_000.0,
    delta_task_ms: float = 900.0,
    blocked_ms: float | None = 1_800.0,
    chars_open: int | None = 0,
    chars_close: int | None = 3_000,
    attempted: bool = True,
    frame_gaps: list[float] | None = None,
    max_frame_ms: float | None = 100.0,
    clocks_agree: bool | None = None,
    timer_clock_ratio: float | None = 0.95,
) -> dict:
    sc: dict = {
        "stream_cost_attempted": attempted,
        "streaming_observed": streaming_observed,
        "streaming_ms": streaming_ms,
        "delta_task_ms": delta_task_ms,
        "sse_chunks": 120 if streaming_observed else 0,
        "reply_chars_open": chars_open,
        "reply_chars_close": chars_close,
    }
    if blocked_ms is None:
        sc["stream_blocked_ms"] = None
        sc["stream_blocked_ms_reason"] = "no timer clamp was established"
    else:
        sc["stream_blocked_ms"] = blocked_ms
        sc["stream_blocked_ms_attempted"] = True
    if chars_open is None or chars_close is None:
        sc["reply_chars_delta"] = None
        sc["reply_chars_delta_reason"] = "no assistant message was on screen"
    elif chars_close < chars_open:
        sc["reply_chars_delta"] = None
        sc["reply_chars_delta_reason"] = "the last assistant message shrank"
    else:
        sc["reply_chars_delta"] = chars_close - chars_open
        sc["reply_chars_delta_attempted"] = True

    return {
        "row_type": "window",
        "name": name,
        "kind": kind,
        "duration_ms": duration_ms,
        "instruments": {
            "stream_cost": sc,
            "frames": {
                "frames_attempted": True,
                "frame_gaps_ms": frame_gaps if frame_gaps is not None else [16.7] * 540,
                "frame_gaps_truncated": False,
                "max_frame_ms": max_frame_ms,
                "clocks_agree": clocks_agree,
                "timer_clock_ratio": timer_clock_ratio,
            },
        },
        "notes": {},
    }


# ── what is selected as "the streaming phase" ────────────────────────────────────────────────


def test_a_window_with_traffic_and_growth_is_the_streaming_phase():
    picked, rejected = _stream_windows([_window("stream:gap1")])
    assert len(picked) == 1
    assert rejected == {}


def test_a_stream_named_window_with_no_traffic_is_refused():
    """THE TRAP. Every inter-slot gap is opened with kind `stream`, and most carry no streaming."""
    picked, rejected = _stream_windows(
        [_window("stream:gap12", streaming_observed = False, chars_open = 0, chars_close = 0)]
    )
    assert picked == []
    assert any("no SSE traffic" in reason for reason in rejected)


def test_stream_drain_after_the_film_is_refused():
    """`stream:drain` is opened after the film; on a measured 100K cell it was 7 ms long."""
    picked, _ = _stream_windows(
        [
            _window(
                "stream:drain",
                duration_ms = 7.0,
                streaming_observed = False,
                streaming_ms = 0.0,
                chars_open = 194_783,
                chars_close = 194_783,
            )
        ]
    )
    assert picked == []


def test_the_drain_is_scored_when_the_reply_outlasts_the_film():
    """The other half of the drain window, and the half `--stream-tail-chars` creates.

    The test above pins the DEFAULT shape: the tail is pinned at 6,000 characters, the reply is
    finished forty seconds before the film ends, and the drain window is 7 ms of nothing that
    `_stream_windows` refuses for having seen no traffic. Raise the tail to 96,000 -- the one
    supported way to vary reply length -- and the reply streams for 291 s at field cadence against
    a 243 s standard film, so the last 48 s of it lands in the drain window with nothing scripted
    running in it.

    That is unaided streaming, and it was dropped: `_unaided` selected `kind == "gap"` and the
    drain window is `kind = "stream"`. The characters and the cost of the last fifth of the reply
    -- the part streamed into the largest thread -- left every streaming metric, and the worst
    frame in that stretch was never seen at all.
    """
    gap = _window("stream:gap3", kind = "gap", delta_task_ms = 900.0, chars_close = 3_000)
    drain = _window(
        "stream:drain",
        kind = "stream",
        duration_ms = 48_000.0,
        streaming_ms = 47_000.0,
        delta_task_ms = 2_700.0,
        blocked_ms = 5_400.0,
        chars_open = 0,
        chars_close = 9_000,
        max_frame_ms = 640.0,
    )
    picked, _ = _stream_windows([drain])
    assert len(picked) == 1, "a drain window that really carried traffic is a streaming window"

    m = _stream_measures([gap, drain])
    assert "2 unaided streaming window(s)" in m["stream_delta_cost_ms_per_kchar"].note
    assert "12000 streamed characters" in m["stream_delta_cost_ms_per_kchar"].note
    # 3,600 ms over 12,000 characters, not 900 ms over 3,000: the drain is in the integral.
    assert m["stream_delta_cost_ms_per_kchar"].value == pytest.approx(300.0)
    assert m["stream_cost_ms_per_kchar"].value == pytest.approx(600.0)
    # A max is not a ratio, so a dropped window is a silently missing worst frame.
    assert m["stream_max_frame_ms"].value == pytest.approx(640.0)


def test_an_action_window_is_still_not_unaided():
    """The kind filter widened by one QUIET kind, and must not have widened to the action windows.

    This is the property `_unaided` exists for. On a fast-tier 100K null, admitting the action
    windows put a 1,738 ms scroll frame into `stream_max_frame_ms` against a 286 ms unaided peak.
    """
    windows = [
        _window("stream:gap3", kind = "gap", max_frame_ms = 286.0),
        _window("stream:drain", kind = "stream", max_frame_ms = 300.0),
        _window("action:scroll_during_generation", kind = "action", max_frame_ms = 1_738.0),
    ]
    m = _stream_measures(windows)
    assert m["stream_max_frame_ms"].value == pytest.approx(300.0)
    assert "2 unaided streaming window(s)" in m["stream_delta_cost_ms_per_kchar"].note


def test_a_window_whose_reply_shrank_is_refused_rather_than_signed():
    """`send_turn` and `thread_reopen` REPLACE the last assistant message."""
    picked, rejected = _stream_windows(
        [_window("action:thread_reopen", kind = "action", chars_open = 96_000, chars_close = 5)]
    )
    assert picked == []
    assert any("shrank" in reason for reason in rejected)


def test_a_trivial_character_gain_is_refused():
    """The measured `action:send_turn` shape: 13 characters against 475 ms of blocked time."""
    picked, rejected = _stream_windows(
        [
            _window(
                "action:send_turn",
                kind = "action",
                chars_open = 193_548,
                chars_close = 193_561,
                blocked_ms = 475.3,
            )
        ]
    )
    assert picked == []
    assert any(str(MIN_STREAM_CHARS_PER_WINDOW) in reason for reason in rejected)


def test_a_window_whose_clocks_disagree_is_excluded_from_scoring():
    picked, rejected = _stream_windows([_window("stream:gap1", clocks_agree = False)])
    assert picked == []
    assert any("clocks disagreed" in reason for reason in rejected)


def test_more_timer_ticks_than_the_clamp_allows_is_excluded():
    picked, rejected = _stream_windows([_window("stream:gap1", timer_clock_ratio = 1.9)])
    assert picked == []
    assert any("clamp is wrong" in reason for reason in rejected)


def _unscoreable(window: dict, reason: str = "") -> dict:
    """What `instruments/streamcost.py` writes when a frame failed to parse inside the window, or
    an unterminated one was still buffered at its close."""
    sc = window["instruments"]["stream_cost"]
    sc["reply_chars_scoreable"] = False
    sc["wire_parse_failures_in_window"] = 1
    sc["wire_pending_chars_at_close"] = 41
    sc["reply_chars_unscoreable_reason"] = (
        reason or "the wire character count is short by an unknown amount"
    )
    return window


def test_a_window_the_instrument_marked_unscoreable_is_excluded_from_scoring():
    """THE DEFECT. The instrument publishes `reply_chars_scoreable: false` when it knows the wire
    count is short -- an SSE event is dispatched only at the blank line that terminates it, so an
    unterminated frame at the close is characters delivered and not counted -- and the scoring path
    summed the delta anyway. Every official cost-per-character was then divided by a denominator
    the instrument had already disowned, which inflates it by an unknown factor.
    """
    picked, rejected = _stream_windows([_unscoreable(_window("stream:gap1"))])
    assert picked == []
    assert any("short by an unknown amount" in reason for reason in rejected)


def test_an_unscoreable_window_does_not_reach_the_cost_per_character():
    """The rate is not merely nudged by an unscoreable window: with nothing else in the cell there
    is no rate at all, and the reason travels instead of a number."""
    m = _stream_measures([_unscoreable(_window("stream:gap1"))])
    for key in STREAM_METRICS:
        assert m[key].value is None, (key, m[key])
    assert "short by an unknown amount" in (m["stream_delta_cost_ms_per_kchar"].note or "")


def test_a_payload_recorded_before_the_flag_existed_is_scored_exactly_as_before():
    """`is False`, not falsiness. Rows written before the instrument published the flag carry no
    key at all, and voiding them would retro-actively delete every streaming metric this project
    has already recorded."""
    w = _window("stream:gap1")
    assert "reply_chars_scoreable" not in w["instruments"]["stream_cost"]
    picked, rejected = _stream_windows([w])
    assert len(picked) == 1
    assert rejected == {}


def test_a_window_the_instrument_scored_is_still_admitted():
    """The control. A flag that is present and true changes nothing."""
    w = _window("stream:gap1")
    w["instruments"]["stream_cost"]["reply_chars_scoreable"] = True
    picked, rejected = _stream_windows([w])
    assert len(picked) == 1
    assert rejected == {}


def test_a_window_the_instrument_never_ran_in_is_refused():
    picked, rejected = _stream_windows([_window("stream:gap1", attempted = False)])
    assert picked == []
    assert any("did not run" in reason for reason in rejected)


# ── the numbers themselves ───────────────────────────────────────────────────────────────────


def test_the_rate_is_cost_per_thousand_streamed_characters():
    m = _stream_measures([_window("stream:gap1", delta_task_ms = 900.0, chars_close = 3_000)])
    assert m["stream_delta_cost_ms_per_kchar"].value == pytest.approx(300.0)


def test_cost_integrates_across_several_streaming_windows():
    """Two windows of 900 ms over 3,000 characters each is the same RATE as one of them."""
    one = _stream_measures([_window("stream:gap1")])
    two = _stream_measures([_window("stream:gap1"), _window("stream:gap3")])
    assert two["stream_delta_cost_ms_per_kchar"].value == pytest.approx(
        one["stream_delta_cost_ms_per_kchar"].value
    )
    # ... but the integral itself doubled, which is what the note records.
    assert "6000 streamed characters" in two["stream_delta_cost_ms_per_kchar"].note


def test_doubling_the_per_frame_cost_doubles_the_metric():
    """The detection property. A metric that cannot move with an injected cost cannot detect one."""
    base = _stream_measures([_window("stream:gap1", delta_task_ms = 900.0)])
    heavy = _stream_measures([_window("stream:gap1", delta_task_ms = 1_800.0)])
    ratio = (
        heavy["stream_delta_cost_ms_per_kchar"].value / base["stream_delta_cost_ms_per_kchar"].value
    )
    assert ratio == pytest.approx(2.0)


def test_a_longer_reply_at_the_same_per_character_cost_reads_the_same_rate():
    """The point of normalising. Twice the characters at twice the cost is not a regression."""
    short = _stream_measures([_window("stream:gap1", delta_task_ms = 900.0, chars_close = 3_000)])
    long = _stream_measures([_window("stream:gap1", delta_task_ms = 1_800.0, chars_close = 6_000)])
    assert short["stream_delta_cost_ms_per_kchar"].value == pytest.approx(
        long["stream_delta_cost_ms_per_kchar"].value
    )


def test_the_broad_and_targeted_numerators_are_different_quantities():
    m = _stream_measures([_window("stream:gap1", delta_task_ms = 900.0, blocked_ms = 1_800.0)])
    assert m["stream_delta_cost_ms_per_kchar"].value == pytest.approx(300.0)
    assert m["stream_cost_ms_per_kchar"].value == pytest.approx(600.0)


def test_busy_percent_is_over_the_streaming_time_not_the_window():
    m = _stream_measures(
        [_window("stream:gap1", duration_ms = 18_000.0, streaming_ms = 9_000.0, blocked_ms = 1_800.0)]
    )
    assert m["stream_busy_pct"].value == pytest.approx(20.0)


def test_the_worst_streaming_frame_excludes_the_action_windows():
    """The separation this exists for: a 1,866 ms reasoning_toggle frame is not a streaming frame."""
    windows = [
        _window("stream:gap1", max_frame_ms = 100.2),
        _window(
            "action:reasoning_toggle",
            kind = "action",
            streaming_observed = False,
            chars_open = 0,
            chars_close = 0,
            max_frame_ms = 1_865.9,
        ),
    ]
    assert _stream_measures(windows)["stream_max_frame_ms"].value == pytest.approx(100.2)


def test_an_action_running_during_generation_does_not_set_the_worst_streaming_frame():
    """MEASURED. `scroll_during_generation` runs mid-stream by design, so its window carries SSE
    traffic AND a scroll. On a fast-tier 100K null it put a 1,738 ms worst frame into this metric
    while the unaided stretch beside it peaked at 286 ms. A scroll is not a streaming stall."""
    windows = [
        _window("stream:gap3", max_frame_ms = 286.0),
        _window("action:scroll_during_generation", kind = "action", max_frame_ms = 1_738.0),
    ]
    m = _stream_measures(windows)
    assert m["stream_max_frame_ms"].value == pytest.approx(286.0)
    # And its SSE task chains are excluded from the targeted numerator too. Measured on a
    # standard-tier 10K null, an `action:keystroke` chain cost 23.77 ms per burst against 1.69 ms
    # in the gap windows either side: the chain runs until the event loop drains, so the typing
    # lands inside it and is billed to the stream.
    assert "1 unaided streaming window(s)" in m["stream_delta_cost_ms_per_kchar"].note


# ── refusing rather than reporting zero ──────────────────────────────────────────────────────


def test_no_clamp_means_null_with_a_reason_not_zero_cost():
    m = _stream_measures([_window("stream:gap1", blocked_ms = None)])
    assert m["stream_cost_ms_per_kchar"].value is None
    assert m["stream_cost_ms_per_kchar"].attempted is True
    assert "clamp" in (m["stream_cost_ms_per_kchar"].note or "")
    # The targeted numerator does not depend on the clamp, so it still reads.
    assert m["stream_delta_cost_ms_per_kchar"].value == pytest.approx(300.0)


def test_a_cell_with_no_streaming_reports_not_attempted_with_every_reason():
    m = _stream_measures(
        [
            _window("stream:gap12", streaming_observed = False, chars_open = 0, chars_close = 0),
            _window("action:send_turn", kind = "action", chars_open = 10, chars_close = 20),
        ]
    )
    for key in STREAM_METRICS:
        assert m[key].value is None
        assert m[key].attempted is False
    assert "no SSE traffic" in (m["stream_cost_ms_per_kchar"].note or "")


def test_an_empty_cell_names_that_rather_than_scoring_it():
    m = _stream_measures([])
    assert set(m) == set(STREAM_METRICS)
    assert all(v.value is None for v in m.values())


def test_a_frameless_streaming_window_poisons_the_pooled_streaming_frame_metrics():
    """A stream window the rAF loop never ran in must not be answered for by the one beside it.

    `instruments/frames.js` emits `frames_attempted: true` with `frame_gaps_ms: []` and a null
    `max_frame_ms` when it saw no callbacks, and rAF stops being scheduled when the renderer
    stalls -- while SSE keeps arriving, so the window still qualifies as streaming. Skipped, it
    contributed no deltas, no wall time and no worst frame, and the cell came back with the other
    window's clean numbers. `_frame_measures` refuses the same shape for the cell-wide metrics.
    """
    smooth = _window("stream:gap1", frame_gaps = [16.7] * 540, max_frame_ms = 16.7)
    frozen = _window("stream:gap2", duration_ms = 4_000.0, frame_gaps = [], max_frame_ms = None)

    m = _stream_measures([smooth, frozen])
    for key in ("stream_max_frame_ms", "stream_time_in_jank_pct", "stream_jank_index"):
        assert m[key].attempted is True
        assert m[key].value is None
        assert "no frames at all" in (m[key].note or "")

    # The cost side is untouched: `stream_cost` measured that window perfectly well.
    assert m["stream_cost_ms_per_kchar"].value is not None
    assert m["stream_delta_cost_ms_per_kchar"].value is not None
    assert m["stream_busy_pct"].value is not None

    # The giveaway: without this the three above were byte-identical to the cell with no freeze.
    alone = _stream_measures([smooth])
    assert alone["stream_max_frame_ms"].value == pytest.approx(16.7)
    assert alone["stream_time_in_jank_pct"].value == pytest.approx(0.0)


def test_a_window_the_frame_recorder_was_never_installed_in_is_still_only_skipped():
    """The control. Not-attempted is an absent instrument, not an unmeasured window."""
    smooth = _window("stream:gap1", frame_gaps = [16.7] * 540, max_frame_ms = 16.7)
    never = _window("stream:gap2", duration_ms = 4_000.0)
    never["instruments"]["frames"] = {"frames_attempted": False}
    m = _stream_measures([smooth, never])
    assert m["stream_max_frame_ms"].value == pytest.approx(16.7)
    assert m["stream_jank_index"].value is not None


def test_a_recorder_that_died_partway_poisons_every_streaming_metric():
    """A cell whose page crashed is truncated, and a truncated cell must not report a rate.

    `_stream_windows` records why it rejected each window, but `_stream_measures` reads that only
    when NOTHING qualified. Once one window has qualified, the rejection that says the page went
    away is discarded with the rest, and the cell publishes a rate computed over the prefix that
    ran before the crash. Both halves of the rate lose the same unmeasured stretch, so the error
    does not show up as a wide interval.

    The shape is not hypothetical: 138 unaided windows in the payload corpus carry
    `TargetClosedError` or `Target crashed` after a qualifying window, with a median duration of
    11.0 s.
    """
    good = _window("stream:gap1")
    dead = _window("stream:gap2", duration_ms = 14_000.0, attempted = False)
    dead["instruments"]["stream_cost"]["unavailable"] = (
        "TargetClosedError: Page.evaluate: Target page, context or browser has been closed"
    )

    m = _stream_measures([good, dead])
    assert set(m) == set(STREAM_METRICS)
    for key, measure in m.items():
        assert measure.attempted is True, key
        assert measure.value is None, key
        assert "stopped partway through this cell" in (measure.note or ""), key
    assert "Target page, context or browser has been closed" in (
        m["stream_cost_ms_per_kchar"].note or ""
    )

    # The giveaway: without this the crashed cell was byte-identical to the one that never
    # crashed, so nothing downstream could tell a truncated run from a complete one.
    assert _stream_measures([good])["stream_cost_ms_per_kchar"].value is not None


def test_a_window_the_instrument_merely_skipped_does_not_poison_the_cell():
    """The control, and the reason this keys on `unavailable` rather than on not-attempted.

    Windows with no `stream_cost` reading are ordinary -- the instrument is not installed in every
    window -- and they are already handled by being left out of `picked`. Only a window that was
    instrumented and reports the recorder going away means the cell is missing a stretch it cannot
    account for. Poisoning on not-attempted alone would void most of the corpus.
    """
    good = _window("stream:gap1")
    skipped = _window("stream:gap2", duration_ms = 14_000.0, attempted = False)
    assert "unavailable" not in skipped["instruments"]["stream_cost"]

    m = _stream_measures([good, skipped])
    assert m["stream_cost_ms_per_kchar"].value is not None
    assert m["stream_max_frame_ms"].value is not None


def test_every_declared_stream_metric_is_produced():
    m = _stream_measures([_window("stream:gap1")])
    assert set(m) == set(STREAM_METRICS)
    assert all(v.value is not None for v in m.values())


# ── the label that caused this ───────────────────────────────────────────────────────────────


def test_a_gap_window_is_not_labelled_stream():
    """REGRESSION. Every inter-slot gap used to be opened with `kind = "stream"`.

    Eighteen of them exist on the standard film and only the first four contain streaming, so the
    label sent anyone filtering on it to a pool of mostly post-stream idle. It is now `gap`, and
    `stream` is left to `stream:drain`, which is the only window the session layer opens that is
    genuinely about the stream.
    """
    from studiobench.runtime.types import WINDOW_KINDS
    from studiobench.scene.schedule import SceneRunner

    assert "gap" in WINDOW_KINDS

    opened: list[tuple[str, str]] = []

    class _Window:
        notes: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def note(self, *_args, **_kwargs):
            return None

    def _open(name, kind):
        opened.append((name, kind))
        return _Window()

    runner = SceneRunner(
        cell = None,
        page = None,
        cdp = None,
        dom = None,
        recorder = None,
        open_window = _open,
        log = lambda _m: None,
    )
    runner._census = lambda: {}
    # A gap far enough ahead that the scheduler actually opens a window for it.
    runner._gap_window("stream:gap12", until_ms = 300, t0 = __import__("time").monotonic())

    assert opened, "the scheduler opened no window for a gap that was wide enough to need one"
    name, kind = opened[0]
    assert kind == "gap"
    assert kind != "stream"
