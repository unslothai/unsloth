# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the scoring layer, run against synthetic payloads.

The scoring layer is pure, so it can be tested exhaustively without a browser, and it is the part
where a mistake is least likely to be noticed: a wrong ms is obvious, a wrong aggregation rule
produces plausible numbers forever.

The tests that matter most are the ADVERSARIAL ones, each of which encodes a specific way a
benchmark lies:

    a crashed rung must not outscore a slow one
    a regression must surface despite a positive headline
    NULL and SPIKE must behave, and must be caught when they misbehave
    a zero must never be printable without saying whether the thing was attempted
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.scoring import (  # noqa: E402
    METRIC_BY_KEY,
    Measure,
    Pair,
    PayloadSchemaError,
    RunIdentity,
    compare,
    compute_frame_stats,
    log_rung_weights,
    measure_refresh_interval_ms,
    noise_floor_from_null_control,
    score_ladder,
    score_metric,
    score_rung,
    validate_payload,
)
from studiobench.scoring.anchors import ONSET_SCORE_THRESHOLD  # noqa: E402


# ---------------------------------------------------------------------------------------
# Measure: the ban on bare zeros
# ---------------------------------------------------------------------------------------


def test_not_attempted_cannot_carry_a_value():
    with pytest.raises(PayloadSchemaError):
        Measure(value=1.0, attempted=False, unit="ms", note="nope")


def test_not_attempted_must_say_why():
    with pytest.raises(PayloadSchemaError):
        Measure(value=None, attempted=False, unit="ms")


def test_sub_floor_delta_prints_as_a_bound_not_a_zero():
    measure = Measure.read(0.04, "ms/update", floor=0.12)
    assert measure.sub_floor is True
    assert measure.display() == "< 0.12 ms/update (instrument floor)"
    assert "0.04" not in measure.display()


def test_negative_sub_floor_is_bounded_from_below():
    measure = Measure.read(-0.03, "ms/update", floor=0.12)
    assert measure.display() == "> -0.12 ms/update (instrument floor)"


def test_not_attempted_reads_differently_from_a_failed_reading():
    skipped = Measure.not_attempted("ms", "profiling alias not verified")
    failed = Measure.failed("ms", "the recorder produced no frames")
    assert "not attempted" in skipped.display()
    assert "no reading" in failed.display()
    assert skipped.to_json()["attempted"] is False
    assert failed.to_json()["attempted"] is True


def test_validate_payload_rejects_a_bare_zero():
    payload = {"excluded_cells": [], "windows": [{"react_stage_ms": 0}]}
    with pytest.raises(PayloadSchemaError) as caught:
        validate_payload(payload)
    assert "react_stage_ms" in str(caught.value)


def test_validate_payload_accepts_a_zero_inside_a_measure():
    payload = {
        "excluded_cells": [],
        "windows": [{"react_stage_ms": Measure.read(0.0, "ms").to_json()}],
    }
    validate_payload(payload)


def test_validate_payload_requires_excluded_cells():
    with pytest.raises(PayloadSchemaError):
        validate_payload({"windows": []})
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": None})


# ---------------------------------------------------------------------------------------
# frames: both directions of jank
# ---------------------------------------------------------------------------------------


def test_uniform_mediocrity_is_caught_by_time_in_jank_and_missed_by_max():
    """Every frame 120 ms. `max` says 120, which sounds survivable. It is not."""

    deltas = [120.0] * 50
    stats = compute_frame_stats(deltas, window_ms=6000.0, declared_refresh_ms=16.7)
    assert stats.time_in_jank_pct.value == pytest.approx(100.0)
    assert stats.max_frame_ms.value == pytest.approx(120.0)


def test_a_single_stall_is_caught_by_jank_index_and_missed_by_percentiles():
    """5,000 good frames and one 3.4 s freeze. p95 is perfect; the app was frozen."""

    deltas = [8.0] * 5000 + [3400.0]
    stats = compute_frame_stats(deltas, window_ms=43_400.0, declared_refresh_ms=16.7)
    assert stats.p95_frame_ms.value == pytest.approx(8.0, abs=0.5)
    assert stats.time_in_jank_pct.value < 10.0
    assert stats.jank_index.value > 200.0
    assert stats.max_frame_ms.value == pytest.approx(3400.0)


def test_refresh_interval_is_measured_not_assumed():
    deltas = [8.3] * 40 + [40.0, 60.0]
    interval, source = measure_refresh_interval_ms(deltas)
    assert source == "measured"
    assert interval == pytest.approx(8.3, abs=0.5)


def test_no_frames_recorded_is_a_failed_reading_not_zero_jank():
    stats = compute_frame_stats([], window_ms=5000.0)
    assert stats.no_frames_recorded is True
    assert stats.time_in_jank_pct.has_reading is False
    assert stats.time_in_jank_pct.attempted is True
    assert "unscheduled" in stats.time_in_jank_pct.display()


def test_histogram_is_always_present_and_totals_the_frames():
    deltas = [4.0, 9.0, 20.0, 120.0, 5000.0]
    stats = compute_frame_stats(deltas, window_ms=5153.0)
    assert sum(b["bucket_count"] for b in stats.histogram) == len(deltas)


# ---------------------------------------------------------------------------------------
# per-metric and per-rung scoring
# ---------------------------------------------------------------------------------------


def test_log_anchors_put_the_geometric_midpoint_at_fifty():
    anchor = METRIC_BY_KEY["keystroke_p95_ms"]
    midpoint = math.sqrt(anchor.good * anchor.bad)
    scored = score_metric(anchor, Measure.read(midpoint, "ms"))
    assert scored.score == pytest.approx(50.0, abs=0.001)


def test_a_reading_at_or_past_the_bad_anchor_scores_zero():
    anchor = METRIC_BY_KEY["keystroke_p95_ms"]
    assert score_metric(anchor, Measure.read(anchor.bad, "ms")).score == 0.0
    assert score_metric(anchor, Measure.read(anchor.bad * 10, "ms")).score == 0.0


def test_an_unmeasured_metric_is_not_scored_at_all():
    anchor = METRIC_BY_KEY["menu_open_ms"]
    scored = score_metric(anchor, Measure.not_attempted("ms", "action not in this scene"))
    assert scored.scored is False
    assert scored.score is None


def _good_metrics(**overrides) -> dict[str, Measure]:
    base = {
        "keystroke_p95_ms": Measure.read(25.0, "ms"),
        "time_in_jank_pct": Measure.read(0.8, "%"),
        "jank_index": Measure.read(0.2, "ms"),
        "max_frame_ms": Measure.read(40.0, "ms"),
        "scroll_settle_ms": Measure.read(130.0, "ms"),
        "menu_open_ms": Measure.read(60.0, "ms"),
    }
    base.update(overrides)
    return base


def test_one_catastrophic_metric_zeroes_the_rung():
    """Four good metrics must not rescue one broken one. That is why the mean is geometric."""

    rung = score_rung(100_000, _good_metrics(scroll_settle_ms=Measure.read(9000.0, "ms")))
    assert rung.score == 0.0
    assert rung.zeroed_by == ["scroll_settle_ms"]
    assert rung.usable is False


def test_an_arithmetic_mean_would_have_rescued_it():
    """Documents the choice: the same inputs average to a passing score."""

    rung = score_rung(100_000, _good_metrics(scroll_settle_ms=Measure.read(9000.0, "ms")))
    scored = [m for m in rung.metric_scores if m.scored]
    arithmetic = sum(m.weight * float(m.score) for m in scored) / sum(m.weight for m in scored)
    assert arithmetic > 60.0
    assert rung.score == 0.0


def test_missing_most_metrics_makes_the_rung_incomplete_rather_than_easy():
    sparse = {
        "keystroke_p95_ms": Measure.read(21.0, "ms"),
        "time_in_jank_pct": Measure.not_attempted("%", "recorder not installed"),
        "jank_index": Measure.not_attempted("ms", "recorder not installed"),
        "max_frame_ms": Measure.not_attempted("ms", "recorder not installed"),
        "scroll_settle_ms": Measure.not_attempted("ms", "action skipped"),
        "menu_open_ms": Measure.not_attempted("ms", "action skipped"),
    }
    rung = score_rung(100_000, sparse)
    assert rung.complete is False
    assert rung.score == 0.0
    assert "weight" in rung.incomplete_reason


# ---------------------------------------------------------------------------------------
# aggregation, and the three ways naive AUC lies
# ---------------------------------------------------------------------------------------


def test_log_rung_weights_do_not_let_the_top_rung_be_the_whole_score():
    weights = log_rung_weights([1_000, 10_000, 100_000, 500_000, 1_000_000])
    assert sum(weights) == pytest.approx(1.0)
    assert max(weights) < 0.40
    linear_span = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    linear_weights = [t / sum(linear_span) for t in linear_span]
    assert max(linear_weights) > 0.60  # what a linear axis would have done


def test_a_crashed_rung_must_not_outscore_a_slow_one():
    """The adversarial case: crashing at 500K must lose to limping through it."""

    rungs = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    slow_metrics = _good_metrics(
        keystroke_p95_ms=Measure.read(180.0, "ms"),
        time_in_jank_pct=Measure.read(25.0, "%"),
        jank_index=Measure.read(8.0, "ms"),
        max_frame_ms=Measure.read(900.0, "ms"),
        scroll_settle_ms=Measure.read(1800.0, "ms"),
        menu_open_ms=Measure.read(700.0, "ms"),
    )

    crasher = score_ladder(
        [
            score_rung(t, _good_metrics(), completed=True)
            if t <= 100_000
            else score_rung(t, {}, completed=False, failure_mode="renderer crashed")
            for t in rungs
        ]
    )
    limper = score_ladder(
        [
            score_rung(t, _good_metrics() if t <= 100_000 else slow_metrics, completed=True)
            for t in rungs
        ]
    )

    assert limper.aggregate > crasher.aggregate
    assert crasher.rungs[-1].score == 0.0
    assert crasher.rungs[-1].complete is False
    # and the crashed rungs are still IN the ladder, keeping their weight
    assert len(crasher.rungs) == len(rungs)


def test_onset_rung_is_the_largest_usable_rung():
    rungs = [1_000, 10_000, 100_000, 500_000]
    bad = _good_metrics(
        keystroke_p95_ms=Measure.read(400.0, "ms"),
        scroll_settle_ms=Measure.read(2800.0, "ms"),
    )
    ladder = score_ladder(
        [
            score_rung(t, _good_metrics() if t <= 10_000 else bad, completed=True)
            for t in rungs
        ]
    )
    assert ladder.onset_rung_tokens == 10_000
    assert ladder.rungs[0].score >= ONSET_SCORE_THRESHOLD


def test_non_monotone_usability_is_flagged_rather_than_maximised():
    rungs = [1_000, 10_000, 100_000]
    bad = _good_metrics(keystroke_p95_ms=Measure.read(450.0, "ms"))
    ladder = score_ladder(
        [
            score_rung(1_000, _good_metrics()),
            score_rung(10_000, bad),
            score_rung(100_000, _good_metrics()),
        ]
    )
    assert ladder.non_monotonic is True
    assert ladder.onset_rung_tokens == 100_000  # reported, but flagged
    assert len(rungs) == 3


# ---------------------------------------------------------------------------------------
# A/B
# ---------------------------------------------------------------------------------------


def _identity(session: str = "s1", **overrides) -> RunIdentity:
    fields = {
        "bench_version": "studiobench/1",
        "corpus_hash": "corpus-abc",
        "rung_ladder_id": "r-123",
        "weights_id": "w-123",
        "session_id": session,
    }
    fields.update(overrides)
    return RunIdentity(**fields)


def _pairs(ratios: dict[str, float], rungs=(1_000, 10_000, 100_000)) -> list[Pair]:
    out = []
    for metric, ratio in ratios.items():
        for index, rung in enumerate(rungs):
            base = 100.0 + index
            out.append(
                Pair(
                    rung_tokens=rung,
                    metric_key=metric,
                    base=Measure.read(base, "ms"),
                    treatment=Measure.read(base * ratio, "ms"),
                )
            )
    return out


def test_ab_refuses_across_weights_ids():
    with pytest.raises(Exception) as caught:
        compare(
            "x",
            _pairs({"keystroke_p95_ms": 0.9}),
            _identity(),
            _identity(weights_id="w-999"),
        )
    assert "weights_id" in str(caught.value)


def test_ab_refuses_across_sessions():
    with pytest.raises(Exception) as caught:
        compare(
            "x",
            _pairs({"keystroke_p95_ms": 0.9}),
            _identity(session="s1"),
            _identity(session="s2"),
        )
    assert "session" in str(caught.value)


def test_a_regression_surfaces_despite_a_positive_headline():
    """The adversarial case: a headline win of 16% hiding a 60% worse worst frame.

    This is the shape that ships. Nobody merges a change whose headline is negative; the change
    that gets merged is the one that improves five metrics and quietly ruins the sixth, and a
    single headline number is exactly the instrument that would let it through.
    """

    pairs = _pairs(
        {
            "keystroke_p95_ms": 0.75,
            "time_in_jank_pct": 0.75,
            "jank_index": 0.75,
            "scroll_settle_ms": 0.75,
            "menu_open_ms": 0.75,
            "max_frame_ms": 1.60,
        }
    )
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct=5.0)
    assert result.headline_ratio < 0.9  # the headline says "16% faster"
    assert result.verdict == "FAIL"
    assert any("max_frame_ms" in r for r in result.regressions)


def test_within_noise_is_not_a_win():
    result = compare(
        "treatment",
        _pairs({"keystroke_p95_ms": 0.98}),
        _identity(),
        _identity(),
        noise_floor_pct=5.0,
    )
    assert result.verdict == "NO DIFFERENCE"
    assert result.metrics[0].verdict == "within noise"


def test_null_control_that_drifts_voids_the_whole_comparison():
    control = compare(
        "base vs base",
        _pairs({"keystroke_p95_ms": 1.30}),
        _identity(),
        _identity(),
        noise_floor_pct=5.0,
        is_null_control=True,
    )
    assert control.void is True
    assert control.verdict == "VOID"
    assert "null-treatment control" in control.void_reason


def test_null_control_that_behaves_sets_the_noise_floor():
    control = compare(
        "base vs base",
        _pairs({"keystroke_p95_ms": 1.02, "jank_index": 0.985}),
        _identity(),
        _identity(),
        noise_floor_pct=5.0,
        is_null_control=True,
    )
    assert control.void is False
    floor, source = noise_floor_from_null_control(control)
    assert 1.0 <= floor <= 3.0
    assert "measured from the null-treatment control" in source


def test_bootstrap_ci_brackets_the_geometric_mean():
    pairs = _pairs({"keystroke_p95_ms": 0.70}, rungs=(1_000, 10_000, 100_000, 500_000))
    result = compare("treatment", pairs, _identity(), _identity(), bootstrap_seed=7)
    metric = result.metrics[0]
    assert metric.ci_low is not None
    assert metric.ci_low <= metric.ratio_geomean <= metric.ci_high
