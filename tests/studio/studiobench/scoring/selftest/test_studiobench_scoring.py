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
        Measure(value = 1.0, attempted = False, unit = "ms", note = "nope")


def test_not_attempted_must_say_why():
    with pytest.raises(PayloadSchemaError):
        Measure(value = None, attempted = False, unit = "ms")


def test_sub_floor_delta_prints_as_a_bound_not_a_zero():
    measure = Measure.read(0.04, "ms/update", floor = 0.12)
    assert measure.sub_floor is True
    assert measure.display() == "< 0.12 ms/update (instrument floor)"
    assert "0.04" not in measure.display()


def test_negative_sub_floor_is_bounded_from_below():
    measure = Measure.read(-0.03, "ms/update", floor = 0.12)
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


def test_validate_payload_accepts_the_first_parity_message_row():
    """`i` is the ordinal scene/parity.js writes for message 0, not a missing measurement.

    Without this exemption every payload carrying parity rows failed validation, so the
    documented `--report` step never ran at all.
    """

    payload = {
        "excluded_cells": [],
        "cells": [
            {
                "parity": {
                    "parity_attempted": True,
                    "digest": "abc123",
                    "chars": 4211,
                    "messages": [
                        {"i": 0, "role": "user", "digest": "d0", "chars": 120},
                        {"i": 1, "role": "assistant", "digest": "d1", "chars": 900},
                    ],
                }
            }
        ],
    }
    validate_payload(payload)


def test_validate_payload_still_rejects_a_bare_zero_beside_a_parity_ordinal():
    """The exemption is the ordinal alone; a zero-length signature stays a loud failure."""

    payload = {
        "excluded_cells": [],
        "cells": [
            {
                "parity": {
                    "parity_attempted": True,
                    "digest": "abc123",
                    "chars": 4211,
                    "messages": [{"i": 0, "role": "user", "digest": "d0", "chars": 0}],
                }
            }
        ],
    }
    with pytest.raises(PayloadSchemaError) as caught:
        validate_payload(payload)
    assert "chars" in str(caught.value)


def test_validate_payload_accepts_an_equivalence_field_absent_from_both_arms():
    """`reasoning_spans` reads 0 streamed and 0 seeded on a real quick-tier run.

    `drift` was already exempt; the two counts it is computed from were not, so every payload
    carrying an equivalence block failed validation.
    """

    payload = {
        "excluded_cells": [],
        "cells": [
            {
                "equivalence": {
                    "fields": {
                        "reasoning_spans": {
                            "streamed": 0,
                            "seeded": 0,
                            "gating": False,
                            "note": "reported, not gated",
                        },
                        "assistant_messages": {"streamed": 6, "seeded": 4, "drift": 0.0},
                    }
                }
            }
        ],
    }
    validate_payload(payload)


def test_validate_payload_accepts_a_zero_sample_in_an_attested_instrument_array():
    """A 0 ms inter-frame gap is two frames in one millisecond, not a missing reading.

    The block attests with `frames_attempted`, which already covers the scalar counters beside
    it; walking into the sample array dropped that attestation.
    """

    payload = {
        "excluded_cells": [],
        "windows": [
            {
                "instruments": {
                    "frames": {
                        "frames_attempted": True,
                        "frame_gaps_ms": [0, 3, 4, 16, 16],
                        "frames_over_33": 0,
                    }
                }
            }
        ],
    }
    validate_payload(payload)


def test_an_unattested_instrument_array_still_fails():
    """The exemption is the attestation, not the brackets. Without it the zero stays loud."""

    payload = {
        "excluded_cells": [],
        "windows": [{"instruments": {"frames": {"frame_gaps_ms": [0, 3, 4]}}}],
    }
    with pytest.raises(PayloadSchemaError) as caught:
        validate_payload(payload)
    assert "frame_gaps_ms" in str(caught.value)


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
    stats = compute_frame_stats(deltas, window_ms = 6000.0, declared_refresh_ms = 16.7)
    assert stats.time_in_jank_pct.value == pytest.approx(100.0)
    assert stats.max_frame_ms.value == pytest.approx(120.0)


def test_a_single_stall_is_caught_by_jank_index_and_missed_by_percentiles():
    """5,000 good frames and one 3.4 s freeze. p95 is perfect; the app was frozen."""

    deltas = [8.0] * 5000 + [3400.0]
    stats = compute_frame_stats(deltas, window_ms = 43_400.0, declared_refresh_ms = 16.7)
    assert stats.p95_frame_ms.value == pytest.approx(8.0, abs = 0.5)
    assert stats.time_in_jank_pct.value < 10.0
    assert stats.jank_index.value > 200.0
    assert stats.max_frame_ms.value == pytest.approx(3400.0)


def test_refresh_interval_is_measured_not_assumed():
    deltas = [8.3] * 40 + [40.0, 60.0]
    interval, source = measure_refresh_interval_ms(deltas)
    assert source == "measured"
    assert interval == pytest.approx(8.3, abs = 0.5)


def test_no_frames_recorded_is_a_failed_reading_not_zero_jank():
    stats = compute_frame_stats([], window_ms = 5000.0)
    assert stats.no_frames_recorded is True
    assert stats.time_in_jank_pct.has_reading is False
    assert stats.time_in_jank_pct.attempted is True
    assert "unscheduled" in stats.time_in_jank_pct.display()


def test_histogram_is_always_present_and_totals_the_frames():
    deltas = [4.0, 9.0, 20.0, 120.0, 5000.0]
    stats = compute_frame_stats(deltas, window_ms = 5153.0)
    assert sum(b["bucket_count"] for b in stats.histogram) == len(deltas)


# ---------------------------------------------------------------------------------------
# per-metric and per-rung scoring
# ---------------------------------------------------------------------------------------


def test_log_anchors_put_the_geometric_midpoint_at_fifty():
    anchor = METRIC_BY_KEY["keystroke_p95_ms"]
    midpoint = math.sqrt(anchor.good * anchor.bad)
    scored = score_metric(anchor, Measure.read(midpoint, "ms"))
    assert scored.score == pytest.approx(50.0, abs = 0.001)


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

    rung = score_rung(100_000, _good_metrics(scroll_settle_ms = Measure.read(9000.0, "ms")))
    assert rung.score == 0.0
    assert rung.zeroed_by == ["scroll_settle_ms"]
    assert rung.usable is False


def test_an_arithmetic_mean_would_have_rescued_it():
    """Documents the choice: the same inputs average to a passing score."""

    rung = score_rung(100_000, _good_metrics(scroll_settle_ms = Measure.read(9000.0, "ms")))
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
        keystroke_p95_ms = Measure.read(180.0, "ms"),
        time_in_jank_pct = Measure.read(25.0, "%"),
        jank_index = Measure.read(8.0, "ms"),
        max_frame_ms = Measure.read(900.0, "ms"),
        scroll_settle_ms = Measure.read(1800.0, "ms"),
        menu_open_ms = Measure.read(700.0, "ms"),
    )

    crasher = score_ladder(
        [
            score_rung(t, _good_metrics(), completed = True)
            if t <= 100_000
            else score_rung(t, {}, completed = False, failure_mode = "renderer crashed")
            for t in rungs
        ]
    )
    limper = score_ladder(
        [
            score_rung(t, _good_metrics() if t <= 100_000 else slow_metrics, completed = True)
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
        keystroke_p95_ms = Measure.read(400.0, "ms"),
        scroll_settle_ms = Measure.read(2800.0, "ms"),
    )
    ladder = score_ladder(
        [score_rung(t, _good_metrics() if t <= 10_000 else bad, completed = True) for t in rungs]
    )
    assert ladder.onset_rung_tokens == 10_000
    assert ladder.rungs[0].score >= ONSET_SCORE_THRESHOLD


def test_non_monotone_usability_is_flagged_rather_than_maximised():
    rungs = [1_000, 10_000, 100_000]
    bad = _good_metrics(keystroke_p95_ms = Measure.read(450.0, "ms"))
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


def _pairs(ratios: dict[str, float], rungs = (1_000, 10_000, 100_000)) -> list[Pair]:
    out = []
    for metric, ratio in ratios.items():
        for index, rung in enumerate(rungs):
            base = 100.0 + index
            out.append(
                Pair(
                    rung_tokens = rung,
                    metric_key = metric,
                    base = Measure.read(base, "ms"),
                    treatment = Measure.read(base * ratio, "ms"),
                )
            )
    return out


def test_ab_refuses_across_weights_ids():
    with pytest.raises(Exception) as caught:
        compare(
            "x",
            _pairs({"keystroke_p95_ms": 0.9}),
            _identity(),
            _identity(weights_id = "w-999"),
        )
    assert "weights_id" in str(caught.value)


def test_ab_refuses_across_sessions():
    with pytest.raises(Exception) as caught:
        compare(
            "x",
            _pairs({"keystroke_p95_ms": 0.9}),
            _identity(session = "s1"),
            _identity(session = "s2"),
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
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct = 5.0)
    assert result.headline_ratio < 0.9  # the headline says "16% faster"
    assert result.verdict == "FAIL"
    assert any("max_frame_ms" in r for r in result.regressions)


def test_within_noise_is_not_a_win():
    result = compare(
        "treatment",
        _pairs({"keystroke_p95_ms": 0.98}),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    assert result.verdict == "NO DIFFERENCE"
    assert result.metrics[0].verdict == "within noise"


def test_null_control_that_drifts_voids_the_whole_comparison():
    control = compare(
        "base vs base",
        _pairs({"keystroke_p95_ms": 1.30}),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
        is_null_control = True,
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
        noise_floor_pct = 5.0,
        is_null_control = True,
    )
    assert control.void is False
    floor, source = noise_floor_from_null_control(control)
    assert 1.0 <= floor <= 3.0
    assert "measured from the null-treatment control" in source


def test_bootstrap_ci_brackets_the_geometric_mean():
    pairs = _pairs({"keystroke_p95_ms": 0.70}, rungs = (1_000, 10_000, 100_000, 500_000))
    result = compare("treatment", pairs, _identity(), _identity(), bootstrap_seed = 7)
    metric = result.metrics[0]
    assert metric.ci_low is not None
    assert metric.ci_low <= metric.ratio_geomean <= metric.ci_high


# ---------------------------------------------------------------------------------------
# A CI that contains 1.0 is not a result: the pairs have to agree on the sign
# ---------------------------------------------------------------------------------------
#
# The noise floor is a fact about the HARNESS -- whether a difference of this size is resolvable
# here at all -- and it was the only gate a direction had to pass. Repetitions that disagree
# produce a geometric mean well clear of the floor anyway: 0.7, 0.7, 1.2, 1.2 averages to 0.917
# with a CI of 0.700-1.200, and the table said "improved" over it, in the one column anybody
# quotes. The CI was computed, printed, and never consulted.


def _split_pairs(metric: str, ratios: list[float]) -> list[Pair]:
    """One metric, one ratio per rung, so the pairs can be made to disagree on the sign."""
    out = []
    for index, ratio in enumerate(ratios):
        base = 100.0 + index
        out.append(
            Pair(
                rung_tokens = 1_000 * (index + 1),
                metric_key = metric,
                base = Measure.read(base, "ms"),
                treatment = Measure.read(base * ratio, "ms"),
            )
        )
    return out


def test_a_ci_that_spans_no_effect_is_not_an_improvement():
    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [0.7, 0.7, 1.2, 1.2]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert metric.beyond_noise is True  # 8.3% clear of a 5% floor
    assert metric.ci_low <= 1.0 <= metric.ci_high
    assert metric.ci_spans_no_effect is True
    assert metric.verdict == "inconclusive"
    # The headline has no interval of its own, so it is the line most likely to be quoted.
    assert result.verdict == "INCONCLUSIVE"
    assert result.regressions == []


def test_a_ci_that_spans_no_effect_does_not_clear_a_regression():
    """An unresolved regression is still a regression: the fail-safe direction is FAIL.

    The mirror of the case above must NOT be symmetric. Refusing to claim a win costs a
    contributor a headline; refusing to raise a fail lets the regression ship. So the metric is
    labelled unresolved and still counted.
    """
    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [1.4, 1.4, 0.9, 0.9]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert metric.ci_spans_no_effect is True
    assert metric.verdict == "regressed (unresolved)"
    assert result.verdict == "FAIL"
    assert any("unresolved" in r for r in result.regressions)


def test_agreeing_pairs_still_carry_their_direction():
    """The gate must not swallow a real effect: four ratios that agree keep their verdict."""
    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [0.70, 0.72, 0.68, 0.74]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert metric.ci_spans_no_effect is False
    assert metric.verdict == "improved"
    assert result.verdict == "IMPROVED"


def test_the_table_does_not_print_a_direction_it_could_not_resolve():
    from studiobench.report.render import render_ab_table

    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [0.7, 0.7, 1.2, 1.2]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    text = render_ab_table(result)
    assert "VERDICT: INCONCLUSIVE" in text
    assert "improved" not in text
    assert "no direction is claimed" in text
    assert "contains it" in text  # the interval straddles 1.0, as opposed to being absent


# ---------------------------------------------------------------------------------------
# A measured zero is a reading: sub-floor arms bound the ratio rather than voiding the pair
# ---------------------------------------------------------------------------------------
#
# `Pair.usable` required both values to be strictly positive. `time_in_jank_pct` and `jank_index`
# are 0.0 on any arm smooth enough to have no over-budget frames, which is the ordinary state of a
# healthy base, so a treatment that introduced jank over a zero-jank base had its pair dropped:
# the table said `no reading` about two arms that had both read, the regression was absent from
# `regressions` and from the headline, and as a null control it neither voided nor reached the
# noise floor derived from it.
#
# The rule applied here is the one `score.py` has always applied to the same reading: a sub-floor
# value is at least as good as the floor, so the floor is what enters the ratio. What that must
# NOT do is admit a reading that was never taken, which is the distinction `frames.py` protects
# when it refuses to score an unscheduled rAF loop as zero jank.

JANK_FLOOR = 0.1
SMOOTH = Measure.read(0.0, "%", floor = JANK_FLOOR)
JANKY = Measure.read(5.0, "%", floor = JANK_FLOOR)


def _jank_pair(base: Measure, treatment: Measure) -> Pair:
    return Pair(rung_tokens = 100_000, metric_key = "time_in_jank_pct", base = base, treatment = treatment)


def test_a_zero_jank_base_still_pairs_against_a_treatment_that_introduced_jank():
    pair = _jank_pair(SMOOTH, JANKY)

    assert pair.base.has_reading and pair.treatment.has_reading
    assert pair.usable is True
    # The floor stands in for the sub-floor arm, so the ratio is a LOWER bound: the true
    # magnitude is larger, never smaller.
    assert pair.ratio == 50.0
    assert pair.bounded is True
    assert pair.to_json()["bounded"] is True


def test_the_regression_over_a_zero_base_reaches_the_table_and_the_headline():
    result = compare("main -> fix", [_jank_pair(SMOOTH, JANKY)], _identity(), _identity())
    metric = result.metrics[0]

    assert metric.n_pairs == 1
    assert metric.verdict == "regressed"
    assert metric.bounded is True
    assert any("time_in_jank_pct" in r for r in result.regressions)


def test_an_improvement_to_zero_jank_is_bounded_the_other_way():
    pair = _jank_pair(JANKY, SMOOTH)

    assert pair.usable is True
    assert pair.ratio == JANK_FLOOR / 5.0
    assert pair.bounded is True


def test_two_arms_below_the_floor_are_not_a_difference():
    """score.py's reason, at the ratio layer: instrument noise on a fast machine must not invent
    a difference between two perfect builds."""

    pair = _jank_pair(SMOOTH, Measure.read(0.02, "%", floor = JANK_FLOOR))

    assert pair.usable is True
    assert pair.ratio == 1.0
    result = compare("main -> fix", [pair], _identity(), _identity())
    assert result.metrics[0].verdict == "within noise"


def test_a_reading_that_was_never_taken_is_still_not_a_zero():
    """THE LINE THIS MUST NOT CROSS. Both of these carry `value is None`, which is a different
    thing from a measured zero, and neither may enter a ratio."""

    never = Measure.not_attempted("%", "no window had the frame recorder installed")
    failed = Measure.failed("%", "the recorder ran but exported no per-frame deltas")

    assert _jank_pair(never, JANKY).usable is False
    assert _jank_pair(failed, JANKY).usable is False
    assert _jank_pair(JANKY, never).usable is False
    assert _jank_pair(JANKY, failed).usable is False
    assert (
        compare("x", [_jank_pair(never, JANKY)], _identity(), _identity()).metrics[0].verdict
        == "no_reading"
    )


def test_a_zero_with_no_declared_floor_stays_unusable():
    """Nothing bounds it, so there is no honest ratio to form."""

    pair = Pair(
        rung_tokens = 100_000,
        metric_key = "keystroke_p95_ms",
        base = Measure.read(0.0, "ms"),
        treatment = Measure.read(40.0, "ms"),
    )

    assert pair.base.has_reading is True
    assert pair.usable is False


def test_a_null_control_voids_on_jank_it_introduced_over_a_zero_base():
    """The coupling. The null control decides whether this machine can tell two identical builds
    apart; blind to the jank transition, it passed and published a floor derived from what was
    left."""

    result = compare(
        "null control: main vs itself",
        [_jank_pair(SMOOTH, JANKY)],
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
        is_null_control = True,
    )

    assert result.void is True
    assert "time_in_jank_pct" in (result.void_reason or "")


def test_a_bounded_ratio_is_not_published_as_this_machines_noise_floor():
    """A bound is not a spread. Left in, the 50x ratio above would publish a 4,900% noise floor
    and swallow every real effect measured on that machine afterwards."""

    result = compare(
        "null control: main vs itself",
        [
            _jank_pair(SMOOTH, JANKY),
            Pair(
                rung_tokens = 100_000,
                metric_key = "max_frame_ms",
                base = Measure.read(30.0, "ms"),
                treatment = Measure.read(31.0, "ms"),
            ),
        ],
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
        is_null_control = True,
    )
    floor, source = noise_floor_from_null_control(result)

    assert floor < 10.0
    assert "1 bounded metric(s) excluded" in source


def test_a_null_control_of_only_bounded_ratios_falls_back_to_the_declared_default():
    result = compare(
        "null control: main vs itself",
        [_jank_pair(SMOOTH, JANKY)],
        _identity(),
        _identity(),
        is_null_control = True,
    )
    floor, source = noise_floor_from_null_control(result)

    assert "declared default" in source
    assert "under an instrument floor" in source


def test_an_unresolved_metric_does_not_lend_its_magnitude_to_the_headline():
    """A metric that cannot resolve its own sign must not supply the number that gets quoted.

    The headline is a weighted geometric mean of point estimates with no interval of its own, so
    an inconclusive metric that happened to move a long way used to dominate it. keystroke_p95_ms
    at 0.2, 0.2, 1.5, 1.5 (geomean 0.548, CI 0.200-1.500) beside a resolved menu_open_ms of 0.900
    produced a headline of 0.631 and the word IMPROVED: a quoted 36.9% win almost entirely made
    of data the same table labels inconclusive.
    """

    pairs = _split_pairs("keystroke_p95_ms", [0.2, 0.2, 1.5, 1.5])
    pairs += _split_pairs("menu_open_ms", [0.9, 0.9, 0.9, 0.9])
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct = 5.0)

    by_key = {m.metric_key: m for m in result.metrics}
    assert by_key["keystroke_p95_ms"].verdict == "inconclusive"
    assert by_key["menu_open_ms"].verdict == "improved"
    # Only the resolved metric survives into the headline, so the quoted size is the real one.
    assert result.headline_ratio == pytest.approx(0.9, abs = 1e-9)
    assert result.verdict == "IMPROVED"


def test_a_run_whose_every_moving_metric_is_unresolved_is_inconclusive_not_no_reading():
    """Dropping unresolved metrics from the headline must not turn "says nothing" into "no data".

    NO READING means there was nothing to read. This run measured fine and simply failed to
    resolve a direction, which is a different answer and the one the operator has to act on.
    """

    pairs = _split_pairs("keystroke_p95_ms", [0.7, 0.7, 1.2, 1.2])
    pairs += _split_pairs("menu_open_ms", [0.6, 0.6, 1.3, 1.3])
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct = 5.0)

    assert all(m.ci_spans_no_effect for m in result.metrics)
    assert result.headline_ratio is None
    assert result.verdict == "INCONCLUSIVE"


def test_an_unresolved_mover_beside_a_flat_metric_is_not_no_difference():
    """ "No difference" asserts the change did nothing, which is stronger than this data supports.

    Dropping an unresolved mover from the headline can leave only flat metrics behind, putting the
    aggregate back inside the noise floor. Reading that as NO DIFFERENCE would convert a refusal
    to answer into a positive finding of no effect, when one metric did move and simply could not
    resolve its own sign.
    """

    pairs = _split_pairs("keystroke_p95_ms", [0.7, 0.7, 1.2, 1.2])
    pairs += _split_pairs("menu_open_ms", [1.0, 1.0, 1.0, 1.0])
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct = 5.0)

    by_key = {m.metric_key: m for m in result.metrics}
    assert by_key["keystroke_p95_ms"].unresolved is True
    assert by_key["menu_open_ms"].verdict == "within noise"
    assert result.headline_ratio == pytest.approx(1.0, abs = 1e-9)
    assert result.verdict == "INCONCLUSIVE"


def test_an_unresolved_metric_never_clears_a_resolved_regression():
    """A fail is never cleared by the exclusion: regressions are collected independently."""

    pairs = _split_pairs("keystroke_p95_ms", [0.7, 0.7, 1.2, 1.2])
    pairs += _split_pairs("menu_open_ms", [1.3, 1.3, 1.3, 1.3])
    result = compare("treatment", pairs, _identity(), _identity(), noise_floor_pct = 5.0)

    assert result.regressions
    assert result.verdict == "FAIL"


def test_a_metric_with_no_ci_at_all_does_not_claim_a_direction():
    """An interval that does not exist cannot clear 1.0, so it must not read as permission.

    `bootstrap_geomean_ci` returns (None, None) below three usable pairs, which a short ladder or
    a partially measured metric reaches easily. The rule added here claims a direction only when
    the CI clears no effect, and testing "does the interval contain 1.0" fails open when there is
    no interval: two pairs at 0.5 printed a 50% win with nothing behind it.
    """

    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [0.5, 0.5]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert metric.n_pairs == 2
    assert (metric.ci_low, metric.ci_high) == (None, None)
    assert metric.beyond_noise is True
    assert metric.ci_spans_no_effect is False  # nothing to span
    assert metric.unresolved is True
    assert metric.verdict == "inconclusive"
    assert result.headline_ratio is None
    assert result.verdict == "INCONCLUSIVE"


def test_a_regression_with_no_ci_is_still_a_regression_and_still_in_the_headline():
    """The refusal is one-sided: a missing interval withholds a win, never a loss.

    Withholding an unresolved win costs a headline; withholding an unresolved loss ships the
    regression. So the worse side keeps its plain "regressed" label, its FAIL, and its place in
    the aggregate, where it can only pull the number toward worse.
    """

    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [1.5, 1.5]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert (metric.ci_low, metric.ci_high) == (None, None)
    assert metric.verdict == "regressed"
    assert metric.withheld is False
    assert result.headline_ratio == pytest.approx(1.5, abs = 1e-9)
    assert result.regressions
    assert result.verdict == "FAIL"


def test_three_pairs_still_resolve_a_direction():
    """The control: the threshold is three, so three agreeing pairs still read as a win."""

    result = compare(
        "treatment",
        _split_pairs("keystroke_p95_ms", [0.5, 0.5, 0.5]),
        _identity(),
        _identity(),
        noise_floor_pct = 5.0,
    )
    metric = result.metrics[0]
    assert metric.ci_low is not None
    assert metric.unresolved is False
    assert metric.verdict == "improved"
    assert result.verdict == "IMPROVED"
