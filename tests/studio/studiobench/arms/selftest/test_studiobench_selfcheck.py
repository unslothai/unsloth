# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the integrity gates in `instruments/selfcheck.py`.

These live under `arms/selftest/` rather than beside the module because `instruments/` is shared
with the layer that owns the frame recorder, and this layer owns only `selfcheck.py` inside it.

Every gate is tested in BOTH directions. The passing direction is the easy half and the one that
gets written; the failing direction is the half that matters, because a gate that cannot fail is
decoration, and the whole point of this file is that a failure ABORTS the run rather than
appearing as a caveat under a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.instruments.selfcheck import (  # noqa: E402
    SelfCheckFailure,
    evaluate_clock_pair,
    evaluate_input_delay_gate,
    evaluate_longtask_support,
    evaluate_scene_contrast_gate,
    evaluate_stall_gate,
    evaluate_tri_clock,
    guard,
    run_gates,
)
from studiobench.scoring.schema import EXCLUSION_REASONS, check_exclusion_reasons  # noqa: E402


# ---------------------------------------------------------------------------------------
# the injected stall
# ---------------------------------------------------------------------------------------


def test_a_120ms_stall_seen_within_tolerance_passes():
    assert evaluate_stall_gate(121.0).passed is True
    assert evaluate_stall_gate(139.0).passed is True


def test_a_stall_the_recorder_missed_fails_the_run():
    gate = evaluate_stall_gate(12.0)
    assert gate.passed is False
    assert "mis-attributing" in gate.detail


def test_no_stall_reading_at_all_is_a_failure_not_a_pass():
    gate = evaluate_stall_gate(None)
    assert gate.passed is False
    assert gate.measured.attempted is True
    assert gate.measured.has_reading is False


# ---------------------------------------------------------------------------------------
# the injected input delay
# ---------------------------------------------------------------------------------------


def test_a_400ms_input_delay_must_move_p95_by_350ms():
    assert evaluate_input_delay_gate(20.0, 415.0).passed is True


def test_an_input_path_that_does_not_move_is_not_measuring_input():
    gate = evaluate_input_delay_gate(20.0, 24.0)
    assert gate.passed is False
    assert "will not move when the app gets slower" in gate.detail


# ---------------------------------------------------------------------------------------
# scene contrast: the blindness check
# ---------------------------------------------------------------------------------------


def test_a_heavy_and_a_trivial_scene_must_differ():
    assert evaluate_scene_contrast_gate(300.0, 100.0).passed is True


def test_an_instrument_that_cannot_tell_heavy_from_trivial_is_blind():
    gate = evaluate_scene_contrast_gate(104.0, 100.0)
    assert gate.passed is False
    assert "BLIND" in gate.detail


# ---------------------------------------------------------------------------------------
# longtask support
# ---------------------------------------------------------------------------------------


def test_longtask_support_is_read_from_supported_entry_types():
    supported = evaluate_longtask_support(["mark", "measure", "longtask"])
    assert "available on this engine" in supported.detail
    unsupported = evaluate_longtask_support(["mark", "measure"])
    assert "NOT ATTEMPTED rather than as zero" in unsupported.detail
    # not supporting longtask is a fact about the engine, not a reason to abort
    assert unsupported.fatal is False
    assert unsupported.passed is True


def test_an_unreadable_support_list_is_recorded_and_not_guessed():
    gate = evaluate_longtask_support(None)
    assert gate.passed is False
    assert gate.fatal is False


# ---------------------------------------------------------------------------------------
# the clock-pair control ratio
# ---------------------------------------------------------------------------------------


def test_a_flat_control_ratio_passes():
    assert evaluate_clock_pair(10_000.0, 10_020.0).passed is True


def test_a_moving_control_ratio_means_the_measurement_moved():
    gate = evaluate_clock_pair(10_000.0, 8_000.0)
    assert gate.passed is False
    assert "the MEASUREMENT moved, not the page" in gate.detail


# ---------------------------------------------------------------------------------------
# three-clock agreement
# ---------------------------------------------------------------------------------------


def test_three_agreeing_clocks_keep_the_window():
    verdict = evaluate_tri_clock(
        wall_ms = 10_000.0,
        raf_span_ms = 9_900.0,
        screencast_span_ms = 9_850.0,
        timer_span_ms = 10_010.0,
        raf_frames = 600,
    )
    assert verdict.agreed is True
    assert verdict.excluded_cell("r3.w1") is None


def test_an_unscheduled_raf_loop_reads_as_no_measurement_not_as_no_dropped_frames():
    """The trap this gate exists for: no frames looks exactly like a perfectly smooth window."""

    verdict = evaluate_tri_clock(
        wall_ms = 10_000.0,
        raf_span_ms = 0.0,
        screencast_span_ms = 9_900.0,
        timer_span_ms = 10_000.0,
        raf_frames = 0,
    )
    assert verdict.agreed is False
    assert "unmeasured one" in verdict.reason
    cell = verdict.excluded_cell("r4.w2")
    assert cell is not None
    assert cell.reason == "clock_disagreement"
    check_exclusion_reasons([cell])


def test_a_clock_more_than_twenty_percent_off_excludes_the_window():
    verdict = evaluate_tri_clock(
        wall_ms = 10_000.0,
        raf_span_ms = 6_000.0,
        screencast_span_ms = 9_900.0,
        timer_span_ms = 10_000.0,
        raf_frames = 120,
    )
    assert verdict.agreed is False
    assert verdict.worst_clock == "raf"
    assert verdict.worst_disagreement_pct == pytest.approx(40.0)


def test_agreement_between_fewer_than_two_clocks_is_not_agreement():
    verdict = evaluate_tri_clock(
        wall_ms = 10_000.0,
        raf_span_ms = 9_900.0,
        screencast_span_ms = None,
        timer_span_ms = None,
        raf_frames = 600,
    )
    assert verdict.agreed is False
    assert "fewer than two clocks" in verdict.reason


def test_clock_disagreement_is_a_declared_exclusion_reason():
    assert "clock_disagreement" in EXCLUSION_REASONS


# ---------------------------------------------------------------------------------------
# the whole gate set
# ---------------------------------------------------------------------------------------


def _healthy(**overrides):
    args = {
        "stall_observed_ms": 122.0,
        "keystroke_p95_baseline_ms": 22.0,
        "keystroke_p95_delayed_ms": 430.0,
        "heavy_scene_ms": 380.0,
        "trivial_scene_ms": 100.0,
        "supported_entry_types": ["longtask"],
        "clock_pair_page_ms": 10_000.0,
        "clock_pair_driver_ms": 10_010.0,
    }
    args.update(overrides)
    return args


def test_a_healthy_instrument_passes_every_gate():
    report = run_gates(**_healthy())
    assert report.ok is True
    assert "all gates held" in report.render()
    guard(report)  # does not raise


def test_one_failed_gate_aborts_the_run_before_any_numbers():
    report = run_gates(**_healthy(stall_observed_ms = 5.0))
    assert report.ok is False
    messages = []
    with pytest.raises(SelfCheckFailure) as caught:
        guard(report, on_abort = messages.append)
    assert messages and "ABORT" in messages[0]
    assert "no cells will be measured" in str(caught.value)


def test_the_abort_message_says_why_reporting_nothing_is_better():
    report = run_gates(**_healthy(heavy_scene_ms = 101.0))
    with pytest.raises(SelfCheckFailure) as caught:
        report.raise_if_failed()
    assert "the numbers get quoted and the blindness does not" in str(caught.value)
