# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The overhead gate, and the layout-cost adapter that feeds it.

Both exist to stop the same lie in two different places: an instrument reporting zero for
something it never measured, and an instrument whose own cost grows with the treatment and
therefore produces the slope everyone is hoping to find.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.arms.layoutcost import (  # noqa: E402
    COUNTER_FAMILIES,
    LAYOUTCOST_JS_PATH,
    LayoutCostInstrument,
    in_situ_overhead,
    load_layoutcost_js,
    reading_from_snapshot,
)
from studiobench.report.overhead import (  # noqa: E402
    log_growth_slope,
    overhead_growth_gate,
    render_overhead_section,
)
from studiobench.scoring import Measure  # noqa: E402


# the layout-cost adapter


# ---------------------------------------------------------------------------------------


def test_a_missing_snapshot_is_not_attempted_rather_than_zero():
    reading = reading_from_snapshot(None)
    assert set(reading.unavailable) == set(COUNTER_FAMILIES)
    for name in COUNTER_FAMILIES:
        assert reading.counters[name].attempted is False


def test_a_patch_the_engine_refused_reads_as_not_attempted():
    """A WebKit build that refuses the getter patch must not report zero forced layouts."""

    reading = reading_from_snapshot(
        {
            "unavailable": ["scrollHeightReads"],
            "counters": {"scrollHeightReads": 0, "moCallbacks": 0},
            "attempted": {"scrollHeightReads": False, "moCallbacks": True},
        }
    )
    refused = reading.counters["scrollHeightReads"]
    assert refused.attempted is False
    assert "could not be installed" in refused.display()
    # a family that DID install and genuinely saw nothing keeps its honest zero
    installed = reading.counters["moCallbacks"]
    assert installed.attempted is True
    assert installed.value == 0.0


def test_the_self_cost_estimate_is_labelled_a_lower_bound():
    reading = reading_from_snapshot(
        {"counters": {}, "attempted": {}, "overheadMsPerCall": 0.001045}
    )
    assert reading.self_cost_ms_per_call.has_reading
    assert "LOWER BOUND" in (reading.self_cost_ms_per_call.note or "")


def test_the_paired_cell_is_what_produces_the_real_cost():
    overhead = in_situ_overhead(Measure.read(48.0, "ms"), Measure.read(41.0, "ms"))
    assert overhead.value == pytest.approx(7.0)


def test_an_incomplete_pair_leaves_the_cost_unknown_not_zero():
    overhead = in_situ_overhead(
        Measure.read(48.0, "ms"), Measure.failed("ms", "the without cell crashed")
    )
    assert overhead.has_reading is False
    assert "unknown rather than zero" in overhead.display()


def test_the_instrument_declares_a_level_above_the_headline():
    assert LayoutCostInstrument.level >= 1
    assert LayoutCostInstrument.name == "layoutcost"


def test_the_browser_half_exists_and_exposes_what_the_adapter_calls():
    source = load_layoutcost_js()
    assert LAYOUTCOST_JS_PATH.is_file()
    for symbol in ("__sbLayoutCost", "snapshot", "reset", "selfCostEstimate", "uninstall"):
        assert symbol in source


@pytest.mark.skipif(shutil.which("node") is None, reason = "node is not installed")
def test_layoutcost_js_parses():
    result = subprocess.run(
        ["node", "--check", str(LAYOUTCOST_JS_PATH)],
        capture_output = True,
        text = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr


# the overhead gate


# ---------------------------------------------------------------------------------------


def test_an_instrument_whose_cost_tracks_the_treatment_is_disqualified():
    verdict = overhead_growth_gate(
        "tracing",
        2,
        {
            1_000: Measure.read(4.0, "ms"),
            10_000: Measure.read(9.0, "ms"),
            100_000: Measure.read(31.0, "ms"),
        },
    )
    assert verdict.disqualified is True
    assert "manufactures the very slope" in verdict.reason
    assert "DISQUALIFIED" in render_overhead_section([verdict])


def test_a_flat_instrument_passes():
    verdict = overhead_growth_gate(
        "tracing",
        2,
        {
            1_000: Measure.read(8.0, "ms"),
            100_000: Measure.read(8.4, "ms"),
        },
    )
    assert verdict.disqualified is False
    assert "within the" in verdict.reason


def test_a_large_ratio_on_a_tiny_number_is_not_a_disqualification():
    verdict = overhead_growth_gate(
        "layoutcost",
        3,
        {1_000: Measure.read(0.01, "ms"), 100_000: Measure.read(0.09, "ms")},
    )
    assert verdict.disqualified is False
    assert "below the" in verdict.reason


def test_a_gate_that_could_not_be_evaluated_is_not_a_passed_gate():
    verdict = overhead_growth_gate("tracing", 2, {1_000: Measure.read(4.0, "ms")})
    assert verdict.disqualified is False
    assert "not a passed gate" in verdict.reason


def test_level_zero_has_nothing_to_declare():
    verdict = overhead_growth_gate("frames", 0, {})
    assert "headline numbers" in verdict.reason


def test_the_log_slope_catches_a_spike_the_end_to_end_ratio_misses():
    by_rung = {
        1_000: Measure.read(5.0, "ms"),
        10_000: Measure.read(40.0, "ms"),
        100_000: Measure.read(5.0, "ms"),
    }
    verdict = overhead_growth_gate("tracing", 2, by_rung)
    assert verdict.growth_ratio == pytest.approx(1.0)  # the ends agree
    assert verdict.disqualified is False
    assert log_growth_slope(by_rung) is not None  # but the slope is reported anyway


def test_the_log_slope_needs_two_points():
    assert log_growth_slope({1_000: Measure.read(5.0, "ms")}) is None
