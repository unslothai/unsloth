# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the report layer: incremental payload, and the editorial policy.

Two things are being tested here and they fail in different ways. The payload writer fails by
losing evidence, which is invisible until the run that mattered is the one that crashed. The
renderer fails by printing something true next to something misleading, which is invisible
forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report import (  # noqa: E402
    HeadlinePolicyError,
    PayloadWriter,
    assemble,
    assert_headline_pair,
    excluded_totals,
    read_records,
    render_ab_table,
    render_excluded,
    render_frame_health,
    render_summary,
)
from studiobench.scoring import (  # noqa: E402
    ExcludedCell,
    Measure,
    PayloadSchemaError,
    Pair,
    RunIdentity,
    compare,
    compute_frame_stats,
    score_ladder,
    score_rung,
)


def _metrics(keystroke_ms: float = 25.0) -> dict[str, Measure]:
    return {
        "keystroke_p95_ms": Measure.read(keystroke_ms, "ms"),
        "time_in_jank_pct": Measure.read(0.8, "%"),
        "jank_index": Measure.read(0.2, "ms"),
        "max_frame_ms": Measure.read(40.0, "ms"),
        "scroll_settle_ms": Measure.read(130.0, "ms"),
        "menu_open_ms": Measure.read(60.0, "ms"),
    }


# ---------------------------------------------------------------------------------------
# incremental payload
# ---------------------------------------------------------------------------------------


def test_a_crash_at_rung_four_still_ships_rungs_one_to_three(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    writer.write("header", info = {"bench_version": "studiobench/1"})
    for rung in (1_000, 10_000, 100_000):
        writer.write("window", rung_tokens = rung, metrics = _metrics())
    writer.write("crash", where = "renderer", error_type = "TargetClosedError", error = "oom")
    writer.close()

    payload = assemble(path)
    assert len(payload["windows"]) == 3
    assert payload["complete"] is False
    assert payload["crashes"][0]["where"] == "renderer"
    assert "did not reach the end" in payload["incomplete_note"]


def test_a_half_written_final_line_is_discarded_not_fatal(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    writer.write("header", info = {"bench_version": "studiobench/1"})
    writer.write("window", rung_tokens = 1_000, metrics = _metrics())
    writer.close()
    with path.open("a", encoding = "utf-8") as handle:
        handle.write('{"kind":"window","rung_tokens":10000,"met')

    records, discarded = read_records(path)
    assert discarded == 1
    assert len(records) == 2
    payload = assemble(path)
    assert payload["truncated_records"] == 1
    assert len(payload["windows"]) == 1


def test_the_writer_records_a_crash_when_the_driver_dies(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    with pytest.raises(RuntimeError):
        with PayloadWriter(path) as writer:
            writer.write("header", info = {})
            raise RuntimeError("the driver fell over")
    payload = assemble(path)
    assert payload["crashes"][0]["error"] == "the driver fell over"
    assert payload["crashes"][0]["where"] == "driver"


def test_assemble_validates_the_bare_zero_ban(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    writer.write("window", rung_tokens = 1_000, react_stage_ms = 0)
    writer.close()
    with pytest.raises(PayloadSchemaError):
        assemble(path)


def test_measures_survive_the_round_trip_with_their_attempted_flag(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    writer.write(
        "window",
        rung_tokens = 1_000,
        metrics = {"react_stage_ms": Measure.not_attempted("ms", "profiling alias unverified")},
    )
    writer.close()
    payload = assemble(path)
    measure = payload["windows"][0]["metrics"]["react_stage_ms"]
    assert measure["attempted"] is False
    assert "not attempted" in measure["display"]


def test_excluded_cells_are_totalled_by_reason(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    for cell in (
        ExcludedCell("r3.w2", "clock_disagreement", 1, "raf 44% off wall"),
        ExcludedCell("r4.w1", "clock_disagreement", 2, "raf produced no frames"),
        ExcludedCell("r4.w7", "arm_voided_invariance", 1, "digest drifted"),
    ):
        writer.write("excluded", **cell.to_json())
    writer.write("footer", info = {"exit": "ok"})
    writer.close()
    payload = assemble(path)
    assert excluded_totals(payload) == {"clock_disagreement": 3, "arm_voided_invariance": 1}
    rendered = render_excluded(payload)
    assert "clock_disagreement" in rendered


def test_an_empty_excluded_block_is_still_printed_as_a_claim():
    rendered = render_excluded({"excluded_cells": []})
    assert "none." in rendered
    with pytest.raises(AssertionError):
        render_excluded({"excluded_cells": None})


# ---------------------------------------------------------------------------------------
# editorial policy
# ---------------------------------------------------------------------------------------


def test_a_single_frame_summary_may_not_be_a_headline():
    with pytest.raises(HeadlinePolicyError):
        assert_headline_pair(["time_in_jank_pct"])
    with pytest.raises(HeadlinePolicyError):
        assert_headline_pair(["jank_index", "max_frame_ms"])
    assert_headline_pair(["time_in_jank_pct", "jank_index", "max_frame_ms"])
    assert_headline_pair(["onset_rung"])  # nothing frame-related, nothing to police


def test_frame_health_always_prints_all_three():
    stats = compute_frame_stats([8.0] * 100 + [900.0], window_ms = 1700.0)
    rendered = render_frame_health(stats)
    assert "time in jank" in rendered
    assert "jank index" in rendered
    assert "worst frame" in rendered
    assert "histogram" in rendered


def test_no_frames_recorded_is_called_out_in_the_render():
    rendered = render_frame_health(compute_frame_stats([], window_ms = 3000.0))
    assert "not zero jank" in rendered


def test_the_headline_is_the_onset_rung_and_the_score_is_labelled_machine_local():
    ladder = score_ladder(
        [
            score_rung(1_000, _metrics()),
            score_rung(10_000, _metrics()),
            score_rung(100_000, _metrics(keystroke_ms = 460.0)),
        ]
    )
    summary = render_summary({"complete": True, "excluded_cells": []}, ladder)
    assert summary.index("ONSET RUNG") < summary.index("aggregate score")
    assert "machine-local" in summary
    assert "10,000 tokens" in summary


def test_a_void_ab_prints_no_numbers_at_all():
    identity = RunIdentity("studiobench/1", "c", "r", "w", "s1")
    pairs = [
        Pair(1_000, "keystroke_p95_ms", Measure.read(100.0, "ms"), Measure.read(140.0, "ms"))
        for _ in range(4)
    ]
    result = compare("base vs base", pairs, identity, identity, is_null_control = True)
    rendered = render_ab_table(result)
    assert "VOID" in rendered
    assert "1.4" not in rendered
    assert "ratio" not in rendered


def test_an_incomplete_run_says_so_before_any_numbers(tmp_path: Path):
    path = tmp_path / "payload.jsonl"
    writer = PayloadWriter(path)
    writer.write("window", rung_tokens = 1_000, metrics = _metrics())
    writer.close()
    payload = assemble(path)
    summary = render_summary(payload, None)
    assert summary.index("RUN DID NOT FINISH") < summary.index("EXCLUDED CELLS")


def test_harness_bias_is_printed_and_never_subtracted():
    summary = render_summary(
        {"complete": True, "excluded_cells": []},
        None,
        harness_bias = {"keystroke_p95_ms": Measure.read(4.2, "%")},
    )
    assert "HARNESS BIAS" in summary
    assert "NOT subtracted" in summary


# ── a cell that failed an invalidating gate is INCOMPLETE, not absent ──────────


def _gated_cell_rows(
    gate_name,
    passed,
    *,
    session = "s1",
):
    """One rung, one cell that completed its film, and that cell's own verdict on itself."""
    return [
        {
            "row_type": "cell",
            "cell_id": "r100K.base.rep0",
            "session_id": session,
            "target_tokens": 100_000,
            "completed": True,
            "cell": {"arm": "base", "rep": 0},
        },
        {
            "row_type": "gate",
            "name": gate_name,
            "passed": passed,
            "cell_id": "r100K.base.rep0",
            "session_id": session,
            "detail": {"reason": "12 of 18 ordinals never mounted"},
        },
    ]


def test_a_cell_that_lost_its_thread_is_scored_incomplete_rather_than_green():
    """REGRESSION. The ladder is ABSOLUTE: there is no second arm to contradict a cheap cell.

    `thread_complete` is advisory where it is emitted, so the cell reaches the report
    `completed=True` with a full set of timings that are cheaper for exactly the wrong reason, and
    the rung was scored against fixed anchors and came out green and fast.
    """
    from studiobench.report.build import _completion_by_rung

    got = _completion_by_rung(_gated_cell_rows("thread_complete", False))
    complete, reason = got[100_000]
    assert complete is False, got
    assert "thread_complete" in (reason or ""), got


def test_a_cell_that_stopped_following_the_stream_is_incomplete_too():
    from studiobench.report.build import _completion_by_rung
    got = _completion_by_rung(_gated_cell_rows("follows_the_stream", False))
    assert got[100_000][0] is False, got


def test_a_gate_that_only_qualifies_one_column_leaves_the_rung_alone():
    """`timer_clamp` failing nulls `busy_pct` and leaves every other column standing. Reading it as
    fatal here would zero a rung over a missing idle floor, most often on a loaded machine."""
    from studiobench.report.build import _completion_by_rung

    got = _completion_by_rung(_gated_cell_rows("timer_clamp", False))
    assert got[100_000] == (True, None), got


def test_a_passing_gate_leaves_the_rung_complete():
    from studiobench.report.build import _completion_by_rung
    got = _completion_by_rung(_gated_cell_rows("thread_complete", True))
    assert got[100_000] == (True, None), got


def test_a_retry_that_passed_is_not_marked_incomplete_by_the_dead_attempt():
    """`latest_attempt_rows` runs before this and drops the superseded cell, action and window
    rows -- but a gate is none of those, and `--resume` reuses the cell id."""
    from studiobench.report.build import _completion_by_rung
    from studiobench.scoring.from_payload import latest_attempt_rows

    rows = [
        {
            "row_type": "cell",
            "cell_id": "r100K.base.rep0",
            "session_id": "s-old",
            "target_tokens": 100_000,
            "completed": False,
            "cell": {"arm": "base", "rep": 0},
        },
        {
            "row_type": "gate",
            "name": "thread_complete",
            "passed": False,
            "cell_id": "r100K.base.rep0",
            "session_id": "s-old",
            "detail": {"reason": "lost the middle"},
        },
    ] + _gated_cell_rows("thread_complete", True, session = "s-new")

    got = _completion_by_rung(latest_attempt_rows(rows))
    assert got[100_000] == (True, None), got


def test_the_floor_table_drops_a_gate_failed_cell():
    """The other scorer that admits a cell on `completed` alone. It builds the floor a result is
    then judged against, and scores the result too."""
    from studiobench.sweep.floor_table import cell_metrics

    rows = _gated_cell_rows("thread_complete", False) + [
        {
            "row_type": "action",
            "cell_id": "r100K.base.rep0",
            "session_id": "s1",
            "action": "keystroke",
            "ran": True,
            "expect_ok": True,
            "timings": {"p95_ms": 50.0},
        }
    ]
    assert cell_metrics(rows) == {}
    ok = [r for r in rows if r.get("row_type") != "gate"]
    assert "r100K.base.rep0" in cell_metrics(ok)


def test_an_absent_sampler_is_not_read_as_a_cell_that_lost_the_stream():
    """REGRESSION. `_read_follow` returns `{"follow_attempted": False}` when the page-side sampler
    is not installed, and the gate then reports `passed: False` because `pinned` is None.

    That is an absent instrument, not a film that went wrong. Read as fatal it marked EVERY cell of
    every run unusable anywhere the sampler is missing, which is a far larger blast radius than the
    defect being closed, and it zeroed the ladder for a reason that says nothing about the build.
    """
    from studiobench.report.build import _completion_by_rung

    rows = _gated_cell_rows("follows_the_stream", False)
    rows[1]["detail"] = {
        "follow_attempted": False,
        "reason": "the follow sampler is not installed",
    }
    assert _completion_by_rung(rows)[100_000] == (True, None), rows


def test_a_measured_stream_follow_failure_is_still_fatal():
    """The positive control for the test above: the sampler ran and the thread fell behind."""
    from studiobench.report.build import _completion_by_rung

    rows = _gated_cell_rows("follows_the_stream", False)
    rows[1]["detail"] = {
        "follow_attempted": True,
        "pinned_fraction": 0.42,
        "reason": "the thread fell behind for 58% of the streaming phase",
    }
    assert _completion_by_rung(rows)[100_000][0] is False, rows


def test_a_bounded_ratio_is_marked_rather_than_printed_as_a_measurement():
    """An arm under its instrument floor contributes the floor, so the ratio understates the true
    magnitude. Printed bare it invites `4900% worse` being quoted as a measurement; the marker
    goes in the ratio cell rather than a footnote for the same reason the void path gives."""

    identity = RunIdentity("studiobench/1", "c", "r", "w", "s1")
    smooth = Measure.read(0.0, "%", floor = 0.1)
    janky = Measure.read(5.0, "%", floor = 0.1)
    result = compare(
        "main -> fix",
        [Pair(100_000, "time_in_jank_pct", smooth, janky)],
        identity,
        identity,
    )
    rendered = render_ab_table(result)

    assert ">=50.000" in rendered
    assert "no reading" not in rendered
    assert "regressed" in rendered


def test_an_ordinary_ratio_is_still_printed_bare():
    """The control: nothing sub-floor, nothing marked."""

    identity = RunIdentity("studiobench/1", "c", "r", "w", "s1")
    result = compare(
        "main -> fix",
        [Pair(100_000, "keystroke_p95_ms", Measure.read(100.0, "ms"), Measure.read(120.0, "ms"))],
        identity,
        identity,
    )
    rendered = render_ab_table(result)

    assert "1.200" in rendered
    assert ">=" not in rendered and "<=" not in rendered


def test_a_missing_viewport_is_not_waived_as_an_absent_instrument():
    """REGRESSION. `probe_attempted: False` has two producers and only one is an absent instrument.

    `window.__sb.dom is not installed` is the harness not being loaded. `no thread viewport` is the
    ARM missing the surface the film measures, and waiving it let a real defect ride the instrument
    allowance -- a cell with no scroller was admitted, scrolled nothing, and was scored.
    """
    from studiobench.report.build import _completion_by_rung

    rows = _gated_cell_rows("thread_complete", False)
    rows[1]["detail"] = {"probe_attempted": False, "reason": "no thread viewport"}
    assert _completion_by_rung(rows)[100_000][0] is False, rows


def test_an_uninstalled_dom_helper_is_still_waived():
    """The positive control: the harness itself not being loaded is not a finding about the build."""
    from studiobench.report.build import _completion_by_rung

    rows = _gated_cell_rows("thread_complete", False)
    rows[1]["detail"] = {
        "probe_attempted": False,
        "reason": "window.__sb.dom is not installed",
    }
    assert _completion_by_rung(rows)[100_000] == (True, None), rows
