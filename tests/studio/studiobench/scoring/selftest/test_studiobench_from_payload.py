# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the session-layer -> scoring-layer seam.

Every test here is written against a failure that would otherwise render as a plausible number.
The seam's whole job is to keep three different absences apart, so each is asserted separately
rather than through "is not None".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.scoring.from_payload import measures_from_records   # noqa: E402
from studiobench.scoring.schema import (                             # noqa: E402
    PayloadSchemaError,
    validate_payload,
)


def _cell(cell_id="c1", tokens=10_000, completed=True):
    return {"row_type": "cell", "cell_id": cell_id, "target_tokens": tokens,
            "completed": completed}


def _action(cell_id, name, ran=True, timings=None, reason=None):
    return {"row_type": "action", "cell_id": cell_id, "action": name, "ran": ran,
            "timings": timings or {}, "reason": reason}


def _window(cell_id, gaps, duration_ms=1000.0, kind="action", attempted=True):
    frames = {"frames_attempted": attempted}
    if attempted:
        frames.update({"frame_gaps_ms": gaps, "frame_gaps_truncated": False,
                       "frame_gaps_total": len(gaps),
                       "max_frame_ms": max(gaps) if gaps else None})
    return {"row_type": "window", "cell_id": cell_id, "kind": kind,
            "duration_ms": duration_ms, "instruments": {"frames": frames}}


# ── the three absences, kept apart ──────────────────────────────────────────────────────────

def test_action_absent_from_scene_is_not_attempted():
    m = measures_from_records([_cell()])["10000"if False else 10_000]
    keystroke = m["keystroke_p95_ms"]
    assert keystroke.attempted is False
    assert keystroke.value is None
    assert "not in this scene" in keystroke.note


def test_action_that_did_not_run_is_attempted_and_failed_with_its_reason():
    """The distinction that matters at 1K: scroll cannot run, and that is not a fast scroll."""
    recs = [_cell(), _action("c1", "scroll_after", ran=False,
                             reason="the thread is shorter than the viewport")]
    scroll = measures_from_records(recs)[10_000]["scroll_settle_ms"]
    assert scroll.attempted is True
    assert scroll.value is None
    assert "shorter than the viewport" in scroll.note


def test_action_that_ran_without_the_timing_key_is_failed_not_zero():
    recs = [_cell(), _action("c1", "keystroke", ran=True, timings={"p50_ms": 10.0})]
    keystroke = measures_from_records(recs)[10_000]["keystroke_p95_ms"]
    assert keystroke.attempted is True
    assert keystroke.value is None
    assert "recorded no p95_ms" in keystroke.note


def test_a_real_reading_carries_the_key_it_came_from():
    recs = [_cell(), _action("c1", "keystroke", timings={"p95_ms": 31.5})]
    keystroke = measures_from_records(recs)[10_000]["keystroke_p95_ms"]
    assert keystroke.value == pytest.approx(31.5)
    assert keystroke.note == "keystroke.p95_ms"


def test_scroll_settle_says_it_is_gesture_time_not_settle_time():
    """The one anchor whose recorded quantity is not what its name says. Never silent."""
    recs = [_cell(), _action("c1", "scroll_after", timings={"gesture_ms": 466.5})]
    scroll = measures_from_records(recs)[10_000]["scroll_settle_ms"]
    assert scroll.value == pytest.approx(466.5)
    assert "not post-gesture settle" in scroll.note


# ── frame metrics ───────────────────────────────────────────────────────────────────────────

def test_frame_metrics_pool_the_active_windows():
    recs = [_cell(), _window("c1", [16.0] * 50 + [200.0], duration_ms=1000.0)]
    m = measures_from_records(recs)[10_000]
    assert m["max_frame_ms"].value == pytest.approx(200.0)
    assert m["jank_index"].value is not None and m["jank_index"].value > 0
    assert m["time_in_jank_pct"].value is not None


def test_idle_windows_are_excluded_so_jank_is_not_diluted():
    busy = _window("c1", [200.0] * 10, duration_ms=2000.0, kind="action")
    idle = _window("c1", [16.0] * 600, duration_ms=10000.0, kind="idle")
    with_idle = measures_from_records([_cell(), busy, idle])[10_000]["time_in_jank_pct"].value
    without = measures_from_records([_cell(), busy])[10_000]["time_in_jank_pct"].value
    assert with_idle == pytest.approx(without)


def test_recorder_never_installed_reads_not_attempted():
    recs = [_cell(), _window("c1", [], attempted=False)]
    m = measures_from_records(recs)[10_000]
    assert m["jank_index"].attempted is False
    assert "frame recorder" in m["jank_index"].note


def test_truncated_window_refuses_a_number_rather_than_scoring_the_fast_frames():
    """A capped window exports no deltas at all, and must not become a clean reading."""
    w = _window("c1", [16.0], duration_ms=1000.0)
    w["instruments"]["frames"]["frame_gaps_truncated"] = True
    w["instruments"]["frames"]["frame_gaps_ms"] = None
    m = measures_from_records([_cell(), w])[10_000]
    assert m["jank_index"].attempted is True
    assert m["jank_index"].value is None
    assert "cap" in m["jank_index"].note


def test_an_incomplete_cell_still_contributes_its_rung():
    """A build that died at a rung must not make that rung disappear from the ladder."""
    recs = [_cell(completed=False), _action("c1", "keystroke", timings={"p95_ms": 900.0})]
    assert 10_000 in measures_from_records(recs)


def test_names_come_from_action_rows_not_the_lossy_embedded_copy():
    """The `actions` list inside a cell row drops names; using it would decode positionally."""
    cell = _cell()
    cell["actions"] = [{"name": None, "ran": True, "timings": {"p95_ms": 1.0}}]
    recs = [cell, _action("c1", "keystroke", timings={"p95_ms": 31.5})]
    assert measures_from_records(recs)[10_000]["keystroke_p95_ms"].value == pytest.approx(31.5)


# ── the bare-zero ban still bites after the harness-row exemptions ───────────────────────────

def test_bare_zero_outside_an_attested_block_still_fails():
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": [], "result": {"cost_ms": 0}})


def test_zero_under_a_false_attestation_still_fails():
    """`attempted: false` plus a zero is the exact conflation the ban exists to catch."""
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": [],
                          "blk": {"thing_attempted": False, "cost_ms": 0}})


def test_zero_in_a_positively_attested_instrument_block_is_accepted():
    validate_payload({"excluded_cells": [],
                      "blk": {"frames_attempted": True, "frames_over_33": 0}})


def test_attestation_does_not_leak_into_nested_blocks():
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": [],
                          "blk": {"frames_attempted": True, "inner": {"cost_ms": 0}}})


def test_real_payload_shape_round_trips(tmp_path):
    """One end-to-end pass over rows shaped exactly as the session layer writes them."""
    rows = [
        {"row_type": "run_meta", "tier": "quick"},
        _cell("c1", 1_000),
        _action("c1", "keystroke", timings={"p95_ms": 32.2}),
        _action("c1", "message_menu", timings={"open_ms": 45.8}),
        _action("c1", "scroll_after", ran=False, reason="the thread is shorter than the viewport"),
        _window("c1", [16.7] * 60 + [68.1], duration_ms=1017.0),
    ]
    path = tmp_path / "payload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    from studiobench.report.build import score_payload

    ladder = score_payload(path, [1_000])
    assert [r.tokens for r in ladder.rungs] == [1_000]
    rung = ladder.rungs[0]
    # keystroke + menu + the three frame metrics = 85% of the weight, over the 60% floor, so the
    # rung scores despite scroll being legitimately unavailable at this size.
    assert rung.usable is True
