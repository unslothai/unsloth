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

from studiobench.scoring.from_payload import (  # noqa: E402
    latest_attempt_rows,
    measures_from_records,
)
from studiobench.scoring.schema import (  # noqa: E402
    PayloadSchemaError,
    validate_payload,
)


def _cell(
    cell_id = "c1",
    tokens = 10_000,
    completed = True,
):
    return {"row_type": "cell", "cell_id": cell_id, "target_tokens": tokens, "completed": completed}


def _action(
    cell_id,
    name,
    ran = True,
    timings = None,
    reason = None,
):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "action": name,
        "ran": ran,
        "timings": timings or {},
        "reason": reason,
    }


def _window(
    cell_id,
    gaps,
    duration_ms = 1000.0,
    kind = "action",
    attempted = True,
):
    frames = {"frames_attempted": attempted}
    if attempted:
        frames.update(
            {
                "frame_gaps_ms": gaps,
                "frame_gaps_truncated": False,
                "frame_gaps_total": len(gaps),
                "max_frame_ms": max(gaps) if gaps else None,
            }
        )
    return {
        "row_type": "window",
        "cell_id": cell_id,
        "kind": kind,
        "duration_ms": duration_ms,
        "instruments": {"frames": frames},
    }


# ── the three absences, kept apart ──────────────────────────────────────────────────────────


def test_action_absent_from_scene_is_not_attempted():
    m = measures_from_records([_cell()])["10000" if False else 10_000]
    keystroke = m["keystroke_p95_ms"]
    assert keystroke.attempted is False
    assert keystroke.value is None
    assert "not in this scene" in keystroke.note


def test_action_that_did_not_run_is_attempted_and_failed_with_its_reason():
    """The distinction that matters at 1K: scroll cannot run, and that is not a fast scroll."""
    recs = [
        _cell(),
        _action("c1", "scroll_after", ran = False, reason = "the thread is shorter than the viewport"),
    ]
    scroll = measures_from_records(recs)[10_000]["scroll_settle_ms"]
    assert scroll.attempted is True
    assert scroll.value is None
    assert "shorter than the viewport" in scroll.note


def test_action_that_ran_without_the_timing_key_is_failed_not_zero():
    recs = [_cell(), _action("c1", "keystroke", ran = True, timings = {"p50_ms": 10.0})]
    keystroke = measures_from_records(recs)[10_000]["keystroke_p95_ms"]
    assert keystroke.attempted is True
    assert keystroke.value is None
    assert "recorded no p95_ms" in keystroke.note


def test_a_real_reading_carries_the_key_it_came_from():
    recs = [_cell(), _action("c1", "keystroke", timings = {"p95_ms": 31.5})]
    keystroke = measures_from_records(recs)[10_000]["keystroke_p95_ms"]
    assert keystroke.value == pytest.approx(31.5)
    assert keystroke.note == "keystroke.p95_ms"


def test_scroll_settle_says_it_is_gesture_time_not_settle_time():
    """The one anchor whose recorded quantity is not what its name says. Never silent."""
    recs = [_cell(), _action("c1", "scroll_after", timings = {"gesture_ms": 466.5})]
    scroll = measures_from_records(recs)[10_000]["scroll_settle_ms"]
    assert scroll.value == pytest.approx(466.5)
    assert "not post-gesture settle" in scroll.note


# ── frame metrics ───────────────────────────────────────────────────────────────────────────


def test_frame_metrics_pool_the_active_windows():
    recs = [_cell(), _window("c1", [16.0] * 50 + [200.0], duration_ms = 1000.0)]
    m = measures_from_records(recs)[10_000]
    assert m["max_frame_ms"].value == pytest.approx(200.0)
    assert m["jank_index"].value is not None and m["jank_index"].value > 0
    assert m["time_in_jank_pct"].value is not None


def test_idle_windows_are_excluded_so_jank_is_not_diluted():
    busy = _window("c1", [200.0] * 10, duration_ms = 2000.0, kind = "action")
    idle = _window("c1", [16.0] * 600, duration_ms = 10000.0, kind = "idle")
    with_idle = measures_from_records([_cell(), busy, idle])[10_000]["time_in_jank_pct"].value
    without = measures_from_records([_cell(), busy])[10_000]["time_in_jank_pct"].value
    assert with_idle == pytest.approx(without)


def test_recorder_never_installed_reads_not_attempted():
    recs = [_cell(), _window("c1", [], attempted = False)]
    m = measures_from_records(recs)[10_000]
    assert m["jank_index"].attempted is False
    assert "frame recorder" in m["jank_index"].note


def test_a_window_that_recorded_no_frames_fails_the_pool_instead_of_dropping_out():
    """A complete freeze during one action must not be answered for by the windows beside it.

    `compute_frame_stats` already calls a single attempted-but-frameless window a failed
    measurement. Pooled, that window used to be skipped, and the cell came back with the
    remaining window's clean numbers.
    """
    normal = _window("c1", [16.0] * 50 + [40.0], duration_ms = 1000.0)
    frozen = _window("c1", [], duration_ms = 4000.0)
    m = measures_from_records([_cell(), normal, frozen])[10_000]
    for key in ("time_in_jank_pct", "jank_index", "max_frame_ms"):
        assert m[key].attempted is True
        assert m[key].value is None
        assert "no frames at all" in m[key].note
    # and the giveaway: without the fix these were byte-identical to the cell without the freeze
    alone = measures_from_records([_cell(), normal])[10_000]
    assert alone["max_frame_ms"].value == pytest.approx(40.0)


def test_a_window_the_recorder_was_never_installed_in_is_still_only_skipped():
    """The control. Not-attempted is an absent instrument, not an unmeasured window."""
    normal = _window("c1", [16.0] * 50 + [40.0], duration_ms = 1000.0)
    never = _window("c1", [], duration_ms = 4000.0, attempted = False)
    m = measures_from_records([_cell(), normal, never])[10_000]
    assert m["max_frame_ms"].value == pytest.approx(40.0)
    assert m["jank_index"].value is not None
    assert m["time_in_jank_pct"].value is not None


def test_a_pool_where_every_window_has_frames_is_unaffected():
    """The other control: the ordinary multi-window cell still pools as before."""
    a = _window("c1", [16.0] * 50 + [40.0], duration_ms = 1000.0)
    b = _window("c1", [16.0] * 20 + [900.0], duration_ms = 2000.0)
    m = measures_from_records([_cell(), a, b])[10_000]
    assert m["max_frame_ms"].value == pytest.approx(900.0)
    assert m["jank_index"].value is not None and m["jank_index"].value > 0
    assert m["time_in_jank_pct"].value is not None


def test_truncated_window_refuses_a_number_rather_than_scoring_the_fast_frames():
    """A capped window exports no deltas at all, and must not become a clean reading."""
    w = _window("c1", [16.0], duration_ms = 1000.0)
    w["instruments"]["frames"]["frame_gaps_truncated"] = True
    w["instruments"]["frames"]["frame_gaps_ms"] = None
    m = measures_from_records([_cell(), w])[10_000]
    assert m["jank_index"].attempted is True
    assert m["jank_index"].value is None
    assert "cap" in m["jank_index"].note


def test_an_incomplete_cell_still_contributes_its_rung():
    """A build that died at a rung must not make that rung disappear from the ladder."""
    recs = [_cell(completed = False), _action("c1", "keystroke", timings = {"p95_ms": 900.0})]
    assert 10_000 in measures_from_records(recs)


def test_names_come_from_action_rows_not_the_lossy_embedded_copy():
    """The `actions` list inside a cell row drops names; using it would decode positionally."""
    cell = _cell()
    cell["actions"] = [{"name": None, "ran": True, "timings": {"p95_ms": 1.0}}]
    recs = [cell, _action("c1", "keystroke", timings = {"p95_ms": 31.5})]
    assert measures_from_records(recs)[10_000]["keystroke_p95_ms"].value == pytest.approx(31.5)


# ── the bare-zero ban still bites after the harness-row exemptions ───────────────────────────


def test_bare_zero_outside_an_attested_block_still_fails():
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": [], "result": {"cost_ms": 0}})


def test_zero_under_a_false_attestation_still_fails():
    """`attempted: false` plus a zero is the exact conflation the ban exists to catch."""
    with pytest.raises(PayloadSchemaError):
        validate_payload({"excluded_cells": [], "blk": {"thing_attempted": False, "cost_ms": 0}})


def test_zero_in_a_positively_attested_instrument_block_is_accepted():
    validate_payload({"excluded_cells": [], "blk": {"frames_attempted": True, "frames_over_33": 0}})


def test_attestation_does_not_leak_into_nested_blocks():
    with pytest.raises(PayloadSchemaError):
        validate_payload(
            {"excluded_cells": [], "blk": {"frames_attempted": True, "inner": {"cost_ms": 0}}}
        )


def test_real_payload_shape_round_trips(tmp_path):
    """One end-to-end pass over rows shaped exactly as the session layer writes them."""
    rows = [
        {"row_type": "run_meta", "tier": "quick"},
        _cell("c1", 1_000),
        _action("c1", "keystroke", timings = {"p95_ms": 32.2}),
        _action("c1", "message_menu", timings = {"open_ms": 45.8}),
        _action("c1", "scroll_after", ran = False, reason = "the thread is shorter than the viewport"),
        _window("c1", [16.7] * 60 + [68.1], duration_ms = 1017.0),
    ]
    path = tmp_path / "payload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")

    from studiobench.report.build import score_payload

    ladder = score_payload(path, [1_000])
    assert [r.tokens for r in ladder.rungs] == [1_000]
    rung = ladder.rungs[0]
    # keystroke + menu + the three frame metrics = 85% of the weight, over the 60% floor, so the
    # rung scores despite scroll being legitimately unavailable at this size.
    assert rung.usable is True


# ── per-cell readings, which is what makes an A/B paired ─────────────────────────────────────


def test_measures_by_cell_keeps_every_repetition():
    """Collapsing reps leaves the bootstrap with one pair per metric and nothing to resample."""
    from studiobench.scoring.from_payload import measures_by_cell

    rows = []
    for rep in (0, 1):
        cid = f"c{rep}"
        cell = _cell(cid)
        cell["cell"] = {"rep": rep}
        rows.append(cell)
        rows.append(_action(cid, "keystroke", timings = {"p95_ms": 30.0 + rep}))

    by_cell = measures_by_cell(rows)
    assert sorted(by_cell) == [(10_000, 0), (10_000, 1)]
    assert by_cell[(10_000, 0)]["keystroke_p95_ms"].value == pytest.approx(30.0)
    assert by_cell[(10_000, 1)]["keystroke_p95_ms"].value == pytest.approx(31.0)


def test_measures_by_cell_does_not_mix_readings_between_cells():
    """Each cell must see only its own action rows, or rep 0 inherits rep 1's timings."""
    from studiobench.scoring.from_payload import measures_by_cell

    a, b = _cell("cA"), _cell("cB")
    a["cell"], b["cell"] = {"rep": 0}, {"rep": 1}
    rows = [
        a,
        b,
        _action("cA", "menu_open" if False else "message_menu", timings = {"open_ms": 10.0}),
        _action("cB", "message_menu", timings = {"open_ms": 900.0}),
    ]
    by_cell = measures_by_cell(rows)
    assert by_cell[(10_000, 0)]["menu_open_ms"].value == pytest.approx(10.0)
    assert by_cell[(10_000, 1)]["menu_open_ms"].value == pytest.approx(900.0)


# ── ran is not "did what it claimed" ────────────────────────────────────────────────────────


def test_an_action_whose_own_assertion_failed_is_not_a_reading():
    """`report/payload.py` lists this cell under EXCLUDED CELLS with "must not be quoted".

    Before this, the same action was excluded in the report and load-bearing in the headline: the
    keystroke row reported `ran: true, expect_ok: false` because the composer never accepted the
    commanded characters, and its p95 still set the rung's highest-weight metric.
    """
    recs = [
        _cell(),
        {
            **_action("c1", "keystroke", ran = True, timings = {"p95_ms": 12.0}),
            "expect_ok": False,
            "reason": "typed 12 characters but the composer value grew by 0",
        },
    ]
    keystroke = measures_from_records(recs)[10_000]["keystroke_p95_ms"]
    assert keystroke.attempted is True
    assert keystroke.value is None
    assert "its own assertion failed" in keystroke.note
    assert "grew by 0" in keystroke.note


def test_an_action_whose_assertion_passed_is_still_a_reading():
    recs = [
        _cell(),
        {**_action("c1", "keystroke", ran = True, timings = {"p95_ms": 12.0}), "expect_ok": True},
    ]
    assert measures_from_records(recs)[10_000]["keystroke_p95_ms"].value == 12.0


# ---------------------------------------------------------------------------------------
# The latest attempt is the last one that WROTE anything, not the last one that finished
# ---------------------------------------------------------------------------------------
#
# `CellRunner.run` writes its terminal cell row in a `finally`, which a SIGKILL, an OOM kill or a
# lost machine never reaches -- while the Recorder has already flushed and fsynced every action and
# window row before it. Keyed on cell rows alone, a resume hard-killed inside a cell left the
# older, completed attempt named as the latest, and `_resume_set` skipped it.


def _stamped(row, session):
    return {**row, "session_id": session}


def test_a_killed_attempt_supersedes_the_completed_one_it_was_re_running():
    records = [
        _stamped(_cell("c1", completed = True), "sess-1"),
        _stamped(_action("c1", "keystroke", timings = {"p95_ms": 40.0}), "sess-1"),
        # sess-2 got this far and was killed: no cell row was ever written.
        _stamped(_action("c1", "keystroke", timings = {"p95_ms": 900.0}), "sess-2"),
    ]

    kept = latest_attempt_rows(records)

    # The old attempt's rows are gone, so nothing reports c1 as completed.
    assert not [r for r in kept if r.get("row_type") == "cell"]
    assert [r["timings"]["p95_ms"] for r in kept if r.get("row_type") == "action"] == [900.0]


def test_a_window_row_alone_is_enough_to_prove_a_newer_attempt():
    records = [
        _stamped(_cell("c1", completed = True), "sess-1"),
        _stamped(_window("c1", [16.0, 17.0]), "sess-1"),
        _stamped(_window("c1", [400.0]), "sess-2"),
    ]

    kept = latest_attempt_rows(records)

    assert not [r for r in kept if r.get("row_type") == "cell"]
    assert len([r for r in kept if r.get("row_type") == "window"]) == 1


def test_a_completed_retry_still_supersedes_the_attempt_that_died():
    """The control this rule already existed for, unchanged: the retry that FINISHED wins."""

    records = [
        _stamped(_cell("c1", completed = False), "sess-1"),
        _stamped(_action("c1", "keystroke", timings = {"p95_ms": 900.0}), "sess-1"),
        _stamped(_action("c1", "keystroke", timings = {"p95_ms": 40.0}), "sess-2"),
        _stamped(_cell("c1", completed = True), "sess-2"),
    ]

    kept = latest_attempt_rows(records)

    assert [r["completed"] for r in kept if r.get("row_type") == "cell"] == [True]
    assert [r["timings"]["p95_ms"] for r in kept if r.get("row_type") == "action"] == [40.0]


def test_a_payload_with_no_session_ids_is_kept_whole():
    """A payload from before the recorder stamped sessions cannot be split into attempts, and
    dropping it would lose the run."""

    records = [_cell("c1"), _action("c1", "keystroke", timings = {"p95_ms": 40.0})]

    assert latest_attempt_rows(records) == records


def test_rows_of_another_cell_are_untouched():
    records = [
        _stamped(_cell("c1", completed = True), "sess-1"),
        _stamped(_action("c2", "keystroke", timings = {"p95_ms": 40.0}), "sess-1"),
        _stamped(_action("c1", "keystroke", timings = {"p95_ms": 900.0}), "sess-2"),
    ]

    kept = latest_attempt_rows(records)

    assert any(r.get("cell_id") == "c2" for r in kept)
