# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What may enter an A/B ratio: a completed cell, from THIS session.

Two ways a ratio was formed out of things that are not comparable, both of them silent:

  A CRASHED ARM CAN WIN. An arm that died part way through a cell still wrote the action rows it
  had already measured. Pairing those against a completed cell on the other side turned a crash
  into an improvement -- a treatment cell holding one 50 ms keystroke against a completed 100 ms
  base cell reported IMPROVED, and the crash appeared only in the excluded list underneath.

  RESUME REOPENS THE SESSION. `--resume` appends to the payload of the run it is continuing, and
  the new cells carry a new session id. `assert_comparable` refuses two sessions, but both sides
  are labelled with the CURRENT one, so the 8% cross-session drift term it exists to refuse came
  back in through the rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.ab import compare_arms, readings_by_arm  # noqa: E402

SESSION = "s-now"
OLD_SESSION = "s-before"


def _cell(
    cell_id,
    arm,
    *,
    completed = True,
    session = SESSION,
    tokens = 10_000,
    rep = 0,
):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "session_id": session,
        "target_tokens": tokens,
        "completed": completed,
        "cell": {"arm": arm, "rep": rep},
    }


def _keystroke(
    cell_id,
    p95,
    *,
    session = SESSION,
):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "session_id": session,
        "action": "keystroke",
        "ran": True,
        "expect_ok": True,
        "timings": {"p95_ms": p95},
    }


def _gate(
    name,
    passed,
    *,
    cell_id = None,
    session = SESSION,
):
    row = {"row_type": "gate", "name": name, "passed": passed, "detail": {}, "session_id": session}
    if cell_id is not None:
        row["cell_id"] = cell_id
    return row


def _pairs(records, session_id = SESSION):
    result = compare_arms(
        records,
        "base",
        "treatment",
        bench_version = "0.1.0",
        corpus_hash = "c0ffee",
        session_id = session_id,
        label = "test",
    )
    return [p for p in result.pairs if p.metric_key == "keystroke_p95_ms"], result


def test_a_crashed_treatment_cell_does_not_become_a_win():
    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment", completed = False),
        _keystroke("r10K.treatment.rep0", 50.0),
    ]
    assert "treatment" not in readings_by_arm(records)
    pairs, result = _pairs(records)
    assert pairs == []
    assert result.verdict == "NO READING"


def test_a_cell_that_failed_a_per_cell_gate_does_not_become_a_win():
    """REGRESSION. A completeness gate is advisory where it is emitted, so the cell arrives here
    `completed=True` with a full set of timings -- and cheaper ones, because a thread that lost its
    middle renders fewer rows. `excluded_from_rows` reads the same gate row into `excluded_cells`,
    but nothing filters on that block, and `ab.md` is scored from `readings_by_arm`.
    """

    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("thread_complete", False, cell_id = "r10K.treatment.rep0"),
    ]
    assert "treatment" not in readings_by_arm(records)
    pairs, result = _pairs(records)
    assert pairs == []
    assert result.verdict == "NO READING"


def test_a_failed_follows_the_stream_gate_disqualifies_its_cell_too():
    """The streamed reply leaving the viewport is the other way a defect gets cheaper to paint."""

    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("follows_the_stream", False, cell_id = "r10K.treatment.rep0"),
    ]
    assert "treatment" not in readings_by_arm(records)


def test_a_passing_gate_leaves_its_cell_alone():
    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _gate("thread_complete", True, cell_id = "r10K.base.rep0"),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("thread_complete", True, cell_id = "r10K.treatment.rep0"),
    ]
    pairs, _result = _pairs(records)
    assert len(pairs) == 1
    assert pairs[0].ratio == 0.5


def test_a_failed_timer_clamp_does_not_throw_away_the_rest_of_the_cell():
    """A per-cell gate is not automatically fatal, and this one says so itself.

    `timer_clamp` fails whenever idle calibration cannot establish a floor -- an overloaded
    machine, or the frames instrument simply not being loaded. `session.py` calls that "NOT fatal,
    and NOT silently zero": blocked time is a subtraction against the floor, so `busy_pct` is null
    with the reason attached and every other column stands. Excluding the cell would delete
    keystroke, frame and census readings that were measured correctly, most often on the machines
    least able to spare a repetition.
    """

    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _gate("timer_clamp", False, cell_id = "r10K.base.rep0"),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("timer_clamp", False, cell_id = "r10K.treatment.rep0"),
    ]
    assert set(readings_by_arm(records)) == {"base", "treatment"}
    pairs, _result = _pairs(records)
    assert len(pairs) == 1
    assert pairs[0].ratio == 0.5


def test_a_failed_run_level_gate_does_not_empty_the_table():
    """`production_build` and `reportable_tier` carry no cell id. Read as per-cell they would
    disqualify every cell on both arms and turn a fast-tier A/B into an empty table."""

    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("reportable_tier", False),
        _gate("production_build", False),
    ]
    assert set(readings_by_arm(records)) == {"base", "treatment"}
    pairs, _result = _pairs(records)
    assert len(pairs) == 1


def test_a_retry_that_passed_is_not_disqualified_by_the_dead_attempt_gate():
    """`latest_attempt_rows` drops a superseded attempt's `cell`, `action` and `window` rows, but
    NOT its gate row, and `--resume` reuses the cell id. Scoping the gate to the winning attempt is
    what keeps the retry countable."""

    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment", completed = False, session = OLD_SESSION),
        _keystroke("r10K.treatment.rep0", 999.0, session = OLD_SESSION),
        _gate("thread_complete", False, cell_id = "r10K.treatment.rep0", session = OLD_SESSION),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
        _gate("thread_complete", True, cell_id = "r10K.treatment.rep0"),
    ]
    assert "treatment" in readings_by_arm(records)
    pairs, _result = _pairs(records)
    assert len(pairs) == 1
    assert pairs[0].ratio == 0.5


def test_two_completed_cells_still_pair():
    records = [
        _cell("r10K.base.rep0", "base"),
        _keystroke("r10K.base.rep0", 100.0),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
    ]
    pairs, _result = _pairs(records)
    assert len(pairs) == 1
    assert pairs[0].ratio == 0.5


def test_cells_from_a_previous_session_are_not_paired_with_this_one():
    records = [
        _cell("r10K.base.rep0", "base", session = OLD_SESSION),
        _keystroke("r10K.base.rep0", 100.0, session = OLD_SESSION),
        _cell("r10K.treatment.rep0", "treatment"),
        _keystroke("r10K.treatment.rep0", 50.0),
    ]
    assert "base" not in readings_by_arm(records, session_id = SESSION)
    pairs, _result = _pairs(records)
    assert pairs == []


def test_a_payload_without_session_ids_is_still_readable():
    """Older payloads, and the hand-built ones in these tests, carry no session id per row."""

    records = [
        {k: v for k, v in _cell("a", "base").items() if k != "session_id"},
        {k: v for k, v in _keystroke("a", 100.0).items() if k != "session_id"},
        {k: v for k, v in _cell("b", "treatment").items() if k != "session_id"},
        {k: v for k, v in _keystroke("b", 50.0).items() if k != "session_id"},
    ]
    assert set(readings_by_arm(records, session_id = SESSION)) == {"base", "treatment"}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
