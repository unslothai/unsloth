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
