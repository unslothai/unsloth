# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cell that was re-run belongs to the run that FINISHED it, not to the one that died.

`--resume` appends to the payload it continues and re-runs the cells that did not complete, and
`make_cell_id` is deterministic, so the retry lands under the cell id of the attempt that died.
Two readings came out of that, both silent and both wrong:

  THE DEAD ATTEMPT'S FRAMES BECAME THE RETRY'S. Cell rows were scoped by session; the `action`
  and `window` rows underneath them were collected by `cell_id` alone, so a 100 ms frame recorded
  by the run that crashed stayed the RETRY's `max_frame_ms` and its gaps stayed in the retry's
  jank distribution -- inside an A/B ratio, which is where cross-session drift does the most
  damage.

  THE SUCCESSFUL RETRY DID NOT COUNT. The rung kept the failure forever, so `--report` on a
  payload whose only failure had already been re-run successfully still printed INCOMPLETE and
  scored the rung zero, dragging the headline down with it.

What must NOT change: two REPETITIONS are two observations, and a rung with a failed rep is still
not a clean rung. Only a later attempt at the SAME cell supersedes an earlier one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.report.build import score_payload  # noqa: E402
from studiobench.runtime.ab import readings_by_arm  # noqa: E402

NOW = "s-now"
OLD = "s-before"


def _cell(
    cell_id,
    arm,
    *,
    completed,
    session,
    tokens = 10_000,
    rep = 0,
):
    row = {
        "row_type": "cell",
        "cell_id": cell_id,
        "session_id": session,
        "target_tokens": tokens,
        "completed": completed,
        "cell": {"arm": arm, "rep": rep},
    }
    if not completed:
        row["failure"] = {"kind": "TimeoutError", "message": "the machine stalled"}
    return row


def _keystroke(cell_id, session, p95):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "session_id": session,
        "action": "keystroke",
        "ran": True,
        "expect_ok": True,
        "timings": {"p95_ms": p95},
    }


def _window(
    cell_id,
    session,
    max_frame,
    gaps,
    duration_ms = 1000.0,
):
    return {
        "row_type": "window",
        "cell_id": cell_id,
        "session_id": session,
        "name": "stream:film",
        "kind": "stream",
        "t_open_ms": 0.0,
        "duration_ms": duration_ms,
        "instruments": {
            "frames": {
                "frames_attempted": True,
                "max_frame_ms": max_frame,
                "frame_gaps_ms": list(gaps),
            }
        },
    }


# ── the ratio ───────────────────────────────────────────────────────


def test_the_dead_attempts_frames_are_not_the_retrys_frames():
    cell_id = "r10K.base.rep0"
    records = [
        _cell(cell_id, "base", completed = False, session = OLD),
        _window(cell_id, OLD, 100.0, [100.0, 16.0]),
        _cell(cell_id, "base", completed = True, session = NOW),
        _window(cell_id, NOW, 17.0, [16.0, 17.0]),
    ]
    reading = readings_by_arm(records, session_id = NOW)["base"][(10_000, 0)]
    assert reading["max_frame_ms"].value == 17.0
    # The 100 ms gap is out of the pooled distribution too, not just out of the maximum.
    assert (
        reading["jank_index"].value
        == readings_by_arm([records[2], records[3]], session_id = NOW)["base"][(10_000, 0)][
            "jank_index"
        ].value
    )


def test_two_windows_of_the_same_attempt_are_still_pooled():
    """The control: filtering by attempt must not turn into filtering out real windows."""

    cell_id = "r10K.base.rep0"
    records = [
        _cell(cell_id, "base", completed = True, session = NOW),
        _window(cell_id, NOW, 17.0, [16.0, 17.0]),
        _window(cell_id, NOW, 100.0, [100.0, 16.0]),
    ]
    reading = readings_by_arm(records, session_id = NOW)["base"][(10_000, 0)]
    assert reading["max_frame_ms"].value == 100.0


# ── the score ───────────────────────────────────────────────────────


def _payload(directory, rows):
    Path(directory).mkdir(parents = True, exist_ok = True)
    path = Path(directory) / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def _finished(
    session,
    *,
    tokens = 10_000,
    rep = 0,
):
    cell_id = f"r{tokens}.A0.rep{rep}"
    return [
        _cell(cell_id, "A0", completed = True, session = session, tokens = tokens, rep = rep),
        _keystroke(cell_id, session, 40.0),
        _window(cell_id, session, 30.0, [16.0, 17.0, 16.0]),
    ]


def _died(
    session,
    *,
    tokens = 10_000,
    rep = 0,
):
    cell_id = f"r{tokens}.A0.rep{rep}"
    return [
        _cell(cell_id, "A0", completed = False, session = session, tokens = tokens, rep = rep),
        _keystroke(cell_id, session, 400.0),
        _window(cell_id, session, 900.0, [900.0, 16.0]),
    ]


def test_a_resumed_cell_scores_as_if_the_crash_had_not_happened(tmp_path):
    resumed = score_payload(_payload(tmp_path / "a", _died(OLD) + _finished(NOW)), [10_000])
    clean = score_payload(_payload(tmp_path / "b", _finished(NOW)), [10_000])

    rung = resumed.rungs[0]
    assert rung.complete is True
    assert rung.incomplete_reason is None
    assert rung.score > 0
    # Not merely "not zero": the superseded attempt must not move the number at all.
    assert rung.to_json() == clean.rungs[0].to_json()
    assert resumed.aggregate == clean.aggregate


def test_a_failed_repetition_still_makes_the_rung_incomplete(tmp_path):
    """The control: reps are independent observations and one of them failing is the answer."""

    rows = _finished(NOW, rep = 0) + _died(NOW, rep = 1)
    ladder = score_payload(_payload(tmp_path / "c", rows), [10_000])
    assert ladder.rungs[0].complete is False
    assert "TimeoutError" in (ladder.rungs[0].incomplete_reason or "")
    assert ladder.rungs[0].score == 0.0


def test_a_cell_that_was_never_re_run_keeps_its_failure(tmp_path):
    """The other control: a crash with no retry behind it is still a crash."""

    ladder = score_payload(_payload(tmp_path / "d", _died(OLD)), [10_000])
    assert ladder.rungs[0].complete is False
    assert "TimeoutError" in (ladder.rungs[0].incomplete_reason or "")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
