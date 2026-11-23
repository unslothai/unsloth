# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cell_id is unique per session only, so keying on it alone blends two concurrent runs."""

from __future__ import annotations

import os

import pytest

from tests.studio.studiobench.runtime.types import Recorder, new_session_id
from tests.studio.studiobench.sweep import floor_table


def _cell(session: str, cell_id: str, p50: float) -> list[dict]:
    """One completed cell plus the keystroke action that carries its timing."""
    return [
        {
            "row_type": "action",
            "session_id": session,
            "cell_id": cell_id,
            "action": "keystroke",
            "ran": True,
            "timings": {"p50_ms": p50},
            "counts": {},
        },
        {"row_type": "cell", "session_id": session, "cell_id": cell_id, "completed": True},
    ]


def _two_session_payload() -> list[dict]:
    """Session B ran concurrently with A, so its treatment cells read much slower for the same cell id."""
    rows: list[dict] = []
    for sess, base0, treat0, base1, treat1 in (
        ("91c4d6d94da8", 45.0, 58.7, 51.1, 73.4),
        ("430f0b831dda", 37.9, 73.6, 47.3, 144.5),
    ):
        rows += _cell(sess, "r1M.base.rep0", base0)
        rows += _cell(sess, "r1M.treatment.rep0", treat0)
        rows += _cell(sess, "r1M.base.rep1", base1)
        rows += _cell(sess, "r1M.treatment.rep1", treat1)
    return rows


def test_cell_metrics_refuses_to_collapse_two_sessions():
    with pytest.raises(SystemExit) as caught:
        floor_table.cell_metrics(_two_session_payload())
    message = str(caught.value)
    assert "more than one session" in message, (
        "cell_metrics keyed on cell_id alone and returned the last writer's values. That is how a "
        "payload holding two concurrent runs reported a 149.8% regression that does not exist."
    )
    assert "r1M.base.rep0" in message
    assert "91c4d6d94da8" in message and "430f0b831dda" in message


def test_a_resumed_run_is_not_mistaken_for_two_concurrent_ones():
    """Refusal keys on a cell completing twice, not on a payload holding two sessions, as resumes do."""
    rows = [
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
        {
            "row_type": "cell",
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s1",
            "completed": False,
        },
        {
            "row_type": "cell",
            "cell_id": "r100K.treatment.rep0",
            "session_id": "s2",
            "completed": True,
        },
    ]
    assert floor_table.collided_cells(rows) == {}
    assert set(floor_table.cell_metrics(rows)) == {"r100K.base.rep0", "r100K.treatment.rep0"}


def test_a_cell_completing_twice_is_what_is_refused():
    """The other direction: two COMPLETED copies of one cell id is the concurrent-run signature."""
    rows = [
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s2", "completed": True},
    ]
    assert floor_table.collided_cells(rows) == {"r100K.base.rep0": {"s1", "s2"}}
    with pytest.raises(SystemExit):
        floor_table.cell_metrics(rows)


def test_a_single_session_payload_is_unaffected():
    rows = [r for r in _two_session_payload() if r["session_id"] == "91c4d6d94da8"]
    cells = floor_table.cell_metrics(rows)
    assert set(cells) == {
        "r1M.base.rep0",
        "r1M.treatment.rep0",
        "r1M.base.rep1",
        "r1M.treatment.rep1",
    }
    assert cells["r1M.treatment.rep1"]["keystroke.p50_ms"] == 73.4


def test_a_session_can_be_selected_explicitly():
    rows = _two_session_payload()
    a = floor_table.cell_metrics(rows, session = "91c4d6d94da8")
    b = floor_table.cell_metrics(rows, session = "430f0b831dda")
    assert a["r1M.treatment.rep1"]["keystroke.p50_ms"] == 73.4
    assert b["r1M.treatment.rep1"]["keystroke.p50_ms"] == 144.5


def _two_session_no_collision() -> list[dict]:
    """Two sessions with disjoint repetitions, as a sharded or continued run produces."""
    rows: list[dict] = []
    for sess, reps in (("91c4d6d94da8", (0, 1)), ("430f0b831dda", (2, 3))):
        for rep in reps:
            rows += _cell(sess, f"r1M.base.rep{rep}", 40.0 + rep)
            rows += _cell(sess, f"r1M.treatment.rep{rep}", 60.0 + rep)
    return rows


def test_paired_keys_on_the_session_and_does_not_cross_match():
    """Pairing must never match one session's base with another session's treatment."""
    pairs = floor_table.paired(_two_session_no_collision())["keystroke.p50_ms"]
    assert sorted(pairs) == sorted([(40.0, 60.0), (41.0, 61.0), (42.0, 62.0), (43.0, 63.0)]), (
        "pairing crossed the sessions. Two sessions both produce rep0, so a key without the "
        "session matches a base measured under one machine load against a treatment measured "
        "under another and calls it a repetition."
    )
    assert len(pairs) == 4


def test_pairing_refuses_a_payload_whose_cells_completed_twice():
    """Pairing must refuse colliding cells itself; paired never reaches the refusal in cell_metrics."""
    with pytest.raises(SystemExit) as caught:
        floor_table.paired(_two_session_payload())
    assert "completed under more than one session" in str(caught.value)


def test_the_refusal_does_not_send_the_reader_to_a_function_that_also_refuses():
    """A refusal must not point the reader to a remedy, like paired, that refuses the same payload."""
    with pytest.raises(SystemExit) as caught:
        floor_table.cell_metrics(_two_session_payload())
    message = str(caught.value)
    assert "use `paired`" not in message
    assert "Split the payload by session" in message


def test_sessions_in_lists_only_completed_cells():
    rows = _two_session_payload() + [
        {
            "row_type": "cell",
            "session_id": "deadbeef",
            "cell_id": "r1M.base.rep2",
            "completed": False,
        }
    ]
    assert floor_table.sessions_in(rows) == {"91c4d6d94da8", "430f0b831dda"}


def test_a_second_live_session_is_refused(tmp_path):
    first = Recorder(tmp_path / "payload.jsonl", new_session_id())
    try:
        with pytest.raises(SystemExit) as caught:
            Recorder(tmp_path / "payload.jsonl", new_session_id())
        assert "still running" in str(caught.value), (
            "a second concurrent run was allowed to append to a live output directory. Both runs "
            "then contend with each other and write the same cell ids into one file."
        )
    finally:
        first.close()


def test_the_directory_is_reusable_once_the_first_run_closes(tmp_path):
    first = Recorder(tmp_path / "payload.jsonl", new_session_id())
    first.close()
    second = Recorder(tmp_path / "payload.jsonl", new_session_id())
    second.close()


def test_a_marker_from_a_dead_process_does_not_block_forever(tmp_path):
    """A crashed run must not lock the directory against every later one."""
    stale = tmp_path / ".running.deadsession"
    tmp_path.mkdir(parents = True, exist_ok = True)
    # A pid that cannot be alive: one past pid_max.
    with open("/proc/sys/kernel/pid_max", encoding = "utf-8") as fh:
        dead_pid = int(fh.read().strip()) - 1
    stale.write_text(f"{dead_pid} deadsession\n", encoding = "utf-8")
    if _pid_alive(dead_pid):
        pytest.skip("the chosen pid happens to be alive")
    rec = Recorder(tmp_path / "payload.jsonl", new_session_id())
    rec.close()
    assert not stale.exists(), "a marker naming a dead process should be cleared, not obeyed"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_the_new_row_types_are_registered_in_the_schema(tmp_path):
    """A new row type must be registered in ROW_TYPES with its emitter, or the first run aborts on it."""
    rec = Recorder(tmp_path / "payload.jsonl", new_session_id())
    try:
        rec.emit(
            {
                "row_type": "cell_aborted",
                "cell_id": "r1M.treatment.rep0",
                "reason": "budget exhausted",
            }
        )
        rec.emit(
            {
                "row_type": "comparability",
                "key": "cmp:0123456789",
                "fields": {"corpus_hash": "ac9d5d8e"},
            }
        )
    finally:
        rec.close()
    written = (tmp_path / "payload.jsonl").read_text(encoding = "utf-8")
    assert "cell_aborted" in written and "comparability" in written
