# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two sessions in one payload must not be collapsed into one, and must not happen in the first
place.

Found by making the mistake. A launcher was started twice, so two full sessions ran concurrently
against one `--out` and both appended. `cell_id` is unique within a session and not across them,
so `cell_metrics` keyed on it alone reported whichever session was written last -- a blend of two
runs that were contending with each other, under one label.

The numbers below are the real ones from that payload.
"""

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
    """The shape that produced the withdrawn +149.8%.

    Session B ran concurrently with session A, so its treatment cells are much slower: 144.5 ms
    against 73.4 ms for the same cell id.
    """
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
    # The colliding cell is NAMED, not just counted. A refusal that says only "two sessions" sends
    # the reader back to the payload to find out which reading it was protecting them from.
    assert "r1M.base.rep0" in message
    assert "91c4d6d94da8" in message and "430f0b831dda" in message


def test_a_resumed_run_is_not_mistaken_for_two_concurrent_ones():
    """The refusal keys on a COLLIDING COMPLETED CELL, not on the payload holding two sessions.

    `--resume` re-runs the arm that died under a NEW session id into the same shard directory, so a
    resumed payload legitimately carries two sessions. The attempt that died is not marked
    completed, so no cell id completes twice and there is nothing to refuse. Keying the refusal on
    the session count instead would reject every resumed run -- deleting good readings to guard
    against a collision that is not there.
    """
    rows = [
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
        # died, then re-run under a new session id
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


def test_paired_keys_on_the_session_and_does_not_cross_match():
    """Pairing must not match one session's base against another session's treatment."""
    pairs = floor_table.paired(_two_session_payload())["keystroke.p50_ms"]
    assert sorted(pairs) == sorted([(45.0, 58.7), (51.1, 73.4), (37.9, 73.6), (47.3, 144.5)]), (
        "pairing crossed the sessions. Two sessions both produce rep0, so a key without the "
        "session matches a base measured under one machine load against a treatment measured "
        "under another and calls it a repetition."
    )
    assert len(pairs) == 4


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


# ── the write-time guard ────────────────────────────────────────────────────


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
    # A pid that cannot be alive: this process's own pid is taken, so use one past the max.
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


# ── the rows these guards emit must actually be emittable ──────────────────


def test_the_new_row_types_are_registered_in_the_schema(tmp_path):
    """A guard that cannot write its own row is a guard that crashes the run it protects.

    Both of these were added and neither was registered in ROW_TYPES, so the first clean run
    aborted two seconds in with `row_type must be one of [...], got 'comparability'`. The emitter
    and the schema have to move together.
    """
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
