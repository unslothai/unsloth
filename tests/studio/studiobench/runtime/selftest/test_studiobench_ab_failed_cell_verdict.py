# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed cell must not yield a verdict over surviving pairs; its healthy partner is dropped too."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import _render_ab  # noqa: E402
from studiobench.runtime.ab import unmeasured_planned_cells  # noqa: E402
from studiobench.runtime.types import Paths, Recorder  # noqa: E402

SIDES = [
    {"label": "base", "ref": "main", "base_url": "http://127.0.0.1:5399", "owns": True},
    {"label": "treatment", "ref": "fix", "base_url": "http://127.0.0.1:5400", "owns": True},
]

LADDER = (("10K", 10_000, 100.0, 200.0), ("100K", 100_000, 100.0, 80.0))


def _cell_row(
    cell_id,
    arm,
    tokens,
    *,
    completed = True,
):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "cell": {"arm": arm, "rep": 0},
        "completed": completed,
        "fidelity": "ok" if completed else "unknown",
        "target_tokens": tokens,
    }


def _keystroke(cell_id, p95):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "action": "keystroke",
        "window": "film",
        "ran": True,
        "expect_ok": True,
        "expect": {},
        "timings": {"p95_ms": p95},
        "slot_missed": False,
    }


def _table(
    tmp_path,
    *,
    failed: str | None,
    planned_known: bool = True,
) -> str:
    """Write the payload one A/B session would write, then render it the way `run()` does."""

    paths = Paths.under(tmp_path / "out")
    rec = Recorder(paths.payload_jsonl, "sess-1")
    planned = []
    for rung, tokens, base_p95, treat_p95 in LADDER:
        for arm, p95 in (("base", base_p95), ("treatment", treat_p95)):
            cell_id = f"r{rung}.{arm}.rep0"
            planned.append(cell_id)
            rec.emit(_cell_row(cell_id, arm, tokens, completed = cell_id != failed))
            rec.emit(_keystroke(cell_id, p95))
    rec.close()

    _render_ab(
        paths,
        SIDES,
        "sess-1",
        "c0ffee",
        planned = planned if planned_known else (),
    )
    return (paths.out / "ab.md").read_text(encoding = "utf-8")


def test_a_failed_base_cell_does_not_publish_a_verdict_from_the_rung_that_survived(tmp_path):
    table = _table(tmp_path, failed = "r10K.base.rep0")

    assert "VOID. No numbers are quotable" in table
    assert "r10K.base.rep0" in table
    assert "IMPROVED" not in table
    assert "20.0% faster" not in table
    assert "headline ratio" not in table


def test_the_same_payload_complete_is_the_fail_the_omission_erased(tmp_path):
    """The control, and the measurement of what was lost: with 10K present this is a FAIL."""

    table = _table(tmp_path, failed = None)

    assert "VERDICT: FAIL" in table
    assert "26.5% worse" in table
    assert "VOID" not in table


def test_the_unguarded_render_is_the_wrong_verdict(tmp_path):
    """Unguarded, the failed cell is never named and its regression vanishes; the guard voids the run."""

    table = _table(tmp_path, failed = "r10K.base.rep0", planned_known = False)

    assert "VERDICT: INCONCLUSIVE" in table
    assert "20.0% faster" not in table
    assert "VOID" not in table
    assert "r10K.base.rep0" not in table


def test_a_gate_failed_cell_is_a_hole_in_the_plan_too(tmp_path):
    """A completed cell that failed thread_complete or follows_the_stream is also a hole in the plan."""

    records = [
        {
            "row_type": "cell",
            "cell_id": "r10K.base.rep0",
            "session_id": "sess-1",
            "completed": True,
        },
        {
            "row_type": "gate",
            "name": "thread_complete",
            "passed": False,
            "cell_id": "r10K.base.rep0",
            "session_id": "sess-1",
            "detail": {"probe_attempted": True, "reason": "the thread lost its middle"},
        },
    ]

    assert unmeasured_planned_cells(records, ["r10K.base.rep0"], session_id = "sess-1") == [
        "r10K.base.rep0"
    ]


def test_a_not_measured_gate_row_is_not_a_hole(tmp_path):
    """The other side of it, and the reason this defers to `failed_invalidating_gates` rather than
    to any failed gate: an absent instrument is not a missing reading of the plan, and voiding on
    it would void every run of a harness that is not loaded."""

    records = [
        {
            "row_type": "cell",
            "cell_id": "r10K.base.rep0",
            "session_id": "sess-1",
            "completed": True,
        },
        {
            "row_type": "gate",
            "name": "follows_the_stream",
            "passed": False,
            "cell_id": "r10K.base.rep0",
            "session_id": "sess-1",
            "detail": {"follow_attempted": False, "reason": "sampler is not installed"},
        },
    ]

    assert unmeasured_planned_cells(records, ["r10K.base.rep0"], session_id = "sess-1") == []


def test_a_gate_failed_cell_voids_the_table_rather_than_publishing_the_survivors(tmp_path):
    """End to end through `_render_ab`, which is where the wrong verdict was published."""

    paths = Paths.under(tmp_path / "out")
    rec = Recorder(paths.payload_jsonl, "sess-1")
    planned = []
    for rung, tokens, base_p95, treat_p95 in LADDER:
        for arm, p95 in (("base", base_p95), ("treatment", treat_p95)):
            cell_id = f"r{rung}.{arm}.rep0"
            planned.append(cell_id)
            rec.emit(_cell_row(cell_id, arm, tokens))
            rec.emit(_keystroke(cell_id, p95))
    rec.gate(
        "thread_complete",
        False,
        {"probe_attempted": True, "reason": "the thread lost its middle"},
        cell_id = "r10K.base.rep0",
    )
    rec.close()

    _render_ab(paths, SIDES, "sess-1", "c0ffee", planned = planned)
    table = (paths.out / "ab.md").read_text(encoding = "utf-8")

    assert "VOID. No numbers are quotable" in table
    assert "r10K.base.rep0" in table


def test_a_complete_plan_is_not_voided(tmp_path):
    records = [_cell_row("r10K.base.rep0", "base", 10_000, completed = True)]
    for row in records:
        row["session_id"] = "sess-1"

    assert unmeasured_planned_cells(records, ["r10K.base.rep0"], session_id = "sess-1") == []


def test_a_cell_from_another_session_does_not_count_as_measured():
    """The session filter `readings_by_arm` applies, applied to the same question. A planned cell
    whose only completed row belongs to a previous session has no reading in THIS one."""

    row = _cell_row("r10K.base.rep0", "base", 10_000, completed = True)
    row["session_id"] = "sess-0"

    assert unmeasured_planned_cells([row], ["r10K.base.rep0"], session_id = "sess-1") == [
        "r10K.base.rep0"
    ]


def test_a_dead_attempt_does_not_mark_a_recovered_cell_unmeasured():
    """A cell id carries every attempt at it. One completed row for the planned cell in this
    session is a reading, and the guard must not void a run that recovered."""

    dead = _cell_row("r10K.base.rep0", "base", 10_000, completed = False)
    dead["session_id"] = "sess-1"
    retry = _cell_row("r10K.base.rep0", "base", 10_000, completed = True)
    retry["session_id"] = "sess-1"

    assert unmeasured_planned_cells([dead, retry], ["r10K.base.rep0"], session_id = "sess-1") == []
