# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A completeness gate row must carry its cell_id, or excluded_from_rows files the failure under run."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.report.payload import excluded_from_rows  # noqa: E402
from studiobench.runtime.readiness import (  # noqa: E402
    COVERAGE_COMPLETE,
    COVERAGE_INCOMPLETE,
    COVERAGE_NOT_APPLICABLE,
    COVERAGE_UNMEASURED,
    ordinal_coverage,
)
from studiobench.runtime.session import record_completeness_gate  # noqa: E402
from studiobench.runtime.types import Cell, Recorder, make_cell_id  # noqa: E402

CELL = Cell(
    cell_id = make_cell_id("100K", "B1", 0),
    rung = "100K",
    rung_tokens = 100_000,
    arm = "B1",
    session_id = "sess0",
)

LOST_MIDDLE = {
    "probe_attempted": True,
    "expected_messages": 18,
    "head_reached": True,
    "ordinal_coverage_complete": False,
    "ordinal_coverage_state": COVERAGE_INCOMPLETE,
    "ordinals_seen_count": 6,
    "ordinals_missing": list(range(4, 16)),
    "ordinals_missing_count": 12,
    "coverage_reason": "the arm is missing messages from the MIDDLE of the thread",
}


def _rows(tmp_path, completeness: dict) -> tuple[list[dict], bool]:
    recorder = Recorder(tmp_path / "payload.jsonl", "sess0")
    try:
        passed = record_completeness_gate(recorder, CELL, completeness)
    finally:
        recorder.close()
    return list(recorder.rows()), passed


def test_the_completeness_verdict_names_the_cell_it_was_taken_from(tmp_path):
    rows, passed = _rows(tmp_path, LOST_MIDDLE)
    assert len(rows) == 1
    row = rows[0]
    assert row["row_type"] == "gate"
    assert row["name"] == "thread_complete"
    assert passed is False and row["passed"] is False
    assert row["cell_id"] == "r100K.B1.rep0"
    assert row["detail"]["ordinals_missing"] == list(range(4, 16))


def test_a_cell_that_lost_messages_is_excluded_as_itself_and_not_as_the_run(tmp_path):
    """The fallback to the synthetic cell id run is deliberate, so the fix belongs on the writing side."""
    rows, _ = _rows(tmp_path, LOST_MIDDLE)
    excluded = excluded_from_rows(rows)
    assert len(excluded) == 1
    assert excluded[0]["cell_id"] == CELL.cell_id
    assert excluded[0]["cell_id"] != "run"
    assert excluded[0]["reason"] == "selfcheck_failed"
    assert "thread_complete" in excluded[0]["detail"]


def test_coverage_that_does_not_apply_does_not_fail_the_cell(tmp_path):
    """A fully mounted arm publishes no `aria-posinset` anywhere, so there are no ordinals to
    cover and none missing. The question does not arise, and failing a cell on a question that was
    never asked of it would fail the shipped build's own completeness gate on every cell."""
    rows, passed = _rows(
        tmp_path,
        {
            "probe_attempted": True,
            "head_reached": True,
            "ordinal_coverage_complete": None,
            "ordinal_coverage_state": COVERAGE_NOT_APPLICABLE,
            "coverage_reason": "no mounted row published aria-posinset during the traversal",
        },
    )
    assert passed is True and rows[0]["passed"] is True
    assert excluded_from_rows(rows) == []


def test_coverage_that_applies_but_was_never_measured_is_not_a_pass(tmp_path):
    """An unmeasured coverage state must not pass the gate, or a first-and-last-page store slips back in."""
    rows, passed = _rows(
        tmp_path,
        {
            "probe_attempted": True,
            "head_reached": True,
            "ordinal_coverage_complete": None,
            "ordinal_coverage_state": COVERAGE_UNMEASURED,
            "coverage_reason": "consecutive stops of the gesture did not overlap",
        },
    )
    assert passed is False and rows[0]["passed"] is False
    assert [c["cell_id"] for c in excluded_from_rows(rows)] == [CELL.cell_id]


def test_an_undifferentiated_coverage_None_is_not_a_pass_either(tmp_path):
    """A payload written before the state existed carries the ambiguity and nothing that resolves
    it. That is exactly the reading the gate must not resolve in favour of a pass."""
    rows, passed = _rows(
        tmp_path,
        {"probe_attempted": True, "head_reached": True, "ordinal_coverage_complete": None},
    )
    assert passed is False and rows[0]["passed"] is False


def _coverage(**traverse) -> dict:
    """What `probe_thread_completeness` would hand the gate, from a real traversal record."""
    got = {"probe_attempted": True, "head_reached": True}
    got.update(ordinal_coverage(traverse, 18))
    return got


def test_a_coarse_sweep_over_a_thread_missing_its_middle_does_not_pass_the_gate(tmp_path):
    """ordinal_coverage and the gate must agree on which None this is, so the test runs both unmodified."""
    rows, passed = _rows(
        tmp_path,
        _coverage(
            reached_target = True,
            ordinals_seen = [1, 2, 3, 16, 17, 18],
            ordinals_in_window_holes = [],
            sweep_continuous = False,
            traversal_stops = 2,
        ),
    )
    assert passed is False and rows[0]["passed"] is False
    assert rows[0]["detail"]["ordinal_coverage_state"] == COVERAGE_UNMEASURED
    assert [c["cell_id"] for c in excluded_from_rows(rows)] == [CELL.cell_id]


def test_a_continuous_sweep_that_saw_every_ordinal_passes_the_gate(tmp_path):
    """The positive control for the test above: the same joining, on a sweep that did establish
    coverage. Without it, "not a pass" could be coming from a gate that passes nothing."""
    rows, passed = _rows(
        tmp_path,
        _coverage(
            reached_target = True,
            ordinals_seen = list(range(1, 19)),
            ordinals_in_window_holes = [],
            sweep_continuous = True,
            traversal_stops = 9,
        ),
    )
    assert passed is True and rows[0]["passed"] is True
    assert rows[0]["detail"]["ordinal_coverage_state"] == COVERAGE_COMPLETE


def test_an_arm_that_publishes_no_ordinals_at_all_passes_the_gate(tmp_path):
    """The other positive control, and the reason a blanket "None fails" would be wrong: this is
    the shipped build, traversed to the top, publishing nothing for the sweep to count."""
    rows, passed = _rows(
        tmp_path,
        _coverage(
            reached_target = True,
            ordinals_seen = [],
            ordinals_in_window_holes = [],
            sweep_continuous = True,
            traversal_stops = 3,
        ),
    )
    assert passed is True and rows[0]["passed"] is True
    assert rows[0]["detail"]["ordinal_coverage_state"] == COVERAGE_NOT_APPLICABLE


def test_a_head_that_never_mounted_still_fails_the_cell(tmp_path):
    """The original verdict, unchanged: the head marker not arriving is data loss on its own."""
    rows, passed = _rows(
        tmp_path,
        {"probe_attempted": True, "head_reached": False, "ordinal_coverage_complete": None},
    )
    assert passed is False and rows[0]["passed"] is False
    assert [c["cell_id"] for c in excluded_from_rows(rows)] == [CELL.cell_id]


def test_a_probe_that_never_ran_fails_the_cell_rather_than_passing_it(tmp_path):
    """A probe that never scrolled reports probe_attempted false and must fail the cell, not pass it."""
    rows, passed = _rows(
        tmp_path,
        {"probe_attempted": False, "reason": "the viewport could not be scrolled"},
    )
    assert passed is False and rows[0]["passed"] is False
    assert [c["cell_id"] for c in excluded_from_rows(rows)] == [CELL.cell_id]


def test_every_per_cell_gate_names_its_cell():
    """Per-cell gates emitted without a cell_id are attributed to the synthetic cell run."""
    import inspect

    from studiobench.runtime import session as S

    src = inspect.getsource(S)

    def _calls(text: str) -> list:
        # Paren counter, not regex: the gate call contains nested calls.
        out = []
        for marker in ("rec.gate(", "recorder.gate("):
            start = 0
            while True:
                i = text.find(marker, start)
                if i < 0:
                    break
                j = i + len(marker)
                depth = 1
                while j < len(text) and depth:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                args = text[i + len(marker) : j - 1]
                # Skip prose: session.py has a comment quoting `recorder.gate(...)`.
                line_start = text.rfind("\n", 0, i) + 1
                is_comment = text[line_start:i].lstrip().startswith("#")
                if args.strip() != "..." and not is_comment:
                    out.append(args)
                start = j
        return out

    calls = _calls(src)
    assert calls, "no gate calls found, so this test is asserting nothing"
    per_cell = [c for c in calls if "instrument_unavailable" not in c]
    missing = [c.strip()[:60] for c in per_cell if "cell_id" not in c]
    assert not missing, (
        "these per-cell gates do not name the cell they describe, so a failure in one arm at one "
        f"rung will be reported against the synthetic cell 'run': {missing}"
    )
