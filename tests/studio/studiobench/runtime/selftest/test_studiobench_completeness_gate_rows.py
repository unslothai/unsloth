# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""WHICH CELL LOST THE MESSAGES, read back out of the payload the session actually writes.

The completeness probe is the only check in the harness that can find a windowed arm dropping
history, and its verdict used to be written with `Recorder.gate`, which emits `{row_type, name,
passed, detail}` and no cell id. `report/payload.py::excluded_from_rows` reads a failed gate as
`row.get("cell_id") or "run"`, so the finding arrived in the report as a run-level self-check
failure: something failed somewhere, and no way to say which arm or which rung. For a probe whose
whole purpose is to attribute data loss to an arm, that is the finding being deleted on the way
out.

So the row is written here through `record_completeness_gate` and read back through the real
`excluded_from_rows`, in that order, rather than asserted against a hand-built dict. No browser:
this is about what reaches the payload, not about what the page did.

AND WHICH `None` THE COVERAGE WAS. `ordinal_coverage` answers `None` for two opposite reasons --
no ordinals were published at all, so the question does not arise, versus the sweep never having
inspected the middle of the thread -- and a gate that passes both lets the first-page-and-last-page
store back in through the unknown state. The pair of positive controls below is what stops the
repair turning into a gate that fails the shipped build instead.

    python -m pytest tests/studio/studiobench/runtime/selftest/test_studiobench_completeness_gate_rows.py -q
"""

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

#: The arm the whole windowed readiness gate exists for, at a rung big enough to be windowed.
CELL = Cell(
    cell_id = make_cell_id("100K", "B1", 0),
    rung = "100K",
    rung_tokens = 100_000,
    arm = "B1",
    session_id = "sess0",
)

#: What the probe returns for a thread that kept its head and its tail and lost the middle: the
#: marker check is satisfied and the ordinals are not.
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
    # `r{rung}.{arm}.rep{rep}`, so the one field answers which arm and which rung lost them.
    assert row["cell_id"] == "r100K.B1.rep0"
    assert row["detail"]["ordinals_missing"] == list(range(4, 16))


def test_a_cell_that_lost_messages_is_excluded_as_itself_and_not_as_the_run(tmp_path):
    """The consuming end, unmodified: `excluded_from_rows` is what the report reads.

    Its fallback to the synthetic cell id "run" is deliberate and stays -- a genuinely run-level
    gate has no cell -- so the fix is on the writing side, and this is the assertion that the two
    ends agree.
    """
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
    """THE COMPLETENESS FIX, RE-ENTERING THROUGH THE UNKNOWN STATE.

    A store that kept the first page and the last one and lost everything between them mounts the
    head when the traversal scrolls to it, so the marker check is satisfied. The ordinals are what
    catch it -- and when the sweep's consecutive stops did not overlap, the ordinals were never
    inspected. Reading that `None` as a pass hands the cell straight back to the arm this probe
    exists to refuse, and the report then carries its frame rate as if it had been earned on a
    complete thread.
    """
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


# ── the two ends, joined: the probe's own output through the gate ───


def _coverage(**traverse) -> dict:
    """What `probe_thread_completeness` would hand the gate, from a real traversal record."""
    got = {"probe_attempted": True, "head_reached": True}
    got.update(ordinal_coverage(traverse, 18))
    return got


def test_a_coarse_sweep_over_a_thread_missing_its_middle_does_not_pass_the_gate(tmp_path):
    """END TO END, with no hand-written state string between the two halves.

    `ordinal_coverage` and `record_completeness_gate` have to agree about which `None` this is, so
    the traversal record goes through the real function and the real gate. Six of eighteen
    ordinals ever mounted, the head among them, and a gesture too coarse to say whether the other
    twelve exist: not a finding, and not a pass.
    """
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
    """A probe that could not scroll the viewport returns neither verdict.

    It reports `probe_attempted: false` and nothing else, and the gate must not read a missing
    `head_reached` as a thread that was fine. This is the reading the harness has always taken and
    it is asserted here so the three-valued coverage rule above cannot quietly widen it.
    """
    rows, passed = _rows(
        tmp_path,
        {"probe_attempted": False, "reason": "the viewport could not be scrolled"},
    )
    assert passed is False and rows[0]["passed"] is False
    assert [c["cell_id"] for c in excluded_from_rows(rows)] == [CELL.cell_id]


# ── every per-cell gate, not just the completeness one ──────────────


def test_every_per_cell_gate_names_its_cell():
    """THE SAME DEFECT, IN FOUR MORE PLACES. `excluded_from_rows` reads
    `row.get("cell_id") or "run"`, so a per-cell gate emitted without one is attributed to the
    synthetic cell "run": a failure that says one arm at one rung lost the thread, or fell behind
    the stream, or had its timer clamped, is presented as a run-level self-check failure and the
    report cannot say which arm or which rung.

    Asserted against the source rather than a live session, because reaching these lines needs a
    browser, a backend and a seeded thread, and the property under test is simply that the call
    passes the identity it already has in scope.
    """
    import inspect

    from studiobench.runtime import session as S

    src = inspect.getsource(S)

    def _calls(text: str) -> list:
        # A PAREN COUNTER, not a regex. `rec.gate("follows_the_stream", bool(...), ...)` contains a
        # nested call, and a non-greedy regex stops at the inner closing paren -- reporting the
        # outer call as missing an argument that is two lines further down.
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
                # Skip prose. `session.py` carries a comment reading "WHY THIS IS NOT
                # `recorder.gate(...)`", and a scanner that counts it reports a defect in a
                # sentence.
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
