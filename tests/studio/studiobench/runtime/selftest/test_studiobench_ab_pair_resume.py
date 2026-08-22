# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An interrupted A/B resumes at PAIR granularity, because a pair is the unit of the comparison.

The two cells of one `(rung, rep)` pair run adjacent in time inside one session, and that adjacency
is the whole method: cross-session drift on the measured machine ran to 8%, larger than most of the
wins anybody argues about, so `readings_by_arm` scopes the ratio to a single session and
`assert_comparable` refuses two session ids outright.

An interruption between those two adjacent cells is the ordinary way a run stops. Skipping the arm
that completed and measuring only its partner in the new session therefore bought a full cell of
measurement that no table can contain: the old arm is dropped by the session filter, the new one
has nothing to pair with, and `_render_ab` writes that repetition out of the table -- at one
repetition, out of the table entirely, as NO READING with an exit code of 0 underneath it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import _render_ab, _resume_set  # noqa: E402
from studiobench.runtime.ab import Target, interleave, skippable_cells  # noqa: E402
from studiobench.runtime.types import Cell, Paths, Recorder, make_cell_id  # noqa: E402

SIDES = [
    {"label": "base", "ref": "main", "base_url": "http://127.0.0.1:5399", "owns": True},
    {"label": "treatment", "ref": "fix", "base_url": "http://127.0.0.1:5400", "owns": True},
]


def _target(label):
    return Target(label = label, ref = label, base_url = f"http://x/{label}", seeder = None, runner = None)


def _work(reps = 1, rungs = ("10K",)):
    """The real `interleave` over real cells: exactly what `run()` iterates."""

    cells = []
    for rung in rungs:
        for rep in range(reps):
            cells.append(
                (
                    Cell(
                        cell_id = make_cell_id(rung, "A0", rep),
                        rung = rung,
                        rung_tokens = 10_000,
                        rep = rep,
                    ),
                    None,
                )
            )
    return interleave(cells, [_target("base"), _target("treatment")])


def _cell_row(
    cell_id,
    arm,
    rep = 0,
):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "cell": {"arm": arm, "rep": rep},
        "completed": True,
        "fidelity": "ok",
        "target_tokens": 10_000,
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


# ── the decision ─────────────────────────────────────────────────────────────────────────────


def test_a_pair_interrupted_between_its_two_arms_is_re_run_whole():
    work = _work()
    done = {"r10K.base.rep0"}  # the run died after the base arm and before the treatment arm

    assert skippable_cells(work, done) == set()


def test_a_pair_recorded_on_both_arms_is_skipped():
    """The control: pair granularity must not turn into refusing to resume anything."""

    work = _work()
    done = {"r10K.base.rep0", "r10K.treatment.rep0"}

    assert skippable_cells(work, done) == done


def test_only_the_whole_pairs_are_skipped():
    work = _work(reps = 2)
    done = {"r10K.base.rep0", "r10K.treatment.rep0", "r10K.base.rep1"}

    assert skippable_cells(work, done) == {"r10K.base.rep0", "r10K.treatment.rep0"}


def test_a_run_without_ab_skips_exactly_what_it_recorded():
    """The other control. Without `--ab` every pair holds one cell, so nothing changes."""

    work = [
        (None, Cell(cell_id = make_cell_id(rung, "A0", 0), rung = rung, rung_tokens = 1), None)
        for rung in ("1K", "10K", "100K")
    ]
    done = {"r1K.A0.rep0", "r100K.A0.rep0"}

    assert skippable_cells(work, done) == done


# ── the consequence, through the table a resumed run actually writes ─────────────────────────


def _resumed_table(tmp_path, *, pair_granular: bool) -> str:
    """Drive a resumed A/B end to end: what the payload holds, what the resume decides to run,
    what it records in the new session, and what `_render_ab` then writes to `ab.md`."""

    paths = Paths.under(tmp_path / "out")
    interrupted = Recorder(paths.payload_jsonl, "sess-1")
    interrupted.emit(_cell_row("r10K.base.rep0", "base"))
    interrupted.emit(_keystroke("r10K.base.rep0", 100.0))
    interrupted.close()  # killed here, between the two arms of the only pair

    work = _work()
    done = _resume_set(paths)
    if pair_granular:
        done = skippable_cells(work, done)

    resumed = Recorder(paths.payload_jsonl, "sess-2")
    for target, cell, _plan in work:
        if cell.cell_id in done:
            continue
        resumed.emit(_cell_row(cell.cell_id, target.label))
        resumed.emit(_keystroke(cell.cell_id, 100.0 if target.label == "base" else 50.0))
    resumed.close()

    _render_ab(paths, SIDES, "sess-2", "c0ffee")
    return (paths.out / "ab.md").read_text(encoding = "utf-8")


def test_the_resumed_session_measures_both_arms_and_gets_a_table(tmp_path):
    table = _resumed_table(tmp_path, pair_granular = True)

    assert "VERDICT: IMPROVED" in table
    assert "NO READING" not in table


def test_without_pair_granularity_the_same_resume_reports_nothing(tmp_path):
    """What the fix is for, driven through the same path: skipping the recorded base arm leaves
    the treatment arm with nothing to pair against and the table says NO READING."""

    table = _resumed_table(tmp_path, pair_granular = False)

    assert "VERDICT: NO READING" in table


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
