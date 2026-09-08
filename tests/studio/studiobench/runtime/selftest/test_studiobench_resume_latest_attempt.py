# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`--resume` decides on the LATEST attempt at a cell, the same one everything else reads.

`latest_attempt_rows` is the rule for the score (`report.build.score_payload`), the ratio
(`ab.readings_by_arm`), the surface parity sweep and `--assert-liveness`. `_resume_set` read the
payload raw, and it is the one reader whose disagreement skips work rather than adding it.

HOW A SUPERSEDED SUCCESS GETS INTO A PAYLOAD WITHOUT ANYBODY DOING ANYTHING UNUSUAL. An A/B pair
is re-run WHOLE (`ab.skippable_cells`), because an arm measured alone in a new session is a
reading no table can contain. So a resume re-runs an arm that had ALREADY succeeded. If that
retry fails while its partner succeeds -- one flaky cell, which is the reason `--resume` exists --
the payload holds a completed row and a later failed row under the same deterministic `cell_id`.

Read raw, the next `--resume` found the old success, skipped the whole pair, ran nothing and
exited 0. Read through `latest_attempt_rows`, the same payload scores INCOMPLETE and
`--assert-liveness` fails on it. A resume that can never re-run the cell that is broken is a gate
nobody can satisfy by fixing the run, and an exit code of 0 over a rung that scored zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import _resume_set  # noqa: E402
from studiobench.report.build import score_payload  # noqa: E402
from studiobench.runtime.ab import Target, interleave, skippable_cells  # noqa: E402
from studiobench.runtime.types import Cell, Paths, Recorder, make_cell_id  # noqa: E402

FIRST = "sess-1"
RETRY = "sess-2"

BASE = "r10K.base.rep0"
TREATMENT = "r10K.treatment.rep0"


def _cell_row(cell_id, arm, *, completed):
    row = {
        "row_type": "cell",
        "cell_id": cell_id,
        "cell": {"arm": arm, "rep": 0},
        "completed": completed,
        "fidelity": "streamed_only",
        "target_tokens": 10_000,
        "actions": [
            {"action": "keystroke", "ran": completed, "reason": None if completed else "died"}
        ],
    }
    if not completed:
        row["failure"] = {"kind": "TimeoutError", "message": "the machine stalled"}
    return row


def _keystroke(cell_id, p95):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "action": "keystroke",
        "ran": True,
        "expect_ok": True,
        "expect": {},
        "timings": {"p95_ms": p95},
        "slot_missed": False,
    }


def _work():
    """The real `interleave` over real cells: exactly the list `run()` iterates."""

    cell = Cell(cell_id = make_cell_id("10K", "A0", 0), rung = "10K", rung_tokens = 10_000, rep = 0)
    targets = [
        Target(
            label = label,
            ref = label,
            base_url = f"http://x/{label}",
            seeder = None,
            runner = None,
        )
        for label in ("base", "treatment")
    ]
    return interleave([(cell, None)], targets)


def _paths(tmp_path, sessions):
    """Write a payload one session at a time, through the real append-mode `Recorder`."""

    paths = Paths.under(tmp_path / "out")
    for session_id, rows in sessions:
        rec = Recorder(paths.payload_jsonl, session_id)
        for row in rows:
            rec.emit(dict(row))
        rec.close()
    return paths


# ── the decision ─────────────────────────────────────────────────────────────────────────────


def test_a_cell_whose_latest_attempt_failed_is_not_skipped(tmp_path):
    """The interrupted A/B: the base arm succeeded, its rerun died, the treatment arm is fine."""

    paths = _paths(
        tmp_path,
        [
            (FIRST, [_cell_row(BASE, "base", completed = True), _keystroke(BASE, 100.0)]),
            (
                RETRY,
                [
                    _cell_row(BASE, "base", completed = False),
                    _cell_row(TREATMENT, "treatment", completed = True),
                    _keystroke(TREATMENT, 50.0),
                ],
            ),
        ],
    )

    assert _resume_set(paths) == {TREATMENT}


def test_the_pair_is_therefore_re_run_and_the_run_is_not_a_no_op(tmp_path):
    """The consequence, through the same pair rule `run()` applies and the score `--report` gives.

    The rung scores INCOMPLETE off the failed retry, so a `--resume` that finds nothing to do
    would exit 0 over a ladder that scored zero.
    """

    paths = _paths(
        tmp_path,
        [
            (FIRST, [_cell_row(BASE, "base", completed = True), _keystroke(BASE, 100.0)]),
            (
                RETRY,
                [
                    _cell_row(BASE, "base", completed = False),
                    _cell_row(TREATMENT, "treatment", completed = True),
                    _keystroke(TREATMENT, 50.0),
                ],
            ),
        ],
    )

    assert score_payload(paths.payload_jsonl, [10_000]).rungs[0].complete is False
    assert skippable_cells(_work(), _resume_set(paths)) == set()


# ── the controls ─────────────────────────────────────────────────────────────────────────────


def test_a_cell_completed_once_and_never_re_run_is_skipped(tmp_path):
    """The control that passes either way: an ordinary resume still skips what it recorded."""

    paths = _paths(
        tmp_path,
        [
            (
                FIRST,
                [
                    _cell_row(BASE, "base", completed = True),
                    _keystroke(BASE, 100.0),
                    _cell_row(TREATMENT, "treatment", completed = True),
                    _keystroke(TREATMENT, 50.0),
                ],
            )
        ],
    )

    done = _resume_set(paths)
    assert done == {BASE, TREATMENT}
    assert skippable_cells(_work(), done) == done


def test_a_cell_that_died_and_was_then_re_run_successfully_is_skipped(tmp_path):
    """The control on the direction: superseding must not turn into refusing to skip anything.

    This is the case `--resume` was built for, and the LATEST attempt is the successful one.
    """

    paths = _paths(
        tmp_path,
        [
            (FIRST, [_cell_row(BASE, "base", completed = False)]),
            (
                RETRY,
                [
                    _cell_row(BASE, "base", completed = True),
                    _keystroke(BASE, 100.0),
                    _cell_row(TREATMENT, "treatment", completed = True),
                    _keystroke(TREATMENT, 50.0),
                ],
            ),
        ],
    )

    assert _resume_set(paths) == {BASE, TREATMENT}


def test_a_payload_without_session_ids_still_resumes(tmp_path):
    """The other control. A payload written before the recorder stamped sessions cannot be split
    into attempts, and `latest_attempt_rows` keeps it whole rather than losing the run."""

    paths = Paths.under(tmp_path / "out")
    paths.payload_jsonl.parent.mkdir(parents = True, exist_ok = True)
    paths.payload_jsonl.write_text(
        '{"row_type": "cell", "cell_id": "r10K.base.rep0", "completed": true}\n',
        encoding = "utf-8",
    )

    assert _resume_set(paths) == {BASE}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
