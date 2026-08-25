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


#: Each rung needs its OWN token count: `measures_by_cell` keys on `(rung_tokens, rep)`, so two
#: rungs sharing one number collapse into one pair and a two-rung table renders as a one-rung one.
RUNG_TOKENS = {"1K": 1_000, "10K": 10_000, "100K": 100_000, "500K": 500_000, "1M": 1_000_000}


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
                        rung_tokens = RUNG_TOKENS.get(rung, 10_000),
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
    tokens = 10_000,
):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "cell": {"arm": arm, "rep": rep},
        "completed": True,
        "fidelity": "ok",
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


def test_a_comparison_with_work_left_re_runs_every_pair():
    """A COMPLETE PAIR FROM THE OLD SESSION IS AS UNUSABLE AS A LONE ARM, and this asserted the
    opposite until a resumed standard tier published `VERDICT: IMPROVED (20.0% faster)` off its
    100K pair while the 10K pair it had already measured -- a 30% regression, complete, in session
    one -- was dropped by the session filter with nothing in `ab.md` naming the missing rung.

    The rule the module already argues for a half-finished pair is the rule for a half-finished
    table: re-run both arms of every pair, adjacent in time, in one session.
    """

    work = _work(reps = 2)
    done = {"r10K.base.rep0", "r10K.treatment.rep0", "r10K.base.rep1"}

    assert skippable_cells(work, done) == set()


def test_a_comparison_with_nothing_left_still_skips_everything():
    """The control that keeps `--resume` on a finished A/B free: nothing runs, and `_render_ab`
    keeps the table that run already wrote rather than replacing it with NO READING."""

    work = _work(reps = 2)
    done = {
        "r10K.base.rep0",
        "r10K.treatment.rep0",
        "r10K.base.rep1",
        "r10K.treatment.rep1",
    }

    assert skippable_cells(work, done) == done


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

    # Both arms measured, so the pair exists and the table has a reading. One pair carries no
    # bootstrap CI, so the verdict is INCONCLUSIVE; the contrast with the test below, where the
    # arms never pair at all and the table says NO READING, is the thing under test.
    assert "VERDICT: INCONCLUSIVE" in table
    assert "NO READING" not in table


def test_without_pair_granularity_the_same_resume_reports_nothing(tmp_path):
    """What the fix is for, driven through the same path: skipping the recorded base arm leaves
    the treatment arm with nothing to pair against and the table says NO READING."""

    table = _resumed_table(tmp_path, pair_granular = False)

    assert "VERDICT: NO READING" in table


def _two_rung_resumed_table(tmp_path, *, whole_table: bool) -> str:
    """The same drive over TWO rungs, with the first pair fully recorded in session one.

    10K measured base 100 ms against treatment 130 ms -- a 30% regression, complete, beyond the
    noise floor. The run then died inside the 100K pair, where the treatment is the faster side.
    """

    paths = Paths.under(tmp_path / "out")
    interrupted = Recorder(paths.payload_jsonl, "sess-1")
    for cell_id, arm, rung, p95 in (
        ("r10K.base.rep0", "base", "10K", 100.0),
        ("r10K.treatment.rep0", "treatment", "10K", 130.0),
        ("r100K.base.rep0", "base", "100K", 100.0),
    ):
        interrupted.emit(_cell_row(cell_id, arm, tokens = RUNG_TOKENS[rung]))
        interrupted.emit(_keystroke(cell_id, p95))
    interrupted.close()

    work = _work(rungs = ("10K", "100K"))
    recorded = _resume_set(paths)
    if whole_table:
        done = skippable_cells(work, recorded)
    else:
        # THE PRE-FIX RULE, inline so the contrast is the change itself: skip any pair whose every
        # arm is already recorded, and let the rest of the table be whatever is left.
        by_pair: dict = {}
        for _t, cell, _p in work:
            by_pair.setdefault((cell.rung, cell.rep), []).append(cell.cell_id)
        done = {cid for ids in by_pair.values() if all(c in recorded for c in ids) for cid in ids}

    resumed = Recorder(paths.payload_jsonl, "sess-2")
    for target, cell, _plan in work:
        if cell.cell_id in done:
            continue
        resumed.emit(_cell_row(cell.cell_id, target.label, tokens = RUNG_TOKENS[cell.rung]))
        p95 = 100.0 if target.label == "base" else (130.0 if cell.rung == "10K" else 80.0)
        resumed.emit(_keystroke(cell.cell_id, p95))
    resumed.close()

    _render_ab(paths, SIDES, "sess-2", "c0ffee")
    return (paths.out / "ab.md").read_text(encoding = "utf-8")


def test_a_resumed_comparison_publishes_a_verdict_over_every_rung(tmp_path):
    """THE CONSEQUENCE. Both pairs are re-measured in the new session, so both are in the table
    and the verdict is computed over the ladder rather than over its remainder."""

    table = _two_rung_resumed_table(tmp_path, whole_table = True)

    assert "keystroke_p95_ms         2" in table, table
    assert "0.800-1.300" in table, table
    assert "VERDICT: IMPROVED" not in table, table


def test_skipping_the_recorded_pair_publishes_a_verdict_over_the_remainder(tmp_path):
    """What the fix is for, driven through the same path. The 30% regression the run had already
    measured is dropped by the session filter and nothing in the file mentions the missing rung."""

    table = _two_rung_resumed_table(tmp_path, whole_table = False)

    assert "keystroke_p95_ms         1" in table, table
    # The remainder is a single pair, so it can no longer be published as a direction; the bug
    # this documents is unchanged and is the two assertions around this one: the 30% regression
    # is gone from the table and nothing names the rung it came from.
    assert "VERDICT: INCONCLUSIVE" in table, table
    assert "10K" not in table


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_a_resume_killed_inside_a_cell_does_not_read_as_a_finished_run(tmp_path):
    """THE CONSEQUENCE, through `_resume_set` and `skippable_cells` together.

    Session one failed the 10K base arm. The resume repaired the 10K pair, then was hard-killed
    inside the 100K base arm: its action and window rows are on disk, fsynced, but the terminal
    cell row that `CellRunner.run` writes in a `finally` never was. Keyed on cell rows alone, the
    older completed 100K attempt stayed the latest, every cell read as done across two sessions,
    and the next `--resume` ran nothing and exited 0 over a stale table.
    """

    paths = Paths.under(tmp_path / "out")
    first = Recorder(paths.payload_jsonl, "sess-1")
    first.emit({**_cell_row("r10K.base.rep0", "base"), "completed": False})
    first.emit(_keystroke("r10K.base.rep0", 40.0))
    first.emit(_cell_row("r10K.treatment.rep0", "treatment"))
    first.emit(_keystroke("r10K.treatment.rep0", 41.0))
    for arm in ("base", "treatment"):
        cid = f"r100K.{arm}.rep0"
        first.emit(_cell_row(cid, arm, tokens = RUNG_TOKENS["100K"]))
        first.emit(_keystroke(cid, 50.0))
    first.close()

    killed = Recorder(paths.payload_jsonl, "sess-2")
    for arm in ("base", "treatment"):
        cid = f"r10K.{arm}.rep0"
        killed.emit(_cell_row(cid, arm))
        killed.emit(_keystroke(cid, 42.0))
    killed.emit(_keystroke("r100K.base.rep0", 900.0))  # killed here: no cell row follows
    killed.close()

    done = _resume_set(paths)
    assert "r100K.base.rep0" not in done

    work = _work(rungs = ("10K", "100K"))
    assert skippable_cells(work, done) == set()
