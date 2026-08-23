# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""`--assert-liveness` is the gate that catches an action which never fired reading as "no effect".

A gate that cannot fail is worse than no gate, because it is quoted as evidence. So every test here
that asserts a pass is paired with one that asserts the corresponding failure, and the empty-payload
case is tested explicitly: a check that passes over zero rows is the same false negative wearing a
different hat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import main  # noqa: E402


def write_payload(tmp_path: Path, rows: list[dict]) -> Path:
    tmp_path.mkdir(parents = True, exist_ok = True)
    path = tmp_path / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def cell(
    actions: list[dict],
    *,
    completed: bool = True,
    cell_id: str = "100K/rep0",
) -> dict:
    return {"row_type": "cell", "cell_id": cell_id, "completed": completed, "actions": actions}


def ran(name: str, **extra) -> dict:
    return {"action": name, "ran": True, **extra}


def test_a_payload_where_everything_ran_passes(tmp_path):
    path = write_payload(tmp_path, [cell([ran("keystroke"), ran("message_menu")])])
    assert main(["--assert-liveness", str(path)]) == 0


def test_an_action_that_did_not_run_fails(tmp_path):
    path = write_payload(
        tmp_path,
        [
            cell(
                [
                    ran("keystroke"),
                    {"action": "message_menu", "ran": False, "reason": "slot closed"},
                ]
            )
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_missed_slot_fails_even_though_the_action_ran(tmp_path):
    # `ran` and `slot_missed` are different failures: the second means the film moved on without
    # it, so its timing describes a different session than every other arm's. Default slack is 0,
    # which is the right default for the quiet machine a measurement is taken on.
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path)]) == 1


def test_slack_excuses_a_missed_slot_and_only_up_to_the_number_given(tmp_path):
    # A missed slot is a fact about the MACHINE. The scene is a fixed-duration film on the wall
    # clock and is designed to roll on through one, precisely so a slow machine does not take a
    # different path through a different-length session. On a two-core shared runner, failing on
    # that makes the gate a speed test of the runner rather than a check that the harness works.
    one = write_payload(tmp_path / "a", [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(one), "--allow-slot-misses", "1"]) == 0
    two = write_payload(
        tmp_path / "b",
        [cell([ran("settings", slot_missed = True), ran("keystroke", slot_missed = True)])],
    )
    assert main(["--assert-liveness", str(two), "--allow-slot-misses", "1"]) == 1


def test_slack_never_excuses_a_scene_problem(tmp_path):
    # The distinction the whole split rests on. An action that was never planned, or whose button
    # was not there, is the harness lying, and no amount of machine slack makes that acceptable.
    path = write_payload(
        tmp_path,
        [cell([{"action": "message_menu", "ran": False, "reason": "no More button in the DOM"}])],
    )
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "99"]) == 1


def test_an_action_that_missed_its_slot_is_not_also_counted_as_a_scene_problem(tmp_path):
    # A missed slot is recorded as `ran: False` WITH `slot_missed: True`, so an implementation
    # that checks `ran` first would file every missed slot under the category that slack cannot
    # excuse, and the flag would do nothing at all.
    path = write_payload(
        tmp_path,
        [
            cell(
                [
                    {
                        "action": "message_menu",
                        "ran": False,
                        "slot_missed": True,
                        "reason": "the slot opened at 32000ms and this machine reached it at "
                        "32901ms, past its 800ms budget",
                    }
                ]
            )
        ],
    )
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "1"]) == 0
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_tolerated_miss_still_says_the_run_is_not_quotable(tmp_path, capsys):
    # Exit 0 here claims only that the harness was not the cause. The payload still has a hole in
    # it, and saying so is the difference between tolerating a miss and hiding one.
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "1"]) == 0
    assert "Do not quote a number from this payload" in capsys.readouterr().out


def test_negative_slack_is_treated_as_none(tmp_path):
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "-5"]) == 1


def test_a_not_run_allowance_does_not_swallow_a_missed_slot(tmp_path):
    # Where the two rules above meet. `--allow-not-run` excuses an action the fixture cannot mount
    # at all, and a missed slot is a fact about the machine, so the slot is classified FIRST and
    # the allowance never reaches it. Checked the other way round, a listed name would vanish from
    # both buckets: not a scene problem, and not a missed slot either, so `--allow-slot-misses`
    # would silently stop counting exactly the actions most likely to overrun.
    path = write_payload(
        tmp_path,
        [cell([{"action": "image_upload", "ran": False, "slot_missed": True, "reason": "late"}])],
    )
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1
    assert (
        main(
            [
                "--assert-liveness",
                str(path),
                "--allow-not-run",
                "image_upload",
                "--allow-slot-misses",
                "1",
            ]
        )
        == 0
    )


def test_an_action_whose_own_assertion_failed_fails(tmp_path):
    """RAN IS NOT DID WHAT IT CLAIMED, and this gate is the machine that checks it.

    `scoring/from_payload.py` refuses to score a timing whose `expect_ok` is False and
    `report/payload.py` lists the cell under EXCLUDED CELLS saying its timings "must not be
    quoted". Read raw, this gate agreed with neither: `ran = True, expect_ok = False` reached
    neither the NOT RUN branch nor the missed-slot branch, so the CI liveness job exited 0 and
    the report that dropped every excluded number exited 0 as well. A selector regression that
    fails EVERY cell's assertion is precisely the "absent read as no effect" failure this gate
    exists for.
    """

    path = write_payload(
        tmp_path,
        [
            cell(
                [
                    ran("keystroke"),
                    ran(
                        "message_menu",
                        expect_ok = False,
                        reason = "the menu opened with no items",
                        timings = {"open_ms": 31.0},
                    ),
                ]
            )
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_an_action_that_passed_its_assertion_still_passes(tmp_path):
    # The pair for the test above: `expect_ok = True` is the healthy recording, and a gate that
    # failed on it would be unusable rather than strict.
    path = write_payload(tmp_path, [cell([ran("keystroke", expect_ok = True)])])
    assert main(["--assert-liveness", str(path)]) == 0


def test_an_unattempted_action_reports_not_run_rather_than_its_assertion(tmp_path):
    # `ActionResult.__post_init__` forces `expect_ok = None` when `ran` is False, so the two
    # failures never blur: the reader is told the action never happened, not that it misbehaved.
    path = write_payload(
        tmp_path,
        [cell([{"action": "message_menu", "ran": False, "expect_ok": None, "reason": "no slot"}])],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_failed_assertion_is_a_scene_problem_that_slack_cannot_excuse(tmp_path):
    # The third bucket meeting the first. An assertion that failed says the surface is broken, not
    # that the machine was slow, so it belongs with the scene problems and no amount of
    # `--allow-slot-misses` may buy it a pass.
    path = write_payload(
        tmp_path,
        [cell([ran("message_menu", expect_ok = False, reason = "the menu opened with no items")])],
    )
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "99"]) == 1


def test_an_incomplete_cell_fails(tmp_path):
    path = write_payload(tmp_path, [cell([ran("keystroke")], completed = False)])
    assert main(["--assert-liveness", str(path)]) == 1


def test_an_empty_payload_is_refused_rather_than_passed(tmp_path):
    # The whole point. Zero rows satisfies "no action failed" vacuously.
    path = write_payload(tmp_path, [{"row_type": "header", "tool": "studiobench"}])
    assert main(["--assert-liveness", str(path)]) == 2


def test_a_missing_payload_is_refused(tmp_path):
    assert main(["--assert-liveness", str(tmp_path / "nope.jsonl")]) == 2


def test_allow_not_run_excuses_only_the_names_given(tmp_path):
    rows = [
        cell([{"action": "image_upload", "ran": False}, {"action": "message_menu", "ran": False}])
    ]
    path = write_payload(tmp_path, rows)
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1
    assert (
        main(["--assert-liveness", str(path), "--allow-not-run", "image_upload,message_menu"]) == 0
    )


def test_non_cell_rows_are_ignored_but_do_not_count_as_cells(tmp_path):
    path = write_payload(
        tmp_path,
        [
            {"row_type": "window", "name": "stream:drain"},
            {"row_type": "action", "action": "keystroke", "ran": False},
            cell([ran("keystroke")]),
        ],
    )
    # The stray top-level action row must not fail the run: actions are read from inside their
    # cell, so counting them twice would make the gate depend on payload layout.
    assert main(["--assert-liveness", str(path)]) == 0


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_lines_are_skipped(tmp_path, blank):
    path = tmp_path / "payload.jsonl"
    path.write_text(json.dumps(cell([ran("keystroke")])) + "\n" + blank + "\n", encoding = "utf-8")
    assert main(["--assert-liveness", str(path)]) == 0


# ── a cell that was re-run is judged on the run that finished it ─────


NOW = "s-now"
OLD = "s-before"


def attempt(
    session: str,
    actions: list[dict],
    *,
    completed: bool,
    cell_id: str = "100K/rep0",
):
    row = cell(actions, completed = completed, cell_id = cell_id)
    row["session_id"] = session
    return row


def test_a_resumed_cell_is_not_failed_by_the_attempt_that_died(tmp_path):
    """`--resume` appends, so both attempts at one `cell_id` are in the file forever.

    The resumed run exits 0 and `--report` scores the retry. Read raw, this gate found the dead
    attempt's `completed: false` and NOT RUN actions on every later invocation, so a payload that
    had already been repaired could never pass again.
    """

    path = write_payload(
        tmp_path,
        [
            attempt(
                OLD, [{"action": "message_menu", "ran": False, "reason": "died"}], completed = False
            ),
            attempt(NOW, [ran("keystroke"), ran("message_menu")], completed = True),
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 0


def test_the_superseded_attempt_does_not_count_as_a_second_cell(tmp_path):
    """Not merely "does not fail": the dead attempt is not a cell this payload contains.

    Counting it would make `--assert-liveness` report two cells where one ran, and the count is
    the only thing standing between this gate and an empty payload passing vacuously.
    """

    rows = [
        attempt(OLD, [{"action": "message_menu", "ran": False}], completed = False),
        attempt(NOW, [ran("keystroke")], completed = True),
    ]
    path = write_payload(tmp_path, rows)
    logged: list[str] = []
    import studiobench.__main__ as m

    real = m._log
    m._log = lambda msg = "": (logged.append(str(msg)), real(msg))[1]
    try:
        assert main(["--assert-liveness", str(path)]) == 0
    finally:
        m._log = real
    # The summary names scene problems and missed slots apart on this branch, so the count is
    # asserted on its own rather than against one spelling of the rest of the line.
    assert any("1 cell(s)" in line and "0 scene problem(s)" in line for line in logged), logged


# ── the controls: what must still fail ──────────────────────────────


def test_a_cell_that_was_never_re_run_still_fails(tmp_path):
    """A crash with no retry behind it is still a crash."""

    path = write_payload(
        tmp_path,
        [attempt(OLD, [{"action": "message_menu", "ran": False}], completed = False)],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_the_latest_attempt_is_judged_on_its_own_failures(tmp_path):
    """Superseding must not become excusing: a retry that itself died is the answer."""

    path = write_payload(
        tmp_path,
        [
            attempt(OLD, [ran("keystroke"), ran("message_menu")], completed = True),
            attempt(
                NOW,
                [{"action": "message_menu", "ran": False, "reason": "died again"}],
                completed = False,
            ),
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_different_cell_in_an_earlier_session_is_not_superseded(tmp_path):
    """Only a later attempt at the SAME cell id supersedes an earlier one.

    A resumed run re-runs only the cells that died; every cell the first session completed stays
    in the payload under its own id and must still be checked.
    """

    path = write_payload(
        tmp_path,
        [
            attempt(
                OLD, [{"action": "message_menu", "ran": False}], completed = True, cell_id = "10K/rep0"
            ),
            attempt(NOW, [ran("keystroke")], completed = True, cell_id = "100K/rep0"),
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_an_attempt_killed_before_its_cell_row_is_not_a_pass(tmp_path):
    """A SIGKILL inside a cell must not delete that cell from the gate's answer.

    `latest_attempt_rows` names the latest attempt from ANY attempt-keyed row, because the
    Recorder flushes and fsyncs action and window rows while the terminal `cell` row is written in
    a `finally` that a SIGKILL never reaches. The killed attempt therefore supersedes the recorded
    one and contributes no cell row itself, so reading cell rows alone dropped the cell entirely:
    a resume killed inside a cell whose first attempt had already recorded `completed: false` with
    a NOT RUN action turned exit 1 into exit 0 while both facts were still in the file.
    """

    path = write_payload(
        tmp_path,
        [
            attempt(
                OLD,
                [
                    ran("keystroke"),
                    {"action": "message_menu", "ran": False, "reason": "slot closed"},
                ],
                completed = False,
                cell_id = "10K/rep0",
            ),
            attempt(OLD, [ran("keystroke")], completed = True, cell_id = "1K/rep0"),
            # All the retry managed to flush before it was killed.
            {
                "row_type": "action",
                "cell_id": "10K/rep0",
                "session_id": NOW,
                "action": "keystroke",
                "ran": True,
            },
            {"row_type": "window", "cell_id": "10K/rep0", "session_id": NOW, "name": "stream"},
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_run_killed_during_a_later_cell_does_not_pass_on_its_earlier_ones(tmp_path):
    """The same hole without a resume: the killed cell has no attempt behind it at all.

    Earlier cells stay complete and live, so a gate that only reads cell rows reports them and
    exits 0 over a run that stopped in the middle.
    """

    path = write_payload(
        tmp_path,
        [
            attempt(NOW, [ran("keystroke")], completed = True, cell_id = "1K/rep0"),
            {
                "row_type": "action",
                "cell_id": "10K/rep0",
                "session_id": NOW,
                "action": "keystroke",
                "ran": True,
            },
        ],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_an_allowed_action_that_ran_and_failed_its_assertion_still_fails(tmp_path):
    """`--allow-not-run` excuses NOT RUNNING, not everything the gate checks.

    The allow-list skip used to sit above all three branches, so a listed name was exempt from
    the gate entirely. `image_upload` is listed in studiobench-ci.yml only because the fixture
    cannot mount an upload; the day it does mount, an upload that produces no attachment has to
    be a failure rather than an excuse inherited from a different reason.
    """

    path = write_payload(
        tmp_path,
        [cell([ran("image_upload", expect_ok = False, reason = "no attachment appeared")])],
    )
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1


def test_an_allowed_action_that_did_not_run_is_still_excused(tmp_path):
    """The pair: scoping the allowance must not break what it is actually for."""

    path = write_payload(
        tmp_path,
        [
            cell(
                [
                    {
                        "action": "image_upload",
                        "ran": False,
                        "reason": "fixture cannot mount an upload",
                    }
                ]
            )
        ],
    )
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 0


def test_an_allowed_action_that_ran_and_missed_its_slot_still_fails(tmp_path):
    """A listed name that ran is still held to its slot, for the same reason."""

    path = write_payload(tmp_path, [cell([ran("image_upload", slot_missed = True)])])
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1


def test_a_missed_slot_is_not_excused_by_a_not_run_allowance(tmp_path):
    """`--allow-not-run` excuses an action the platform cannot perform, not one it was late for.

    `scene/schedule.py` records an overrun as `ran = False, slot_missed = True`, so checking the
    allowance first let a listed name inherit an excuse written for a different failure. The
    action could have been performed; the machine was too slow to reach the window. On
    `image_upload`, the only allowance this repo ships, that is the difference between a timed
    benchmark and an untimed one, and the help text already promises a listed action is still
    held to its slot.
    """

    path = write_payload(
        tmp_path,
        [
            cell(
                [
                    {
                        "action": "image_upload",
                        "ran": False,
                        "slot_missed": True,
                        "reason": "the slot opened at 1000ms and this machine reached it at 2000ms",
                    }
                ]
            )
        ],
    )
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1
