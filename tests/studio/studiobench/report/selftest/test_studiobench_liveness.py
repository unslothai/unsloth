# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""--assert-liveness must fail when an action never fired; a gate that cannot fail is worse than none."""

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
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path)]) == 1


def test_slack_excuses_a_missed_slot_and_only_up_to_the_number_given(tmp_path):
    # A missed slot is a machine fact; failing on it would make the gate a runner speed test.
    one = write_payload(tmp_path / "a", [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(one), "--allow-slot-misses", "1"]) == 0
    two = write_payload(
        tmp_path / "b",
        [cell([ran("settings", slot_missed = True), ran("keystroke", slot_missed = True)])],
    )
    assert main(["--assert-liveness", str(two), "--allow-slot-misses", "1"]) == 1


def test_slack_never_excuses_a_scene_problem(tmp_path):
    path = write_payload(
        tmp_path,
        [cell([{"action": "message_menu", "ran": False, "reason": "no More button in the DOM"}])],
    )
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "99"]) == 1


def test_an_action_that_missed_its_slot_is_not_also_counted_as_a_scene_problem(tmp_path):
    # Missed slots are ran=False with slot_missed=True, so the slot must be checked before ran.
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
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "1"]) == 0
    assert "Do not quote a number from this payload" in capsys.readouterr().out


def test_negative_slack_is_treated_as_none(tmp_path):
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "-5"]) == 1


def test_a_not_run_allowance_does_not_swallow_a_missed_slot(tmp_path):
    # Slots are classified before --allow-not-run applies, so the allowance never hides a missed slot.
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
    """An action that ran but has expect_ok False must fail the liveness gate."""

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
    path = write_payload(tmp_path, [cell([ran("keystroke", expect_ok = True)])])
    assert main(["--assert-liveness", str(path)]) == 0


def test_an_unattempted_action_reports_not_run_rather_than_its_assertion(tmp_path):
    path = write_payload(
        tmp_path,
        [cell([{"action": "message_menu", "ran": False, "expect_ok": None, "reason": "no slot"}])],
    )
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_failed_assertion_is_a_scene_problem_that_slack_cannot_excuse(tmp_path):
    path = write_payload(
        tmp_path,
        [cell([ran("message_menu", expect_ok = False, reason = "the menu opened with no items")])],
    )
    assert main(["--assert-liveness", str(path), "--allow-slot-misses", "99"]) == 1


def test_an_incomplete_cell_fails(tmp_path):
    path = write_payload(tmp_path, [cell([ran("keystroke")], completed = False)])
    assert main(["--assert-liveness", str(path)]) == 1


def test_an_empty_payload_is_refused_rather_than_passed(tmp_path):
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
    assert main(["--assert-liveness", str(path)]) == 0


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_lines_are_skipped(tmp_path, blank):
    path = tmp_path / "payload.jsonl"
    path.write_text(json.dumps(cell([ran("keystroke")])) + "\n" + blank + "\n", encoding = "utf-8")
    assert main(["--assert-liveness", str(path)]) == 0


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
    """A resumed retry supersedes the dead attempt at the same cell_id, and the gate must ignore it."""

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
    """The superseded attempt is not a second cell; counting it would report two cells where one ran."""

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
    assert any("1 cell(s)" in line and "0 scene problem(s)" in line for line in logged), logged


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
    """Only the same cell id is superseded; cells an earlier session completed stay checked."""

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
    """A cell killed before its cell row is written is not a pass; it must not vanish from the gate."""

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
    """A cell killed before writing any attempt row must fail the gate, not vanish from it."""

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
    """--allow-not-run excuses only NOT RUNNING, so a listed action that ran and failed still fails."""

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
    """--allow-not-run excuses an action the platform cannot perform, not one whose slot was missed."""

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
