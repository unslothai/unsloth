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
