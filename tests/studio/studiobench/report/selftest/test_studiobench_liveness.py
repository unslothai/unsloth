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

from studiobench.__main__ import main                                           # noqa: E402


def write_payload(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "payload.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return path


def cell(actions: list[dict], *, completed: bool = True, cell_id: str = "100K/rep0") -> dict:
    return {"row_type": "cell", "cell_id": cell_id, "completed": completed, "actions": actions}


def ran(name: str, **extra) -> dict:
    return {"action": name, "ran": True, **extra}


def test_a_payload_where_everything_ran_passes(tmp_path):
    path = write_payload(tmp_path, [cell([ran("keystroke"), ran("message_menu")])])
    assert main(["--assert-liveness", str(path)]) == 0


def test_an_action_that_did_not_run_fails(tmp_path):
    path = write_payload(tmp_path, [
        cell([ran("keystroke"), {"action": "message_menu", "ran": False, "reason": "slot closed"}])])
    assert main(["--assert-liveness", str(path)]) == 1


def test_a_missed_slot_fails_even_though_the_action_ran(tmp_path):
    # `ran` and `slot_missed` are different failures: the second means the film moved on without
    # it, so its timing describes a different session than every other arm's.
    path = write_payload(tmp_path, [cell([ran("settings", slot_missed = True)])])
    assert main(["--assert-liveness", str(path)]) == 1


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
    rows = [cell([{"action": "image_upload", "ran": False},
                  {"action": "message_menu", "ran": False}])]
    path = write_payload(tmp_path, rows)
    assert main(["--assert-liveness", str(path), "--allow-not-run", "image_upload"]) == 1
    assert main(["--assert-liveness", str(path),
                 "--allow-not-run", "image_upload,message_menu"]) == 0


def test_non_cell_rows_are_ignored_but_do_not_count_as_cells(tmp_path):
    path = write_payload(tmp_path, [
        {"row_type": "window", "name": "stream:drain"},
        {"row_type": "action", "action": "keystroke", "ran": False},
        cell([ran("keystroke")]),
    ])
    # The stray top-level action row must not fail the run: actions are read from inside their
    # cell, so counting them twice would make the gate depend on payload layout.
    assert main(["--assert-liveness", str(path)]) == 0


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_lines_are_skipped(tmp_path, blank):
    path = tmp_path / "payload.jsonl"
    path.write_text(json.dumps(cell([ran("keystroke")])) + "\n" + blank + "\n", encoding = "utf-8")
    assert main(["--assert-liveness", str(path)]) == 0
