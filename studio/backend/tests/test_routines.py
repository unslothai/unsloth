# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused checks for routine scheduling and run history."""

import sqlite3
import time

import pytest

from storage import routines_db as db


@pytest.fixture
def routine_db(tmp_path, monkeypatch):
    path = tmp_path / "studio.db"

    def connect():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    monkeypatch.setattr(db, "get_connection", connect)
    return path


def test_manual_run_snapshots_steps_and_prevents_overlap(routine_db):
    first = [{"kind": "recipe", "label": "Original"}]
    routine = db.save_routine(routine_id = None, name = "Daily dataset", steps = first,
                              interval_minutes = 60, enabled = False)
    run_id = db.run_now(routine["id"])
    db.save_routine(routine_id = routine["id"], name = "Daily dataset",
                    steps = [{"kind": "recipe", "label": "Edited"}],
                    interval_minutes = 60, enabled = False)

    assert db.get_run(run_id)["steps"] == first
    with pytest.raises(RuntimeError, match = "already running"):
        db.run_now(routine["id"])

    claimed = db.claim_queued_run()
    assert claimed["id"] == run_id
    assert claimed["status"] == "running"
    assert db.claim_queued_run() is None
    db.update_run(run_id, status = "completed", current_step = 1,
                  step_statuses = [{"kind": "recipe", "label": "Original", "status": "completed"}])
    assert db.get_run(run_id)["finished_at"] is not None


def test_schedule_skips_missed_intervals_and_pause_blocks_claim(routine_db):
    routine = db.save_routine(routine_id = None, name = "Frequent sync",
                              steps = [{"kind": "folder_sync", "label": "Folder"}],
                              interval_minutes = 15, enabled = True,
                              next_run_at = time.time() - 3600)
    run_id = db.claim_due()
    assert run_id is not None
    assert db.get_run(run_id)["trigger"] == "schedule"
    assert db.get_routine(routine["id"])["next_run_at"] > time.time()
    assert db.claim_due() is None

    db.claim_queued_run()
    db.update_run(run_id, status = "completed")
    with sqlite3.connect(routine_db) as conn:
        conn.execute("UPDATE studio_routines SET next_run_at=? WHERE id=?",
                     (time.time() - 1, routine["id"]))
    db.set_enabled(routine["id"], False)
    assert db.claim_due() is None
    resumed = db.set_enabled(routine["id"], True)
    assert resumed["next_run_at"] > time.time() + 14 * 60


def test_missing_folder_has_actionable_error(monkeypatch):
    from core.rag import folder_sync
    from core.routines import RoutineSupervisor

    def missing(_folder_id):
        raise KeyError(_folder_id)

    monkeypatch.setattr(folder_sync, "request_sync", missing)
    with pytest.raises(RuntimeError, match = "linked folder was removed"):
        RoutineSupervisor(None)._start_step({"kind": "folder_sync", "folder_id": "gone"})
