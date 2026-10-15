# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-account durable definitions and execution history for Studio routines."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import closing
from typing import Any

from storage.studio_db import get_connection


def _connect():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS studio_routines (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            interval_minutes INTEGER NOT NULL,
            next_run_at REAL NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS studio_routine_runs (
            id TEXT PRIMARY KEY,
            routine_id TEXT NOT NULL REFERENCES studio_routines(id) ON DELETE CASCADE,
            trigger TEXT NOT NULL,
            status TEXT NOT NULL,
            steps_json TEXT NOT NULL,
            current_step INTEGER NOT NULL DEFAULT 0,
            step_status_json TEXT NOT NULL DEFAULT '[]',
            error TEXT,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL
        );
        CREATE INDEX IF NOT EXISTS studio_routines_due ON studio_routines(enabled, next_run_at);
        CREATE INDEX IF NOT EXISTS studio_routine_runs_recent ON studio_routine_runs(routine_id, created_at DESC);
    """)
    return conn


def _routine(row) -> dict[str, Any]:
    value = dict(row)
    value["steps"] = json.loads(value.pop("steps_json"))
    value["enabled"] = bool(value["enabled"])
    return value


def _run(row) -> dict[str, Any]:
    value = dict(row)
    value["steps"] = json.loads(value.pop("steps_json"))
    value["step_statuses"] = json.loads(value.pop("step_status_json"))
    return value


def list_routines() -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM studio_routines ORDER BY updated_at DESC").fetchall()
        return [_routine(row) for row in rows]


def get_routine(routine_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM studio_routines WHERE id=?", (routine_id,)).fetchone()
        return _routine(row) if row else None


def save_routine(
    *,
    routine_id: str | None,
    name: str,
    steps: list[dict],
    interval_minutes: int,
    enabled: bool,
    next_run_at: float | None = None,
) -> dict[str, Any]:
    now = time.time()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if routine_id is None:
            routine_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO studio_routines VALUES (?,?,?,?,?,?,?,?)",
                (
                    routine_id,
                    name,
                    json.dumps(steps),
                    interval_minutes,
                    next_run_at or now + interval_minutes * 60,
                    int(enabled),
                    now,
                    now,
                ),
            )
        else:
            old = conn.execute(
                "SELECT id FROM studio_routines WHERE id=?", (routine_id,)
            ).fetchone()
            if old is None:
                raise KeyError(routine_id)
            conn.execute(
                "UPDATE studio_routines SET name=?, steps_json=?, interval_minutes=?, "
                "next_run_at=?, enabled=?, updated_at=? WHERE id=?",
                (
                    name,
                    json.dumps(steps),
                    interval_minutes,
                    next_run_at or now + interval_minutes * 60,
                    int(enabled),
                    now,
                    routine_id,
                ),
            )
        conn.commit()
    return get_routine(routine_id)


def set_enabled(routine_id: str, enabled: bool) -> dict[str, Any] | None:
    now = time.time()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        changed = conn.execute(
            "UPDATE studio_routines SET enabled=?, next_run_at=CASE WHEN ? THEN ? + interval_minutes * 60 ELSE next_run_at END, "
            "updated_at=? WHERE id=?",
            (int(enabled), int(enabled), now, now, routine_id),
        ).rowcount
        conn.commit()
    return get_routine(routine_id) if changed else None


def list_runs(routine_id: str, limit: int = 30) -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM studio_routine_runs WHERE routine_id=? ORDER BY created_at DESC LIMIT ?",
            (routine_id, limit),
        ).fetchall()
        return [_run(row) for row in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM studio_routine_runs WHERE id=?", (run_id,)).fetchone()
        return _run(row) if row else None


def _create_run(conn, routine, trigger: str, now: float) -> str:
    run_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO studio_routine_runs "
        "(id,routine_id,trigger,status,steps_json,created_at) VALUES (?,?,?,?,?,?)",
        (run_id, routine["id"], trigger, "queued", routine["steps_json"], now),
    )
    return run_id


def run_now(routine_id: str) -> str:
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        routine = conn.execute("SELECT * FROM studio_routines WHERE id=?", (routine_id,)).fetchone()
        if routine is None:
            raise KeyError(routine_id)
        active = conn.execute(
            "SELECT 1 FROM studio_routine_runs WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone()
        if active:
            raise RuntimeError("A routine is already running for this account")
        run_id = _create_run(conn, routine, "manual", time.time())
        conn.commit()
        return run_id


def claim_due() -> str | None:
    now = time.time()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute(
            "SELECT 1 FROM studio_routine_runs WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone()
        if active:
            conn.commit()
            return None
        routine = conn.execute(
            "SELECT * FROM studio_routines WHERE enabled=1 AND next_run_at<=? "
            "ORDER BY next_run_at LIMIT 1",
            (now,),
        ).fetchone()
        if routine is None:
            conn.commit()
            return None
        # Missed occurrences are skipped. One scheduled occurrence is claimed per wake.
        interval = routine["interval_minutes"] * 60
        next_at = (
            routine["next_run_at"] + ((now - routine["next_run_at"]) // interval + 1) * interval
        )
        conn.execute(
            "UPDATE studio_routines SET next_run_at=? WHERE id=?", (next_at, routine["id"])
        )
        run_id = _create_run(conn, routine, "schedule", now)
        conn.commit()
        return run_id


def claim_queued_run() -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM studio_routine_runs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE studio_routine_runs SET status='running', started_at=? WHERE id=?",
            (time.time(), row["id"]),
        )
        conn.commit()
        return get_run(row["id"])


def update_run(
    run_id: str,
    *,
    status: str,
    current_step: int | None = None,
    step_statuses: list[dict] | None = None,
    error: str | None = None,
) -> None:
    now = time.time()
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE studio_routine_runs SET status=?, current_step=COALESCE(?,current_step), "
            "step_status_json=COALESCE(?,step_status_json), error=?, "
            "started_at=CASE WHEN ?='running' THEN COALESCE(started_at,?) ELSE started_at END, "
            "finished_at=CASE WHEN ? IN ('completed','failed','cancelled') THEN ? ELSE finished_at END "
            "WHERE id=?",
            (
                status,
                current_step,
                json.dumps(step_statuses) if step_statuses is not None else None,
                error,
                status,
                now,
                status,
                now,
                run_id,
            ),
        )
        conn.commit()


def fail_interrupted_runs() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE studio_routine_runs SET status='failed', error='Studio stopped during this run', "
            "finished_at=? WHERE status IN ('queued','running')",
            (time.time(),),
        )
        conn.commit()
