# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled tasks and their runs in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Callable, Optional

from storage.studio_db import delete_chat_threads, get_connection

# Per account. Every run opens a chat, so this also bounds how many chats schedules can make.
MAX_TASKS = 100
# Finished runs kept per task, newest first. The chats they made are kept regardless.
RUN_HISTORY_LIMIT = 20

_ACTIVE = "('queued', 'running')"
_NO_ACTIVE_RUN = (
    "NOT EXISTS (SELECT 1 FROM scheduled_task_runs r"
    f" WHERE r.task_id = t.id AND r.status IN {_ACTIVE})"
)
_DUE_TASK = (
    f"t.enabled = 1 AND t.next_run_at IS NOT NULL AND t.next_run_at <= ? AND {_NO_ACTIVE_RUN}"
)


class TaskLimitReached(Exception):
    pass


class TaskRunActive(Exception):
    """The task already has a run queued or in progress."""


# Any fixed value: it only keeps run chat ids apart from every other uuid5.
_RUN_CHAT_NAMESPACE = uuid.UUID("5b0e7a1e-36a4-4a39-9c1f-2f6d3c8e9a41")


def now_ms() -> int:
    return int(time.time() * 1000)


def chat_id_for_run(run_id: str) -> str:
    """The id of the chat a run writes. Derived from the run, so a chat whose run never settled
    (Studio stopped in between) can be found and removed without the run having recorded it."""
    return str(uuid.uuid5(_RUN_CHAT_NAMESPACE, run_id))


def _task_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "ownerSubject": row["owner_subject"],
        "title": row["title"],
        "prompt": row["prompt"],
        "schedule": json.loads(row["schedule_json"]),
        "timezone": row["timezone"],
        "enabled": bool(row["enabled"]),
        "nextRunAt": row["next_run_at"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _run_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "taskId": row["task_id"],
        "source": row["source"],
        "status": row["status"],
        "scheduledFor": row["scheduled_for"],
        "createdAt": row["created_at"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "threadId": row["thread_id"],
        "model": row["model"],
        "errorCode": row["error_code"],
        "error": row["error"],
    }


def _latest_runs(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, dict]:
    if not task_ids:
        return {}
    rows = conn.execute(
        f"""
        SELECT * FROM scheduled_task_runs WHERE rowid IN (
            SELECT (
                SELECT r.rowid FROM scheduled_task_runs r WHERE r.task_id = t.id
                ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1
            )
            FROM scheduled_tasks t WHERE t.id IN ({", ".join("?" for _ in task_ids)})
        )
        """,
        task_ids,
    ).fetchall()
    return {row["task_id"]: _run_from_row(row) for row in rows}


def _with_last_run(conn: sqlite3.Connection, tasks: list[dict]) -> list[dict]:
    latest = _latest_runs(conn, [task["id"] for task in tasks])
    for task in tasks:
        task["lastRun"] = latest.get(task["id"])
    return tasks


def _get_task(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
    return None if row is None else _with_last_run(conn, [_task_from_row(row)])[0]


def list_tasks() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return _with_last_run(conn, [_task_from_row(row) for row in rows])
    finally:
        conn.close()


def get_task(task_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        return _get_task(conn, task_id)
    finally:
        conn.close()


def create_task(task: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        (count,) = conn.execute("SELECT COUNT(*) FROM scheduled_tasks").fetchone()
        if count >= MAX_TASKS:
            raise TaskLimitReached()
        conn.execute(
            """
            INSERT INTO scheduled_tasks
                (id, owner_subject, title, prompt, schedule_json, timezone, enabled, next_run_at,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task["id"],
                task["ownerSubject"],
                task["title"],
                task["prompt"],
                json.dumps(task["schedule"]),
                task["timezone"],
                1 if task["enabled"] else 0,
                task["nextRunAt"],
                task["createdAt"],
                task["updatedAt"],
            ),
        )
        conn.commit()
        return _get_task(conn, task["id"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# What decides when a task runs; any other change keeps its next run, an overdue one included.
_TIMING_FIELDS = ("schedule", "timezone", "enabled")


def update_task(
    task_id: str,
    changes: dict,
    reschedule: Optional[Callable[[dict], Optional[int]]] = None,
) -> Optional[dict]:
    """Apply ``changes``. When they change when the task runs, ``reschedule`` sees the updated task
    and returns its next run time; resending the values it already has changes nothing.

    Both happen in one transaction, so a run claimed meanwhile cannot be scheduled from a stale
    copy. Whatever ``reschedule`` raises rolls the update back.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            conn.rollback()
            return None
        stored = _task_from_row(row)
        task = {**stored, **changes}
        if reschedule is not None and any(task[key] != stored[key] for key in _TIMING_FIELDS):
            task["nextRunAt"] = reschedule(task)
        conn.execute(
            """
            UPDATE scheduled_tasks
            SET title = ?, prompt = ?, schedule_json = ?, timezone = ?, enabled = ?,
                next_run_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                task["title"],
                task["prompt"],
                json.dumps(task["schedule"]),
                task["timezone"],
                1 if task["enabled"] else 0,
                task["nextRunAt"],
                now_ms(),
                task_id,
            ),
        )
        conn.commit()
        return _get_task(conn, task_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_task(task_id: str) -> bool:
    """Delete a task and its run history. Chats its runs made are left alone."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def request_run(task_id: str) -> Optional[dict]:
    """Queue a run now, outside the schedule. None when the task does not exist."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if (
            conn.execute("SELECT 1 FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            is None
        ):
            conn.rollback()
            return None
        active = conn.execute(
            f"SELECT 1 FROM scheduled_task_runs WHERE task_id = ? AND status IN {_ACTIVE}",
            (task_id,),
        ).fetchone()
        if active is not None:
            raise TaskRunActive()
        run_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO scheduled_task_runs (id, task_id, source, status, created_at)"
            " VALUES (?, ?, 'manual', 'queued', ?)",
            (run_id, task_id, now_ms()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _has_claimable(conn: sqlite3.Connection, now: int) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM scheduled_task_runs WHERE status = 'queued')"
        f" OR EXISTS (SELECT 1 FROM scheduled_tasks t WHERE {_DUE_TASK})",
        (now,),
    ).fetchone()
    return bool(row[0])


def claim_next(
    now: int, worker_id: str, next_run_after: Callable[[dict, int], Optional[int]]
) -> Optional[tuple[dict, dict]]:
    """Start the next run: a queued Run now first, then the most overdue scheduled task.

    A scheduled claim moves the task on to its next run time in the same transaction, so each
    occurrence runs once however the run itself ends. Occurrences missed while Studio was closed
    collapse into this one run: ``next_run_after`` is asked for the first run after now.
    Returns ``(task, run)``, or None when nothing is due.
    """
    conn = get_connection()
    try:
        # Read-only first, so an idle poll never takes the writer lock.
        if not _has_claimable(conn, now):
            return None
        conn.execute("BEGIN IMMEDIATE")
        queued = conn.execute(
            "SELECT * FROM scheduled_task_runs WHERE status = 'queued'"
            " ORDER BY created_at, rowid LIMIT 1"
        ).fetchone()
        if queued is not None:
            conn.execute(
                "UPDATE scheduled_task_runs"
                " SET status = 'running', started_at = ?, heartbeat_at = ?, worker_id = ?"
                " WHERE id = ?",
                (now, now, worker_id, queued["id"]),
            )
            task_row = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE id = ?", (queued["task_id"],)
            ).fetchone()
            run_id = queued["id"]
        else:
            task_row = conn.execute(
                f"SELECT * FROM scheduled_tasks t WHERE {_DUE_TASK}"
                " ORDER BY t.next_run_at, t.created_at LIMIT 1",
                (now,),
            ).fetchone()
            if task_row is None:
                conn.rollback()
                return None
            task = _task_from_row(task_row)
            try:
                following = next_run_after(task, max(now, task["nextRunAt"]))
            except Exception:
                # A schedule that no longer resolves (a time zone dropped from tzdata) runs this
                # once and then waits for an edit, rather than firing on every poll.
                following = None
            run_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO scheduled_task_runs
                    (id, task_id, source, status, scheduled_for, created_at, started_at,
                     heartbeat_at, worker_id)
                VALUES (?, ?, 'schedule', 'running', ?, ?, ?, ?, ?)
                """,
                (run_id, task["id"], task["nextRunAt"], now, now, now, worker_id),
            )
            conn.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?",
                (following, task["id"]),
            )
        conn.commit()
        task_row = conn.execute(
            "SELECT * FROM scheduled_tasks WHERE id = ?", (task_row["id"],)
        ).fetchone()
        run_row = conn.execute(
            "SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _task_from_row(task_row), _run_from_row(run_row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def next_due_at() -> Optional[int]:
    """When the next run should start: now for a queued one, else the earliest due task."""
    conn = get_connection()
    try:
        if conn.execute(
            "SELECT 1 FROM scheduled_task_runs WHERE status = 'queued' LIMIT 1"
        ).fetchone():
            return now_ms()
        row = conn.execute(
            "SELECT MIN(t.next_run_at) FROM scheduled_tasks t"
            f" WHERE t.enabled = 1 AND t.next_run_at IS NOT NULL AND {_NO_ACTIVE_RUN}"
        ).fetchone()
        return row[0]
    finally:
        conn.close()


def heartbeat(run_id: str, worker_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE scheduled_task_runs SET heartbeat_at = ?"
            " WHERE id = ? AND worker_id = ? AND status = 'running'",
            (now_ms(), run_id, worker_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def task_exists(task_id: str) -> bool:
    conn = get_connection()
    try:
        return (
            conn.execute("SELECT 1 FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            is not None
        )
    finally:
        conn.close()


def _trim_history(conn: sqlite3.Connection, task_id: str) -> None:
    """Drop the task's finished runs past the newest RUN_HISTORY_LIMIT."""
    conn.execute(
        f"""
        DELETE FROM scheduled_task_runs
        WHERE task_id = ? AND status NOT IN {_ACTIVE} AND rowid NOT IN (
            SELECT rowid FROM scheduled_task_runs
            WHERE task_id = ? AND status NOT IN {_ACTIVE}
            ORDER BY created_at DESC, rowid DESC LIMIT ?
        )
        """,
        (task_id, task_id, RUN_HISTORY_LIMIT),
    )


def finish_run(
    run_id: str,
    worker_id: str,
    *,
    status: str,
    thread_id: Optional[str] = None,
    model: Optional[str] = None,
    error_code: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """Settle a run this worker owns, and trim its task's history. False when the run is gone."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT task_id FROM scheduled_task_runs"
            " WHERE id = ? AND worker_id = ? AND status = 'running'",
            (run_id, worker_id),
        ).fetchone()
        if row is None:
            conn.rollback()
            return False
        # The chat can already be gone; the foreign key would refuse the whole update.
        if (
            thread_id is not None
            and conn.execute("SELECT 1 FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
            is None
        ):
            thread_id = None
        conn.execute(
            """
            UPDATE scheduled_task_runs
            SET status = ?, finished_at = ?, thread_id = ?, model = ?, error_code = ?, error = ?
            WHERE id = ?
            """,
            (status, now_ms(), thread_id, model, error_code, error, run_id),
        )
        _trim_history(conn, row["task_id"])
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def recover_stale_runs(stale_before: int) -> int:
    """Fail runs whose heartbeat stopped before ``stale_before``: Studio quit or crashed mid-run,
    or the worker could not settle the run, which would otherwise hold its task as running.

    A run still being heartbeated is being worked on and is left alone. The scheduler sweeps only
    between its own runs, so none of its runs is in progress then.
    """
    stale = "status = 'running' AND COALESCE(heartbeat_at, started_at, created_at) < ?"
    conn = get_connection()
    try:
        if (
            conn.execute(
                f"SELECT 1 FROM scheduled_task_runs WHERE {stale} LIMIT 1", (stale_before,)
            ).fetchone()
            is None
        ):
            return 0
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            f"SELECT id, task_id FROM scheduled_task_runs WHERE {stale}", (stale_before,)
        ).fetchall()
        run_ids = [row["id"] for row in rows]
        conn.executemany(
            "UPDATE scheduled_task_runs"
            " SET status = 'failed', finished_at = ?, error_code = 'interrupted', error = NULL"
            " WHERE id = ?",
            [(now_ms(), run_id) for run_id in run_ids],
        )
        # As finish_run does: a task whose runs keep ending this way must not pile them up.
        for task_id in {row["task_id"] for row in rows}:
            _trim_history(conn, task_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    # One that stopped after writing its chat but before settling left the chat behind, and a
    # failed run has none. Deleting also keeps a save still under way from bringing it back.
    if run_ids:
        delete_chat_threads([chat_id_for_run(run_id) for run_id in run_ids])
    return len(run_ids)


def list_runs(task_id: str, limit: int = RUN_HISTORY_LIMIT) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM scheduled_task_runs WHERE task_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [_run_from_row(row) for row in rows]
    finally:
        conn.close()


# Finished runs per answer. Scheduled runs alone fit, at most one a day per task, but Run now has
# no such bound, so a client pages through the rest (see ``more``).
ACTIVITY_LIMIT = 200


def activity(since: Optional[int]) -> dict:
    """What a client watching for results needs: runs finished at or after ``since`` (oldest
    first, each with its task's title), how many are queued or running, and the next run time.

    At most ACTIVITY_LIMIT runs; ``more`` says there are others, which a call from the last
    run's ``finishedAt`` returns (repeating any that finished in that same millisecond).
    """
    conn = get_connection()
    try:
        runs: list[dict] = []
        more = False
        if since is not None:
            rows = conn.execute(
                """
                SELECT r.*, t.title AS task_title FROM scheduled_task_runs r
                JOIN scheduled_tasks t ON t.id = r.task_id
                WHERE r.finished_at IS NOT NULL AND r.finished_at >= ?
                ORDER BY r.finished_at, r.rowid LIMIT ?
                """,
                (since, ACTIVITY_LIMIT + 1),
            ).fetchall()
            more = len(rows) > ACTIVITY_LIMIT
            runs = [
                {**_run_from_row(row), "taskTitle": row["task_title"]}
                for row in rows[:ACTIVITY_LIMIT]
            ]
        (active,) = conn.execute(
            f"SELECT COUNT(*) FROM scheduled_task_runs WHERE status IN {_ACTIVE}"
        ).fetchone()
        (next_run_at,) = conn.execute(
            "SELECT MIN(next_run_at) FROM scheduled_tasks WHERE enabled = 1"
        ).fetchone()
        return {"runs": runs, "more": more, "activeRuns": active, "nextRunAt": next_run_at}
    finally:
        conn.close()
