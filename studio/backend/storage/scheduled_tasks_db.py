# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled tasks and their runs in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from storage.studio_db import delete_chat_threads_in, get_connection

# Per account. Every run opens a chat, so this also bounds how many chats schedules can make.
MAX_TASKS = 100
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


@contextmanager
def _connect(begin: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """One connection: commits what the block wrote, rolls back on any exception."""
    conn = get_connection()
    try:
        if begin:
            conn.execute(begin)
        yield conn
        if conn.in_transaction:
            conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_ms() -> int:
    return int(time.time() * 1000)


def chat_id_for_run(run_id: str) -> str:
    """The run's chat id, derived from the run so an unsettled run's chat can still be found."""
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
        "notify": row["notify"],
        "named": bool(row["named"]),
        "model": row["model"],
        "reasoningEffort": row["reasoning_effort"],
        "allowModelLoad": bool(row["allow_model_load"]),
        "archiveSuccessful": bool(row["archive_successful"]),
        "projectId": row["project_id"],
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
        "allowModelLoad": bool(row["allow_model_load"]),
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
    with _connect("BEGIN") as conn:
        rows = conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return _with_last_run(conn, [_task_from_row(row) for row in rows])


def get_task(task_id: str) -> Optional[dict]:
    with _connect("BEGIN") as conn:
        return _get_task(conn, task_id)


# The columns create and update both write, in _task_values order.
_TASK_COLUMNS = (
    "title",
    "prompt",
    "schedule_json",
    "timezone",
    "enabled",
    "notify",
    "named",
    "model",
    "reasoning_effort",
    "allow_model_load",
    "archive_successful",
    "project_id",
    "next_run_at",
)


def _task_values(task: dict) -> tuple:
    return (
        task["title"],
        task["prompt"],
        json.dumps(task["schedule"]),
        task["timezone"],
        1 if task["enabled"] else 0,
        task["notify"],
        1 if task["named"] else 0,
        task.get("model"),
        task.get("reasoningEffort"),
        1 if task.get("allowModelLoad") else 0,
        1 if task.get("archiveSuccessful") else 0,
        task.get("projectId"),
        task["nextRunAt"],
    )


def create_task(task: dict) -> dict:
    with _connect("BEGIN IMMEDIATE") as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM scheduled_tasks").fetchone()
        if count >= MAX_TASKS:
            raise TaskLimitReached()
        conn.execute(
            f"INSERT INTO scheduled_tasks (id, owner_subject, created_at, {', '.join(_TASK_COLUMNS)},"
            f" updated_at) VALUES ({', '.join('?' * (len(_TASK_COLUMNS) + 4))})",
            (
                task["id"],
                task["ownerSubject"],
                task["createdAt"],
                *_task_values(task),
                task["updatedAt"],
            ),
        )
        return _get_task(conn, task["id"])


# What decides when a task runs; any other change keeps its next run, an overdue one included.
_TIMING_FIELDS = ("schedule", "timezone", "enabled")


def update_task(
    task_id: str,
    changes: dict,
    reschedule: Optional[Callable[[dict], Optional[int]]] = None,
) -> Optional[dict]:
    """Apply ``changes``; ``reschedule`` returns the next run time in the same transaction (raising rolls back)."""
    with _connect("BEGIN IMMEDIATE") as conn:
        row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        stored = _task_from_row(row)
        task = {**stored, **changes}
        if reschedule is not None and any(task[key] != stored[key] for key in _TIMING_FIELDS):
            task["nextRunAt"] = reschedule(task)
        conn.execute(
            f"UPDATE scheduled_tasks SET {', '.join(f'{column} = ?' for column in _TASK_COLUMNS)},"
            " updated_at = ? WHERE id = ?",
            (*_task_values(task), now_ms(), task_id),
        )
        return _get_task(conn, task_id)


def delete_task(task_id: str) -> bool:
    """Delete a task and its run history; finished runs' chats stay, an in-progress run's chat goes too."""
    with _connect("BEGIN IMMEDIATE") as conn:
        running = conn.execute(
            "SELECT id FROM scheduled_task_runs WHERE task_id = ? AND status = 'running'",
            (task_id,),
        ).fetchall()
        if running:
            delete_chat_threads_in(conn, [chat_id_for_run(row["id"]) for row in running])
        cursor = conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0


def request_run(task_id: str, allow_model_load: bool = False) -> Optional[dict]:
    """Queue a run now, outside the schedule. None when the task does not exist."""
    with _connect("BEGIN IMMEDIATE") as conn:
        if (
            conn.execute("SELECT 1 FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            is None
        ):
            return None
        active = conn.execute(
            f"SELECT 1 FROM scheduled_task_runs WHERE task_id = ? AND status IN {_ACTIVE}",
            (task_id,),
        ).fetchone()
        if active is not None:
            raise TaskRunActive()
        run_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO scheduled_task_runs"
            " (id, task_id, source, status, created_at, allow_model_load)"
            " VALUES (?, ?, 'manual', 'queued', ?, ?)",
            (run_id, task_id, now_ms(), 1 if allow_model_load else 0),
        )
        row = conn.execute("SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row)


_STALE_RUN = "status = 'running' AND COALESCE(heartbeat_at, started_at, created_at) < ?"


def _state(
    conn: sqlite3.Connection, now: int, stale_before: Optional[int]
) -> tuple[bool, Optional[int]]:
    """(stale runs to fail, when the next run is due: ``now`` for a queued one), in one read."""
    stale, queued, due = conn.execute(
        f"SELECT ? IS NOT NULL AND EXISTS (SELECT 1 FROM scheduled_task_runs WHERE {_STALE_RUN}),"
        " EXISTS (SELECT 1 FROM scheduled_task_runs WHERE status = 'queued'),"
        " (SELECT MIN(t.next_run_at) FROM scheduled_tasks t"
        f" WHERE t.enabled = 1 AND t.next_run_at IS NOT NULL AND {_NO_ACTIVE_RUN})",
        (stale_before, stale_before),
    ).fetchone()
    return bool(stale), now if queued else due


def poll(
    now: int,
    stale_before: Optional[int],
    worker_id: str,
    next_run_after: Callable[[dict, int], Optional[int]],
) -> tuple[Optional[tuple[dict, dict]], Optional[int]]:
    """Fails stale runs and claims due work; reads first so an idle pass never takes the writer lock."""
    with _connect() as conn:
        stale, due = _state(conn, now, stale_before)
        if stale:
            conn.execute("BEGIN IMMEDIATE")
            _fail_stale(conn, stale_before)
            conn.commit()
            _stale, due = _state(conn, now, None)
        if due is None or due > now:
            return None, due
        conn.execute("BEGIN IMMEDIATE")
        return _claim(conn, now, worker_id, next_run_after), due


def claim_next(
    now: int, worker_id: str, next_run_after: Callable[[dict, int], Optional[int]]
) -> Optional[tuple[dict, dict]]:
    """Claim the next run (queued Run now first, then most overdue); missed occurrences collapse into one."""
    return poll(now, None, worker_id, next_run_after)[0]


def _claim(
    conn: sqlite3.Connection,
    now: int,
    worker_id: str,
    next_run_after: Callable[[dict, int], Optional[int]],
) -> Optional[tuple[dict, dict]]:
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
        task_id, run_id = queued["task_id"], queued["id"]
    else:
        task_row = conn.execute(
            f"SELECT * FROM scheduled_tasks t WHERE {_DUE_TASK}"
            " ORDER BY t.next_run_at, t.created_at LIMIT 1",
            (now,),
        ).fetchone()
        if task_row is None:
            return None
        task = _task_from_row(task_row)
        try:
            following = next_run_after(task, max(now, task["nextRunAt"]))
        except Exception:
            # A schedule that no longer resolves runs once, then waits for an edit instead of firing every poll.
            following = None
        task_id, run_id = task["id"], str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO scheduled_task_runs
                (id, task_id, source, status, scheduled_for, created_at, started_at,
                 heartbeat_at, worker_id)
            VALUES (?, ?, 'schedule', 'running', ?, ?, ?, ?, ?)
            """,
            (run_id, task_id, task["nextRunAt"], now, now, now, worker_id),
        )
        conn.execute(
            "UPDATE scheduled_tasks SET next_run_at = ? WHERE id = ?", (following, task_id)
        )
    task_row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
    run_row = conn.execute("SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)).fetchone()
    return _task_from_row(task_row), _run_from_row(run_row)


def release_claim(run_id: str, worker_id: str, claimed_next_run_at: Optional[int]) -> bool:
    """Undo a claim that will not run; False when the run is no longer this worker's to give back."""
    with _connect("BEGIN IMMEDIATE") as conn:
        run = conn.execute(
            "SELECT * FROM scheduled_task_runs WHERE id = ? AND worker_id = ? AND status = 'running'",
            (run_id, worker_id),
        ).fetchone()
        if run is None:
            return False
        if run["source"] == "manual":
            conn.execute(
                "UPDATE scheduled_task_runs"
                " SET status = 'queued', started_at = NULL, heartbeat_at = NULL, worker_id = NULL"
                " WHERE id = ?",
                (run_id,),
            )
        else:
            conn.execute("DELETE FROM scheduled_task_runs WHERE id = ?", (run_id,))
            conn.execute(
                "UPDATE scheduled_tasks SET next_run_at = ?"
                " WHERE id = ? AND enabled = 1 AND next_run_at IS ?",
                (run["scheduled_for"], run["task_id"], claimed_next_run_at),
            )
        return True


def heartbeat(run_id: str, worker_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE scheduled_task_runs SET heartbeat_at = ?"
            " WHERE id = ? AND worker_id = ? AND status = 'running'",
            (now_ms(), run_id, worker_id),
        )
        return cursor.rowcount > 0


def task_exists(task_id: str) -> bool:
    with _connect() as conn:
        return (
            conn.execute("SELECT 1 FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            is not None
        )


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
    discard_chat: bool = False,
) -> bool:
    """Settle a run this worker owns and trim history; ``discard_chat`` removes a failed run's chat atomically."""
    with _connect("BEGIN IMMEDIATE") as conn:
        row = conn.execute(
            "SELECT task_id FROM scheduled_task_runs"
            " WHERE id = ? AND worker_id = ? AND status = 'running'",
            (run_id, worker_id),
        ).fetchone()
        if row is None:
            return False
        if discard_chat:
            delete_chat_threads_in(conn, [chat_id_for_run(run_id)])
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
        return True


def _fail_stale(conn: sqlite3.Connection, stale_before: int) -> None:
    """Fail runs whose heartbeat stopped before ``stale_before`` (Studio quit or crashed mid-run)."""
    rows = conn.execute(
        f"SELECT id, task_id FROM scheduled_task_runs WHERE {_STALE_RUN}", (stale_before,)
    ).fetchall()
    run_ids = [row["id"] for row in rows]
    conn.executemany(
        "UPDATE scheduled_task_runs"
        " SET status = 'failed', finished_at = ?, error_code = 'interrupted', error = NULL"
        " WHERE id = ?",
        [(now_ms(), run_id) for run_id in run_ids],
    )
    # Delete its chat in this transaction, so a failed run never keeps one and a late save cannot restore it.
    if run_ids:
        delete_chat_threads_in(conn, [chat_id_for_run(run_id) for run_id in run_ids])
    for task_id in {row["task_id"] for row in rows}:
        _trim_history(conn, task_id)


def list_runs(task_id: str, limit: int = RUN_HISTORY_LIMIT) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scheduled_task_runs WHERE task_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [_run_from_row(row) for row in rows]


# Finished runs per answer; Run now is unbounded, so clients page via ``more``.
ACTIVITY_LIMIT = 200


def activity(since: Optional[int]) -> dict:
    """Runs finished since ``since`` (oldest first, at most ACTIVITY_LIMIT, ``more`` pages), active count, next run."""
    with _connect() as conn:
        # One snapshot, so a run settling between two reads is not missed by both.
        conn.execute("BEGIN")
        runs: list[dict] = []
        more = False
        if since is not None:
            rows = conn.execute(
                """
                SELECT r.*, t.title AS task_title, t.notify AS task_notify, t.model AS task_model
                FROM scheduled_task_runs r
                JOIN scheduled_tasks t ON t.id = r.task_id
                WHERE r.finished_at IS NOT NULL AND r.finished_at >= ?
                ORDER BY r.finished_at, r.rowid LIMIT ?
                """,
                (since, ACTIVITY_LIMIT + 1),
            ).fetchall()
            more = len(rows) > ACTIVITY_LIMIT
            runs = [
                {
                    **_run_from_row(row),
                    "taskTitle": row["task_title"],
                    "taskNotify": row["task_notify"],
                    "taskModel": row["task_model"],
                }
                for row in rows[:ACTIVITY_LIMIT]
            ]
        (active,) = conn.execute(
            f"SELECT COUNT(*) FROM scheduled_task_runs WHERE status IN {_ACTIVE}"
        ).fetchone()
        (next_run_at,) = conn.execute(
            "SELECT MIN(next_run_at) FROM scheduled_tasks WHERE enabled = 1"
        ).fetchone()
        return {"runs": runs, "more": more, "activeRuns": active, "nextRunAt": next_run_at}
