# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-off scheduled training starts and their saved request, in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from storage import credential_secrets
from storage.studio_db import get_connection

# A start still runs this long after its time; later it is missed rather than run at a surprise hour.
START_GRACE_MS = 5 * 60 * 1000
# Finished schedules kept so the UI can say what became of the last one.
HISTORY_LIMIT = 5

PENDING = "pending"
STARTING = "starting"
STARTED = "started"
CANCELLED = "cancelled"
FAILED = "failed"
MISSED = "missed"

_ACTIVE = "('pending', 'starting')"
_TERMINAL = "('started', 'cancelled', 'failed', 'missed')"

# The request is encrypted with the installation key, keyed by schedule id.
PAYLOAD_KIND = "training_schedule_payload"


class TrainingScheduleExists(Exception):
    """The account already has a training start waiting or in flight."""


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


def _schedule_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "ownerSubject": row["owner_subject"],
        "status": row["status"],
        "startAt": row["start_at"],
        "startAtLocal": row["start_at_local"],
        "timezone": row["timezone"],
        "modelName": row["model_name"],
        "projectName": row["project_name"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "jobId": row["job_id"],
        "error": row["error"],
    }


def _get(conn: sqlite3.Connection, schedule_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM training_schedules WHERE id = ?", (schedule_id,)).fetchone()
    return None if row is None else _schedule_from_row(row)


def list_schedules() -> list[dict]:
    """The waiting start first, then the newest finished ones."""
    with _connect("BEGIN") as conn:
        rows = conn.execute(
            "SELECT * FROM training_schedules ORDER BY"
            " CASE status WHEN 'pending' THEN 0 WHEN 'starting' THEN 1 ELSE 2 END,"
            " CASE WHEN status IN ('pending', 'starting') THEN start_at ELSE 0 END,"
            " CASE WHEN status IN ('pending', 'starting') THEN 0 ELSE updated_at END DESC,"
            " rowid DESC"
        ).fetchall()
        return [_schedule_from_row(row) for row in rows]


def get_schedule(schedule_id: str) -> Optional[dict]:
    with _connect("BEGIN") as conn:
        return _get(conn, schedule_id)


def pending_schedule() -> Optional[dict]:
    with _connect("BEGIN") as conn:
        row = conn.execute(
            "SELECT * FROM training_schedules WHERE status = 'pending' ORDER BY start_at LIMIT 1"
        ).fetchone()
        return None if row is None else _schedule_from_row(row)


def create_schedule(schedule: dict, payload: dict) -> dict:
    """Store a start and its encrypted request; one waiting start per account."""
    credential_secrets.ensure_schema()
    with _connect("BEGIN IMMEDIATE") as conn:
        if (
            conn.execute(f"SELECT 1 FROM training_schedules WHERE status IN {_ACTIVE}").fetchone()
            is not None
        ):
            raise TrainingScheduleExists()
        conn.execute(
            """
            INSERT INTO training_schedules
                (id, owner_subject, status, start_at, start_at_local, timezone,
                 model_name, project_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule["id"],
                schedule["ownerSubject"],
                schedule["status"],
                schedule["startAt"],
                schedule["startAtLocal"],
                schedule["timezone"],
                schedule["modelName"],
                schedule["projectName"],
                schedule["createdAt"],
                schedule["updatedAt"],
            ),
        )
        credential_secrets.upsert_secret(
            PAYLOAD_KIND,
            schedule["id"],
            json.dumps(payload),
            connection = conn,
        )
        return _get(conn, schedule["id"])


def get_payload(schedule_id: str) -> Optional[dict]:
    """The saved start request, or None when it is missing or unreadable."""
    plaintext = credential_secrets.get_secret(PAYLOAD_KIND, schedule_id)
    if plaintext is None:
        return None
    try:
        payload = json.loads(plaintext)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _delete_payload(conn: sqlite3.Connection, schedule_id: str) -> None:
    credential_secrets.delete_secret(PAYLOAD_KIND, schedule_id, connection = conn)


def _trim(conn: sqlite3.Connection) -> None:
    """Keep only the newest finished schedules, with any payload they somehow still hold."""
    rows = conn.execute(
        f"SELECT id FROM training_schedules WHERE status IN {_TERMINAL}"
        " AND rowid NOT IN ("
        f"  SELECT rowid FROM training_schedules WHERE status IN {_TERMINAL}"
        "   ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        ")",
        (HISTORY_LIMIT,),
    ).fetchall()
    for row in rows:
        _delete_payload(conn, row["id"])
    conn.executemany("DELETE FROM training_schedules WHERE id = ?", [(row["id"],) for row in rows])


def _set_status(
    conn: sqlite3.Connection,
    schedule_id: str,
    status: str,
    *,
    job_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    conn.execute(
        "UPDATE training_schedules SET status = ?, job_id = ?, error = ?, updated_at = ?"
        " WHERE id = ?",
        (status, job_id, error, now_ms(), schedule_id),
    )
    _delete_payload(conn, schedule_id)
    _trim(conn)


def _miss_overdue(conn: sqlite3.Connection, now: int) -> None:
    rows = conn.execute(
        "SELECT id FROM training_schedules WHERE status = 'pending' AND start_at < ?",
        (now - START_GRACE_MS,),
    ).fetchall()
    for row in rows:
        _set_status(
            conn,
            row["id"],
            MISSED,
            error = "The scheduled start time passed while Unsloth Studio was not running.",
        )


def _reconcile(conn: sqlite3.Connection, now: int) -> None:
    """Settle what a previous Studio left behind: a claimed start never began, overdue ones are missed."""
    conn.execute(
        "UPDATE training_schedules SET status = 'pending', claimed_at = NULL, updated_at = ?"
        " WHERE status = 'starting'",
        (now,),
    )
    _miss_overdue(conn, now)


def reconcile_startup(now: int) -> None:
    with _connect("BEGIN IMMEDIATE") as conn:
        _reconcile(conn, now)


def _next_pending(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        "SELECT MIN(start_at) FROM training_schedules WHERE status = 'pending'"
    ).fetchone()
    return row[0]


def poll(now: int) -> tuple[Optional[dict], Optional[int]]:
    """One scheduler pass: claim the due start and say when the next one is due.

    Reads first, so an idle pass never takes the writer lock. A row still 'starting' belongs to a
    live claim of this process; reconcile_startup() settles the ones a previous process left."""
    with _connect() as conn:
        due = _next_pending(conn)
        if due is None or due > now:
            return None, due
        conn.execute("BEGIN IMMEDIATE")
        _miss_overdue(conn, now)
        row = conn.execute(
            "SELECT * FROM training_schedules WHERE status = 'pending' AND start_at <= ?"
            " ORDER BY start_at, created_at LIMIT 1",
            (now,),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE training_schedules SET status = 'starting', claimed_at = ?, updated_at = ?"
                " WHERE id = ?",
                (now, now, row["id"]),
            )
        return (None if row is None else _get(conn, row["id"])), _next_pending(conn)


def claim_due(now: int) -> Optional[dict]:
    return poll(now)[0]


def release_claim(schedule_id: str) -> bool:
    """Give a claimed start back; False when it is no longer this process's to release."""
    with _connect("BEGIN IMMEDIATE") as conn:
        cursor = conn.execute(
            "UPDATE training_schedules SET status = 'pending', claimed_at = NULL, updated_at = ?"
            " WHERE id = ? AND status = 'starting'",
            (now_ms(), schedule_id),
        )
        return cursor.rowcount > 0


def finish_claim(
    schedule_id: str,
    status: str,
    *,
    job_id: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """Settle a claimed start; False when it was already settled or released."""
    if status not in (STARTED, FAILED, MISSED, CANCELLED):
        raise ValueError(f"not a terminal schedule status: {status}")
    with _connect("BEGIN IMMEDIATE") as conn:
        if (
            conn.execute(
                "SELECT 1 FROM training_schedules WHERE id = ? AND status = 'starting'",
                (schedule_id,),
            ).fetchone()
            is None
        ):
            return False
        _set_status(conn, schedule_id, status, job_id = job_id, error = error)
        return True


def cancel_pending(schedule_id: str) -> tuple[Optional[dict], bool]:
    """Cancel a waiting start: the row (None when it does not exist) and whether it was still waiting."""
    with _connect("BEGIN IMMEDIATE") as conn:
        cursor = conn.execute(
            "UPDATE training_schedules SET status = 'cancelled', updated_at = ?"
            " WHERE id = ? AND status = 'pending'",
            (now_ms(), schedule_id),
        )
        cancelled = cursor.rowcount > 0
        if cancelled:
            _delete_payload(conn, schedule_id)
            _trim(conn)
        return _get(conn, schedule_id), cancelled


def cancel_all_active() -> int:
    """Drop every waiting start of this account, e.g. when the account is retired."""
    with _connect("BEGIN IMMEDIATE") as conn:
        rows = conn.execute(
            f"SELECT id FROM training_schedules WHERE status IN {_ACTIVE}"
        ).fetchall()
        for row in rows:
            _set_status(conn, row["id"], CANCELLED, error = "The account was removed.")
        return len(rows)
