# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted subagent records (see studio/SUBAGENTS.md) in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from storage.studio_db import get_connection as _studio_connection
from utils.paths import studio_db_path

_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subagents (
            id TEXT NOT NULL PRIMARY KEY,
            parent_thread_id TEXT,
            child_thread_id TEXT NOT NULL DEFAULT '',
            session_id TEXT,
            description TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            model TEXT,
            tools_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'running',
            error TEXT,
            result_preview TEXT,
            tool_call_count INTEGER NOT NULL DEFAULT 0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            transcript_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subagents_parent ON subagents(parent_thread_id, started_at)"
    )


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    """The account's studio.db connection, with the subagents table created once per database file."""
    conn = _studio_connection()
    schema_path = studio_db_path().resolve()
    if schema_path not in _schema_ready:
        with _schema_lock:
            if schema_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    conn.commit()
                    _schema_ready.add(schema_path)
                except Exception:
                    conn.close()
                    raise
    return conn


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


def _now_ms() -> int:
    return int(time.time() * 1000)


def _loads(value: Optional[str], fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _summary_from_row(row: sqlite3.Row) -> dict:
    tools = _loads(row["tools_json"], [])
    return {
        "id": row["id"],
        "parentThreadId": row["parent_thread_id"],
        "childThreadId": row["child_thread_id"],
        "sessionId": row["session_id"],
        "description": row["description"],
        "prompt": row["prompt"],
        "model": row["model"],
        "tools": tools if isinstance(tools, list) else [],
        "status": row["status"],
        "error": row["error"],
        "resultPreview": row["result_preview"],
        "toolCallCount": row["tool_call_count"],
        # Live only: a stored run has no one waiting on an approval.
        "pendingApproval": None,
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
    }


_SUMMARY_COLUMNS = (
    "id, parent_thread_id, child_thread_id, session_id, description, prompt, model, tools_json,"
    " status, error, result_preview, tool_call_count, started_at, finished_at"
)


def upsert_subagent(record: dict) -> None:
    """Insert or replace by ``record["id"]``; ``pendingApproval`` is never stored."""
    transcript = record.get("transcript")
    with _connect("BEGIN IMMEDIATE") as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO subagents (
                id, parent_thread_id, child_thread_id, session_id, description, prompt, model,
                tools_json, status, error, result_preview, tool_call_count, started_at,
                finished_at, transcript_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record.get("parentThreadId"),
                record.get("childThreadId") or "",
                record.get("sessionId"),
                record.get("description") or "",
                record.get("prompt") or "",
                record.get("model"),
                json.dumps(list(record.get("tools") or []), ensure_ascii = False),
                record.get("status") or "running",
                record.get("error"),
                record.get("resultPreview"),
                int(record.get("toolCallCount") or 0),
                int(record["startedAt"]) if record.get("startedAt") is not None else _now_ms(),
                record.get("finishedAt"),
                json.dumps(list(transcript or []), ensure_ascii = False),
            ),
        )


def get_subagent(subagent_id: str) -> Optional[dict]:
    """The full record including ``transcript``, or None."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM subagents WHERE id = ?", (subagent_id,)).fetchone()
        if row is None:
            return None
        record = _summary_from_row(row)
        transcript = _loads(row["transcript_json"], [])
        record["transcript"] = transcript if isinstance(transcript, list) else []
        return record


def list_subagents(parent_thread_id: str) -> list[dict]:
    """Summaries (no ``transcript``) for a parent thread, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM subagents WHERE parent_thread_id = ?"
            " ORDER BY started_at, rowid",
            (parent_thread_id,),
        ).fetchall()
        return [_summary_from_row(row) for row in rows]


def delete_subagents_for_parent(parent_thread_id: str) -> int:
    with _connect("BEGIN IMMEDIATE") as conn:
        return conn.execute(
            "DELETE FROM subagents WHERE parent_thread_id = ?", (parent_thread_id,)
        ).rowcount
