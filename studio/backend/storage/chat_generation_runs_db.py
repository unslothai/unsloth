# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transactional state and cursor events for durable Studio chat generations."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from typing import Any, Iterable, Union

from storage.studio_db import get_connection

ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
_EVENTS_CHANGED = threading.Condition()
_RUN_TOMBSTONE_PREFIX = "chat-generation-run-tombstone:"
ChatGenerationEventInput = Union[tuple[str, dict[str, Any]], tuple[str, dict[str, Any], int]]


class ChatGenerationConflictError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def canonical_request(
    *,
    thread_id: str,
    user_message_id: str,
    assistant_message_id: str,
    request_payload: dict[str, Any],
) -> tuple[str, str]:
    request_json = json.dumps(
        request_payload,
        sort_keys = True,
        separators = (",", ":"),
        ensure_ascii = False,
    )
    identity = json.dumps(
        {
            "threadId": thread_id,
            "userMessageId": user_message_id,
            "assistantMessageId": assistant_message_id,
            "requestPayload": request_payload,
        },
        sort_keys = True,
        separators = (",", ":"),
        ensure_ascii = False,
    )
    return request_json, hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "threadId": row["thread_id"],
        "userMessageId": row["user_message_id"],
        "assistantMessageId": row["assistant_message_id"],
        "requestHash": row["request_hash"],
        "requestPayload": _loads(row["request_json"], {}),
        "status": row["status"],
        "cancelRequested": bool(row["cancel_requested"]),
        "lastEventSeq": int(row["last_event_seq"]),
        "finishReason": row["finish_reason"],
        "error": row["error_message"],
        "createdAt": int(row["created_at"]),
        "updatedAt": int(row["updated_at"]),
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def _append_events_locked(
    conn: sqlite3.Connection, run_id: str, events: Iterable[ChatGenerationEventInput]
) -> list[int]:
    row = conn.execute(
        "SELECT last_event_seq FROM chat_generation_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    seq = int(row["last_event_seq"])
    batch_created = now_ms()
    sequences: list[int] = []
    for event in events:
        event_type, payload = event[:2]
        created = event[2] if len(event) == 3 else batch_created
        seq += 1
        conn.execute(
            """INSERT INTO chat_generation_events
               (run_id, seq, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                seq,
                event_type,
                json.dumps(payload, ensure_ascii = False, separators = (",", ":")),
                created,
            ),
        )
        sequences.append(seq)
    if sequences:
        conn.execute(
            "UPDATE chat_generation_runs SET last_event_seq=?, updated_at=? WHERE id=?",
            (seq, batch_created, run_id),
        )
    return sequences


def _commit(conn: sqlite3.Connection, *, notify: bool = False) -> None:
    conn.commit()
    if notify:
        with _EVENTS_CHANGED:
            _EVENTS_CHANGED.notify_all()


def _sync_assistant_status_locked(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    row = conn.execute(
        """SELECT r.assistant_message_id, r.finish_reason, m.metadata_json
           FROM chat_generation_runs r
           LEFT JOIN chat_messages m ON m.id=r.assistant_message_id
           WHERE r.id=?""",
        (run_id,),
    ).fetchone()
    if row is None or row["metadata_json"] is None:
        return
    metadata = _loads(row["metadata_json"], {})
    if not isinstance(metadata, dict) or metadata.get("generationRunId") not in (None, run_id):
        return
    metadata.update(
        {
            "generationRunId": run_id,
            "generationStatus": status,
            "serverManaged": True,
        }
    )
    if status == "cancelled":
        metadata["incomplete"] = {"reason": "cancelled"}
    elif status == "failed":
        metadata["incomplete"] = {"reason": "interrupted"}
    elif status == "completed":
        if row["finish_reason"] == "length":
            metadata["incomplete"] = {"reason": "length"}
        else:
            metadata.pop("incomplete", None)
    conn.execute(
        "UPDATE chat_messages SET metadata_json=? WHERE id=?",
        (json.dumps(metadata, ensure_ascii = False), row["assistant_message_id"]),
    )


def create_run(
    *,
    run_id: str,
    owner_subject: str,
    thread_id: str,
    user_message_id: str,
    assistant_message_id: str,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    request_json, request_hash = canonical_request(
        thread_id = thread_id,
        user_message_id = user_message_id,
        assistant_message_id = assistant_message_id,
        request_payload = request_payload,
    )
    created = now_ms()
    worker_token = secrets.token_hex(16)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["owner_subject"] != owner_subject
                or existing["request_hash"] != request_hash
            ):
                raise ChatGenerationConflictError("Run ID is already bound to another request")
            conn.commit()
            return _run_from_row(existing), False
        tombstone = conn.execute(
            "SELECT 1 FROM app_settings WHERE key=?",
            (f"{_RUN_TOMBSTONE_PREFIX}{run_id}",),
        ).fetchone()
        if tombstone is not None:
            raise ChatGenerationConflictError("Run ID has already been used")

        thread = conn.execute("SELECT 1 FROM chat_threads WHERE id=?", (thread_id,)).fetchone()
        user_message = conn.execute(
            "SELECT thread_id, role FROM chat_messages WHERE id=?",
            (user_message_id,),
        ).fetchone()
        if thread is None:
            raise KeyError("thread")
        if (
            user_message is None
            or user_message["thread_id"] != thread_id
            or user_message["role"] != "user"
        ):
            raise ValueError("userMessageId must identify a user message in the thread")
        active = conn.execute(
            """SELECT 1 FROM chat_generation_runs
               WHERE thread_id=? AND status IN ('queued','running','cancelling')""",
            (thread_id,),
        ).fetchone()
        if active is not None:
            raise ChatGenerationConflictError("This thread already has an active generation")

        metadata = {
            "generationRunId": run_id,
            "generationSeq": 0,
            "generationStatus": "queued",
            "serverManaged": True,
        }
        assistant = conn.execute(
            "SELECT * FROM chat_messages WHERE id=?",
            (assistant_message_id,),
        ).fetchone()
        if assistant is None:
            conn.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, parent_id, role, content_json, metadata_json, created_at)
                   VALUES (?, ?, ?, 'assistant', '[]', ?, ?)""",
                (
                    assistant_message_id,
                    thread_id,
                    user_message_id,
                    json.dumps(metadata, ensure_ascii = False),
                    created,
                ),
            )
        else:
            assistant_metadata = _loads(assistant["metadata_json"], {})
            existing_run_id = (
                assistant_metadata.get("generationRunId")
                if isinstance(assistant_metadata, dict)
                else None
            )
            content = _loads(assistant["content_json"], [])
            has_content = isinstance(content, list) and any(
                isinstance(part, dict)
                and (
                    (part.get("type") == "text" and str(part.get("text") or "").strip())
                    or part.get("type") not in (None, "text")
                )
                for part in content
            )
            if (
                assistant["thread_id"] != thread_id
                or assistant["parent_id"] != user_message_id
                or assistant["role"] != "assistant"
                or existing_run_id not in (None, run_id)
                or (existing_run_id is None and has_content)
            ):
                raise ChatGenerationConflictError(
                    "Assistant message does not match this generation run"
                )
            merged_metadata = (
                dict(assistant_metadata) if isinstance(assistant_metadata, dict) else {}
            )
            merged_metadata.update(metadata)
            conn.execute(
                "UPDATE chat_messages SET metadata_json=? WHERE id=?",
                (json.dumps(merged_metadata, ensure_ascii = False), assistant_message_id),
            )

        try:
            conn.execute(
                """INSERT INTO chat_generation_runs
                   (id, owner_subject, thread_id, user_message_id, assistant_message_id,
                    request_hash, request_json, worker_token, status, cancel_requested, last_event_seq,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?, ?)""",
                (
                    run_id,
                    owner_subject,
                    thread_id,
                    user_message_id,
                    assistant_message_id,
                    request_hash,
                    request_json,
                    worker_token,
                    created,
                    created,
                ),
            )
        except sqlite3.IntegrityError as exc:
            active = conn.execute(
                """SELECT 1 FROM chat_generation_runs
                   WHERE thread_id=?
                     AND status IN ('queued','running','cancelling')""",
                (thread_id,),
            ).fetchone()
            if active is not None:
                raise ChatGenerationConflictError(
                    "This thread already has an active generation"
                ) from exc
            raise
        _append_events_locked(conn, run_id, [("run.created", {"status": "queued"})])
        row = conn.execute("SELECT * FROM chat_generation_runs WHERE id=?", (run_id,)).fetchone()
        _commit(conn, notify = True)
        return _run_from_row(row), True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_run(run_id: str, owner_subject: str | None = None) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        if owner_subject is None:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=? AND owner_subject=?",
                (run_id, owner_subject),
            ).fetchone()
        return _run_from_row(row) if row is not None else None
    finally:
        conn.close()


def get_worker_token(run_id: str) -> str | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT worker_token FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        return str(row["worker_token"]) if row is not None else None
    finally:
        conn.close()


def get_worker_run(
    run_id: str, worker_token: str | None = None
) -> tuple[dict[str, Any], str, str] | None:
    """Return one fenced producer snapshot and its owner from the same row read."""
    conn = get_connection()
    try:
        if worker_token is None:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=? AND worker_token=?",
                (run_id, worker_token),
            ).fetchone()
        if row is None:
            return None
        return _run_from_row(row), str(row["owner_subject"]), str(row["worker_token"])
    finally:
        conn.close()


def list_active(thread_id: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE thread_id=?
                 AND status IN ('queued','running','cancelling')
               ORDER BY created_at, id""",
            (thread_id,),
        ).fetchall()
        return [_run_from_row(row) for row in rows]
    finally:
        conn.close()


def append_events(
    run_id: str, worker_token: str, events: Iterable[ChatGenerationEventInput]
) -> list[int]:
    batch = list(events)
    if not batch:
        return []
    conn = get_connection()
    try:
        # These are recoverable stream checkpoints, not the authoritative final
        # message.  In WAL mode FULL issues FlushFileBuffers/fsync after every
        # commit; at the live-stream cadence that turns small batches into sustained
        # disk barriers on Windows.  NORMAL keeps the database consistent and the
        # checkpoints durable across an application crash, while the terminal FULL
        # transaction in finish_run() syncs the completed stream.
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM chat_generation_runs WHERE id=? AND worker_token=?",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] not in ACTIVE_STATUSES:
            conn.commit()
            return []
        sequences = _append_events_locked(conn, run_id, batch)
        _commit(conn, notify = True)
        return sequences
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_running(run_id: str, worker_token: str) -> bool:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT status, cancel_requested FROM chat_generation_runs
               WHERE id=? AND worker_token=?""",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] == "running":
            conn.commit()
            return True
        if row["status"] != "queued" or bool(row["cancel_requested"]):
            conn.commit()
            return False
        started = now_ms()
        conn.execute(
            """UPDATE chat_generation_runs
               SET status='running', started_at=COALESCE(started_at, ?), updated_at=?
               WHERE id=?""",
            (started, started, run_id),
        )
        _sync_assistant_status_locked(conn, run_id, "running")
        _append_events_locked(conn, run_id, [("run.started", {"status": "running"})])
        _commit(conn, notify = True)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def request_cancel(run_id: str, owner_subject: str | None = None) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        sql = "SELECT * FROM chat_generation_runs WHERE id=?"
        args: tuple[Any, ...] = (run_id,)
        if owner_subject is not None:
            sql += " AND owner_subject=?"
            args += (owner_subject,)
        row = conn.execute(sql, args).fetchone()
        if row is None:
            conn.commit()
            return None
        status = row["status"]
        if status in TERMINAL_STATUSES or status == "cancelling":
            conn.commit()
            return _run_from_row(row)
        updated = now_ms()
        if status == "queued":
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status='cancelled', cancel_requested=1, finish_reason='cancelled',
                       updated_at=?, completed_at=? WHERE id=?""",
                (updated, updated, run_id),
            )
            _append_events_locked(
                conn,
                run_id,
                [("run.cancelled", {"status": "cancelled", "finishReason": "cancelled"})],
            )
            _sync_assistant_status_locked(conn, run_id, "cancelled")
        else:
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status='cancelling', cancel_requested=1, updated_at=? WHERE id=?""",
                (updated, run_id),
            )
            _append_events_locked(conn, run_id, [("run.cancelling", {"status": "cancelling"})])
            _sync_assistant_status_locked(conn, run_id, "cancelling")
        updated_row = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify = True)
        return _run_from_row(updated_row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_run(
    run_id: str,
    *,
    worker_token: str,
    status: str,
    finish_reason: str | None = None,
    error: str | None = None,
    pending_events: Iterable[ChatGenerationEventInput] = (),
) -> dict[str, Any] | None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Invalid terminal status: {status}")
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=? AND worker_token=?",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        if row["status"] in TERMINAL_STATUSES:
            conn.commit()
            return _run_from_row(row)
        if bool(row["cancel_requested"]):
            status = "cancelled"
            finish_reason = "cancelled"
            error = None
        _append_events_locked(conn, run_id, list(pending_events))
        terminal_payload: dict[str, Any] = {
            "status": status,
            "finishReason": finish_reason,
        }
        if error:
            terminal_payload["error"] = error
        _append_events_locked(conn, run_id, [(f"run.{status}", terminal_payload)])
        completed = now_ms()
        conn.execute(
            """UPDATE chat_generation_runs
               SET status=?, finish_reason=?, error_message=?, updated_at=?, completed_at=?
               WHERE id=?""",
            (status, finish_reason, error, completed, completed, run_id),
        )
        _sync_assistant_status_locked(conn, run_id, status)
        updated = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify = True)
        return _run_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_events(
    run_id: str,
    after: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT seq, event_type, payload_json, created_at
               FROM chat_generation_events
               WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?""",
            (run_id, after, limit),
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "type": row["event_type"],
                "payload": _loads(row["payload_json"], {}),
                "createdAt": int(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def wait_for_events(
    run_id: str,
    after: int = 0,
    timeout: float = 15,
) -> list[dict[str, Any]]:
    events = list_events(run_id, after)
    if events:
        return events
    with _EVENTS_CHANGED:
        events = list_events(run_id, after)
        if events:
            return events
        _EVENTS_CHANGED.wait(timeout)
    return list_events(run_id, after)


def reconcile_orphaned_runs(error: str = "Studio restarted during generation") -> int:
    conn = get_connection()
    changed = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """SELECT id, status, cancel_requested FROM chat_generation_runs
               WHERE status IN ('queued','running','cancelling') ORDER BY created_at, id"""
        ).fetchall()
        completed = now_ms()
        for row in rows:
            run_id = row["id"]
            # A Stop that was already recorded outlives the restart. Reporting it as a
            # backend failure would tell the user Studio broke when they stopped it, and
            # finish_run settles this same case as cancelled.
            if str(row["status"]) == "cancelling" or bool(row["cancel_requested"]):
                status, finish_reason, message = "cancelled", "cancelled", None
                terminal = ("run.cancelled", {"status": status, "finishReason": finish_reason})
            else:
                status, finish_reason, message = "failed", "interrupted", error
                terminal = ("run.failed", {"status": status, "error": error, "interrupted": True})
            _append_events_locked(conn, run_id, [terminal])
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status=?, finish_reason=?, error_message=?,
                       updated_at=?, completed_at=? WHERE id=?""",
                (status, finish_reason, message, completed, completed, run_id),
            )
            _sync_assistant_status_locked(conn, run_id, status)
            changed += 1
        _commit(conn, notify = bool(changed))
        return changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
