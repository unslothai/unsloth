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

# Progress lease columns, added here rather than in _ensure_schema so the base table
# definition stays owned by studio_db. Named _schema_ready to match the flag the test
# harness resets on every storage module when it swaps UNSLOTH_STUDIO_HOME.
_schema_ready = False
_schema_lock = threading.Lock()


class ChatGenerationConflictError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def _connect() -> sqlite3.Connection:
    """get_connection plus the one-off progress-lease migration for this database."""
    global _schema_ready
    conn = get_connection()
    if _schema_ready:
        return conn
    try:
        with _schema_lock:
            if not _schema_ready:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(chat_generation_runs)").fetchall()
                }
                for column, spec in (
                    ("progress_at", "INTEGER"),
                    ("progress_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    if column in columns:
                        continue
                    try:
                        conn.execute(f"ALTER TABLE chat_generation_runs ADD COLUMN {column} {spec}")
                    except sqlite3.OperationalError as exc:
                        # Another process migrated the same database first.
                        if "duplicate column" not in str(exc).lower():
                            raise
                conn.commit()
                _schema_ready = True
    except sqlite3.OperationalError:
        # A writer holds the database. The columns are additive, so let this call through
        # and migrate later rather than turning contention into a failed history read;
        # _without_lease_columns below covers the paths that then cannot assume them.
        conn.rollback()
    except Exception:
        conn.close()
        raise
    return conn


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


def _missing_lease_columns(exc: sqlite3.OperationalError) -> bool:
    """Whether `exc` is this database still waiting on the progress-lease migration.

    _connect lets a call through when contention blocks the ALTER, so every statement
    naming progress_at or progress_tokens can meet a table that predates them. Degrading
    to the pre-migration behaviour keeps that window harmless: without it a blocked
    migration would abort a generation with `no such column` the moment the writer let go.
    """
    message = str(exc).lower()
    return "no such column" in message and (
        "progress_at" in message or "progress_tokens" in message
    )


def _touch_progress_locked(conn: sqlite3.Connection, run_id: str, tokens: int) -> None:
    """Stamp the progress lease for one flush of streamed output.

    Monotonic in both fields, the same rule studio_db._safe_generation_assistant_update
    applies to the assistant row this run owns: the token counter only ever accumulates,
    and progress_at takes MAX(stored, now) so a wall-clock step backwards (NTP, suspend)
    cannot age a live run into the sweep below. One chunk carries at most one token
    delta, so the count of chunk events is the token count.

    updated_at moves with it, as it already does on every event append. That is what the
    follower's snapshot poll compares, so a client watching a run through a long model
    preparation or an admission wait, neither of which emits events, sees the server is
    alive and rearms its own no-progress deadline instead of reporting an interruption
    over healthy work.
    """
    now = now_ms()
    try:
        conn.execute(
            """UPDATE chat_generation_runs
               SET progress_at=MAX(COALESCE(progress_at, 0), ?),
                   updated_at=MAX(COALESCE(updated_at, 0), ?),
                   progress_tokens=COALESCE(progress_tokens, 0) + ?
               WHERE id=?""",
            (now, now, max(0, int(tokens)), run_id),
        )
    except sqlite3.OperationalError as exc:
        # The migration has not landed yet. The run keeps streaming and simply ages out
        # on started_at/created_at, which the sweep already falls back to.
        if not _missing_lease_columns(exc):
            raise


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
    conn = _connect()
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
    conn = _connect()
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
    conn = _connect()
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
    conn = _connect()
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


def touch_progress(run_id: str) -> None:
    """Renew one run's progress lease without recording any streamed output.

    For work the lease cannot see. Automatic model loading, idle reload and auto-download
    all happen between mark_running and the first token, and the engine's own first-token
    budget does not start until after them, so ageing a run from mark_running could reap a
    legitimate load followed by a legitimate prefill.
    """
    conn = _connect()
    try:
        _touch_progress_locked(conn, run_id, 0)
        conn.commit()
    finally:
        conn.close()


def get_progress(run_id: str) -> tuple[int | None, int] | None:
    """(last progress timestamp, tokens streamed) for one run, or None if unknown."""
    conn = _connect()
    try:
        try:
            row = conn.execute(
                """SELECT COALESCE(progress_at, started_at, created_at) AS progress_at,
                          COALESCE(progress_tokens, 0) AS progress_tokens
                   FROM chat_generation_runs WHERE id=?""",
                (run_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if not _missing_lease_columns(exc):
                raise
            row = conn.execute(
                """SELECT COALESCE(started_at, created_at) AS progress_at,
                          0 AS progress_tokens
                   FROM chat_generation_runs WHERE id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        progress_at = row["progress_at"]
        return (int(progress_at) if progress_at is not None else None, int(row["progress_tokens"]))
    finally:
        conn.close()


def list_active(thread_id: str) -> list[dict[str, Any]]:
    conn = _connect()
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
    conn = _connect()
    try:
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
        # The producer's only regular write, so it is also the lease renewal: output
        # reaching the database is the definition of progress this sweep reaps on.
        _touch_progress_locked(conn, run_id, sum(1 for event in batch if event[0] == "chunk"))
        _commit(conn, notify = True)
        return sequences
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_running(run_id: str, worker_token: str) -> bool:
    conn = _connect()
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
        _touch_progress_locked(conn, run_id, 0)
        _commit(conn, notify = True)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def request_cancel(run_id: str, owner_subject: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
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
    conn = _connect()
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
    conn = _connect()
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


def reconcile_runs(
    *, error: str = "Studio restarted during generation", stale_after_ms: int | None = None
) -> list[str]:
    """Settle active runs, returning the ids settled.

    ``stale_after_ms`` is what makes this safe to run while Studio is serving: with it,
    only runs whose progress lease has not moved for that long are settled, so a slow
    but advancing generation is never touched. Without it (process boot) every active
    run is orphaned by definition and all of them are settled.

    Partial output survives either way: only the run row and the assistant message's
    status metadata are rewritten, never the streamed content or the event log.
    """
    conn = _connect()
    settled: list[str] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        completed = now_ms()
        sql = """SELECT id, status, cancel_requested FROM chat_generation_runs
                 WHERE status IN ('queued','running','cancelling')"""
        args: tuple[Any, ...] = ()
        if stale_after_ms is not None:
            # started_at/created_at carry a run that has not streamed anything yet,
            # so a producer wedged before its first token still ages out.
            sql += " AND COALESCE(progress_at, started_at, created_at) <= ?"
            args = (completed - int(stale_after_ms),)
        try:
            rows = conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
        except sqlite3.OperationalError as exc:
            if not _missing_lease_columns(exc):
                raise
            # Contention blocked the migration, so no run can persist progress. Falling
            # back to started_at/created_at is not conservative but the opposite: those
            # stamps are older by the whole life of the run, so one that streamed a chunk
            # moments ago is reaped once its total AGE passes the timeout. Boot reconcile
            # passes no stale_after_ms and is unaffected.
            if stale_after_ms is not None:
                conn.rollback()
                return []
            rows = conn.execute(
                sql.replace(" AND COALESCE(progress_at, started_at, created_at) <= ?", "")
                + " ORDER BY created_at, id",
                (),
            ).fetchall()
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
            # Stamps incomplete: {reason: "interrupted"} on the assistant message, which
            # is what releases the frontend's "generating" state and restores Send.
            _sync_assistant_status_locked(conn, run_id, status)
            settled.append(str(run_id))
        _commit(conn, notify = bool(settled))
        return settled
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reconcile_orphaned_runs(error: str = "Studio restarted during generation") -> int:
    return len(reconcile_runs(error = error))
