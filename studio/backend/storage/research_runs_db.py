# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transactional durable state for inline Deep Research runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any

from core.inference.web_access_policy import check_url_access
from storage.studio_db import get_connection

ACTIVE_STATUSES = frozenset(
    {"planning", "awaiting_approval", "queued", "running", "paused", "cancelling"}
)
TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
_EVENTS_CHANGED = threading.Condition()


class ResearchConflictError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical_plan(plan: dict[str, Any]) -> tuple[str, str]:
    raw = json.dumps(plan, sort_keys = True, separators = (",", ":"), ensure_ascii = False)
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _event_locked(conn: sqlite3.Connection, run_id: str, event_type: str, data: dict) -> int:
    row = conn.execute(
        "SELECT next_event_seq, retry_count FROM research_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    seq = int(row["next_event_seq"])
    created = now_ms()
    event_data = dict(data)
    event_data.setdefault("attempt", int(row["retry_count"]))
    conn.execute(
        "INSERT INTO research_events (run_id, seq, event_type, data_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, seq, event_type, json.dumps(event_data, ensure_ascii = False), created),
    )
    conn.execute(
        "UPDATE research_runs SET next_event_seq = ?, updated_at = ? WHERE id = ?",
        (seq + 1, created, run_id),
    )
    return seq


def _commit_event(conn: sqlite3.Connection) -> None:
    conn.commit()
    with _EVENTS_CHANGED:
        _EVENTS_CHANGED.notify_all()


def _worker_can_write_locked(
    conn: sqlite3.Connection, run_id: str, worker_id: str, statuses: set[str],
) -> bool:
    row = conn.execute(
        "SELECT status, lease_owner, lease_expires_at, cancel_requested "
        "FROM research_runs WHERE id = ?", (run_id,),
    ).fetchone()
    return bool(
        row is not None
        and row["lease_owner"] == worker_id
        and row["status"] in statuses
        and not bool(row["cancel_requested"])
        and row["lease_expires_at"] is not None
        and int(row["lease_expires_at"]) >= now_ms()
    )


def append_event(run_id: str, event_type: str, data: dict[str, Any]) -> int:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        seq = _event_locked(conn, run_id, event_type, data)
        _commit_event(conn)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_worker_event(
    run_id: str, worker_id: str, event_type: str, data: dict[str, Any],
) -> int | None:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not _worker_can_write_locked(
            conn, run_id, worker_id, {"planning", "running"},
        ):
            conn.commit()
            return None
        seq = _event_locked(conn, run_id, event_type, data)
        _commit_event(conn)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_run(
    *, run_id: str, owner_subject: str, thread_id: str, user_message_id: str,
    assistant_message_id: str | None, config: dict[str, Any], created_at: int | None = None,
) -> dict:
    created = created_at or now_ms()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO research_thread_claims (owner_subject, thread_id, created_at) "
                "VALUES (?, ?, ?)",
                (owner_subject, thread_id, created),
            )
        except sqlite3.IntegrityError as exc:
            claim = conn.execute(
                "SELECT 1 FROM research_thread_claims "
                "WHERE owner_subject=? AND thread_id=?",
                (owner_subject, thread_id),
            ).fetchone()
            if claim is not None:
                raise ResearchConflictError(
                    "This thread already has a Deep Research run"
                ) from exc
            raise
        if assistant_message_id:
            message = conn.execute(
                "SELECT * FROM chat_messages WHERE id=?", (assistant_message_id,)
            ).fetchone()
            metadata = {
                "researchRunId": run_id, "researchStatus": "planning",
                "researchPlanRevision": 0, "serverManaged": True,
            }
            if message is None:
                conn.execute(
                    """INSERT INTO chat_messages
                       (id, thread_id, parent_id, role, content_json, metadata_json, created_at)
                       VALUES (?, ?, ?, 'assistant', '[]', ?, ?)""",
                    (assistant_message_id, thread_id, user_message_id,
                     json.dumps(metadata, ensure_ascii = False), created),
                )
                conn.execute(
                    "UPDATE chat_threads SET updated_at=MAX(COALESCE(updated_at, created_at), ?) "
                    "WHERE id=?",
                    (created, thread_id),
                )
            else:
                existing_metadata = _loads(message["metadata_json"], {})
                existing_run_id = (
                    existing_metadata.get("researchRunId")
                    if isinstance(existing_metadata, dict) else None
                )
                if (
                    message["thread_id"] != thread_id
                    or message["role"] != "assistant"
                    or message["parent_id"] != user_message_id
                    or existing_run_id not in (None, run_id)
                ):
                    raise ResearchConflictError(
                        "Assistant message does not match this research run"
                    )
                merged_metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
                merged_metadata.update(metadata)
                conn.execute(
                    "UPDATE chat_messages SET metadata_json=? WHERE id=?",
                    (json.dumps(merged_metadata, ensure_ascii = False), assistant_message_id),
                )
        conn.execute(
            """
            INSERT INTO research_runs
                (id, owner_subject, thread_id, user_message_id, assistant_message_id,
                 status, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'planning', ?, ?, ?)
            """,
            (run_id, owner_subject, thread_id, user_message_id, assistant_message_id,
             json.dumps(config, ensure_ascii = False), created, created),
        )
        _event_locked(conn, run_id, "run.created", {"status": "planning"})
        _commit_event(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_run(run_id, owner_subject)


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": data["id"], "ownerSubject": data["owner_subject"],
        "threadId": data["thread_id"], "userMessageId": data["user_message_id"],
        "assistantMessageId": data["assistant_message_id"], "status": data["status"],
        "plan": _loads(data["plan_json"], None), "planRevision": data["plan_revision"],
        "planHash": data["plan_hash"], "config": _loads(data["config_json"], {}),
        "cancelRequested": bool(data["cancel_requested"]), "retryCount": data["retry_count"],
        "error": data["error_message"], "report": data.get("report_text"),
        "createdAt": data["created_at"],
        "updatedAt": data["updated_at"], "startedAt": data["started_at"],
        "completedAt": data["completed_at"], "heartbeatAt": data["heartbeat_at"],
        "lastEventSeq": int(data["next_event_seq"]) - 1,
    }


def get_run(run_id: str, owner_subject: str | None = None) -> dict | None:
    conn = get_connection()
    try:
        sql = "SELECT * FROM research_runs WHERE id = ?"
        args: tuple = (run_id,)
        if owner_subject is not None:
            sql += " AND owner_subject = ?"
            args += (owner_subject,)
        row = conn.execute(sql, args).fetchone()
        if row is None:
            return None
        result = _row_to_run(row)
        result["steps"] = [dict(r) for r in conn.execute(
            "SELECT position, title, query, status, result_json AS resultJson, "
            "started_at AS startedAt, completed_at AS completedAt FROM research_plan_steps "
            "WHERE run_id = ? ORDER BY position", (run_id,)
        ).fetchall()]
        for step in result["steps"]:
            step["result"] = _loads(step.pop("resultJson"), None)
            step["input"] = step["query"]
        result["sources"] = [dict(r) for r in conn.execute(
            "SELECT id, step_position AS stepPosition, url, title, snippet, "
            "fetched_at AS fetchedAt FROM research_sources WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()]
        return result
    finally:
        conn.close()


def list_active(owner_subject: str, thread_id: str) -> list[dict]:
    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        rows = conn.execute(
            f"SELECT id FROM research_runs WHERE owner_subject = ? AND thread_id = ? "
            f"AND status IN ({placeholders}) ORDER BY created_at",
            (owner_subject, thread_id, *sorted(ACTIVE_STATUSES)),
        ).fetchall()
    finally:
        conn.close()
    return [run for row in rows if (run := get_run(row["id"], owner_subject)) is not None]


def has_thread_claim(owner_subject: str, thread_id: str) -> bool:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT 1 FROM research_thread_claims "
            "WHERE owner_subject=? AND thread_id=?",
            (owner_subject, thread_id),
        ).fetchone() is not None
    finally:
        conn.close()


def _discover_assistant_locked(conn: sqlite3.Connection, run: sqlite3.Row) -> str | None:
    bound_id = run["assistant_message_id"]
    if bound_id:
        bound = conn.execute(
            "SELECT id FROM chat_messages WHERE id=? AND thread_id=? AND role='assistant'",
            (bound_id, run["thread_id"]),
        ).fetchone()
        if bound is not None:
            return str(bound["id"])
    rows = conn.execute(
        """SELECT id, metadata_json FROM chat_messages
           WHERE thread_id=? AND parent_id=? AND role='assistant' ORDER BY created_at, id""",
        (run["thread_id"], run["user_message_id"]),
    ).fetchall()
    for message in rows:
        metadata = _loads(message["metadata_json"], {})
        if isinstance(metadata, dict) and metadata.get("researchRunId") == run["id"]:
            message_id = str(message["id"])
            conn.execute(
                "UPDATE research_runs SET assistant_message_id=?, updated_at=? WHERE id=?",
                (message_id, now_ms(), run["id"]),
            )
            return message_id
    return None


def discover_and_bind_assistant_message(run_id: str) -> str | None:
    """Atomically bind the assistant-ui child carrying this run's metadata."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute("SELECT * FROM research_runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(run_id)
        message_id = _discover_assistant_locked(conn, run)
        _commit_event(conn)
        return message_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_and_bind_terminal_fallback(
    run_id: str, *, text: str, status: str, sources: list[dict] | None = None,
    completion_worker_id: str | None = None,
) -> tuple[str, bool]:
    """Discover a frontend message or atomically create exactly one fallback."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(status)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute("SELECT * FROM research_runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(run_id)
        can_prepare_completion = (
            completion_worker_id is not None
            and status == "completed"
            and run["status"] == "running"
            and run["lease_owner"] == completion_worker_id
            and run["lease_expires_at"] is not None
            and int(run["lease_expires_at"]) >= now_ms()
            and not bool(run["cancel_requested"])
        )
        if run["status"] != status and not can_prepare_completion:
            raise ResearchConflictError(
                f"Cannot create a {status} fallback for a {run['status']} run"
            )
        message_id = _discover_assistant_locked(conn, run)
        if message_id is not None:
            conn.commit()
            return message_id, False

        message_id = f"research-{run_id}"
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": text, "researchRunId": run_id}
        ]
        for source in sources or []:
            parts.append({
                "type": "source", "sourceType": "url", "id": source["url"],
                "url": source["url"], "title": source.get("title") or source["url"],
                "metadata": {"description": source.get("snippet") or ""},
                "researchRunId": run_id,
            })
        metadata = {
            "researchRunId": run_id, "researchStatus": status,
            "researchPlanRevision": int(run["plan_revision"]), "serverManaged": True,
        }
        created = now_ms()
        conn.execute(
            """INSERT INTO chat_messages
               (id, thread_id, parent_id, role, content_json, metadata_json, created_at)
               VALUES (?, ?, ?, 'assistant', ?, ?, ?)""",
            (message_id, run["thread_id"], run["user_message_id"],
             json.dumps(parts, ensure_ascii = False),
             json.dumps(metadata, ensure_ascii = False), created),
        )
        conn.execute(
            "UPDATE research_runs SET assistant_message_id=?, updated_at=? WHERE id=?",
            (message_id, created, run_id),
        )
        conn.execute(
            "UPDATE chat_threads SET updated_at=MAX(COALESCE(updated_at, created_at), ?) WHERE id=?",
            (created, run["thread_id"]),
        )
        _commit_event(conn)
        return message_id, True
    except sqlite3.IntegrityError:
        conn.rollback()
        # A concurrent terminal path may have inserted the deterministic fallback.
        message_id = discover_and_bind_assistant_message(run_id)
        if message_id is None:
            raise
        return message_id, False
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_plan(
    run_id: str, plan: dict, expected_revision: int | None = None,
    worker_id: str | None = None,
) -> dict:
    raw, digest = canonical_plan(plan)
    steps = plan.get("steps") or []
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, plan_revision, lease_owner, lease_expires_at, cancel_requested "
            "FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if worker_id is not None and (
            row["status"] != "planning"
            or row["lease_owner"] != worker_id
            or row["lease_expires_at"] is None
            or int(row["lease_expires_at"]) < now_ms()
            or bool(row["cancel_requested"])
        ):
            raise ResearchConflictError("Planner no longer owns this research run")
        if worker_id is None and row["status"] not in {"planning", "awaiting_approval"}:
            raise ResearchConflictError("Plan can only be changed before approval")
        revision = int(row["plan_revision"])
        if expected_revision is not None and revision != expected_revision:
            raise ResearchConflictError(f"Plan revision is {revision}, not {expected_revision}")
        revision += 1
        conn.execute(
            "UPDATE research_runs SET plan_json = ?, plan_revision = ?, plan_hash = ?, "
            "status = 'awaiting_approval', error_message = NULL, lease_owner = NULL, "
            "lease_expires_at = NULL, updated_at = ? WHERE id = ?",
            (raw, revision, digest, now_ms(), run_id),
        )
        conn.execute("DELETE FROM research_plan_steps WHERE run_id = ?", (run_id,))
        conn.executemany(
            "INSERT INTO research_plan_steps (run_id, position, title, query) VALUES (?, ?, ?, ?)",
            [(run_id, i, str(s["title"]), str(s.get("query") or s["title"]))
             for i, s in enumerate(steps)],
        )
        _event_locked(conn, run_id, "plan.ready", {
            "status": "awaiting_approval", "plan": plan,
            "planRevision": revision, "planHash": digest,
        })
        _commit_event(conn)
        return {"plan": plan, "planRevision": revision, "planHash": digest}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def approve(run_id: str, revision: int, plan_hash: str) -> str:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, plan_revision, plan_hash FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if int(row["plan_revision"]) != revision or row["plan_hash"] != plan_hash:
            raise ResearchConflictError("Plan revision or hash no longer matches")
        if row["status"] in {"queued", "running", "completed"}:
            conn.commit()
            return row["status"]
        if row["status"] != "awaiting_approval":
            raise ResearchConflictError(f"Cannot approve a {row['status']} run")
        conn.execute(
            "UPDATE research_runs SET status = 'queued', updated_at = ? WHERE id = ?",
            (now_ms(), run_id),
        )
        _event_locked(conn, run_id, "run.approved", {"status": "queued"})
        _commit_event(conn)
        return "queued"
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def request_cancel(run_id: str) -> str:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM research_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        status = row["status"]
        if status in TERMINAL_STATUSES or status == "cancelling":
            conn.commit()
            return status
        new_status = "cancelled" if status in {"awaiting_approval", "queued", "paused"} else "cancelling"
        completed = now_ms() if new_status == "cancelled" else None
        conn.execute(
            "UPDATE research_runs SET cancel_requested = 1, status = ?, completed_at = ?, "
            "updated_at = ? WHERE id = ?", (new_status, completed, now_ms(), run_id),
        )
        event_type = "run.cancelled" if new_status == "cancelled" else "run.cancelRequested"
        _event_locked(conn, run_id, event_type, {"status": new_status})
        _commit_event(conn)
        return new_status
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retry(run_id: str, max_retries: int = 3) -> str:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, retry_count, plan_json, owner_subject, thread_id "
            "FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] not in {"failed", "cancelled"}:
            raise ResearchConflictError("Only failed or cancelled runs can be retried")
        if int(row["retry_count"]) >= max_retries:
            raise ResearchConflictError("Retry budget exhausted")
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        active = conn.execute(
            f"SELECT id FROM research_runs WHERE owner_subject=? AND thread_id=? AND id<>? "
            f"AND status IN ({placeholders}) LIMIT 1",
            (row["owner_subject"], row["thread_id"], run_id, *sorted(ACTIVE_STATUSES)),
        ).fetchone()
        if active is not None:
            raise ResearchConflictError("This thread already has an active research run")
        plan_was_approved = False
        if row["plan_json"]:
            plan_was_approved = conn.execute(
                "SELECT 1 FROM research_events WHERE run_id=? AND event_type='run.approved' LIMIT 1",
                (run_id,),
            ).fetchone() is not None
        status = (
            "queued" if plan_was_approved
            else "awaiting_approval" if row["plan_json"]
            else "planning"
        )
        conn.execute(
            "UPDATE research_runs SET status = ?, cancel_requested = 0, retry_count = retry_count + 1, "
            "error_message = NULL, report_text = NULL, completed_at = NULL, lease_owner = NULL, "
            "lease_expires_at = NULL, updated_at = ? WHERE id = ?", (status, now_ms(), run_id),
        )
        if status != "awaiting_approval":
            conn.execute("DELETE FROM research_plan_steps WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM research_sources WHERE run_id = ?", (run_id,))
        _event_locked(conn, run_id, "run.retried", {"status": status})
        _commit_event(conn)
        return status
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def claim_next(worker_id: str, lease_ms: int = 120_000) -> dict | None:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = now_ms()
        row = conn.execute(
            """SELECT * FROM research_runs
               WHERE status IN ('planning','queued','running','cancelling')
                 AND (lease_owner IS NULL OR lease_expires_at < ?)
               ORDER BY created_at LIMIT 1""", (now,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        status = row["status"]
        next_status = (
            "running" if status in {"queued", "running"}
            else "cancelling" if status == "cancelling"
            else "planning"
        )
        conn.execute(
            "UPDATE research_runs SET status=?, lease_owner=?, lease_expires_at=?, heartbeat_at=?, "
            "started_at=COALESCE(started_at, ?), updated_at=? WHERE id=?",
            (next_status, worker_id, now + lease_ms, now, now, now, row["id"]),
        )
        _event_locked(conn, row["id"], "run.started", {"status": next_status})
        _commit_event(conn)
        return get_run(row["id"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def heartbeat(run_id: str, worker_id: str, lease_ms: int = 120_000) -> bool:
    conn = get_connection()
    try:
        now = now_ms()
        cur = conn.execute(
            "UPDATE research_runs SET heartbeat_at=?, lease_expires_at=? "
            "WHERE id=? AND lease_owner=? AND lease_expires_at>=?",
            (now, now + lease_ms, run_id, worker_id, now),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def is_cancel_requested(run_id: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT cancel_requested FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return row is None or bool(row[0])
    finally:
        conn.close()


def finish(
    run_id: str, worker_id: str, status: str, error: str | None = None,
    event_payload: dict[str, Any] | None = None, allow_expired: bool = False,
) -> str | None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(status)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = now_ms()
        row = conn.execute(
            "SELECT status, cancel_requested, lease_expires_at "
            "FROM research_runs WHERE id=? AND lease_owner=?",
            (run_id, worker_id),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        if (
            not allow_expired
            and not bool(row["cancel_requested"])
            and (row["lease_expires_at"] is None or int(row["lease_expires_at"]) < now)
        ):
            conn.commit()
            return None
        actual_status = (
            "cancelled"
            if bool(row["cancel_requested"]) or row["status"] == "cancelling"
            else status
        )
        actual_error = None if actual_status == "cancelled" else error
        report_text = None
        if actual_status == "completed" and event_payload:
            candidate = event_payload.get("report")
            if isinstance(candidate, str):
                report_text = candidate
        conn.execute(
            "UPDATE research_runs SET status=?, error_message=?, report_text=?, completed_at=?, updated_at=?, "
            "lease_owner=NULL, lease_expires_at=NULL WHERE id=? AND lease_owner=?",
            (actual_status, actual_error, report_text, now, now, run_id, worker_id),
        )
        payload = {"status": actual_status, "error": actual_error}
        if event_payload and actual_status == status:
            payload.update(event_payload)
        _event_locked(conn, run_id, f"run.{actual_status}", payload)
        _commit_event(conn)
        return actual_status
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_report_progress(
    run_id: str, report: str, delta: str | None = None,
    worker_id: str | None = None,
) -> bool:
    """Persist partial report text and notify followers while synthesis runs."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, lease_owner, lease_expires_at, cancel_requested "
            "FROM research_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "running"
            or worker_id is not None and (
                row["lease_owner"] != worker_id or bool(row["cancel_requested"])
                or row["lease_expires_at"] is None
                or int(row["lease_expires_at"]) < now_ms()
            )
        ):
            conn.commit()
            return False
        now = now_ms()
        conn.execute(
            "UPDATE research_runs SET report_text = ?, updated_at = ? WHERE id = ?",
            (report, now, run_id),
        )
        event_data: dict[str, Any] = {"length": len(report)}
        if delta:
            event_data.update({"delta": delta, "offset": len(report) - len(delta)})
        _event_locked(conn, run_id, "report.updated", event_data)
        _commit_event(conn)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_step(run_id: str, position: int, status: str, result: Any = None) -> None:
    conn = get_connection()
    try:
        now = now_ms()
        conn.execute(
            "UPDATE research_plan_steps SET status=?, result_json=?, "
            "started_at=CASE WHEN ?='running' THEN COALESCE(started_at, ?) ELSE started_at END, "
            "completed_at=CASE WHEN ? IN ('completed','failed') THEN ? ELSE completed_at END "
            "WHERE run_id=? AND position=?",
            (status, json.dumps(result, ensure_ascii = False) if result is not None else None,
             status, now, status, now, run_id, position),
        )
        conn.commit()
    finally:
        conn.close()


def reset_execution_steps(run_id: str, worker_id: str | None = None) -> bool:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if worker_id is not None and not _worker_can_write_locked(
            conn, run_id, worker_id, {"running"},
        ):
            conn.commit()
            return False
        conn.execute("DELETE FROM research_plan_steps WHERE run_id = ?", (run_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_execution_step(
    run_id: str, position: int, title: str, query: str, status: str,
    result: Any = None, worker_id: str | None = None,
) -> bool:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if worker_id is not None and not _worker_can_write_locked(
            conn, run_id, worker_id, {"running"},
        ):
            conn.commit()
            return False
        now = now_ms()
        conn.execute(
            """INSERT INTO research_plan_steps
               (run_id, position, title, query, status, result_json, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, position) DO UPDATE SET
                 title=excluded.title, query=excluded.query, status=excluded.status,
                 result_json=excluded.result_json,
                 started_at=COALESCE(research_plan_steps.started_at, excluded.started_at),
                 completed_at=excluded.completed_at""",
            (
                run_id, position, title[:200], query[:500], status,
                json.dumps(result, ensure_ascii = False) if result is not None else None,
                now, now if status in {"completed", "failed"} else None,
            ),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_reasoning_text(run_id: str) -> str:
    conn = get_connection()
    try:
        run = conn.execute(
            "SELECT retry_count FROM research_runs WHERE id=?", (run_id,)
        ).fetchone()
        if run is None:
            return ""
        attempt = int(run["retry_count"])
        rows = conn.execute(
            "SELECT data_json FROM research_events WHERE run_id=? "
            "AND event_type='reasoning.updated' ORDER BY seq",
            (run_id,),
        ).fetchall()
        return "".join(
            str(data.get("reasoningDelta") or "")
            for row in rows
            if int((data := _loads(row["data_json"], {})).get("attempt", 0)) == attempt
        )
    finally:
        conn.close()


def upsert_source(
    run_id: str, position: int, url: str, title: str, snippet: str,
    worker_id: str | None = None,
) -> bool:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if worker_id is not None and not _worker_can_write_locked(
            conn, run_id, worker_id, {"running"},
        ):
            conn.commit()
            return False
        run = conn.execute(
            "SELECT config_json FROM research_runs WHERE id=?", (run_id,),
        ).fetchone()
        if run is None:
            conn.commit()
            return False
        config = _loads(run["config_json"], {})
        allowed, reason, _hostname = check_url_access(
            url, config.get("websitePolicy") if isinstance(config, dict) else None,
        )
        if not allowed:
            raise ValueError(reason)
        fetched_at = now_ms()
        conn.execute(
            """INSERT INTO research_sources (run_id, step_position, url, title, snippet, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, url) DO UPDATE SET step_position=excluded.step_position,
               title=excluded.title,
               snippet=excluded.snippet, fetched_at=excluded.fetched_at""",
            (run_id, position, url, title[:500], snippet[:4000], fetched_at),
        )
        _event_locked(conn, run_id, "source.added", {
            "position": position, "stepPosition": position, "url": url,
            "title": title[:500], "snippet": snippet[:4000], "fetchedAt": fetched_at,
        })
        _commit_event(conn)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_events(run_id: str, owner_subject: str, after: int = 0, limit: int = 1000) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT e.seq, e.event_type, e.data_json, e.created_at
               FROM research_events e JOIN research_runs r ON r.id=e.run_id
               WHERE e.run_id=? AND r.owner_subject=? AND e.seq>? ORDER BY e.seq LIMIT ?""",
            (run_id, owner_subject, after, limit),
        ).fetchall()
        return [{"seq": r["seq"], "type": r["event_type"],
                 "data": _loads(r["data_json"], {}), "createdAt": r["created_at"]} for r in rows]
    finally:
        conn.close()


def wait_for_events(
    run_id: str, owner_subject: str, after: int = 0, timeout: float = 15,
) -> list[dict]:
    """Block until committed events are available or the keep-alive timeout expires."""
    events = list_events(run_id, owner_subject, after)
    if events:
        return events
    with _EVENTS_CHANGED:
        # Recheck under the condition lock so a commit cannot be missed between
        # the initial query and waiting for its notification.
        events = list_events(run_id, owner_subject, after)
        if events:
            return events
        _EVENTS_CHANGED.wait(timeout)
    return list_events(run_id, owner_subject, after)


def recover_expired(now: int | None = None) -> int:
    conn = get_connection()
    try:
        now = now or now_ms()
        cur = conn.execute(
            """UPDATE research_runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE status IN ('planning','queued','running','cancelling')
                 AND lease_owner IS NOT NULL AND lease_expires_at < ?""", (now, now),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def owns_lease(run_id: str, worker_id: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM research_runs WHERE id=? AND lease_owner=? AND lease_expires_at>=?",
            (run_id, worker_id, now_ms()),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def release_worker_leases(worker_id: str) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            """UPDATE research_runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE lease_owner=? AND status IN ('planning','queued','running','cancelling')""",
            (now_ms(), worker_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
