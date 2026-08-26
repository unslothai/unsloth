# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable one-use confirmation state for Studio-prepared Git commits."""

import hashlib
import hmac
import json
import sqlite3
import threading
from typing import Optional

from storage.studio_db import get_connection

from .common import AgentWorkspaceError


_SCHEMA_LOCK = threading.Lock()
_READY_DATABASES: set[str] = set()
_MAX_PENDING_PREPARATIONS = 10_000
_STATUSES = frozenset({"awaiting_confirmation", "confirming", "confirmed", "failed", "expired"})


def _database_key(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return str(row[2])


def _connection() -> sqlite3.Connection:
    conn = get_connection()
    key = _database_key(conn)
    if key in _READY_DATABASES:
        return conn
    with _SCHEMA_LOCK:
        if key not in _READY_DATABASES:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_prepared_commits (
                    id TEXT NOT NULL PRIMARY KEY,
                    project_id TEXT NOT NULL
                        REFERENCES chat_projects(id) ON DELETE CASCADE,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    token_digest BLOB,
                    branch_ref TEXT NOT NULL,
                    head_sha TEXT NOT NULL,
                    git_root TEXT NOT NULL,
                    message TEXT NOT NULL,
                    owned_paths_json TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    ref_name TEXT NOT NULL,
                    commit_sha TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    confirmed_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_agent_prepared_commits_project
                    ON agent_prepared_commits(project_id, created_at DESC);
                """
            )
            conn.commit()
            _READY_DATABASES.add(key)
    return conn


def token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _record(row: sqlite3.Row) -> dict:
    try:
        owned_paths = json.loads(row["owned_paths_json"])
    except (TypeError, ValueError):
        owned_paths = None
    if not isinstance(owned_paths, list) or not all(isinstance(path, str) for path in owned_paths):
        raise AgentWorkspaceError("Prepared commit state is invalid.")
    status = str(row["status"])
    if status not in _STATUSES:
        raise AgentWorkspaceError("Prepared commit state is invalid.")
    return {
        "id": str(row["id"]),
        "projectId": str(row["project_id"]),
        "operation": str(row["operation"]),
        "status": status,
        "tokenDigest": row["token_digest"],
        "branchRef": str(row["branch_ref"]),
        "headSha": str(row["head_sha"]),
        "gitRoot": str(row["git_root"]),
        "message": str(row["message"]),
        "ownedPaths": owned_paths,
        "sourceFingerprint": str(row["source_fingerprint"]),
        "payloadDigest": str(row["payload_digest"]),
        "refName": str(row["ref_name"]),
        "commitSha": str(row["commit_sha"]) if row["commit_sha"] else None,
        "createdAt": int(row["created_at"]),
        "expiresAt": int(row["expires_at"]),
        "confirmedAt": (int(row["confirmed_at"]) if row["confirmed_at"] is not None else None),
    }


def save_preparation(record: dict, raw_token: str, *, now: int) -> None:
    conn = _connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            DELETE FROM agent_prepared_commits
            WHERE status IN ('awaiting_confirmation', 'failed', 'expired')
              AND expires_at < ?
            """,
            (now,),
        )
        pending = conn.execute(
            """
            SELECT COUNT(*) FROM agent_prepared_commits
            WHERE status IN ('awaiting_confirmation', 'confirming')
            """
        ).fetchone()[0]
        if int(pending) >= _MAX_PENDING_PREPARATIONS:
            raise AgentWorkspaceError("Too many prepared commits are awaiting confirmation.")
        conn.execute(
            """
            INSERT INTO agent_prepared_commits(
                id, project_id, operation, status, token_digest,
                branch_ref, head_sha, git_root, message, owned_paths_json,
                source_fingerprint, payload_digest, ref_name, commit_sha,
                created_at, expires_at, confirmed_at
            ) VALUES (?, ?, ?, 'awaiting_confirmation', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)
            """,
            (
                record["id"],
                record["projectId"],
                record["operation"],
                token_digest(raw_token),
                record["branchRef"],
                record["headSha"],
                record["gitRoot"],
                record["message"],
                json.dumps(record["ownedPaths"], separators = (",", ":")),
                record["sourceFingerprint"],
                record["payloadDigest"],
                record["refName"],
                record["createdAt"],
                record["expiresAt"],
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reserve_confirmation(preparation_id: str, project_id: str, raw_token: str, *, now: int) -> dict:
    conn = _connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_prepared_commits WHERE id = ?",
            (preparation_id,),
        ).fetchone()
        if row is None or str(row["project_id"]) != project_id:
            raise AgentWorkspaceError("Prepared commit not found.")
        record = _record(row)
        if record["operation"] != "prepare_commit":
            raise AgentWorkspaceError("Prepared commit confirmation is invalid.")
        if record["status"] != "awaiting_confirmation":
            raise AgentWorkspaceError(
                "Prepared commit confirmation was already used or is unavailable."
            )
        if record["expiresAt"] < now:
            conn.execute(
                """
                UPDATE agent_prepared_commits
                SET status = 'expired', token_digest = NULL WHERE id = ?
                """,
                (preparation_id,),
            )
            conn.commit()
            raise AgentWorkspaceError("Prepared commit confirmation expired.")
        persisted_digest = record["tokenDigest"]
        supplied_digest = token_digest(raw_token)
        if not isinstance(persisted_digest, bytes) or not hmac.compare_digest(
            persisted_digest, supplied_digest
        ):
            raise AgentWorkspaceError("Prepared commit confirmation token is invalid.")
        cursor = conn.execute(
            """
            UPDATE agent_prepared_commits
            SET status = 'confirming', token_digest = NULL
            WHERE id = ? AND status = 'awaiting_confirmation'
            """,
            (preparation_id,),
        )
        if cursor.rowcount != 1:
            raise AgentWorkspaceError(
                "Prepared commit confirmation was already used or is unavailable."
            )
        conn.commit()
        record["status"] = "confirming"
        record["tokenDigest"] = None
        return record
    except AgentWorkspaceError:
        if conn.in_transaction:
            conn.rollback()
        raise
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def mark_confirmed(preparation_id: str, commit_sha: str, *, now: int) -> None:
    conn = _connection()
    try:
        cursor = conn.execute(
            """
            UPDATE agent_prepared_commits
            SET status = 'confirmed', commit_sha = ?, confirmed_at = ?
            WHERE id = ? AND status = 'confirming'
            """,
            (commit_sha, now, preparation_id),
        )
        if cursor.rowcount != 1:
            raise AgentWorkspaceError("Prepared commit state changed during confirmation.")
        conn.commit()
    finally:
        conn.close()


def save_candidate_commit(preparation_id: str, commit_sha: str) -> None:
    conn = _connection()
    try:
        cursor = conn.execute(
            """
            UPDATE agent_prepared_commits SET commit_sha = ?
            WHERE id = ? AND status = 'confirming' AND commit_sha IS NULL
            """,
            (commit_sha, preparation_id),
        )
        if cursor.rowcount != 1:
            raise AgentWorkspaceError("Prepared commit state changed during confirmation.")
        conn.commit()
    finally:
        conn.close()


def mark_failed(preparation_id: str) -> None:
    conn = _connection()
    try:
        conn.execute(
            """
            UPDATE agent_prepared_commits SET status = 'failed', token_digest = NULL
            WHERE id = ? AND status = 'confirming'
            """,
            (preparation_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_preparation(preparation_id: str) -> Optional[dict]:
    conn = _connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_prepared_commits WHERE id = ?", (preparation_id,)
        ).fetchone()
        return _record(row) if row is not None else None
    finally:
        conn.close()


def list_ref_bearing_preparations(project_id: str) -> list[dict]:
    conn = _connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_prepared_commits
            WHERE project_id = ?
              AND commit_sha IS NOT NULL
              AND status IN ('confirming', 'confirmed')
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
        return [_record(row) for row in rows]
    finally:
        conn.close()


def delete_preparation(preparation_id: str, project_id: str) -> None:
    conn = _connection()
    try:
        conn.execute(
            "DELETE FROM agent_prepared_commits WHERE id = ? AND project_id = ?",
            (preparation_id, project_id),
        )
        conn.commit()
    finally:
        conn.close()
