# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small durable store for Studio-owned worktree lifecycle records.

Worktree ownership, checkpoint refs, and project retirement stay durable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional

from storage.studio_db import get_connection

from .git_context import AgentWorkspaceError


_STATE_LOCK = threading.Lock()
_READY_DATABASES: set[str] = set()
_WORKTREE_STATUSES = frozenset({"creating", "active", "removing", "removed", "needs_attention"})
_MERGE_RECORD_LIMIT = 64 * 1024
_CHECKPOINT_PATHS_LIMIT = 256 * 1024


def _now_ms() -> int:
    return int(time.time() * 1000)


def _database_key(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return str(row[2])


def _ensure_schema(conn: sqlite3.Connection) -> None:
    key = _database_key(conn)
    if key in _READY_DATABASES:
        return
    with _STATE_LOCK:
        if key in _READY_DATABASES:
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_worktrees (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                git_root TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                branch TEXT NOT NULL,
                base_ref TEXT NOT NULL,
                marker_path TEXT NOT NULL,
                marker_token_hash TEXT NOT NULL,
                background_task_id TEXT,
                status TEXT NOT NULL,
                merge_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_worktrees_project "
            "ON agent_worktrees(project_id, created_at, id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_git_checkpoints (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                git_root TEXT NOT NULL,
                ref_name TEXT NOT NULL UNIQUE,
                commit_sha TEXT NOT NULL,
                owned_paths_json TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_git_checkpoints_project "
            "ON agent_git_checkpoints(project_id, created_at, id)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_git_retirement ("
            "project_id TEXT PRIMARY KEY REFERENCES chat_projects(id) ON DELETE CASCADE, "
            "retired INTEGER NOT NULL DEFAULT 0)"
        )
        conn.commit()
        _READY_DATABASES.add(key)


def connection() -> sqlite3.Connection:
    conn = get_connection()
    _ensure_schema(conn)
    return conn


def _json(value: Optional[str]) -> object:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _record(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "gitRoot": row["git_root"],
        "path": row["path"],
        "branch": row["branch"],
        "baseRef": row["base_ref"],
        "markerPath": row["marker_path"],
        "markerTokenHash": row["marker_token_hash"],
        "backgroundTaskId": row["background_task_id"],
        "status": row["status"],
        "merge": _json(row["merge_json"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def save_worktree(record: dict) -> None:
    status = str(record.get("status") or "")
    if status not in _WORKTREE_STATUSES:
        raise AgentWorkspaceError("Invalid worktree status.")
    merge = record.get("merge")
    merge_json = None
    if merge is not None:
        try:
            merge_json = json.dumps(merge, separators = (",", ":"))
        except (TypeError, ValueError) as exc:
            raise AgentWorkspaceError("Worktree merge record is not valid JSON.") from exc
        if len(merge_json.encode("utf-8")) > _MERGE_RECORD_LIMIT:
            raise AgentWorkspaceError("Worktree merge record is too large.")
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_worktrees(
                id, project_id, git_root, path, branch, base_ref,
                marker_path, marker_token_hash, background_task_id,
                status, merge_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record["id"]),
                str(record["projectId"]),
                str(record["gitRoot"]),
                str(record["path"]),
                str(record["branch"]),
                str(record["baseRef"]),
                str(record["markerPath"]),
                str(record["markerTokenHash"]),
                record.get("backgroundTaskId"),
                status,
                merge_json,
                int(record["createdAt"]),
                int(record["updatedAt"]),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise AgentWorkspaceError("Worktree durable state already exists or is invalid.") from exc
    finally:
        conn.close()


def get_worktree(worktree_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_worktrees WHERE id = ?", (str(worktree_id),)
        ).fetchone()
        return _record(row) if row is not None else None
    finally:
        conn.close()


def list_worktrees(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_worktrees WHERE project_id = ? ORDER BY created_at, id",
            (str(project_id),),
        ).fetchall()
        return [_record(row) for row in rows]
    finally:
        conn.close()


def list_all_worktrees(limit: int = 4096) -> list[dict]:
    bounded = max(1, min(int(limit), 16_384))
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_worktrees ORDER BY created_at, id LIMIT ?", (bounded,)
        ).fetchall()
        return [_record(row) for row in rows]
    finally:
        conn.close()


def list_active_worktrees(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_worktrees
            WHERE project_id = ? AND status != 'removed'
            ORDER BY created_at, id
            """,
            (str(project_id),),
        ).fetchall()
        return [_record(row) for row in rows]
    finally:
        conn.close()


def transition_worktree_status(
    worktree_id: str, expected_statuses: set[str] | frozenset[str], status: str
) -> Optional[dict]:
    expected = sorted(set(expected_statuses))
    if not expected or any(value not in _WORKTREE_STATUSES for value in expected):
        raise AgentWorkspaceError("Invalid expected worktree status.")
    if status not in _WORKTREE_STATUSES:
        raise AgentWorkspaceError("Invalid worktree status.")
    placeholders = ", ".join("?" for _ in expected)
    conn = connection()
    try:
        cursor = conn.execute(
            f"""
            UPDATE agent_worktrees SET status = ?, updated_at = ?
            WHERE id = ? AND status IN ({placeholders})
            """,
            (status, _now_ms(), str(worktree_id), *expected),
        )
        conn.commit()
        if cursor.rowcount == 0:
            row = conn.execute(
                "SELECT status FROM agent_worktrees WHERE id = ?", (str(worktree_id),)
            ).fetchone()
            if row is None:
                return None
            raise AgentWorkspaceError(
                "Worktree state changed while the operation was running. Refresh and try again."
            )
        row = conn.execute(
            "SELECT * FROM agent_worktrees WHERE id = ?", (str(worktree_id),)
        ).fetchone()
        return _record(row) if row is not None else None
    finally:
        conn.close()


def record_worktree_merge(worktree_id: str, merge: dict) -> dict:
    try:
        encoded = json.dumps(merge, separators = (",", ":"))
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Worktree merge record is not valid JSON.") from exc
    if len(encoded.encode("utf-8")) > _MERGE_RECORD_LIMIT:
        raise AgentWorkspaceError("Worktree merge record is too large.")
    conn = connection()
    try:
        cursor = conn.execute(
            "UPDATE agent_worktrees SET merge_json = ?, updated_at = ? WHERE id = ?",
            (encoded, _now_ms(), str(worktree_id)),
        )
        conn.commit()
        if not cursor.rowcount:
            raise AgentWorkspaceError("Studio worktree not found.")
    finally:
        conn.close()
    result = get_worktree(worktree_id)
    if result is None:
        raise AgentWorkspaceError("Studio worktree not found.")
    return result


def save_checkpoint(record: dict) -> None:
    try:
        paths = json.dumps(record["ownedPaths"], separators = (",", ":"))
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Checkpoint paths are not valid JSON.") from exc
    if len(paths.encode("utf-8")) > _CHECKPOINT_PATHS_LIMIT:
        raise AgentWorkspaceError("Checkpoint paths are too large.")
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_git_checkpoints(
                id, project_id, git_root, ref_name, commit_sha,
                owned_paths_json, source_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record["id"]),
                str(record["projectId"]),
                str(record["gitRoot"]),
                str(record["refName"]),
                str(record["commitSha"]),
                paths,
                str(record["sourceFingerprint"]),
                int(record["createdAt"]),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise AgentWorkspaceError("Checkpoint durable state already exists or is invalid.") from exc
    finally:
        conn.close()


def _checkpoint(row: sqlite3.Row) -> dict:
    try:
        paths = json.loads(row["owned_paths_json"])
    except (TypeError, ValueError):
        paths = []
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "gitRoot": row["git_root"],
        "refName": row["ref_name"],
        "commitSha": row["commit_sha"],
        "ownedPaths": paths if isinstance(paths, list) else [],
        "sourceFingerprint": row["source_fingerprint"],
        "createdAt": row["created_at"],
    }


def get_checkpoint(checkpoint_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_git_checkpoints WHERE id = ?", (str(checkpoint_id),)
        ).fetchone()
        return _checkpoint(row) if row is not None else None
    finally:
        conn.close()


def list_checkpoints(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_git_checkpoints WHERE project_id = ? ORDER BY created_at, id",
            (str(project_id),),
        ).fetchall()
        return [_checkpoint(row) for row in rows]
    finally:
        conn.close()


def list_all_checkpoints(limit: int = 4096) -> list[dict]:
    bounded = max(1, min(int(limit), 16_384))
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_git_checkpoints ORDER BY created_at, id LIMIT ?",
            (bounded,),
        ).fetchall()
        return [_checkpoint(row) for row in rows]
    finally:
        conn.close()


def delete_checkpoint(checkpoint_id: str, project_id: str) -> bool:
    conn = connection()
    try:
        cursor = conn.execute(
            "DELETE FROM agent_git_checkpoints WHERE id = ? AND project_id = ?",
            (str(checkpoint_id), str(project_id)),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


__all__ = [
    "connection",
    "delete_checkpoint",
    "get_checkpoint",
    "get_worktree",
    "list_active_worktrees",
    "list_all_worktrees",
    "list_worktrees",
    "list_checkpoints",
    "list_all_checkpoints",
    "record_worktree_merge",
    "save_worktree",
    "save_checkpoint",
    "transition_worktree_status",
]


def require_git_admission(project_id: str) -> None:
    conn = connection()
    try:
        project = conn.execute(
            "SELECT archived FROM chat_projects WHERE id = ?", (project_id,)
        ).fetchone()
        retired = conn.execute(
            "SELECT retired FROM agent_git_retirement WHERE project_id = ?", (project_id,)
        ).fetchone()
        if project is None or project[0] or (retired is not None and retired[0]):
            raise AgentWorkspaceError("Project Git operations are unavailable during retirement.")
    finally:
        conn.close()


def set_git_retirement(project_id: str, retired: bool) -> None:
    conn = connection()
    try:
        conn.execute(
            "INSERT INTO agent_git_retirement(project_id, retired) "
            "SELECT id, ? FROM chat_projects WHERE id = ? "
            "ON CONFLICT(project_id) DO UPDATE SET retired = excluded.retired",
            (int(retired), project_id),
        )
        conn.commit()
    finally:
        conn.close()
