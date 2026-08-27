# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable SQLite state for project-agent features."""

import json
import sqlite3
import threading
import uuid
from typing import Any, Optional

from storage.studio_db import get_connection

from .common import AgentWorkspaceError, now_ms


_STATE_LOCK = threading.Lock()
_READY_DATABASES: set[str] = set()
_PLAN_STATUSES = frozenset({"active", "blocked", "completed", "cancelled"})
_TASK_STATUSES = frozenset({"pending", "running", "blocked", "completed", "cancelled"})
_BACKGROUND_STATUSES = frozenset(
    {"queued", "running", "cancelling", "cancelled", "completed", "failed", "interrupted"}
)
_WORKTREE_STATUSES = frozenset({"creating", "active", "removing", "removed", "needs_attention"})
_BACKGROUND_PAYLOAD_LIMIT = 256 * 1024
_BACKGROUND_RESULT_LIMIT = 1024 * 1024
_PLAN_SNAPSHOT_LIMIT = 512 * 1024
_NOT_PROVIDED = object()


def _database_key(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return str(row[2])


def _ensure_state_schema(conn: sqlite3.Connection) -> None:
    key = _database_key(conn)
    if key in _READY_DATABASES:
        return
    with _STATE_LOCK:
        if key in _READY_DATABASES:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_verification_configs (
                project_id TEXT NOT NULL PRIMARY KEY
                    REFERENCES chat_projects(id) ON DELETE CASCADE,
                checks_json TEXT NOT NULL,
                require_for_goal_completion INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_verification_runs (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                worktree_id TEXT,
                status TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                final_fingerprint TEXT,
                config_revision INTEGER NOT NULL DEFAULT 0,
                results_json TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_agent_verification_runs_project
                ON agent_verification_runs(project_id, started_at DESC);

            CREATE TABLE IF NOT EXISTS agent_plans (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                goal_snapshot TEXT,
                goal_updated_at INTEGER,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_plans_project
                ON agent_plans(project_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS agent_plan_tasks (
                id TEXT NOT NULL PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES agent_plans(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                blocker TEXT,
                verification_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(plan_id, position)
            );

            CREATE TABLE IF NOT EXISTS agent_background_tasks (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                goal_snapshot TEXT,
                goal_status_snapshot TEXT,
                goal_updated_at INTEGER,
                plan_id TEXT,
                plan_revision INTEGER,
                plan_task_id TEXT,
                plan_snapshot_json TEXT,
                worktree_id TEXT,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                parent_task_id TEXT REFERENCES agent_background_tasks(id) ON DELETE SET NULL,
                result_json TEXT,
                error TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                started_at INTEGER,
                completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_agent_background_tasks_project
                ON agent_background_tasks(project_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS agent_git_checkpoints (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                git_root TEXT NOT NULL,
                ref_name TEXT NOT NULL UNIQUE,
                commit_sha TEXT NOT NULL,
                owned_paths_json TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_worktrees (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                git_root TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                branch TEXT NOT NULL,
                base_ref TEXT NOT NULL,
                marker_path TEXT NOT NULL,
                marker_token_hash TEXT NOT NULL,
                background_task_id TEXT REFERENCES agent_background_tasks(id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                merge_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        plan_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_plans)").fetchall()}
        if "goal_updated_at" not in plan_columns:
            conn.execute("ALTER TABLE agent_plans ADD COLUMN goal_updated_at INTEGER")
        if "revision" not in plan_columns:
            conn.execute("ALTER TABLE agent_plans ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
        worktree_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(agent_worktrees)").fetchall()
        }
        if "marker_token_hash" not in worktree_columns:
            conn.execute(
                "ALTER TABLE agent_worktrees ADD COLUMN marker_token_hash TEXT NOT NULL DEFAULT ''"
            )
        if "background_task_id" not in worktree_columns:
            conn.execute("ALTER TABLE agent_worktrees ADD COLUMN background_task_id TEXT")
        if "merge_json" not in worktree_columns:
            conn.execute("ALTER TABLE agent_worktrees ADD COLUMN merge_json TEXT")
        background_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(agent_background_tasks)").fetchall()
        }
        background_migrations = {
            "goal_snapshot": "TEXT",
            "goal_status_snapshot": "TEXT",
            "goal_updated_at": "INTEGER",
            "plan_id": "TEXT",
            "plan_revision": "INTEGER",
            "plan_task_id": "TEXT",
            "plan_snapshot_json": "TEXT",
            "worktree_id": "TEXT",
        }
        for column, type_name in background_migrations.items():
            if column not in background_columns:
                conn.execute(f"ALTER TABLE agent_background_tasks ADD COLUMN {column} {type_name}")
        verification_run_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(agent_verification_runs)").fetchall()
        }
        if "worktree_id" not in verification_run_columns:
            conn.execute("ALTER TABLE agent_verification_runs ADD COLUMN worktree_id TEXT")
        if "config_revision" not in verification_run_columns:
            conn.execute(
                "ALTER TABLE agent_verification_runs "
                "ADD COLUMN config_revision INTEGER NOT NULL DEFAULT 0"
            )
        verification_config_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(agent_verification_configs)").fetchall()
        }
        if "require_for_goal_completion" not in verification_config_columns:
            conn.execute(
                "ALTER TABLE agent_verification_configs "
                "ADD COLUMN require_for_goal_completion INTEGER NOT NULL DEFAULT 0"
            )
        if "revision" not in verification_config_columns:
            conn.execute(
                "ALTER TABLE agent_verification_configs "
                "ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()
        # A desktop restart cannot prove an old child is still controlled by this
        # process. Record interruption instead of reporting false success/running.
        current = now_ms()
        conn.execute(
            """
            UPDATE agent_background_tasks
            SET status = 'interrupted', updated_at = ?, completed_at = ?,
                error = COALESCE(error, 'Studio restarted while the task was active.')
            WHERE status IN ('running', 'cancelling')
            """,
            (current, current),
        )
        conn.commit()
        _READY_DATABASES.add(key)


def connection() -> sqlite3.Connection:
    conn = get_connection()
    _ensure_state_schema(conn)
    return conn


def _loads(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _encoded_json(value: Any, *, limit: int, label: str) -> str:
    try:
        encoded = json.dumps(value, separators = (",", ":"))
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} must be valid JSON.") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise AgentWorkspaceError(f"{label} is too large.")
    return encoded


def _verification_run(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "worktreeId": row["worktree_id"],
        "status": row["status"],
        "configRevision": row["config_revision"],
        "sourceFingerprint": row["source_fingerprint"],
        "finalFingerprint": row["final_fingerprint"],
        "results": _loads(row["results_json"], []),
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def set_verification_config(
    project_id: str,
    checks: list[dict],
    *,
    require_for_goal_completion: bool = False,
    expected_revision: Optional[int] = None,
) -> dict:
    normalized_checks = []
    seen_names: set[str] = set()
    for check in checks:
        normalized = dict(check)
        name = str(normalized.get("name") or "").strip()
        command = str(normalized.get("command") or "").strip()
        if not name or not command:
            raise AgentWorkspaceError("Verification check names and commands cannot be blank.")
        name_key = name.casefold()
        if name_key in seen_names:
            raise AgentWorkspaceError("Verification check names must be unique.")
        seen_names.add(name_key)
        normalized["name"] = name
        normalized["command"] = command
        normalized_checks.append(normalized)
    checks = normalized_checks
    current = now_ms()
    encoded = json.dumps(checks, separators = (",", ":"))
    if len(encoded.encode("utf-8")) > 128 * 1024:
        raise AgentWorkspaceError("Verification configuration is too large.")
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT revision FROM agent_verification_configs
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        current_revision = int(row["revision"]) if row else 0
        if expected_revision is not None and current_revision != expected_revision:
            raise AgentWorkspaceError(
                "Verification settings changed in another session. Refresh and retry."
            )
        revision = current_revision + 1
        if row:
            conn.execute(
                """
                UPDATE agent_verification_configs
                SET checks_json = ?, require_for_goal_completion = ?,
                    revision = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (
                    encoded,
                    1 if require_for_goal_completion else 0,
                    revision,
                    current,
                    project_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO agent_verification_configs(
                    project_id, checks_json, require_for_goal_completion,
                    revision, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    encoded,
                    1 if require_for_goal_completion else 0,
                    revision,
                    current,
                ),
            )
        conn.commit()
        return {
            "projectId": project_id,
            "checks": checks,
            "requireForGoalCompletion": require_for_goal_completion,
            "revision": revision,
            "updatedAt": current,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_verification_config(project_id: str) -> dict:
    conn = connection()
    try:
        row = conn.execute(
            """
            SELECT checks_json, require_for_goal_completion, revision, updated_at
            FROM agent_verification_configs WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        return {
            "projectId": project_id,
            "checks": _loads(row["checks_json"], []) if row else [],
            "requireForGoalCompletion": bool(row["require_for_goal_completion"] if row else False),
            "revision": int(row["revision"] if row else 0),
            "updatedAt": row["updated_at"] if row else None,
        }
    finally:
        conn.close()


def begin_verification_run(
    project_id: str,
    source_fingerprint: str,
    worktree_id: Optional[str] = None,
    *,
    config_revision: int = 0,
) -> dict:
    run_id = str(uuid.uuid4())
    started = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_verification_runs(
                id, project_id, worktree_id, status,
                source_fingerprint, config_revision, results_json, started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, '[]', ?)
            """,
            (
                run_id,
                project_id,
                worktree_id,
                source_fingerprint,
                max(0, int(config_revision)),
                started,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": run_id,
        "projectId": project_id,
        "worktreeId": worktree_id,
        "status": "running",
        "configRevision": max(0, int(config_revision)),
        "sourceFingerprint": source_fingerprint,
        "finalFingerprint": None,
        "results": [],
        "startedAt": started,
        "completedAt": None,
    }


def finish_verification_run(
    run_id: str, status: str, final_fingerprint: str, results: list[dict]
) -> dict:
    completed = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            UPDATE agent_verification_runs
            SET status = ?, final_fingerprint = ?, results_json = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status,
                final_fingerprint,
                json.dumps(results, separators = (",", ":")),
                completed,
                run_id,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_verification_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Verification run not found.")
        return _verification_run(row)
    finally:
        conn.close()


def get_verification_run(run_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_verification_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _verification_run(row) if row else None
    finally:
        conn.close()


def list_verification_runs(project_id: str, limit: int = 20) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_verification_runs
            WHERE project_id = ? ORDER BY started_at DESC LIMIT ?
            """,
            (project_id, max(1, min(limit, 100))),
        ).fetchall()
        return [_verification_run(row) for row in rows]
    finally:
        conn.close()


def latest_primary_verification_run(project_id: str) -> Optional[dict]:
    """Return the newest run executed in the project's primary workspace."""
    conn = connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM agent_verification_runs
            WHERE project_id = ? AND worktree_id IS NULL
            ORDER BY started_at DESC, rowid DESC LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        return _verification_run(row) if row else None
    finally:
        conn.close()


def _task(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "planId": row["plan_id"],
        "position": row["position"],
        "title": row["title"],
        "status": row["status"],
        "blocker": row["blocker"],
        "verification": _loads(row["verification_json"], []),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _plan(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    tasks = conn.execute(
        "SELECT * FROM agent_plan_tasks WHERE plan_id = ? ORDER BY position, id",
        (row["id"],),
    ).fetchall()
    rendered_tasks = [_task(task) for task in tasks]
    counts = {status: 0 for status in sorted(_TASK_STATUSES)}
    blockers = []
    for task in rendered_tasks:
        counts[task["status"]] += 1
        if task["status"] == "blocked" and task["blocker"]:
            blockers.append({"taskId": task["id"], "text": task["blocker"]})
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "title": row["title"],
        "goalSnapshot": row["goal_snapshot"],
        "goalUpdatedAt": row["goal_updated_at"],
        "status": row["status"],
        "revision": row["revision"],
        "tasks": rendered_tasks,
        "completionSummary": {
            "counts": counts,
            "blockers": blockers,
            "remaining": sum(counts[status] for status in ("pending", "running", "blocked")),
        },
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def create_plan(
    project_id: str,
    title: str,
    goal_snapshot: Optional[str],
    tasks: list[dict],
    *,
    goal_updated_at: Optional[int] = None,
) -> dict:
    title = title.strip()
    if not title or len(title) > 500:
        raise AgentWorkspaceError("Plan title is invalid.")
    if len(tasks) > 500:
        raise AgentWorkspaceError("A plan can contain at most 500 tasks.")
    _encoded_json(tasks, limit = _PLAN_SNAPSHOT_LIMIT, label = "Plan")
    for task in tasks:
        task_title = str(task.get("title") or "").strip()
        task_status = str(task.get("status") or "pending")
        if not task_title or len(task_title) > 500 or task_status not in _TASK_STATUSES:
            raise AgentWorkspaceError("Plan task is invalid.")
    plan_id = str(uuid.uuid4())
    current = now_ms()
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO agent_plans(
                id, project_id, title, goal_snapshot, goal_updated_at,
                status, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
            """,
            (plan_id, project_id, title, goal_snapshot, goal_updated_at, current, current),
        )
        for position, task in enumerate(tasks):
            conn.execute(
                """
                INSERT INTO agent_plan_tasks(
                    id, plan_id, position, title, status, blocker,
                    verification_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    plan_id,
                    position,
                    str(task["title"]).strip(),
                    str(task.get("status") or "pending"),
                    task.get("blocker"),
                    json.dumps(task.get("verification") or [], separators = (",", ":")),
                    current,
                    current,
                ),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
        return _plan(conn, row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_plan(plan_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
        return _plan(conn, row) if row else None
    finally:
        conn.close()


def list_plans(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_plans WHERE project_id = ? ORDER BY updated_at DESC",
            (project_id,),
        ).fetchall()
        return [_plan(conn, row) for row in rows]
    finally:
        conn.close()


def update_plan_status(
    plan_id: str,
    status: str,
    *,
    expected_revision: Optional[int] = None,
) -> Optional[dict]:
    if status not in _PLAN_STATUSES:
        raise AgentWorkspaceError("Invalid plan status.")
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT revision FROM agent_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if current is None:
            return None
        if expected_revision is not None and current["revision"] != expected_revision:
            raise AgentWorkspaceError("Plan changed in another session. Refresh and retry.")
        if status == "completed":
            incomplete = conn.execute(
                """
                SELECT COUNT(*) AS count FROM agent_plan_tasks
                WHERE plan_id = ? AND status NOT IN ('completed', 'cancelled')
                """,
                (plan_id,),
            ).fetchone()
            if incomplete and int(incomplete["count"]) > 0:
                raise AgentWorkspaceError(
                    "Complete or cancel every plan task before completing the plan."
                )
        cursor = conn.execute(
            """
            UPDATE agent_plans
            SET status = ?, updated_at = ?, revision = revision + 1
            WHERE id = ? AND (? IS NULL OR revision = ?)
            """,
            (status, now_ms(), plan_id, expected_revision, expected_revision),
        )
        if expected_revision is not None and not cursor.rowcount:
            existing = conn.execute("SELECT 1 FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
            if existing:
                raise AgentWorkspaceError("Plan changed in another session. Refresh and retry.")
        conn.commit()
        row = conn.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
        return _plan(conn, row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_plan_task(
    plan_id: str,
    task_id: str,
    *,
    status: Optional[str] = None,
    blocker: Any = _NOT_PROVIDED,
    verification: Optional[list[dict]] = None,
    expected_revision: Optional[int] = None,
) -> Optional[dict]:
    if status is not None and status not in _TASK_STATUSES:
        raise AgentWorkspaceError("Invalid task status.")
    assignments = ["updated_at = ?"]
    values: list[Any] = [now_ms()]
    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if blocker is not _NOT_PROVIDED:
        assignments.append("blocker = ?")
        values.append(blocker or None)
    elif status is not None and status != "blocked":
        assignments.append("blocker = NULL")
    if verification is not None:
        assignments.append("verification_json = ?")
        values.append(
            _encoded_json(
                verification,
                limit = 128 * 1024,
                label = "Plan task verification requirements",
            )
        )
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        revision = conn.execute(
            "SELECT revision, project_id FROM agent_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if revision is None:
            conn.rollback()
            return None
        if expected_revision is not None and revision["revision"] != expected_revision:
            conn.rollback()
            raise AgentWorkspaceError("Plan changed in another session. Refresh and retry.")
        existing_task = conn.execute(
            """
            SELECT verification_json FROM agent_plan_tasks
            WHERE id = ? AND plan_id = ?
            """,
            (task_id, plan_id),
        ).fetchone()
        if existing_task is None:
            conn.rollback()
            return None
        effective_verification = (
            verification
            if verification is not None
            else _loads(existing_task["verification_json"], [])
        )
        if status == "completed":
            _require_plan_task_verification(str(revision["project_id"]), effective_verification)
        cursor = conn.execute(
            f"UPDATE agent_plan_tasks SET {', '.join(assignments)} WHERE id = ? AND plan_id = ?",
            (*values, task_id, plan_id),
        )
        if cursor.rowcount:
            conn.execute(
                """
                UPDATE agent_plans
                SET updated_at = ?, revision = revision + 1 WHERE id = ?
                """,
                (now_ms(), plan_id),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,)).fetchone()
        return _plan(conn, row) if row and cursor.rowcount else None
    finally:
        conn.close()


def _verification_requirement_name(
    requirement: Any,
) -> tuple[str, bool, Optional[str], Optional[str]]:
    if isinstance(requirement, str):
        return requirement.strip(), True, None, None
    if not isinstance(requirement, dict):
        raise AgentWorkspaceError("Plan task verification requirements are invalid.")
    name = str(requirement.get("name") or "").strip()
    required = bool(requirement.get("required", True))
    run_id = requirement.get("runId")
    worktree_id = requirement.get("worktreeId")
    return (
        name,
        required,
        str(run_id) if run_id is not None else None,
        str(worktree_id) if worktree_id is not None else None,
    )


def _latest_verification_row(
    conn: sqlite3.Connection, project_id: str, *, run_id: Optional[str], worktree_id: Optional[str]
) -> Optional[sqlite3.Row]:
    if run_id is not None:
        return conn.execute(
            """
            SELECT * FROM agent_verification_runs
            WHERE id = ? AND project_id = ?
            """,
            (run_id, project_id),
        ).fetchone()
    if worktree_id is None:
        return conn.execute(
            """
            SELECT * FROM agent_verification_runs
            WHERE project_id = ? AND worktree_id IS NULL
            ORDER BY started_at DESC, rowid DESC LIMIT 1
            """,
            (project_id,),
        ).fetchone()
    return conn.execute(
        """
        SELECT * FROM agent_verification_runs
        WHERE project_id = ? AND worktree_id = ?
        ORDER BY started_at DESC, rowid DESC LIMIT 1
        """,
        (project_id, worktree_id),
    ).fetchone()


def _require_plan_task_verification(project_id: str, requirements: list[Any]) -> None:
    """Require current passing evidence before a verified plan task can complete."""
    normalized = [
        parsed for item in requirements if (parsed := _verification_requirement_name(item))[1]
    ]
    if not normalized:
        return
    if any(not name for name, _required, _run_id, _worktree_id in normalized):
        raise AgentWorkspaceError("Required plan checks must have a name.")

    conn = connection()
    try:
        config_row = conn.execute(
            """
            SELECT revision FROM agent_verification_configs WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        config_revision = int(config_row["revision"] if config_row else 0)
        grouped: dict[tuple[Optional[str], Optional[str]], list[str]] = {}
        for name, _required, run_id, worktree_id in normalized:
            grouped.setdefault((run_id, worktree_id), []).append(name)
        run_ids: list[str] = []
        for (run_id, worktree_id), names in grouped.items():
            row = _latest_verification_row(
                conn,
                project_id,
                run_id = run_id,
                worktree_id = worktree_id,
            )
            if row is None:
                raise AgentWorkspaceError(
                    "Plan task completion requires fresh passing verification evidence."
                )
            record = _verification_run(row)
            if worktree_id is not None and record["worktreeId"] != worktree_id:
                raise AgentWorkspaceError(
                    "Plan task verification evidence came from the wrong workspace."
                )
            if record["configRevision"] != config_revision:
                raise AgentWorkspaceError(
                    "Plan task verification evidence is stale after a configuration change."
                )
            passed = {
                str(result.get("name") or "").strip()
                for result in record["results"]
                if result.get("status") == "passed"
            }
            if record["status"] != "passed" or any(name not in passed for name in names):
                raise AgentWorkspaceError(
                    "Plan task completion requires every required check to pass."
                )
            run_ids.append(record["id"])
    finally:
        conn.close()

    # Import after the state connection closes. Freshness computes filesystem
    # fingerprints and can open its own state connection.
    from .verification import verification_run_with_freshness

    for run_id in run_ids:
        fresh = verification_run_with_freshness(run_id)
        if not fresh or fresh.get("evidenceComplete") is not True or fresh.get("stale"):
            raise AgentWorkspaceError(
                "Plan task completion requires fresh passing verification evidence."
            )


def _background_task(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "kind": row["kind"],
        "payload": _loads(row["payload_json"], {}),
        "goalSnapshot": row["goal_snapshot"],
        "goalStatusSnapshot": row["goal_status_snapshot"],
        "goalUpdatedAt": row["goal_updated_at"],
        "planId": row["plan_id"],
        "planRevision": row["plan_revision"],
        "planTaskId": row["plan_task_id"],
        "planSnapshot": _loads(row["plan_snapshot_json"], None),
        "worktreeId": row["worktree_id"],
        "status": row["status"],
        "attempt": row["attempt"],
        "parentTaskId": row["parent_task_id"],
        "result": _loads(row["result_json"], None),
        "error": row["error"],
        "cancelRequested": bool(row["cancel_requested"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
        "appExitPolicy": "interrupt",
        "appExitContract": {
            "activeTaskState": "interrupted",
            "managedCommandsSurvive": False,
            "adapterMustHonorCancellation": True,
        },
    }


def create_background_task(
    project_id: str,
    kind: str,
    payload: dict,
    *,
    parent_task_id: Optional[str] = None,
    attempt: int = 1,
    goal_snapshot: Optional[str] = None,
    goal_status_snapshot: Optional[str] = None,
    goal_updated_at: Optional[int] = None,
    plan_id: Optional[str] = None,
    plan_revision: Optional[int] = None,
    plan_task_id: Optional[str] = None,
    plan_snapshot: Optional[dict] = None,
    worktree_id: Optional[str] = None,
) -> dict:
    task_id = str(uuid.uuid4())
    current = now_ms()
    encoded_payload = _encoded_json(
        payload, limit = _BACKGROUND_PAYLOAD_LIMIT, label = "Background task payload"
    )
    encoded_plan = (
        _encoded_json(
            plan_snapshot,
            limit = _PLAN_SNAPSHOT_LIMIT,
            label = "Plan snapshot",
        )
        if plan_snapshot is not None
        else None
    )
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_background_tasks(
                id, project_id, kind, payload_json,
                goal_snapshot, goal_status_snapshot, goal_updated_at,
                plan_id, plan_revision, plan_task_id, plan_snapshot_json,
                worktree_id, status, attempt, parent_task_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                task_id,
                project_id,
                kind,
                encoded_payload,
                goal_snapshot,
                goal_status_snapshot,
                goal_updated_at,
                plan_id,
                plan_revision,
                plan_task_id,
                encoded_plan,
                worktree_id,
                attempt,
                parent_task_id,
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _background_task(row)
    finally:
        conn.close()


def create_agent_background_task(
    project_id: str,
    instruction: str,
    *,
    runtime_snapshot: Optional[dict] = None,
    plan_id: Optional[str] = None,
    plan_task_id: Optional[str] = None,
    worktree_id: Optional[str] = None,
    cleanup_worktree_on_cancel: bool = False,
) -> dict:
    """Atomically snapshot project context and queue one provider-neutral agent run."""
    instruction = instruction.strip()
    if not instruction:
        raise AgentWorkspaceError("Agent task instructions cannot be empty.")
    if len(instruction) > 32_768:
        raise AgentWorkspaceError("Agent task instructions are too large.")
    if plan_task_id and not plan_id:
        raise AgentWorkspaceError("A plan task must include its plan ID.")

    task_id = str(uuid.uuid4())
    current = now_ms()
    payload = {
        "instruction": instruction,
        "cleanupWorktreeOnCancel": bool(cleanup_worktree_on_cancel),
        "runtime": runtime_snapshot,
    }
    encoded_payload = _encoded_json(
        payload, limit = _BACKGROUND_PAYLOAD_LIMIT, label = "Background task payload"
    )
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        project = conn.execute(
            """
            SELECT goal, goal_status, goal_updated_at
            FROM chat_projects WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
        if project is None:
            raise AgentWorkspaceError("Project not found.")

        plan_snapshot = None
        plan_revision = None
        if plan_id is not None:
            plan_row = conn.execute(
                "SELECT * FROM agent_plans WHERE id = ? AND project_id = ?",
                (plan_id, project_id),
            ).fetchone()
            if plan_row is None:
                raise AgentWorkspaceError("Plan not found.")
            plan_snapshot = _plan(conn, plan_row)
            plan_revision = int(plan_row["revision"])
            if plan_task_id is not None and not any(
                task["id"] == plan_task_id for task in plan_snapshot["tasks"]
            ):
                raise AgentWorkspaceError("Plan task not found.")
        encoded_plan = (
            _encoded_json(
                plan_snapshot,
                limit = _PLAN_SNAPSHOT_LIMIT,
                label = "Plan snapshot",
            )
            if plan_snapshot is not None
            else None
        )

        if worktree_id is not None:
            worktree = conn.execute(
                """
                SELECT project_id, status, background_task_id
                FROM agent_worktrees WHERE id = ?
                """,
                (worktree_id,),
            ).fetchone()
            if worktree is None or worktree["project_id"] != project_id:
                raise AgentWorkspaceError("Studio worktree not found.")
            if worktree["status"] != "active":
                raise AgentWorkspaceError("Studio worktree is not active.")
            if worktree["background_task_id"] is not None:
                raise AgentWorkspaceError("Studio worktree is already linked to another task.")

        conn.execute(
            """
            INSERT INTO agent_background_tasks(
                id, project_id, kind, payload_json,
                goal_snapshot, goal_status_snapshot, goal_updated_at,
                plan_id, plan_revision, plan_task_id, plan_snapshot_json,
                worktree_id, status, attempt, parent_task_id,
                created_at, updated_at
            ) VALUES (?, ?, 'agent', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1, NULL, ?, ?)
            """,
            (
                task_id,
                project_id,
                encoded_payload,
                project["goal"],
                project["goal_status"],
                project["goal_updated_at"],
                plan_id,
                plan_revision,
                plan_task_id,
                encoded_plan,
                worktree_id,
                current,
                current,
            ),
        )
        if worktree_id is not None:
            conn.execute(
                """
                UPDATE agent_worktrees
                SET background_task_id = ?, updated_at = ? WHERE id = ?
                """,
                (task_id, current, worktree_id),
            )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _background_task(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_background_task(
    task_id: str,
    status: str,
    *,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    cancel_requested: Optional[bool] = None,
) -> Optional[dict]:
    if status not in _BACKGROUND_STATUSES:
        raise AgentWorkspaceError("Invalid background task status.")
    current = now_ms()
    assignments = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, current]
    if status == "running":
        assignments.append("started_at = COALESCE(started_at, ?)")
        values.append(current)
    if status in {"cancelled", "completed", "failed", "interrupted"}:
        assignments.append("completed_at = ?")
        values.append(current)
    if result is not None:
        assignments.append("result_json = ?")
        values.append(
            _encoded_json(
                result,
                limit = _BACKGROUND_RESULT_LIMIT,
                label = "Background task result",
            )
        )
    if error is not None:
        assignments.append("error = ?")
        values.append(error[:4096])
    if cancel_requested is not None:
        assignments.append("cancel_requested = ?")
        values.append(1 if cancel_requested else 0)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        previous_row = conn.execute(
            "SELECT status FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if previous_row is None:
            conn.rollback()
            return None
        previous = str(previous_row["status"])
        allowed = {
            "queued": {"running", "cancelled"},
            "running": {"cancelling", "cancelled", "completed", "failed", "interrupted"},
            "cancelling": {"cancelled", "failed", "interrupted"},
            "cancelled": set(),
            "completed": set(),
            "failed": set(),
            "interrupted": set(),
        }
        if status != previous and status not in allowed[previous]:
            conn.rollback()
            raise AgentWorkspaceError(
                f"Background task cannot transition from {previous} to {status}."
            )
        cursor = conn.execute(
            f"UPDATE agent_background_tasks SET {', '.join(assignments)} "
            "WHERE id = ? AND status = ?",
            (*values, task_id, previous),
        )
        if not cursor.rowcount:
            conn.rollback()
            raise AgentWorkspaceError("Background task changed in another worker.")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _background_task(row) if row else None
    finally:
        conn.close()


def claim_background_task(task_id: str) -> Optional[dict]:
    """Atomically claim one queued task for execution."""
    current = now_ms()
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            """
            SELECT id, project_id, kind, status, worktree_id
            FROM agent_background_tasks WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if task is None:
            conn.rollback()
            return None
        if task["status"] != "queued":
            raise AgentWorkspaceError("Only an unclaimed queued background task can be started.")
        if task["worktree_id"] is not None:
            worktree = conn.execute(
                """
                SELECT project_id, status, background_task_id
                FROM agent_worktrees WHERE id = ?
                """,
                (task["worktree_id"],),
            ).fetchone()
            if (
                worktree is None
                or worktree["project_id"] != task["project_id"]
                or worktree["status"] != "active"
                or (task["kind"] == "agent" and worktree["background_task_id"] != task_id)
            ):
                raise AgentWorkspaceError("The task's linked worktree is not ready for execution.")
        cursor = conn.execute(
            """
            UPDATE agent_background_tasks
            SET status = 'running', updated_at = ?, started_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (current, current, task_id),
        )
        if not cursor.rowcount:
            conn.rollback()
            raise AgentWorkspaceError("Only an unclaimed queued background task can be started.")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _background_task(row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_background_task(task_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _background_task(row) if row else None
    finally:
        conn.close()


def list_background_tasks(project_id: str, limit: int = 100) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_background_tasks
            WHERE project_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (project_id, max(1, min(limit, 500))),
        ).fetchall()
        return [_background_task(row) for row in rows]
    finally:
        conn.close()


def list_active_background_tasks(project_id: str) -> list[dict]:
    """Return every task that must stop before its project row can be deleted."""
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_background_tasks
            WHERE project_id = ? AND status IN ('queued', 'running', 'cancelling')
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
        return [_background_task(row) for row in rows]
    finally:
        conn.close()


def list_all_active_background_tasks(limit: int = 4096) -> list[dict]:
    """Return bounded process-active rows for application shutdown handling."""
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_background_tasks
            WHERE status IN ('running', 'cancelling')
            ORDER BY started_at, id LIMIT ?
            """,
            (max(1, min(limit, 16_384)),),
        ).fetchall()
        return [_background_task(row) for row in rows]
    finally:
        conn.close()


def retry_background_task(task_id: str) -> dict:
    retried_id = str(uuid.uuid4())
    current = now_ms()
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if previous is None:
            raise AgentWorkspaceError("Background task not found.")
        if previous["status"] not in {"failed", "cancelled", "interrupted"}:
            raise AgentWorkspaceError("Only stopped background tasks can be retried.")
        worktree_id = previous["worktree_id"]
        if worktree_id is not None:
            worktree = conn.execute(
                """
                SELECT project_id, status, background_task_id
                FROM agent_worktrees WHERE id = ?
                """,
                (worktree_id,),
            ).fetchone()
            if (
                worktree is None
                or worktree["project_id"] != previous["project_id"]
                or worktree["status"] != "active"
            ):
                raise AgentWorkspaceError("Studio worktree is not active.")
            if previous["kind"] == "agent" and worktree["background_task_id"] != task_id:
                raise AgentWorkspaceError("Studio worktree belongs to another background task.")
        conn.execute(
            """
            INSERT INTO agent_background_tasks(
                id, project_id, kind, payload_json,
                goal_snapshot, goal_status_snapshot, goal_updated_at,
                plan_id, plan_revision, plan_task_id, plan_snapshot_json,
                worktree_id, status, attempt, parent_task_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                retried_id,
                previous["project_id"],
                previous["kind"],
                previous["payload_json"],
                previous["goal_snapshot"],
                previous["goal_status_snapshot"],
                previous["goal_updated_at"],
                previous["plan_id"],
                previous["plan_revision"],
                previous["plan_task_id"],
                previous["plan_snapshot_json"],
                worktree_id,
                int(previous["attempt"]) + 1,
                task_id,
                current,
                current,
            ),
        )
        if worktree_id is not None and previous["kind"] == "agent":
            cursor = conn.execute(
                """
                UPDATE agent_worktrees
                SET background_task_id = ?, updated_at = ?
                WHERE id = ? AND background_task_id = ? AND status = 'active'
                """,
                (retried_id, current, worktree_id, task_id),
            )
            if cursor.rowcount != 1:
                raise AgentWorkspaceError(
                    "Studio worktree changed while retrying the background task."
                )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_background_tasks WHERE id = ?", (retried_id,)
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Background task retry could not be loaded.")
        return _background_task(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def bind_background_task_worktree(
    task_id: str,
    worktree_id: str,
    *,
    previous_task_id: Optional[str] = None,
) -> dict:
    """Link one queued task to one active, Studio-owned worktree exactly once."""
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            "SELECT project_id, status, worktree_id FROM agent_background_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        worktree = conn.execute(
            """
            SELECT project_id, status, background_task_id
            FROM agent_worktrees WHERE id = ?
            """,
            (worktree_id,),
        ).fetchone()
        if task is None:
            raise AgentWorkspaceError("Background task not found.")
        if worktree is None:
            raise AgentWorkspaceError("Studio worktree not found.")
        if task["status"] != "queued":
            raise AgentWorkspaceError("Only a queued task can be linked to a worktree.")
        if task["project_id"] != worktree["project_id"]:
            raise AgentWorkspaceError("Worktree does not belong to this task's project.")
        if worktree["status"] != "active":
            raise AgentWorkspaceError("Studio worktree is not active.")
        if task["worktree_id"] not in {None, worktree_id}:
            raise AgentWorkspaceError("Background task is already linked to another worktree.")
        allowed_task_ids = {None, task_id}
        if previous_task_id is not None:
            allowed_task_ids.add(previous_task_id)
        if worktree["background_task_id"] not in allowed_task_ids:
            raise AgentWorkspaceError("Studio worktree is already linked to another task.")
        conn.execute(
            "UPDATE agent_background_tasks SET worktree_id = ?, updated_at = ? WHERE id = ?",
            (worktree_id, now_ms(), task_id),
        )
        conn.execute(
            "UPDATE agent_worktrees SET background_task_id = ?, updated_at = ? WHERE id = ?",
            (task_id, now_ms(), worktree_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    linked = get_background_task(task_id)
    if linked is None:
        raise AgentWorkspaceError("Background task not found.")
    return linked


def release_failed_worktree_task_reservation(task_id: str, worktree_id: str) -> None:
    """Release only a queued task reservation whose checkout never existed."""
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            """
            SELECT status, worktree_id FROM agent_background_tasks WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        worktree = conn.execute(
            """
            SELECT status, background_task_id FROM agent_worktrees WHERE id = ?
            """,
            (worktree_id,),
        ).fetchone()
        if task is None or worktree is None:
            raise AgentWorkspaceError("Worktree task reservation is unavailable.")
        if (
            task["status"] != "queued"
            or task["worktree_id"] != worktree_id
            or worktree["status"] != "removed"
            or worktree["background_task_id"] != task_id
        ):
            raise AgentWorkspaceError("Worktree task reservation is not safe to release.")
        task_cursor = conn.execute(
            """
            UPDATE agent_background_tasks SET worktree_id = NULL, updated_at = ?
            WHERE id = ? AND status = 'queued' AND worktree_id = ?
            """,
            (now_ms(), task_id, worktree_id),
        )
        worktree_cursor = conn.execute(
            """
            UPDATE agent_worktrees SET background_task_id = NULL, updated_at = ?
            WHERE id = ? AND status = 'removed' AND background_task_id = ?
            """,
            (now_ms(), worktree_id, task_id),
        )
        if task_cursor.rowcount != 1 or worktree_cursor.rowcount != 1:
            raise AgentWorkspaceError("Worktree task reservation changed during release.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_checkpoint(record: dict) -> None:
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
                record["id"],
                record["projectId"],
                record["gitRoot"],
                record["refName"],
                record["commitSha"],
                json.dumps(record["ownedPaths"], separators = (",", ":")),
                record["sourceFingerprint"],
                record["createdAt"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_checkpoint(checkpoint_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_git_checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "projectId": row["project_id"],
            "gitRoot": row["git_root"],
            "refName": row["ref_name"],
            "commitSha": row["commit_sha"],
            "ownedPaths": _loads(row["owned_paths_json"], []),
            "sourceFingerprint": row["source_fingerprint"],
            "createdAt": row["created_at"],
        }
    finally:
        conn.close()


def list_checkpoints(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT id FROM agent_git_checkpoints
            WHERE project_id = ? ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [record for row in rows if (record := get_checkpoint(row["id"])) is not None]


def delete_checkpoint(checkpoint_id: str, project_id: str) -> bool:
    """Forget one checkpoint only after its owned Git ref has been reconciled."""
    conn = connection()
    try:
        cursor = conn.execute(
            """
            DELETE FROM agent_git_checkpoints
            WHERE id = ? AND project_id = ?
            """,
            (checkpoint_id, project_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_worktree(record: dict) -> None:
    if record["status"] not in _WORKTREE_STATUSES:
        raise AgentWorkspaceError("Invalid worktree status.")
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        background_task_id = record.get("backgroundTaskId")
        if background_task_id is not None:
            task = conn.execute(
                """
                SELECT project_id, status, worktree_id
                FROM agent_background_tasks WHERE id = ?
                """,
                (background_task_id,),
            ).fetchone()
            if task is None or task["project_id"] != record["projectId"]:
                raise AgentWorkspaceError("Background task does not belong to this project.")
            if task["status"] != "queued":
                raise AgentWorkspaceError("Only a queued background task can reserve a worktree.")
            if task["worktree_id"] is not None:
                raise AgentWorkspaceError("Background task is already linked to another worktree.")
        conn.execute(
            """
            INSERT INTO agent_worktrees(
                id, project_id, git_root, path, branch, base_ref,
                marker_path, marker_token_hash, background_task_id,
                status, merge_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["projectId"],
                record["gitRoot"],
                record["path"],
                record["branch"],
                record["baseRef"],
                record["markerPath"],
                record["markerTokenHash"],
                background_task_id,
                record["status"],
                (
                    _encoded_json(
                        record["merge"],
                        limit = 64 * 1024,
                        label = "Worktree merge record",
                    )
                    if record.get("merge") is not None
                    else None
                ),
                record["createdAt"],
                record["updatedAt"],
            ),
        )
        if background_task_id is not None:
            cursor = conn.execute(
                """
                UPDATE agent_background_tasks
                SET worktree_id = ?, updated_at = ?
                WHERE id = ? AND status = 'queued' AND worktree_id IS NULL
                """,
                (record["id"], now_ms(), background_task_id),
            )
            if cursor.rowcount != 1:
                raise AgentWorkspaceError("Background task changed while reserving its worktree.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _worktree(row: sqlite3.Row) -> dict:
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
        "merge": _loads(row["merge_json"], None),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def get_worktree(worktree_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute("SELECT * FROM agent_worktrees WHERE id = ?", (worktree_id,)).fetchone()
        return _worktree(row) if row is not None else None
    finally:
        conn.close()


def update_worktree_status(worktree_id: str, status: str) -> Optional[dict]:
    if status not in _WORKTREE_STATUSES:
        raise AgentWorkspaceError("Invalid worktree status.")
    conn = connection()
    try:
        conn.execute(
            "UPDATE agent_worktrees SET status = ?, updated_at = ? WHERE id = ?",
            (status, now_ms(), worktree_id),
        )
        conn.commit()
    finally:
        conn.close()
    return get_worktree(worktree_id)


def transition_worktree_status(
    worktree_id: str, expected_statuses: set[str] | frozenset[str], status: str
) -> Optional[dict]:
    """Atomically move a worktree lifecycle row from an expected state."""
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
            (status, now_ms(), worktree_id, *expected),
        )
        conn.commit()
        if cursor.rowcount == 0:
            row = conn.execute(
                "SELECT status FROM agent_worktrees WHERE id = ?", (worktree_id,)
            ).fetchone()
            if row is None:
                return None
            raise AgentWorkspaceError(
                "Worktree state changed while the operation was running. Refresh and try again."
            )
    finally:
        conn.close()
    return get_worktree(worktree_id)


def record_worktree_merge(worktree_id: str, merge: dict) -> dict:
    encoded = _encoded_json(merge, limit = 64 * 1024, label = "Worktree merge record")
    conn = connection()
    try:
        cursor = conn.execute(
            """
            UPDATE agent_worktrees SET merge_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (encoded, now_ms(), worktree_id),
        )
        conn.commit()
        if not cursor.rowcount:
            raise AgentWorkspaceError("Studio worktree not found.")
    finally:
        conn.close()
    record = get_worktree(worktree_id)
    if record is None:
        raise AgentWorkspaceError("Studio worktree not found.")
    return record


def list_worktrees(project_id: str) -> list[dict]:
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_worktrees WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_worktree(row) for row in rows]


def list_all_worktrees(limit: int = 4096) -> list[dict]:
    """Return bounded lifecycle rows for startup reconciliation."""
    conn = connection()
    try:
        rows = conn.execute(
            "SELECT * FROM agent_worktrees ORDER BY created_at, id LIMIT ?",
            (max(1, min(limit, 16_384)),),
        ).fetchall()
    finally:
        conn.close()
    return [_worktree(row) for row in rows]


def list_active_worktrees(project_id: str) -> list[dict]:
    """Return every worktree lifecycle row that blocks project deletion."""
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_worktrees
            WHERE project_id = ? AND status != 'removed'
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_worktree(row) for row in rows]
