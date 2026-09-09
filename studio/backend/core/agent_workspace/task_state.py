# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable project task ownership, bounded delegation, and explicit retries.

Runtime selections and workspace identities are captured by the server before
creation. A lease token never leaves the worker API. Reads reconcile expired
leases; neither a restart nor a read silently repeats a model/tool operation.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any

from storage.studio_db import get_connection


class TaskStateError(RuntimeError):
    """A task request cannot preserve the durable ownership contract."""


TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
ACTIVE = frozenset({"queued", "running", "cancelling"})
MAX_CHILDREN = 8
MAX_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 32768
MAX_DELEGATED_TOKENS = 131072
MAX_ACTIVE_TASKS = 128
LEASE_MS = 30000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studio_project_tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES studio_project_tasks(id),
    root_id TEXT NOT NULL REFERENCES studio_project_tasks(id),
    retry_of TEXT REFERENCES studio_project_tasks(id),
    attempt INTEGER NOT NULL CHECK(attempt BETWEEN 1 AND 3),
    role TEXT NOT NULL CHECK(role IN ('root', 'reviewer', 'implementer')),
    instruction TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','running','cancelling','completed','failed','cancelled','interrupted')),
    revision INTEGER NOT NULL DEFAULT 1,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
    owner TEXT,
    lease_expires_at INTEGER,
    max_output_tokens INTEGER NOT NULL CHECK(max_output_tokens BETWEEN 1 AND 32768),
    child_limit INTEGER NOT NULL CHECK(child_limit BETWEEN 0 AND 8),
    child_budget INTEGER NOT NULL CHECK(child_budget BETWEEN 0 AND 131072),
    child_allocated INTEGER NOT NULL DEFAULT 0,
    child_count INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER
);
CREATE INDEX IF NOT EXISTS studio_project_tasks_project ON studio_project_tasks(project_id,created_at,id);
CREATE INDEX IF NOT EXISTS studio_project_tasks_parent ON studio_project_tasks(parent_id,status);
CREATE INDEX IF NOT EXISTS studio_project_tasks_lease ON studio_project_tasks(status,lease_expires_at);
CREATE TABLE IF NOT EXISTS studio_project_task_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES studio_project_tasks(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(task_id,revision)
);
CREATE TABLE IF NOT EXISTS studio_project_task_retirements (
    project_id TEXT PRIMARY KEY REFERENCES chat_projects(id) ON DELETE CASCADE,
    owner TEXT NOT NULL,
    lease_expires_at INTEGER NOT NULL
);
"""


def _now() -> int:
    return time.time_ns() // 1000000


def _integer(value: Any, lower: int, upper: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise TaskStateError(f"Invalid {label}.")
    return value


def _json(value: Any, limit: int, label: str) -> str:
    try:
        result = json.dumps(value, ensure_ascii = False, allow_nan = False, separators = (",", ":"))
        if len(result.encode("utf-8")) > limit:
            raise TaskStateError(f"{label} is too large.")
        return result
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TaskStateError(f"Invalid {label}.") from exc


def _instruction(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TaskStateError("Task instructions are required.")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise TaskStateError("Invalid task instructions.") from exc
    if size > 65536:
        raise TaskStateError("Task instructions are too large.")
    return value


def _event(conn, task_id: str, kind: str, now: int) -> None:
    conn.execute(
        "INSERT INTO studio_project_task_events(task_id,revision,kind,created_at) "
        "SELECT id,revision,?,? FROM studio_project_tasks WHERE id=?",
        (kind, now, task_id),
    )


def _row(conn, task_id: str):
    row = conn.execute("SELECT * FROM studio_project_tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        raise TaskStateError("Task not found.")
    return row


def _public(row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "parentId": row["parent_id"],
        "rootId": row["root_id"],
        "retryOf": row["retry_of"],
        "attempt": row["attempt"],
        "role": row["role"],
        "instruction": row["instruction"],
        "snapshot": json.loads(row["snapshot_json"]),
        "status": row["status"],
        "revision": row["revision"],
        "cancelRequested": bool(row["cancel_requested"]),
        "maxOutputTokens": row["max_output_tokens"],
        "childLimit": row["child_limit"],
        "childBudget": row["child_budget"],
        "childAllocated": row["child_allocated"],
        "childCount": row["child_count"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def _project_available(conn, project_id: str, now: int) -> None:
    row = conn.execute("SELECT archived FROM chat_projects WHERE id=?", (project_id,)).fetchone()
    if row is None or row["archived"]:
        raise TaskStateError("Project not found.")
    retiring = conn.execute(
        "SELECT 1 FROM studio_project_task_retirements WHERE project_id=? AND lease_expires_at>?",
        (project_id, now),
    ).fetchone()
    if retiring is not None:
        raise TaskStateError("Project retirement is in progress.")


def _owned(
    conn,
    task_id: str,
    owner: str,
    now: int,
    *,
    allow_cancel: bool = False,
):
    row = _row(conn, task_id)
    statuses = {"running", "cancelling"} if allow_cancel else {"running"}
    if (
        not owner
        or row["owner"] != owner
        or row["status"] not in statuses
        or row["lease_expires_at"] is None
        or row["lease_expires_at"] <= now
        or (row["cancel_requested"] and not allow_cancel)
    ):
        raise TaskStateError("Task ownership is no longer active.")
    return row


def _cancel_rows(conn, rows, now: int) -> None:
    for row in rows:
        row = _row(conn, row["id"])
        if row["status"] not in ACTIVE or row["cancel_requested"]:
            continue
        queued = row["status"] == "queued"
        conn.execute(
            "UPDATE studio_project_tasks SET cancel_requested=1,status=?,revision=revision+1,"
            "updated_at=?,completed_at=? WHERE id=?",
            ("cancelled" if queued else "cancelling", now, now if queued else None, row["id"]),
        )
        _event(conn, row["id"], "cancelled" if queued else "cancel_requested", now)


def _reconcile(conn, now: int) -> None:
    expired = conn.execute(
        "SELECT * FROM studio_project_tasks WHERE status IN ('queued','running','cancelling') AND lease_expires_at<=?",
        (now,),
    ).fetchall()
    for row in expired:
        row = _row(conn, row["id"])
        if row["status"] not in ACTIVE:
            continue
        conn.execute(
            "UPDATE studio_project_tasks SET status='interrupted',owner=NULL,lease_expires_at=NULL,"
            "error='Worker ownership expired; retry requires an explicit request.',revision=revision+1,"
            "updated_at=?,completed_at=? WHERE id=?",
            (now, now, row["id"]),
        )
        _event(conn, row["id"], "interrupted", now)
        children = conn.execute(
            "SELECT * FROM studio_project_tasks WHERE parent_id=?", (row["id"],)
        ).fetchall()
        _cancel_rows(conn, children, now)

    archived = conn.execute(
        "SELECT task.* FROM studio_project_tasks AS task JOIN chat_projects AS project "
        "ON project.id=task.project_id WHERE project.archived!=0 "
        "AND task.status IN ('queued','running','cancelling')"
    ).fetchall()
    _cancel_rows(conn, archived, now)


@contextmanager
def _transaction():
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        now = _now()
        _reconcile(conn, now)
        # Expiry is durable even if the requested operation is refused (for
        # example, a late worker attempting to renew its revoked ownership).
        conn.execute("SAVEPOINT task_request")
        try:
            yield conn, now
        except BaseException:
            conn.execute("ROLLBACK TO task_request")
            conn.execute("RELEASE task_request")
            conn.commit()
            raise
        conn.execute("RELEASE task_request")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _insert(
    conn,
    *,
    project_id,
    instruction,
    snapshot,
    role,
    parent_id,
    root_id,
    max_output_tokens,
    child_limit,
    child_budget,
    now,
    retry_of = None,
    attempt = 1,
):
    active = conn.execute(
        "SELECT COUNT(*) FROM studio_project_tasks WHERE status IN ('queued','running','cancelling')"
    ).fetchone()[0]
    if active >= MAX_ACTIVE_TASKS:
        raise TaskStateError("The project task queue is full.")
    task_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO studio_project_tasks(id,project_id,parent_id,root_id,retry_of,attempt,role,"
        "instruction,snapshot_json,status,max_output_tokens,child_limit,child_budget,created_at,updated_at,lease_expires_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,'queued',?,?,?,?,?,?)",
        (
            task_id,
            project_id,
            parent_id,
            root_id or task_id,
            retry_of,
            attempt,
            role,
            instruction,
            snapshot,
            max_output_tokens,
            child_limit,
            child_budget,
            now,
            now,
            now + LEASE_MS,
        ),
    )
    _event(conn, task_id, "queued", now)
    return _public(_row(conn, task_id))


def create_task(
    project_id: str,
    instruction: str,
    snapshot: dict,
    *,
    max_output_tokens: int = 8192,
    child_limit: int = 0,
    child_budget: int = 0,
) -> dict:
    instruction = _instruction(instruction)
    snapshot_json = _json(snapshot, 131072, "Runtime/workspace snapshot")
    if not isinstance(snapshot, dict):
        raise TaskStateError("A runtime/workspace snapshot is required.")
    _integer(max_output_tokens, 1, MAX_OUTPUT_TOKENS, "output-token bound")
    _integer(child_limit, 0, MAX_CHILDREN, "child limit")
    _integer(child_budget, 0, MAX_DELEGATED_TOKENS, "delegated-token bound")
    if bool(child_limit) != bool(child_budget):
        raise TaskStateError("Delegation requires both a child limit and a token budget.")
    with _transaction() as (conn, now):
        _project_available(conn, project_id, now)
        return _insert(
            conn,
            project_id = project_id,
            instruction = instruction,
            snapshot = snapshot_json,
            role = "root",
            parent_id = None,
            root_id = None,
            max_output_tokens = max_output_tokens,
            child_limit = child_limit,
            child_budget = child_budget,
            now = now,
        )


def get_task(project_id: str, task_id: str) -> dict:
    with _transaction() as (conn, _now_ms):
        row = _row(conn, task_id)
        if row["project_id"] != project_id:
            raise TaskStateError("Task not found.")
        return _public(row)


def list_tasks(project_id: str, *, limit: int = 100) -> list[dict]:
    _integer(limit, 1, 500, "task-list limit")
    with _transaction() as (conn, _now_ms):
        return [
            _public(row)
            for row in conn.execute(
                "SELECT * FROM studio_project_tasks WHERE project_id=? ORDER BY created_at DESC,id DESC LIMIT ?",
                (project_id, limit),
            )
        ]


def claim_task(project_id: str, task_id: str, owner: str) -> dict:
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise TaskStateError("A worker ownership token is required.")
    with _transaction() as (conn, now):
        _project_available(conn, project_id, now)
        row = _row(conn, task_id)
        if row["project_id"] != project_id or row["status"] != "queued":
            raise TaskStateError("Only a queued task in this project can be claimed.")
        if row["owner"] is not None and row["owner"] != owner:
            raise TaskStateError("Task dispatch is owned by another worker.")
        if row["parent_id"]:
            parent = _row(conn, row["parent_id"])
            if parent["status"] != "running" or parent["cancel_requested"]:
                raise TaskStateError("The parent task is no longer running.")
        conn.execute(
            "UPDATE studio_project_tasks SET status='running',owner=?,lease_expires_at=?,"
            "revision=revision+1,started_at=?,updated_at=? WHERE id=?",
            (owner, now + LEASE_MS, now, now, task_id),
        )
        _event(conn, task_id, "running", now)
        return _public(_row(conn, task_id))


def heartbeat(task_id: str, owner: str) -> bool:
    with _transaction() as (conn, now):
        row = _row(conn, task_id)
        if row["status"] != "queued" or row["owner"] != owner or not owner:
            row = _owned(conn, task_id, owner, now, allow_cancel = True)
        conn.execute(
            "UPDATE studio_project_tasks SET lease_expires_at=? WHERE id=?",
            (now + LEASE_MS, task_id),
        )
        return not row["cancel_requested"]


def validate_owner(task_id: str, owner: str) -> dict:
    with _transaction() as (conn, now):
        row = _owned(conn, task_id, owner, now)
        _project_available(conn, row["project_id"], now)
        return _public(row)


def cancel_task(project_id: str, task_id: str) -> dict:
    with _transaction() as (conn, now):
        row = _row(conn, task_id)
        if row["project_id"] != project_id:
            raise TaskStateError("Task not found.")
        children = conn.execute(
            "SELECT * FROM studio_project_tasks WHERE parent_id=?", (task_id,)
        ).fetchall()
        _cancel_rows(conn, [row, *children], now)
        return _public(_row(conn, task_id))


def finish_task(
    task_id: str,
    owner: str,
    status: str,
    *,
    result: dict | None = None,
    error: str | None = None,
) -> dict:
    if status not in TERMINAL:
        raise TaskStateError("A terminal task status is required.")
    if result is not None and not isinstance(result, dict):
        raise TaskStateError("A task result must be an object.")
    result_json = _json(result, 1024 * 1024, "Task result") if result is not None else None
    if error is not None and (not isinstance(error, str) or len(error.encode("utf-8")) > 4096):
        raise TaskStateError("Task error is too large.")
    with _transaction() as (conn, now):
        row = _owned(conn, task_id, owner, now, allow_cancel = True)
        if conn.execute(
            "SELECT 1 FROM studio_project_tasks WHERE parent_id=? AND status IN ('queued','running','cancelling') LIMIT 1",
            (task_id,),
        ).fetchone():
            raise TaskStateError("Child tasks must settle before the parent completes.")
        if row["cancel_requested"]:
            status = "cancelled"
        conn.execute(
            "UPDATE studio_project_tasks SET status=?,result_json=?,error=?,owner=NULL,lease_expires_at=NULL,"
            "revision=revision+1,updated_at=?,completed_at=? WHERE id=?",
            (status, result_json, error, now, now, task_id),
        )
        _event(conn, task_id, status, now)
        return _public(_row(conn, task_id))


def create_child(
    parent_id: str,
    owner: str,
    instruction: str,
    snapshot: dict,
    *,
    role: str,
    max_output_tokens: int = 4096,
) -> dict:
    instruction = _instruction(instruction)
    snapshot_json = _json(snapshot, 131072, "Runtime/workspace snapshot")
    _integer(max_output_tokens, 1, MAX_OUTPUT_TOKENS, "output-token bound")
    if not isinstance(snapshot, dict) or role not in {"reviewer", "implementer"}:
        raise TaskStateError("Invalid child-task request.")
    with _transaction() as (conn, now):
        parent = _owned(conn, parent_id, owner, now)
        _project_available(conn, parent["project_id"], now)
        if parent["parent_id"] is not None:
            raise TaskStateError("Child tasks cannot delegate further.")
        if parent["child_count"] >= parent["child_limit"]:
            raise TaskStateError("The root task's child limit is exhausted.")
        if parent["child_allocated"] + max_output_tokens > parent["child_budget"]:
            raise TaskStateError("The root task's delegated-token budget is exhausted.")
        conn.execute(
            "UPDATE studio_project_tasks SET child_count=child_count+1,child_allocated=child_allocated+?,"
            "revision=revision+1,updated_at=? WHERE id=?",
            (max_output_tokens, now, parent_id),
        )
        _event(conn, parent_id, "child_reserved", now)
        return _insert(
            conn,
            project_id = parent["project_id"],
            instruction = instruction,
            snapshot = snapshot_json,
            role = role,
            parent_id = parent_id,
            root_id = parent_id,
            max_output_tokens = max_output_tokens,
            child_limit = 0,
            child_budget = 0,
            now = now,
        )


def retry_task(
    project_id: str,
    task_id: str,
    *,
    parent_owner: str | None = None,
) -> dict:
    with _transaction() as (conn, now):
        _project_available(conn, project_id, now)
        row = _row(conn, task_id)
        if row["project_id"] != project_id or row["status"] not in {
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise TaskStateError("Only unsuccessful tasks in this project can be retried.")
        if row["attempt"] >= MAX_ATTEMPTS:
            raise TaskStateError("The task attempt limit is exhausted.")
        if conn.execute(
            "SELECT 1 FROM studio_project_tasks WHERE retry_of=?", (task_id,)
        ).fetchone():
            raise TaskStateError("This attempt already has a retry.")
        if conn.execute(
            "SELECT 1 FROM studio_project_tasks WHERE parent_id=? AND status IN ('queued','running','cancelling') LIMIT 1",
            (task_id,),
        ).fetchone():
            raise TaskStateError("Child tasks must settle before retrying the parent.")
        if row["parent_id"]:
            parent = _owned(conn, row["parent_id"], parent_owner, now)
            if parent["child_allocated"] + row["max_output_tokens"] > parent["child_budget"]:
                raise TaskStateError("The root task's delegated-token budget is exhausted.")
            conn.execute(
                "UPDATE studio_project_tasks SET child_allocated=child_allocated+?,revision=revision+1,updated_at=? WHERE id=?",
                (row["max_output_tokens"], now, parent["id"]),
            )
            _event(conn, parent["id"], "child_retry_reserved", now)
        return _insert(
            conn,
            project_id = project_id,
            instruction = row["instruction"],
            snapshot = row["snapshot_json"],
            role = row["role"],
            parent_id = row["parent_id"],
            root_id = row["root_id"] if row["parent_id"] else None,
            max_output_tokens = row["max_output_tokens"],
            child_limit = row["child_limit"],
            child_budget = row["child_budget"],
            now = now,
            retry_of = row["id"],
            attempt = row["attempt"] + 1,
        )


def task_events(
    project_id: str,
    task_id: str,
    *,
    after: int = 0,
    limit: int = 100,
) -> list[dict]:
    _integer(after, 0, 2**63 - 1, "event cursor")
    _integer(limit, 1, 500, "event-list limit")
    with _transaction() as (conn, _now_ms):
        if _row(conn, task_id)["project_id"] != project_id:
            raise TaskStateError("Task not found.")
        return [
            dict(row)
            for row in conn.execute(
                "SELECT sequence,revision,kind,created_at AS createdAt FROM studio_project_task_events "
                "WHERE task_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (task_id, after, limit),
            )
        ]


def reserve_dispatch(project_id: str, task_id: str, owner: str) -> None:
    """Reserve a queued task while a bounded worker lane is busy."""
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise TaskStateError("A worker ownership token is required.")
    with _transaction() as (conn, now):
        _project_available(conn, project_id, now)
        row = _row(conn, task_id)
        if row["project_id"] != project_id or row["status"] != "queued" or row["owner"] is not None:
            raise TaskStateError("Task dispatch is no longer available.")
        conn.execute(
            "UPDATE studio_project_tasks SET owner=?,lease_expires_at=? WHERE id=?",
            (owner, now + LEASE_MS, task_id),
        )


def reconcile_expired_tasks() -> None:
    """Recover expired attempts without replaying queued or running work."""
    with _transaction():
        pass


def begin_project_retirement(project_id: str, owner: str) -> None:
    """Fence admission before signalling existing work to stop."""
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise TaskStateError("A retirement ownership token is required.")
    with _transaction() as (conn, now):
        _project_available(conn, project_id, now)
        conn.execute(
            "INSERT INTO studio_project_task_retirements(project_id,owner,lease_expires_at) VALUES(?,?,?) "
            "ON CONFLICT(project_id) DO UPDATE SET owner=excluded.owner,lease_expires_at=excluded.lease_expires_at",
            (project_id, owner, now + LEASE_MS),
        )
        _cancel_rows(
            conn,
            conn.execute(
                "SELECT * FROM studio_project_tasks WHERE project_id=?",
                (project_id,),
            ).fetchall(),
            now,
        )


def renew_project_retirement(project_id: str, owner: str) -> None:
    with _transaction() as (conn, now):
        changed = conn.execute(
            "UPDATE studio_project_task_retirements SET lease_expires_at=? "
            "WHERE project_id=? AND owner=? AND lease_expires_at>?",
            (now + LEASE_MS, project_id, owner, now),
        ).rowcount
        if not changed:
            raise TaskStateError("Project retirement ownership expired.")


def finish_project_retirement(project_id: str, owner: str) -> None:
    with _transaction() as (conn, _now_ms):
        conn.execute(
            "DELETE FROM studio_project_task_retirements WHERE project_id=? AND owner=?",
            (project_id, owner),
        )


def project_has_active_tasks(project_id: str) -> bool:
    with _transaction() as (conn, _now_ms):
        return (
            conn.execute(
                "SELECT 1 FROM studio_project_tasks WHERE project_id=? "
                "AND status IN ('queued','running','cancelling') LIMIT 1",
                (project_id,),
            ).fetchone()
            is not None
        )


def list_children(project_id: str, task_id: str) -> list[dict]:
    with _transaction() as (conn, _now_ms):
        if _row(conn, task_id)["project_id"] != project_id:
            raise TaskStateError("Task not found.")
        return [
            _public(row)
            for row in conn.execute(
                "SELECT * FROM studio_project_tasks WHERE parent_id=? ORDER BY created_at,id",
                (task_id,),
            )
        ]


def interrupt_task(task_id: str, owner: str) -> dict:
    """Relinquish an attempt whose children could not be drained in time."""
    with _transaction() as (conn, now):
        _owned(conn, task_id, owner, now, allow_cancel = True)
        _cancel_rows(
            conn,
            conn.execute(
                "SELECT * FROM studio_project_tasks WHERE parent_id=?",
                (task_id,),
            ).fetchall(),
            now,
        )
        conn.execute(
            "UPDATE studio_project_tasks SET status='interrupted',owner=NULL,lease_expires_at=NULL,"
            "cancel_requested=1,error='Child tasks did not settle before the shutdown deadline.',"
            "revision=revision+1,updated_at=?,completed_at=? WHERE id=?",
            (now, now, task_id),
        )
        _event(conn, task_id, "interrupted", now)
        return _public(_row(conn, task_id))
