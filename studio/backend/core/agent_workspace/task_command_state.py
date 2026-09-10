# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded command evidence survives task cancellation, failure and worker loss."""

import json
import secrets
import uuid

from . import task_state as state
from .task_runner import TaskContext

MAX_COMMANDS = 6
MAX_OUTPUT_BYTES = 64 * 1024
MAX_TIMEOUT = 120


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS studio_task_commands (
        id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES studio_project_tasks(id) ON DELETE CASCADE,
        project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL, receipt TEXT NOT NULL, argv_json TEXT NOT NULL,
        timeout INTEGER NOT NULL, status TEXT NOT NULL, exit_code INTEGER,
        output TEXT NOT NULL DEFAULT '', output_bytes INTEGER NOT NULL DEFAULT 0,
        truncated INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, completed_at INTEGER,
        UNIQUE(task_id, sequence))""")


def require_context(context):
    if not isinstance(context, TaskContext):
        raise state.TaskStateError("A live task worker is required for commands.")
    task = context.check()
    if task["role"] != "implementer" or task["snapshot"].get("commandsEnabled") is not True:
        raise state.TaskStateError("This task is not permitted to run commands.")
    return task


def begin(context, argv, timeout):
    task = require_context(context)
    from .review import redact_review_text
    with state._transaction() as (conn, now):
        owned = state._owned(conn, task["id"], context._work.owner, now)
        if owned["project_id"] != task["projectId"]:
            raise state.TaskStateError("Task command ownership changed.")
        _schema(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM studio_task_commands WHERE task_id=?", (task["id"],)
        ).fetchone()[0]
        if count >= MAX_COMMANDS:
            raise state.TaskStateError("This attempt has used its six command executions.")
        command_id, receipt = str(uuid.uuid4()), secrets.token_hex(32)
        conn.execute(
            "INSERT INTO studio_task_commands(id,task_id,project_id,sequence,receipt,argv_json,timeout,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                command_id,
                task["id"],
                task["projectId"],
                count + 1,
                receipt,
                json.dumps([redact_review_text(arg) for arg in argv]),
                timeout,
                "running",
                now,
            ),
        )
        return command_id, receipt


def finish(
    command_id,
    receipt,
    status,
    *,
    exit_code = None,
    output = "",
    output_bytes = 0,
    truncated = False,
):
    if status not in {
        "passed",
        "failed",
        "cancelled",
        "timed_out",
        "unavailable",
        "interrupted",
        "containment_pending",
    }:
        raise state.TaskStateError("Invalid command outcome.")
    from .review import redact_review_text

    # Include the supervisor's truncation notice within the persisted cap too.
    rendered = redact_review_text(output)
    payload = rendered.encode("utf-8")
    truncated = truncated or len(payload) > MAX_OUTPUT_BYTES
    rendered = payload[:MAX_OUTPUT_BYTES].decode("utf-8", errors = "ignore")
    with state._transaction() as (conn, now):
        _schema(conn)
        conn.execute(
            "UPDATE studio_task_commands SET status=?,exit_code=?,output=?,output_bytes=?,truncated=?,completed_at=? "
            "WHERE id=? AND receipt=? AND status='running'",
            (status, exit_code, rendered, output_bytes, int(truncated), now, command_id, receipt),
        )


def _public(row, task, *, summary):
    result = {
        "id": row["id"],
        "taskId": row["task_id"],
        "sequence": row["sequence"],
        "argv": json.loads(row["argv_json"]),
        "timeout": row["timeout"],
        "status": row["status"],
        "exitCode": row["exit_code"],
        "output": row["output"],
        "outputBytes": row["output_bytes"],
        "outputTruncated": bool(row["truncated"]),
        "createdAt": row["created_at"],
        "completedAt": row["completed_at"],
    }
    if result["status"] == "running" and task["status"] in state.TERMINAL:
        result["status"] = "interrupted"
        result["output"] = (
            "Worker ended without a confirmed command outcome. Process cleanup may still be pending."
        )
    result["previewTruncated"] = summary and len(result["output"]) > 4096
    if summary:
        result["output"] = result["output"][:4096]
    return result


def read(
    project_id,
    task_id,
    command_id = None,
    *,
    summary = False,
):
    task = state.get_task(project_id, task_id)
    with state._transaction() as (conn, _now):
        _schema(conn)
        rows = conn.execute(
            "SELECT * FROM studio_task_commands WHERE project_id=? AND task_id=? ORDER BY sequence LIMIT ?",
            (project_id, task_id, MAX_COMMANDS),
        ).fetchall()
    if command_id is not None:
        for row in rows:
            if row["id"] == command_id:
                return _public(row, task, summary = summary)
        raise state.TaskStateError("Task command not found.")
    return [_public(row, task, summary = True) for row in rows]
