# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server-owned task admission, public views, and project retirement."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from storage import studio_db
from . import task_state as state
from .task_runner import ProjectTaskRunner
from .task_runtime import capture_runtime, validate_runtime
from .task_workspaces import (
    binding,
    bindings,
    capture_workspace,
    validate_workspace,
    acquire_task_project_fence,
    release_task_project_fence,
)

_lock = threading.RLock()
_runner: ProjectTaskRunner | None = None
_closing = False


def require_prerequisites():
    from core.project_retirement import PROJECT_TASK_RETIREMENT_PROTOCOL
    from .worktrees import TASK_WORKTREE_GUARD_PROTOCOL
    if PROJECT_TASK_RETIREMENT_PROTOCOL != 1 or TASK_WORKTREE_GUARD_PROTOCOL != 1:
        raise state.TaskStateError("Project task lifecycle support is unavailable.")


def runner() -> ProjectTaskRunner:
    global _runner
    require_prerequisites()
    with _lock:
        if _closing:
            raise state.TaskStateError("Project tasks are shutting down.")
        if _runner is None:
            from .task_executor import execute_task
            _runner = ProjectTaskRunner(execute_task)
        return _runner


def submit(
    project_id: str,
    instruction: str,
    *,
    kind: str,
    model: str,
    provider_id = None,
    max_output_tokens: int = 8192,
    child_limit: int = 2,
    child_budget: int = 8192,
    timeout: int = 900,
    allow_commands: bool = False,
) -> dict:
    require_prerequisites()
    if not isinstance(allow_commands, bool):
        raise state.TaskStateError("Command opt-in must be a boolean.")
    if allow_commands:
        _require_commands()
    runtime = capture_runtime(kind, model, provider_id)
    workspace = capture_workspace(project_id)
    project = studio_db.get_chat_project(project_id)
    if project is None or project.get("archived"):
        raise state.TaskStateError("Project not found.")
    snapshot = {
        "runtime": runtime,
        "commandsEnabled": allow_commands,
        "workspace": workspace,
        "instructions": str(project.get("instructions") or "")[:24000],
    }
    return public_task(
        runner().submit(
            project_id,
            instruction,
            snapshot,
            max_output_tokens = max_output_tokens,
            child_limit = child_limit,
            child_budget = child_budget,
            timeout = timeout,
        )
    )


def public_tasks(project_id: str, *, limit: int = 100) -> list[dict]:
    tasks = state.list_tasks(project_id, limit = limit)
    owned = bindings(project_id, [task["id"] for task in tasks])
    return [public_task(task, summary = True, owned = owned) for task in tasks]


def public_task(
    task: dict,
    *,
    summary: bool = False,
    owned: dict | None = None,
) -> dict:
    # Return model labels, never routing/credential fingerprints, root identity,
    # full project instructions, owner tokens, or raw workspace paths.
    snapshot = task["snapshot"]
    result = {key: value for key, value in task.items() if key != "snapshot"}
    runtime = snapshot.get("runtime", {})
    result["commandsEnabled"] = snapshot.get("commandsEnabled", False)
    result["runtime"] = {key: runtime.get(key) for key in ("kind", "model", "providerId")}
    record = binding(task["projectId"], task["id"]) if owned is None else owned.get(task["id"])
    result["worktreeId"] = record["worktree_id"] if record else None
    if summary and result.get("result"):
        output = str(result["result"].get("output") or "")
        result["result"] = {"output": output[:4096]}
        result["resultTruncated"] = len(output) > 4096
    return result


def _require_commands():
    try:
        from .task_commands import require_support
    except ImportError:
        raise state.TaskStateError("Task command support is not installed.") from None
    require_support()


def retry(project_id: str, task_id: str) -> dict:
    task = state.get_task(project_id, task_id)
    if task["snapshot"].get("commandsEnabled"):
        _require_commands()
    validate_runtime(task["snapshot"]["runtime"])
    validate_workspace(project_id, task["snapshot"]["workspace"])
    return public_task(runner().retry(project_id, task_id))


def cancel(project_id: str, task_id: str) -> dict:
    with _lock:
        current = _runner
    return public_task(
        current.cancel(project_id, task_id) if current else state.cancel_task(project_id, task_id)
    )


def shutdown() -> bool:
    global _closing
    with _lock:
        _closing = True
        current = _runner
    return current is None or current.shutdown(timeout = 10)


@dataclass
class TaskRetirement:
    project_id: str
    owner: str = field(default_factory = lambda: str(uuid.uuid4()))
    stop: threading.Event = field(default_factory = threading.Event)
    thread: threading.Thread | None = None
    descriptor: int | None = None


def begin_task_retirement(project_id: str) -> TaskRetirement:
    token = TaskRetirement(project_id)
    state.begin_project_retirement(project_id, token.owner)
    try:
        deadline = time.monotonic() + 10
        with _lock:
            current = _runner
        # DB cancellation covered every task, including rows older than a list
        # page. The local watcher signals every physical executor on renewal.
        while not (
            current.project_is_idle(project_id)
            if current
            else not state.project_has_active_tasks(project_id)
        ):
            if time.monotonic() >= deadline:
                raise state.TaskStateError("A project task has not stopped. Retry after it exits.")
            state.renew_project_retirement(project_id, token.owner)
            token.stop.wait(0.05)

        # A different process may have a revoked DB lease while its native tool
        # still runs. Require the physical shared leases to drain as well.
        token.descriptor = acquire_task_project_fence(project_id, exclusive = True, deadline = deadline)

        def renew():
            while not token.stop.wait(1):
                try:
                    state.renew_project_retirement(project_id, token.owner)
                except Exception:
                    # The exclusive OS fence still blocks new physical executors
                    # until the archive/delete transaction and Git cleanup finish.
                    return

        token.thread = threading.Thread(target = renew, daemon = True, name = "project-task-retirement")
        token.thread.start()
        return token
    except BaseException:
        finish_task_retirement(project_id, token)
        raise


def finish_task_retirement(project_id: str, token: TaskRetirement):
    if token.project_id != project_id:
        raise state.TaskStateError("Project task retirement ownership changed.")
    token.stop.set()
    if token.thread:
        token.thread.join(timeout = 2)
    try:
        state.finish_project_retirement(project_id, token.owner)
    finally:
        release_task_project_fence(token.descriptor)
        token.descriptor = None
