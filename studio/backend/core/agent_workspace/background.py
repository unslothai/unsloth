# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded in-process scheduling over durable background-task records."""

import os
import stat
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from storage.studio_db import get_chat_project

from .common import AgentWorkspaceError, project_workspace
from .state import (
    claim_background_task,
    create_agent_background_task,
    create_background_task,
    get_background_task,
    get_verification_config,
    list_all_active_background_tasks,
    list_active_background_tasks,
    list_active_child_tasks,
    retry_background_task,
    update_background_task,
)
from .verification import run_project_verification
from .worktrees import (
    cleanup_worktree,
    owned_worktree_path,
    sync_worktree_background_task_marker,
)


@dataclass(frozen = True)
class AgentTaskContext:
    """Immutable context supplied to a registered provider or coding-agent adapter."""

    task_id: str
    project_id: str
    instruction: str
    runtime_snapshot: Optional[dict]
    goal_snapshot: Optional[str]
    goal_status_snapshot: Optional[str]
    goal_updated_at: Optional[int]
    plan_id: Optional[str]
    plan_revision: Optional[int]
    plan_task_id: Optional[str]
    plan_snapshot: Optional[dict]
    worktree_id: Optional[str]
    cwd: Path
    expected_root_identity: Optional[tuple[int, int]]
    project_root: Path
    expected_project_root_identity: tuple[int, int]
    project_workspace_binding: tuple[Optional[str], ...]
    parent_task_id: Optional[str]
    root_task_id: str
    delegation_role: Optional[str]
    delegation_depth: int
    delegation_budget: Optional[dict]

    def run_command(
        self,
        command: str,
        cancel_event: threading.Event,
        *,
        timeout_seconds: int = 3600,
        log_limit_bytes: int = 1024 * 1024,
    ) -> dict:
        """Run a CLI adapter inside the project boundary with process-tree cancellation."""
        from .verification import execute_check
        return execute_check(
            {
                "name": "agent",
                "kind": "custom",
                "command": command,
                "required": True,
                "timeoutSeconds": timeout_seconds,
                "logLimitBytes": log_limit_bytes,
                "projectId": self.project_id,
            },
            root = self.cwd,
            cancel_event = cancel_event,
            run_id = f"agent:{self.task_id}",
            expected_root_identity = self.expected_root_identity,
        )


AgentTaskExecutor = Callable[[AgentTaskContext, threading.Event], dict[str, Any]]


def _bounded_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise AgentWorkspaceError("Agent executor returned an invalid result.")
    bounded = dict(result)
    output = bounded.get("output")
    if isinstance(output, str):
        encoded = output.encode("utf-8", errors = "replace")
        limit = 900 * 1024
        reported_bytes = bounded.get("outputBytes")
        if not (
            bounded.get("outputTruncated") is True
            and isinstance(reported_bytes, int)
            and reported_bytes >= len(encoded)
        ):
            bounded["outputBytes"] = len(encoded)
        if len(encoded) > limit:
            bounded["output"] = encoded[:limit].decode("utf-8", errors = "replace")
            bounded["outputTruncated"] = True
        else:
            bounded.setdefault("outputTruncated", False)
    return bounded


class BackgroundTaskManager:
    """Runs durable verification and agent tasks with a bounded scheduler."""

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(
            max_workers = max_workers, thread_name_prefix = "studio-agent-task"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._verification_runs: dict[str, str] = {}
        self._deleting_projects: set[str] = set()
        self._agent_executor: Optional[AgentTaskExecutor] = None

    def register_agent_executor(self, executor: Optional[AgentTaskExecutor]) -> None:
        """Register the provider-neutral adapter used by subsequently started jobs."""
        if executor is not None and not callable(executor):
            raise AgentWorkspaceError("Agent executor must be callable.")
        with self._lock:
            self._agent_executor = executor

    def begin_project_deletion(self, project_id: str) -> None:
        """Fence new work while the project deletion preflight is in progress."""
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is already in progress.")
            self._deleting_projects.add(project_id)

    def finish_project_deletion(self, project_id: str) -> None:
        with self._lock:
            self._deleting_projects.discard(project_id)

    def _require_project_available_locked(self, project_id: str) -> None:
        if project_id in self._deleting_projects:
            raise AgentWorkspaceError("Project deletion is in progress.")

    def enqueue_verification(
        self,
        project_id: str,
        selected_names: Optional[list[str]] = None,
        *,
        worktree_id: Optional[str] = None,
        config_revision: Optional[int] = None,
        start: bool = True,
    ) -> dict:
        with self._lock:
            self._require_project_available_locked(project_id)
            revision = (
                get_verification_config(project_id)["revision"]
                if config_revision is None
                else config_revision
            )
            task = create_background_task(
                project_id,
                "verification",
                {
                    "selectedNames": selected_names,
                    "worktreeId": worktree_id,
                    "configRevision": revision,
                },
            )
        return self.start(task["id"]) if start else task

    def enqueue_dream(
        self,
        project_id: str,
        *,
        thread_ids: list[str],
        instructions: str = "",
        start: bool = True,
    ) -> dict:
        """Queue an asynchronous, review-only transcript curation pass."""
        with self._lock:
            self._require_project_available_locked(project_id)
            task = create_background_task(
                project_id,
                "dream",
                {
                    "threadIds": list(thread_ids),
                    "instructions": instructions,
                },
            )
        return self.start(task["id"]) if start else task

    def enqueue_agent(
        self,
        project_id: str,
        instruction: str,
        *,
        task_id: Optional[str] = None,
        runtime_selection: Optional[dict] = None,
        runtime_snapshot: Optional[dict] = None,
        plan_id: Optional[str] = None,
        plan_task_id: Optional[str] = None,
        worktree_id: Optional[str] = None,
        cleanup_worktree_on_cancel: bool = False,
        delegation_policy: Optional[dict] = None,
        start: bool = True,
    ) -> dict:
        with self._lock:
            self._require_project_available_locked(project_id)
            if start and self._agent_executor is None:
                raise AgentWorkspaceError(
                    "No background agent executor is registered for this runtime."
                )
            if runtime_selection is not None and runtime_snapshot is not None:
                raise AgentWorkspaceError(
                    "Provide either a runtime selection or a durable runtime snapshot, not both."
                )
            if runtime_selection is not None:
                from .inference_executor import capture_runtime_snapshot
                runtime_snapshot = capture_runtime_snapshot(runtime_selection)
            elif runtime_snapshot is not None:
                from .inference_executor import validate_runtime_snapshot
                runtime_snapshot = validate_runtime_snapshot(runtime_snapshot)
            task = create_agent_background_task(
                project_id,
                instruction,
                task_id = task_id,
                runtime_snapshot = runtime_snapshot,
                plan_id = plan_id,
                plan_task_id = plan_task_id,
                worktree_id = worktree_id,
                cleanup_worktree_on_cancel = cleanup_worktree_on_cancel,
                delegation_policy = delegation_policy,
            )
            if worktree_id:
                try:
                    sync_worktree_background_task_marker(project_id, worktree_id, task["id"])
                except Exception:
                    update_background_task(
                        task["id"],
                        "cancelled",
                        error = "The durable worktree task link could not be finalized.",
                    )
                    raise
        return self.start(task["id"]) if start else task

    def enqueue_child_agent(
        self,
        project_id: str,
        parent_task_id: str,
        instruction: str,
        *,
        role: str,
        budget: dict,
        worktree_id: str,
        cleanup_worktree_on_cancel: bool = False,
        start: bool = True,
    ) -> dict:
        """Queue a role-bound child using the parent's immutable runtime ceiling."""
        with self._lock:
            self._require_project_available_locked(project_id)
            if start and self._agent_executor is None:
                raise AgentWorkspaceError(
                    "No background agent executor is registered for this runtime."
                )
            task = create_agent_background_task(
                project_id,
                instruction,
                worktree_id = worktree_id,
                cleanup_worktree_on_cancel = cleanup_worktree_on_cancel,
                parent_task_id = parent_task_id,
                delegation_role = role,
                delegation_budget = budget,
            )
            try:
                sync_worktree_background_task_marker(project_id, worktree_id, task["id"])
            except Exception:
                update_background_task(
                    task["id"],
                    "cancelled",
                    error = "The durable child worktree link could not be finalized.",
                )
                raise
        return self.start(task["id"]) if start else task

    @staticmethod
    def _project_workspace_binding(project: dict) -> tuple[Optional[str], ...]:
        """Return only the immutable row fields that bind a task to a workspace."""
        return (
            str(project.get("workspaceKind") or "managed"),
            str(project.get("rootPath")) if project.get("rootPath") else None,
            str(project.get("sandboxPath")) if project.get("sandboxPath") else None,
            str(project.get("workspaceDeviceId"))
            if project.get("workspaceDeviceId") is not None
            else None,
            str(project.get("workspaceFileId"))
            if project.get("workspaceFileId") is not None
            else None,
            str(project.get("managedRootDeviceId"))
            if project.get("managedRootDeviceId") is not None
            else None,
            str(project.get("managedRootFileId"))
            if project.get("managedRootFileId") is not None
            else None,
        )

    @staticmethod
    def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
        try:
            metadata = path.stat(follow_symlinks = False)
            resolved = path.resolve(strict = True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AgentWorkspaceError(f"The {label} is unavailable.") from exc
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise AgentWorkspaceError(f"The {label} identity changed.")
        if os.path.normcase(str(resolved)) != os.path.normcase(str(path)):
            raise AgentWorkspaceError(f"The {label} identity changed.")
        return int(metadata.st_dev), int(metadata.st_ino)

    @staticmethod
    def _agent_context(task: dict) -> AgentTaskContext:
        workspace = project_workspace(task["projectId"])
        project = get_chat_project(task["projectId"])
        if project is None:
            raise AgentWorkspaceError("Project not found.")
        row_workspace = project.get(
            "rootPath"
            if str(project.get("workspaceKind") or "managed") == "folder"
            else "sandboxPath"
        )
        try:
            row_workspace_path = Path(str(row_workspace)).expanduser().resolve(strict = True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AgentWorkspaceError(
                "The project workspace binding changed before execution."
            ) from exc
        if os.path.normcase(str(row_workspace_path)) != os.path.normcase(str(workspace.root)):
            raise AgentWorkspaceError("The project workspace binding changed before execution.")
        project_root_identity = BackgroundTaskManager._directory_identity(
            workspace.root, label = "project workspace"
        )
        worktree_id = task.get("worktreeId")
        cwd = (
            owned_worktree_path(
                task["projectId"],
                worktree_id,
                background_task_id = task["id"],
            )
            if worktree_id
            else workspace.root
        )
        if worktree_id:
            expected_identity = BackgroundTaskManager._directory_identity(
                cwd, label = "background agent workspace"
            )
        else:
            expected_identity = project_root_identity
        return AgentTaskContext(
            task_id = task["id"],
            project_id = task["projectId"],
            instruction = str(task["payload"].get("instruction") or ""),
            runtime_snapshot = task["payload"].get("runtime"),
            goal_snapshot = task.get("goalSnapshot"),
            goal_status_snapshot = task.get("goalStatusSnapshot"),
            goal_updated_at = task.get("goalUpdatedAt"),
            plan_id = task.get("planId"),
            plan_revision = task.get("planRevision"),
            plan_task_id = task.get("planTaskId"),
            plan_snapshot = task.get("planSnapshot"),
            worktree_id = worktree_id,
            cwd = cwd,
            expected_root_identity = expected_identity,
            project_root = workspace.root,
            expected_project_root_identity = project_root_identity,
            project_workspace_binding = BackgroundTaskManager._project_workspace_binding(project),
            parent_task_id = task.get("parentTaskId"),
            root_task_id = str(task.get("rootTaskId") or task["id"]),
            delegation_role = task.get("delegationRole"),
            delegation_depth = int(task.get("delegationDepth") or 0),
            delegation_budget = task.get("delegationBudget"),
        )

    @staticmethod
    def _revalidate_agent_completion(task: dict, context: AgentTaskContext) -> None:
        """Fail closed if a task's durable or filesystem binding changed in flight.

        This path deliberately reads the project row directly. Calling
        ``project_workspace`` here could recreate a missing managed workspace just
        before persisting a false successful result.
        """
        current = get_background_task(context.task_id)
        if (
            current is None
            or current.get("kind") != "agent"
            or current.get("status") not in {"running", "cancelling"}
            or current.get("projectId") != context.project_id
            or current.get("projectId") != task.get("projectId")
            or current.get("worktreeId") != context.worktree_id
            or current.get("worktreeId") != task.get("worktreeId")
        ):
            raise AgentWorkspaceError(
                "The background agent task binding changed before completion."
            )

        project = get_chat_project(context.project_id)
        if project is None:
            raise AgentWorkspaceError(
                "The project was removed before the background task completed."
            )
        if (
            BackgroundTaskManager._project_workspace_binding(project)
            != context.project_workspace_binding
        ):
            raise AgentWorkspaceError("The project workspace binding changed before completion.")
        if (
            BackgroundTaskManager._directory_identity(
                context.project_root, label = "project workspace"
            )
            != context.expected_project_root_identity
        ):
            raise AgentWorkspaceError("The project workspace identity changed before completion.")

        from core.inference.tools import (
            background_task_session_id,
            resolve_sandbox_workdir,
        )

        try:
            session_cwd = Path(
                resolve_sandbox_workdir(background_task_session_id(context.task_id))
            ).resolve(strict = True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AgentWorkspaceError(
                "The background agent session binding changed before completion."
            ) from exc
        if os.path.normcase(str(session_cwd)) != os.path.normcase(str(context.cwd)):
            raise AgentWorkspaceError(
                "The background agent session binding changed before completion."
            )

        if context.worktree_id:
            bound_cwd = owned_worktree_path(
                context.project_id,
                context.worktree_id,
                background_task_id = context.task_id,
            )
            if os.path.normcase(str(bound_cwd)) != os.path.normcase(str(context.cwd)):
                raise AgentWorkspaceError(
                    "The background agent worktree binding changed before completion."
                )
        if (
            BackgroundTaskManager._directory_identity(
                context.cwd, label = "background agent workspace"
            )
            != context.expected_root_identity
        ):
            raise AgentWorkspaceError(
                "The background agent workspace identity changed before completion."
            )

    def start(self, task_id: str) -> dict:
        event = threading.Event()
        submit_error: Optional[Exception] = None
        with self._lock:
            task = get_background_task(task_id)
            if task is None:
                raise AgentWorkspaceError("Background task not found.")
            self._require_project_available_locked(task["projectId"])
            if task["status"] != "queued":
                raise AgentWorkspaceError("Only queued background tasks can be started.")
            if task["kind"] not in {"verification", "agent", "dream"}:
                raise AgentWorkspaceError("Unsupported background task kind.")
            executor = self._agent_executor if task["kind"] == "agent" else None
            if task["kind"] == "agent" and executor is None:
                raise AgentWorkspaceError(
                    "No background agent executor is registered for this runtime."
                )
            running = claim_background_task(task_id)
            if running is None:
                raise AgentWorkspaceError("Background task not found.")
            self._cancellations[task_id] = event
            try:
                target = self._run_verification
                args = (task_id, event)
                if task["kind"] == "agent":
                    target = self._run_agent
                    args = (task_id, event, executor)
                elif task["kind"] == "dream":
                    target = self._run_dream
                    args = (task_id, event)
                future = self._executor.submit(target, *args)
                self._futures[task_id] = future
            except Exception as exc:
                self._cancellations.pop(task_id, None)
                self._futures.pop(task_id, None)
                submit_error = exc
        if submit_error is not None:
            update_background_task(task_id, "failed", error = str(submit_error))
            # Submission can fail after a parent was claimed while delegated
            # children are already queued or running. Reuse the terminal-parent
            # cancellation path so every descendant is durably fenced too.
            self.cancel(task_id)
            raise submit_error
        future.add_done_callback(lambda _future: self._forget(task_id))
        return running

    def _run_agent(self, task_id: str, event: threading.Event, executor: AgentTaskExecutor) -> None:
        task = get_background_task(task_id)
        if task is None:
            return
        try:
            context = self._agent_context(task)
            result = _bounded_agent_result(executor(context, event))
            current = get_background_task(task_id)
            if current is None:
                return
            status = "cancelled" if event.is_set() else "completed"
            if status == "cancelled":
                # The worktree cleanup guard rejects every live task state. Persist
                # cancellation first, then remove only the now-stopped owned tree
                # and attach that cleanup result to the same terminal task row.
                update_background_task(task_id, "cancelled", result = result)
                result = self._cancelled_worktree_result(task, result)
                update_background_task(task_id, "cancelled", result = result)
            else:
                if list_active_child_tasks(task_id):
                    raise AgentWorkspaceError("The parent agent still has active child agents.")
                self._revalidate_agent_completion(task, context)
                update_background_task(task_id, "completed", result = result)
        except Exception as exc:
            current = get_background_task(task_id)
            if current and current["status"] in {"running", "cancelling"}:
                if event.is_set():
                    update_background_task(task_id, "cancelled", result = {})
                    result = self._cancelled_worktree_result(task, {})
                    update_background_task(task_id, "cancelled", result = result)
                else:
                    # A parent that fails while a delegated child is still live
                    # must not leave that child running without its authority.
                    # Persist the terminal parent state first, which fences any
                    # new child admission, then cancel every descendant that was
                    # already queued or running. Descendant cancellation is best
                    # effort here, but each child is durably fenced before its
                    # process event is signalled.
                    update_background_task(task_id, "failed", error = str(exc))
                    for child in list_active_child_tasks(task_id):
                        self.cancel(child["id"])

    def _run_verification(self, task_id: str, event: threading.Event) -> None:
        task = get_background_task(task_id)
        if task is None:
            return
        try:
            result = run_project_verification(
                task["projectId"],
                task["payload"].get("selectedNames"),
                cancel_event = event,
                on_run_started = lambda run_id: self._remember_run(task_id, run_id),
                worktree_id = task["payload"].get("worktreeId"),
                config_revision = task["payload"].get("configRevision"),
            )
            result["worktreeId"] = task["payload"].get("worktreeId")
            current = get_background_task(task_id)
            if current is None:
                return
            if result["status"] == "cancelled":
                final_status = "cancelled"
            elif result["status"] == "passed":
                final_status = "completed"
            else:
                final_status = "failed"
            update_background_task(task_id, final_status, result = result)
        except Exception as exc:
            current = get_background_task(task_id)
            if current and current["status"] in {"running", "cancelling"}:
                status = "cancelled" if event.is_set() else "failed"
                update_background_task(task_id, status, error = str(exc))

    def _run_dream(self, task_id: str, event: threading.Event) -> None:
        task = get_background_task(task_id)
        if task is None:
            return
        try:
            from .memory import run_dream_task

            result = run_dream_task(task["projectId"], task["payload"], event)
            current = get_background_task(task_id)
            if current is None or current["status"] not in {"running", "cancelling"}:
                return
            status = "cancelled" if event.is_set() else "completed"
            update_background_task(task_id, status, result = result)
        except Exception as exc:
            current = get_background_task(task_id)
            if current and current["status"] in {"running", "cancelling"}:
                update_background_task(
                    task_id,
                    "cancelled" if event.is_set() else "failed",
                    error = str(exc),
                )

    def _remember_run(self, task_id: str, run_id: str) -> None:
        with self._lock:
            self._verification_runs[task_id] = run_id

    def _forget(self, task_id: str) -> None:
        with self._lock:
            self._futures.pop(task_id, None)
            self._cancellations.pop(task_id, None)
            self._verification_runs.pop(task_id, None)

    def cancel(self, task_id: str) -> dict:
        # Children are independent workers but not independent authority. Stop
        # descendants before the parent can settle or release its worktree. Fence
        # the parent first so a concurrent child admission cannot slip between
        # the child listing and the cancellation transition.
        queued_cancelled: Optional[dict] = None
        with self._lock:
            task = get_background_task(task_id)
            if task is None:
                raise AgentWorkspaceError("Background task not found.")
            if task["status"] == "queued":
                queued_cancelled = (
                    update_background_task(task_id, "cancelled", cancel_requested = True) or task
                )
            elif task["status"] in {"running", "cancelling"}:
                if task["status"] == "running":
                    task = (
                        update_background_task(task_id, "cancelling", cancel_requested = True) or task
                    )
                event = self._cancellations.get(task_id)
                if event is not None:
                    event.set()

        for child in list_active_child_tasks(task_id):
            self.cancel(child["id"])

        if queued_cancelled is not None:
            if task["kind"] == "agent":
                result = self._cancelled_worktree_result(task, {})
                queued_cancelled = (
                    update_background_task(task_id, "cancelled", result = result) or queued_cancelled
                )
            return queued_cancelled

        return task

    @staticmethod
    def _cancelled_worktree_result(task: dict, result: dict) -> dict:
        worktree_id = task.get("worktreeId")
        if not worktree_id:
            return result
        cleanup_requested = bool(task.get("payload", {}).get("cleanupWorktreeOnCancel", False))
        if not cleanup_requested:
            return {**result, "worktreeCleanup": "retained"}
        try:
            cleaned = cleanup_worktree(task["projectId"], worktree_id)
            status = "removed" if cleaned.get("status") == "removed" else "retained"
            return {**result, "worktreeCleanup": status}
        except Exception:
            # Dirty, tampered, foreign, or otherwise unproven worktrees are retained.
            return {**result, "worktreeCleanup": "retained_needs_attention"}

    def cancel_project_tasks_and_wait(
        self,
        project_id: str,
        *,
        timeout_seconds: float = 30,
    ) -> list[dict]:
        """Cancel every active project task and wait until no worker can outlive it."""
        if timeout_seconds <= 0:
            raise AgentWorkspaceError("Project task cancellation timed out.")
        deadline = time.monotonic() + timeout_seconds
        tasks = list_active_background_tasks(project_id)
        for task in tasks:
            self.cancel(task["id"])

        with self._lock:
            futures = {
                task["id"]: self._futures.get(task["id"])
                for task in tasks
                if self._futures.get(task["id"]) is not None
            }
        for task_id, future in futures.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentWorkspaceError("Timed out while stopping project background tasks.")
            try:
                future.result(timeout = remaining)
            except FutureTimeoutError as exc:
                raise AgentWorkspaceError(
                    "Timed out while stopping project background tasks."
                ) from exc
            except Exception as exc:
                raise AgentWorkspaceError(
                    f"Failed while stopping background task {task_id}."
                ) from exc

        remaining_tasks = list_active_background_tasks(project_id)
        if remaining_tasks:
            raise AgentWorkspaceError(
                "Project still has active background tasks. Wait for them to stop and retry."
            )
        stopped = []
        for original in tasks:
            current = get_background_task(original["id"])
            if current is not None:
                stopped.append(current)
        return stopped

    def prepare_for_app_exit(self, *, timeout_seconds: float = 10) -> list[dict]:
        """Request cancellation and persist interruption before desktop shutdown."""
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        active = list_all_active_background_tasks()
        for task in active:
            try:
                self.cancel(task["id"])
            except AgentWorkspaceError:
                pass
        with self._lock:
            futures = [self._futures[task["id"]] for task in active if task["id"] in self._futures]
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout = remaining)
            except Exception:
                pass
        interrupted = []
        for task in active:
            current = get_background_task(task["id"])
            if current and current["status"] in {"running", "cancelling"}:
                try:
                    current = (
                        update_background_task(
                            task["id"],
                            "interrupted",
                            error = "Studio exited while the task was active.",
                        )
                        or current
                    )
                except AgentWorkspaceError:
                    current = get_background_task(task["id"]) or current
            interrupted.append(current)
        return interrupted

    def retry(
        self,
        task_id: str,
        *,
        start: bool = True,
    ) -> dict:
        previous = get_background_task(task_id)
        if previous is None:
            raise AgentWorkspaceError("Background task not found.")
        with self._lock:
            self._require_project_available_locked(previous["projectId"])
            task = retry_background_task(task_id)
            if task.get("worktreeId"):
                try:
                    sync_worktree_background_task_marker(
                        task["projectId"], task["worktreeId"], task["id"]
                    )
                except Exception:
                    update_background_task(
                        task["id"],
                        "cancelled",
                        error = "The durable worktree task link could not be finalized.",
                    )
                    raise
        return self.start(task["id"]) if start else task


manager = BackgroundTaskManager()


def register_agent_executor(executor: Optional[AgentTaskExecutor]) -> None:
    """Production registration hook for local, Codex, or external agent adapters."""
    manager.register_agent_executor(executor)


__all__ = [
    "AgentTaskContext",
    "AgentTaskExecutor",
    "BackgroundTaskManager",
    "manager",
    "register_agent_executor",
]
