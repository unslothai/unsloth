# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind one attempt to one owned checkout, retaining native fences until it exits."""

from __future__ import annotations

import os
import logging
import time
import difflib
import stat
from contextlib import contextmanager

from storage.studio_db import get_connection
from .task_state import TaskStateError

TASK_COMMAND_BINDING_PROTOCOL = 1


def acquire_task_project_fence(
    project_id: str,
    *,
    exclusive: bool,
    deadline: float,
    cancel_event = None,
):
    """Shared for physical executors, exclusive for retirement across processes."""
    if os.name != "posix":
        if exclusive:
            # This platform cannot start a worktree task, but existing projects
            # must still be archivable when the optional feature is installed.
            return None
        raise TaskStateError("Owned task worktrees are unavailable on this platform.")
    import fcntl
    from .process_fence import _project_execution_fence_path

    path = _project_execution_fence_path("project-tasks:" + project_id)
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TaskStateError("Project task fence storage is unsafe.")
        while True:
            if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= deadline:
                raise TaskStateError(
                    "A project task is still running or the operation was cancelled."
                )
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(0.05)
                else:
                    time.sleep(0.05)
    except BaseException:
        os.close(fd)
        raise


def release_task_project_fence(descriptor):
    if descriptor is not None:
        from .process_fence import _release_project_execution_fence
        _release_project_execution_fence(descriptor)


@contextmanager
def _task_project_fence(context):
    context.check()
    fd = acquire_task_project_fence(
        context.task["projectId"],
        exclusive = False,
        deadline = context.deadline,
        cancel_event = context.cancel_event,
    )
    try:
        context.check()
        yield
    finally:
        release_task_project_fence(fd)


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS studio_task_workspaces (
        task_id TEXT PRIMARY KEY REFERENCES studio_project_tasks(id) ON DELETE CASCADE,
        project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
        worktree_id TEXT NOT NULL UNIQUE,
        device INTEGER NOT NULL, inode INTEGER NOT NULL)""")


def bindings(project_id: str, task_ids: list[str]) -> dict[str, dict]:
    if not task_ids:
        return {}
    conn = get_connection()
    try:
        _schema(conn)
        placeholders = ",".join("?" for _ in task_ids)
        rows = conn.execute(
            f"SELECT * FROM studio_task_workspaces WHERE project_id=? AND task_id IN ({placeholders})",
            (project_id, *task_ids),
        ).fetchall()
        conn.commit()
        return {row["task_id"]: dict(row) for row in rows}
    finally:
        conn.close()


def binding(project_id: str, task_id: str) -> dict | None:
    return bindings(project_id, [task_id]).get(task_id)


def capture_workspace(project_id: str) -> dict:
    from .git_context import project_workspace
    from .git_service import git_root, repository_head
    from .worktrees import project_operation

    if os.name != "posix":
        raise TaskStateError("Owned task worktrees are unavailable on this platform.")
    with project_operation(project_id):
        workspace = project_workspace(project_id)
        if git_root(workspace.root) != workspace.root:
            raise TaskStateError("Tasks require a project that owns its repository root.")
        return {
            "device": workspace.device_id,
            "inode": workspace.file_id,
            "revision": workspace.revision,
            "head": repository_head(workspace.root),
        }


def validate_workspace(project_id: str, snapshot: dict):
    from .git_context import project_workspace

    workspace = project_workspace(project_id)
    if [workspace.device_id, workspace.file_id, workspace.revision] != [
        snapshot["device"],
        snapshot["inode"],
        snapshot["revision"],
    ]:
        raise TaskStateError("The project workspace changed. Start a new task.")
    return workspace


@contextmanager
def worktree_idle_guard(project_id: str, worktree_id: str):
    """Called inside Git's project_operation, also serializing new task binding.

    An expired database lease does not release this OS lock while a tool lives.
    """
    from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence
    from .git_context import AgentWorkspaceError

    try:
        descriptor = _acquire_project_execution_fence(
            "task-worktree:" + project_id + ":" + worktree_id, None, time.monotonic() + 0.1
        )
    except TimeoutError:
        raise AgentWorkspaceError(
            "A task is still using this worktree. Cancel it and wait for it to stop."
        ) from None
    try:
        yield
    finally:
        _release_project_execution_fence(descriptor)


@contextmanager
def task_workspace(context):
    from .common import ProjectWorkspace
    from .git_context import project_workspace_access
    from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence
    from .worktrees import create_worktree, cleanup_worktree, owned_worktree_path, project_operation

    task = context.check()
    project_id = task["projectId"]
    descriptor = None
    # This session lease prevents the project folder from being retired while
    # the executor or its tool worker still has the checkout open.
    with _task_project_fence(context), project_workspace_access(project_id):
        try:
            with project_operation(project_id):
                context.check()
                validate_workspace(project_id, task["snapshot"]["workspace"])
                if binding(project_id, task["id"]) is not None:
                    raise TaskStateError("This attempt already has a checkout. Retry explicitly.")
                record = create_worktree(project_id, base_ref = task["snapshot"]["workspace"]["head"])
                bound = False
                try:
                    root = owned_worktree_path(project_id, record["id"])
                    descriptor = _acquire_project_execution_fence(
                        "task-worktree:" + project_id + ":" + record["id"],
                        context.cancel_event,
                        context.deadline,
                    )
                    metadata = root.stat(follow_symlinks = False)
                    conn = get_connection()
                    try:
                        _schema(conn)
                        context.check()
                        conn.execute(
                            "INSERT INTO studio_task_workspaces VALUES(?,?,?,?,?)",
                            (
                                task["id"],
                                project_id,
                                record["id"],
                                metadata.st_dev,
                                metadata.st_ino,
                            ),
                        )
                        conn.commit()
                        bound = True
                    finally:
                        conn.close()
                except BaseException:
                    if not bound:
                        # No task tool has received this checkout. Release our
                        # own lease before cleanup reacquires the same fence,
                        # while retaining project_operation across the rollback.
                        if descriptor is not None:
                            _release_project_execution_fence(descriptor)
                            descriptor = None
                        try:
                            cleanup_worktree(project_id, record["id"])
                        except Exception:
                            # Cleanup proves ownership and refuses dirty/unsafe
                            # paths. Preserve that durable Git record for recovery.
                            logging.getLogger(__name__).warning(
                                "Task checkout setup failed; worktree %s requires recovery.",
                                record["id"],
                            )
                    raise
            workspace = ProjectWorkspace(
                project_id, root, "managed", metadata.st_dev, metadata.st_ino
            )
            yield workspace, record["id"]
        finally:
            if descriptor is not None:
                _release_project_execution_fence(descriptor)


def review_task_workspace(project_id: str, task_id: str) -> dict:
    """Read a stopped attempt's tracked diff and bounded new-file previews."""
    from .common import ProjectWorkspace
    from .git_service import repository_command
    from .mutation import ProjectFileMutation
    from .review import redact_review_text
    from .worktrees import owned_worktree_path, project_operation
    from .task_state import get_task

    task = get_task(project_id, task_id)
    record = binding(project_id, task_id)
    if record is None:
        raise TaskStateError("This attempt has no owned worktree.")
    with project_operation(project_id), worktree_idle_guard(project_id, record["worktree_id"]):
        root = owned_worktree_path(project_id, record["worktree_id"])
        workspace = ProjectWorkspace(project_id, root, "managed", record["device"], record["inode"])
        metadata = root.stat(follow_symlinks = False)
        if (metadata.st_dev, metadata.st_ino) != (record["device"], record["inode"]):
            raise TaskStateError("The task worktree identity changed.")
        diff, truncated = repository_command(
            root,
            ["diff", "--no-ext-diff", "--no-textconv", task["snapshot"]["workspace"]["head"], "--"],
            output_limit = 128 * 1024,
            neutralize_filters = True,
        )
        names, names_truncated = repository_command(
            root, ["ls-files", "--others", "--exclude-standard", "-z"], output_limit = 16 * 1024
        )
        new_files = []
        paths = names.split("\0")[:-1]
        preview_bytes = 0
        for path in paths[:20]:
            if preview_bytes >= 64 * 1024:
                truncated = True
                break
            try:
                with ProjectFileMutation.open(workspace, path, max_bytes = 16 * 1024) as mutation:
                    content = mutation.read(16 * 1024)[0].decode("utf-8")
                patch = "".join(
                    difflib.unified_diff(
                        [], content.splitlines(keepends = True), fromfile = "/dev/null", tofile = path
                    )
                )
                preview_bytes += len(patch.encode("utf-8"))
                new_files.append(
                    {"path": path, "diff": redact_review_text(patch, root), "unavailable": False}
                )
            except Exception:
                new_files.append({"path": path, "diff": "", "unavailable": True})
        return {
            "worktreeId": record["worktree_id"],
            "diff": redact_review_text(diff, root),
            "newFiles": new_files,
            "truncated": truncated or names_truncated or len(paths) > len(new_files),
        }


__all__ = [
    "binding",
    "capture_workspace",
    "validate_workspace",
    "task_workspace",
    "worktree_idle_guard",
]
