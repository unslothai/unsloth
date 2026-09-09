# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coordinate Git writes with project retirement and supervised tool execution."""

import threading
import time
from contextlib import contextmanager

from .git_context import AgentWorkspaceError, project_workspace_access
from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence


def project_retirement_available() -> bool:
    try:
        from core.project_retirement import PROJECT_RETIREMENT_PROTOCOL
    except ImportError:
        return False
    return PROJECT_RETIREMENT_PROTOCOL == 1


@contextmanager
def project_git_guard(project_id: str):
    if not project_retirement_available():
        raise AgentWorkspaceError(
            "Git changes are unavailable until project lifecycle support is installed."
        )
    from .git_state import require_git_admission, set_git_retirement
    with project_workspace_access(project_id) as workspace:
        try:
            from .mutation import acquire_workspace_mutation_slot, release_workspace_mutation_slot
        except ModuleNotFoundError as exc:
            if exc.name != __package__ + ".mutation":
                raise
            acquire_workspace_mutation_slot = release_workspace_mutation_slot = None
        identity = (workspace.device_id, workspace.file_id)
        cancelled = threading.Event()
        timer = threading.Timer(30, cancelled.set)
        timer.daemon = True
        acquired = False
        fence = None
        timer.start()
        try:
            if acquire_workspace_mutation_slot is not None:
                acquired = acquire_workspace_mutation_slot(identity, cancelled)
                if not acquired:
                    raise AgentWorkspaceError(
                        "A project tool is still running. Try Git review again."
                    )
            fence = _acquire_project_execution_fence(
                "project:" + project_id,
                cancelled,
                time.monotonic() + 30,
            )
            # Holding the OS fence also permits recovery of a crashed retirement owner.
            set_git_retirement(project_id, False)
            require_git_admission(project_id)
            from .git_service import git_root, repository_fence

            with repository_fence(git_root(workspace.root)):
                yield workspace
        except (InterruptedError, TimeoutError, OSError) as exc:
            raise AgentWorkspaceError("Project Git execution is busy or unavailable.") from exc
        finally:
            timer.cancel()
            if fence is not None:
                _release_project_execution_fence(fence)
            if acquired:
                release_workspace_mutation_slot(identity)
