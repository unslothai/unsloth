# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hold project retirement until owned Git state and active writers are accounted for."""

import os
import time

from .git_context import AgentWorkspaceError
from .git_state import list_checkpoints, list_worktrees, set_git_retirement
from .prepared_commit_state import list_ref_bearing_preparations
from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence
from .worktrees import begin_project_deletion, finish_project_deletion


def begin_git_retirement(project_id: str, *, deleting: bool = False):
    set_git_retirement(project_id, True)
    begun = False
    descriptor = None
    try:
        begin_project_deletion(project_id)
        begun = True
        if os.name == "posix":
            descriptor = _acquire_project_execution_fence(
                "project:" + project_id, None, time.monotonic() + 30
            )
        if deleting:
            if any(item["status"] != "removed" for item in list_worktrees(project_id)):
                raise AgentWorkspaceError(
                    "Remove or recover the project's owned worktrees before deleting it."
                )
            if list_checkpoints(project_id) or list_ref_bearing_preparations(project_id):
                raise AgentWorkspaceError(
                    "Remove the project's checkpoint and prepared commit refs before deleting it."
                )
        return descriptor
    except Exception:
        if descriptor is not None:
            _release_project_execution_fence(descriptor)
        if begun:
            finish_project_deletion(project_id)
        set_git_retirement(project_id, False)
        raise


def finish_git_retirement(project_id: str, descriptor):
    try:
        set_git_retirement(project_id, False)
    finally:
        if descriptor is not None:
            _release_project_execution_fence(descriptor)
        finish_project_deletion(project_id)
