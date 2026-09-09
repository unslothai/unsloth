# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hold project retirement until owned Git state and active writers are accounted for."""

import importlib
import os
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

from .git_context import AgentWorkspaceError
from .git_state import list_checkpoints, list_worktrees, set_git_retirement
from .prepared_commit_state import list_ref_bearing_preparations
from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence
from .worktrees import begin_project_deletion, finish_project_deletion


@dataclass
class _GitRetirement:
    descriptor: Optional[int] = None
    verification: Optional[ModuleType] = None


def begin_git_retirement(project_id: str, *, deleting: bool = False):
    set_git_retirement(project_id, True)
    begun = False
    retirement = _GitRetirement()
    try:
        begin_project_deletion(project_id)
        begun = True
        try:
            verification = importlib.import_module(".verification", __package__)
        except ModuleNotFoundError as exc:
            if exc.name != __package__ + ".verification":
                raise
            verification = None
        if verification is not None:
            # Verification cancellation owns the same process fence. Retire its
            # admission first, then let it stop the process tree and acquire it
            # once; nesting two flock descriptors would deadlock this request.
            verification.begin_project_deletion(project_id)
            retirement.verification = verification
            verification.cancel_project_verifications_and_wait(project_id)
        elif os.name == "posix":
            retirement.descriptor = _acquire_project_execution_fence(
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
        return retirement
    except BaseException:
        if begun:
            finish_git_retirement(project_id, retirement)
        else:
            set_git_retirement(project_id, False)
        raise


def finish_git_retirement(project_id: str, retirement: _GitRetirement):
    try:
        set_git_retirement(project_id, False)
    finally:
        try:
            if retirement.descriptor is not None:
                _release_project_execution_fence(retirement.descriptor)
        finally:
            try:
                if retirement.verification is not None:
                    retirement.verification.finish_project_deletion(project_id)
            finally:
                finish_project_deletion(project_id)
