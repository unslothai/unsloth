# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One lifecycle entry point for optional project features during archive/delete."""

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional


PROJECT_RETIREMENT_PROTOCOL = 1
PROJECT_TASK_RETIREMENT_PROTOCOL = 1


@dataclass
class ProjectRetirement:
    project_id: str
    release: Optional[Callable[[], None]] = None


def _feature(name: str):
    module = "core.agent_workspace." + name
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in {module, "core.agent_workspace"}:
            raise
        return None


def _begin_workspace_retirement(project_id: str, *, deleting: bool = False) -> ProjectRetirement:
    """Leave existing projects unchanged when no optional retirement feature exists.

    Git retirement also retires verification when present; choosing it first
    avoids nesting two owners of the same process-shared execution fence.
    """
    git = _feature("git_retirement")
    if git is not None:
        token = git.begin_git_retirement(project_id, deleting = deleting)
        return ProjectRetirement(project_id, partial(git.finish_git_retirement, project_id, token))
    verification = _feature("verification")
    if verification is None:
        return ProjectRetirement(project_id)
    verification.begin_project_deletion(project_id)
    try:
        verification.cancel_project_verifications_and_wait(project_id)
    except BaseException:
        verification.finish_project_deletion(project_id)
        raise
    return ProjectRetirement(project_id, partial(verification.finish_project_deletion, project_id))


def begin_project_retirement(project_id: str, *, deleting: bool = False) -> ProjectRetirement:
    # Cancel/drain tasks BEFORE acquiring Git's exclusive execution fence. A
    # running task may need that same fence to finish its current operation.
    tasks = _feature("task_service")
    if tasks is None:
        return _begin_workspace_retirement(project_id, deleting = deleting)
    task_token = tasks.begin_task_retirement(project_id)
    try:
        workspace = _begin_workspace_retirement(project_id, deleting = deleting)
    except BaseException:
        tasks.finish_task_retirement(project_id, task_token)
        raise

    def release():
        try:
            finish_project_retirement(project_id, workspace)
        finally:
            tasks.finish_task_retirement(project_id, task_token)

    return ProjectRetirement(project_id, release)


def finish_project_retirement(project_id: str, token: ProjectRetirement) -> None:
    if token.project_id != project_id:
        raise RuntimeError("Project retirement ownership changed.")
    release = token.release
    token.release = None
    if release is not None:
        release()
