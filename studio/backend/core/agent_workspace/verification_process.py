# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verification's restrictive adapter to the secure command prerequisite.

Importing review/history APIs does not enable commands. Execution requires the
native supervisor, including its reviewed-command preflight and inherited
process-tree fence. There is no shell fallback when that prerequisite is absent.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Optional

from . import verification_context as common
from .verification_context import AgentWorkspaceError


class ProjectExecutionUnavailable(AgentWorkspaceError):
    """The secure command prerequisite or native boundary is unavailable."""


@dataclass(frozen = True)
class ExecutionBoundaryStatus:
    available: bool
    backend: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen = True)
class ProjectProcessResult:
    status: str
    exit_code: Optional[int]
    output: str
    output_bytes: int
    output_truncated: bool
    truncation_notice: str = ""


def _native_runner():
    try:
        from core.project_retirement import PROJECT_RETIREMENT_PROTOCOL
    except ImportError as exc:
        raise ProjectExecutionUnavailable(
            "Project lifecycle support must be installed before verification can run."
        ) from exc
    if PROJECT_RETIREMENT_PROTOCOL != 1:
        raise ProjectExecutionUnavailable(
            "Project lifecycle support must be updated before verification can run."
        )
    try:
        runner = importlib.import_module(__package__ + ".supervisor")
    except ModuleNotFoundError as exc:
        if exc.name != __package__ + ".supervisor":
            raise
        raise ProjectExecutionUnavailable(
            "Secure project commands must be installed before verification or hooks can run."
        ) from exc
    if "before_start" not in inspect.signature(
        runner.run_project_process
    ).parameters or not hasattr(runner, "_acquire_project_execution_fence"):
        raise ProjectExecutionUnavailable(
            "The secure command runner needs reviewed-command and process-tree fencing support."
        )
    return runner


def supervised_process_status():
    try:
        return _native_runner().supervised_process_status()
    except ProjectExecutionUnavailable as exc:
        return ExecutionBoundaryStatus(False, reason = str(exc))


def run_project_process(project_id, argv, **options):
    return _native_runner().run_project_process(project_id, argv, **options)


def _run_project_verification_process(capability, **options):
    from . import verification  # noqa: PLC0415

    if (
        not isinstance(capability, verification._VerificationProcessCapability)
        or capability._seal is not verification._CAPABILITY_SEAL
    ):
        raise AgentWorkspaceError("Project verification authority changed before execution.")

    def revalidate(opened_workspace, argv):
        workspace = common.project_workspace(capability.project_id)
        if (
            tuple(argv) != capability.argv
            or opened_workspace.root != workspace.root
            or opened_workspace.device_id != workspace.device_id
            or opened_workspace.file_id != workspace.file_id
        ):
            raise AgentWorkspaceError("Project verification workspace changed before execution.")
        verification._revalidate_verification_capability(capability, workspace)

    return run_project_process(
        capability.project_id, capability.argv, before_start = revalidate, **options
    )


from .process_fence import (  # noqa: E402
    _acquire_project_execution_fence,
    _release_project_execution_fence,
)
