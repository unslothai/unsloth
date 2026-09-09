# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional verification authority for hook review and execution."""

import importlib
from contextlib import contextmanager

try:
    from .verification_context import AgentWorkspaceError
except ModuleNotFoundError as exc:
    if exc.name != __package__ + ".verification_context":
        raise

    class AgentWorkspaceError(RuntimeError):
        """Hook execution prerequisites are unavailable."""


def _required(name):
    try:
        return importlib.import_module(__package__ + "." + name)
    except ModuleNotFoundError as exc:
        if exc.name != __package__ + "." + name:
            raise
        raise AgentWorkspaceError(
            "Project verification support must be installed before hooks can be reviewed or run."
        ) from exc


def project_workspace(project_id):
    return _required("verification_context").project_workspace(project_id)


@contextmanager
def project_workspace_access(project_id):
    with _required("verification_context").project_workspace_access(project_id) as workspace:
        yield workspace


def execution_status():
    try:
        return _required("verification").execution_status()
    except AgentWorkspaceError as exc:
        return {"available": False, "backend": None, "reason": str(exc)}


class _OptionalModule:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        return getattr(_required(self.name), name)


processes = _OptionalModule("verification_process")
verification_state = _OptionalModule("verification_state")
