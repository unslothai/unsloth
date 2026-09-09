# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted project-root authority shared by agent context services."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from storage.studio_db import ensure_chat_project_workspace, get_chat_project


class AgentWorkspaceError(RuntimeError):
    """A safe, user-readable workspace operation failure."""


@dataclass(frozen = True)
class ProjectWorkspace:
    project_id: str
    root: Path
    kind: str
    device_id: Optional[int]
    file_id: Optional[int]


def project_workspace(project_id: str) -> ProjectWorkspace:
    """Resolve a project root exclusively from its persisted storage record."""
    project = get_chat_project(project_id)
    if project is None:
        raise AgentWorkspaceError("Project not found.")
    try:
        project = ensure_chat_project_workspace(project_id)
    except OSError as exc:
        raise AgentWorkspaceError(
            "The project folder is unavailable. Reconnect it and reopen the project."
        ) from exc

    if project is None or project.get("archived"):
        raise AgentWorkspaceError("Project not found.")

    kind = str(project.get("workspaceKind") or "managed")
    if kind != "managed":
        raise AgentWorkspaceError("This project workspace type is not supported.")
    raw_root = project.get("sandboxPath")
    if not raw_root:
        raise AgentWorkspaceError("The project has no workspace folder.")
    root = Path(str(raw_root)).expanduser()
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        metadata = root.stat(follow_symlinks = False)
        resolved = root.resolve(strict = True)
        resolved_metadata = resolved.stat(follow_symlinks = False)
    except AgentWorkspaceError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project folder is unavailable.") from exc
    if not resolved.is_dir() or (
        metadata.st_dev,
        metadata.st_ino,
    ) != (resolved_metadata.st_dev, resolved_metadata.st_ino):
        raise AgentWorkspaceError("The project folder identity changed.")

    expected_device = int(metadata.st_dev)
    expected_file = int(metadata.st_ino)

    return ProjectWorkspace(
        project_id = project_id,
        root = resolved,
        kind = kind,
        device_id = expected_device,
        file_id = expected_file,
    )


@contextmanager
def project_workspace_access(project_id: str):
    """Resolve and hold a project workspace against concurrent folder changes."""
    from core.inference.tools import project_workspace_in_flight
    with project_workspace_in_flight(project_id):
        yield project_workspace(project_id)


__all__ = [
    "AgentWorkspaceError",
    "ProjectWorkspace",
    "project_workspace",
    "project_workspace_access",
]
