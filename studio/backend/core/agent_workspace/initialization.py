# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Create-only `/init` scaffolding for a persisted project root."""

import os
import stat

from storage.studio_db import get_chat_project

from .common import AgentWorkspaceError, project_workspace_access
from .instructions import resolve_agents_instructions


MAX_INITIAL_AGENTS_BYTES = 64 * 1024
_ROOT_INSTRUCTION_FILENAMES = ("AGENTS.override.md", "AGENTS.md")


def _scaffold(project: dict) -> bytes:
    project_name = str(project.get("name") or "Project").strip()[:200]
    stored = str(project.get("instructions") or "").strip()
    lines = [
        "# AGENTS.md",
        "",
        f"## {project_name}",
        "",
        "## Working agreements",
        "",
        "- Keep changes scoped to the current task.",
        "- Preserve unrelated user changes and repository history.",
        "- Run the relevant tests, linters, and build checks before claiming completion.",
        "- Report validation that was not run or could not run.",
    ]
    if stored:
        lines.extend(("", "## Project instructions", "", stored))
    payload = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    if len(payload) > MAX_INITIAL_AGENTS_BYTES:
        raise AgentWorkspaceError("Project instructions are too large for an AGENTS.md scaffold.")
    return payload


def _open_root(workspace) -> int:
    descriptor = None
    try:
        descriptor = os.open(
            workspace.root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened = os.fstat(descriptor)
        current = workspace.root.stat(follow_symlinks = False)
        expected = (int(workspace.device_id), int(workspace.file_id))
        actual = (int(opened.st_dev), int(opened.st_ino))
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or actual != (int(current.st_dev), int(current.st_ino))
            or actual != expected
        ):
            raise AgentWorkspaceError("Project root identity changed.")
        return descriptor
    except AgentWorkspaceError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise AgentWorkspaceError("The project root is unavailable.") from exc


def _existing_root_instruction(root_fd: int) -> str | None:
    for name in _ROOT_INSTRUCTION_FILENAMES:
        try:
            os.stat(name, dir_fd = root_fd, follow_symlinks = False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise AgentWorkspaceError(
                "Root agent instructions could not be inspected safely."
            ) from exc
        return name
    return None


def _create_posix(workspace, payload: bytes) -> tuple[bool, str]:
    root_fd = _open_root(workspace)
    descriptor = None
    committed = False
    identity = None
    try:
        existing = _existing_root_instruction(root_fd)
        if existing is not None:
            return False, existing
        try:
            descriptor = os.open(
                "AGENTS.md",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd = root_fd,
            )
        except FileExistsError:
            return False, "AGENTS.md"
        identity = os.fstat(descriptor)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Could not write AGENTS.md")
            offset += written
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        # Reopen and revalidate the root while its original descriptor remains
        # held, then publish the directory entry durably.
        check_fd = _open_root(workspace)
        os.close(check_fd)
        named = os.stat("AGENTS.md", dir_fd = root_fd, follow_symlinks = False)
        if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (
            identity.st_dev,
            identity.st_ino,
        ):
            raise AgentWorkspaceError("AGENTS.md changed while it was being created.")
        selected = _existing_root_instruction(root_fd)
        if selected != "AGENTS.md":
            return False, selected or "AGENTS.md"
        os.fsync(root_fd)
        committed = True
        return True, "AGENTS.md"
    finally:
        if descriptor is not None:
            if not committed and identity is not None:
                try:
                    named = os.stat("AGENTS.md", dir_fd = root_fd, follow_symlinks = False)
                    if (named.st_dev, named.st_ino) == (identity.st_dev, identity.st_ino):
                        os.unlink("AGENTS.md", dir_fd = root_fd)
                except OSError:
                    pass
            os.close(descriptor)
        os.close(root_fd)


def initialize_project_agents(project_id: str) -> dict:
    with project_workspace_access(project_id) as workspace:
        project = get_chat_project(project_id)
        if project is None:
            raise AgentWorkspaceError("Project not found.")
        payload = _scaffold(project)
        if os.name == "nt":
            raise AgentWorkspaceError("Secure /init is not available on Windows yet.")
        created, path = _create_posix(workspace, payload)
        instructions = resolve_agents_instructions(
            workspace.root,
            expected_identity = (workspace.device_id, workspace.file_id),
        )
    return {
        "status": "created" if created else "already_exists",
        "created": created,
        "path": path,
        "instructions": instructions,
    }


__all__ = ["MAX_INITIAL_AGENTS_BYTES", "initialize_project_agents"]
