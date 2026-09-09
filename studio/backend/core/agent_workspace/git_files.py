# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded descriptor-relative reads for untracked Git review evidence."""

import os
import stat

from .git_context import AgentWorkspaceError


def read_project_file(workspace, target: str, limit: int):
    if os.name == "nt":
        # The secure-tools split supplies native handle-relative traversal.
        try:
            from .mutation import ProjectFileMutation
        except ModuleNotFoundError as exc:
            if exc.name != __package__ + ".mutation":
                raise
            raise AgentWorkspaceError("Secure untracked-file reads are unavailable.") from exc
        with ProjectFileMutation.open(workspace, target, max_bytes = limit) as reader:
            return reader.read(limit)
    parts = target.split("/")
    if not parts or any(p in {"", ".", ".."} or p.lower() == ".git" for p in parts):
        raise AgentWorkspaceError("Git review path is invalid.")
    if "\\" in target or "\x00" in target:
        raise AgentWorkspaceError("Git review path is invalid.")
    handles = []
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        root = os.open(workspace.root, flags | os.O_DIRECTORY)
        handles.append(root)
        metadata = os.fstat(root)
        if (metadata.st_dev, metadata.st_ino) != (workspace.device_id, workspace.file_id):
            raise AgentWorkspaceError("Project workspace identity changed.")
        parent = root
        for part in parts[:-1]:
            parent = os.open(part, flags | os.O_DIRECTORY, dir_fd = parent)
            handles.append(parent)
        descriptor = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd = parent)
        handles.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AgentWorkspaceError("Only ordinary unlinked files can be reviewed.")
        if before.st_size > limit:
            raise OverflowError("untracked-file-limit")
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(descriptor, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > limit:
            raise OverflowError("untracked-file-limit")
        after = os.fstat(descriptor)
        named = os.stat(parts[-1], dir_fd = parent, follow_symlinks = False)

        def identity(value):
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                value.st_mode,
                value.st_nlink,
            )

        if identity(before) != identity(after) or identity(after) != identity(named):
            raise AgentWorkspaceError("Untracked file changed during review.")
        return bytes(data), stat.S_IMODE(after.st_mode), (after.st_dev, after.st_ino)
    finally:
        for descriptor in reversed(handles):
            os.close(descriptor)
