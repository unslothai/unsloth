# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copy a gallery image or clip into a chat project's folder.

Files land in ``<project root>/sandbox/{images,videos,audio}``, where the project's chats run their
tools. It is a copy, so the gallery keeps its item.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import uuid
from pathlib import Path

from loggers import get_logger

logger = get_logger(__name__)


class ProjectNotFound(LookupError):
    """No live project has this id."""


# POSIX: walk into the folder by descriptor so a symlink swapped in cannot redirect the write.
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_USE_DIR_FD = (
    hasattr(os, "O_NOFOLLOW")
    and {
        os.open,
        os.mkdir,
        os.stat,
        os.rename,
        os.unlink,
    }
    <= os.supports_dir_fd
)


def _sandbox_dir(project_id: str) -> str:
    from storage.studio_db import ensure_chat_project_workspace

    project = ensure_chat_project_workspace(project_id)
    if not project:
        raise ProjectNotFound(project_id)
    root_path = project.get("rootPath")
    sandbox_path = project.get("sandboxPath")
    if not root_path or not sandbox_path:
        raise ProjectNotFound(project_id)
    # Same containment check as the chat tools.
    root_real = os.path.realpath(root_path)
    sandbox_real = os.path.realpath(sandbox_path)
    if sandbox_real != root_real and not sandbox_real.startswith(root_real + os.sep):
        raise ProjectNotFound(project_id)
    return sandbox_real


def _tmp_name(name: str) -> str:
    # Unique per call: two adds of one item can run at once in this process.
    return f".{name}.tmp-{uuid.uuid4().hex}"


def _copy_with_dir_fd(source: Path, sandbox: str, folder: str, name: str) -> bool:
    """Copy into ``sandbox/folder`` by descriptor; returns whether it was already there."""
    sandbox_fd = os.open(sandbox, _DIR_FLAGS)
    try:
        try:
            os.mkdir(folder, 0o755, dir_fd = sandbox_fd)
        except FileExistsError:
            pass
        # O_NOFOLLOW makes a symlinked folder fail here (ELOOP) instead of writing through it.
        folder_fd = os.open(folder, _DIR_FLAGS, dir_fd = sandbox_fd)
    finally:
        os.close(sandbox_fd)
    try:
        # Keyed by gallery id and written atomically, so a file here is this item, maybe since edited.
        try:
            st = os.stat(name, dir_fd = folder_fd, follow_symlinks = False)
            if stat.S_ISREG(st.st_mode):
                return True
        except FileNotFoundError:
            pass
        tmp = _tmp_name(name)
        out = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd = folder_fd,
        )
        try:
            with os.fdopen(out, "wb") as dst, open(source, "rb") as src:
                shutil.copyfileobj(src, dst)
            os.rename(tmp, name, src_dir_fd = folder_fd, dst_dir_fd = folder_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp, dir_fd = folder_fd)
            raise
    finally:
        os.close(folder_fd)
    return False


def _copy_by_path(source: Path, sandbox: str, folder: str, name: str) -> bool:
    """Fallback without dir_fd support (Windows): resolve the folder and re-check containment."""
    target = Path(sandbox) / folder
    target.mkdir(exist_ok = True)
    real = os.path.realpath(target)
    if os.path.dirname(real) != sandbox:
        raise PermissionError(f"{target} resolves outside the project sandbox")
    dest = Path(real) / name
    try:
        if not dest.is_symlink() and dest.is_file():
            return True
    except OSError:
        pass
    tmp = Path(real) / _tmp_name(name)
    try:
        shutil.copyfile(source, tmp)
        # Without dir_fd the folder can be swapped after the check; check again before the rename.
        if os.path.realpath(tmp.parent) != real:
            raise PermissionError(f"{target} moved outside the project sandbox")
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok = True)
        raise
    return False


def copy_into_project(
    source: Path,
    project_id: str,
    folder: str,
    name: str | None = None,
) -> dict[str, object]:
    """Copy ``source`` into the project's ``folder`` and return ``{"path", "already"}``.

    Keyed by the file name (the gallery id unless ``name`` is given), so a second add is a no-op.
    Written via a temp file so a failed copy leaves nothing behind. Refuses a ``folder`` that leads
    outside the sandbox. Raises ProjectNotFound, ValueError for a bad ``name``, or OSError."""
    name = name or source.name
    if name in (".", "..") or name.startswith(".") or any(sep in name for sep in "/\\\0"):
        raise ValueError(f"Bad file name: {name!r}")
    sandbox = _sandbox_dir(project_id)
    copy = _copy_with_dir_fd if _USE_DIR_FD else _copy_by_path
    already = copy(source, sandbox, folder, name)
    return {"path": str(Path(sandbox) / folder / name), "already": already}
