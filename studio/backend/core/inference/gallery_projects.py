# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copy a gallery image or clip into a chat project's folder.

Files land in ``<project root>/sandbox/{images,videos}``, where the project's chats run their
tools. It is a copy, so the gallery keeps its item.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


class ProjectNotFound(LookupError):
    """No live project has this id."""


def _project_media_dir(project_id: str, folder: str) -> Path:
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
    target = Path(sandbox_real) / folder
    target.mkdir(parents = True, exist_ok = True)
    return target


def copy_into_project(source: Path, project_id: str, folder: str) -> dict[str, object]:
    """Copy ``source`` into the project's ``folder`` and return ``{"path", "already"}``.

    Keyed by the gallery id, so a second add is a no-op. Written via a temp file so a failed copy
    leaves nothing behind. Raises ProjectNotFound or OSError."""
    target_dir = _project_media_dir(project_id, folder)
    dest = target_dir / source.name
    try:
        if dest.is_file() and dest.stat().st_size == source.stat().st_size:
            return {"path": str(dest), "already": True}
    except OSError:
        pass
    tmp: Optional[Path] = target_dir / f".{source.name}.tmp-{os.getpid()}"
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok = True)
            except OSError:
                pass
    return {"path": str(dest), "already": False}
