# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local file manager for reveal_in_file_manager, from /api/health; loopback alone is not enough."""

from __future__ import annotations

import os
import sys
from typing import Literal, Optional

FileManager = Literal["finder", "explorer", "files"]


def _in_container() -> bool:
    return (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or bool(os.environ.get("container"))
        or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )


def file_manager_kind() -> Optional[FileManager]:
    """finder on macOS, explorer on Windows and WSL (revealing in the Windows host's Explorer),
    files on a Linux desktop session, else None (a container, or Linux with no display)."""
    if sys.platform == "darwin":
        return "finder"
    if os.name == "nt":
        return "explorer"
    from utils.paths.path_utils import _IS_WSL

    if _IS_WSL:
        return "explorer"
    if _in_container():
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "files"
    return None
