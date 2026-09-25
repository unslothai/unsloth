# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which file manager ``reveal_in_file_manager`` would open on this host, if any.

Reported by ``/api/health`` so the UI offers Reveal only where it can show something to the person
at the machine, and names it the way the platform does. A loopback URL is not enough on its own: an
SSH port-forward, a container or a headless server all look like localhost to the browser.
"""

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
    """``finder`` on macOS, ``explorer`` on Windows and WSL (which reveals in the Windows host's
    Explorer), ``files`` on a Linux desktop session, else None: a container, or Linux with no
    display to open a window on."""
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
