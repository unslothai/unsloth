# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-user folders in the system temp dir, for hosts where that dir is shared between accounts."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from pathlib import Path


def private_tmp_path(name: str) -> Path:
    """Where ``private_tmp_dir(name)`` lives; no I/O. Windows ``%TEMP%`` is already per user."""
    if hasattr(os, "getuid"):
        name = f"{name}-{os.getuid()}"
    return Path(tempfile.gettempdir()) / name


def private_tmp_dir(name: str) -> Path:
    """Create (or reuse) a 0700 per-user temp folder; refuse a link, foreign owner or lingering g/o write."""
    path = private_tmp_path(name)
    path.mkdir(mode = 0o700, parents = True, exist_ok = True)
    if not hasattr(os, "getuid"):
        return path
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError(
            errno.EPERM, "refusing a temp folder that is a link or owned by another user", str(path)
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
        # Re-read: some network and FUSE mounts ignore chmod and report success.
        if stat.S_IMODE(os.lstat(path).st_mode) & 0o022:
            raise PermissionError(
                errno.EPERM, "refusing a group- or world-writable temp folder", str(path)
            )
    return path
