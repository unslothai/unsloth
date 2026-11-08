# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extend a test file to a large size without paying for it on disk.

``truncate`` past the end leaves a hole on Linux and macOS. NTFS allocates the whole extension unless the
file is marked sparse first, so a suite of multi-GB fake checkpoints fills a Windows runner's disk.
"""

from __future__ import annotations

import os


def _mark_sparse(handle) -> bool:
    import ctypes
    import msvcrt

    fsctl_set_sparse = 0x000900C4
    returned = ctypes.c_ulong(0)
    return bool(
        ctypes.windll.kernel32.DeviceIoControl(
            ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno())),
            fsctl_set_sparse,
            None,
            0,
            None,
            0,
            ctypes.byref(returned),
            None,
        )
    )


def extend_sparse(handle, size: int) -> None:
    """``handle.truncate(size)``, marking the file sparse first on Windows."""
    if os.name == "nt":
        handle.flush()
        if not _mark_sparse(handle):
            raise OSError(
                "could not mark the test file sparse; extending it would commit the full size"
            )
    handle.truncate(size)
