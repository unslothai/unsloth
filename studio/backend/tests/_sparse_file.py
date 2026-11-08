# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extend a test file to a large size without paying for it on disk.

``truncate`` past the end leaves a hole on Linux and macOS. On Windows it does not: NTFS allocates the
extension unless the file is sparse, and Python's ``truncate`` extends through the CRT's ``_chsize_s``,
which writes the new bytes as zeros, so even a sparse file gets them allocated. A suite of multi-GB fake
checkpoints then fills a Windows runner's disk.
"""

from __future__ import annotations

import os


def _extend_sparse_windows(handle, size: int) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    fsctl_set_sparse = 0x000900C4
    file_begin = 0

    handle.flush()
    position = handle.tell()
    native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
    returned = wintypes.DWORD(0)
    # Sparse first, then move the end of file: SetEndOfFile on a sparse file allocates nothing.
    if not kernel32.DeviceIoControl(
        native, fsctl_set_sparse, None, 0, None, 0, ctypes.byref(returned), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.SetFilePointerEx(native, size, None, file_begin) or not kernel32.SetEndOfFile(
        native
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    handle.seek(position)


def extend_sparse(handle, size: int) -> None:
    """``handle.truncate(size)`` that stays sparse on Windows too."""
    if os.name == "nt":
        _extend_sparse_windows(handle, size)
    else:
        handle.truncate(size)
