# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Exits a desktop-owned backend whose app died without cleanup, so the port is freed.

from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_DEFAULT_POLL_SECONDS = 2.0


def _fire(on_parent_exit: Callable[[], None]) -> None:
    logger.info("parent_watchdog.parent_exited: shutting down")
    try:
        on_parent_exit()
    except Exception as exc:
        logger.warning("parent_watchdog: shutdown callback failed: %s", exc)


def _watch_unix(parent_pid, on_parent_exit, stop, poll_seconds) -> None:
    # Reparenting (parent != owner pid) is the death signal: immune to pid reuse and reaping.
    while True:
        if os.getppid() != parent_pid:
            _fire(on_parent_exit)
            return
        if stop.wait(poll_seconds):
            return


def _watch_windows(parent_pid, on_parent_exit, stop, poll_seconds) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    # Explicit HANDLE signatures: c_int defaults truncate handles on 64-bit.
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
    if not handle:
        _fire(on_parent_exit)
        return
    try:
        while not stop.is_set():
            if kernel32.WaitForSingleObject(handle, int(poll_seconds * 1000)) == WAIT_OBJECT_0:
                _fire(on_parent_exit)
                return
    finally:
        kernel32.CloseHandle(handle)


# Returns the stop event, or None if the parent was already gone (callback fires at once).
def start_parent_watchdog(
    on_parent_exit: Callable[[], None],
    parent_pid: Optional[int] = None,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
) -> Optional[threading.Event]:
    if parent_pid is None:
        parent_pid = os.getppid()
    if parent_pid <= 1:
        _fire(on_parent_exit)
        return None
    stop = threading.Event()
    watch = _watch_windows if sys.platform == "win32" else _watch_unix
    thread = threading.Thread(
        target = watch,
        args = (parent_pid, on_parent_exit, stop, poll_seconds),
        name = "unsloth-parent-watchdog",
        daemon = True,
    )
    thread.start()
    logger.info("parent_watchdog.started ppid=%s", parent_pid)
    return stop
