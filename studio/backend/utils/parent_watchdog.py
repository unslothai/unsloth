# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Exits a desktop-owned backend when the app that spawned it dies without
# running its cleanup. The orphan would otherwise keep the port and make the
# next launch's preflight refuse to start.

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
    # The app spawns the backend as its direct child on Unix, so "my parent is
    # no longer the owner pid" is the death signal: kernel truth, immune to pid
    # reuse, and indifferent to whether the corpse was reaped (a kill-0 probe
    # would count an unreaped zombie as alive). Checked before the first wait
    # so an owner that died during backend startup is caught immediately.
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
    # Explicit signatures: handles are pointer-sized and would be truncated by
    # the c_int defaults on 64-bit, leaving the wait on an invalid handle.
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
        # A same-user owner is always openable, so failure means the parent
        # already died and its pid went stale: exit, don't abandon the watch.
        _fire(on_parent_exit)
        return
    try:
        # Bounded waits so a stop request is honored.
        while not stop.is_set():
            if kernel32.WaitForSingleObject(handle, int(poll_seconds * 1000)) == WAIT_OBJECT_0:
                _fire(on_parent_exit)
                return
    finally:
        kernel32.CloseHandle(handle)


# Watches parent_pid (the spawning app's own pid when it passes one, else the
# direct parent). Returns the stop event, or None when the parent was already
# gone at arm time, in which case the callback fires immediately.
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
