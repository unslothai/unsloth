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
    # Reparenting (to init or a subreaper) is the death signal; pids are never
    # probed, so pid reuse cannot fool this.
    while not stop.wait(poll_seconds):
        if os.getppid() != parent_pid:
            _fire(on_parent_exit)
            return


def _watch_windows(parent_pid, on_parent_exit, stop, poll_seconds) -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
    if not handle:
        logger.debug("parent_watchdog: OpenProcess(%s) failed", parent_pid)
        return
    try:
        # Bounded waits so a stop request is honored.
        while not stop.is_set():
            if kernel32.WaitForSingleObject(handle, int(poll_seconds * 1000)) == WAIT_OBJECT_0:
                _fire(on_parent_exit)
                return
    finally:
        kernel32.CloseHandle(handle)


# Returns the stop event, or None when the parent was already gone (the
# callback fires immediately: reparented to init means the app died before
# the watch could arm, e.g. during a slow backend startup).
def start_parent_watchdog(
    on_parent_exit: Callable[[], None],
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
) -> Optional[threading.Event]:
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
