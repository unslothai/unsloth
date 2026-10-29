# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform subprocess helpers for the Unsloth Studio backend."""

import contextlib
import subprocess
import sys
import threading
from typing import Iterator

_SEM_FAILCRITICALERRORS = 0x0001
_error_mode_lock = threading.Lock()
_error_mode_holders = 0
_error_mode_before = 0


def windows_hidden_subprocess_kwargs() -> dict[str, object]:
    """Empty off Windows, so callers can always unpack it without a platform check."""
    if sys.platform != "win32":
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_factory is not None and startf_use_showwindow:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


@contextlib.contextmanager
def windows_loader_error_boxes_suppressed() -> Iterator[None]:
    """Hold around a test-launch: the child inherits SEM_FAILCRITICALERRORS, so a missing DLL
    exits 0xC0000135 instead of blocking on a modal box until someone dismisses it. The mode is
    process-wide, so concurrent holders share one change and the last one out restores it."""
    global _error_mode_holders, _error_mode_before
    if sys.platform != "win32":
        yield
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
    except Exception:  # noqa: BLE001 -- best effort; the launch still runs
        yield
        return
    with _error_mode_lock:
        if _error_mode_holders == 0:
            mode = kernel32.GetErrorMode() | _SEM_FAILCRITICALERRORS
            _error_mode_before = kernel32.SetErrorMode(mode)
        _error_mode_holders += 1
    try:
        yield
    finally:
        with _error_mode_lock:
            _error_mode_holders -= 1
            if _error_mode_holders == 0:
                kernel32.SetErrorMode(_error_mode_before)
