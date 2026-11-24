# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform subprocess helpers for the Unsloth Studio backend."""

import contextlib
import subprocess
import sys
import threading
import types
from typing import Iterator

_SEM_FAILCRITICALERRORS = 0x0001


def _process_error_mode_state() -> types.ModuleType:
    """One per process: the installer imports this file as backend.utils.subprocess_compat, the backend as utils.subprocess_compat."""
    name = "_unsloth_windows_error_mode"
    state = sys.modules.get(name)
    if state is None:
        fresh = types.ModuleType(name)
        fresh.lock = threading.Lock()
        fresh.held = {}  # token -> flags
        fresh.before = 0
        state = sys.modules.setdefault(name, fresh)
    return state


_ERROR_MODE = _process_error_mode_state()


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


def _apply_held_error_mode(kernel32) -> None:
    flags = 0
    for held in _ERROR_MODE.held.values():
        flags |= held
    kernel32.SetErrorMode(_ERROR_MODE.before | flags)


@contextlib.contextmanager
def windows_loader_error_boxes_suppressed(flags: int = _SEM_FAILCRITICALERRORS) -> Iterator[None]:
    """The child inherits SEM_FAILCRITICALERRORS, so a missing DLL exits 0xC0000135 instead of a modal box.
    Process-wide: mode = the one before the first holder | every holder's flags; the last out restores it."""
    if sys.platform != "win32":
        yield
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
    except Exception:  # noqa: BLE001 -- best effort; the launch still runs
        yield
        return
    token = object()
    with _ERROR_MODE.lock:
        if not _ERROR_MODE.held:
            _ERROR_MODE.before = kernel32.GetErrorMode()
        _ERROR_MODE.held[token] = flags
        _apply_held_error_mode(kernel32)
    try:
        yield
    finally:
        with _ERROR_MODE.lock:
            del _ERROR_MODE.held[token]
            _apply_held_error_mode(kernel32)
