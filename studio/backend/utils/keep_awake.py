# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Windows awake (not the display) while a generation or training job runs; no-op elsewhere.

The execution state is per thread and jobs start/end on different threads, so one holder thread owns it."""

from __future__ import annotations

import functools
import logging
import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_F = TypeVar("_F", bound = Callable[..., Any])

_lock = threading.Lock()
_active = 0
_holder: Optional["_Holder"] = None


def _kernel32() -> Any:
    """``kernel32`` on Windows, else None. Patched in tests."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        return ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


class _Holder:
    def __init__(self, kernel32: Any) -> None:
        self._kernel32 = kernel32
        self._release = threading.Event()
        self._applied = threading.Event()
        self._thread = threading.Thread(target = self._run, name = "studio-keep-awake", daemon = True)

    def start(self) -> None:
        self._thread.start()
        self._applied.wait(timeout = 5.0)

    def _call(self, flags: int) -> None:
        try:
            if not self._kernel32.SetThreadExecutionState(flags):
                logger.debug("SetThreadExecutionState(0x%08X) failed", flags)
        except Exception as exc:  # noqa: BLE001 - never fail a job over a power hint
            logger.debug("SetThreadExecutionState(0x%08X) raised: %s", flags, exc)

    def _run(self) -> None:
        try:
            self._call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        finally:
            self._applied.set()
        self._release.wait()
        self._call(ES_CONTINUOUS)

    def stop(self) -> None:
        self._release.set()
        self._thread.join(timeout = 5.0)


def _acquire() -> None:
    global _active, _holder
    with _lock:
        _active += 1
        if _active != 1:
            return
        kernel32 = _kernel32()
        if kernel32 is None:
            return
        holder = _Holder(kernel32)
        try:
            holder.start()
        except Exception as exc:  # noqa: BLE001
            logger.debug("keep-awake holder did not start: %s", exc)
            return
        _holder = holder


def _release() -> None:
    global _active, _holder
    with _lock:
        if _active <= 0:
            return
        _active -= 1
        if _active or _holder is None:
            return
        holder, _holder = _holder, None
        holder.stop()


@contextmanager
def keep_system_awake() -> Iterator[None]:
    """Hold the system awake for the block; nested and overlapping holders are refcounted."""
    _acquire()
    try:
        yield
    finally:
        _release()


def keeps_system_awake(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with keep_system_awake():
            return fn(*args, **kwargs)

    wrapper.keeps_system_awake = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def active_holders() -> int:
    return _active
