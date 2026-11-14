# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

from loggers import get_logger

from .base import Backend, BackendError, Display, FrontWindow

logger = get_logger(__name__)

__all__ = ["Backend", "BackendError", "Display", "FrontWindow", "get_backend", "reset_backend"]

BACKEND_ENV = "UNSLOTH_STUDIO_COMPUTER_USE_BACKEND"

_lock = threading.Lock()
_backend: Optional[Backend] = None


class _Unsupported(Backend):
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def unavailable_reason(self) -> Optional[str]:
        return self._reason


def _create() -> Backend:
    if os.environ.get(BACKEND_ENV, "").strip().lower() == "virtual":
        from .virtual import VirtualBackend
        return VirtualBackend()
    try:
        if sys.platform == "darwin":
            from .macos import MacBackend
            return MacBackend()
        if sys.platform == "win32":
            from .windows import WindowsBackend
            return WindowsBackend()
        if sys.platform.startswith("linux"):
            from .linux import LinuxBackend
            return LinuxBackend()
    except Exception:  # noqa: BLE001 - a missing system library must not take the server down
        logger.warning("computer use backend failed to load", exc_info = True)
        return _Unsupported("Computer use could not start on this system.")
    return _Unsupported("Computer use isn't supported on this operating system.")


def get_backend() -> Backend:
    global _backend
    with _lock:
        if _backend is None:
            _backend = _create()
        return _backend


def reset_backend() -> None:
    """Test hook: the next get_backend() reads the environment again."""
    global _backend
    with _lock:
        _backend = None
