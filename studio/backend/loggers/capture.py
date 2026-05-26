# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Thread-scoped structlog capture: route an eval run's log lines to a sink."""

from __future__ import annotations

import threading
from typing import Callable

# thread ident -> sink(level: str, message: str)
_SINKS: dict[int, Callable[[str, str], None]] = {}
_LOCK = threading.Lock()

# event_dict keys that are structural, not message content
_STD_KEYS = {"timestamp", "level", "event", "logger", "logger_name", "exc_info", "stack"}


def register_sink(ident: int, sink: Callable[[str, str], None]) -> None:
    with _LOCK:
        _SINKS[ident] = sink


def unregister_sink(ident: int) -> None:
    with _LOCK:
        _SINKS.pop(ident, None)


def capture_processor(logger, method_name, event_dict):
    """structlog pass-through: if the current thread has a registered sink
    (an active eval run), forward a readable line. Never raises; returns the
    event_dict unchanged so normal rendering continues."""
    try:
        sink = _SINKS.get(threading.get_ident())  # lock-free read on hot path
        if sink is not None:
            level = str(event_dict.get("level", "info"))
            event = event_dict.get("event", "")
            msg = str(event) if event is not None else ""
            extras = " ".join(
                f"{k}={v}"
                for k, v in event_dict.items()
                if k not in _STD_KEYS
            )
            if extras:
                msg = f"{msg} {extras}".strip()
            if msg:
                sink(level, msg)
    except Exception:
        pass
    return event_dict
