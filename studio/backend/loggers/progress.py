# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Heartbeat throttle for repeated progress logs.

A long download / model load / training run streams the same progress line over
and over. ProgressThrottle keeps progress visible but not redundant: it logs the
first message for a key, any time the message changes (a new phase), then at most
once per interval while it stays the same. Start/completion/error lines live at
their own call sites and are never gated here. Verbose restores every line.
"""

import os
import threading
import time


def _interval_default() -> float:
    try:
        return max(0.0, float(os.environ.get("UNSLOTH_STUDIO_PROGRESS_LOG_INTERVAL_S", "10")))
    except ValueError:
        return 10.0


def _verbose() -> bool:
    from loggers.config import logs_verbose
    return logs_verbose()


class ProgressThrottle:
    """Emit a progress line at most once per ``interval_s`` per key, plus whenever
    the message changes. ``interval_s=0`` (or verbose) logs everything. Pass a
    stable/empty ``message`` for a pure time heartbeat (e.g. a step counter that
    changes every tick), or the real status text to also log on phase changes."""

    def __init__(self, interval_s: "float | None" = None) -> None:
        self._interval = _interval_default() if interval_s is None else interval_s
        self._last_emit: dict = {}
        self._last_msg: dict = {}
        self._guard = threading.Lock()

    def should_log(
        self,
        key,
        message: str = "",
    ) -> bool:
        if self._interval <= 0 or _verbose():
            return True
        now = time.monotonic()
        with self._guard:
            changed = self._last_msg.get(key) != message
            last = self._last_emit.get(key)
            if changed or last is None or (now - last) >= self._interval:
                self._last_emit[key] = now
                self._last_msg[key] = message
                return True
            return False

    def reset(self, key) -> None:
        """Forget a key so its next message logs immediately (call on completion)."""
        with self._guard:
            self._last_emit.pop(key, None)
            self._last_msg.pop(key, None)


# Shared instance for the subprocess status / training heartbeats.
progress_throttle = ProgressThrottle()
