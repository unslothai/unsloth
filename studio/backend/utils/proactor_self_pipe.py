# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Keeps a Windows ProactorEventLoop from pinning a core when its self-pipe peer goes away.
#
# The proactor wakes itself through a loopback socketpair. CPython's _loop_self_reading re-posts the read as soon as
# it completes and never checks for EOF, so once the peer end is closed (traffic filters such as AdGuard's Browsing
# Security do this to loopback connections when the PC wakes from sleep) every read completes instantly with b"" and
# the loop spins on one core for the rest of the process. Unfixed through CPython 3.14. On EOF we rebuild the pair.

from __future__ import annotations

import sys
import time

from loggers import get_logger

logger = get_logger(__name__)

# A filter that closes every new pair immediately must not turn the fix into a socket churn loop: after a quick
# repeat, wait this long before re-arming. Cross-thread wakeups wait at most this long while backed off.
_REBUILD_BACKOFF_SECONDS = 1.0
_installed = False


def _self_pipe_hit_eof(f) -> bool:
    if f is None or not f.done() or f.cancelled() or f.exception() is not None:
        return False
    return f.result() == b""


def install_proactor_self_pipe_guard() -> bool:
    """Patch BaseProactorEventLoop._loop_self_reading once. Returns whether the guard is active."""
    global _installed
    if _installed:
        return True
    if sys.platform != "win32":
        return False
    try:
        from asyncio import proactor_events

        base = proactor_events.BaseProactorEventLoop
        original = base._loop_self_reading
        # Private API: stand down rather than guess if a future CPython reshapes it.
        if not all(hasattr(base, name) for name in ("_make_self_pipe", "_close_self_pipe")):
            return False
    except Exception as exc:
        logger.debug("proactor self-pipe guard unavailable: %s", exc)
        return False

    def _loop_self_reading(self, f = None):
        if f is None or f is not self._self_reading_future or not _self_pipe_hit_eof(f):
            return original(self, f)
        if self.is_closed():
            return None
        now = time.monotonic()
        last = getattr(self, "_unsloth_self_pipe_rebuilt_at", None)
        self._unsloth_self_pipe_rebuilt_at = now
        quick_repeat = last is not None and now - last < 5 * _REBUILD_BACKOFF_SECONDS
        if not quick_repeat:
            logger.warning(
                "Event loop self-pipe was closed from outside (often a network filter after sleep); rebuilding it"
            )
        try:
            self._close_self_pipe()
            self._make_self_pipe()
        except Exception as exc:
            logger.warning("Could not rebuild the event loop self-pipe: %s", exc)
            return None
        if quick_repeat:
            self.call_later(_REBUILD_BACKOFF_SECONDS, original, self, None)
            return None
        return original(self, None)

    base._loop_self_reading = _loop_self_reading
    _installed = True
    return True
