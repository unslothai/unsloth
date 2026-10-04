# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Keeps a Windows ProactorEventLoop from pinning a core when its self-pipe peer goes away.
#
# The proactor wakes itself through a loopback socketpair. CPython's _loop_self_reading re-posts the read as soon as
# it completes and never checks for EOF, so once the peer end is closed (traffic filters such as AdGuard's Browsing
# Security do this to loopback connections when the PC wakes from sleep) every read completes instantly with b"" and
# the loop spins on one core for the rest of the process. Unfixed through CPython 3.14. On EOF we swap in a new pair.

from __future__ import annotations

import signal
import socket
import sys
import threading
import time

from loggers import get_logger

logger = get_logger(__name__)

# A filter that closes every new pair immediately must not turn the fix into a socket churn loop: after a quick
# repeat, wait this long before re-arming. A failed socketpair() is retried on the same period, warning once per
# streak. With no read armed, this timer is what wakes an idle loop, so cross-thread work waits at most this long.
_REBUILD_BACKOFF_SECONDS = 1.0
_installed = False
_cpython_loop_self_reading = None
# Indirection so tests can stand in a filter that closes each new pair.
_socketpair = socket.socketpair


def _self_pipe_hit_eof(f) -> bool:
    if f is None or not f.done() or f.cancelled() or f.exception() is not None:
        return False
    return f.result() == b""


def _swap_failed(loop, exc) -> bool:
    failures = getattr(loop, "_unsloth_self_pipe_failures", 0)
    loop._unsloth_self_pipe_failures = failures + 1
    log = logger.warning if failures == 0 else logger.debug
    log("Could not rebuild the event loop self-pipe (attempt %d): %s", failures + 1, exc)
    return False


def _swap_self_pipe(loop) -> bool:
    """Replace the loop's self-pipe with a fresh pair. On failure the old pair stays, so close() still works."""
    try:
        ssock, csock = _socketpair()
    except OSError as exc:
        return _swap_failed(loop, exc)
    try:
        ssock.setblocking(False)
        csock.setblocking(False)
    except OSError as exc:
        ssock.close()
        csock.close()
        return _swap_failed(loop, exc)
    loop._unsloth_self_pipe_failures = 0
    old_ssock, old_csock = loop._ssock, loop._csock
    loop._self_reading_future = None
    loop._ssock, loop._csock = ssock, csock
    # BaseProactorEventLoop points the signal wakeup fd at _csock on the main thread; follow it to the new socket,
    # but only when it still names ours, so another owner's registration is left alone.
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.set_wakeup_fd(csock.fileno())
            if previous != old_csock.fileno():
                signal.set_wakeup_fd(previous)
        except (ValueError, OSError) as exc:
            logger.debug("Could not move the signal wakeup fd to the new self-pipe: %s", exc)
    old_ssock.close()
    old_csock.close()
    return True


def install_proactor_self_pipe_guard() -> bool:
    """Patch BaseProactorEventLoop._loop_self_reading once. Returns whether the guard is active."""
    global _installed, _cpython_loop_self_reading
    if _installed:
        return True
    if sys.platform != "win32":
        return False
    try:
        from asyncio import proactor_events
        base = proactor_events.BaseProactorEventLoop
        original = base._loop_self_reading
    except Exception as exc:
        logger.debug("proactor self-pipe guard unavailable: %s", exc)
        return False

    def _loop_self_reading(self, f = None):
        if f is None or f is not self._self_reading_future or not _self_pipe_hit_eof(f):
            return original(self, f)
        if self.is_closed() or self._ssock is None:
            return None
        now = time.monotonic()
        last = getattr(self, "_unsloth_self_pipe_rebuilt_at", None)
        self._unsloth_self_pipe_rebuilt_at = now
        quick_repeat = last is not None and now - last < 5 * _REBUILD_BACKOFF_SECONDS
        if not quick_repeat and not getattr(self, "_unsloth_self_pipe_failures", 0):
            logger.warning(
                "Event loop self-pipe was closed from outside (often a network filter after sleep); rebuilding it"
            )
        if not _swap_self_pipe(self):
            # f stays current, so the retry lands back here and tries the swap again.
            self.call_later(_REBUILD_BACKOFF_SECONDS, self._loop_self_reading, f)
            return None
        if quick_repeat:
            self.call_later(_REBUILD_BACKOFF_SECONDS, original, self, None)
            return None
        return original(self, None)

    _cpython_loop_self_reading = _cpython_loop_self_reading or original
    base._loop_self_reading = _loop_self_reading
    _installed = True
    return True
