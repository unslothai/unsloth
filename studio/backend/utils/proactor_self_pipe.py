# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# CPython's proactor self-pipe read ignores EOF, so a closed peer (AdGuard) pins a core.

from __future__ import annotations

import errno
import signal
import socket
import sys
import threading
import time

from loggers import get_logger

logger = get_logger(__name__)

# Fixed, not growing: with no read armed this timer is what wakes an idle loop.
_REBUILD_BACKOFF_SECONDS = 1.0
# Windows ephemeral loopback binds in socket.socketpair() can hit a dynamic Hyper-V/WinNAT
# excluded port range or endpoint filter and fail with WinError 10013 / 10048 (#18637).
_SOCKETPAIR_PORT_RETRIES = 32
_EPHEMERAL_PORT_WINERRORS = (10013, 10048)
_EPHEMERAL_PORT_ERRNOS = (errno.EACCES, errno.EADDRINUSE)
_installed = False
_cpython_loop_self_reading = None
_cpython_make_self_pipe = None
_cpython_close_self_pipe = None
_socketpair = socket.socketpair


def _is_ephemeral_port_conflict(exc: OSError) -> bool:
    return (
        isinstance(exc, PermissionError)
        or getattr(exc, "winerror", None) in _EPHEMERAL_PORT_WINERRORS
        or getattr(exc, "errno", None) in _EPHEMERAL_PORT_ERRNOS
    )


def _socketpair_with_retry():
    last_exc: OSError | None = None
    for _ in range(_SOCKETPAIR_PORT_RETRIES):
        try:
            return _socketpair()
        except OSError as exc:
            if not _is_ephemeral_port_conflict(exc):
                raise
            last_exc = exc
    assert last_exc is not None
    raise last_exc


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
        ssock, csock = _socketpair_with_retry()
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
    # Follow the main-thread signal wakeup fd only if it still names our old socket.
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
    """Patch BaseProactorEventLoop self-pipe creation, teardown, and read loop once. Returns whether the guard is active."""
    global _installed, _cpython_loop_self_reading, _cpython_make_self_pipe, _cpython_close_self_pipe
    if _installed:
        return True
    if sys.platform != "win32":
        return False
    try:
        from asyncio import proactor_events

        base = proactor_events.BaseProactorEventLoop
        original = base._loop_self_reading
        original_make = base._make_self_pipe
        original_close = base._close_self_pipe
    except Exception as exc:
        logger.debug("proactor self-pipe guard unavailable: %s", exc)
        return False

    def _make_self_pipe(self):
        # Initialize attributes up front and mark the half-constructed loop closed on failure
        # so BaseEventLoop.__del__ never raises AttributeError in _close_self_pipe (#18637).
        self._ssock = None
        self._csock = None
        try:
            ssock, csock = _socketpair_with_retry()
            try:
                ssock.setblocking(False)
                csock.setblocking(False)
            except OSError:
                ssock.close()
                csock.close()
                raise
        except OSError:
            proactor = getattr(self, "_proactor", None)
            if proactor is not None:
                try:
                    proactor.close()
                except Exception:
                    pass
                self._proactor = None
                self._selector = None
            self._closed = True
            raise
        self._ssock, self._csock = ssock, csock
        self._internal_fds += 1

    def _close_self_pipe(self):
        if getattr(self, "_self_reading_future", None) is not None:
            self._self_reading_future.cancel()
            self._self_reading_future = None
        ssock = getattr(self, "_ssock", None)
        if ssock is not None:
            ssock.close()
            self._ssock = None
        csock = getattr(self, "_csock", None)
        if csock is not None:
            csock.close()
            self._csock = None
        if ssock is not None or csock is not None:
            self._internal_fds -= 1

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
    _cpython_make_self_pipe = _cpython_make_self_pipe or original_make
    _cpython_close_self_pipe = _cpython_close_self_pipe or original_close
    base._make_self_pipe = _make_self_pipe
    base._close_self_pipe = _close_self_pipe
    base._loop_self_reading = _loop_self_reading
    _installed = True
    return True
