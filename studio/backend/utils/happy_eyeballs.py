# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Race address families when connecting, instead of walking them in order (RFC 8305).

``socket.create_connection`` tries ``getaddrinfo()`` results one at a time and applies
the caller's ``timeout`` to each, with no overall deadline. On a network that black-holes
one family (a VPN carrying no IPv6, say) every address in that family burns the full
timeout before the other family is tried.

Every sync HTTP stack in this process bottoms out there -- ``httpx`` via ``httpcore``,
``requests`` via ``urllib3``, ``urllib.request`` via ``http.client`` -- so the fix goes
in at the socket rather than per client. Async httpx already races through anyio.

CPython tracks this as python/cpython#88810 (open since 2021). Waiting is not a plan: it
would land in a future version, stdlib features are not backported, and Studio supports
3.9 through 3.14.

Attempts are interleaved across families and staggered 250ms apart under one deadline,
so a black-holed family costs one stagger rather than one timeout per address. Installed
process-wide like :mod:`utils.native_tls`; injection does not survive a spawn, so every
network-touching entry point activates it before its first connection.
"""

from __future__ import annotations

import errno
import logging
import os
import selectors
import socket
import sys
import time

_ENV = "UNSLOTH_STUDIO_HAPPY_EYEBALLS"
_DELAY_ENV = "UNSLOTH_STUDIO_HAPPY_EYEBALLS_DELAY"
_FALSEY = ("0", "false", "no", "off")

# Connection Attempt Delay, RFC 8305 section 5.
_DEFAULT_DELAY = 0.25

# connect_ex() results for an attempt still in flight: EINPROGRESS on POSIX,
# EWOULDBLOCK on Windows, EALREADY on a retry.
_IN_FLIGHT = frozenset({errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY})
# ExceptionGroup and create_connection's all_errors are both 3.11. Studio supports 3.9.
_HAS_EXCEPTION_GROUP = sys.version_info >= (3, 11)

_logger = logging.getLogger(__name__)
_original_create_connection = socket.create_connection
_activated = False


def happy_eyeballs_enabled() -> bool:
    """On unless ``UNSLOTH_STUDIO_HAPPY_EYEBALLS`` opts out; the bug is CPython's, so no
    platform default."""
    return os.environ.get(_ENV, "").strip().lower() not in _FALSEY


def attempt_delay() -> float:
    """Stagger between attempts, overridable for tests and odd links."""
    try:
        value = float(os.environ.get(_DELAY_ENV, "").strip() or _DEFAULT_DELAY)
    except ValueError:
        return _DEFAULT_DELAY
    return max(0.0, value)


def _interleave(infos: list) -> list:
    """Round-robin across families, resolver order kept within each; ``getaddrinfo``
    groups by family."""
    by_family: dict = {}
    for info in infos:
        by_family.setdefault(info[0], []).append(info)
    queues = list(by_family.values())
    ordered = []
    for i in range(max((len(q) for q in queues), default = 0)):
        for q in queues:
            if i < len(q):
                ordered.append(q[i])
    return ordered


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return deadline - time.monotonic()


def happy_eyeballs_connection(
    address,
    timeout = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address = None,
    *,
    all_errors = False,
):
    """Drop-in ``socket.create_connection`` that races families under one deadline.

    ``timeout`` bounds the whole connect, not each resolved address.
    """
    host, port = address
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
        resolved_timeout = socket.getdefaulttimeout()
    else:
        resolved_timeout = timeout
    deadline = None if resolved_timeout is None else time.monotonic() + resolved_timeout

    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    if len(infos) <= 1:
        # Nothing to race; delegate so CPython's semantics and error types are kept.
        if _HAS_EXCEPTION_GROUP:
            return _original_create_connection(
                address, timeout, source_address, all_errors = all_errors
            )
        return _original_create_connection(address, timeout, source_address)

    ordered = _interleave(infos)
    delay = attempt_delay()
    exceptions: list = []
    pending: dict = {}  # socket -> sockaddr
    winner = None

    def _settle(sock):
        """Hand back a socket with the blocking mode the caller expects."""
        sock.setblocking(True)
        sock.settimeout(resolved_timeout)
        return sock

    sel = selectors.DefaultSelector()
    try:
        index = 0
        while True:
            started_one = False
            if index < len(ordered) and (deadline is None or _remaining(deadline) > 0):
                af, socktype, proto, _canon, sa = ordered[index]
                index += 1
                sock = None
                try:
                    sock = socket.socket(af, socktype, proto)
                    sock.setblocking(False)
                    if source_address:
                        sock.bind(source_address)
                    err = sock.connect_ex(sa)
                    if err == 0:
                        winner = sock
                        break
                    if err not in _IN_FLIGHT:
                        raise OSError(err, os.strerror(err))
                    sel.register(sock, selectors.EVENT_WRITE, sa)
                    pending[sock] = sa
                    started_one = True
                except OSError as exc:
                    exceptions.append(exc)
                    if sock is not None:
                        sock.close()

            if not pending:
                if index >= len(ordered):
                    break
                if deadline is not None and _remaining(deadline) <= 0:
                    break
                continue

            # Only until the next attempt is due, so a silent family does not hold the
            # others back; once all are in flight, wait out the deadline.
            budget = _remaining(deadline)
            if index < len(ordered):
                wait = delay if started_one else 0.0
                if budget is not None:
                    wait = min(wait, max(0.0, budget))
            else:
                wait = budget
            if wait is not None and wait <= 0 and budget is not None and budget <= 0:
                break

            for key, _mask in sel.select(wait):
                sock = key.fileobj
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                sel.unregister(sock)
                pending.pop(sock, None)
                if err == 0:
                    winner = sock
                    break
                exceptions.append(OSError(err, os.strerror(err)))
                sock.close()
            if winner is not None:
                break
            if index >= len(ordered) and not pending:
                break
            if deadline is not None and _remaining(deadline) <= 0:
                break
    finally:
        for sock in list(pending):
            try:
                sel.unregister(sock)
            except Exception:  # noqa: BLE001
                pass
            if sock is not winner:
                sock.close()
        sel.close()

    if winner is not None:
        return _settle(winner)

    if not exceptions:
        exceptions.append(socket.timeout("timed out"))
    if all_errors and _HAS_EXCEPTION_GROUP:
        raise ExceptionGroup("create_connection failed", exceptions)
    raise exceptions[0]


def activate_happy_eyeballs() -> bool:
    """Idempotently install the racing connector process-wide.

    Returns True when it is active. Failure is non-fatal: the stdlib's sequential walk
    is the pre-existing behaviour, slow rather than wrong.
    """
    global _activated
    if _activated:
        return True
    if not happy_eyeballs_enabled():
        return False
    try:
        # http.client reads this in HTTPConnection.__init__, not at class definition, so
        # patching before the first request reaches urllib and requests too.
        socket.create_connection = happy_eyeballs_connection
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "happy eyeballs unavailable (%s); connects keep the stdlib's "
            "one-timeout-per-address walk",
            exc,
        )
        return False
    _activated = True
    return True
