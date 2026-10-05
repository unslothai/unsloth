# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RFC 8305 Happy Eyeballs for sync connects.

``socket.create_connection`` applies ``timeout`` per resolved address, so a black-holed
family (VPN without IPv6) costs one timeout per address (python/cpython#88810). httpx's
sync backend and ``http.client`` call it; ``requests`` does not (urllib3 2.x has its own
loop). Spawned children must re-activate, like :mod:`utils.native_tls`.
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

_IN_FLIGHT = frozenset({errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY})
_HAS_EXCEPTION_GROUP = sys.version_info >= (3, 11)

_logger = logging.getLogger(__name__)
_original_create_connection = socket.create_connection
_activated = False


def happy_eyeballs_enabled() -> bool:
    return os.environ.get(_ENV, "").strip().lower() not in _FALSEY


def attempt_delay() -> float:
    try:
        value = float(os.environ.get(_DELAY_ENV, "").strip() or _DEFAULT_DELAY)
    except ValueError:
        return _DEFAULT_DELAY
    return max(0.0, value)


def _interleave(infos: list) -> list:
    """Round-robin across families, resolver order kept within each."""
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
    """Drop-in ``socket.create_connection``: ends within ``timeout + (n - 1) * delay``."""
    host, port = address
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
        resolved_timeout = socket.getdefaulttimeout()
    else:
        resolved_timeout = timeout

    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    if len(infos) <= 1:
        if _HAS_EXCEPTION_GROUP:
            return _original_create_connection(  # novermin
                address,
                timeout,
                source_address,
                all_errors = all_errors,
            )
        return _original_create_connection(address, timeout, source_address)

    ordered = _interleave(infos)
    delay = attempt_delay()
    # The last attempt opens (n - 1) staggers in and must still get the whole timeout.
    deadline = (
        None
        if resolved_timeout is None
        else time.monotonic() + resolved_timeout + (len(ordered) - 1) * delay
    )
    exceptions: list = []
    failed: dict = {}  # sockaddr -> its failure
    pending: dict = {}  # socket -> sockaddr
    winner = None
    timed_out = False

    def _settle(sock):
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
                    failed[sa] = exc
                    if sock is not None:
                        sock.close()

            if not pending:
                if index >= len(ordered):
                    break
                if deadline is not None and _remaining(deadline) <= 0:
                    break
                continue

            budget = _remaining(deadline)
            if index < len(ordered):
                wait = delay if started_one else 0.0
                if budget is not None:
                    wait = min(wait, max(0.0, budget))
            else:
                wait = budget
            if wait is not None and wait <= 0 and budget is not None and budget <= 0:
                timed_out = True
                break

            for key, _mask in sel.select(wait):
                sock = key.fileobj
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                sel.unregister(sock)
                sa = pending.pop(sock, None)
                if err == 0:
                    winner = sock
                    break
                exc = OSError(err, os.strerror(err))
                exceptions.append(exc)
                failed[sa] = exc
                sock.close()
            if winner is not None:
                break
            if index >= len(ordered) and not pending:
                break
            if deadline is not None and _remaining(deadline) <= 0:
                timed_out = bool(pending)
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

    if timed_out or not exceptions:
        exceptions.append(socket.timeout("timed out"))
    if all_errors and _HAS_EXCEPTION_GROUP:
        raise ExceptionGroup("create_connection failed", exceptions)  # novermin
    # Like the stdlib, the last address in resolver order decides the error:
    # hf_tcp_reachable reads ECONNREFUSED as "the endpoint answered".
    raise failed.get(infos[-1][4]) or socket.timeout("timed out")


def activate_happy_eyeballs() -> bool:
    """Idempotently install the connector process-wide; True when active."""
    global _activated
    if _activated:
        return True
    if not happy_eyeballs_enabled():
        return False
    try:
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


def inline_activation_source() -> str:
    """Activation as source, for a ``python -c`` child that cannot import backend modules."""
    return (
        "try:\n"
        "    import importlib.util as _he_util\n"
        f"    _he_spec = _he_util.spec_from_file_location('_studio_happy_eyeballs', {os.path.abspath(__file__)!r})\n"
        "    _he_mod = _he_util.module_from_spec(_he_spec)\n"
        "    _he_spec.loader.exec_module(_he_mod)\n"
        "    _he_mod.activate_happy_eyeballs()\n"
        "except Exception:\n"
        "    pass\n"
    )
