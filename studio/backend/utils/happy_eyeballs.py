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


def _walk(infos, timeout, source_address, all_errors):
    """The stdlib's sequential loop over already-resolved addresses: no second lookup."""
    if not infos:
        raise OSError("getaddrinfo returns an empty list")
    exceptions = []
    for af, socktype, proto, _canon, sa in infos:
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            return sock
        except OSError as exc:
            if not all_errors:
                exceptions.clear()
            exceptions.append(exc)
            if sock is not None:
                sock.close()
    if all_errors and _HAS_EXCEPTION_GROUP:
        raise ExceptionGroup("create_connection failed", exceptions)  # novermin
    raise exceptions[0]


def happy_eyeballs_connection(
    address,
    timeout = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address = None,
    *,
    all_errors = False,
):
    """Drop-in ``socket.create_connection``: attempts overlap, each with the whole ``timeout``."""
    host, port = address
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
        resolved_timeout = socket.getdefaulttimeout()
    else:
        resolved_timeout = timeout

    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    # A fixed source port can only be bound by one socket at a time, so it cannot race.
    if len(infos) <= 1 or (source_address and source_address[1]):
        return _walk(infos, timeout, source_address, all_errors)

    ordered = _interleave(infos)
    delay = attempt_delay()
    exceptions: list = []
    failed: dict = {}
    pending: dict = {}  # every attempt gets the whole timeout
    winner = None

    def _fail(sa, exc):
        exceptions.append(exc)
        failed[sa] = exc

    sel = selectors.DefaultSelector()
    try:
        index = 0
        next_due = 0.0
        kick = True
        while True:
            now = time.monotonic()
            for sock, (sa, expiry) in list(pending.items()):
                if expiry is not None and expiry <= now:
                    sel.unregister(sock)
                    del pending[sock]
                    sock.close()
                    _fail(sa, socket.timeout("timed out"))
                    kick = True
            if index < len(ordered) and (kick or not pending or now >= next_due):
                kick = False
                af, socktype, proto, _canon, sa = ordered[index]
                index += 1
                next_due = now + delay
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
                    sel.register(sock, selectors.EVENT_WRITE)
                    pending[sock] = (
                        sa,
                        None if resolved_timeout is None else now + resolved_timeout,
                    )
                except OSError as exc:
                    _fail(sa, exc)
                    kick = True
                    if sock is not None:
                        sock.close()
                    continue
            if not pending:
                if index >= len(ordered):
                    break
                continue

            wakes = [e for _sa, e in pending.values() if e is not None]
            if index < len(ordered):
                wakes.append(next_due)
            wait = max(0.0, min(wakes) - time.monotonic()) if wakes else None
            for key, _mask in sel.select(wait):
                sock = key.fileobj
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                sel.unregister(sock)
                sa, _expiry = pending.pop(sock)
                if err == 0:
                    winner = sock
                    break
                sock.close()
                _fail(sa, OSError(err, os.strerror(err)))
                kick = True
            if winner is not None:
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
        winner.setblocking(True)
        winner.settimeout(resolved_timeout)
        return winner

    if not exceptions:
        exceptions.append(socket.timeout("timed out"))
    if all_errors and _HAS_EXCEPTION_GROUP:
        raise ExceptionGroup("create_connection failed", exceptions)  # novermin
    # Like the stdlib, the last address decides the error; hf_tcp_reachable reads ECONNREFUSED as up.
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
