# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Landlock ABI 6 scoping to block host abstract AF_UNIX sockets; best effort, pre-6.12 kernels record a limitation."""

from __future__ import annotations

import ctypes
import functools
import os
import select
import signal
import struct
from typing import Callable

_NR_LANDLOCK_CREATE_RULESET = 444
_NR_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
# The size is the version check: an older kernel answers E2BIG.
_RULESET_ATTR = struct.pack("=QQQ", 0, 0, _LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
_PR_SET_NO_NEW_PRIVS = 38
# Built at import, never in the forked child: see apply_abstract_scope.
_RULESET_ATTR_BUFFER = ctypes.create_string_buffer(_RULESET_ATTR, len(_RULESET_ATTR))
_RULESET_ATTR_REF = ctypes.byref(_RULESET_ATTR_BUFFER)
_RULESET_ATTR_SIZE = ctypes.c_size_t(len(_RULESET_ATTR))
_ZERO_FLAGS = ctypes.c_uint32(0)
_PROBE_TIMEOUT_SECONDS = 5.0

try:
    # Resolved at import, never in the forked child, where an import can deadlock.
    _libc = ctypes.CDLL(None, use_errno = True)
    _libc.syscall.restype = ctypes.c_long
except (OSError, TypeError, AttributeError):  # pragma: no cover
    # TypeError is Windows, where CDLL(None) is not the running process.
    _libc = None


@functools.lru_cache(maxsize = 1)
def abstract_scope_supported() -> bool:
    """Cache whether a child can apply the scope; probed in a child because restrict_self is irreversible."""
    if _libc is None:
        return False
    if not _abi_reports_scope():
        return False
    # Bind in the unscoped parent, so the child must not reach it once scoped.
    import socket as _socket

    name = b"\0unsloth-scope-" + os.urandom(6).hex().encode()
    listener = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        listener.bind(name)
        listener.listen(1)
    except OSError:
        listener.close()
        return False
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        try:
            os.close(read_fd)
            listener.close()
            apply_abstract_scope()
            client = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            client.settimeout(2)
            try:
                client.connect(name)
                os.write(write_fd, b"n")  # reached it, so nothing was applied
            except OSError:
                os.write(write_fd, b"y")  # refused, so the scope is in force
        except BaseException:
            try:
                os.write(write_fd, b"n")
            except OSError:
                pass
        finally:
            os._exit(0)
    os.close(write_fd)
    answer = b""
    try:
        # Bound the read and kill a stalled child; a timeout leaves the scope unproven.
        if select.select([read_fd], [], [], _PROBE_TIMEOUT_SECONDS)[0]:
            answer = os.read(read_fd, 1)
    except OSError:
        answer = b""
    finally:
        os.close(read_fd)
        listener.close()
        _reap(pid)
    return answer == b"y"


def _reap(pid: int) -> None:
    """SIGKILL then wait; harmless for an already-exited child."""
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def _abi_reports_scope() -> bool:
    """NULL attr only reports the ABI version and changes nothing."""
    if _libc is None:
        return False
    try:
        version = _libc.syscall(
            _NR_LANDLOCK_CREATE_RULESET,
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except (OSError, AttributeError, TypeError):
        return False
    return version >= 6


def apply_abstract_scope() -> None:
    """Apply in the forked child without raising or logging; the probe checks success."""
    if _libc is None:
        return
    ctypes.set_errno(0)
    ruleset = _libc.syscall(
        _NR_LANDLOCK_CREATE_RULESET,
        _RULESET_ATTR_REF,
        _RULESET_ATTR_SIZE,
        _ZERO_FLAGS,
    )
    if ruleset < 0:
        return
    try:
        # Required before restrict_self for an unprivileged caller; set here too.
        _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        _libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, ctypes.c_int(int(ruleset)), _ZERO_FLAGS)
    finally:
        try:
            os.close(int(ruleset))
        except OSError:
            pass


def with_abstract_scope(preexec_fn: "Callable[[], None] | None") -> "Callable[[], None]":
    """*preexec_fn* first (it is the setsid the kill paths signal), then the scope."""

    def preexec() -> None:
        if preexec_fn is not None:
            preexec_fn()
        apply_abstract_scope()

    return preexec
