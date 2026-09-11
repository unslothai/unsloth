# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Landlock ABI 6 scoping, which blocks host abstract AF_UNIX sockets.

Those live in the shared network namespace, so no mount or bind rule hides them
and seccomp cannot close it either (the address is behind a pointer). Best
effort: on a pre-6.12 kernel nothing is applied and ``LIMITATIONS`` says so.
"""

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
# struct landlock_ruleset_attr; the third member is ABI 6's, so this size is
# itself the version check -- an older kernel answers E2BIG.
_RULESET_ATTR = struct.pack("=QQQ", 0, 0, _LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
_PR_SET_NO_NEW_PRIVS = 38
# Built at import, never in the forked child: see apply_abstract_scope.
_RULESET_ATTR_BUFFER = ctypes.create_string_buffer(_RULESET_ATTR, len(_RULESET_ATTR))
_RULESET_ATTR_REF = ctypes.byref(_RULESET_ATTR_BUFFER)
_RULESET_ATTR_SIZE = ctypes.c_size_t(len(_RULESET_ATTR))
_ZERO_FLAGS = ctypes.c_uint32(0)
# The child binds one socket and calls two syscalls, so this is a hang budget,
# not a work budget.
_PROBE_TIMEOUT_SECONDS = 5.0

try:
    # Resolved at import, never in the forked child, where an import can
    # deadlock on a lock a thread held at fork time.
    _libc = ctypes.CDLL(None, use_errno = True)
    _libc.syscall.restype = ctypes.c_long
except (OSError, TypeError, AttributeError):  # pragma: no cover
    # TypeError is Windows: CDLL(None) means "the running process" only where
    # dlopen has that convention, and ctypes there tests the name for a
    # separator before anything else. Importing this module raised, which is not
    # something a Linux-only helper should do on a platform that never calls it.
    _libc = None


@functools.lru_cache(maxsize = 1)
def abstract_scope_supported() -> bool:
    """Whether the scope can actually be APPLIED here, not whether the ABI has it.

    The two differ, and the difference disqualified a working sandbox. An outer
    sandbox or an exhausted nesting limit lets the ABI query succeed while
    create_ruleset or restrict_self is denied; apply_abstract_scope() cannot
    report that, since it runs post-fork in the child. Both callers then behaved
    as if the scope were in force: LIMITATIONS dropped
    host_abstract_sockets_reachable, and the probe armed a negative control that
    could not pass, so the whole bubblewrap backend was reported unavailable and
    auto gave up filesystem and PID isolation it could have had.

    So this proves it in a forked child that applies the scope for real. The
    child is where restrict_self is irreversible, which is exactly why the answer
    cannot be taken in this process. Cached: it costs a fork, both callers ask,
    and it cannot change without a restart.
    """
    if _libc is None:
        return False
    if not _abi_reports_scope():
        return False
    # The listener is bound HERE, in the unscoped parent, so it sits outside the
    # child's Landlock domain. That is the whole test: the scope stops a domain
    # reaching sockets outside itself and leaves its own alone, so a child that
    # binds and connects its own name is permitted and proves nothing. Measured:
    # scoped child -> parent's socket is EPERM, scoped child -> its own socket
    # connects.
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
        # Deadlined, and the child is killed rather than waited on: this forks a
        # MULTITHREADED server, and the child then runs real Python and libc
        # work, so a lock another thread held at fork time can leave it never
        # writing and never exiting. An unbounded read here is not one hung
        # probe, it is every tool call for the life of the process, since the
        # capability is asked for before any of them and the answer is cached.
        # No answer is a NO: the scope is unproven, LIMITATIONS keeps
        # host_abstract_sockets_reachable, and the launch still happens.
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
    """SIGKILL then wait. Harmless for a child that already exited: it is a
    zombie until reaped, so the signal lands on nothing and the wait returns."""
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
    """Runs in the forked child, so it must never raise or log. Success cannot be
    inferred from the ABI version, since an outer sandbox or the nesting limit can
    still deny restrict_self; the live probe decides.

    Everything allocatable is built at IMPORT and reused here. This is a pre-exec,
    so it runs after fork() in a process whose other threads are gone but whose
    locks are not, and an allocation that wants one of them deadlocks the child
    before exec -- where Popen is still waiting on the error pipe and its own
    timeout has not started. That does not make this async-signal-safe, which no
    Python pre-exec is, but it is the difference between the work this adds and
    the setsid the tool launches already did on main.
    """
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
        # Required before restrict_self for an unprivileged caller. Set again
        # here so this does not depend on which pre-exec it was composed with.
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
