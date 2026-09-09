# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Landlock ABI 6 scoping, which blocks host abstract AF_UNIX sockets.

Those live in the shared network namespace, so no mount or bind rule hides them
and seccomp cannot close it either (the address is behind a pointer). Best
effort: on a pre-6.12 kernel nothing is applied and ``LIMITATIONS`` says so.
"""

from __future__ import annotations

import ctypes
import os
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

try:
    # Resolved at import, never in the forked child, where an import can
    # deadlock on a lock a thread held at fork time.
    _libc = ctypes.CDLL(None, use_errno = True)
    _libc.syscall.restype = ctypes.c_long
except OSError:  # pragma: no cover - a libc that will not load
    _libc = None


def abstract_scope_supported() -> bool:
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
    still deny restrict_self; the live probe decides."""
    if _libc is None:
        return
    attr = ctypes.create_string_buffer(_RULESET_ATTR, len(_RULESET_ATTR))
    ctypes.set_errno(0)
    ruleset = _libc.syscall(
        _NR_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.c_size_t(len(_RULESET_ATTR)),
        ctypes.c_uint32(0),
    )
    if ruleset < 0:
        return
    try:
        # Required before restrict_self for an unprivileged caller. Set again
        # here so this does not depend on which pre-exec it was composed with.
        _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        _libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, ctypes.c_int(int(ruleset)), ctypes.c_uint32(0))
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
