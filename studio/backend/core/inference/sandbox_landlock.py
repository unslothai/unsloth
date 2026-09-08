# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Landlock scope that closes the one hole left in the network namespace.

The sandbox shares the host's network namespace on purpose: tool calls
pip-install and download models. An ABSTRACT AF_UNIX socket lives in that
namespace and not in the filesystem, so no mount, bind or seccomp rule in here
touches it -- ``/proc/net/unix`` lists every one of them by name and a connect
needs nothing but the name. On an ordinary Linux desktop that reaches the
session bus and the X server, and a peer that authenticates on the inherited uid
will start a process outside the jail. It is a way out of the filesystem
boundary, through the operation the design leaves open.

seccomp cannot close it: the address is behind a pointer and a filter cannot
dereference one. ``--unshare-net`` closes it and takes the network with it,
which is the trade the whole backend exists to avoid. Landlock ABI 6 (Linux
6.12) added exactly the missing scope, it is designed to be applied unprivileged
with no_new_privs already set, and it is inherited across the exec into bwrap
and everything bwrap starts.

Best effort by construction. On an older kernel nothing is applied and
``sandbox_linux.LIMITATIONS`` says so, because a boundary that fails silently is
worse than one that is named.
"""

from __future__ import annotations

import ctypes
import os
import struct
from typing import Callable

# Same numbers on x86_64 and aarch64: Landlock landed in the shared range.
_NR_LANDLOCK_CREATE_RULESET = 444
_NR_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
# struct landlock_ruleset_attr: handled_access_fs, handled_access_net, scoped.
# The third member is what ABI 6 added, so passing this size is itself the
# version check: an older kernel answers E2BIG rather than silently ignoring it.
_RULESET_ATTR = struct.pack("=QQQ", 0, 0, _LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
_PR_SET_NO_NEW_PRIVS = 38

try:
    # Resolved at import, never after the fork: the pre-exec below runs in the
    # forked child, where an import can deadlock on the lock a thread held at
    # fork time. Same rule as tools._sandbox_preexec.
    _libc = ctypes.CDLL(None, use_errno = True)
    _libc.syscall.restype = ctypes.c_long
except OSError:  # pragma: no cover - a libc that will not load
    _libc = None


def abstract_scope_supported() -> bool:
    """Whether this kernel has the Landlock scope, asked once at import.

    One syscall with a NULL attribute, which only reports the ABI version and
    changes nothing. The answer cannot change under a running kernel.
    """
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
    # ABI 6 is where `scoped` appears; anything below it cannot express this.
    return version >= 6


def apply_abstract_scope() -> None:
    """Put this process, and everything it execs, out of reach of host abstract sockets.

    Runs in the forked child. Silent on a kernel that cannot do it, because
    ``auto`` is the mode that never refuses and the limitation already says the
    boundary is not there.
    """
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
        # Required before restrict_self for an unprivileged caller. tools.py's
        # pre-exec sets it too; setting it twice is free and this must not depend
        # on which pre-exec it was composed with.
        _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        _libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, ctypes.c_int(int(ruleset)), ctypes.c_uint32(0))
    finally:
        try:
            os.close(int(ruleset))
        except OSError:
            pass


def with_abstract_scope(preexec_fn: "Callable[[], None] | None") -> "Callable[[], None]":
    """*preexec_fn* followed by the scope, as one pre-exec.

    The caller's comes first: it is the setsid every kill path in tools.py
    signals, and it must run whether or not this kernel has the scope.
    """

    def preexec() -> None:
        if preexec_fn is not None:
            preexec_fn()
        apply_abstract_scope()

    return preexec
