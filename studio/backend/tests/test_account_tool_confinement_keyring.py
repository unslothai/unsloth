# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's tool shares the server's UID and Landlock has no right for keys, so the
managed seccomp filter refuses keyctl / add_key / request_key, as the owner's bwrap filter does."""

import platform
import subprocess
import sys

import pytest

from core.inference import sandbox_seccomp


def test_the_managed_filter_refuses_the_keyring_syscalls():
    for machine in ("x86_64", "aarch64"):
        program = sandbox_seccomp.unix_socket_program(machine)
        for number in sandbox_seccomp._KEYRING_SYSCALLS[machine]:
            index = program.index((sandbox_seccomp._JEQ, 0, 1, number))
            assert program[index + 1] == (sandbox_seccomp._RET, 0, 0, sandbox_seccomp._EPERM)


_CHILD = r"""
import ctypes, errno, platform, struct, sys
sys.path.insert(0, sys.argv[1])
from core.inference import sandbox_seccomp
code = sandbox_seccomp.unix_socket_program(platform.machine())
class F(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]
class P(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(F))]
arr = (F * len(code))(*(F(*c) for c in code))
prog = P(len(code), ctypes.cast(arr, ctypes.POINTER(F)))
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(38, 1, 0, 0, 0) == 0
assert libc.prctl(22, 2, ctypes.byref(prog), 0, 0) == 0
nr = sandbox_seccomp._KEYRING_SYSCALLS[platform.machine().lower()][2]
rc = libc.syscall(nr, 0, ctypes.c_long(-4), 0)  # KEYCTL_GET_KEYRING_ID, KEY_SPEC_USER_KEYRING
print("KEYCTL:" + ("refused" if rc < 0 and ctypes.get_errno() == errno.EPERM else "open"))
"""


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine().lower() not in sandbox_seccomp._ABIS,
    reason = "Linux x86_64 / aarch64 seccomp",
)
def test_a_filtered_child_gets_eperm_from_keyctl():
    import os

    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, backend], capture_output = True, text = True, timeout = 60
    )
    assert "KEYCTL:refused" in out.stdout, out.stdout + out.stderr
