# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seccomp filter for bwrap: blocks AF_VSOCK, io_uring, keyrings and nested user namespaces; AF_UNIX/AF_INET allowed on purpose."""

from __future__ import annotations

import errno
import platform
import struct
import sys
import tempfile
from typing import BinaryIO

# machine -> (AUDIT_ARCH, socket, socketpair); anything else is refused.
_ABIS = {
    "x86_64": (0xC000003E, 41, 53),
    "amd64": (0xC000003E, 41, 53),
    "aarch64": (0xC00000B7, 198, 199),
    "arm64": (0xC00000B7, 198, 199),
}
_USERNS_SYSCALLS = {
    "x86_64": (56, 272, 435),
    "amd64": (56, 272, 435),
    "aarch64": (220, 97, 435),
    "arm64": (220, 97, 435),
}
_IO_URING = (425, 426, 427)
# Denied rather than joining an empty keyring, since joining needs the syscall being removed.
_KEYRING_SYSCALLS = {
    "x86_64": (248, 249, 250),
    "amd64": (248, 249, 250),
    "aarch64": (217, 218, 219),
    "arm64": (217, 218, 219),
}
_X32_SYSCALL_BIT = 0x40000000
_CLONE_NEWUSER = 0x10000000
_AF_VSOCK = 40

# struct seccomp_data: nr@0, arch@4, args[0]@16.
_LOAD, _JEQ, _JSET, _RET = 0x20, 0x15, 0x45, 0x06
_KILL = 0x80000000
_EPERM = 0x00050000 | errno.EPERM
_ENOSYS = 0x00050000 | errno.ENOSYS
_ALLOW = 0x7FFF0000


def program(machine: str, *, block_userns: bool = False) -> tuple[tuple[int, int, int, int], ...]:
    key = machine.lower()
    if key not in _ABIS or sys.byteorder != "little":
        raise RuntimeError(
            "the Studio sandbox seccomp filter requires little-endian x86_64 or aarch64"
        )
    audit_arch, socket_nr, socketpair_nr = _ABIS[key]
    code: list[tuple[int, int, int, int]] = [
        (_LOAD, 0, 0, 4),
        (_JEQ, 1, 0, audit_arch),
        (_RET, 0, 0, _KILL),  # an ABI switch would read args at other offsets
        (_LOAD, 0, 0, 0),
    ]
    if block_userns:
        # clone3() must report ENOSYS so glibc falls back to clone(), whose flags are checked.
        clone_nr, unshare_nr, clone3_nr = _USERNS_SYSCALLS[key]
        code += [
            # unshare() is filtered on its flags, not refused: only CLONE_NEWUSER is the target.
            (_JEQ, 0, 4, unshare_nr),
            (_LOAD, 0, 0, 16),
            (_JSET, 0, 1, _CLONE_NEWUSER),
            (_RET, 0, 0, _EPERM),
            (_LOAD, 0, 0, 0),
            (_JEQ, 0, 1, clone3_nr),
            (_RET, 0, 0, _ENOSYS),
            (_JEQ, 0, 4, clone_nr),
            (_LOAD, 0, 0, 16),
            (_JSET, 0, 1, _CLONE_NEWUSER),
            (_RET, 0, 0, _EPERM),
            (_LOAD, 0, 0, 0),
        ]
    if key in ("x86_64", "amd64"):
        # x32 numbers alias the 64-bit table, so an unfiltered x32 call bypasses this filter.
        code += [(_JSET, 0, 1, _X32_SYSCALL_BIT), (_RET, 0, 0, _KILL)]
    for number in (*_IO_URING, *_KEYRING_SYSCALLS[key]):
        code += [(_JEQ, 0, 1, number), (_RET, 0, 0, _EPERM)]
    code += [
        (_JEQ, 1, 0, socket_nr),
        (_JEQ, 0, 3, socketpair_nr),
        (_LOAD, 0, 0, 16),
        (_JEQ, 0, 1, _AF_VSOCK),
        (_RET, 0, 0, _EPERM),
        (_RET, 0, 0, _ALLOW),
    ]
    return tuple(code)


def program_bytes(*, block_userns: bool = False, machine: str | None = None) -> bytes:
    instructions = program(machine or platform.machine(), block_userns = block_userns)
    return b"".join(struct.pack("=HBBI", *instruction) for instruction in instructions)


def filter_file(*, block_userns: bool = False) -> BinaryIO:
    """Rewound because bwrap reads ``--seccomp FD`` to EOF."""
    stream = tempfile.TemporaryFile(prefix = "unsloth-sandbox-seccomp-")
    try:
        stream.write(program_bytes(block_userns = block_userns))
        stream.flush()
        stream.seek(0)
    except Exception:
        stream.close()
        raise
    return stream
