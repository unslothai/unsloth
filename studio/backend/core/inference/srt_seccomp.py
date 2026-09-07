# SPDX-License-Identifier: AGPL-3.0-only
"""Inherited Process Guard rules for channels outside network namespaces.

Prepare ctypes state before fork; install only in the disposable launch child.
The filesystem namespace, not this filter, confines private Unix IPC paths.
"""
import ctypes
import errno
import platform
import sys

KILL = 0x80000000
DENY = 0x00050000 | errno.EPERM
ALLOW = 0x7FFF0000
ABIS = {"x86_64": (0xC000003E, 41, 53), "aarch64": (0xC00000B7, 198, 199)}


def program(machine: str) -> tuple[tuple[int, int, int, int], ...]:
    """Return classic BPF instructions for the two supported native ABIs."""
    if machine not in ABIS:
        raise RuntimeError("SRT syscall guard requires Linux x86_64 or aarch64")
    arch, socket_nr, socketpair_nr = ABIS[machine]
    code = [
        (0x20, 0, 0, 4),  # seccomp_data.arch
        (0x15, 1, 0, arch),
        (0x06, 0, 0, KILL),
        (0x20, 0, 0, 0),  # seccomp_data.nr
    ]
    if machine == "x86_64":
        code += [(0x45, 0, 1, 0x40000000), (0x06, 0, 0, KILL)]
    for number in (425, 426, 427):  # setup, enter, register; same on both ABIs
        code += [(0x15, 0, 1, number), (0x06, 0, 0, DENY)]
    code += [
        (0x15, 1, 0, socket_nr),
        (0x15, 0, 3, socketpair_nr),
        (0x20, 0, 0, 16),  # args[0], family (low word, native little endian)
        (0x15, 0, 1, 40),  # AF_VSOCK
        (0x06, 0, 0, DENY),
        (0x06, 0, 0, ALLOW),
    ]
    return tuple(code)


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


_prctl = None
_pointer = 0
if sys.platform == "linux" and sys.byteorder == "little" and platform.machine().lower() in ABIS:
    _instructions = program(platform.machine().lower())
    _array = (_Filter * len(_instructions))(*(_Filter(*item) for item in _instructions))
    _descriptor = _Program(len(_instructions), _array)
    _pointer = ctypes.addressof(_descriptor)
    _libc = ctypes.CDLL(None)
    _prctl = _libc.prctl
    _prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    _prctl.restype = ctypes.c_int


def install() -> None:
    """Fail closed in the fork child; never call in the Studio parent process."""
    if _prctl is None:
        raise RuntimeError("SRT syscall guard unavailable on this platform")
    if _prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise RuntimeError("SRT syscall guard could not set no_new_privs")
    if _prctl(22, 2, _pointer, 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        raise RuntimeError("SRT syscall guard installation failed")
