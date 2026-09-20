# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned identity of the MXC binaries, and how to check it.

Deliberately dependency-free. The installer runs at setup time, before the
backend's imports are necessarily satisfiable, and it must never degrade to
installing an unverified executor; the backend needs the same values at launch
time. One module with no imports beyond the standard library is readable from
both, so the two sides cannot hold different ideas of what the binary is.

The digests are checked in TWO places on purpose. At install time, so a
compromised registry response or mirror is refused before anything is written.
At launch time, because the managed directory is user-writable: a same-user
process, including a software-safeguarded tool call made before isolation
became available, could replace the executor afterwards, and the whole boundary
rests on this one binary being the one we pinned.
"""

from __future__ import annotations

import hashlib
import os

MXC_VERSION = "0.8.0"

# sha256 of the published npm tarball, cross-checked against the registry's own
# integrity metadata for 0.8.0
# (sha512-pnf5QsASwp+qtRi5uth2GDjwuyG0rHWRpxCf3RbAjQ4wDTNfBX/9l0A+RVZspU2agpF3/11uWB1JisIS7WrNYg==).
TARBALL_SHA256 = "06bb2399d7e98ab1907acf851e12a4e44748dd467b79d3e53c2f2fbf569da14e"

# sha256 of the binaries extracted from that tarball, per architecture.
# wxc-exec.exe is the sandbox itself; wxc-host-prep.exe carries
# requireAdministrator in its manifest and is run elevated, so it is the other
# binary whose identity matters.
EXECUTOR_SHA256 = {
    "x64": "6049c64723af1173c3739dc6cd6b2f33f6c021bb2832c4216233cba7f71aee9a",
    "arm64": "dde1c592270e9a659b01dccad70362da7b99fec114885fa4d625507aa775a503",
}
HOST_PREP_SHA256 = {
    "x64": "a9b8b14a11a1c5888641297c26abca547c2afa4435085c03ccfebd1deface310",
    "arm64": "c8ddcf0461ae3d7ddff4656abb87b51236146742bb711a2e03fdc23cb8a966b0",
}


def arch_dir() -> str:
    machine = (os.environ.get("PROCESSOR_ARCHITECTURE") or "").lower()
    if "arm64" in machine:
        return "arm64"
    return "x64"


def digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def matches_pin(path: str, expected: "dict[str, str]") -> bool:
    """Whether the file at ``path`` is the pinned binary for this architecture.

    Any unreadable file, unknown architecture or mismatch is False: this is a
    trust check, so it fails closed rather than assuming the good case.
    """
    wanted = expected.get(arch_dir())
    if not wanted:
        return False
    try:
        return digest(path) == wanted
    except OSError:
        return False


class FileHold:
    """An open handle that denies other processes write, rename and delete.

    Verifying a digest and then handing the PATHNAME to a process launcher
    leaves a window, and the window is the interesting part: for the executor
    it spans policy construction, the live probe and the cleanup that runs the
    same file again; for the elevated helper it spans the UAC prompt the user
    has to answer. A same-user process, which is precisely the attacker these
    pins exist for, can swap the file inside it.

    Windows honours a deny-write share mode at the filesystem level, so the
    verified bytes cannot change while this is open.
    """

    def __init__(self, handle: object) -> None:
        self._handle = handle

    def close(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            import ctypes

            if not hasattr(ctypes, "WinDLL"):
                return
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            # argtypes is not optional here. Without it ctypes converts the
            # handle with the default c_int, which on 64-bit Windows truncates
            # it, so nothing is closed and every probe and launch leaks a
            # deny-write handle on the executor until Studio exits, at which
            # point the file cannot be replaced or reinstalled either.
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.CloseHandle(ctypes.c_void_p(handle))
        except Exception:  # noqa: BLE001 - a released handle is not worth a failed call
            pass

    def __enter__(self) -> "FileHold":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def hold_file(path: str) -> "FileHold | None":
    """Open ``path`` so nothing else can modify or replace it.

    None on a non-Windows interpreter, keyed on WinDLL being absent rather
    than on sys.platform, because the policy tests spoof the platform to
    exercise the Windows branches from any host. Raises OSError when the file
    cannot be opened that way, which means somebody else already holds it in a
    manner that would let them write to it.
    """
    import ctypes
    import sys

    if sys.platform != "win32" or not hasattr(ctypes, "WinDLL"):
        return None
    from ctypes import wintypes

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    invalid_handle = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    )
    handle = kernel32.CreateFileW(path, generic_read, file_share_read, None, open_existing, 0, None)
    if handle == invalid_handle:
        raise OSError(ctypes.get_last_error(), f"WinError {ctypes.get_last_error()}")
    return FileHold(handle)
