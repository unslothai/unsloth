# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows handle leases and private creation for broker-owned runtime content.

No permissions on installed interpreters or existing directories are changed.
Sharing locks stop replacement/writing while handles are held; ACL checks are
separate because sharing modes do not prevent WRITE_DAC by a trusted host owner.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes as W
from functools import lru_cache
import os
from pathlib import Path

from .dependencies import checked_path
from .profiles import WindowsRuntimeError

READ = 0x80000000
WRITE = 0x40000000
READ_CONTROL = 0x20000
SHARE_READ = 1
SHARE_WRITE = 2
STREAM_CHUNK_BYTES = 1024 * 1024
_INVALID_HANDLE = ctypes.c_void_p(-1).value


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("length", W.DWORD), ("descriptor", ctypes.c_void_p), ("inherit", W.BOOL)]


class _FileInfo(ctypes.Structure):
    _fields_ = [
        ("attributes", W.DWORD),
        ("created", W.FILETIME),
        ("accessed", W.FILETIME),
        ("written", W.FILETIME),
        ("volume", W.DWORD),
        ("size_high", W.DWORD),
        ("size_low", W.DWORD),
        ("links", W.DWORD),
        ("index_high", W.DWORD),
        ("index_low", W.DWORD),
    ]

    @property
    def size(self):
        return (self.size_high << 32) | self.size_low


def _error(operation):
    error = ctypes.get_last_error()
    code = "WINDOWS_SANDBOX_STORE_BUSY" if error == 32 else "WINDOWS_SANDBOX_CONTENT_INVALID"
    return WindowsRuntimeError(code, f"{operation} failed (WinError {error}).")


class NativeFiles:
    def __init__(self):
        if os.name != "nt":
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_UNSUPPORTED", "Runtime storage requires Windows NTFS."
            )
        self.kernel = ctypes.WinDLL("kernel32", use_last_error = True, winmode = 0x800)
        self.security = ctypes.WinDLL("advapi32", use_last_error = True, winmode = 0x800)
        pvoid = ctypes.POINTER(ctypes.c_void_p)
        pdword = ctypes.POINTER(W.DWORD)
        declarations = (
            (self.kernel, "GetDriveTypeW", W.UINT, [W.LPCWSTR]),
            (
                self.kernel,
                "CreateFileW",
                W.HANDLE,
                [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, W.HANDLE],
            ),
            (self.kernel, "CloseHandle", W.BOOL, [W.HANDLE]),
            (self.kernel, "GetCurrentProcess", W.HANDLE, []),
            (
                self.kernel,
                "GetFileInformationByHandle",
                W.BOOL,
                [W.HANDLE, ctypes.POINTER(_FileInfo)],
            ),
            (
                self.kernel,
                "GetFinalPathNameByHandleW",
                W.DWORD,
                [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD],
            ),
            (self.kernel, "CreateDirectoryW", W.BOOL, [W.LPCWSTR, ctypes.c_void_p]),
            (
                self.kernel,
                "ReadFile",
                W.BOOL,
                [W.HANDLE, ctypes.c_void_p, W.DWORD, pdword, ctypes.c_void_p],
            ),
            (
                self.kernel,
                "WriteFile",
                W.BOOL,
                [W.HANDLE, ctypes.c_void_p, W.DWORD, pdword, ctypes.c_void_p],
            ),
            (self.kernel, "FlushFileBuffers", W.BOOL, [W.HANDLE]),
            (self.kernel, "LocalFree", ctypes.c_void_p, [ctypes.c_void_p]),
            (
                self.kernel,
                "GetVolumeInformationW",
                W.BOOL,
                [W.LPCWSTR, W.LPWSTR, W.DWORD, pdword, pdword, pdword, W.LPWSTR, W.DWORD],
            ),
            (
                self.security,
                "OpenProcessToken",
                W.BOOL,
                [W.HANDLE, W.DWORD, ctypes.POINTER(W.HANDLE)],
            ),
            (
                self.security,
                "GetTokenInformation",
                W.BOOL,
                [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, pdword],
            ),
            (
                self.security,
                "ConvertSidToStringSidW",
                W.BOOL,
                [ctypes.c_void_p, ctypes.POINTER(W.LPWSTR)],
            ),
            (
                self.security,
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                W.BOOL,
                [W.LPCWSTR, W.DWORD, pvoid, pdword],
            ),
            (
                self.security,
                "ConvertSecurityDescriptorToStringSecurityDescriptorW",
                W.BOOL,
                [ctypes.c_void_p, W.DWORD, W.DWORD, ctypes.POINTER(W.LPWSTR), pdword],
            ),
            (
                self.security,
                "GetSecurityInfo",
                W.DWORD,
                [W.HANDLE, W.DWORD, W.DWORD, pvoid, pvoid, pvoid, pvoid, pvoid],
            ),
            (
                self.security,
                "GetSecurityDescriptorDacl",
                W.BOOL,
                [ctypes.c_void_p, ctypes.POINTER(W.BOOL), pvoid, ctypes.POINTER(W.BOOL)],
            ),
            (
                self.security,
                "SetSecurityInfo",
                W.DWORD,
                [
                    W.HANDLE,
                    W.DWORD,
                    W.DWORD,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
            ),
        )
        for dll, name, result, args in declarations:
            function = getattr(dll, name)
            function.restype, function.argtypes = result, args
        token = W.HANDLE()
        if not self.security.OpenProcessToken(
            self.kernel.GetCurrentProcess(), 8, ctypes.byref(token)
        ):
            raise _error("OpenProcessToken(store owner)")
        try:
            data = ctypes.create_string_buffer(4096)
            needed = W.DWORD()
            if not self.security.GetTokenInformation(
                token, 1, data, len(data), ctypes.byref(needed)
            ):
                raise _error("GetTokenInformation(store owner)")
            sid = ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0]
            text = W.LPWSTR()
            if not self.security.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                raise _error("ConvertSidToStringSidW")
            try:
                self.owner = text.value
            finally:
                self.kernel.LocalFree(text)
        finally:
            self.kernel.CloseHandle(token)
        # Explicit protected DACL at creation, not chmod after public creation.
        sddl = f"O:{self.owner}D:P(A;;FA;;;{self.owner})(A;;FA;;;SY)"
        with self.security_attributes(sddl) as attributes:
            self.private_sddl = self._sddl(attributes.descriptor)

    def _sddl(self, descriptor):
        text, size = W.LPWSTR(), W.DWORD()
        if not self.security.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, 1, 5, ctypes.byref(text), ctypes.byref(size)
        ):
            raise _error("ConvertSecurityDescriptorToStringSecurityDescriptorW")
        try:
            if size.value > 65536:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CONTENT_INVALID", "Oversized file security descriptor."
                )
            # SetSecurityInfo may mark a protected DACL auto-inherited even
            # when it contains only explicit ACEs. Normalize that control bit,
            # not any ACE's inheritance flags or the protection bit itself.
            return text.value.replace("D:PAI(", "D:P(")
        finally:
            self.kernel.LocalFree(text)

    @contextmanager
    def security_attributes(self, sddl = None):
        descriptor = ctypes.c_void_p()
        if not self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl or self.private_sddl, 1, ctypes.byref(descriptor), None
        ):
            raise _error("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        try:
            yield _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        finally:
            self.kernel.LocalFree(descriptor)

    def security_text(self, handle):
        descriptor = ctypes.c_void_p()
        result = self.security.GetSecurityInfo(
            handle, 1, 5, None, None, None, None, ctypes.byref(descriptor)
        )
        if result:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID", f"GetSecurityInfo failed ({result})."
            )
        try:
            return self._sddl(descriptor)
        finally:
            self.kernel.LocalFree(descriptor)

    def require_private(self, handle):
        if self.security_text(handle) != self.private_sddl:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID", "Runtime store owner or private DACL changed."
            )

    def set_owned_dacl(self, handle, sddl):
        """Set a previously validated store DACL, never the owner or a host ACL.

        Caller holds the store mutation lock and validates the current complete
        DACL before deriving the new one. No ACE is inheritable; metadata and
        unlisted files never acquire payload access through inheritance.
        """
        with self.security_attributes(sddl) as attributes:
            present, defaulted, acl = W.BOOL(), W.BOOL(), ctypes.c_void_p()
            if (
                not self.security.GetSecurityDescriptorDacl(
                    attributes.descriptor,
                    ctypes.byref(present),
                    ctypes.byref(acl),
                    ctypes.byref(defaulted),
                )
                or not present.value
                or not acl
            ):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CONTENT_INVALID", "Missing private content DACL."
                )
            result = self.security.SetSecurityInfo(
                handle,
                1,
                4 | 0x80000000,
                None,
                None,
                acl,
                None,
            )
            if result:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CONTENT_INVALID",
                    f"SetSecurityInfo(content) failed ({result}).",
                )

    def require_ntfs(self, path):
        if self.kernel.GetDriveTypeW(Path(path).anchor) != 3:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_UNSUPPORTED", "Runtime storage requires a fixed local drive."
            )
        name = ctypes.create_unicode_buffer(32)
        flags = W.DWORD()
        if not self.kernel.GetVolumeInformationW(
            Path(path).anchor, None, 0, None, None, ctypes.byref(flags), name, len(name)
        ):
            raise _error("GetVolumeInformationW")
        if name.value != "NTFS" or not flags.value & 8:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_UNSUPPORTED", "Runtime storage requires NTFS persistent ACLs."
            )

    @staticmethod
    def native_path(path):
        # Only called with checked local absolute paths, not user device paths.
        return "\\\\?\\" + str(path)

    def open(
        self,
        path,
        *,
        directory = False,
        exclusive = False,
        write_dac = False,
    ):
        # Metadata-only directory handles do not enforce delete-sharing denial.
        # Include FILE_LIST_DIRECTORY so this lease also prevents rename/removal.
        access = (0x81 if directory else READ) | READ_CONTROL | (0x40000 if write_dac else 0)
        share = (SHARE_READ | SHARE_WRITE) if directory else SHARE_READ
        handle = self.kernel.CreateFileW(
            self.native_path(path),
            access,
            0 if exclusive else share,
            None,
            3,
            0x00200000 | (0x02000000 if directory else 0),
            None,
        )
        if handle == _INVALID_HANDLE:
            raise _error("CreateFileW(content lease)")
        try:
            self.require_path(handle, path, directory = directory)
            return handle
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise

    def require_path(
        self,
        handle,
        path,
        *,
        directory = False,
    ):
        self.info(handle, directory = directory)
        final = ctypes.create_unicode_buffer(32768)
        size = self.kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        if not 0 < size < len(final) or os.path.normcase(final.value) != os.path.normcase(
            self.native_path(path)
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID",
                "Runtime handle path differs from validated path.",
            )

    def info(
        self,
        handle,
        *,
        directory = False,
    ):
        info = _FileInfo()
        if not self.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise _error("GetFileInformationByHandle")
        if (
            info.attributes & (0x400 | 0x1000)
            or bool(info.attributes & 0x10) != directory
            or (not directory and info.links != 1)
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID",
                "Reparse, offline, hardlinked or wrong-type runtime object.",
            )
        return info

    def iter_read(self, handle, limit):
        """Read a pinned file once, retaining bounded memory and exact size checks."""
        info = self.info(handle)
        if info.size > limit:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID", "Runtime content exceeds its byte limit."
            )
        remaining = info.size
        while remaining:
            chunk = ctypes.create_string_buffer(min(STREAM_CHUNK_BYTES, remaining))
            count = W.DWORD()
            if not self.kernel.ReadFile(handle, chunk, len(chunk), ctypes.byref(count), None):
                raise _error("ReadFile(content)")
            if not 0 < count.value <= len(chunk):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CONTENT_INVALID", "Runtime content was truncated."
                )
            remaining -= count.value
            yield chunk.raw[: count.value]
        after = self.info(handle)
        if (after.volume, after.index_high, after.index_low, after.size) != (
            info.volume,
            info.index_high,
            info.index_low,
            info.size,
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CONTENT_INVALID", "Runtime content changed during reading."
            )

    def read(self, handle, limit):
        return b"".join(self.iter_read(handle, limit))

    def mkdir(self, path):
        with self.security_attributes() as attributes:
            if not self.kernel.CreateDirectoryW(self.native_path(path), ctypes.byref(attributes)):
                raise _error("CreateDirectoryW(private content)")

    def create(self, path, data):
        with self.security_attributes() as attributes:
            handle = self.kernel.CreateFileW(
                self.native_path(path),
                WRITE | READ_CONTROL,
                0,
                ctypes.byref(attributes),
                1,
                0x00200000,
                None,
            )
        if handle == _INVALID_HANDLE:
            raise _error("CreateFileW(private content)")
        try:
            self.require_private(handle)
            chunks = (
                (
                    data[offset : offset + STREAM_CHUNK_BYTES]
                    for offset in range(0, len(data), STREAM_CHUNK_BYTES)
                )
                if isinstance(data, bytes)
                else data
            )
            for chunk in chunks:
                if not isinstance(chunk, bytes) or not 0 < len(chunk) <= STREAM_CHUNK_BYTES:
                    raise WindowsRuntimeError(
                        "WINDOWS_SANDBOX_CONTENT_INVALID", "Invalid runtime copy chunk."
                    )
                count = W.DWORD()
                if not self.kernel.WriteFile(
                    handle, chunk, len(chunk), ctypes.byref(count), None
                ) or count.value != len(chunk):
                    raise _error("WriteFile(private content)")
            if not self.kernel.FlushFileBuffers(handle):
                raise _error("FlushFileBuffers(private content)")
        finally:
            self.kernel.CloseHandle(handle)


@lru_cache(maxsize = 1)
def native_files():
    return NativeFiles()


class PathLease:
    """Pin directories before their children; release all handles on failure/exit."""

    def __init__(self):
        self.api = native_files()
        self.handles = {}

    def directory(self, path):
        spelled = Path(path)
        if spelled in self.handles:
            # Reuse only this lease's still-open pin, not a cached path verdict.
            # Recheck the handle's current type and normalized native path.
            handle = self.handles[spelled]
            self.api.require_path(handle, spelled, directory = True)
            return handle
        path = checked_path(path)
        for item in (*reversed(path.parents), path):
            if item not in self.handles:
                self.handles[item] = self.api.open(item, directory = True)
        return self.handles[path]

    def file(
        self,
        path,
        *,
        exclusive = False,
    ):
        spelled = Path(path)
        self.directory(spelled.parent)
        path = checked_path(spelled)
        if path in self.handles:
            raise WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", "Duplicate file lease.")
        handle = self.api.open(path, exclusive = exclusive)
        self.handles[path] = handle
        return handle

    def close(self):
        for path, handle in reversed(tuple(self.handles.items())):
            if not self.api.kernel.CloseHandle(handle):
                raise _error("CloseHandle(runtime pin)")
            del self.handles[path]

    def duplicate_from(self, process_handle, entries, check_deadline):
        """Adopt pins validated by the fixed worker, never model-supplied handles.

        The broker validates the invocation and complete inventory first. No
        filesystem reopen here: duplicates preserve the worker's sharing locks.
        Partial duplicates stay in this owner if any later operation fails.
        """
        if self.handles or type(entries) is not tuple or not 0 < len(entries) <= 16384:
            raise WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", "Invalid pin handoff.")
        paths, handles = set(), set()
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", "Invalid pin entry.")
            path, handle = entry
            if (
                type(path) is not str
                or not path
                or len(path) > 32768
                or "\0" in path
                or not Path(path).is_absolute()
                or type(handle) is not int
                or not 0 < handle < 2**32 - 16
                or ".." in Path(path).parts
                or os.path.normcase(os.path.normpath(path)) in paths
                or handle in handles
            ):
                raise WindowsRuntimeError("WINDOWS_SANDBOX_CONTENT_INVALID", "Invalid pin entry.")
            paths.add(os.path.normcase(os.path.normpath(path)))
            handles.add(handle)
        duplicate = self.api.kernel.DuplicateHandle
        duplicate.argtypes = [
            W.HANDLE,
            W.HANDLE,
            W.HANDLE,
            ctypes.POINTER(W.HANDLE),
            W.DWORD,
            W.BOOL,
            W.DWORD,
        ]
        duplicate.restype = W.BOOL
        current = self.api.kernel.GetCurrentProcess()
        for path, handle in entries:
            check_deadline()
            copied = W.HANDLE()
            if not duplicate(process_handle, handle, current, ctypes.byref(copied), 0, False, 2):
                raise _error("DuplicateHandle(runtime pin)")
            self.handles[Path(path)] = copied.value
        check_deadline()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
