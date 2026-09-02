# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Handle-verified Windows traversal for project metadata and instructions.

Windows does not expose POSIX openat directory descriptors through Python. This
module opens every traversed object with CreateFileW, refuses reparse points,
and checks the final handle path against the expected path below an already
identity-bound root. Path strings are used only to request a handle. Content or
metadata is returned only after the handle proves containment and identity.
"""

from __future__ import annotations

import ctypes
import ntpath
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .common import AgentWorkspaceError


_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_READ_ATTRIBUTES = 0x00000080
_GENERIC_READ = 0x80000000
_FILE_ID_INFO_CLASS = 18
_MAX_READ_CHUNK = 64 * 1024
_WINDOWS_EPOCH_OFFSET_100NS = 116_444_736_000_000_000


class WindowsTraversalRejected(OSError):
    """A Windows path could not be proven to be the expected local object."""

    def __init__(
        self,
        reason: str,
        *,
        reparse: bool = False,
    ):
        super().__init__(reason)
        self.reparse = reparse


@dataclass(frozen = True)
class WindowsEntry:
    name: str
    is_directory: bool
    is_file: bool
    is_reparse: bool = False


@dataclass(frozen = True)
class WindowsFileData:
    raw: bytes
    truncated: bool
    size: int
    modified_ns: int


@dataclass(frozen = True)
class _HandleInfo:
    attributes: int
    identity_options: tuple[tuple[int, int], ...]
    size: int
    modified_ns: int
    final_path: str

    @property
    def is_directory(self) -> bool:
        return bool(self.attributes & _FILE_ATTRIBUTE_DIRECTORY)

    @property
    def is_reparse(self) -> bool:
        return bool(self.attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


class _Win32Api:
    def __init__(self) -> None:
        from ctypes import wintypes

        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)

        class FileTime(ctypes.Structure):
            _fields_ = [
                ("low", wintypes.DWORD),
                ("high", wintypes.DWORD),
            ]

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("attributes", wintypes.DWORD),
                ("creation_time", FileTime),
                ("access_time", FileTime),
                ("write_time", FileTime),
                ("volume_serial", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("file_index_high", wintypes.DWORD),
                ("file_index_low", wintypes.DWORD),
            ]

        class FileId128(ctypes.Structure):
            _fields_ = [("identifier", ctypes.c_ubyte * 16)]

        class FileIdInfo(ctypes.Structure):
            _fields_ = [
                ("volume_serial", ctypes.c_ulonglong),
                ("file_id", FileId128),
            ]

        self.ByHandleFileInformation = ByHandleFileInformation
        self.FileIdInfo = FileIdInfo
        handle = wintypes.HANDLE
        dword = wintypes.DWORD
        bool_type = wintypes.BOOL
        self.kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            dword,
            dword,
            ctypes.c_void_p,
            dword,
            dword,
            handle,
        ]
        self.kernel32.CreateFileW.restype = handle
        self.kernel32.CloseHandle.argtypes = [handle]
        self.kernel32.CloseHandle.restype = bool_type
        self.kernel32.GetFileInformationByHandle.argtypes = [
            handle,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self.kernel32.GetFileInformationByHandle.restype = bool_type
        self.kernel32.GetFileInformationByHandleEx.argtypes = [
            handle,
            ctypes.c_int,
            ctypes.c_void_p,
            dword,
        ]
        self.kernel32.GetFileInformationByHandleEx.restype = bool_type
        self.kernel32.GetFinalPathNameByHandleW.argtypes = [
            handle,
            wintypes.LPWSTR,
            dword,
            dword,
        ]
        self.kernel32.GetFinalPathNameByHandleW.restype = dword
        self.kernel32.ReadFile.argtypes = [
            handle,
            ctypes.c_void_p,
            dword,
            ctypes.POINTER(dword),
            ctypes.c_void_p,
        ]
        self.kernel32.ReadFile.restype = bool_type
        self.invalid_handle = ctypes.c_void_p(-1).value


_API: Optional[_Win32Api] = None


def _api() -> _Win32Api:
    global _API
    if _API is None:
        _API = _Win32Api()
    return _API


def windows_secure_traversal_supported() -> bool:
    if os.name != "nt":
        return False
    try:
        _api()
    except (AttributeError, ImportError, OSError):
        return False
    return True


def normalize_windows_path(path: str | os.PathLike[str]) -> str:
    """Normalize a DOS path without depending on the host's os.path module."""
    value = os.fspath(path).replace("/", "\\")
    folded = value.casefold()
    if folded.startswith("\\\\?\\unc\\"):
        value = "\\\\" + value[8:]
    elif folded.startswith("\\\\?\\"):
        value = value[4:]
    elif folded.startswith("\\??\\"):
        value = value[4:]
    return ntpath.normpath(value)


def windows_path_key(path: str | os.PathLike[str]) -> str:
    return ntpath.normcase(normalize_windows_path(path)).rstrip("\\")


def windows_path_is_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    candidate = windows_path_key(path)
    parent = windows_path_key(root)
    try:
        return ntpath.commonpath((candidate, parent)) == parent
    except ValueError:
        return False


def _validated_local_path(path: str | os.PathLike[str]) -> str:
    value = normalize_windows_path(path)
    if not value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise WindowsTraversalRejected("Windows path contains invalid characters.")
    folded = value.casefold()
    if folded.startswith(("\\\\", "\\.\\", "\\??\\")):
        raise WindowsTraversalRejected("Network and device paths are not supported.")
    drive, tail = ntpath.splitdrive(value)
    if len(drive) != 2 or drive[1] != ":" or not tail.startswith("\\"):
        raise WindowsTraversalRejected("Windows project paths must use a local drive.")
    if ":" in tail:
        raise WindowsTraversalRejected("Alternate data streams are not supported.")
    return value


def _extended_path(path: str | os.PathLike[str]) -> str:
    return "\\\\?\\" + _validated_local_path(path)


def _validated_part(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or any(ord(char) < 32 for char in value)
        or any(char in value for char in ("\\", "/", ":"))
    ):
        raise WindowsTraversalRejected("Workspace path is invalid.")
    return value


def _win32_error(message: str) -> OSError:
    code = ctypes.get_last_error()
    factory = getattr(ctypes, "WinError", None)
    if factory is not None:
        error = factory(code)
        error.args = (*error.args, message)
        return error
    return OSError(code, f"{message} (WinError {code})")


def _open_handle(path: str, *, read: bool = False) -> int:
    api = _api()
    access = _FILE_READ_ATTRIBUTES | (_GENERIC_READ if read else 0)
    handle = api.kernel32.CreateFileW(
        _extended_path(path),
        access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle in {None, -1, api.invalid_handle}:
        raise _win32_error("Windows path could not be opened")
    return int(handle)


def _close_handle(handle: int) -> None:
    if handle:
        _api().kernel32.CloseHandle(handle)


def _final_path(handle: int) -> str:
    api = _api()
    size = 1024
    while size <= 65_536:
        buffer = ctypes.create_unicode_buffer(size)
        length = int(api.kernel32.GetFinalPathNameByHandleW(handle, buffer, size, 0))
        if length == 0:
            raise _win32_error("Windows could not resolve an opened path")
        if length < size:
            return _validated_local_path(buffer.value)
        size = length + 1
    raise WindowsTraversalRejected("Windows path exceeds the supported length.")


def _handle_info(handle: int) -> _HandleInfo:
    api = _api()
    legacy = api.ByHandleFileInformation()
    if not api.kernel32.GetFileInformationByHandle(handle, ctypes.byref(legacy)):
        raise _win32_error("Windows could not inspect an opened path")
    legacy_identity = (
        int(legacy.volume_serial),
        (int(legacy.file_index_high) << 32) | int(legacy.file_index_low),
    )
    identities = [legacy_identity]
    extended = api.FileIdInfo()
    if api.kernel32.GetFileInformationByHandleEx(
        handle,
        _FILE_ID_INFO_CLASS,
        ctypes.byref(extended),
        ctypes.sizeof(extended),
    ):
        file_id = int.from_bytes(bytes(extended.file_id.identifier), "little")
        extended_identity = (int(extended.volume_serial), file_id)
        if extended_identity != legacy_identity:
            identities.append(extended_identity)
    write_100ns = (int(legacy.write_time.high) << 32) | int(legacy.write_time.low)
    modified_ns = max(0, write_100ns - _WINDOWS_EPOCH_OFFSET_100NS) * 100
    return _HandleInfo(
        attributes = int(legacy.attributes),
        identity_options = tuple(identities),
        size = (int(legacy.size_high) << 32) | int(legacy.size_low),
        modified_ns = modified_ns,
        final_path = _final_path(handle),
    )


def _assert_expected_handle(
    info: _HandleInfo,
    expected_path: str,
    root_path: str,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    directory: Optional[bool] = None,
) -> None:
    if info.is_reparse:
        raise WindowsTraversalRejected("Windows reparse points are not supported.", reparse = True)
    if not windows_path_is_within(info.final_path, root_path):
        raise WindowsTraversalRejected("Workspace path escaped the project root.")
    if windows_path_key(info.final_path) != windows_path_key(expected_path):
        raise WindowsTraversalRejected("Workspace path changed while it was being inspected.")
    if directory is True and not info.is_directory:
        raise NotADirectoryError(expected_path)
    if directory is False and info.is_directory:
        raise IsADirectoryError(expected_path)
    if expected_identity is not None:
        try:
            expected = (int(expected_identity[0]), int(expected_identity[1]))
        except (IndexError, TypeError, ValueError) as exc:
            raise AgentWorkspaceError("Project root identity is invalid.") from exc
        if expected not in info.identity_options:
            raise WindowsTraversalRejected("Project root identity changed.")


class WindowsVerifiedRoot:
    """An opened local-drive root used to verify every subsequent handle."""

    def __init__(
        self,
        path: Path | str,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> None:
        requested = _validated_local_path(str(path))
        handle = _open_handle(requested)
        try:
            info = _handle_info(handle)
            _assert_expected_handle(
                info,
                requested,
                requested,
                expected_identity = expected_identity,
                directory = True,
            )
        except Exception:
            _close_handle(handle)
            raise
        self.path = info.final_path
        self.identity_options = info.identity_options
        self._handle = handle
        self._closed = False

    @classmethod
    def open(
        cls,
        path: Path | str,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> "WindowsVerifiedRoot":
        if not windows_secure_traversal_supported():
            raise AgentWorkspaceError(
                "Secure Windows workspace traversal is unavailable on this host."
            )
        try:
            return cls(path, expected_identity)
        except AgentWorkspaceError:
            raise
        except OSError as exc:
            raise AgentWorkspaceError(
                "The project root is unavailable or changed identity."
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_handle(self._handle)

    def __enter__(self) -> "WindowsVerifiedRoot":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    def _expected_path(self, parts: Sequence[str]) -> str:
        checked = [_validated_part(str(part)) for part in parts]
        return ntpath.join(self.path, *checked) if checked else self.path

    def _open_verified(
        self,
        parts: Sequence[str],
        *,
        read: bool = False,
        directory: Optional[bool] = None,
    ) -> tuple[int, _HandleInfo]:
        if self._closed:
            raise WindowsTraversalRejected("Windows project root is closed.")
        expected = self._expected_path(parts)
        handle = _open_handle(expected, read = read)
        try:
            info = _handle_info(handle)
            _assert_expected_handle(
                info,
                expected,
                self.path,
                directory = directory,
            )
            return handle, info
        except Exception:
            _close_handle(handle)
            raise

    def recheck(self) -> None:
        handle, info = self._open_verified((), directory = True)
        try:
            if not any(identity in self.identity_options for identity in info.identity_options):
                raise WindowsTraversalRejected("Project root identity changed.")
            original = _handle_info(self._handle)
            if not any(identity in self.identity_options for identity in original.identity_options):
                raise WindowsTraversalRejected("Project root identity changed.")
        finally:
            _close_handle(handle)

    def path_kind(self, parts: Sequence[str]) -> str:
        handle, info = self._open_verified(parts)
        try:
            return "directory" if info.is_directory else "file"
        finally:
            _close_handle(handle)

    def read_file(self, parts: Sequence[str], limit: int) -> WindowsFileData:
        if limit < 0:
            raise ValueError("Windows read limit cannot be negative.")
        handle, before = self._open_verified(parts, read = True, directory = False)
        output = bytearray()
        try:
            while len(output) <= limit:
                requested = min(_MAX_READ_CHUNK, limit + 1 - len(output))
                if requested <= 0:
                    break
                buffer = ctypes.create_string_buffer(requested)
                count = _api().wintypes.DWORD(0)
                ok = _api().kernel32.ReadFile(
                    handle,
                    buffer,
                    requested,
                    ctypes.byref(count),
                    None,
                )
                if not ok:
                    raise _win32_error("Windows could not read an opened file")
                if count.value == 0:
                    break
                output.extend(buffer.raw[: count.value])
            after = _handle_info(handle)
            if (
                before.identity_options != after.identity_options
                or before.size != after.size
                or before.modified_ns != after.modified_ns
                or windows_path_key(before.final_path) != windows_path_key(after.final_path)
            ):
                raise WindowsTraversalRejected("File changed while it was being read.")
            return WindowsFileData(
                raw = bytes(output[:limit]),
                truncated = len(output) > limit,
                size = before.size,
                modified_ns = before.modified_ns,
            )
        finally:
            _close_handle(handle)

    def list_directory(self, parts: Sequence[str]) -> list[WindowsEntry]:
        directory, before = self._open_verified(parts, directory = True)
        try:
            expected = self._expected_path(parts)
            try:
                names = sorted(entry.name for entry in os.scandir(_extended_path(expected)))
            except OSError as exc:
                raise WindowsTraversalRejected(
                    "Windows directory could not be enumerated."
                ) from exc
            result: list[WindowsEntry] = []
            for name in names:
                try:
                    handle, info = self._open_verified((*parts, name))
                except WindowsTraversalRejected as exc:
                    if exc.reparse:
                        result.append(
                            WindowsEntry(
                                name = name,
                                is_directory = False,
                                is_file = False,
                                is_reparse = True,
                            )
                        )
                    continue
                except OSError:
                    continue
                try:
                    result.append(
                        WindowsEntry(
                            name = name,
                            is_directory = info.is_directory,
                            is_file = not info.is_directory,
                        )
                    )
                finally:
                    _close_handle(handle)
            after_handle, after = self._open_verified(parts, directory = True)
            _close_handle(after_handle)
            if before.identity_options != after.identity_options:
                raise WindowsTraversalRejected("Directory changed while it was being enumerated.")
            return result
        finally:
            _close_handle(directory)


__all__ = [
    "normalize_windows_path",
    "windows_path_is_within",
    "windows_path_key",
    "windows_secure_traversal_supported",
    "WindowsEntry",
    "WindowsFileData",
    "WindowsTraversalRejected",
    "WindowsVerifiedRoot",
]
