# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Handle-verified Windows traversal and atomic project-file mutation.

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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from .common import AgentWorkspaceError


_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_READ_ATTRIBUTES = 0x00000080
_DELETE = 0x00010000
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_ID_INFO_CLASS = 18
_FILE_DISPOSITION_INFO_CLASS = 4
_MOVEFILE_WRITE_THROUGH = 0x00000008
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_MAX_READ_CHUNK = 64 * 1024
_MAX_MUTATION_BYTES = 16 * 1024 * 1024
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

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("delete_file", wintypes.BOOL)]

        self.ByHandleFileInformation = ByHandleFileInformation
        self.FileIdInfo = FileIdInfo
        self.FileDispositionInfo = FileDispositionInfo
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
        self.kernel32.WriteFile.argtypes = [
            handle,
            ctypes.c_void_p,
            dword,
            ctypes.POINTER(dword),
            ctypes.c_void_p,
        ]
        self.kernel32.WriteFile.restype = bool_type
        self.kernel32.FlushFileBuffers.argtypes = [handle]
        self.kernel32.FlushFileBuffers.restype = bool_type
        self.kernel32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
        self.kernel32.CreateDirectoryW.restype = bool_type
        self.kernel32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, dword]
        self.kernel32.MoveFileExW.restype = bool_type
        self.kernel32.ReplaceFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            dword,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.kernel32.ReplaceFileW.restype = bool_type
        self.kernel32.SetFileInformationByHandle.argtypes = [
            handle,
            ctypes.c_int,
            ctypes.c_void_p,
            dword,
        ]
        self.kernel32.SetFileInformationByHandle.restype = bool_type
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


def _open_handle(
    path: str,
    *,
    read: bool = False,
    write: bool = False,
    delete: bool = False,
    share_write: bool = True,
    share_delete: bool = True,
    creation: int = _OPEN_EXISTING,
) -> int:
    api = _api()
    access = _FILE_READ_ATTRIBUTES
    if read:
        access |= _GENERIC_READ
    if write:
        access |= _GENERIC_WRITE
    if delete:
        access |= _DELETE
    share = _FILE_SHARE_READ
    if share_write:
        share |= _FILE_SHARE_WRITE
    if share_delete:
        share |= _FILE_SHARE_DELETE
    handle = api.kernel32.CreateFileW(
        _extended_path(path),
        access,
        share,
        None,
        creation,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle in {None, -1, api.invalid_handle}:
        raise _win32_error("Windows path could not be opened")
    return int(handle)


def _close_handle(handle: int) -> None:
    if handle:
        _api().kernel32.CloseHandle(handle)


def _read_handle(handle: int, limit: int) -> bytes:
    if limit < 0:
        raise ValueError("Windows read limit cannot be negative.")
    output = bytearray()
    while len(output) < limit:
        requested = min(_MAX_READ_CHUNK, limit - len(output))
        buffer = ctypes.create_string_buffer(requested)
        count = _api().wintypes.DWORD(0)
        if not _api().kernel32.ReadFile(
            handle,
            buffer,
            requested,
            ctypes.byref(count),
            None,
        ):
            raise _win32_error("Windows could not read an opened file")
        if count.value == 0:
            break
        output.extend(buffer.raw[: count.value])
    return bytes(output)


def _write_handle(handle: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        chunk = payload[offset : offset + _MAX_READ_CHUNK]
        buffer = ctypes.create_string_buffer(chunk)
        count = _api().wintypes.DWORD(0)
        if not _api().kernel32.WriteFile(
            handle,
            buffer,
            len(chunk),
            ctypes.byref(count),
            None,
        ):
            raise _win32_error("Windows could not write a temporary edit file")
        if count.value <= 0:
            raise OSError("Windows reported a short write to a temporary edit file.")
        offset += int(count.value)
    if not _api().kernel32.FlushFileBuffers(handle):
        raise _win32_error("Windows could not flush a temporary edit file")


def _mark_handle_for_deletion(handle: int) -> None:
    api = _api()
    disposition = api.FileDispositionInfo(delete_file = True)
    if not api.kernel32.SetFileInformationByHandle(
        handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise _win32_error("Windows could not clean up a temporary edit file")


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


class _WindowsMutationOps(Protocol):
    """Narrow Win32 seam used by the mutation primitive and its unit tests."""

    def open_existing(
        self,
        path: str,
        *,
        read: bool = False,
        delete: bool = False,
        share_write: bool = True,
        share_delete: bool = False,
    ) -> int: ...

    def create_temp(self, path: str) -> int: ...

    def close(self, handle: int) -> None: ...

    def info(self, handle: int) -> _HandleInfo: ...

    def read(self, handle: int, limit: int) -> bytes: ...

    def write_and_flush(self, handle: int, payload: bytes) -> None: ...

    def create_directory(self, path: str) -> None: ...

    def move_new(self, source: str, target: str) -> bool: ...

    def replace(self, source: str, target: str) -> None: ...

    def mark_delete(self, handle: int) -> None: ...


class _NativeWindowsMutationOps:
    def open_existing(
        self,
        path: str,
        *,
        read: bool = False,
        delete: bool = False,
        share_write: bool = True,
        share_delete: bool = False,
    ) -> int:
        return _open_handle(
            path,
            read = read,
            delete = delete,
            share_write = share_write,
            share_delete = share_delete,
        )

    def create_temp(self, path: str) -> int:
        return _open_handle(
            path,
            write = True,
            delete = True,
            share_write = False,
            share_delete = False,
            creation = _CREATE_NEW,
        )

    def close(self, handle: int) -> None:
        _close_handle(handle)

    def info(self, handle: int) -> _HandleInfo:
        return _handle_info(handle)

    def read(self, handle: int, limit: int) -> bytes:
        return _read_handle(handle, limit)

    def write_and_flush(self, handle: int, payload: bytes) -> None:
        _write_handle(handle, payload)

    def create_directory(self, path: str) -> None:
        if _api().kernel32.CreateDirectoryW(_extended_path(path), None):
            return
        code = ctypes.get_last_error()
        if code not in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
            raise _win32_error("Windows could not create a project directory")

    def move_new(self, source: str, target: str) -> bool:
        if _api().kernel32.MoveFileExW(
            _extended_path(source),
            _extended_path(target),
            _MOVEFILE_WRITE_THROUGH,
        ):
            return True
        code = ctypes.get_last_error()
        if code in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
            return False
        raise _win32_error("Windows could not atomically create the project file")

    def replace(self, source: str, target: str) -> None:
        if not _api().kernel32.ReplaceFileW(
            _extended_path(target),
            _extended_path(source),
            None,
            0,
            None,
            None,
        ):
            raise _win32_error("Windows could not atomically replace the project file")

    def mark_delete(self, handle: int) -> None:
        _mark_handle_for_deletion(handle)


@dataclass(frozen = True)
class _MutationGuard:
    path: str
    handle: int
    identity_options: tuple[tuple[int, int], ...]


_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def _validated_mutation_part(value: str) -> str:
    checked = _validated_part(value)
    if checked.endswith((" ", ".")):
        raise WindowsTraversalRejected(
            "Windows edit paths cannot end a component with a space or period."
        )
    device_name = checked.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED_COMPONENTS:
        raise WindowsTraversalRejected("Windows device names are not valid edit paths.")
    return checked


class WindowsVerifiedMutation:
    """Atomic, no-follow mutation of one file below an identity-bound root.

    This mirrors the ``read`` / ``create`` / ``replace`` surface used by the
    POSIX project editor. Directory handles deny delete sharing for the entire
    operation. File bytes are written only to a new sibling temporary file and
    committed with a Win32 rename or replace operation.
    """

    def __init__(
        self,
        root: Path | str,
        target: Path | str,
        expected_root_identity: Optional[tuple[int, int]] = None,
        *,
        max_bytes: int = _MAX_MUTATION_BYTES,
        _ops: Optional[_WindowsMutationOps] = None,
    ) -> None:
        if max_bytes < 0:
            raise ValueError("Windows mutation limit cannot be negative.")
        requested_root = _validated_local_path(str(root))
        raw_target = normalize_windows_path(str(target))
        if ntpath.isabs(raw_target):
            requested_target = _validated_local_path(raw_target)
        else:
            requested_target = _validated_local_path(ntpath.join(requested_root, raw_target))
        if not windows_path_is_within(requested_target, requested_root):
            raise WindowsTraversalRejected("The edit path escapes the project root.")
        relative = ntpath.relpath(requested_target, requested_root)
        raw_parts = tuple(part for part in relative.split("\\") if part)
        if not raw_parts or raw_parts == (".",):
            raise WindowsTraversalRejected("The project root cannot be edited as a file.")
        self.parts = tuple(_validated_mutation_part(part) for part in raw_parts)
        self.max_bytes = int(max_bytes)
        self._ops = _ops or _NativeWindowsMutationOps()
        self._closed = False
        handle = self._ops.open_existing(
            requested_root,
            share_write = True,
            share_delete = False,
        )
        try:
            info = self._ops.info(handle)
            _assert_expected_handle(
                info,
                requested_root,
                requested_root,
                expected_identity = expected_root_identity,
                directory = True,
            )
        except Exception:
            self._ops.close(handle)
            raise
        self.path = info.final_path
        self.identity_options = info.identity_options
        self._root = _MutationGuard(self.path, handle, info.identity_options)

    @classmethod
    def open(
        cls,
        root: Path | str,
        target: Path | str,
        expected_root_identity: Optional[tuple[int, int]] = None,
        *,
        max_bytes: int = _MAX_MUTATION_BYTES,
        _ops: Optional[_WindowsMutationOps] = None,
    ) -> "WindowsVerifiedMutation":
        if _ops is None and not windows_secure_traversal_supported():
            raise AgentWorkspaceError(
                "Secure Windows workspace mutation is unavailable on this host."
            )
        try:
            return cls(
                root,
                target,
                expected_root_identity,
                max_bytes = max_bytes,
                _ops = _ops,
            )
        except AgentWorkspaceError:
            raise
        except OSError as exc:
            raise AgentWorkspaceError(
                "The project root is unavailable, changed identity, or unsafe to edit."
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ops.close(self._root.handle)

    def __enter__(self) -> "WindowsVerifiedMutation":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    @property
    def target_path(self) -> str:
        return ntpath.join(self.path, *self.parts)

    def _assert_open(self) -> None:
        if self._closed:
            raise WindowsTraversalRejected("Windows project mutation is closed.")

    def _assert_guard(
        self,
        guard: _MutationGuard,
        *,
        root: bool = False,
    ) -> None:
        info = self._ops.info(guard.handle)
        _assert_expected_handle(
            info,
            guard.path,
            self.path,
            directory = True,
        )
        if not any(identity in guard.identity_options for identity in info.identity_options):
            raise WindowsTraversalRejected("A project directory changed during the edit.")
        reopened = self._ops.open_existing(
            guard.path,
            share_write = True,
            share_delete = False,
        )
        try:
            current = self._ops.info(reopened)
            _assert_expected_handle(
                current,
                guard.path,
                self.path,
                directory = True,
            )
            if not any(identity in guard.identity_options for identity in current.identity_options):
                message = (
                    "Project root identity changed."
                    if root
                    else "A project directory changed during the edit."
                )
                raise WindowsTraversalRejected(message)
        finally:
            self._ops.close(reopened)

    def _recheck_guards(self, guards: Sequence[_MutationGuard]) -> None:
        self._assert_guard(self._root, root = True)
        for guard in guards:
            self._assert_guard(guard)

    def _open_parent(self, *, create: bool) -> list[_MutationGuard]:
        self._assert_open()
        guards: list[_MutationGuard] = []
        current = self.path
        try:
            for component in self.parts[:-1]:
                expected = ntpath.join(current, component)
                try:
                    handle = self._ops.open_existing(
                        expected,
                        share_write = True,
                        share_delete = False,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    self._ops.create_directory(expected)
                    handle = self._ops.open_existing(
                        expected,
                        share_write = True,
                        share_delete = False,
                    )
                try:
                    info = self._ops.info(handle)
                    _assert_expected_handle(
                        info,
                        expected,
                        self.path,
                        directory = True,
                    )
                except Exception:
                    self._ops.close(handle)
                    raise
                guard = _MutationGuard(info.final_path, handle, info.identity_options)
                guards.append(guard)
                current = info.final_path
            self._recheck_guards(guards)
            return guards
        except Exception:
            self._close_guards(guards)
            raise

    def _close_guards(self, guards: Sequence[_MutationGuard]) -> None:
        for guard in reversed(guards):
            self._ops.close(guard.handle)

    def _open_target(self) -> tuple[int, _HandleInfo]:
        handle = self._ops.open_existing(
            self.target_path,
            read = True,
            share_write = False,
            share_delete = False,
        )
        try:
            info = self._ops.info(handle)
            _assert_expected_handle(
                info,
                self.target_path,
                self.path,
                directory = False,
            )
            return handle, info
        except Exception:
            self._ops.close(handle)
            raise

    @staticmethod
    def _identity_matches(info: _HandleInfo, identity: tuple[int, int]) -> bool:
        try:
            expected = (int(identity[0]), int(identity[1]))
        except (IndexError, TypeError, ValueError):
            return False
        return expected in info.identity_options

    def _validate_payload(self, payload: bytes) -> bytes:
        if not isinstance(payload, bytes):
            raise TypeError("Windows edit payloads must be bytes.")
        if len(payload) > self.max_bytes:
            raise OverflowError("Windows edit payload exceeds the configured limit.")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Windows edit payload is not UTF-8 text.") from exc
        if "\x00" in text:
            raise ValueError("Windows edit payload contains a NUL character.")
        return payload

    def read(self, limit: int) -> tuple[bytes, int, tuple[int, int]]:
        if limit < 0 or limit > self.max_bytes:
            raise ValueError("Windows read limit is outside the configured mutation bound.")
        guards = self._open_parent(create = False)
        try:
            handle, before = self._open_target()
            try:
                if before.size > limit:
                    raise OverflowError("Windows edit target exceeds the read limit.")
                raw = self._ops.read(handle, limit + 1)
                after = self._ops.info(handle)
                if (
                    before.identity_options != after.identity_options
                    or before.size != after.size
                    or before.modified_ns != after.modified_ns
                    or windows_path_key(before.final_path) != windows_path_key(after.final_path)
                    or len(raw) > limit
                ):
                    raise WindowsTraversalRejected("File changed while it was being read.")
                self._recheck_guards(guards)
                return raw, before.attributes, before.identity_options[0]
            finally:
                self._ops.close(handle)
        finally:
            self._close_guards(guards)

    def _prepare_temp(self, payload: bytes, parent_path: str) -> tuple[str, tuple[int, int]]:
        for _attempt in range(8):
            temp_path = ntpath.join(parent_path, f".unsloth_edit_{uuid.uuid4().hex}")
            try:
                handle = self._ops.create_temp(temp_path)
            except FileExistsError:
                continue
            try:
                before = self._ops.info(handle)
                _assert_expected_handle(
                    before,
                    temp_path,
                    self.path,
                    directory = False,
                )
                self._ops.write_and_flush(handle, payload)
                after = self._ops.info(handle)
                if (
                    before.identity_options != after.identity_options
                    or after.size != len(payload)
                    or windows_path_key(after.final_path) != windows_path_key(temp_path)
                ):
                    raise WindowsTraversalRejected(
                        "Temporary edit file changed while it was being written."
                    )
                identity = after.identity_options[0]
            except Exception:
                try:
                    self._ops.mark_delete(handle)
                finally:
                    self._ops.close(handle)
                raise
            self._ops.close(handle)
            return temp_path, identity
        raise FileExistsError("Windows could not allocate a unique temporary edit file.")

    def _cleanup_temp(self, temp_path: str, identity: tuple[int, int]) -> None:
        try:
            handle = self._ops.open_existing(
                temp_path,
                delete = True,
                share_write = False,
                share_delete = False,
            )
        except FileNotFoundError:
            return
        try:
            info = self._ops.info(handle)
            _assert_expected_handle(
                info,
                temp_path,
                self.path,
                directory = False,
            )
            if self._identity_matches(info, identity):
                self._ops.mark_delete(handle)
        finally:
            self._ops.close(handle)

    def _current_matches(self, expect: bytes, identity: tuple[int, int]) -> bool:
        try:
            handle, info = self._open_target()
        except WindowsTraversalRejected as exc:
            if exc.reparse:
                raise
            return False
        except (FileNotFoundError, NotADirectoryError):
            return False
        try:
            if not self._identity_matches(info, identity) or info.size != len(expect):
                return False
            current = self._ops.read(handle, len(expect) + 1)
            after = self._ops.info(handle)
            return (
                current == expect
                and self._identity_matches(after, identity)
                and after.size == len(expect)
                and after.modified_ns == info.modified_ns
                and windows_path_key(after.final_path) == windows_path_key(info.final_path)
            )
        finally:
            self._ops.close(handle)

    def _committed_matches(self, payload: bytes, identity: tuple[int, int]) -> bool:
        """Verify that the committed name still identifies the prepared temp file."""
        return self._current_matches(payload, identity)

    def _target_exists_safely(self) -> bool:
        try:
            handle = self._ops.open_existing(
                self.target_path,
                share_write = False,
                share_delete = False,
            )
        except FileNotFoundError:
            return False
        try:
            info = self._ops.info(handle)
            _assert_expected_handle(info, self.target_path, self.path)
            return True
        finally:
            self._ops.close(handle)

    def create(
        self,
        payload: bytes,
        mode: int = 0o666,
    ) -> Optional[str]:
        del mode
        data = self._validate_payload(payload)
        guards = self._open_parent(create = True)
        parent_path = guards[-1].path if guards else self.path
        temp_path = ""
        temp_identity: Optional[tuple[int, int]] = None
        committed = False
        try:
            if self._target_exists_safely():
                return "exists"
            temp_path, temp_identity = self._prepare_temp(data, parent_path)
            self._recheck_guards(guards)
            committed = self._ops.move_new(temp_path, self.target_path)
            if not committed:
                return "exists"
            if not self._committed_matches(data, temp_identity):
                raise WindowsTraversalRejected(
                    "The created project file changed before the commit could be verified."
                )
            return None
        finally:
            if temp_path and not committed and temp_identity is not None:
                self._cleanup_temp(temp_path, temp_identity)
            self._close_guards(guards)

    def replace(
        self, payload: bytes, *, expect: bytes, mode: int, identity: tuple[int, int]
    ) -> Optional[str]:
        del mode
        data = self._validate_payload(payload)
        if not isinstance(expect, bytes) or len(expect) > self.max_bytes:
            raise ValueError("Windows expected edit content is outside the configured bound.")
        guards = self._open_parent(create = False)
        parent_path = guards[-1].path if guards else self.path
        temp_path = ""
        temp_identity: Optional[tuple[int, int]] = None
        committed = False
        try:
            if not self._current_matches(expect, identity):
                return "changed"
            temp_path, temp_identity = self._prepare_temp(data, parent_path)
            self._recheck_guards(guards)
            if not self._current_matches(expect, identity):
                return "changed"
            self._ops.replace(temp_path, self.target_path)
            committed = True
            if not self._committed_matches(data, temp_identity):
                raise WindowsTraversalRejected(
                    "The replaced project file changed before the commit could be verified."
                )
            return None
        finally:
            if temp_path and not committed and temp_identity is not None:
                self._cleanup_temp(temp_path, temp_identity)
            self._close_guards(guards)


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
    "WindowsVerifiedMutation",
    "WindowsVerifiedRoot",
]
