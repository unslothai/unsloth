# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Identity-bound project file mutation primitives.

The public editor passes a persisted project workspace and one target path to
``ProjectFileMutation``. POSIX traversal stays relative to opened directory
descriptors and refuses symbolic links and mount crossings. Windows traversal
opens every component with ``FILE_FLAG_OPEN_REPARSE_POINT`` and verifies its
final path and file identity before content is used or a mutation is committed.

This module deliberately has no chat or tool-schema policy. It is the narrow
filesystem boundary shared by callers that already decided an edit is allowed.
"""

from __future__ import annotations

import contextlib
import ctypes
import ntpath
import os
import stat
import struct
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from .common import AgentWorkspaceError, ProjectWorkspace


_MAX_MUTATION_BYTES = 16 * 1024 * 1024
_READ_CHUNK = 64 * 1024


_MUTATION_CONDITION = threading.Condition()
_ACTIVE_MUTATION_ROOTS: set[tuple[int, int]] = set()


def acquire_workspace_mutation_slot(identity: tuple[int, int], cancel_event = None) -> bool:
    """Serialize processes and edits that can mutate one project root."""
    key = (int(identity[0]), int(identity[1]))
    with _MUTATION_CONDITION:
        while key in _ACTIVE_MUTATION_ROOTS:
            if cancel_event is not None and cancel_event.is_set():
                return False
            _MUTATION_CONDITION.wait(timeout = 0.05)
        if cancel_event is not None and cancel_event.is_set():
            return False
        _ACTIVE_MUTATION_ROOTS.add(key)
        return True


def release_workspace_mutation_slot(identity: tuple[int, int]) -> None:
    key = (int(identity[0]), int(identity[1]))
    with _MUTATION_CONDITION:
        _ACTIVE_MUTATION_ROOTS.discard(key)
        _MUTATION_CONDITION.notify_all()


def _expected_identity(workspace: ProjectWorkspace) -> tuple[int, int]:
    try:
        return int(workspace.device_id), int(workspace.file_id)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project root identity is missing or invalid.") from exc


def _require_posix_mutation_primitives() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    required_dir_fd = (os.open, os.stat, os.mkdir, os.link)
    if (
        os.name != "posix"
        or any(not hasattr(os, name) for name in required_flags)
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.stat not in os.supports_follow_symlinks
        or os.link not in os.supports_follow_symlinks
    ):
        raise AgentWorkspaceError(
            "Secure descriptor-relative project mutation is unavailable on this host."
        )


def _posix_parts(root: Path, target: Path | str) -> tuple[str, ...]:
    raw = os.fspath(target)
    if not raw or "\x00" in raw:
        raise AgentWorkspaceError("The edit path is invalid.")
    if os.path.isabs(raw):
        root_text = os.path.abspath(os.fspath(root))
        target_text = os.path.abspath(raw)
        try:
            if os.path.commonpath((root_text, target_text)) != root_text:
                raise AgentWorkspaceError("The edit path escapes the project root.")
        except ValueError as exc:
            raise AgentWorkspaceError("The edit path escapes the project root.") from exc
        raw = os.path.relpath(target_text, root_text)
    normalized = os.path.normpath(raw)
    parts = tuple(part for part in normalized.split(os.sep) if part)
    if not parts or parts == (".",) or any(part in {".", ".."} for part in parts):
        raise AgentWorkspaceError("The project root cannot be edited as a file.")
    return parts


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _file_read_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset : offset + _READ_CHUNK])
        if written <= 0:
            raise OSError("Could not write the temporary edit file.")
        offset += written


def _validate_payload(payload: bytes, max_bytes: int) -> bytes:
    if not isinstance(payload, bytes):
        raise TypeError("Project edit payloads must be bytes.")
    if len(payload) > max_bytes:
        raise OverflowError("Project edit payload exceeds the configured limit.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Project edit payload is not UTF-8 text.") from exc
    if "\x00" in text:
        raise ValueError("Project edit payload contains a NUL character.")
    return payload


def _read_all(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = os.read(descriptor, min(_READ_CHUNK, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


class _PosixVerifiedMutation:
    def __init__(self, workspace: ProjectWorkspace, target: Path | str, *, max_bytes: int) -> None:
        _require_posix_mutation_primitives()
        self.parts = _posix_parts(workspace.root, target)
        self.max_bytes = int(max_bytes)
        self._closed = False
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(workspace.root, _directory_flags())
            opened = os.fstat(descriptor)
            current = workspace.root.stat(follow_symlinks = False)
            expected = _expected_identity(workspace)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or _identity(opened) != _identity(current)
                or _identity(opened) != expected
            ):
                raise AgentWorkspaceError("Project root identity changed.")
            self._root_fd = descriptor
            self.root_identity = expected
            self.root_device = int(opened.st_dev)
            descriptor = None
        except AgentWorkspaceError:
            raise
        except OSError as exc:
            raise AgentWorkspaceError(
                "The project root is unavailable, changed identity, or unsafe to edit."
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @property
    def target_path(self) -> str:
        return os.path.join(*self.parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._root_fd)

    def _assert_open(self) -> None:
        if self._closed:
            raise AgentWorkspaceError("Project mutation boundary is closed.")
        metadata = os.fstat(self._root_fd)
        if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != self.root_identity:
            raise AgentWorkspaceError("Project root identity changed.")

    def _open_parent(self, *, create: bool) -> int:
        self._assert_open()
        current = os.dup(self._root_fd)
        try:
            for component in self.parts[:-1]:
                try:
                    child = os.open(component, _directory_flags(), dir_fd = current)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, 0o777, dir_fd = current)
                    except FileExistsError:
                        pass
                    child = os.open(component, _directory_flags(), dir_fd = current)
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode) or int(metadata.st_dev) != self.root_device:
                    os.close(child)
                    raise AgentWorkspaceError(
                        "Project edit paths cannot cross symbolic links or mounted filesystems."
                    )
                os.close(current)
                current = child
            self._assert_open()
            return current
        except Exception:
            os.close(current)
            raise

    def _open_target(self, parent_fd: int) -> tuple[int, os.stat_result]:
        descriptor = os.open(self.parts[-1], _file_read_flags(), dir_fd = parent_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_dev) != self.root_device:
                raise AgentWorkspaceError("The edit target is not a regular project file.")
            if metadata.st_nlink != 1:
                raise AgentWorkspaceError("Hard-linked edit targets are not supported.")
            return descriptor, metadata
        except Exception:
            os.close(descriptor)
            raise

    def _read_target(self, parent_fd: int, limit: int) -> tuple[bytes, os.stat_result]:
        descriptor, before = self._open_target(parent_fd)
        try:
            if before.st_size > limit:
                raise OverflowError("Project edit target exceeds the configured limit.")
            raw = _read_all(descriptor, limit)
            after = os.fstat(descriptor)
            if (
                _identity(before) != _identity(after)
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or after.st_nlink != 1
                or len(raw) > limit
            ):
                raise AgentWorkspaceError("The project file changed while it was being read.")
            return raw, before
        finally:
            os.close(descriptor)

    def read(self, limit: int) -> tuple[bytes, int, tuple[int, int]]:
        if limit < 0 or limit > self.max_bytes:
            raise ValueError("Project read limit is outside the configured mutation bound.")
        parent_fd = self._open_parent(create = False)
        try:
            raw, metadata = self._read_target(parent_fd, limit)
            self._assert_open()
            return raw, stat.S_IMODE(metadata.st_mode), _identity(metadata)
        finally:
            os.close(parent_fd)

    def _temporary(
        self, parent_fd: int, payload: bytes, mode: int, *, preserve_mode: bool
    ) -> tuple[str, tuple[int, int]]:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _attempt in range(8):
            name = f".unsloth_edit_{uuid.uuid4().hex}"
            try:
                requested_mode = 0o600 if preserve_mode else stat.S_IMODE(mode)
                descriptor = os.open(name, flags, requested_mode, dir_fd = parent_fd)
            except FileExistsError:
                continue
            committed = False
            before: Optional[os.stat_result] = None
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or int(before.st_dev) != self.root_device:
                    raise AgentWorkspaceError("Temporary edit file is unsafe.")
                _write_all(descriptor, payload)
                if preserve_mode:
                    # Replacement keeps the old file's exact permission bits.
                    # Creation leaves open(2)'s mode alone so umask applies once.
                    os.fchmod(descriptor, stat.S_IMODE(mode))
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                if _identity(before) != _identity(after) or after.st_size != len(payload):
                    raise AgentWorkspaceError("Temporary edit file changed while it was written.")
                committed = True
                return name, _identity(after)
            finally:
                os.close(descriptor)
                if not committed and before is not None:
                    with contextlib.suppress(OSError):
                        named = os.stat(name, dir_fd = parent_fd, follow_symlinks = False)
                        if _identity(named) == _identity(before):
                            os.unlink(name, dir_fd = parent_fd)
        raise FileExistsError("Could not allocate a unique temporary edit file.")

    @staticmethod
    def _unlink_if_identity(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
        with contextlib.suppress(OSError):
            metadata = os.stat(name, dir_fd = parent_fd, follow_symlinks = False)
            if _identity(metadata) == expected:
                os.unlink(name, dir_fd = parent_fd)

    def create(
        self,
        payload: bytes,
        mode: int = 0o666,
    ) -> Optional[str]:
        payload = _validate_payload(payload, self.max_bytes)
        parent_fd = self._open_parent(create = True)
        temp_name = ""
        temp_identity: Optional[tuple[int, int]] = None
        try:
            temp_name, temp_identity = self._temporary(
                parent_fd,
                payload,
                mode,
                preserve_mode = False,
            )
            try:
                os.link(
                    temp_name,
                    self.parts[-1],
                    src_dir_fd = parent_fd,
                    dst_dir_fd = parent_fd,
                    follow_symlinks = False,
                )
            except FileExistsError:
                return "exists"
            named = os.stat(self.parts[-1], dir_fd = parent_fd, follow_symlinks = False)
            if not stat.S_ISREG(named.st_mode) or _identity(named) != temp_identity:
                raise AgentWorkspaceError("Created project file changed during publication.")
            os.unlink(temp_name, dir_fd = parent_fd)
            temp_name = ""
            os.fsync(parent_fd)
            return None
        finally:
            if temp_name and temp_identity is not None:
                self._unlink_if_identity(parent_fd, temp_name, temp_identity)
            os.close(parent_fd)

    def replace(
        self, payload: bytes, *, expect: bytes, mode: int, identity: tuple[int, int]
    ) -> Optional[str]:
        payload = _validate_payload(payload, self.max_bytes)
        if not isinstance(expect, bytes) or len(expect) > self.max_bytes:
            raise OverflowError("Project edit content exceeds the configured limit.")
        parent_fd = self._open_parent(create = False)
        temp_name = ""
        temp_identity: Optional[tuple[int, int]] = None
        try:
            current, metadata = self._read_target(parent_fd, self.max_bytes)
            if current != expect or _identity(metadata) != tuple(identity):
                return "changed"
            temp_name, temp_identity = self._temporary(
                parent_fd,
                payload,
                mode,
                preserve_mode = True,
            )
            current, metadata = self._read_target(parent_fd, self.max_bytes)
            if current != expect or _identity(metadata) != tuple(identity):
                return "changed"
            os.replace(
                temp_name,
                self.parts[-1],
                src_dir_fd = parent_fd,
                dst_dir_fd = parent_fd,
            )
            temp_name = ""
            named = os.stat(self.parts[-1], dir_fd = parent_fd, follow_symlinks = False)
            if not stat.S_ISREG(named.st_mode) or _identity(named) != temp_identity:
                raise AgentWorkspaceError("Replaced project file changed during publication.")
            os.fsync(parent_fd)
            return None
        finally:
            if temp_name and temp_identity is not None:
                self._unlink_if_identity(parent_fd, temp_name, temp_identity)
            os.close(parent_fd)


# Win32 constants used by the handle-verified backend.
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_ATTRIBUTE_SPARSE_FILE = 0x00000200
_FILE_ATTRIBUTE_COMPRESSED = 0x00000800
_FILE_ATTRIBUTE_ENCRYPTED = 0x00004000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_LIST_DIRECTORY = 0x00000001
_DELETE = 0x00010000
_READ_CONTROL = 0x00020000
_WRITE_DAC = 0x00040000
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_BASIC_INFO_CLASS = 0
_FILE_STREAM_INFO_CLASS = 7
_FILE_ID_INFO_CLASS = 18
_FILE_DISPOSITION_INFO_CLASS = 4
_FILE_RENAME_INFO_CLASS = 3
_ERROR_HANDLE_EOF = 38
_ERROR_FILE_EXISTS = 80
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_ALREADY_EXISTS = 183
_ERROR_MORE_DATA = 234
_DACL_SECURITY_INFORMATION = 0x00000004
_UNPROTECTED_DACL_SECURITY_INFORMATION = 0x20000000
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_SE_DACL_PRESENT = 0x0004
_SE_DACL_PROTECTED = 0x1000
_SE_FILE_OBJECT = 1


_UNSUPPORTED_REPLACEMENT_ATTRIBUTES = (
    _FILE_ATTRIBUTE_SPARSE_FILE | _FILE_ATTRIBUTE_COMPRESSED | _FILE_ATTRIBUTE_ENCRYPTED
)


class WindowsMutationRejected(OSError):
    """A Windows path could not be proven to be an allowed local object."""

    def __init__(
        self,
        reason: str,
        *,
        reparse: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reparse = reparse


@dataclass(frozen = True)
class _WindowsHandleInfo:
    attributes: int
    identity_options: tuple[tuple[int, int], ...]
    size: int
    modified_ns: int
    final_path: str
    links: int = 1

    @property
    def is_directory(self) -> bool:
        return bool(self.attributes & _FILE_ATTRIBUTE_DIRECTORY)

    @property
    def is_reparse(self) -> bool:
        return bool(self.attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


@dataclass(frozen = True)
class _WindowsBasicMetadata:
    creation_time: int
    attributes: int


@dataclass(frozen = True)
class _WindowsDacl:
    present: bool
    protected: bool
    acl: Optional[bytes]

    @property
    def comparison_key(self) -> tuple[bool, bool, Optional[bytes]]:
        return self.present, self.protected, self.acl


class _Win32Api:
    def __init__(self) -> None:
        from ctypes import wintypes

        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

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
            _fields_ = [("volume_serial", ctypes.c_ulonglong), ("file_id", FileId128)]

        class FileBasicInfo(ctypes.Structure):
            _fields_ = [
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("attributes", wintypes.DWORD),
            ]

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("delete_file", wintypes.BOOL)]

        self.ByHandleFileInformation = ByHandleFileInformation
        self.FileIdInfo = FileIdInfo
        self.FileBasicInfo = FileBasicInfo
        self.FileDispositionInfo = FileDispositionInfo
        handle = wintypes.HANDLE
        dword = wintypes.DWORD
        boolean = wintypes.BOOL
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
        self.kernel32.CloseHandle.restype = boolean
        self.kernel32.GetFileInformationByHandle.argtypes = [
            handle,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self.kernel32.GetFileInformationByHandle.restype = boolean
        self.kernel32.GetFileInformationByHandleEx.argtypes = [
            handle,
            ctypes.c_int,
            ctypes.c_void_p,
            dword,
        ]
        self.kernel32.GetFileInformationByHandleEx.restype = boolean
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
        self.kernel32.ReadFile.restype = boolean
        self.kernel32.WriteFile.argtypes = [
            handle,
            ctypes.c_void_p,
            dword,
            ctypes.POINTER(dword),
            ctypes.c_void_p,
        ]
        self.kernel32.WriteFile.restype = boolean
        self.kernel32.FlushFileBuffers.argtypes = [handle]
        self.kernel32.FlushFileBuffers.restype = boolean
        self.kernel32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
        self.kernel32.CreateDirectoryW.restype = boolean
        self.kernel32.SetFileInformationByHandle.argtypes = [
            handle,
            ctypes.c_int,
            ctypes.c_void_p,
            dword,
        ]
        self.kernel32.SetFileInformationByHandle.restype = boolean
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
        void_pointer = ctypes.c_void_p
        self.advapi32.GetSecurityInfo.argtypes = [
            handle,
            dword,
            dword,
            ctypes.POINTER(void_pointer),
            ctypes.POINTER(void_pointer),
            ctypes.POINTER(void_pointer),
            ctypes.POINTER(void_pointer),
            ctypes.POINTER(void_pointer),
        ]
        self.advapi32.GetSecurityInfo.restype = dword
        self.advapi32.SetSecurityInfo.argtypes = [
            handle,
            dword,
            dword,
            void_pointer,
            void_pointer,
            void_pointer,
            void_pointer,
        ]
        self.advapi32.SetSecurityInfo.restype = dword
        self.advapi32.GetSecurityDescriptorControl.argtypes = [
            void_pointer,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(dword),
        ]
        self.advapi32.GetSecurityDescriptorControl.restype = boolean
        self.kernel32.LocalFree.argtypes = [void_pointer]
        self.kernel32.LocalFree.restype = void_pointer
        self.invalid_handle = ctypes.c_void_p(-1).value


_WINDOWS_API: Optional[_Win32Api] = None


def _win32_api() -> _Win32Api:
    global _WINDOWS_API
    if _WINDOWS_API is None:
        _WINDOWS_API = _Win32Api()
    return _WINDOWS_API


def windows_secure_mutation_supported() -> bool:
    if os.name != "nt":
        return False
    try:
        _win32_api()
    except (AttributeError, ImportError, OSError):
        return False
    return True


def _normalize_windows_path(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path).replace("/", "\\")
    folded = value.casefold()
    if folded.startswith("\\\\?\\unc\\"):
        value = "\\\\" + value[8:]
    elif folded.startswith("\\\\?\\"):
        value = value[4:]
    elif folded.startswith("\\??\\"):
        value = value[4:]
    return ntpath.normpath(value)


def _windows_key(path: str | os.PathLike[str]) -> str:
    normalized = ntpath.normcase(_normalize_windows_path(path))
    drive, tail = ntpath.splitdrive(normalized)
    # Keep the separator for a filesystem root. Stripping it turns ``C:\``
    # into the drive-relative path ``C:``, which makes commonpath reject every
    # descendant even though it is inside the root.
    if drive and tail == "\\":
        return drive + tail
    return normalized.rstrip("\\")


def _windows_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    candidate = _windows_key(path)
    parent = _windows_key(root)
    try:
        return ntpath.commonpath((candidate, parent)) == parent
    except ValueError:
        return False


def _validated_windows_path(path: str | os.PathLike[str]) -> str:
    value = _normalize_windows_path(path)
    if not value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise WindowsMutationRejected("Windows path contains invalid characters.")
    folded = value.casefold()
    if folded.startswith(("\\\\", "\\.\\", "\\??\\")):
        raise WindowsMutationRejected("Network and device paths are not supported.")
    drive, tail = ntpath.splitdrive(value)
    if len(drive) != 2 or drive[1] != ":" or not tail.startswith("\\"):
        raise WindowsMutationRejected("Windows project paths must use a local drive.")
    if ":" in tail:
        raise WindowsMutationRejected("Alternate data streams are not supported.")
    return value


def _extended_windows_path(path: str | os.PathLike[str]) -> str:
    return "\\\\?\\" + _validated_windows_path(path)


_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def _validated_windows_part(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or any(ord(char) < 32 for char in value)
        or any(char in value for char in ("\\", "/", ":"))
        or value.endswith((" ", "."))
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise WindowsMutationRejected("Windows edit path is invalid.")
    return value


def _windows_error(message: str) -> OSError:
    code = ctypes.get_last_error()
    factory = getattr(ctypes, "WinError", None)
    if factory is None:
        return OSError(code, f"{message} (WinError {code})")
    error = factory(code)
    # OSError.__str__ uses strerror rather than additional args. Preserve the
    # native error subclass/code while identifying the failing operation.
    error.strerror = f"{message}: {error.strerror}"
    return error


def _open_windows_handle(
    path: str,
    *,
    directory: bool = False,
    read: bool = False,
    write: bool = False,
    delete: bool = False,
    read_control: bool = False,
    write_dac: bool = False,
    share_write: bool = True,
    share_delete: bool = False,
    creation: int = _OPEN_EXISTING,
) -> int:
    api = _win32_api()
    access = _FILE_READ_ATTRIBUTES
    if directory:
        access |= _FILE_LIST_DIRECTORY
    if read:
        access |= _GENERIC_READ
    if write:
        access |= _GENERIC_WRITE
    if delete:
        access |= _DELETE
    if read_control:
        access |= _READ_CONTROL
    if write_dac:
        access |= _WRITE_DAC
    share = _FILE_SHARE_READ
    if share_write:
        share |= _FILE_SHARE_WRITE
    if share_delete:
        share |= _FILE_SHARE_DELETE
    handle = api.kernel32.CreateFileW(
        _extended_windows_path(path),
        access,
        share,
        None,
        creation,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle in {None, -1, api.invalid_handle}:
        raise _windows_error("Windows path could not be opened")
    return int(handle)


def _windows_final_path(handle: int) -> str:
    api = _win32_api()
    size = 1024
    while size <= 65_536:
        buffer = ctypes.create_unicode_buffer(size)
        length = int(api.kernel32.GetFinalPathNameByHandleW(handle, buffer, size, 0))
        if length == 0:
            raise _windows_error("Windows could not resolve an opened path")
        if length < size:
            return _validated_windows_path(buffer.value)
        size = length + 1
    raise WindowsMutationRejected("Windows path exceeds the supported length.")


def _windows_handle_info(handle: int) -> _WindowsHandleInfo:
    api = _win32_api()
    legacy = api.ByHandleFileInformation()
    if not api.kernel32.GetFileInformationByHandle(handle, ctypes.byref(legacy)):
        raise _windows_error("Windows could not inspect an opened path")
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
        extended_identity = (
            int(extended.volume_serial),
            int.from_bytes(bytes(extended.file_id.identifier), "little"),
        )
        if extended_identity != legacy_identity:
            identities.append(extended_identity)
    write_time = (int(legacy.write_time.high) << 32) | int(legacy.write_time.low)
    return _WindowsHandleInfo(
        attributes = int(legacy.attributes),
        identity_options = tuple(identities),
        size = (int(legacy.size_high) << 32) | int(legacy.size_low),
        modified_ns = write_time * 100,
        final_path = _windows_final_path(handle),
        links = int(legacy.links),
    )


def _windows_basic_metadata(handle: int) -> _WindowsBasicMetadata:
    api = _win32_api()
    basic = api.FileBasicInfo()
    if not api.kernel32.GetFileInformationByHandleEx(
        handle,
        _FILE_BASIC_INFO_CLASS,
        ctypes.byref(basic),
        ctypes.sizeof(basic),
    ):
        raise _windows_error("Windows could not inspect basic file metadata")
    return _WindowsBasicMetadata(
        creation_time = int(basic.creation_time),
        attributes = int(basic.attributes),
    )


def _set_windows_basic_metadata(handle: int, metadata: _WindowsBasicMetadata) -> None:
    api = _win32_api()
    basic = api.FileBasicInfo(
        creation_time = metadata.creation_time,
        last_access_time = 0,
        last_write_time = 0,
        change_time = 0,
        attributes = metadata.attributes,
    )
    if not api.kernel32.SetFileInformationByHandle(
        handle,
        _FILE_BASIC_INFO_CLASS,
        ctypes.byref(basic),
        ctypes.sizeof(basic),
    ):
        raise _windows_error("Windows could not preserve basic file metadata")


def _windows_dacl(handle: int) -> _WindowsDacl:
    api = _win32_api()
    dacl_pointer = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = int(
        api.advapi32.GetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(dacl_pointer),
            None,
            ctypes.byref(descriptor),
        )
    )
    if status:
        raise OSError(status, "Windows could not read the file DACL")
    try:
        control = ctypes.c_ushort(0)
        revision = api.wintypes.DWORD(0)
        if not api.advapi32.GetSecurityDescriptorControl(
            descriptor,
            ctypes.byref(control),
            ctypes.byref(revision),
        ):
            raise _windows_error("Windows could not inspect the file DACL")
        if not control.value & _SE_DACL_PRESENT:
            raise WindowsMutationRejected("Windows files without a DACL cannot be edited safely.")
        acl: Optional[bytes] = None
        if dacl_pointer.value:
            header = ctypes.string_at(dacl_pointer, 8)
            acl_size = int(struct.unpack_from("<H", header, 2)[0])
            if acl_size < 8 or acl_size > 1024 * 1024:
                raise WindowsMutationRejected("Windows returned an invalid file DACL.")
            acl = ctypes.string_at(dacl_pointer, acl_size)
        return _WindowsDacl(
            present = True,
            protected = bool(control.value & _SE_DACL_PROTECTED),
            acl = acl,
        )
    finally:
        if descriptor.value:
            api.kernel32.LocalFree(descriptor)


def _set_windows_dacl(handle: int, dacl: _WindowsDacl) -> None:
    if not dacl.present:
        raise WindowsMutationRejected("Windows files without a DACL cannot be edited safely.")
    api = _win32_api()
    acl_buffer = ctypes.create_string_buffer(dacl.acl, len(dacl.acl)) if dacl.acl else None
    acl_pointer = ctypes.cast(acl_buffer, ctypes.c_void_p) if acl_buffer is not None else None
    protection = (
        _PROTECTED_DACL_SECURITY_INFORMATION
        if dacl.protected
        else _UNPROTECTED_DACL_SECURITY_INFORMATION
    )
    status = int(
        api.advapi32.SetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | protection,
            None,
            None,
            acl_pointer,
            None,
        )
    )
    if status:
        raise OSError(status, "Windows could not preserve the file DACL")


def _windows_stream_names(handle: int) -> tuple[str, ...]:
    api = _win32_api()
    size = 4096
    while size <= 1024 * 1024:
        buffer = ctypes.create_string_buffer(size)
        if api.kernel32.GetFileInformationByHandleEx(
            handle,
            _FILE_STREAM_INFO_CLASS,
            buffer,
            size,
        ):
            names: list[str] = []
            offset = 0
            while True:
                if offset + 24 > size:
                    raise WindowsMutationRejected("Windows returned invalid stream metadata.")
                next_offset, name_length = struct.unpack_from("<II", buffer.raw, offset)
                if name_length % 2 or offset + 24 + name_length > size:
                    raise WindowsMutationRejected("Windows returned invalid stream metadata.")
                raw_name = bytes(buffer.raw[offset + 24 : offset + 24 + name_length])
                try:
                    names.append(raw_name.decode("utf-16-le"))
                except UnicodeDecodeError as exc:
                    raise WindowsMutationRejected(
                        "Windows returned an invalid stream name."
                    ) from exc
                if next_offset == 0:
                    return tuple(names)
                if next_offset < 24 + name_length or next_offset % 8:
                    raise WindowsMutationRejected("Windows returned invalid stream metadata.")
                offset += int(next_offset)
                if offset >= size:
                    raise WindowsMutationRejected("Windows returned invalid stream metadata.")
        code = ctypes.get_last_error()
        if code == _ERROR_HANDLE_EOF:
            return ()
        if code not in {_ERROR_INSUFFICIENT_BUFFER, _ERROR_MORE_DATA}:
            raise _windows_error("Windows could not inspect file streams")
        size *= 2
    raise WindowsMutationRejected("Windows file stream metadata exceeds the supported limit.")


def _assert_windows_replacement_metadata_supported(
    info: _WindowsHandleInfo, stream_names: Sequence[str]
) -> None:
    if info.attributes & _UNSUPPORTED_REPLACEMENT_ATTRIBUTES:
        raise WindowsMutationRejected(
            "Compressed, encrypted, and sparse Windows files cannot be edited safely."
        )
    if any(name.casefold() != "::$data" for name in stream_names):
        raise WindowsMutationRejected("Windows files with named streams cannot be edited safely.")


def _assert_windows_handle(
    info: _WindowsHandleInfo,
    expected_path: str,
    root_path: str,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    directory: Optional[bool] = None,
) -> None:
    if info.is_reparse:
        raise WindowsMutationRejected("Windows reparse points are not supported.", reparse = True)
    if not _windows_within(info.final_path, root_path):
        raise WindowsMutationRejected("Workspace path escaped the project root.")
    if _windows_key(info.final_path) != _windows_key(expected_path):
        raise WindowsMutationRejected("Workspace path changed while it was inspected.")
    if directory is True and not info.is_directory:
        raise NotADirectoryError(expected_path)
    if directory is False and info.is_directory:
        raise IsADirectoryError(expected_path)
    if not info.is_directory and info.links != 1:
        raise WindowsMutationRejected("Hard-linked edit targets are not supported.")
    if expected_identity is not None and tuple(expected_identity) not in info.identity_options:
        raise WindowsMutationRejected("Project root identity changed.")


class _WindowsMutationOps(Protocol):
    def open_existing(
        self,
        path: str,
        *,
        directory: bool = False,
        read: bool = False,
        delete: bool = False,
        read_control: bool = False,
        write_dac: bool = False,
        share_write: bool = True,
        share_delete: bool = False,
    ) -> int: ...

    def create_temp(self, path: str) -> int: ...
    def close(self, handle: int) -> None: ...
    def info(self, handle: int) -> _WindowsHandleInfo: ...
    def read(self, handle: int, limit: int) -> bytes: ...
    def write_and_flush(self, handle: int, payload: bytes) -> None: ...
    def create_directory(self, path: str) -> None: ...
    def basic_metadata(self, handle: int) -> _WindowsBasicMetadata: ...
    def dacl(self, handle: int) -> _WindowsDacl: ...
    def stream_names(self, handle: int) -> tuple[str, ...]: ...
    def apply_basic_metadata(self, handle: int, metadata: _WindowsBasicMetadata) -> None: ...
    def apply_dacl(self, handle: int, dacl: _WindowsDacl) -> None: ...
    def rename(
        self, handle: int, parent_handle: int, target_name: str, *, replace: bool
    ) -> bool: ...
    def mark_delete(self, handle: int) -> None: ...


class _NativeWindowsMutationOps:
    def open_existing(
        self,
        path: str,
        *,
        directory: bool = False,
        read: bool = False,
        delete: bool = False,
        read_control: bool = False,
        write_dac: bool = False,
        share_write: bool = True,
        share_delete: bool = False,
    ) -> int:
        return _open_windows_handle(
            path,
            directory = directory,
            read = read,
            delete = delete,
            read_control = read_control,
            write_dac = write_dac,
            share_write = share_write,
            share_delete = share_delete,
        )

    def create_temp(self, path: str) -> int:
        return _open_windows_handle(
            path,
            write = True,
            delete = True,
            read_control = True,
            write_dac = True,
            share_write = False,
            share_delete = False,
            creation = _CREATE_NEW,
        )

    def close(self, handle: int) -> None:
        _win32_api().kernel32.CloseHandle(handle)

    def info(self, handle: int) -> _WindowsHandleInfo:
        return _windows_handle_info(handle)

    def read(self, handle: int, limit: int) -> bytes:
        output = bytearray()
        while len(output) < limit:
            requested = min(_READ_CHUNK, limit - len(output))
            buffer = ctypes.create_string_buffer(requested)
            count = _win32_api().wintypes.DWORD(0)
            if not _win32_api().kernel32.ReadFile(
                handle, buffer, requested, ctypes.byref(count), None
            ):
                raise _windows_error("Windows could not read an opened file")
            if count.value == 0:
                break
            output.extend(buffer.raw[: count.value])
        return bytes(output)

    def write_and_flush(self, handle: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + _READ_CHUNK]
            buffer = ctypes.create_string_buffer(chunk)
            count = _win32_api().wintypes.DWORD(0)
            if not _win32_api().kernel32.WriteFile(
                handle, buffer, len(chunk), ctypes.byref(count), None
            ):
                raise _windows_error("Windows could not write a temporary edit file")
            if count.value <= 0:
                raise OSError("Windows reported a short write to a temporary edit file.")
            offset += int(count.value)
        if not _win32_api().kernel32.FlushFileBuffers(handle):
            raise _windows_error("Windows could not flush a temporary edit file")

    def create_directory(self, path: str) -> None:
        if _win32_api().kernel32.CreateDirectoryW(_extended_windows_path(path), None):
            return
        code = ctypes.get_last_error()
        if code not in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
            raise _windows_error("Windows could not create a project directory")

    def basic_metadata(self, handle: int) -> _WindowsBasicMetadata:
        return _windows_basic_metadata(handle)

    def dacl(self, handle: int) -> _WindowsDacl:
        return _windows_dacl(handle)

    def stream_names(self, handle: int) -> tuple[str, ...]:
        return _windows_stream_names(handle)

    def apply_basic_metadata(self, handle: int, metadata: _WindowsBasicMetadata) -> None:
        _set_windows_basic_metadata(handle, metadata)

    def apply_dacl(self, handle: int, dacl: _WindowsDacl) -> None:
        _set_windows_dacl(handle, dacl)

    def rename(self, handle: int, parent_handle: int, target_name: str, *, replace: bool) -> bool:
        """Publish an open temp relative to an already-open parent handle."""
        name = _validated_windows_part(target_name)
        api = _win32_api()
        encoded_name = name.encode("utf-16-le")

        class FileRenameInfo(ctypes.Structure):
            _fields_ = [
                ("replace_if_exists", api.wintypes.BOOL),
                ("root_directory", api.wintypes.HANDLE),
                ("file_name_length", api.wintypes.DWORD),
                # Keep the WCHAR terminator and native structure padding in
                # the buffer passed to Win32, outside FileNameLength.
                ("file_name", ctypes.c_ubyte * (len(encoded_name) + 2)),
            ]

        info = FileRenameInfo()
        info.replace_if_exists = bool(replace)
        info.root_directory = parent_handle
        info.file_name_length = len(encoded_name)
        info.file_name[: len(encoded_name)] = encoded_name
        if api.kernel32.SetFileInformationByHandle(
            handle,
            _FILE_RENAME_INFO_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            return True
        code = ctypes.get_last_error()
        if not replace and code in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
            return False
        raise _windows_error("Windows could not atomically publish the project file")

    def mark_delete(self, handle: int) -> None:
        api = _win32_api()
        disposition = api.FileDispositionInfo(delete_file = True)
        if not api.kernel32.SetFileInformationByHandle(
            handle,
            _FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise _windows_error("Windows could not clean up a temporary edit file")


@dataclass(frozen = True)
class _WindowsGuard:
    path: str
    handle: int
    identities: tuple[tuple[int, int], ...]


class _WindowsVerifiedMutation:
    def __init__(
        self,
        workspace: ProjectWorkspace,
        target: Path | str,
        *,
        max_bytes: int,
        ops: Optional[_WindowsMutationOps] = None,
    ) -> None:
        root = _validated_windows_path(str(workspace.root))
        raw_target_text = str(target).replace("/", "\\")
        if raw_target_text.casefold().startswith(("\\\\?\\", "\\??\\", "\\.\\")):
            raise WindowsMutationRejected(
                "Extended-length and device namespace edit paths are not supported."
            )
        raw_target = _normalize_windows_path(raw_target_text)
        requested = (
            _validated_windows_path(raw_target)
            if ntpath.isabs(raw_target)
            else _validated_windows_path(ntpath.join(root, raw_target))
        )
        if not _windows_within(requested, root):
            raise WindowsMutationRejected("The edit path escapes the project root.")
        relative = ntpath.relpath(requested, root)
        parts = tuple(part for part in relative.split("\\") if part)
        if not parts or parts == (".",):
            raise WindowsMutationRejected("The project root cannot be edited as a file.")
        self.parts = tuple(_validated_windows_part(part) for part in parts)
        self.max_bytes = int(max_bytes)
        self._ops = ops or _NativeWindowsMutationOps()
        self._closed = False
        handle = self._ops.open_existing(root, directory = True, share_delete = False)
        try:
            info = self._ops.info(handle)
            _assert_windows_handle(
                info,
                root,
                root,
                expected_identity = _expected_identity(workspace),
                directory = True,
            )
        except Exception:
            self._ops.close(handle)
            raise
        self.path = info.final_path
        self.root_identity = _expected_identity(workspace)
        self._root = _WindowsGuard(self.path, handle, info.identity_options)

    @property
    def target_path(self) -> str:
        return ntpath.join(self.path, *self.parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ops.close(self._root.handle)

    def _assert_guard(
        self,
        guard: _WindowsGuard,
        *,
        root: bool = False,
    ) -> None:
        info = self._ops.info(guard.handle)
        _assert_windows_handle(info, guard.path, self.path, directory = True)
        if not any(identity in guard.identities for identity in info.identity_options):
            raise WindowsMutationRejected("A project directory changed during the edit.")
        reopened = self._ops.open_existing(guard.path, directory = True, share_delete = False)
        try:
            current = self._ops.info(reopened)
            _assert_windows_handle(current, guard.path, self.path, directory = True)
            if not any(identity in guard.identities for identity in current.identity_options):
                raise WindowsMutationRejected(
                    "Project root identity changed."
                    if root
                    else "A project directory changed during the edit."
                )
        finally:
            self._ops.close(reopened)

    def _recheck(self, guards: Sequence[_WindowsGuard]) -> None:
        self._assert_guard(self._root, root = True)
        for guard in guards:
            self._assert_guard(guard)

    def _open_parent(self, *, create: bool) -> list[_WindowsGuard]:
        guards: list[_WindowsGuard] = []
        current = self.path
        try:
            for component in self.parts[:-1]:
                expected = ntpath.join(current, component)
                try:
                    handle = self._ops.open_existing(
                        expected,
                        directory = True,
                        share_delete = False,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    self._ops.create_directory(expected)
                    handle = self._ops.open_existing(
                        expected,
                        directory = True,
                        share_delete = False,
                    )
                try:
                    info = self._ops.info(handle)
                    _assert_windows_handle(info, expected, self.path, directory = True)
                except Exception:
                    self._ops.close(handle)
                    raise
                guards.append(_WindowsGuard(info.final_path, handle, info.identity_options))
                current = info.final_path
            self._recheck(guards)
            return guards
        except Exception:
            self._close_guards(guards)
            raise

    def _close_guards(self, guards: Sequence[_WindowsGuard]) -> None:
        for guard in reversed(guards):
            self._ops.close(guard.handle)

    def _open_target(self) -> tuple[int, _WindowsHandleInfo]:
        handle = self._ops.open_existing(
            self.target_path,
            read = True,
            read_control = True,
            share_write = False,
            share_delete = False,
        )
        try:
            info = self._ops.info(handle)
            _assert_windows_handle(info, self.target_path, self.path, directory = False)
            return handle, info
        except Exception:
            self._ops.close(handle)
            raise

    def _read_target(self, limit: int) -> tuple[bytes, _WindowsHandleInfo]:
        handle, before = self._open_target()
        try:
            if before.size > limit:
                raise OverflowError("Project edit target exceeds the configured limit.")
            raw = self._ops.read(handle, limit + 1)
            after = self._ops.info(handle)
            if (
                before.identity_options != after.identity_options
                or before.size != after.size
                or before.modified_ns != after.modified_ns
                or after.links != 1
                or _windows_key(before.final_path) != _windows_key(after.final_path)
                or len(raw) > limit
            ):
                raise WindowsMutationRejected("File changed while it was being read.")
            return raw, before
        finally:
            self._ops.close(handle)

    def read(self, limit: int) -> tuple[bytes, int, tuple[int, int]]:
        if limit < 0 or limit > self.max_bytes:
            raise ValueError("Project read limit is outside the configured mutation bound.")
        guards = self._open_parent(create = False)
        try:
            raw, info = self._read_target(limit)
            self._recheck(guards)
            return raw, info.attributes, info.identity_options[0]
        finally:
            self._close_guards(guards)

    @staticmethod
    def _identity_matches(info: _WindowsHandleInfo, expected: tuple[int, int]) -> bool:
        return tuple(expected) in info.identity_options

    def _temporary(self, payload: bytes, parent: str) -> tuple[str, tuple[int, int], int]:
        for _attempt in range(8):
            path = ntpath.join(parent, f".unsloth_edit_{uuid.uuid4().hex}")
            try:
                handle = self._ops.create_temp(path)
            except FileExistsError:
                continue
            try:
                before = self._ops.info(handle)
                _assert_windows_handle(before, path, self.path, directory = False)
                self._ops.write_and_flush(handle, payload)
                after = self._ops.info(handle)
                if (
                    before.identity_options != after.identity_options
                    or after.size != len(payload)
                    or _windows_key(after.final_path) != _windows_key(path)
                ):
                    raise WindowsMutationRejected(
                        "Temporary edit file changed while it was written."
                    )
                identity = after.identity_options[0]
            except Exception:
                try:
                    self._ops.mark_delete(handle)
                finally:
                    self._ops.close(handle)
                raise
            return path, identity, handle
        raise FileExistsError("Could not allocate a unique temporary edit file.")

    def _cleanup_temp(self, handle: int, path: str, identity: tuple[int, int]) -> None:
        try:
            info = self._ops.info(handle)
            _assert_windows_handle(info, path, self.path, directory = False)
            if self._identity_matches(info, identity):
                self._ops.mark_delete(handle)
        finally:
            self._ops.close(handle)

    def _matches(self, expect: bytes, identity: tuple[int, int]) -> bool:
        try:
            current, info = self._read_target(len(expect))
        except WindowsMutationRejected as exc:
            if exc.reparse:
                raise
            return False
        except (FileNotFoundError, NotADirectoryError):
            return False
        return current == expect and self._identity_matches(info, identity)

    def _copy_verified_replacement_metadata(
        self, temp_handle: int, expect: bytes, identity: tuple[int, int]
    ) -> bool:
        """Copy stable target metadata to the temp while both handles are verified."""
        handle, before = self._open_target()
        try:
            if before.size > len(expect):
                return False
            current = self._ops.read(handle, len(expect) + 1)
            after_read = self._ops.info(handle)
            if (
                current != expect
                or not self._identity_matches(after_read, identity)
                or before.identity_options != after_read.identity_options
                or before.size != after_read.size
                or before.modified_ns != after_read.modified_ns
                or _windows_key(before.final_path) != _windows_key(after_read.final_path)
            ):
                return False

            streams = self._ops.stream_names(handle)
            basic = self._ops.basic_metadata(handle)
            dacl = self._ops.dacl(handle)
            streams_after = self._ops.stream_names(handle)
            basic_after = self._ops.basic_metadata(handle)
            dacl_after = self._ops.dacl(handle)
            final = self._ops.info(handle)
            if (
                streams != streams_after
                or basic != basic_after
                or dacl.comparison_key != dacl_after.comparison_key
                or after_read.identity_options != final.identity_options
                or after_read.size != final.size
                or after_read.modified_ns != final.modified_ns
                or _windows_key(after_read.final_path) != _windows_key(final.final_path)
            ):
                return False
            if basic.attributes != final.attributes:
                return False
            _assert_windows_replacement_metadata_supported(final, streams)

            self._ops.apply_dacl(temp_handle, dacl)
            self._ops.apply_basic_metadata(temp_handle, basic)
            temp_info = self._ops.info(temp_handle)
            temp_streams = self._ops.stream_names(temp_handle)
            temp_basic = self._ops.basic_metadata(temp_handle)
            temp_dacl = self._ops.dacl(temp_handle)
            _assert_windows_replacement_metadata_supported(temp_info, temp_streams)
            if (
                temp_basic != basic
                or temp_info.attributes != basic.attributes
                or temp_dacl.comparison_key != dacl.comparison_key
            ):
                raise WindowsMutationRejected("Windows could not verify replacement file metadata.")
            return True
        finally:
            self._ops.close(handle)

    def _target_exists_safely(self) -> bool:
        try:
            handle = self._ops.open_existing(
                self.target_path, share_write = False, share_delete = False
            )
        except FileNotFoundError:
            return False
        try:
            info = self._ops.info(handle)
            _assert_windows_handle(info, self.target_path, self.path, directory = False)
            return True
        finally:
            self._ops.close(handle)

    def create(
        self,
        payload: bytes,
        mode: int = 0o666,
    ) -> Optional[str]:
        del mode
        payload = _validate_payload(payload, self.max_bytes)
        guards = self._open_parent(create = True)
        parent = guards[-1].path if guards else self.path
        temp_path = ""
        temp_identity: Optional[tuple[int, int]] = None
        temp_handle: Optional[int] = None
        committed = False
        try:
            if self._target_exists_safely():
                return "exists"
            temp_path, temp_identity, temp_handle = self._temporary(payload, parent)
            self._recheck(guards)
            parent_handle = guards[-1].handle if guards else self._root.handle
            committed = self._ops.rename(
                temp_handle,
                parent_handle,
                self.parts[-1],
                replace = False,
            )
            if not committed:
                return "exists"
            after = self._ops.info(temp_handle)
            _assert_windows_handle(after, self.target_path, self.path, directory = False)
            if not self._identity_matches(after, temp_identity) or after.size != len(payload):
                raise WindowsMutationRejected("Created project file changed during publication.")
            return None
        finally:
            if temp_handle is not None:
                if temp_path and not committed and temp_identity is not None:
                    self._cleanup_temp(temp_handle, temp_path, temp_identity)
                else:
                    self._ops.close(temp_handle)
            self._close_guards(guards)

    def replace(
        self, payload: bytes, *, expect: bytes, mode: int, identity: tuple[int, int]
    ) -> Optional[str]:
        del mode
        payload = _validate_payload(payload, self.max_bytes)
        if not isinstance(expect, bytes) or len(expect) > self.max_bytes:
            raise OverflowError("Project edit content exceeds the configured limit.")
        guards = self._open_parent(create = False)
        parent = guards[-1].path if guards else self.path
        temp_path = ""
        temp_identity: Optional[tuple[int, int]] = None
        temp_handle: Optional[int] = None
        committed = False
        try:
            if not self._matches(expect, identity):
                return "changed"
            temp_path, temp_identity, temp_handle = self._temporary(payload, parent)
            self._recheck(guards)
            if not self._copy_verified_replacement_metadata(temp_handle, expect, identity):
                return "changed"
            parent_handle = guards[-1].handle if guards else self._root.handle
            committed = self._ops.rename(
                temp_handle,
                parent_handle,
                self.parts[-1],
                replace = True,
            )
            after = self._ops.info(temp_handle)
            _assert_windows_handle(after, self.target_path, self.path, directory = False)
            if not self._identity_matches(after, temp_identity) or after.size != len(payload):
                raise WindowsMutationRejected("Replaced project file changed during publication.")
            return None
        finally:
            if temp_handle is not None:
                if temp_path and not committed and temp_identity is not None:
                    self._cleanup_temp(temp_handle, temp_path, temp_identity)
                else:
                    self._ops.close(temp_handle)
            self._close_guards(guards)


class ProjectFileMutation:
    """Cross-platform, serialized project file mutation boundary."""

    def __init__(self, backend, identity: tuple[int, int]) -> None:
        self._backend = backend
        self._identity = identity
        self._closed = False

    @classmethod
    def open(
        cls,
        workspace: ProjectWorkspace,
        target: Path | str,
        *,
        max_bytes: int = _MAX_MUTATION_BYTES,
        cancel_event = None,
        _windows_ops: Optional[_WindowsMutationOps] = None,
    ) -> "ProjectFileMutation":
        if max_bytes < 0:
            raise ValueError("Project mutation limit cannot be negative.")
        identity = _expected_identity(workspace)
        if not acquire_workspace_mutation_slot(identity, cancel_event):
            raise AgentWorkspaceError("Project edit was cancelled before it started.")
        backend = None
        try:
            if os.name == "nt" or _windows_ops is not None:
                if _windows_ops is None and not windows_secure_mutation_supported():
                    raise AgentWorkspaceError(
                        "Secure Windows workspace mutation is unavailable on this host."
                    )
                backend = _WindowsVerifiedMutation(
                    workspace,
                    target,
                    max_bytes = max_bytes,
                    ops = _windows_ops,
                )
            else:
                backend = _PosixVerifiedMutation(workspace, target, max_bytes = max_bytes)
            return cls(backend, identity)
        except Exception:
            if backend is not None:
                backend.close()
            release_workspace_mutation_slot(identity)
            raise

    @property
    def target_path(self) -> str:
        return self._backend.target_path

    def read(self, limit: int) -> tuple[bytes, int, tuple[int, int]]:
        return self._backend.read(limit)

    def create(
        self,
        payload: bytes,
        mode: int = 0o666,
    ) -> Optional[str]:
        return self._backend.create(payload, mode)

    def replace(
        self, payload: bytes, *, expect: bytes, mode: int, identity: tuple[int, int]
    ) -> Optional[str]:
        return self._backend.replace(payload, expect = expect, mode = mode, identity = identity)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.close()
        finally:
            release_workspace_mutation_slot(self._identity)

    def __enter__(self) -> "ProjectFileMutation":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()


__all__ = [
    "ProjectFileMutation",
    "WindowsMutationRejected",
    "acquire_workspace_mutation_slot",
    "release_workspace_mutation_slot",
    "windows_secure_mutation_supported",
]
