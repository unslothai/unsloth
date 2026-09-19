# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Private per-run Windows named-pipe transport for the MXC supervisor."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import secrets
import time

if os.name == "nt":
    import _winapi


PIPE_ACCESS_DUPLEX = 0x00000003
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
PIPE_TYPE_BYTE = 0x00000000
PIPE_READMODE_BYTE = 0x00000000
PIPE_WAIT = 0x00000000
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
ERROR_BROKEN_PIPE = 109
ERROR_PIPE_NOT_CONNECTED = 233
SDDL_REVISION_1 = 1


class PipeError(RuntimeError):
    pass


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


def _win_functions():
    if os.name != "nt":
        raise PipeError("the MXC named-pipe transport is Windows-only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SecurityAttributes),
    ]
    kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.WriteFile.restype = wintypes.BOOL
    kernel32.PeekNamedPipe.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.PeekNamedPipe.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    return kernel32, advapi32


class PipeChannel:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def close(self) -> None:
        if self._handle is not None:
            kernel32, _ = _win_functions()
            kernel32.CloseHandle(self._handle)
            self._handle = None

    def available(self) -> int | None:
        if self._handle is None:
            return None
        kernel32, _ = _win_functions()
        available = wintypes.DWORD()
        if kernel32.PeekNamedPipe(self._handle, None, 0, None, ctypes.byref(available), None):
            return int(available.value)
        error = ctypes.get_last_error()
        if error in {ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED}:
            return None
        raise PipeError(f"PeekNamedPipe failed: {ctypes.WinError(error)}")

    def read_available(self, limit: int) -> bytes | None:
        available = self.available()
        if available is None:
            return b""
        if available == 0:
            return None
        size = min(available, limit)
        buffer = ctypes.create_string_buffer(size)
        read = wintypes.DWORD()
        kernel32, _ = _win_functions()
        if not kernel32.ReadFile(self._handle, buffer, size, ctypes.byref(read), None):
            error = ctypes.get_last_error()
            if error in {ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED}:
                return b""
            raise PipeError(f"ReadFile failed: {ctypes.WinError(error)}")
        return buffer.raw[: read.value]

    def write(self, data: bytes) -> None:
        if self._handle is None:
            raise PipeError("named-pipe channel is closed")
        kernel32, _ = _win_functions()
        offset = 0
        while offset < len(data):
            written = wintypes.DWORD()
            chunk = data[offset:]
            buffer = ctypes.create_string_buffer(chunk)
            if not kernel32.WriteFile(
                self._handle, buffer, len(chunk), ctypes.byref(written), None
            ):
                raise PipeError(f"WriteFile failed: {ctypes.WinError(ctypes.get_last_error())}")
            if written.value == 0:
                raise PipeError("named-pipe write made no progress")
            offset += written.value


class PrivatePipeServer:
    def __init__(self, *, buffer_size: int) -> None:
        kernel32, advapi32 = _win_functions()
        self.name = rf"\\.\pipe\unsloth-mxc-{secrets.token_hex(24)}"
        descriptor = wintypes.LPVOID()
        # Protected DACL: only LocalSystem and the object owner (the Studio
        # process identity) may open the endpoint. The auth token remains a
        # second, independent protocol check.
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;;GA;;;SY)(A;;GA;;;OW)",
            SDDL_REVISION_1,
            ctypes.byref(descriptor),
            None,
        ):
            raise PipeError(
                f"failed to create named-pipe security descriptor: {ctypes.WinError(ctypes.get_last_error())}"
            )
        self._descriptor = descriptor
        attributes = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
        handle = kernel32.CreateNamedPipeW(
            self.name,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            buffer_size,
            buffer_size,
            0,
            ctypes.byref(attributes),
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, invalid}:
            error = ctypes.get_last_error()
            kernel32.LocalFree(descriptor)
            self._descriptor = None
            raise PipeError(f"CreateNamedPipeW failed: {ctypes.WinError(error)}")
        self._handle = handle

    def accept(self, *, deadline: float, cancel_event=None, proc=None) -> PipeChannel:
        try:
            overlapped = _winapi.ConnectNamedPipe(self._handle, overlapped=True)
        except OSError as exc:
            if exc.winerror != _winapi.ERROR_PIPE_CONNECTED:
                raise PipeError(f"ConnectNamedPipe failed: {exc}") from exc
            overlapped = None
        if overlapped is not None:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    overlapped.cancel()
                    raise PipeError("named-pipe connection cancelled")
                if proc is not None and proc.poll() is not None:
                    overlapped.cancel()
                    raise PipeError("MXC supervisor exited before connecting to its control pipe")
                if time.monotonic() >= deadline:
                    overlapped.cancel()
                    raise PipeError("MXC supervisor control-pipe connection timed out")
                result = _winapi.WaitForSingleObject(overlapped.event, 50)
                if result == WAIT_OBJECT_0:
                    _, error = overlapped.GetOverlappedResult(True)
                    if error:
                        raise PipeError(f"ConnectNamedPipe completion failed with {error}")
                    break
                if result != WAIT_TIMEOUT:
                    overlapped.cancel()
                    raise PipeError(f"unexpected named-pipe wait result: {result}")
        handle = self._handle
        self._handle = None
        return PipeChannel(handle)

    def close(self) -> None:
        kernel32, _ = _win_functions()
        if self._handle is not None:
            kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._descriptor is not None:
            kernel32.LocalFree(self._descriptor)
            self._descriptor = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
