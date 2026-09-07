# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checked native pipe I/O shared by Python and Terminal launch owners."""

import ctypes
from ctypes import wintypes as W
import io


def pipe_api():
    from .native_bindings import _api

    api = _api().kernel32
    api.PeekNamedPipe.argtypes = [
        W.HANDLE,
        ctypes.c_void_p,
        W.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(W.DWORD),
        ctypes.c_void_p,
    ]
    api.PeekNamedPipe.restype = W.BOOL
    api.ReadFile.argtypes = [
        W.HANDLE,
        ctypes.c_void_p,
        W.DWORD,
        ctypes.POINTER(W.DWORD),
        ctypes.c_void_p,
    ]
    api.ReadFile.restype = W.BOOL
    api.WriteFile.argtypes = api.ReadFile.argtypes
    api.WriteFile.restype = W.BOOL
    return api


class NativePipeReader(io.RawIOBase):
    """Read stdout without surrendering its checked-close native owner."""

    def __init__(self, handle, close_handle):
        self._handle = handle
        self._close_handle = close_handle
        self._api = pipe_api()

    def readable(self):
        return True

    def readinto(self, buffer):
        from .native_bindings import _winerror

        if self.closed:
            raise ValueError("read of closed sandbox stdout")
        view = memoryview(buffer).cast("B")
        if view.readonly:
            raise TypeError("readinto requires a writable buffer")
        size = min(len(view), 65536)
        if not size:
            return 0
        target = (ctypes.c_char * size).from_buffer(view)
        count = W.DWORD()
        if not self._api.ReadFile(self._handle, target, size, ctypes.byref(count), None):
            if ctypes.get_last_error() == 109:
                return 0
            raise _winerror("ReadFile(sandbox stdout)")
        return count.value

    def close(self):
        if not self.closed:
            self._close_handle(self._handle)
            super().close()
