# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fake multi-GB checkpoints in the diffusion and video suites must cost (almost) nothing on disk."""

from __future__ import annotations

import os

from ._sparse_file import extend_sparse

_GIB = 1 << 30


def _allocated_bytes(path) -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetCompressedFileSizeW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetCompressedFileSizeW.restype = wintypes.DWORD
        high = wintypes.DWORD(0)
        low = kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
        return (high.value << 32) | low
    return os.stat(path).st_blocks * 512


def test_a_gigabyte_fake_file_allocates_almost_nothing(tmp_path):
    path = tmp_path / "fake.safetensors"
    with open(path, "wb") as fh:
        fh.write(b"header")
        extend_sparse(fh, _GIB)
    assert os.path.getsize(path) == _GIB
    with open(path, "rb") as fh:
        assert fh.read(6) == b"header"
    assert _allocated_bytes(path) < 64 << 20, _allocated_bytes(path)
