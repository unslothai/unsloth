# SPDX-License-Identifier: AGPL-3.0-only

"""Studio verifies wxc-exec.exe by SHA-256 on every launch: the CNG path must hash exactly what hashlib does."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys

import pytest

from core.inference import mxc_runtime

CHUNK = mxc_runtime._HASH_CHUNK
SIZES = [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 7]


def _blob(size: int) -> bytes:
    # Embedded zero runs and every byte value, not only random data.
    pattern = bytes(range(256)) + b"\x00" * 300
    return (pattern * (size // len(pattern) + 1))[:size]


class _FakeBcrypt:
    """The BCrypt calls _Cng.digest makes, backed by hashlib, so the chunk loop runs on any OS."""

    def __init__(self, fail = None):
        self.fail = fail
        self.live = {}
        self.destroyed = 0

    def BCryptCreateHash(self, _algorithm, handle_ref, *_rest):
        if self.fail == "create":
            return -1
        key = len(self.live) + 1
        self.live[key] = hashlib.sha256()
        handle_ref._obj.value = key
        return 0

    def BCryptHashData(self, handle, native, count, _flags):
        if self.fail == "data":
            return -1
        self.live[handle.value].update(bytes(native[:count]))
        return 0

    def BCryptFinishHash(self, handle, output, size, _flags):
        digest = self.live[handle.value].digest()
        assert size == len(digest) == 32
        ctypes.memmove(output, digest, size)
        return 0

    def BCryptDestroyHash(self, handle):
        self.destroyed += 1
        self.live.pop(handle.value, None)
        return 0


def _fake_cng(fail = None):
    cng = mxc_runtime._Cng.__new__(mxc_runtime._Cng)
    cng._ctypes = ctypes
    cng._bcrypt = _FakeBcrypt(fail)
    cng._algorithm = ctypes.c_void_p(1)
    return cng


@pytest.mark.parametrize("size", SIZES)
def test_the_cng_chunk_loop_hashes_every_byte(tmp_path, monkeypatch, size):
    path = tmp_path / "wxc-exec.exe"
    path.write_bytes(_blob(size))
    cng = _fake_cng()
    monkeypatch.setattr(mxc_runtime, "_native_sha256", lambda: cng)
    assert mxc_runtime._sha256_file(path) == hashlib.sha256(_blob(size)).hexdigest()
    assert cng._bcrypt.destroyed == 1 and not cng._bcrypt.live


@pytest.mark.parametrize("fail", ["create", "data"])
def test_a_failing_cng_call_falls_back_to_hashlib_never_to_no_check(tmp_path, monkeypatch, fail):
    path = tmp_path / "wxc-exec.exe"
    path.write_bytes(_blob(CHUNK + 5))
    monkeypatch.setattr(mxc_runtime, "_native_sha256", lambda: _fake_cng(fail))
    assert mxc_runtime._sha256_file(path) == hashlib.sha256(_blob(CHUNK + 5)).hexdigest()


@pytest.mark.parametrize("size", SIZES)
def test_hashlib_path_is_unchanged(tmp_path, monkeypatch, size):
    path = tmp_path / "wxc-exec.exe"
    path.write_bytes(_blob(size))
    monkeypatch.setattr(mxc_runtime, "_native_sha256", lambda: None)
    assert mxc_runtime._sha256_file(path) == hashlib.sha256(_blob(size)).hexdigest()


def test_the_native_hash_is_windows_only_and_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_cng_state", [])
    if os.name != "nt":
        assert mxc_runtime._native_sha256() is None
    monkeypatch.setattr(mxc_runtime.os, "name", "nt")
    monkeypatch.setenv(mxc_runtime.NATIVE_HASH_ENV, "0")
    assert mxc_runtime._native_sha256() is None
    assert mxc_runtime._cng_state == []  # never even tried


def test_an_unloadable_cng_is_remembered_as_unavailable(monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_cng_state", [])
    monkeypatch.setattr(mxc_runtime.os, "name", "nt")
    monkeypatch.delenv(mxc_runtime.NATIVE_HASH_ENV, raising = False)
    tries = []

    def broken():
        tries.append(1)
        raise OSError("no bcrypt.dll")

    monkeypatch.setattr(mxc_runtime, "_Cng", broken)
    assert mxc_runtime._native_sha256() is None
    assert mxc_runtime._native_sha256() is None
    assert tries == [1]


def test_an_unreadable_file_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(mxc_runtime, "_native_sha256", lambda: _fake_cng())
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable):
        mxc_runtime._sha256_file(tmp_path / "missing.exe")


@pytest.mark.skipif(sys.platform != "win32", reason = "real Windows CNG")
@pytest.mark.parametrize("size", SIZES + [7_278_392])
def test_real_cng_matches_hashlib(tmp_path, monkeypatch, size):
    monkeypatch.setattr(mxc_runtime, "_cng_state", [])
    monkeypatch.delenv(mxc_runtime.NATIVE_HASH_ENV, raising = False)
    assert mxc_runtime._native_sha256() is not None, "CNG SHA-256 should load on Windows"
    data = _blob(size) if size < 8 * CHUNK else os.urandom(size)
    path = tmp_path / "wxc-exec.exe"
    path.write_bytes(data)
    assert mxc_runtime._sha256_file(path) == hashlib.sha256(data).hexdigest()
