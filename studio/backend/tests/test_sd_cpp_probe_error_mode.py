# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp test-launches run with SEM_FAILCRITICALERRORS, so a missing DLL exits instead of opening a box."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import types

import pytest

from core.inference import sd_cpp_backend as bk
from core.inference import sd_cpp_engine as eng
from utils import subprocess_compat

_STATUS_DLL_NOT_FOUND = 0xC0000135


@pytest.fixture
def error_mode(monkeypatch):
    """A fake kernel32 on win32; ``state["value"]`` is the process error mode."""
    state = {"value": 0x8000, "sets": 0}

    class _Kernel32:
        def GetErrorMode(self):
            return state["value"]

        def SetErrorMode(self, value):
            previous, state["value"] = state["value"], value
            state["sets"] += 1
            return previous

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda _name: _Kernel32(), raising = False)
    return state


def _record_mode(
    monkeypatch,
    state,
    returncode = _STATUS_DLL_NOT_FOUND,
):
    seen = []

    def fake_run(*_args, **_kwargs):
        seen.append(state["value"])
        return types.SimpleNamespace(returncode = returncode, stdout = "", stderr = "")

    # The backend probes import subprocess inside the function; the engine at module level.
    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_the_bit_is_or_ed_in_and_the_previous_mode_restored(error_mode):
    with subprocess_compat.windows_loader_error_boxes_suppressed():
        assert error_mode["value"] == 0x8001
    assert error_mode["value"] == 0x8000


def test_nested_holders_keep_the_bit_until_the_last_one_leaves(error_mode):
    outer = subprocess_compat.windows_loader_error_boxes_suppressed()
    inner = subprocess_compat.windows_loader_error_boxes_suppressed()
    outer.__enter__()
    inner.__enter__()
    outer.__exit__(None, None, None)
    # A plain save/restore would clear the bit here while the inner probe is still spawning.
    assert error_mode["value"] == 0x8001
    inner.__exit__(None, None, None)
    assert error_mode["value"] == 0x8000
    assert error_mode["sets"] == 2


def test_concurrent_holders_never_see_the_bit_cleared(error_mode):
    barrier = threading.Barrier(8)
    cleared = []

    def hold():
        with subprocess_compat.windows_loader_error_boxes_suppressed():
            barrier.wait()
            cleared.append(not error_mode["value"] & 1)

    threads = [threading.Thread(target = hold) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cleared == [False] * 8
    assert error_mode["value"] == 0x8000


def test_the_mode_is_restored_when_the_launch_raises(error_mode):
    with pytest.raises(subprocess.TimeoutExpired):
        with subprocess_compat.windows_loader_error_boxes_suppressed():
            raise subprocess.TimeoutExpired("sd-server", 20)
    assert error_mode["value"] == 0x8000


def test_off_windows_the_mode_is_untouched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda _name: pytest.fail("kernel32 loaded off Windows"), raising = False
    )
    with subprocess_compat.windows_loader_error_boxes_suppressed():
        pass


def test_server_runnable_probe_launches_with_the_bit_and_reads_dll_not_found(
    error_mode, monkeypatch
):
    seen = _record_mode(monkeypatch, error_mode)
    assert bk._server_binary_runnable("sd-server.exe") is False
    assert seen == [0x8001]
    assert error_mode["value"] == 0x8000


def test_probe_output_launches_with_the_bit(error_mode, monkeypatch):
    seen = _record_mode(monkeypatch, error_mode)
    assert bk._sd_cpp_probe_output("sd-cli.exe", "--help") is None
    assert seen == [0x8001]
    assert error_mode["value"] == 0x8000


def test_engine_identity_probe_launches_with_the_bit(error_mode, monkeypatch, tmp_path):
    binary = tmp_path / "sd-cli.exe"
    binary.write_bytes(b"stub")
    seen = _record_mode(monkeypatch, error_mode)
    assert eng.sd_cpp_binary_identifies(str(binary)) is False
    assert seen == [0x8001]
    assert error_mode["value"] == 0x8000


def test_engine_version_probe_launches_with_the_bit(error_mode, monkeypatch, tmp_path):
    binary = tmp_path / "sd-cli.exe"
    binary.write_bytes(b"stub")
    seen = _record_mode(monkeypatch, error_mode)
    engine = eng.SdCppEngine(binary = str(binary))
    monkeypatch.setattr(engine, "is_available", lambda: True)
    assert engine.version() is None
    assert seen == [0x8001]
    assert error_mode["value"] == 0x8000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
