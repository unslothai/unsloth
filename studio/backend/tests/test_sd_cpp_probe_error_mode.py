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
    state = {"value": 0x8000}

    class _Kernel32:
        def GetErrorMode(self):
            return state["value"]

        def SetErrorMode(self, value):
            previous, state["value"] = state["value"], value
            return previous

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda _name: _Kernel32(), raising = False)
    yield state
    # A failed test can leave a holder behind; the record is process-wide.
    subprocess_compat._ERROR_MODE.held.clear()


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


def _installer_and_backend_helpers():
    installer = bk._installer_module()
    # The installer loads its own copy of this module; the record must still be one per process.
    assert sys.modules["backend.utils.subprocess_compat"] is not subprocess_compat
    return (
        installer._windows_error_mode_suppressed,
        subprocess_compat.windows_loader_error_boxes_suppressed,
    )


def test_installer_leaving_first_keeps_the_backend_probes_bit(error_mode):
    installer_scope, backend_scope = _installer_and_backend_helpers()
    install = installer_scope()
    probe = backend_scope()
    install.__enter__()
    probe.__enter__()
    assert error_mode["value"] == 0x8003
    install.__exit__(None, None, None)
    # The install's crash-box flags go; the probe still holds SEM_FAILCRITICALERRORS.
    assert error_mode["value"] == 0x8001
    with backend_scope():
        assert error_mode["value"] == 0x8001
    probe.__exit__(None, None, None)
    assert error_mode["value"] == 0x8000


def test_backend_probe_leaving_first_keeps_the_installers_flags(error_mode):
    installer_scope, backend_scope = _installer_and_backend_helpers()
    probe = backend_scope()
    install = installer_scope()
    probe.__enter__()
    install.__enter__()
    probe.__exit__(None, None, None)
    assert error_mode["value"] == 0x8003
    install.__exit__(None, None, None)
    # Nothing left set for good, SEM_NOGPFAULTERRORBOX included.
    assert error_mode["value"] == 0x8000


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


_windows_only = pytest.mark.skipif(sys.platform != "win32", reason = "real kernel32")


def _child_error_mode() -> int:
    out = subprocess.run(
        [sys.executable, "-c", "import ctypes; print(ctypes.WinDLL('kernel32').GetErrorMode())"],
        capture_output = True,
        text = True,
        check = True,
        timeout = 60,
    )
    return int(out.stdout.strip())


@_windows_only
def test_a_real_child_inherits_the_bit_only_while_held():
    kernel32 = ctypes.WinDLL("kernel32")
    original = kernel32.GetErrorMode()
    kernel32.SetErrorMode(original & ~1)
    try:
        assert not _child_error_mode() & 1
        with subprocess_compat.windows_loader_error_boxes_suppressed():
            assert _child_error_mode() & 1
        assert not _child_error_mode() & 1
    finally:
        kernel32.SetErrorMode(original)


@_windows_only
def test_a_real_binary_missing_its_dll_is_reported_unrunnable(tmp_path, monkeypatch):
    # python.exe copied away from pythonXY.dll, PATH cut to System32: a loader failure like ROCm without HIP.
    import os
    import shutil

    binary = tmp_path / "sd-server.exe"
    shutil.copy(getattr(sys, "_base_executable", sys.executable), binary)
    monkeypatch.setenv("PATH", os.path.join(os.environ["SystemRoot"], "System32"))
    with subprocess_compat.windows_loader_error_boxes_suppressed():
        rc = subprocess.run([str(binary), "--help"], capture_output = True, timeout = 60).returncode
    assert rc & 0xFFFFFFFF == _STATUS_DLL_NOT_FOUND
    assert bk._server_binary_runnable(str(binary)) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
