# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for terminal Studio admission to the Windows runtime gate."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from ctypes import wintypes
from pathlib import Path

import pytest

from unsloth_cli import _studio_runtime_gate as gate


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"


def test_runtime_mutex_name_matches_installer_and_tauri():
    sid = "S-1-5-21-111-222-333-1001"
    expected = f"Global\\UnslothStudioManagedEnvironment-{sid}"
    assert gate.runtime_mutex_name_for_sid(sid) == expected

    install_source = (REPO_ROOT / "install.ps1").read_text(encoding = "utf-8")
    rust_source = (REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs").read_text(
        encoding = "utf-8"
    )
    assert '"Global\\UnslothStudioManagedEnvironment-$Sid"' in install_source
    assert '"Global\\\\UnslothStudioManagedEnvironment-"' in rust_source


def test_custom_root_mutex_name_matches_installer_hash(monkeypatch):
    root = Path(r"C:\\custom\\studio")
    canonical = r"C:\\custom\\studio"
    monkeypatch.setattr(gate, "uses_tauri_managed_root", lambda _path: False)
    monkeypatch.setattr(gate, "_resolved_windows_path", lambda _path: canonical)
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert gate.runtime_mutex_name_for_studio_home(root) == (
        f"Global\\UnslothStudioManagedEnvironmentPath-{expected_hash}"
    )
    install_source = (REPO_ROOT / "install.ps1").read_text(encoding = "utf-8")
    assert (
        '"Global\\UnslothStudioManagedEnvironmentPath-$(Get-StudioRuntimePathHash -Path $Path)"'
        in install_source
    )


def test_runtime_gate_handoff_is_one_shot(monkeypatch):
    monkeypatch.setenv(gate._RUNTIME_GATE_HANDOFF_ENV, "1")
    assert gate.consume_runtime_gate_handoff() is True
    assert gate.consume_runtime_gate_handoff() is False


def test_terminal_launch_boundaries_use_the_runtime_gate():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    assert source.count("with _studio_runtime_launch_guard(") >= 4
    assert "runtime_gate_child_environment()" in source
    assert "runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()" in source


def test_terminal_update_holds_the_gate_through_environment_mutation():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    body = source[source.index("def update(") : source.index("def _release_self_exe_lock_windows")]
    consume = body.index("_studio_runtime_gate.consume_runtime_gate_handoff()")
    guard = body.index("with _studio_runtime_launch_guard(", consume)
    idle_scan = body.index("_studio_runtime_gate.ensure_managed_environment_is_idle", guard)
    release_self = body.index("_release_self_exe_lock_windows()", idle_scan)
    setup = body.index("_run_setup_script(", release_self)
    verify = body.index("_fail_if_install_damaged()", setup)
    shortcuts = body.index("_refresh_desktop_shortcuts(", verify)
    assert consume < guard < idle_scan < release_self < setup < verify < shortcuts

    update_source = (REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs").read_text(
        encoding = "utf-8"
    )
    handoff = update_source.index("STUDIO_RUNTIME_GATE_HANDOFF_ENV")
    spawn = update_source.index(".spawn()", handoff)
    assert handoff < spawn


def test_command_line_path_matching_requires_component_boundaries():
    root = r"C:\Users\pc\.unsloth\studio\unsloth_studio"
    assert gate._command_line_references_windows_path(rf'python.exe "{root}\Lib\worker.py"', root)
    assert not gate._command_line_references_windows_path(
        rf'python.exe "X{root}\Lib\worker.py"',
        root,
    )
    assert not gate._command_line_references_windows_path(
        rf'python.exe "{root}_backup\Lib\worker.py"',
        root,
    )


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_terminal_update_idle_scan_excludes_self_and_blocks_another_consumer(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    worker = studio_home / "unsloth_studio" / "Lib" / "worker.py"
    worker.parent.mkdir(parents = True)
    worker.write_text("pass", encoding = "utf-8")
    base_process = {
        "ParentProcessId": os.getppid(),
        "Name": "python.exe",
        "ExecutablePath": str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
        "CommandLine": r"python.exe worker.py",
    }

    payload = [dict(base_process, ProcessId = os.getpid())]
    fake_psutil = SimpleNamespace(
        Error = Exception,
        Process = lambda _process_id: SimpleNamespace(cwd = lambda: str(worker.parent)),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(payload),
            stderr = "",
        ),
    )
    gate.ensure_managed_environment_is_idle(studio_home)

    payload.append(dict(base_process, ProcessId = os.getpid() + 100000))
    with pytest.raises(RuntimeError, match = "managed Studio environment is in use"):
        gate.ensure_managed_environment_is_idle(studio_home)


@pytest.mark.skipif(os.name != "nt", reason = "Windows named mutexes are required")
def test_runtime_gate_blocks_another_thread_and_recovers():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    managed_root = gate._windows_profile_path() / ".unsloth" / "studio"
    name = gate.runtime_mutex_name_for_sid(gate._current_windows_user_sid())
    holder = kernel32.CreateMutexW(None, True, name)
    assert holder

    observed: list[str] = []

    def contend() -> None:
        try:
            with gate.studio_runtime_launch_guard(managed_root):
                observed.append("acquired")
        except gate.StudioRuntimeGateBusy:
            observed.append("blocked")

    contender = threading.Thread(target = contend)
    contender.start()
    contender.join(timeout = 10)
    assert not contender.is_alive()
    assert observed == ["blocked"]

    assert kernel32.ReleaseMutex(holder)
    assert kernel32.CloseHandle(holder)

    with gate.studio_runtime_launch_guard(managed_root) as acquired:
        assert acquired is True


@pytest.mark.skipif(os.name != "nt", reason = "Windows named mutexes are required")
def test_custom_root_runtime_gate_blocks_another_thread(tmp_path):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    name = gate.runtime_mutex_name_for_studio_home(tmp_path)
    holder = kernel32.CreateMutexW(None, True, name)
    assert holder
    observed: list[str] = []

    def contend() -> None:
        try:
            with gate.studio_runtime_launch_guard(tmp_path):
                observed.append("acquired")
        except gate.StudioRuntimeGateBusy:
            observed.append("blocked")

    contender = threading.Thread(target = contend)
    contender.start()
    contender.join(timeout = 10)
    assert not contender.is_alive()
    assert observed == ["blocked"]
    assert kernel32.ReleaseMutex(holder)
    assert kernel32.CloseHandle(holder)
