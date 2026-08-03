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
    outer_shim = studio_home / "bin" / "unsloth.exe"
    inner_shim = studio_home / "unsloth_studio" / "Scripts" / "unsloth.exe"
    for shim in (outer_shim, inner_shim):
        shim.parent.mkdir(parents = True, exist_ok = True)
        shim.write_bytes(b"MZ")
    parent_pid = os.getppid()
    grandparent_pid = parent_pid + 1_000_000
    base_process = {
        "ParentProcessId": parent_pid,
        "Name": "python.exe",
        "ExecutablePath": str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
        "CommandLine": r"python.exe worker.py",
    }

    payload = [
        dict(base_process, ProcessId = -1),
        dict(base_process, ProcessId = os.getpid()),
        dict(
            base_process,
            ProcessId = parent_pid,
            ParentProcessId = grandparent_pid,
            Name = "unsloth.exe",
            ExecutablePath = "",
            CommandLine = "unsloth studio update",
        ),
        dict(
            base_process,
            ProcessId = grandparent_pid,
            ParentProcessId = 0,
            Name = "unsloth.exe",
            ExecutablePath = "",
            CommandLine = "unsloth studio update",
        ),
    ]
    grandparent_process = SimpleNamespace(
        pid = grandparent_pid,
        exe = lambda: str(outer_shim),
        parent = lambda: None,
    )
    parent_process = SimpleNamespace(
        pid = parent_pid,
        exe = lambda: str(inner_shim),
        parent = lambda: grandparent_process,
    )

    def fake_process(process_id):
        if process_id <= 0:
            raise ValueError(f"pid must be positive: {process_id}")
        return SimpleNamespace(
            cwd = lambda: str(worker.parent),
            parent = lambda: parent_process if process_id == os.getpid() else None,
        )

    fake_psutil = SimpleNamespace(
        Error = Exception,
        Process = fake_process,
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


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_terminal_update_only_excludes_its_verified_python_redirector(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    scripts_dir = studio_home / "unsloth_studio" / "Scripts"
    managed_python = scripts_dir / "python.exe"
    outer_shim = studio_home / "bin" / "unsloth.exe"
    worker = scripts_dir / "worker.py"
    for executable in (managed_python, outer_shim):
        executable.parent.mkdir(parents = True, exist_ok = True)
        executable.write_bytes(b"MZ")
    worker.write_text("pass", encoding = "utf-8")

    redirector_pid = os.getpid() + 1_100_000
    shim_pid = redirector_pid + 1
    shell_pid = shim_pid + 1
    base_process = {
        "Name": "python.exe",
        "ExecutablePath": str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
        "CommandLine": r"python.exe worker.py",
    }
    redirector_command = f'"{managed_python}" "{outer_shim}" studio update --local'
    payload = [
        dict(
            base_process,
            ProcessId = os.getpid(),
            ParentProcessId = redirector_pid,
        ),
        dict(
            base_process,
            ProcessId = redirector_pid,
            ParentProcessId = shim_pid,
            ExecutablePath = str(managed_python),
            CommandLine = redirector_command,
        ),
        dict(
            base_process,
            ProcessId = shim_pid,
            ParentProcessId = shell_pid,
            Name = "unsloth.exe",
            ExecutablePath = str(outer_shim),
            CommandLine = "unsloth studio update --local",
        ),
        dict(
            base_process,
            ProcessId = shell_pid,
            ParentProcessId = 0,
            Name = "bash.exe",
        ),
    ]
    shell_process = SimpleNamespace(
        pid = shell_pid,
        exe = lambda: str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
        parent = lambda: None,
    )
    shim_process = SimpleNamespace(
        pid = shim_pid,
        exe = lambda: str(outer_shim),
        parent = lambda: shell_process,
    )
    redirector_process = SimpleNamespace(
        pid = redirector_pid,
        exe = lambda: str(managed_python),
        parent = lambda: shim_process,
    )

    def fake_process(process_id):
        return SimpleNamespace(
            cwd = lambda: str(tmp_path),
            parent = lambda: redirector_process if process_id == os.getpid() else None,
        )

    fake_psutil = SimpleNamespace(Error = Exception, Process = fake_process)
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

    payload[1]["CommandLine"] = f'"{managed_python}" "{outer_shim}" studio --api-only'
    with pytest.raises(RuntimeError, match = "managed Studio environment is in use"):
        gate.ensure_managed_environment_is_idle(studio_home)

    payload[1]["CommandLine"] = f'"{managed_python}" "{worker}" studio update --local'
    with pytest.raises(RuntimeError, match = "managed Studio environment is in use"):
        gate.ensure_managed_environment_is_idle(studio_home)

    payload[1]["CommandLine"] = redirector_command
    payload.append(
        dict(
            base_process,
            ProcessId = shell_pid + 100,
            ParentProcessId = 0,
            ExecutablePath = str(managed_python),
            CommandLine = redirector_command,
        )
    )
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
