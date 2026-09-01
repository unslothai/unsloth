# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.skipif(os.name != "nt", reason = "Windows ordinal comparison is required")
def test_tauri_root_classification_uses_windows_ordinal_case_semantics(monkeypatch):
    profile = Path(r"C:\Users\Straße")
    monkeypatch.setattr(gate, "_windows_profile_path", lambda: profile)

    assert gate.uses_tauri_managed_root(Path(r"C:\Users\straße\.unsloth\studio"))
    assert not gate.uses_tauri_managed_root(Path(r"C:\Users\Strasse\.unsloth\studio"))


def test_runtime_gate_handoff_is_one_shot(monkeypatch):
    monkeypatch.setenv(gate._RUNTIME_GATE_HANDOFF_ENV, "1")
    assert gate.consume_runtime_gate_handoff() is True
    assert gate.consume_runtime_gate_handoff() is False


def test_runtime_gate_acquire_is_one_shot(monkeypatch):
    monkeypatch.setenv(gate._RUNTIME_GATE_ACQUIRE_ENV, "1")
    assert gate.consume_runtime_gate_acquire() is True
    assert gate.consume_runtime_gate_acquire() is False


def test_terminal_launch_boundaries_use_the_runtime_gate():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    assert source.count("with _studio_runtime_launch_guard(") >= 5
    assert "runtime_gate_child_environment()" in source
    assert "runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()" in source


def test_terminal_update_holds_the_gate_through_environment_mutation():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    start = source.index("def update(")
    # To the next top-level def, rather than naming the one that happens to follow today.
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    consume = body.index("_studio_runtime_gate.consume_runtime_gate_handoff()")
    guard = body.index("with _studio_runtime_launch_guard(", consume)
    idle_scan = body.index("_studio_runtime_gate.ensure_managed_environment_is_idle", guard)
    # The launcher transaction wraps the mutation;
    launcher = body.index("_WindowsLauncherUpdateTransaction()", idle_scan)
    setup = body.index("_run_setup_script(", launcher)
    verify = body.index("_fail_if_install_damaged(", setup)
    assert consume < guard < idle_scan < launcher < setup < verify


def test_terminal_setup_holds_the_gate_through_environment_mutation():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    body = source[source.index("def setup(") : source.index("def _fail_if_install_damaged")]
    consume = body.index("_studio_runtime_gate.consume_runtime_gate_handoff()")
    guard = body.index("with _studio_runtime_launch_guard(", consume)
    idle_scan = body.index("_studio_runtime_gate.ensure_managed_environment_is_idle", guard)
    launcher = body.index("_WindowsLauncherUpdateTransaction()", idle_scan)
    setup = body.index("_run_setup_script(", launcher)
    assert consume < guard < idle_scan < launcher < setup


def test_terminal_stage_holds_the_gate_through_the_clone():
    source = STUDIO_COMMAND.read_text(encoding = "utf-8")
    body = source[source.index("def _stage_update(") : source.index("class _WindowsLauncher")]
    consume = body.index("_studio_runtime_gate.consume_runtime_gate_handoff()")
    guard = body.index("with _studio_runtime_launch_guard(", consume)
    stage = body.index("_studio_stage.stage(", guard)
    assert consume < guard < stage


@pytest.mark.skipif(os.name == "nt", reason = "POSIX orphan process semantics are required")
def test_staging_cli_keeps_the_gate_after_its_parent_exits(tmp_path):
    home = tmp_path / "studio"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    worker_pid = tmp_path / "worker.pid"
    log = tmp_path / "worker.log"
    worker = r"""
import sys
import time
from pathlib import Path
from unsloth_cli.commands import studio

home, ready, release = map(Path, sys.argv[1:4])
studio.STUDIO_HOME = home

def fake_stage(*args, **kwargs):
    (home / ".update-stage").mkdir(parents=True, exist_ok=True)
    ready.write_text("ready", encoding="utf-8")
    while not release.exists():
        time.sleep(0.02)
    return {"backend_version": "test", "root": str(home / ".update-stage")}

studio._studio_stage.stage = fake_stage
studio._stage_update(local=False, package="unsloth", verbose=False, verify=True)
"""
    parent = r"""
import os
import subprocess
import sys

log = open(sys.argv[6], "w", encoding="utf-8")
child = subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], *sys.argv[2:5]],
    start_new_session=True,
    stdout=log,
    stderr=subprocess.STDOUT,
)
with open(sys.argv[5], "w", encoding="utf-8") as pid_file:
    pid_file.write(str(child.pid))
os._exit(0)
"""
    env = os.environ.copy()
    env.pop(gate._RUNTIME_GATE_HANDOFF_ENV, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            parent,
            worker,
            str(home),
            str(ready),
            str(release),
            str(worker_pid),
            str(log),
        ],
        env = env,
        check = False,
    )
    assert result.returncode == 0

    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), log.read_text(encoding = "utf-8") if log.exists() else ""
        assert worker_pid.is_file()
        with pytest.raises(gate.StudioRuntimeGateBusy):
            with gate.studio_runtime_launch_guard(home):
                pass
    finally:
        release.touch()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with gate.studio_runtime_launch_guard(home):
                break
        except gate.StudioRuntimeGateBusy:
            time.sleep(0.02)
    else:
        pytest.fail(log.read_text(encoding = "utf-8") if log.exists() else "gate stayed busy")


@pytest.mark.skipif(os.name == "nt", reason = "POSIX flock is required")
def test_posix_runtime_gate_blocks_another_process_and_recovers(tmp_path):
    script = """
import sys
from pathlib import Path
from unsloth_cli import _studio_runtime_gate as gate
try:
    with gate.studio_runtime_launch_guard(Path(sys.argv[1])):
        pass
except gate.StudioRuntimeGateBusy:
    raise SystemExit(7)
"""
    with gate.studio_runtime_launch_guard(tmp_path) as acquired:
        assert acquired is True
        blocked = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check = False)
        assert blocked.returncode == 7

    recovered = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check = False)
    assert recovered.returncode == 0


@pytest.mark.skipif(os.name == "nt", reason = "POSIX flock is required")
def test_posix_runtime_gate_waits_for_parent_handoff(tmp_path):
    script = """
import sys
from pathlib import Path
from unsloth_cli import _studio_runtime_gate as gate
print("waiting", flush=True)
with gate.studio_runtime_launch_guard(Path(sys.argv[1]), wait=True):
    print("acquired", flush=True)
"""
    with gate.studio_runtime_launch_guard(tmp_path):
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout = subprocess.PIPE,
            text = True,
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "waiting"
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout = 0.2)

    stdout, _ = child.communicate(timeout = 5)
    assert child.returncode == 0
    assert stdout.strip() == "acquired"


def test_interrupted_windows_setup_kills_tree_before_return(monkeypatch):
    from unsloth_cli.commands import studio as studio_command

    events = []

    class InterruptedProcess:
        pid = 4242
        returncode = None

        def wait(self):
            events.append("wait")
            if self.returncode is None and events.count("wait") == 1:
                raise KeyboardInterrupt
            self.returncode = -1
            return self.returncode

        def poll(self):
            return self.returncode

    def fake_taskkill(argv, **kwargs):
        events.append(("taskkill", argv, kwargs))
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(studio_command.subprocess, "run", fake_taskkill)
    monkeypatch.setattr(studio_command, "_windows_hidden_subprocess_kwargs", lambda: {})

    with pytest.raises(KeyboardInterrupt):
        studio_command._wait_for_windows_setup_process(InterruptedProcess())

    assert events[0] == "wait"
    assert events[1][0] == "taskkill"
    assert events[1][1] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert events[1][2]["check"] is False
    assert events[2] == "wait"


@pytest.mark.skipif(os.name != "nt", reason = "Windows ordinal comparison is required")
def test_windows_path_containment_requires_component_boundaries():
    root = r"C:\Users\pc\.unsloth\studio\unsloth_studio"
    assert gate._windows_path_is_within(root + r"\Scripts\python.exe", root)
    assert not gate._windows_path_is_within(root + "_backup" + r"\python.exe", root)


@pytest.mark.skipif(os.name != "nt", reason = "Windows ordinal comparison is required")
def test_windows_path_containment_uses_ordinal_case_semantics():
    root = r"D:\Straße\studio"
    assert gate._windows_path_is_within(r"D:\straße\studio\python.exe", root)
    assert not gate._windows_path_is_within(r"D:\Strasse\studio\python.exe", root)


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_excludes_verified_launcher_and_blocks_another_managed_image(
    tmp_path, monkeypatch
):
    studio_home = tmp_path / "studio"
    managed_python = studio_home / "unsloth_studio" / "Scripts" / "python.exe"
    managed_launcher = studio_home / "unsloth_studio" / "Scripts" / "unsloth.exe"
    managed_python.parent.mkdir(parents = True)
    managed_python.write_bytes(b"MZ")
    managed_launcher.write_bytes(b"MZ")
    current_pid = os.getpid()
    parent_pid = current_pid + 1_000_000
    consumer_pid = parent_pid + 1
    payload = [
        {
            "ProcessId": current_pid,
            "ParentProcessId": parent_pid,
            "Name": "python.exe",
            "ExecutablePath": str(managed_python),
        },
        {
            "ProcessId": parent_pid,
            "ParentProcessId": 0,
            "Name": "unsloth.exe",
            "ExecutablePath": str(managed_launcher),
        },
    ]
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
    payload.append(
        {
            "ProcessId": consumer_pid,
            "ParentProcessId": 0,
            "Name": "python.exe",
            "ExecutablePath": str(managed_python),
        }
    )
    with pytest.raises(RuntimeError, match = rf"PID {consumer_pid}"):
        gate.ensure_managed_environment_is_idle(studio_home)


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_excludes_the_venv_python_redirector(tmp_path, monkeypatch):
    # install.ps1 runs `Scripts\unsloth.exe studio setup` and Tauri runs the venv interpreter, so both arrive through
    studio_home = tmp_path / "studio"
    scripts = studio_home / "unsloth_studio" / "Scripts"
    managed_python = scripts / "python.exe"
    managed_launcher = scripts / "unsloth.exe"
    scripts.mkdir(parents = True)
    managed_python.write_bytes(b"MZ")
    managed_launcher.write_bytes(b"MZ")
    current_pid = os.getpid()
    redirector_pid = current_pid + 1_075_000
    launcher_pid = redirector_pid + 1
    payload = [
        {
            "ProcessId": current_pid,
            "ParentProcessId": redirector_pid,
            "Name": "python.exe",
            "ExecutablePath": str(tmp_path / "base" / "python.exe"),
        },
        {
            "ProcessId": redirector_pid,
            "ParentProcessId": launcher_pid,
            "Name": "python.exe",
            "ExecutablePath": str(managed_python),
        },
        {
            "ProcessId": launcher_pid,
            "ParentProcessId": 0,
            "Name": "unsloth.exe",
            "ExecutablePath": str(managed_launcher),
        },
    ]
    monkeypatch.setattr(sys, "executable", str(managed_python))
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

    # Tauri runs the redirector directly, with no shim above it.
    payload[2]["ExecutablePath"] = str(tmp_path / "Unsloth" / "unsloth.exe")
    gate.ensure_managed_environment_is_idle(studio_home)

    # Only the direct parent is the redirector;
    payload[2]["ExecutablePath"] = str(managed_python)
    with pytest.raises(RuntimeError, match = rf"PID {launcher_pid}"):
        gate.ensure_managed_environment_is_idle(studio_home)

    # We were not launched through the venv, so nothing above us is a redirector.
    payload[2]["ExecutablePath"] = str(managed_launcher)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "base" / "python.exe"))
    with pytest.raises(RuntimeError, match = rf"PID {redirector_pid}"):
        gate.ensure_managed_environment_is_idle(studio_home)


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_does_not_exclude_managed_parent_of_updater(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    scripts = studio_home / "unsloth_studio" / "Scripts"
    managed_python = scripts / "python.exe"
    managed_launcher = scripts / "unsloth.exe"
    scripts.mkdir(parents = True)
    managed_python.write_bytes(b"MZ")
    managed_launcher.write_bytes(b"MZ")
    current_pid = os.getpid()
    launcher_pid = current_pid + 1_050_000
    managed_parent_pid = launcher_pid + 1
    payload = [
        {
            "ProcessId": current_pid,
            "ParentProcessId": launcher_pid,
            "Name": "python.exe",
            "ExecutablePath": str(managed_python),
        },
        {
            "ProcessId": launcher_pid,
            "ParentProcessId": managed_parent_pid,
            "Name": "unsloth.exe",
            "ExecutablePath": str(managed_launcher),
        },
        {
            "ProcessId": managed_parent_pid,
            "ParentProcessId": 0,
            "Name": "python.exe",
            "ExecutablePath": str(managed_python),
        },
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(payload),
            stderr = "",
        ),
    )

    with pytest.raises(RuntimeError, match = rf"PID {managed_parent_pid}"):
        gate.ensure_managed_environment_is_idle(studio_home)


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_blocks_exact_outer_shim(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    outer_shim = studio_home / "bin" / "unsloth.exe"
    outer_shim.parent.mkdir(parents = True)
    outer_shim.write_bytes(b"MZ")
    consumer_pid = os.getpid() + 1_100_000
    payload = [
        {
            "ProcessId": os.getpid(),
            "ParentProcessId": 0,
            "Name": "python.exe",
            "ExecutablePath": str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
        },
        {
            "ProcessId": consumer_pid,
            "ParentProcessId": 0,
            "Name": "unsloth.exe",
            "ExecutablePath": str(outer_shim),
        },
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 0,
            stdout = json.dumps(payload),
            stderr = "",
        ),
    )
    with pytest.raises(RuntimeError, match = rf"PID {consumer_pid}"):
        gate.ensure_managed_environment_is_idle(studio_home)


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_ignores_command_line_only_path_mentions(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    mentioned = studio_home / "unsloth_studio" / "Lib" / "worker.py"
    payload = [
        {
            "ProcessId": os.getpid(),
            "ParentProcessId": 0,
            "Name": "python.exe",
            "ExecutablePath": str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"),
            "CommandLine": f'python.exe "{mentioned}"',
        }
    ]
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode = 0, stdout = json.dumps(payload), stderr = "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    gate.ensure_managed_environment_is_idle(studio_home)
    assert "CommandLine" not in " ".join(captured["command"])


@pytest.mark.skipif(os.name != "nt", reason = "Windows process inspection is required")
def test_idle_scan_fails_closed_when_process_inventory_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode = 1,
            stdout = "",
            stderr = "access denied",
        ),
    )
    with pytest.raises(RuntimeError, match = "access denied"):
        gate.ensure_managed_environment_is_idle(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason = "Windows named mutexes are required")
def test_runtime_gate_blocks_another_thread_and_recovers():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    managed_root = gate._windows_profile_path() / ".unsloth" / "studio"
    holder = kernel32.CreateMutexW(
        None,
        True,
        gate.runtime_mutex_name_for_sid(gate._current_windows_user_sid()),
    )
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
    assert observed == ["blocked"]
    assert kernel32.ReleaseMutex(holder)
    assert kernel32.CloseHandle(holder)
    with gate.studio_runtime_launch_guard(managed_root) as acquired:
        assert acquired is True


@pytest.mark.skipif(os.name != "nt", reason = "Windows named mutexes are required")
def test_custom_root_runtime_gate_blocks_another_thread(tmp_path):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    holder = kernel32.CreateMutexW(None, True, gate.runtime_mutex_name_for_studio_home(tmp_path))
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
    assert observed == ["blocked"]
    assert kernel32.ReleaseMutex(holder)
    assert kernel32.CloseHandle(holder)
