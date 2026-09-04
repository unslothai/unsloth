# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused contract and live-enforcement tests for Studio's native OS sandbox."""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.inference import os_sandbox
from core.inference import tool_isolation
from core.inference import tools as inference_tools


def _spec(workdir: Path, *argv: str) -> os_sandbox.ToolLaunchPlan:
    return os_sandbox.ToolLaunchPlan(
        argv = tuple(argv) or (sys.executable, "-c", "pass"),
        workdir = str(workdir),
        env = {
            "HOME": str(workdir),
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONIOENCODING": "utf-8",
            "TMPDIR": "/tmp",
        },
    )


@pytest.fixture
def isolated_capability_cache():
    before = dict(os_sandbox._capability_cache)
    os_sandbox._capability_cache.clear()
    try:
        yield
    finally:
        os_sandbox._capability_cache.clear()
        os_sandbox._capability_cache.update(before)


class _RecordingBackend:
    identity = "test-recording-backend"
    profile_id = "test-recording-profile-v1"

    def __init__(self, capabilities: list[os_sandbox.SandboxCapability] | None = None):
        self.capabilities = capabilities or [
            os_sandbox.SandboxCapability(
                self.identity, True, "qualified", profile_id = self.profile_id
            )
        ]
        self.probe_calls = 0
        self.prepared_specs: list[os_sandbox.ToolLaunchPlan] = []

    def probe(self) -> os_sandbox.SandboxCapability:
        result = self.capabilities[min(self.probe_calls, len(self.capabilities) - 1)]
        self.probe_calls += 1
        return result

    def prepare(self, spec: os_sandbox.ToolLaunchPlan) -> os_sandbox.PreparedSandboxLaunch:
        self.prepared_specs.append(spec)
        return os_sandbox.PreparedSandboxLaunch(
            argv = ("native-sandbox", *spec.argv),
            workdir = spec.workdir,
            env = dict(spec.env),
            preexec_fn = spec.preexec_fn,
            backend = self.identity,
        )


def test_backend_neutral_launch_spec_is_canonicalized_and_prepared(
    monkeypatch, tmp_path, isolated_capability_cache
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend = _RecordingBackend()
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)

    spec = _spec(workdir / ".." / "work", sys.executable, "-c", "print('ok')")
    prepared = os_sandbox.prepare_tool_launch(spec)

    assert backend.probe_calls == 1
    assert len(backend.prepared_specs) == 1
    assert backend.prepared_specs[0].argv == spec.argv
    assert backend.prepared_specs[0].workdir == os.path.realpath(workdir)
    assert backend.prepared_specs[0].env == spec.env
    assert prepared.backend == backend.identity
    assert prepared.argv[:2] == ("native-sandbox", sys.executable)


def test_prepare_rejects_empty_inner_argv(tmp_path):
    with pytest.raises(ValueError, match = "argv must not be empty"):
        os_sandbox.prepare_tool_launch(
            os_sandbox.ToolLaunchPlan(argv = (), workdir = str(tmp_path), env = {})
        )


def test_backend_prepare_error_is_fail_closed(monkeypatch, tmp_path, isolated_capability_cache):
    backend = _RecordingBackend()
    backend.prepare = Mock(side_effect = OSError("preparation exploded"))
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)

    with pytest.raises(
        os_sandbox.SandboxUnavailableError,
        match = "test-recording-backend could not prepare the process: preparation exploded",
    ):
        os_sandbox.prepare_tool_launch(_spec(tmp_path))


def test_unsupported_platform_fails_closed(monkeypatch, tmp_path, isolated_capability_cache):
    monkeypatch.setattr(os_sandbox.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)

    capability = os_sandbox.sandbox_capability()
    assert not capability.qualified
    assert capability.backend == "unsupported-win32"
    assert "Limited mode" in capability.remediation
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "unsupported on win32"):
        os_sandbox.prepare_tool_launch(_spec(tmp_path))


def test_available_preview_backend_serves_required_mode_and_records_limitations(
    monkeypatch, tmp_path, isolated_capability_cache
):
    backend = _RecordingBackend(
        [
            os_sandbox.SandboxCapability(
                _RecordingBackend.identity,
                False,
                "preview policy passed",
                available = True,
                protection_state = "preview",
                profile_id = _RecordingBackend.profile_id,
                limitations = ("lifecycle_unverified",),
            )
        ]
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)

    prepared = os_sandbox.prepare_tool_launch(_spec(tmp_path))

    assert prepared.execution_record is not None
    assert prepared.execution_record.os_isolation is True
    assert prepared.execution_record.profile_id == backend.profile_id
    assert prepared.execution_record.limitations == ("lifecycle_unverified",)


def test_spawn_and_cleanup_are_backend_owned_and_lifo(tmp_path):
    events: list[str] = []

    def spawn(_prepared, kwargs):
        events.append(f"spawn:{kwargs['text']}")
        return "process-adapter"

    def failed_cleanup():
        events.append("cleanup-second")
        raise OSError("cleanup failed")

    prepared = os_sandbox.PreparedSandboxLaunch(
        argv = ("opaque",),
        workdir = str(tmp_path),
        env = {},
        preexec_fn = None,
        backend = "custom",
        spawn_callback = spawn,
        cleanup_callbacks = [lambda: events.append("cleanup-first"), failed_cleanup],
    )

    assert os_sandbox.spawn_prepared_launch(prepared, text = True) == "process-adapter"
    prepared.cleanup()

    assert events == ["spawn:True", "cleanup-second", "cleanup-first"]
    assert prepared.cleanup_diagnostics == ["OSError: cleanup failed"]


def test_full_mode_keeps_lifecycle_plan_without_claiming_os_isolation(
    monkeypatch, tmp_path, isolated_capability_cache
):
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    plan = os_sandbox.replace(_spec(tmp_path), requested_mode = "full")

    prepared = os_sandbox.prepare_tool_launch(plan)

    assert prepared.argv == plan.argv
    assert prepared.execution_record is not None
    assert prepared.execution_record.effective_mode == "full"
    assert prepared.execution_record.os_isolation is False
    assert prepared.execution_record.backend == "none"


def test_limited_mode_requires_current_generation_bound_grant(
    monkeypatch, tmp_path, isolated_capability_cache
):
    store = tool_isolation.LimitedGrantStore(ttl_seconds = 60, max_entries = 4)
    monkeypatch.setattr(tool_isolation, "_LIMITED_GRANTS", store)
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    fingerprint = {"value": "environment-a"}
    monkeypatch.setattr(
        os_sandbox, "_environment_fingerprint", lambda _backend: fingerprint["value"]
    )
    capability = os_sandbox.capability_snapshot()
    grant = tool_isolation.issue_limited_grant(
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        probe_generation = capability.probe_generation,
    )
    plan = os_sandbox.replace(
        _spec(tmp_path),
        requested_mode = "limited",
        current_subject = "actor-a",
        tool_ui_session_id = "page-a",
        limited_grant = grant.token,
    )

    prepared = os_sandbox.prepare_tool_launch(plan)

    assert prepared.execution_record is not None
    assert prepared.execution_record.effective_mode == "limited"
    assert prepared.execution_record.os_isolation is False
    assert "process_guard" in prepared.execution_record.retained_safeguards

    fingerprint["value"] = "environment-b"
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "capability changed"):
        os_sandbox.prepare_tool_launch(plan)


@pytest.mark.parametrize("qualified", [False, True])
def test_definitive_probe_result_is_cached(qualified, monkeypatch, isolated_capability_cache):
    backend = _RecordingBackend(
        [os_sandbox.SandboxCapability("test-recording-backend", qualified, "definitive")]
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)

    first = os_sandbox.sandbox_capability()
    second = os_sandbox.sandbox_capability()

    assert first is second
    assert first.qualified is qualified
    assert backend.probe_calls == 1


def test_transient_probe_failure_is_retried_then_success_is_cached(
    monkeypatch, isolated_capability_cache
):
    backend = _RecordingBackend(
        [
            os_sandbox.SandboxCapability(
                "test-recording-backend", False, "temporarily unavailable", transient = True
            ),
            os_sandbox.SandboxCapability("test-recording-backend", True, "qualified"),
        ]
    )
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)

    assert not os_sandbox.sandbox_capability().qualified
    assert os_sandbox.sandbox_capability().qualified
    assert os_sandbox.sandbox_capability().qualified
    assert backend.probe_calls == 2


@pytest.mark.parametrize(
    ("environment", "expected"),
    [("native_linux", "protected"), ("wsl2", "preview"), ("container", "preview")],
)
def test_qualified_capability_label_depends_on_environment(
    environment, expected, monkeypatch, isolated_capability_cache
):
    backend = _RecordingBackend()
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)
    monkeypatch.setattr(os_sandbox, "_environment_class", lambda: environment)
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "fingerprint")

    capability = os_sandbox.capability_snapshot()

    assert capability.protection_state == expected
    assert capability.environment == environment
    assert capability.profile_id == backend.profile_id
    assert capability.probe_generation


def test_mount_topology_changes_environment_fingerprint(monkeypatch):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_environment_class", lambda: "container")
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda _name: None)
    mounts = [os_sandbox._LinuxMount("1", "0", "0:1", "/", "/", "rw", "overlay", "overlay", "rw")]
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: tuple(mounts))

    before = os_sandbox._environment_fingerprint(None)
    mounts.append(
        os_sandbox._LinuxMount("2", "1", "0:2", "/", "/run/secrets", "ro", "tmpfs", "tmpfs", "ro")
    )

    assert os_sandbox._environment_fingerprint(None) != before


@pytest.mark.parametrize("field", ["close_fds", "terminate_descendants"])
def test_prepare_rejects_weakened_launch_lifecycle(field, tmp_path):
    plan = os_sandbox.replace(_spec(tmp_path), **{field: False})

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "descriptors.*descendant"):
        os_sandbox.prepare_tool_launch(plan)


def _mount_sources(argv: tuple[str, ...], option: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv[:-2]) if value == option]


def test_linux_bubblewrap_argv_exposes_only_selected_read_roots_and_workdir(monkeypatch, tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    identity = tmp_path / "identity"
    identity.mkdir()
    passwd = identity / "passwd"
    group = identity / "group"
    passwd.write_text("studio:x:1:1::/nonexistent:/bin/sh\n", encoding = "utf-8")
    group.write_text("studio:x:1:\n", encoding = "utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (str(runtime),))
    monkeypatch.setattr(
        os_sandbox,
        "_identity_files",
        lambda: (str(identity), str(passwd), str(group)),
    )
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    prepared = backend.prepare(_spec(workdir, sys.executable, "-c", "pass"))
    argv = prepared.argv

    assert argv[0] == "/usr/bin/bwrap"
    assert "--die-with-parent" in argv
    assert "--new-session" in argv
    assert "--unshare-all" in argv
    assert "--unshare-user" in argv
    assert "--disable-userns" in argv
    assert "os.execvpe" in argv[argv.index("-c") + 1]
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--seccomp") + 1] == str(prepared.pass_fds[0])
    assert ("--proc", "/proc") == argv[argv.index("--proc") : argv.index("--proc") + 2]
    assert ("--dev", "/dev") == argv[argv.index("--dev") : argv.index("--dev") + 2]
    assert ("--remount-ro", "/") == argv[
        argv.index("--remount-ro") : argv.index("--remount-ro") + 2
    ]
    assert argv.index("--remount-ro") < argv.index("--tmpfs")
    assert "/tmp" in _mount_sources(argv, "--tmpfs")
    assert "/dev/shm" in _mount_sources(argv, "--tmpfs")
    assert _mount_sources(argv, "--bind") == [os.path.realpath(workdir)]
    assert os.path.realpath(runtime) in _mount_sources(argv, "--ro-bind")
    assert "/run" not in argv
    assert "/var/run" not in argv
    assert str(tmp_path) not in argv
    assert str(Path.home()) not in argv
    assert prepared.env["HOME"] == os.path.realpath(workdir)
    assert prepared.env["TMPDIR"] == "/tmp"


def test_linux_nested_mount_beneath_exposed_root_is_masked(monkeypatch, tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    runtime = tmp_path / "runtime"
    nested = runtime / "foreign"
    nested.mkdir(parents = True)
    identity = tmp_path / "identity"
    identity.mkdir()
    passwd = identity / "passwd"
    group = identity / "group"
    passwd.touch()
    group.touch()
    mount = os_sandbox._LinuxMount("2", "1", "0:2", "/", str(nested), "rw", "9p", "drvfs", "rw")
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (str(runtime),))
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: (mount,))
    monkeypatch.setattr(os_sandbox, "_validate_runtime_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(
        os_sandbox, "_identity_files", lambda: (str(identity), str(passwd), str(group))
    )
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    prepared = backend.prepare(_spec(workdir))
    try:
        index = prepared.argv.index(str(nested))
        assert prepared.argv[index - 1] == "--tmpfs"
    finally:
        prepared.cleanup()


def test_wsl_workdir_on_drvfs_is_ineligible(monkeypatch, tmp_path):
    mount = os_sandbox._LinuxMount("2", "1", "0:2", "/", str(tmp_path), "rw", "9p", "drvfs", "rw")
    monkeypatch.setattr(os_sandbox, "_linux_environment", lambda: "wsl2")
    monkeypatch.setattr(os_sandbox, "_linux_mount_for_path", lambda _path: mount)

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Linux filesystem"):
        os_sandbox._validate_linux_workdir_environment(str(tmp_path))


def test_wsl_interop_paths_and_environment_are_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(os_sandbox.os, "pathsep", ":")
    env = {
        "PATH": "/usr/bin:/mnt/c/Windows/System32:/usr/lib/wsl/lib",
        "WSL_INTEROP": "/run/WSL/1_interop",
        "WSLENV": "TOKEN/u",
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "PULSE_SERVER": "unix:/mnt/wslg/PulseServer",
        "XDG_RUNTIME_DIR": "/mnt/wslg/runtime-dir",
    }
    sanitized = os_sandbox._sanitize_linux_environment(env, "wsl2")

    assert sanitized["PATH"] == "/usr/bin"
    assert sanitized["XDG_RUNTIME_DIR"] == "/tmp/runtime"
    assert not set(env).intersection(sanitized) - {"PATH", "XDG_RUNTIME_DIR"}

    workdir = tmp_path / "session"
    workdir.mkdir()
    identity = tmp_path / "identity"
    identity.mkdir()
    passwd = identity / "passwd"
    group = identity / "group"
    passwd.touch()
    group.touch()
    monkeypatch.setattr(os_sandbox, "_linux_environment", lambda: "wsl2")
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    monkeypatch.setattr(os_sandbox, "_validate_linux_workdir_environment", lambda _path: None)
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: ())
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(
        os_sandbox, "_identity_files", lambda: (str(identity), str(passwd), str(group))
    )
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    prepared = backend.prepare(_spec(workdir))
    try:
        assert "/usr/lib/wsl" in _mount_sources(prepared.argv, "--tmpfs")
    finally:
        prepared.cleanup()


def test_runtime_paths_preserve_virtualenv_executable_spelling_and_configuration(
    monkeypatch, tmp_path
):
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents = True)
    base_python.write_text("python", encoding = "utf-8")
    venv = tmp_path / "venv"
    venv_python = venv / "bin" / "python"
    venv_python.parent.mkdir(parents = True)
    try:
        venv_python.symlink_to(base_python)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    pyvenv_cfg = venv / "pyvenv.cfg"
    pyvenv_cfg.write_text("home = ../base/bin\n", encoding = "utf-8")
    monkeypatch.setattr(os_sandbox.sys, "executable", str(venv_python))
    monkeypatch.setattr(os_sandbox.sys, "prefix", str(venv))
    monkeypatch.setattr(os_sandbox.sys, "base_prefix", str(base_python.parents[1]))
    monkeypatch.setattr(os_sandbox.sysconfig, "get_paths", lambda: {})

    paths = os_sandbox._runtime_read_paths()

    assert str(venv_python) in paths
    assert str(base_python) in paths
    assert str(pyvenv_cfg) in paths
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", (str(base_python.parent),))
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    prepared = backend.prepare(_spec(workdir, str(venv_python), "-c", "pass"))
    try:
        assert str(venv_python) in _mount_sources(prepared.argv, "--ro-bind")
    finally:
        prepared.cleanup()


def test_linux_runtime_beneath_tmp_is_mounted_after_private_tmpfs(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    runtime = "/tmp/studio-venv/bin/python"
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (runtime,))
    monkeypatch.setattr(os_sandbox, "_validate_runtime_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    identity = tmp_path / "identity"
    identity.mkdir()
    passwd = identity / "passwd"
    group = identity / "group"
    passwd.touch()
    group.touch()
    monkeypatch.setattr(
        os_sandbox,
        "_identity_files",
        lambda: (str(identity), str(passwd), str(group)),
    )
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    prepared = backend.prepare(_spec(workdir, runtime, "-c", "pass"))
    try:
        runtime_mount = prepared.argv.index(runtime) - 1
        private_tmpfs = prepared.argv.index("/tmp", prepared.argv.index("--tmpfs"))
        assert prepared.argv[runtime_mount] == "--ro-bind"
        assert private_tmpfs < runtime_mount
    finally:
        prepared.cleanup()


def test_linux_seccomp_filter_rejects_x32_syscalls_on_x86(monkeypatch):
    monkeypatch.setattr(os_sandbox.platform, "machine", lambda: "x86_64")
    with os_sandbox._linux_seccomp_filter() as stream:
        raw = stream.read()
    instructions = [
        struct.unpack("=HBBI", raw[offset : offset + 8]) for offset in range(0, len(raw), 8)
    ]
    assert (0x45, 7, 0, 0x40000000) in instructions


def test_linux_system_roots_are_scanned_for_host_ipc(monkeypatch, tmp_path):
    root = tmp_path / "system-root"
    root.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    sentinel = root / "host.sock"
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", (str(root),))
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda path: True)
    monkeypatch.setattr(
        os_sandbox.subprocess,
        "run",
        Mock(return_value = subprocess.CompletedProcess([], 0, stdout = str(sentinel), stderr = "")),
    )

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "socket, FIFO, or device"):
        os_sandbox._validate_runtime_paths((str(root),), str(workdir), include_system_roots = True)


def test_linux_system_roots_reject_nested_host_mounts(monkeypatch, tmp_path):
    root = tmp_path / "system-root"
    root.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    nested_mount = root / "host-volume"
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: (str(nested_mount),))

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "nested host mount"):
        os_sandbox._validate_runtime_paths((str(root),), str(workdir), include_system_roots = True)


def test_linux_system_root_scan_does_not_prune_searchable_directory(monkeypatch, tmp_path):
    root = tmp_path / "system-root"
    root.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda path: True)
    run = Mock(return_value = subprocess.CompletedProcess([], 0, stdout = "", stderr = ""))
    monkeypatch.setattr(os_sandbox.subprocess, "run", run)

    os_sandbox._validate_runtime_paths((str(root),), str(workdir), include_system_roots = True)

    command = run.call_args.args[0]
    assert "-readable" not in command
    assert command.count("-executable") == 1


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(socket, "AF_UNIX"),
    reason = "POSIX directory symlinks and Unix sockets are required",
)
def test_linux_runtime_scan_follows_symlinked_root(monkeypatch, tmp_path):
    short_root = tempfile.mkdtemp(prefix = "us-runtime-", dir = "/tmp")
    target = Path(short_root)
    runtime_link = tmp_path / "runtime-link"
    runtime_link.symlink_to(target, target_is_directory = True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(target / "host.sock"))
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda _path: False)
    try:
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "socket, FIFO, or device"):
            os_sandbox._validate_runtime_paths((str(runtime_link),), str(workdir))
    finally:
        server.close()
        shutil.rmtree(short_root, ignore_errors = True)


@pytest.mark.parametrize("detector_returncode", [0, 1, 2])
def test_linux_environment_classifies_containers_without_blanket_rejection(
    detector_returncode, monkeypatch
):
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda path: True)
    monkeypatch.setattr(os_sandbox.os.path, "exists", lambda path: False)
    monkeypatch.delenv("container", raising = False)
    monkeypatch.delenv("WSL_INTEROP", raising = False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising = False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising = False)
    monkeypatch.delitem(os_sandbox.sys.modules, "google.colab", raising = False)
    monkeypatch.setattr(
        os_sandbox.subprocess,
        "run",
        Mock(return_value = subprocess.CompletedProcess([], detector_returncode)),
    )
    monkeypatch.setattr("builtins.open", Mock(side_effect = OSError("unavailable")))

    expected = (
        "container"
        if detector_returncode == 0
        else "native_linux"
        if detector_returncode == 1
        else "linux_unknown"
    )
    assert os_sandbox._linux_environment() == expected
    assert os_sandbox._excluded_linux_environment() is None


def test_linux_environment_without_detector_is_eligible_for_live_probe(monkeypatch):
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda path: False)
    monkeypatch.setattr(os_sandbox.os.path, "exists", lambda path: False)
    monkeypatch.delenv("container", raising = False)
    monkeypatch.delenv("WSL_INTEROP", raising = False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising = False)
    monkeypatch.setattr("builtins.open", Mock(side_effect = OSError("unavailable")))

    assert os_sandbox._linux_environment() == "linux_unknown"
    assert os_sandbox._excluded_linux_environment() is None


def test_workdir_validation_rejects_boundary_crossing_hardlink(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("host secret", encoding = "utf-8")
    try:
        os.link(outside, workdir / "hardlink")
    except OSError as exc:
        pytest.skip(f"hard links are unsupported on this filesystem: {exc}")

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "hard-linked outside"):
        os_sandbox._validate_workdir(str(workdir))


def test_workdir_validation_allows_hardlinks_fully_contained_in_workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    original = workdir / "original"
    original.write_text("session data", encoding = "utf-8")
    try:
        os.link(original, workdir / "second-name")
    except OSError as exc:
        pytest.skip(f"hard links are unsupported on this filesystem: {exc}")

    assert os_sandbox._validate_workdir(str(workdir)) == os.path.realpath(workdir)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "filesystem FIFOs are unavailable")
def test_workdir_validation_rejects_fifo(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    os.mkfifo(workdir / "host.fifo")

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "socket, FIFO, or device"):
        os_sandbox._validate_workdir(str(workdir))


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason = "pathname Unix sockets are unavailable")
def test_workdir_validation_rejects_unix_socket(tmp_path):
    with tempfile.TemporaryDirectory(prefix = "us-wd-", dir = "/tmp") as short_root:
        workdir = Path(short_root) / "work"
        workdir.mkdir()
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(str(workdir / "host.sock"))
            with pytest.raises(os_sandbox.SandboxUnavailableError, match = "socket, FIFO, or device"):
                os_sandbox._validate_workdir(str(workdir))
        finally:
            server.close()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX directory permissions are required")
@pytest.mark.parametrize("hidden_kind", ["fifo", "unix-socket", "external-hardlink"])
def test_workdir_validation_fails_closed_on_unreadable_subtree(tmp_path, hidden_kind):
    short_root = tempfile.mkdtemp(prefix = "us-hidden-", dir = "/tmp")
    workdir = Path(short_root) / "work"
    hidden = workdir / "hidden"
    hidden.mkdir(parents = True)
    server = None
    if hidden_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("filesystem FIFOs are unavailable")
        os.mkfifo(hidden / "host.fifo")
    elif hidden_kind == "unix-socket":
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("pathname Unix sockets are unavailable")
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(hidden / "host.sock"))
    else:
        outside = tmp_path / "outside"
        outside.write_text("secret", encoding = "utf-8")
        try:
            os.link(outside, hidden / "host-hardlink")
        except OSError as exc:
            pytest.skip(f"hard links are unavailable: {exc}")

    hidden.chmod(0)
    try:
        if os.access(hidden, os.R_OK | os.X_OK):
            pytest.skip("the test user can inspect mode-000 directories")
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "cannot be fully inspected"):
            os_sandbox._validate_workdir(str(workdir))
    finally:
        hidden.chmod(0o700)
        if server is not None:
            server.close()
        shutil.rmtree(short_root, ignore_errors = True)


def test_macos_backend_is_fail_closed_when_seatbelt_is_unavailable(tmp_path):
    backend = os_sandbox.MacOSSeatbeltBackend()

    capability = backend.probe()

    assert not capability.qualified
    assert capability.available is False
    assert capability.backend == "macos-seatbelt"
    assert "deprecated_undocumented_sbpl" in capability.limitations
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "not proven available"):
        backend.prepare(_spec(tmp_path))


class _FakeProcess:
    pid = 424242
    returncode = 0

    def poll(self):
        return self.returncode


def _patch_tool_harness(monkeypatch, workdir: Path):
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(inference_tools, "_snapshot_workdir_files", lambda _path: {})
    monkeypatch.setattr(inference_tools, "_created_file_sentinels", lambda *_a, **_k: "")
    monkeypatch.setattr(inference_tools, "_spill_scope", lambda *_a, **_k: None)
    monkeypatch.setattr(inference_tools, "_call_started", lambda _path: object())
    monkeypatch.setattr(inference_tools, "_call_finished", lambda _token: None)
    monkeypatch.setattr(inference_tools, "_capture_process_group", lambda _proc: None)
    monkeypatch.setattr(inference_tools, "_adopt_tool_pid", lambda _pid: None)
    monkeypatch.setattr(inference_tools, "_forget_tool_pid", lambda _proc: None)
    monkeypatch.setattr(inference_tools, "_drain_process_output", lambda *_a, **_k: ("ok\n", False))
    monkeypatch.setattr(inference_tools, "_check_code_safety", lambda _code: None)
    monkeypatch.setattr(inference_tools, "_find_blocked_commands", lambda _command: set())
    monkeypatch.setattr(inference_tools, "_harden_parent_against_proc_env_leak", lambda: True)
    monkeypatch.setattr(inference_tools, "_build_safe_env", lambda _path: {"MODE": "safe"})
    monkeypatch.setattr(inference_tools, "_build_bypass_env", lambda _path: {"MODE": "bypass"})


def _invoke_tool(
    kind: str,
    *,
    disable_sandbox: bool = False,
    **kwargs,
) -> str:
    if kind == "python":
        return inference_tools._python_exec(
            "print('ok')", disable_sandbox = disable_sandbox, **kwargs
        )
    return inference_tools._bash_exec("printf ok", disable_sandbox = disable_sandbox, **kwargs)


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_real_tool_path_prepares_before_launch_and_never_popen_inner_argv(
    kind, monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _patch_tool_harness(monkeypatch, workdir)
    specs: list[os_sandbox.ToolLaunchPlan] = []
    prepared = os_sandbox.PreparedSandboxLaunch(
        argv = ("qualified-native-sandbox", "opaque-inner-command"),
        workdir = str(workdir),
        env = {"MODE": "prepared"},
        preexec_fn = None,
        backend = "test-native",
        execution_record = os_sandbox.ToolExecutionRecord(
            requested_mode = "os_isolation_required",
            effective_mode = "os_isolation_required",
            environment = "native_linux",
            backend = "test-native",
            profile_id = "test-v1",
            probe_generation = "generation",
            os_isolation = True,
            retained_safeguards = ("os_isolation",),
        ),
    )
    prepared.cleanup = Mock()

    def prepare(spec):
        specs.append(spec)
        return prepared

    popen_calls = []
    lifecycle_events = []

    def popen(argv, **kwargs):
        lifecycle_events.append("popen")
        popen_calls.append((tuple(argv), kwargs))
        return _FakeProcess()

    monkeypatch.setattr(inference_tools, "prepare_tool_launch", prepare)
    monkeypatch.setattr(inference_tools.subprocess, "Popen", popen)

    assert (
        _invoke_tool(
            kind,
            launch_record_callback = lambda record: lifecycle_events.append(record),
        )
        == "ok\n"
    )
    assert len(specs) == 1
    assert len(popen_calls) == 1
    launched_argv, launched_kwargs = popen_calls[0]
    assert launched_argv == prepared.argv
    assert launched_argv != specs[0].argv
    assert launched_kwargs["cwd"] == str(workdir)
    assert launched_kwargs["env"] == {"MODE": "prepared"}
    assert launched_kwargs["close_fds"] is True
    assert launched_kwargs["stdin"] is subprocess.DEVNULL
    assert prepared.cleanup.call_count == 1
    assert lifecycle_events == ["popen", prepared.execution_record]
    if kind == "python":
        assert specs[0].argv[0:2] == (sys.executable, "-u")
        assert specs[0].argv[2].endswith(".py")
    else:
        assert specs[0].argv == tuple(inference_tools._get_shell_cmd("printf ok"))


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_failed_prepare_blocks_before_popen(kind, monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _patch_tool_harness(monkeypatch, workdir)
    popen = Mock(side_effect = AssertionError("Popen must not run"))
    monkeypatch.setattr(inference_tools.subprocess, "Popen", popen)
    monkeypatch.setattr(
        inference_tools,
        "prepare_tool_launch",
        Mock(side_effect = os_sandbox.SandboxUnavailableError("native sandbox unavailable")),
    )

    result = _invoke_tool(kind)

    assert result.startswith("Execution error:")
    assert "native sandbox unavailable" in result
    popen.assert_not_called()


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_explicit_disable_sandbox_uses_full_launch_plan(kind, monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _patch_tool_harness(monkeypatch, workdir)
    specs = []

    def prepare(spec):
        specs.append(spec)
        return os_sandbox.PreparedSandboxLaunch(
            argv = spec.argv,
            workdir = spec.workdir,
            env = spec.env,
            preexec_fn = spec.preexec_fn,
            backend = "none",
        )

    monkeypatch.setattr(inference_tools, "prepare_tool_launch", prepare)
    popen_calls = []

    def popen(argv, **kwargs):
        popen_calls.append((tuple(argv), kwargs))
        return _FakeProcess()

    monkeypatch.setattr(inference_tools.subprocess, "Popen", popen)

    assert _invoke_tool(kind, disable_sandbox = True) == "ok\n"
    assert len(specs) == 1
    assert specs[0].requested_mode == "full"
    assert len(popen_calls) == 1
    argv, kwargs = popen_calls[0]
    assert kwargs["env"]["MODE"] == "bypass"
    assert kwargs["close_fds"] is True
    assert "stdin" not in kwargs
    if kind == "python":
        assert argv[0:2] == (sys.executable, "-u")
        assert argv[2].endswith(".py")
    else:
        assert argv == tuple(inference_tools._get_shell_cmd("printf ok"))


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"),
    reason = "Limited compatibility evidence is collected on macOS and Windows",
)
@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_live_unqualified_hosts_run_only_with_a_current_limited_grant(kind, monkeypatch, tmp_path):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    capability = os_sandbox.capability_snapshot(force = True)
    if capability.available:
        pytest.skip(f"{capability.backend} is available; Limited grants are correctly disabled")
    grant = tool_isolation.issue_limited_grant(
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        probe_generation = capability.probe_generation,
    )
    records = []

    result = _invoke_tool(
        kind,
        timeout = 20,
        tool_execution_mode = "limited",
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        limited_grant = grant.token,
        launch_record_callback = records.append,
    )

    assert result.strip() == "ok"
    assert len(records) == 1
    assert records[0].effective_mode == "limited"
    assert records[0].os_isolation is False
    assert "resource_limits" in records[0].retained_safeguards


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows Job Object behavior")
def test_windows_limited_resource_setup_fails_before_payload_runs(
    monkeypatch, tmp_path, isolated_capability_cache
):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    sentinel = workdir / "payload-ran"
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", "invalid")
    capability = os_sandbox.capability_snapshot(force = True)
    grant = tool_isolation.issue_limited_grant(
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page-fail",
        probe_generation = capability.probe_generation,
    )

    result = inference_tools._python_exec(
        f"open({str(sentinel)!r}, 'w').write('ran')",
        timeout = 20,
        tool_execution_mode = "limited",
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page-fail",
        limited_grant = grant.token,
    )

    assert "could not establish Windows process resource limits" in result
    assert not sentinel.exists()


@pytest.mark.skipif(os.name != "posix", reason = "POSIX pre-exec resource limits")
def test_posix_limited_resource_setup_fails_before_payload_runs(
    monkeypatch, tmp_path, isolated_capability_cache
):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    sentinel = workdir / "payload-ran"
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: None)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", "invalid")
    capability = os_sandbox.capability_snapshot(force = True)
    grant = tool_isolation.issue_limited_grant(
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page-posix-fail",
        probe_generation = capability.probe_generation,
    )

    result = inference_tools._python_exec(
        f"open({str(sentinel)!r}, 'w').write('ran')",
        timeout = 20,
        tool_execution_mode = "limited",
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page-posix-fail",
        limited_grant = grant.token,
    )

    assert "Limited mode resource-limit configuration is invalid" in result
    assert not sentinel.exists()


@pytest.fixture(scope = "module")
def qualified_native_capability():
    capability = os_sandbox.sandbox_capability()
    if capability.backend != "linux-bubblewrap" or not capability.qualified:
        pytest.skip(
            f"qualified Bubblewrap is unavailable: {capability.backend}: {capability.reason}"
        )
    return capability


def _run_native(
    workdir: Path,
    code: str,
    *,
    timeout: int = 20,
) -> subprocess.CompletedProcess:
    prepared = os_sandbox.prepare_tool_launch(_spec(workdir, sys.executable, "-I", "-c", code))
    try:
        kwargs = dict(
            cwd = prepared.workdir,
            env = prepared.env,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            timeout = timeout,
            close_fds = True,
            stdin = subprocess.DEVNULL,
        )
        if prepared.preexec_fn is not None and os.name == "posix":
            kwargs["preexec_fn"] = prepared.preexec_fn
        if prepared.pass_fds:
            kwargs["pass_fds"] = prepared.pass_fds
        return subprocess.run(prepared.argv, **kwargs)
    finally:
        prepared.cleanup()


def _assert_native_ok(completed: subprocess.CompletedProcess) -> None:
    assert completed.returncode == 0, (
        f"sandboxed command failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_live_filesystem_proc_devices_and_interpreter_boundary(
    qualified_native_capability, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    readable = workdir / "input.txt"
    readable.write_text("inside", encoding = "utf-8")
    outside = tmp_path / "outside-secret"
    outside.write_text("outside", encoding = "utf-8")
    home_secret_handle = tempfile.NamedTemporaryFile(
        mode = "w", prefix = ".unsloth-sandbox-test-", dir = Path.home(), delete = False
    )
    escape_target = Path(f"{home_secret_handle.name}-escape")
    try:
        home_secret_handle.write("home secret")
        home_secret_handle.close()
        home_secret = home_secret_handle.name
        repo_file = str(Path(__file__).resolve())
        escape_read = workdir / "escape-read"
        escape_write = workdir / "escape-write"
        escape_read.symlink_to(outside)
        escape_write.symlink_to(escape_target)
        high_fd = 240
        source_fd = os.open(outside, os.O_RDONLY)
        os.dup2(source_fd, high_fd, inheritable = True)
        os.close(source_fd)
        code = f"""
import _sqlite3, _ssl, errno, json, os, sqlite3, ssl, zlib

def denied_read(path):
    try:
        with open(path, 'rb') as stream:
            stream.read(1)
    except OSError:
        return
    raise AssertionError('unexpectedly read ' + path)

with open('input.txt', encoding='utf-8') as stream:
    assert stream.read() == 'inside'
with open('output.txt', 'w', encoding='utf-8') as stream:
    stream.write('written')
denied_read({str(outside)!r})
denied_read({home_secret!r})
denied_read({repo_file!r})
denied_read('escape-read')
try:
    with open('escape-write', 'w', encoding='utf-8') as stream:
        stream.write('escape')
except OSError:
    pass
else:
    raise AssertionError('wrote through an escaping symlink')
assert not os.path.exists('/proc/{os.getpid()}/environ')
assert os.read(0, 1) == b''
for path in ('/run', '/var/run', '/dev/kvm', '/dev/sda', '/dev/dri', '/dev/fuse'):
    assert not os.path.exists(path), path
runtime_paths = [
    {sys.executable!r}, json.__file__, sqlite3.__file__, ssl.__file__,
    _sqlite3.__file__, _ssl.__file__,
]
if getattr(zlib, '__file__', None):
    runtime_paths.append(zlib.__file__)
for path in runtime_paths:
    with open(path, 'rb') as stream:
        assert stream.read(1)
for path in ({sys.executable!r}, '/etc/passwd'):
    try:
        with open(path, 'ab') as stream:
            stream.write(b'x')
    except OSError:
        pass
    else:
        raise AssertionError('modified read-only runtime path ' + path)
try:
    os.fstat({high_fd})
except OSError as exc:
    assert exc.errno == errno.EBADF
else:
    raise AssertionError('inherited host fd remained accessible')
print('FILESYSTEM_BOUNDARY_OK')
"""
        try:
            completed = _run_native(workdir, code)
        finally:
            os.close(high_fd)
        _assert_native_ok(completed)
        assert "FILESYSTEM_BOUNDARY_OK" in completed.stdout
        assert (workdir / "output.txt").read_text(encoding = "utf-8") == "written"
        assert outside.read_text(encoding = "utf-8") == "outside"
        assert not escape_target.exists()
    finally:
        try:
            home_secret_handle.close()
        except OSError:
            pass
        try:
            os.unlink(home_secret_handle.name)
        except OSError:
            pass
        try:
            escape_target.unlink()
        except OSError:
            pass


def test_live_private_tmp_is_fresh_between_invocations(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    marker = "unsloth-private-tmp-does-not-persist"
    first = _run_native(
        workdir,
        f"import os; open(os.path.join(os.environ['TMPDIR'], {marker!r}), 'w').write('x')",
    )
    _assert_native_ok(first)
    second = _run_native(
        workdir,
        f"import os; assert not os.path.exists(os.path.join(os.environ['TMPDIR'], {marker!r})); print('FRESH_TMP_OK')",
    )
    _assert_native_ok(second)
    assert "FRESH_TMP_OK" in second.stdout


def _listen(family: socket.AddressFamily, address):
    server = socket.socket(family)
    server.bind(address)
    server.listen(1)
    return server, server.getsockname()


def test_live_network_and_host_unix_socket_are_unreachable(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    ipv4, ipv4_address = _listen(socket.AF_INET, ("127.0.0.1", 0))
    ipv6 = None
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp_address = udp.getsockname()
    socket_directory = tempfile.mkdtemp(prefix = "unsloth-host-sockets-")
    host_unix_servers = []
    try:
        try:
            ipv6, ipv6_address = _listen(socket.AF_INET6, ("::1", 0))
        except OSError as exc:
            pytest.fail(f"qualified backend cannot prove IPv6 denial: {exc}")
        host_unix_paths = []
        for name in ("docker", "containerd", "ssh-agent", "studio", "mcp"):
            server = socket.socket(socket.AF_UNIX)
            path = os.path.join(socket_directory, f"{name}.sock")
            server.bind(path)
            server.listen(1)
            host_unix_servers.append(server)
            host_unix_paths.append(path)
        try:
            code = f"""
import os, socket

def denied(family, address):
    client = socket.socket(family)
    client.settimeout(1)
    try:
        client.connect(address)
    except OSError:
        return
    finally:
        client.close()
    raise AssertionError('host endpoint was reachable: ' + repr(address))

if hasattr(socket, 'AF_VSOCK'):
    try:
        socket.socket(socket.AF_VSOCK)
    except OSError:
        pass
    else:
        raise AssertionError('AF_VSOCK remained available')
denied(socket.AF_INET, {ipv4_address!r})
denied(socket.AF_INET6, {ipv6_address!r})
datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    datagram.sendto(b'UNSLOTH_TEST_UDP_PROBE', {udp_address!r})
except OSError:
    pass
finally:
    datagram.close()
for host_socket in {host_unix_paths!r}:
    denied(socket.AF_UNIX, host_socket)
for variable in ('DOCKER_HOST', 'SSH_AUTH_SOCK', 'DISPLAY', 'WAYLAND_DISPLAY', 'MCP_SOCKET'):
    assert variable not in os.environ, variable
try:
    assert not os.path.exists('/etc/resolv.conf')
    socket.getaddrinfo('example.com', 443)
except socket.gaierror:
    pass
else:
    raise AssertionError('DNS resolution escaped the sandbox')
print('NETWORK_BOUNDARY_OK')
"""
            completed = _run_native(workdir, code)
            udp.settimeout(0.05)
            with pytest.raises((TimeoutError, socket.timeout)):
                udp.recvfrom(256)
        finally:
            for server in host_unix_servers:
                server.close()
    finally:
        ipv4.close()
        if ipv6 is not None:
            ipv6.close()
        udp.close()
        shutil.rmtree(socket_directory, ignore_errors = True)
    _assert_native_ok(completed)
    assert "NETWORK_BOUNDARY_OK" in completed.stdout


def test_live_private_unix_socket_and_resource_sharer_work(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import os, socket
from multiprocessing.resource_sharer import DupFd, stop

path = os.path.join(os.environ['TMPDIR'], 'private.sock')
server = socket.socket(socket.AF_UNIX)
client = socket.socket(socket.AF_UNIX)
try:
    server.bind(path)
    server.listen(1)
    client.connect(path)
    accepted, _ = server.accept()
    accepted.sendall(b'ok')
    assert client.recv(2) == b'ok'
    accepted.close()
finally:
    client.close()
    server.close()
read_fd, write_fd = os.pipe()
shared_fd = DupFd(read_fd).detach()
os.write(write_fd, b'R')
assert os.read(shared_fd, 1) == b'R'
for fd in (read_fd, write_fd, shared_fd):
    os.close(fd)
stop()
print('PRIVATE_UNIX_AND_RESOURCE_SHARER_OK')
"""
    completed = _run_native(workdir, code)
    _assert_native_ok(completed)
    assert "PRIVATE_UNIX_AND_RESOURCE_SHARER_OK" in completed.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "PyTorch is unavailable")
def test_live_pytorch_tensor_transfer_uses_private_resource_sharing(
    qualified_native_capability, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import torch
import torch.multiprocessing as mp

def send(queue, received):
    queue.put(torch.arange(4))
    assert received.wait(10)

context = mp.get_context('fork')
queue = context.Queue()
received = context.Event()
process = context.Process(target=send, args=(queue, received))
process.start()
tensor = queue.get(timeout=10)
received.set()
process.join(10)
assert process.exitcode == 0
assert torch.equal(tensor, torch.arange(4))
print('PYTORCH_TENSOR_TRANSFER_OK')
"""
    completed = _run_native(workdir, code, timeout = 30)
    _assert_native_ok(completed)
    assert "PYTORCH_TENSOR_TRANSFER_OK" in completed.stdout


@pytest.mark.parametrize("kind", ["python-timeout", "terminal-cancel"])
def test_live_real_tool_path_terminates_detached_sandbox_descendant(
    kind, qualified_native_capability, monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    marker = workdir / "detached-child-survived"
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(inference_tools, "_check_code_safety", lambda _code: None)
    monkeypatch.setattr(inference_tools, "_find_blocked_commands", lambda _command: set())

    if kind == "python-timeout":
        child = (
            "import time; time.sleep(2); "
            f"open({str(marker)!r}, 'w', encoding='utf-8').write('escaped')"
        )
        code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True, "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "time.sleep(30)\n"
        )
        result = inference_tools._python_exec(code, timeout = 1)
        assert result == "Execution timed out after 1 seconds."
    else:
        cancel = threading.Event()
        timer = threading.Timer(0.5, cancel.set)
        timer.start()
        try:
            command = (
                "setsid sh -c "
                f"'sleep 2; printf escaped > {str(marker)!r}' "
                "</dev/null >/dev/null 2>&1 & sleep 30"
            )
            result = inference_tools._bash_exec(command, cancel_event = cancel, timeout = 20)
        finally:
            timer.cancel()
        assert result == "Execution cancelled."

    time.sleep(2.5)
    assert not marker.exists()


def test_live_symlink_virtualenv_interpreter_runs_inside_sandbox(
    qualified_native_capability, tmp_path
):
    venv = tmp_path / "venv"
    workdir = tmp_path / "work"
    workdir.mkdir()
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 30,
        close_fds = True,
    )
    assert created.returncode == 0, created.stderr
    venv_python = venv / "bin" / "python"
    assert venv_python.exists()
    if not venv_python.is_symlink():
        venv_python.unlink()
        venv_python.symlink_to(sys.executable)
    assert venv_python.is_symlink()
    backend_root = str(Path(__file__).resolve().parents[1])
    runtime_config = venv / "pyvenv.cfg"
    inner = f"""
import _ssl, sys
assert sys.prefix == {str(venv)!r}
try:
    with open({str(runtime_config)!r}, 'a', encoding='utf-8') as stream:
        stream.write('escape')
except OSError:
    pass
else:
    raise AssertionError('modified host-writable runtime configuration')
print('VENV_OK')
"""
    helper = f"""
import os, subprocess, sys, types
logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
loggers = types.ModuleType('loggers')
loggers.get_logger = lambda name: logger
sys.modules['loggers'] = loggers
from core.inference import os_sandbox

assert os.path.abspath(sys.executable) == {str(venv_python)!r}
capability = os_sandbox.sandbox_capability()
assert capability.qualified, capability
spec = os_sandbox.SandboxLaunchSpec(
    argv=(sys.executable, '-I', '-c', {inner!r}),
    workdir={str(workdir)!r},
    env={{'HOME': {str(workdir)!r}, 'TMPDIR': '/tmp', 'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8'}},
)
prepared = os_sandbox.prepare_tool_launch(spec)
try:
    completed = subprocess.run(
        prepared.argv, cwd=prepared.workdir, env=prepared.env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=20, close_fds=True, preexec_fn=prepared.preexec_fn,
        pass_fds=prepared.pass_fds,
    )
finally:
    prepared.cleanup()
assert completed.returncode == 0, completed.stderr
assert 'VENV_OK' in completed.stdout
print('SYMLINK_VENV_SANDBOX_OK')
"""
    completed = subprocess.run(
        [str(venv_python), "-c", helper],
        cwd = backend_root,
        env = {**os.environ, "PYTHONPATH": backend_root},
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 60,
        close_fds = True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SYMLINK_VENV_SANDBOX_OK" in completed.stdout
    assert "escape" not in runtime_config.read_text(encoding = "utf-8")


def test_live_twenty_launch_startup_measurement(
    qualified_native_capability, tmp_path, record_property
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    samples_ms = []
    for _ in range(20):
        started = time.perf_counter()
        completed = _run_native(workdir, "print('STARTED')")
        samples_ms.append((time.perf_counter() - started) * 1000)
        _assert_native_ok(completed)
        assert "STARTED" in completed.stdout

    median_ms = statistics.median(samples_ms)
    ordered = sorted(samples_ms)
    p95_ms = ordered[math.ceil(0.95 * len(ordered)) - 1]
    record_property("sandbox_startup_samples", len(samples_ms))
    record_property("sandbox_startup_median_ms", round(median_ms, 3))
    record_property("sandbox_startup_p95_ms", round(p95_ms, 3))
    print(f"sandbox startup over 20 launches: median={median_ms:.3f}ms p95={p95_ms:.3f}ms")
