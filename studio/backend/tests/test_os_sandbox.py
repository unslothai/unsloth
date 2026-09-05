# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused contract and live-enforcement tests for Studio's native OS sandbox."""

from __future__ import annotations

import errno
import importlib.util
import math
import os
import shlex
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
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda: pytest.fail("Full access must not run an OS sandbox capability probe"),
    )
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
        os_sandbox._LinuxMount("2", "1", "0:2", "/", "/usr/share/secrets", "ro", "tmpfs", "tmpfs", "ro")
    )

    assert os_sandbox._environment_fingerprint(None) != before


def test_unrelated_mounts_do_not_change_environment_fingerprint(monkeypatch):
    """A USB stick, a container volume or a gvfs mount never enter the sandbox."""
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_environment_class", lambda: "container")
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda _name: None)
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: ("/opt/python/bin/python",))
    mounts = [os_sandbox._LinuxMount("1", "0", "0:1", "/", "/", "rw", "overlay", "overlay", "rw")]
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: tuple(mounts))

    before = os_sandbox._environment_fingerprint(None)
    for point in ("/media/usb", "/run/user/1000/gvfs", "/var/lib/docker/overlay2/abc/merged", "/mnt/data"):
        mounts.append(os_sandbox._LinuxMount("9", "1", "0:9", "/", point, "rw", "ext4", "/dev/sdb1", "rw"))
    assert os_sandbox._environment_fingerprint(None) == before

    # A mount that contains the interpreter is relevant, as is one under a system root.
    mounts.append(os_sandbox._LinuxMount("3", "1", "0:3", "/", "/opt", "rw", "ext4", "/dev/sdc1", "rw"))
    after_runtime_mount = os_sandbox._environment_fingerprint(None)
    assert after_runtime_mount != before
    mounts.append(os_sandbox._LinuxMount("4", "1", "0:4", "/", "/usr/lib/x", "ro", "squashfs", "loop0", "ro"))
    assert os_sandbox._environment_fingerprint(None) != after_runtime_mount


def test_probe_generation_ignores_reason_wording(monkeypatch, isolated_capability_cache):
    """Grants are bound to security facts, not to the text of a diagnostic."""
    monkeypatch.setattr(os_sandbox, "_environment_class", lambda: "container")
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _backend: "fingerprint")
    first = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability("linux-bubblewrap", False, "probe failed: /tmp/a1b2/host"),
        environment = "container",
        fingerprint = "fingerprint",
    )
    second = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability("linux-bubblewrap", False, "probe failed: /tmp/z9y8/host"),
        environment = "container",
        fingerprint = "fingerprint",
    )
    assert first.probe_generation == second.probe_generation
    qualified = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability("linux-bubblewrap", True, "restrictive live probe passed"),
        environment = "container",
        fingerprint = "fingerprint",
    )
    assert qualified.probe_generation != first.probe_generation
    with_limitation = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability(
            "linux-bubblewrap", True, "restrictive live probe passed",
            limitations = (os_sandbox._LIMITATION_NESTED_USERNS_SECCOMP,),
        ),
        environment = "container",
        fingerprint = "fingerprint",
    )
    assert with_limitation.probe_generation != qualified.probe_generation


def test_backend_remediation_survives_identity_stamping():
    generic = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability("linux-bubblewrap", False, "no bwrap"),
        environment = "native_linux",
        fingerprint = "fp",
    )
    assert "Limited mode" in generic.remediation
    specific = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability(
            "linux-bubblewrap", False, "blocked", remediation = "install the profile"
        ),
        environment = "native_linux",
        fingerprint = "fp",
    )
    assert specific.remediation == "install the profile"
    assert specific.probe_generation == generic.probe_generation
    available = os_sandbox._capability_with_identity(
        os_sandbox.SandboxCapability("linux-bubblewrap", True, "ok", remediation = "ignored"),
        environment = "native_linux",
        fingerprint = "fp",
    )
    assert available.remediation == "No remediation required."


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
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
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


def test_prepare_revalidates_runtime_roots_before_mounts(monkeypatch, tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    validation_calls = []

    def validate(paths, checked_workdir, **kwargs):
        validation_calls.append((tuple(paths), checked_workdir, kwargs))
        if tuple(paths) == (str(runtime),):
            raise os_sandbox.SandboxUnavailableError("unsafe runtime sentinel")

    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (str(runtime),))
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(os_sandbox, "_validate_runtime_paths", validate)
    mount_construction = Mock(side_effect = AssertionError("mount construction started"))
    monkeypatch.setattr(os_sandbox, "_nested_exposed_mounts", mount_construction)
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "unsafe runtime sentinel"):
        backend.prepare(_spec(workdir))

    assert validation_calls[-1][0] == (str(runtime),)
    assert validation_calls[-1][1] == os.path.realpath(workdir)
    mount_construction.assert_not_called()


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(socket, "AF_UNIX"),
    reason = "Linux pathname Unix sockets are required",
)
def test_unsafe_runtime_socket_never_reaches_bwrap_argv(monkeypatch, tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    # A short root keeps the socket path under sun_path's 108 bytes on any checkout.
    runtime_dir = tempfile.TemporaryDirectory(prefix = "us-rt-", dir = "/tmp")
    runtime = Path(runtime_dir.name)
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(runtime / "host.sock"))
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (str(runtime),))
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    mount_construction = Mock(side_effect = AssertionError("unsafe root reached mount construction"))
    monkeypatch.setattr(os_sandbox, "_nested_exposed_mounts", mount_construction)
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"
    try:
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Unix socket"):
            backend.prepare(_spec(workdir))
    finally:
        server.close()
        runtime_dir.cleanup()
    mount_construction.assert_not_called()


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
    framework_image = base_python.parents[1] / "Python"
    framework_image.touch()
    monkeypatch.setattr(os_sandbox.sys, "executable", str(venv_python))
    monkeypatch.setattr(os_sandbox.sys, "prefix", str(venv))
    monkeypatch.setattr(os_sandbox.sys, "base_prefix", str(base_python.parents[1]))
    monkeypatch.setattr(os_sandbox.sysconfig, "get_paths", lambda: {})

    paths = os_sandbox._runtime_read_paths()

    assert str(venv_python) in paths
    assert str(base_python) in paths
    assert str(venv_python.parent) in paths
    assert str(pyvenv_cfg) in paths
    assert str(framework_image) in paths
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

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Unix socket"):
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
    assert any(left == "-type" and right == "s" for left, right in zip(command, command[1:]))


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(socket, "AF_UNIX"),
    reason = "the trusted /usr/bin/find fast path is Linux-specific",
)
def test_find_fast_path_rejects_bound_runtime_unix_socket(monkeypatch, tmp_path):
    # A short root keeps the socket path under sun_path's 108 bytes on any checkout.
    runtime_dir = tempfile.TemporaryDirectory(prefix = "us-rt-", dir = "/tmp")
    runtime = Path(runtime_dir.name)
    workdir = tmp_path / "work"
    workdir.mkdir()
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(runtime / "host.sock"))
    real_run = subprocess.run
    commands: list[tuple[str, ...]] = []

    def recording_run(command, **kwargs):
        commands.append(tuple(command))
        return real_run(command, **kwargs)

    assert os_sandbox._trusted_linux_executable("/usr/bin/find")
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox.subprocess, "run", recording_run)
    try:
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Unix socket"):
            os_sandbox._validate_runtime_paths((str(runtime),), str(workdir))
    finally:
        server.close()
        runtime_dir.cleanup()

    assert commands and commands[0][0] == "/usr/bin/find"
    assert ("-type", "s") in tuple(zip(commands[0], commands[0][1:]))


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(socket, "AF_UNIX"),
    reason = "pathname Unix sockets are required",
)
def test_python_fallback_rejects_bound_runtime_unix_socket(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    with tempfile.TemporaryDirectory(prefix = "us-rt-", dir = "/tmp") as short_root:
        runtime = Path(short_root)
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(runtime / "host.sock"))
        run = Mock(side_effect = AssertionError("the untrusted find binary must not run"))
        monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
        monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda _path: False)
        monkeypatch.setattr(os_sandbox.subprocess, "run", run)
        try:
            with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Unix socket"):
                os_sandbox._validate_runtime_paths((str(runtime),), str(workdir))
        finally:
            server.close()
    run.assert_not_called()


def test_trusted_find_failure_fails_closed_without_fallback(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda _path: True)
    monkeypatch.setattr(
        os_sandbox.subprocess,
        "run",
        Mock(side_effect = subprocess.TimeoutExpired("/usr/bin/find", 8)),
    )
    monkeypatch.setattr(
        os_sandbox.os,
        "walk",
        Mock(side_effect = AssertionError("trusted-scanner failure must not use the fallback")),
    )

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "cannot scan"):
        os_sandbox._validate_runtime_paths((str(runtime),), str(workdir))


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
        with pytest.raises(os_sandbox.SandboxUnavailableError, match = "Unix socket"):
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

    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "FIFOs"):
        os_sandbox._validate_workdir(str(workdir))


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason = "pathname Unix sockets are unavailable")
def test_workdir_validation_rejects_unix_socket(tmp_path):
    with tempfile.TemporaryDirectory(prefix = "us-wd-", dir = "/tmp") as short_root:
        workdir = Path(short_root) / "work"
        workdir.mkdir()
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(str(workdir / "host.sock"))
            with pytest.raises(os_sandbox.SandboxUnavailableError, match = "FIFOs"):
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


def test_macos_backend_is_fail_closed_when_seatbelt_is_unavailable(monkeypatch, tmp_path):
    backend = os_sandbox.MacOSSeatbeltBackend()
    monkeypatch.setattr(
        os_sandbox.os,
        "stat",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("unavailable")),
    )

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
    payload_sentinel: Path | None = None,
    **kwargs,
) -> str:
    if kind == "python":
        code = "print('ok')"
        if payload_sentinel is not None:
            code = f"from pathlib import Path; Path({str(payload_sentinel)!r}).write_text('ran')"
        return inference_tools._python_exec(code, disable_sandbox = disable_sandbox, **kwargs)
    command = "printf ok"
    if payload_sentinel is not None:
        command = (
            f'type nul > "{payload_sentinel}"'
            if os.name == "nt"
            else f"printf ran > {shlex.quote(str(payload_sentinel))}"
        )
    return inference_tools._bash_exec(command, disable_sandbox = disable_sandbox, **kwargs)


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
def test_required_runtime_validation_failure_never_spawns(kind, monkeypatch, tmp_path):
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

    sentinel = tmp_path / "payload-ran"
    records = []
    result = _invoke_tool(
        kind,
        launch_record_callback = records.append,
        payload_sentinel = sentinel,
    )

    assert result.startswith("Execution error:")
    assert "native sandbox unavailable" in result
    popen.assert_not_called()
    assert records == []
    assert not sentinel.exists()


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_popen_failure_emits_no_execution_record(kind, monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _patch_tool_harness(monkeypatch, workdir)
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
    monkeypatch.setattr(inference_tools, "prepare_tool_launch", lambda _spec: prepared)
    monkeypatch.setattr(
        inference_tools.subprocess,
        "Popen",
        Mock(side_effect = OSError("launcher failed before payload")),
    )
    sentinel = tmp_path / "payload-ran"
    records = []

    result = _invoke_tool(
        kind,
        launch_record_callback = records.append,
        payload_sentinel = sentinel,
    )

    assert result.startswith("Execution error:")
    assert "launcher failed before payload" in result
    assert records == []
    assert not sentinel.exists()
    prepared.cleanup.assert_called_once()


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


_LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason = "Linux Bubblewrap backend")


def _fake_prepare_environment(monkeypatch, tmp_path, *, identity_only: bool = False):
    """The monkeypatch set the existing argv tests use so prepare() runs without bwrap."""
    identity = tmp_path / "identity"
    identity.mkdir(exist_ok = True)
    passwd = identity / "passwd"
    group = identity / "group"
    passwd.touch()
    group.touch()
    monkeypatch.setattr(
        os_sandbox, "_identity_files", lambda: (str(identity), str(passwd), str(group))
    )
    if identity_only:
        return
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: ())
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    monkeypatch.setattr(os_sandbox, "_validate_runtime_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(os_sandbox, "_LINUX_SYSTEM_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_LINUX_ETC_FILES", ())


@_LINUX_ONLY
def test_system_root_scan_timeout_is_transient_and_never_cached(
    monkeypatch, tmp_path, isolated_capability_cache
):
    def timing_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda _path: True)
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    monkeypatch.setattr(os_sandbox.subprocess, "run", timing_out)
    os_sandbox._forget_system_scan_memo()

    with pytest.raises(os_sandbox.SandboxUnavailableError) as excinfo:
        os_sandbox._validate_runtime_paths(
            ("/usr/lib",), str(tmp_path), include_system_roots = True, allow_nested_mounts = True
        )
    assert excinfo.value.transient is True
    assert f"exceeded {os_sandbox._SYSTEM_SCAN_TIMEOUT_SECONDS} s" in str(excinfo.value)

    # Interpreter roots keep the short budget and the same transient marker.
    with pytest.raises(os_sandbox.SandboxUnavailableError) as excinfo:
        os_sandbox._validate_runtime_paths((sys.executable,), str(tmp_path), allow_nested_mounts = True)
    assert excinfo.value.transient is True
    assert f"exceeded {os_sandbox._RUNTIME_SCAN_TIMEOUT_SECONDS} s" in str(excinfo.value)

    # Through the backend and the snapshot: retryable, and nothing is cached.
    monkeypatch.setattr(os_sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(os_sandbox, "_bwrap_supported_options", lambda _p: frozenset({"--disable-userns"}))
    backend = os_sandbox.LinuxBubblewrapBackend()
    monkeypatch.setattr(os_sandbox, "_platform_backend", lambda: backend)
    monkeypatch.setattr(os_sandbox, "_environment_class", lambda **_k: "native_linux")
    monkeypatch.setattr(os_sandbox, "_environment_fingerprint", lambda _b, **_k: "fp-timeout")
    capability = os_sandbox.capability_snapshot(force = True)
    assert not capability.available
    assert capability.transient is True and capability.retryable is True
    assert "unsafe to expose" in capability.reason and "exceeded" in capability.reason
    assert os_sandbox._capability_cache == {}


@_LINUX_ONLY
def test_system_root_scan_is_memoized_but_interpreter_scan_is_not(monkeypatch, tmp_path):
    calls: list[tuple[str, ...]] = []

    def recording_run(command, **kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout = "", stderr = "")

    monkeypatch.setattr(os_sandbox, "_trusted_linux_executable", lambda _path: True)
    monkeypatch.setattr(os_sandbox, "_linux_mount_points", lambda: ())
    monkeypatch.setattr(os_sandbox, "_linux_mounts", lambda: ())
    monkeypatch.setattr(os_sandbox.subprocess, "run", recording_run)
    os_sandbox._forget_system_scan_memo()
    try:
        for _ in range(3):
            os_sandbox._validate_runtime_paths(
                ("/usr/lib",), str(tmp_path), include_system_roots = True, allow_nested_mounts = True
            )
        assert len(calls) == 1, "a passed system-root scan is remembered"
        assert "-executable" in calls[0]

        for _ in range(2):
            os_sandbox._validate_runtime_paths((sys.executable,), str(tmp_path), allow_nested_mounts = True)
        assert len(calls) == 3, "user-writable interpreter roots are scanned every launch"

        # A mount change under the roots invalidates the memo.
        monkeypatch.setattr(
            os_sandbox,
            "_linux_mounts",
            lambda: (os_sandbox._LinuxMount("7", "1", "0:7", "/", "/usr/lib/x", "ro", "squashfs", "loop0", "ro"),),
        )
        os_sandbox._validate_runtime_paths(
            ("/usr/lib",), str(tmp_path), include_system_roots = True, allow_nested_mounts = True
        )
        assert len(calls) == 4
        os_sandbox._forget_system_scan_memo()
        os_sandbox._validate_runtime_paths(
            ("/usr/lib",), str(tmp_path), include_system_roots = True, allow_nested_mounts = True
        )
        assert len(calls) == 5
    finally:
        os_sandbox._forget_system_scan_memo()


@_LINUX_ONLY
def test_bwrap_supported_options_parses_usage_and_caches(monkeypatch, tmp_path):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\n")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        usage = (
            "usage: bwrap [OPTIONS...] [--] COMMAND [ARGS...]\n\n"
            "    --unshare-all                Unshare every namespace we support by default\n"
            "    --disable-userns             Disable further use of user namespaces inside sandbox\n"
            "    --cap-drop CAP               Drop cap CAP when running as privileged user\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout = "", stderr = usage)

    monkeypatch.setattr(os_sandbox.subprocess, "run", fake_run)
    os_sandbox._bwrap_options_cache.clear()
    options = os_sandbox._bwrap_supported_options(str(bwrap))
    assert {"--unshare-all", "--disable-userns", "--cap-drop"} <= options
    assert os_sandbox._bwrap_supported_options(str(bwrap)) is options
    assert calls == [(str(bwrap), "--help")]

    def old_run(command, **kwargs):
        usage = "usage: bwrap [OPTIONS...]\n    --unshare-all    Unshare every namespace\n"
        return subprocess.CompletedProcess(command, 0, stdout = "", stderr = usage)

    monkeypatch.setattr(os_sandbox.subprocess, "run", old_run)
    os_sandbox._bwrap_options_cache.clear()
    assert "--disable-userns" not in os_sandbox._bwrap_supported_options(str(bwrap))

    def failing_run(command, **kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr(os_sandbox.subprocess, "run", failing_run)
    os_sandbox._bwrap_options_cache.clear()
    assert os_sandbox._bwrap_supported_options(str(bwrap)) == frozenset()
    os_sandbox._bwrap_options_cache.clear()


@_LINUX_ONLY
def test_bwrap_argv_is_unchanged_when_disable_userns_is_supported(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _fake_prepare_environment(monkeypatch, tmp_path)
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"
    prepared = backend.prepare(_spec(workdir))
    try:
        argv = prepared.argv
        assert argv[:6] == (
            "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-all",
            "--unshare-user", "--disable-userns",
        )
        assert prepared.execution_record is None
        assert backend._limitations() == ()
        seccomp = prepared.owned_files[0]
        seccomp.seek(0)
        assert seccomp.read() == os_sandbox._linux_seccomp_program()
    finally:
        prepared.cleanup()


@_LINUX_ONLY
def test_bwrap_without_disable_userns_uses_seccomp_fallback(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _fake_prepare_environment(monkeypatch, tmp_path)
    backend = os_sandbox.LinuxBubblewrapBackend()
    backend._bwrap = "/usr/bin/bwrap"
    backend._disable_userns_supported = False
    prepared = backend.prepare(_spec(workdir))
    try:
        argv = prepared.argv
        assert "--disable-userns" not in argv
        assert argv[:5] == (
            "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--unshare-user",
        )
        assert argv[5:7] == ("--cap-drop", "ALL")
        assert backend._limitations() == (os_sandbox._LIMITATION_NESTED_USERNS_SECCOMP,)
        seccomp = prepared.owned_files[0]
        seccomp.seek(0)
        assert seccomp.read() == os_sandbox._linux_seccomp_program(block_userns = True)
    finally:
        prepared.cleanup()


@_LINUX_ONLY
def test_seccomp_fallback_program_denies_nested_user_namespaces():
    base = os_sandbox._linux_seccomp_program()
    fallback = os_sandbox._linux_seccomp_program(block_userns = True)
    decode = lambda program: [struct.unpack("=HBBI", program[i:i + 8]) for i in range(0, len(program), 8)]
    base_ops = decode(base)
    fallback_ops = decode(fallback)
    assert len(base_ops) == 14
    assert len(fallback_ops) == 14 + 9
    # Prefix (arch check + nr load) is shared, the socket/io_uring tail is identical.
    assert fallback_ops[:4] == base_ops[:4]
    assert fallback_ops[-10:] == base_ops[-10:]
    clone_nr, unshare_nr, clone3_nr = os_sandbox._LINUX_USERNS_SYSCALLS[os_sandbox.platform.machine().lower()]
    eperm = 0x00050000 | os_sandbox.errno.EPERM
    enosys = 0x00050000 | os_sandbox.errno.ENOSYS
    middle = fallback_ops[4:13]
    assert middle[0][3] == unshare_nr and middle[1][3] == eperm
    assert middle[2][3] == clone3_nr and middle[3][3] == enosys
    assert middle[4][3] == clone_nr
    assert middle[5] == (0x20, 0, 0, 16)  # load args[0] (clone flags)
    assert middle[6][3] == os_sandbox._CLONE_NEWUSER and middle[7][3] == eperm
    assert middle[8] == (0x20, 0, 0, 0)  # reload nr for the socket checks
    # Every jump target stays inside the program.
    for index, (code, jt, jf, _k) in enumerate(fallback_ops):
        if code in (0x15, 0x45):
            assert index + 1 + jt < len(fallback_ops) and index + 1 + jf < len(fallback_ops)


def _run_seccomp_program(program: bytes, *, nr: int, arch: int, arg0: int = 0) -> str:
    """Interpret the classic-BPF seccomp program the way the kernel would, for one syscall."""
    ops = [struct.unpack_from("=HBBI", program, i) for i in range(0, len(program), 8)]
    data = {0: nr, 4: arch, 16: arg0 & 0xFFFFFFFF, 20: arg0 >> 32}
    verdicts = {
        0x80000000: "KILL",
        0x7FFF0000: "ALLOW",
        0x00050000 | errno.EPERM: "EPERM",
        0x00050000 | errno.ENOSYS: "ENOSYS",
    }
    accumulator = 0
    pc = 0
    for _ in range(64):
        assert pc < len(ops), "program fell off its end"
        code, jt, jf, k = ops[pc]
        if code == 0x20:
            accumulator = data[k]
            pc += 1
        elif code == 0x15:
            pc += 1 + (jt if accumulator == k else jf)
        elif code == 0x45:
            pc += 1 + (jt if accumulator & k else jf)
        elif code == 0x06:
            return verdicts[k]
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")
    raise AssertionError("program did not terminate")


@pytest.mark.parametrize(
    "machine, arch, syscalls",
    [
        ("x86_64", 0xC000003E, {"read": 0, "socket": 41, "socketpair": 53, "clone": 56, "unshare": 272}),
        ("aarch64", 0xC00000B7, {"read": 63, "socket": 198, "socketpair": 199, "clone": 220, "unshare": 97}),
    ],
)
def test_seccomp_programs_decide_each_syscall_as_documented(monkeypatch, machine, arch, syscalls):
    monkeypatch.setattr(os_sandbox.platform, "machine", lambda: machine)
    for block_userns in (False, True):
        program = os_sandbox._linux_seccomp_program(block_userns = block_userns)
        run = lambda nr, arg0 = 0, arch = arch: _run_seccomp_program(program, nr = nr, arch = arch, arg0 = arg0)
        assert run(syscalls["read"], arch = 0x1234) == "KILL"
        assert run(syscalls["read"]) == "ALLOW"
        assert run(syscalls["socket"], 2) == "ALLOW"  # AF_INET
        assert run(syscalls["socket"], os_sandbox._AF_VSOCK) == "EPERM"
        assert run(syscalls["socketpair"], os_sandbox._AF_VSOCK) == "EPERM"
        assert run(425) == "EPERM"  # io_uring_setup
        if machine == "x86_64":
            assert run(syscalls["read"] | 0x40000000) == "EPERM"  # x32 ABI bit
        nested = "EPERM" if block_userns else "ALLOW"
        assert run(syscalls["unshare"], os_sandbox._CLONE_NEWUSER) == nested
        assert run(syscalls["clone"], os_sandbox._CLONE_NEWUSER | 17) == nested
        assert run(435) == ("ENOSYS" if block_userns else "ALLOW")  # clone3
        # A plain fork-style clone keeps working under the fallback filter.
        assert run(syscalls["clone"], 17) == "ALLOW"


@_LINUX_ONLY
def test_apparmor_userns_restriction_gets_profile_remediation(monkeypatch, isolated_capability_cache):
    raw = os_sandbox.SandboxCapability(
        "linux-bubblewrap",
        False,
        "the restrictive live probe failed (1): bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
    )
    monkeypatch.setattr(
        os_sandbox, "_read_text",
        lambda path: "1\n" if path == os_sandbox._APPARMOR_USERNS_SYSCTL else "",
    )
    explained = os_sandbox._explain_linux_probe_failure(raw)
    assert explained.reason.startswith("AppArmor restricts unprivileged user namespaces")
    assert "the restrictive live probe failed (1)" in explained.reason
    assert "/etc/apparmor.d/bwrap" in explained.remediation
    assert "apparmor_parser -r" in explained.remediation
    assert "Limited mode" in explained.remediation
    stamped = os_sandbox._capability_with_identity(
        explained, environment = "native_linux", fingerprint = "fp"
    )
    assert stamped.remediation == os_sandbox._APPARMOR_USERNS_REMEDIATION
    assert stamped.protection_state == "unavailable"

    # Same symptom without the sysctl: the explanation is not attached.
    monkeypatch.setattr(os_sandbox, "_read_text", lambda path: "0\n")
    assert os_sandbox._explain_linux_probe_failure(raw) == raw
    # A qualified result or an unrelated failure is never rewritten.
    monkeypatch.setattr(os_sandbox, "_read_text", lambda path: "1\n")
    other = os_sandbox.SandboxCapability("linux-bubblewrap", False, "Bubblewrap is not installed")
    assert os_sandbox._explain_linux_probe_failure(other) == other
    passed = os_sandbox.SandboxCapability("linux-bubblewrap", True, "restrictive live probe passed")
    assert os_sandbox._explain_linux_probe_failure(passed) == passed


def test_runtime_read_paths_include_every_symlink_hop(monkeypatch, tmp_path):
    real = Path(os.path.realpath(sys.executable))
    hop_c = tmp_path / "c" / "bin"
    hop_b = tmp_path / "b" / "bin"
    hop_a = tmp_path / "a" / "bin"
    for hop in (hop_a, hop_b, hop_c):
        hop.mkdir(parents = True)
    (hop_c / "python3.12").symlink_to(real)
    (hop_b / "python3").symlink_to(hop_c / "python3.12")
    (hop_a / "python").symlink_to("../../b/bin/python3")  # relative link
    chain = os_sandbox._symlink_chain(str(hop_a / "python"))
    assert chain == [str(hop_a / "python"), str(hop_b / "python3"), str(hop_c / "python3.12"), str(real)]

    monkeypatch.setattr(os_sandbox.sys, "executable", str(hop_a / "python"))
    paths = os_sandbox._runtime_read_paths()
    for expected in (
        hop_a / "python", hop_b / "python3", hop_c / "python3.12", real,
        hop_a, hop_b, hop_c, real.parent,
    ):
        assert str(expected) in paths, expected
    assert os.path.join(sys.prefix, "bin") in paths or not os.path.isdir(os.path.join(sys.prefix, "bin"))

    # A link loop terminates.
    loop = tmp_path / "loop"
    loop.symlink_to(loop.name)
    assert os_sandbox._symlink_chain(str(loop)) == [str(loop)]


class _TokenPrintingBackend:
    """Stands in for a qualified backend: the probe child only has to print the token."""

    identity = "test-token-backend"
    profile_id = "test-token-profile-v1"

    def __init__(self):
        self.specs = []

    def prepare(self, spec):
        self.specs.append(spec)
        return os_sandbox.PreparedSandboxLaunch(
            argv = (sys.executable, "-c", f"print({os_sandbox._PROBE_TOKEN!r})"),
            workdir = spec.workdir,
            env = spec.env,
            preexec_fn = None,
            backend = self.identity,
        )


def test_live_probe_tolerates_missing_ipv6_loopback(monkeypatch):
    real_socket = socket.socket

    class NoIPv6Socket(real_socket):
        def __init__(self, family = -1, *args, **kwargs):
            if family == socket.AF_INET6:
                raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
            super().__init__(family, *args, **kwargs)

    monkeypatch.setattr(os_sandbox.socket, "socket", NoIPv6Socket)
    backend = _TokenPrintingBackend()
    result = os_sandbox._live_probe(backend)
    assert result.qualified, result
    assert result.limitations == (os_sandbox._LIMITATION_IPV6_UNAVAILABLE,)
    payload = backend.specs[0].argv[-1]
    assert "::1" not in payload
    assert "127.0.0.1" in payload
    assert "libc.unshare(" in payload

    backend = _TokenPrintingBackend()
    monkeypatch.setattr(os_sandbox.socket, "socket", real_socket)
    result = os_sandbox._live_probe(backend)
    assert result.qualified, result
    assert result.limitations == ()
    if socket.has_ipv6:
        try:
            probe6 = real_socket(socket.AF_INET6)
            probe6.bind(("::1", 0))
            probe6.close()
        except OSError:
            pass
        else:
            assert "::1" in backend.specs[0].argv[-1]


@_LINUX_ONLY
def test_limited_mode_sweeps_marked_setsid_descendants(monkeypatch, tmp_path, isolated_capability_cache):
    """A `setsid` grandchild escapes the process group; the run marker sweep still reaps it."""
    if not shutil.which("setsid") or not shutil.which("sleep"):
        pytest.skip("setsid and sleep are required")
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    unavailable = os_sandbox.SandboxCapability(
        "linux-bubblewrap", False, "test: forced unavailable", available = False,
        environment = "native_linux", protection_state = "unavailable",
        probe_generation = "sweep-generation", environment_fingerprint = "fp",
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_k: unavailable)
    grant = tool_isolation.issue_limited_grant(
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        probe_generation = unavailable.probe_generation,
    )
    swept: list[str] = []
    real_sweep = inference_tools._sweep_marked_descendants

    def recording_sweep(marker):
        swept.append(marker)
        return real_sweep(marker)

    monkeypatch.setattr(inference_tools, "_sweep_marked_descendants", recording_sweep)
    duration = 1285
    result = inference_tools._bash_exec(
        f"setsid sleep {duration} >/dev/null 2>&1 < /dev/null & echo started; "
        f"printf '%s' \"${inference_tools._LIMITED_RUN_MARKER_ENV}\" > marker.txt",
        None,
        20,
        "t",
        tool_execution_mode = "limited",
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        limited_grant = grant.token,
    )
    assert "started" in result, result
    assert len(swept) == 1
    marker = swept[0]
    assert (workdir / "marker.txt").read_text() == marker, "the marker reached the child environment"
    time.sleep(0.2)
    needle = f"{inference_tools._LIMITED_RUN_MARKER_ENV}={marker}".encode()
    survivors = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as stream:
                if needle in stream.read().split(b"\0"):
                    survivors.append(entry)
        except OSError:
            continue
    assert survivors == [], survivors
    pgrep = subprocess.run(["pgrep", "-f", f"sleep {duration}$"], capture_output = True, text = True)
    assert pgrep.stdout.strip() == "", pgrep.stdout


@_LINUX_ONLY
def test_limited_child_has_no_new_privs_and_marker(monkeypatch, tmp_path, isolated_capability_cache):
    workdir = tmp_path / "limited-work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    unavailable = os_sandbox.SandboxCapability(
        "linux-bubblewrap", False, "test: forced unavailable", available = False,
        environment = "native_linux", protection_state = "unavailable",
        probe_generation = "nnp-generation", environment_fingerprint = "fp",
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_k: unavailable)
    grant = tool_isolation.issue_limited_grant(
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        probe_generation = unavailable.probe_generation,
    )
    records = []
    result = inference_tools._python_exec(
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "print('NNP', libc.prctl(39, 0, 0, 0, 0))\n"
        f"print('MARKER', len(os.environ.get({inference_tools._LIMITED_RUN_MARKER_ENV!r}, '')))\n",
        None,
        20,
        "t",
        tool_execution_mode = "limited",
        current_subject = "test:limited-user",
        tool_ui_session_id = "test-page",
        limited_grant = grant.token,
        launch_record_callback = records.append,
    )
    assert "NNP 1" in result, result
    assert "MARKER 32" in result, result
    assert records and records[0].effective_mode == "limited"
    assert records[0].limitations == ()


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


def test_private_tmp_unix_socket_round_trip(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import os, socket

path = os.path.join(os.environ['TMPDIR'], 'private.sock')
server = socket.socket(socket.AF_UNIX)
client = socket.socket(socket.AF_UNIX)
accepted = None
try:
    server.bind(path)
    server.listen(1)
    client.connect(path)
    accepted, _ = server.accept()
    client.sendall(b'client-to-server')
    assert accepted.recv(16) == b'client-to-server'
    accepted.sendall(b'server-to-client')
    assert client.recv(16) == b'server-to-client'
finally:
    if accepted is not None:
        accepted.close()
    client.close()
    server.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
assert not os.path.exists(path)
print('PRIVATE_UNIX_ROUND_TRIP_OK')
"""
    completed = _run_native(workdir, code)
    _assert_native_ok(completed)
    assert "PRIVATE_UNIX_ROUND_TRIP_OK" in completed.stdout


def test_private_socket_cleanup_after_assertion_failure(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import os, socket

path = os.path.join(os.environ['TMPDIR'], 'failure.sock')
server = socket.socket(socket.AF_UNIX)
try:
    try:
        server.bind(path)
        raise AssertionError('deliberate probe failure')
    finally:
        server.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
except AssertionError:
    pass
assert not os.path.exists(path)
print('PRIVATE_UNIX_FAILURE_CLEANUP_OK')
"""
    completed = _run_native(workdir, code)
    _assert_native_ok(completed)
    assert "PRIVATE_UNIX_FAILURE_CLEANUP_OK" in completed.stdout


def test_multiprocessing_resource_sharer_remains_supported(qualified_native_capability, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import os
from multiprocessing.resource_sharer import DupFd, stop

read_fd, write_fd = os.pipe()
shared_fd = DupFd(read_fd).detach()
os.write(write_fd, b'R')
assert os.read(shared_fd, 1) == b'R'
for fd in (read_fd, write_fd, shared_fd):
    os.close(fd)
stop()
print('RESOURCE_SHARER_OK')
"""
    completed = _run_native(workdir, code)
    _assert_native_ok(completed)
    assert "RESOURCE_SHARER_OK" in completed.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "PyTorch is unavailable")
def test_pytorch_tensor_sharing_remains_supported(qualified_native_capability, tmp_path):
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


def test_live_pyenv_style_symlink_chain_runs_inside_sandbox(qualified_native_capability, tmp_path):
    """`versions/3.x/bin/python -> python3 -> python3.x -> real` must exec inside the jail.

    Hosted toolcaches, pyenv and many venvs reach the interpreter through more
    than one link; each hop has to be visible inside the sandbox.
    """
    real = Path(os.path.realpath(sys.executable))
    version_bin = tmp_path / "versions" / "3.x" / "bin"
    version_bin.mkdir(parents = True)
    (version_bin / "python3.x").symlink_to(real)
    (version_bin / "python3").symlink_to("python3.x")
    (version_bin / "python").symlink_to("python3")
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend_root = str(Path(__file__).resolve().parents[1])
    helper = f"""
import os, subprocess, sys, types
logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
loggers = types.ModuleType('loggers')
loggers.get_logger = lambda name: logger
sys.modules['loggers'] = loggers
from core.inference import os_sandbox
assert os.path.abspath(sys.executable) == {str(version_bin / "python")!r}, sys.executable
capability = os_sandbox.sandbox_capability()
assert capability.qualified, capability
spec = os_sandbox.SandboxLaunchSpec(
    argv=(sys.executable, '-I', '-c', 'import sys; print("CHAIN_OK", sys.executable)'),
    workdir={str(workdir)!r},
    env={{'HOME': {str(workdir)!r}, 'TMPDIR': '/tmp', 'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8'}},
)
prepared = os_sandbox.prepare_tool_launch(spec)
try:
    for hop in ({str(version_bin / "python")!r}, {str(version_bin / "python3")!r}, {str(version_bin / "python3.x")!r}):
        assert hop in prepared.argv, hop
    completed = subprocess.run(
        prepared.argv, cwd=prepared.workdir, env=prepared.env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=20, close_fds=True, preexec_fn=prepared.preexec_fn,
        pass_fds=prepared.pass_fds,
    )
finally:
    prepared.cleanup()
assert completed.returncode == 0, completed.stderr
assert 'CHAIN_OK' in completed.stdout, completed.stdout
print('SYMLINK_CHAIN_SANDBOX_OK')
"""
    completed = subprocess.run(
        [str(version_bin / "python"), "-c", helper],
        cwd = backend_root,
        env = {**os.environ, "PYTHONPATH": backend_root},
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 90,
        close_fds = True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SYMLINK_CHAIN_SANDBOX_OK" in completed.stdout


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
