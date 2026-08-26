# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import shlex
import socket
import sys
import threading
import uuid
from pathlib import Path

import pytest

from core.agent_workspace import execution
from core.agent_workspace.execution import (
    ExecutionBoundaryStatus,
    ProjectExecutionBoundary,
    ProjectExecutionUnavailable,
    execution_boundary_status,
)
from core.agent_workspace.verification import execute_check


def _python_command(source: str, *arguments: object) -> str:
    argv = [str(Path(sys.executable).resolve()), "-c", source, *map(str, arguments)]
    return " ".join(shlex.quote(part) for part in argv)


def test_unsupported_platforms_do_not_advertise_project_execution():
    windows = execution_boundary_status("win32", probe = False)
    unknown = execution_boundary_status("plan9", probe = False)

    assert windows.available is False
    assert windows.backend is None
    assert "Windows" in str(windows.reason)
    assert unknown.available is False
    assert unknown.backend is None


def test_linux_without_bubblewrap_does_not_advertise_execution(monkeypatch):
    monkeypatch.setattr(execution, "_bubblewrap_path", lambda: None)

    status = execution_boundary_status("linux", probe = False)

    assert status.available is False
    assert status.backend is None
    assert "bubblewrap" in str(status.reason)


def test_boundary_rejects_root_replacement_between_identity_check_and_open(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    original = tmp_path / "original"
    root.mkdir()
    metadata = root.stat()
    expected = (metadata.st_dev, metadata.st_ino)
    real_open = execution.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == root:
            swapped = True
            root.rename(original)
            root.mkdir()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(True, "sandbox-exec", None),
    )
    monkeypatch.setattr(execution.os, "open", racing_open)

    with pytest.raises(ProjectExecutionUnavailable, match = "identity changed"):
        ProjectExecutionBoundary.open(root, expected)


def test_policy_parameters_reject_control_character_paths(tmp_path, monkeypatch):
    root = tmp_path / "repository\nreplacement"
    root.mkdir()
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(True, "sandbox-exec", None),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "control characters"):
        ProjectExecutionBoundary.open(root)


def test_macos_profile_is_self_contained_and_default_deny():
    profile = execution._MACOS_PROFILE

    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "system.sb" not in profile
    assert "(allow file-read*)" not in profile
    assert '(subpath "/Applications")' not in profile
    assert '(subpath "/Library")' not in profile
    assert '(subpath "/private/var/db")' not in profile


@pytest.mark.skipif(not hasattr(os, "link"), reason = "hardlinks are unavailable")
def test_boundary_rejects_regular_file_hardlinked_outside_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding = "utf-8")
    os.link(outside, root / "alias.txt")
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(True, "sandbox-exec", None),
    )

    with ProjectExecutionBoundary.open(root) as boundary:
        with pytest.raises(ProjectExecutionUnavailable, match = "hard-linked outside"):
            boundary.popen_kwargs()

    assert outside.read_text(encoding = "utf-8") == "preserve"


@pytest.mark.skipif(not hasattr(os, "link"), reason = "hardlinks are unavailable")
def test_boundary_fails_closed_when_a_hardlink_directory_cannot_be_listed(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    hidden = root / "execute-only"
    hidden.mkdir(parents = True)
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding = "utf-8")
    os.link(outside, hidden / "alias.txt")
    hidden.chmod(0o300)
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(True, "sandbox-exec", None),
    )

    try:
        with ProjectExecutionBoundary.open(root) as boundary:
            with pytest.raises(ProjectExecutionUnavailable, match = "safety was checked"):
                boundary.popen_kwargs()
    finally:
        hidden.chmod(0o700)

    assert outside.read_text(encoding = "utf-8") == "preserve"


@pytest.mark.skipif(not hasattr(os, "link"), reason = "hardlinks are unavailable")
def test_boundary_allows_hardlinks_fully_contained_in_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    original = root / "original.txt"
    original.write_text("internal", encoding = "utf-8")
    os.link(original, root / "alias.txt")
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(True, "sandbox-exec", None),
    )

    with ProjectExecutionBoundary.open(root) as boundary:
        options = boundary.popen_kwargs()

    assert "pass_fds" in options


def test_linux_runtime_socket_trees_are_masked_as_one_private_root(tmp_path):
    home = tmp_path / "home"
    temp = tmp_path / "temp"
    home.mkdir()
    temp.mkdir()

    masked = execution._linux_masked_roots(home.resolve(), temp.resolve())

    for requested in (home.resolve(), temp.resolve()):
        assert any(requested == hidden or requested.is_relative_to(hidden) for hidden in masked)
    assert not any(
        child != parent and child.is_relative_to(parent) for child in masked for parent in masked
    )
    if Path("/run").is_dir():
        run_root = Path("/run").resolve()
        var_run_root = Path("/var/run").resolve()
        assert run_root in masked
        assert masked.count(run_root) == 1
        if var_run_root != run_root:
            assert var_run_root not in masked
        assert not any(path == run_root / "media" for path in masked)


def _writable_linux_runtime_directory() -> Path | None:
    candidates = []
    configured = os.environ.get("XDG_RUNTIME_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend((Path(f"/run/user/{os.getuid()}"), Path("/run/user"), Path("/run")))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict = True)
            resolved.relative_to(Path("/run"))
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_dir() and os.access(resolved, os.W_OK | os.X_OK):
            return resolved
    return None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux bubblewrap integration")
def test_linux_boundary_denies_host_runtime_unix_socket(tmp_path):
    status = execution_boundary_status()
    if not status.available or status.backend != "bubblewrap":
        pytest.skip(status.reason or "bubblewrap is unavailable")
    runtime = _writable_linux_runtime_directory()
    if runtime is None:
        pytest.skip("no writable pathname below /run for the AF_UNIX integration fixture")

    project = tmp_path / "project"
    project.mkdir()
    socket_path = runtime / f"unsloth-boundary-{uuid.uuid4().hex[:12]}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(4)
    source = """
import socket, sys
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(0.5)
try:
    client.connect(sys.argv[1])
except OSError:
    print('denied')
else:
    print('connected')
finally:
    client.close()
"""
    try:
        # Prove the fixture is reachable on the host before asking the sandbox
        # to deny the same pathname socket.
        host_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        host_client.settimeout(0.5)
        try:
            host_client.connect(str(socket_path))
        finally:
            host_client.close()
        result = execute_check(
            {
                "name": "runtime-socket-boundary",
                "command": _python_command(source, socket_path),
                "timeoutSeconds": 10,
                "logLimitBytes": 8192,
            },
            root = project,
            cancel_event = threading.Event(),
            run_id = "linux-runtime-socket-boundary",
        )
    finally:
        listener.close()
        socket_path.unlink(missing_ok = True)

    assert result["status"] == "passed", result
    assert result["output"].strip() == "denied"


def _coordination_boundary(identity: tuple[int, int]) -> ProjectExecutionBoundary:
    boundary = object.__new__(ProjectExecutionBoundary)
    boundary._root_identity = identity
    boundary._execution_slot = None
    boundary._closed = False
    boundary.recheck = lambda: None
    return boundary


def test_same_workspace_commands_are_serialized_and_waiters_can_cancel():
    first = _coordination_boundary((101, 202))
    second = _coordination_boundary((101, 202))
    started = threading.Event()
    acquired = threading.Event()
    result: list[bool] = []
    assert first.acquire_execution_slot()

    def wait_for_slot() -> None:
        started.set()
        result.append(second.acquire_execution_slot())
        acquired.set()

    waiter = threading.Thread(target = wait_for_slot)
    try:
        waiter.start()
        assert started.wait(timeout = 1)
        assert not acquired.wait(timeout = 0.1)
        first.release_execution_slot()
        assert acquired.wait(timeout = 1)
        waiter.join(timeout = 1)
        assert result == [True]
    finally:
        first.release_execution_slot()
        second.release_execution_slot()

    holder = _coordination_boundary((101, 202))
    cancelled = _coordination_boundary((101, 202))
    cancel_event = threading.Event()
    try:
        assert holder.acquire_execution_slot()
        cancel_event.set()
        assert cancelled.acquire_execution_slot(cancel_event) is False
    finally:
        holder.release_execution_slot()
        cancelled.release_execution_slot()


@pytest.mark.skipif(sys.platform != "darwin", reason = "macOS Seatbelt integration")
def test_macos_boundary_denies_private_reads_and_network_egress(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    host_temp = tmp_path / "host-temp-secret.txt"
    host_temp.write_text("secret", encoding = "utf-8")

    tcp_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_listener.bind(("127.0.0.1", 0))
    tcp_listener.listen(1)
    unix_path = Path("/private/tmp") / f"unsloth-boundary-{uuid.uuid4().hex[:12]}.sock"
    unix_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix_listener.bind(str(unix_path))
    unix_listener.listen(1)
    source = """
import json, os, socket, sys
from pathlib import Path

def readable(path):
    try:
        with open(path, 'rb') as source:
            source.read(1)
        return True
    except OSError:
        return False

def connected(family, address):
    client = socket.socket(family, socket.SOCK_STREAM)
    client.settimeout(0.5)
    try:
        client.connect(address)
        return True
    except OSError:
        return False
    finally:
        client.close()

Path('inside.txt').write_text('confined')
scratch_ok = True
try:
    Path(os.environ['HOME'], 'scratch.txt').write_text('scratch')
except OSError:
    scratch_ok = False
print(json.dumps({
    'home': readable(sys.argv[1]),
    'temp': readable(sys.argv[2]),
    'volumes': readable('/Volumes'),
    'tcp': connected(socket.AF_INET, ('127.0.0.1', int(sys.argv[3]))),
    'unix': connected(socket.AF_UNIX, sys.argv[4]),
    'scratch': scratch_ok,
}))
"""
    try:
        result = execute_check(
            {
                "name": "boundary",
                "command": _python_command(
                    source,
                    Path.home(),
                    host_temp,
                    tcp_listener.getsockname()[1],
                    unix_path,
                ),
                "timeoutSeconds": 10,
                "logLimitBytes": 8192,
            },
            root = project,
            cancel_event = threading.Event(),
            run_id = "macos-boundary",
        )
    finally:
        unix_listener.close()
        tcp_listener.close()
        unix_path.unlink(missing_ok = True)

    assert result["status"] == "passed", result
    observed = json.loads(result["output"])
    assert observed == {
        "home": False,
        "temp": False,
        "volumes": False,
        "tcp": False,
        "unix": False,
        "scratch": True,
    }
    assert (project / "inside.txt").read_text(encoding = "utf-8") == "confined"
