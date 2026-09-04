# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused profile and Darwin-live tests for the macOS Seatbelt preview."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import os_sandbox
from core.inference import tools as inference_tools


def _spec(
    workdir: Path,
    *argv: str,
    env: dict[str, str] | None = None,
):
    return os_sandbox.ToolLaunchPlan(
        argv = tuple(argv) or (sys.executable, "-I", "-c", "pass"),
        workdir = str(workdir),
        env = env
        or {
            "HOME": str(workdir),
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONIOENCODING": "utf-8",
        },
    )


def test_probe_accepts_only_the_system_owned_launcher_but_reports_preview(monkeypatch):
    backend = os_sandbox.MacOSSeatbeltBackend()
    launcher = SimpleNamespace(st_mode = stat.S_IFREG | 0o755, st_uid = 0)
    monkeypatch.setattr(os_sandbox.os, "stat", lambda *_a, **_k: launcher)
    monkeypatch.setattr(
        os_sandbox,
        "_live_macos_probe",
        lambda current: os_sandbox.SandboxCapability(
            current.identity,
            False,
            "live probe passed",
            available = True,
            protection_state = "preview",
        ),
    )

    capability = backend.probe()

    assert backend._sandbox_exec == "/usr/bin/sandbox-exec"
    assert capability.available is True
    assert capability.qualified is False
    assert capability.protection_state == "preview"
    assert capability.profile_id == "macos-seatbelt-preview-v1"
    assert capability.limitations == (
        "deprecated_undocumented_sbpl",
        "detached_descendant_cleanup_unverified",
    )
    assert "detached setsid/double-fork descendants remains unverified" in capability.reason


@pytest.mark.parametrize(
    "mode,uid",
    [
        (stat.S_IFDIR | 0o755, 0),
        (stat.S_IFREG | 0o755, 501),
        (stat.S_IFREG | 0o775, 0),
        (stat.S_IFREG | 0o757, 0),
    ],
)
def test_probe_rejects_an_untrusted_launcher(mode, uid, monkeypatch):
    backend = os_sandbox.MacOSSeatbeltBackend()
    monkeypatch.setattr(
        os_sandbox.os,
        "stat",
        lambda *_a, **_k: SimpleNamespace(st_mode = mode, st_uid = uid),
    )

    capability = backend.probe()

    assert capability.available is False
    assert capability.qualified is False
    assert backend._sandbox_exec is None
    assert "root-owned, non-user-writable regular file" in capability.reason


def test_sbpl_path_json_encodes_a_canonical_path(tmp_path):
    path = tmp_path / "unicode π"
    path.mkdir()

    encoded, is_directory = os_sandbox._sbpl_path(str(path))

    assert json.loads(encoded) == os.path.realpath(path)
    assert is_directory is True


@pytest.mark.parametrize("unsafe", ["", "relative", "bad\0path", "bad\npath", "bad\rpath"])
def test_sbpl_path_rejects_unsafe_spellings(unsafe):
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "absolute.*NUL/newline"):
        os_sandbox._sbpl_path(unsafe)


def test_sbpl_path_rejects_a_missing_absolute_path(tmp_path):
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "does not exist"):
        os_sandbox._sbpl_path(str(tmp_path / "missing"))


def test_sbpl_filters_include_a_validated_runtime_alias(monkeypatch, tmp_path):
    alias = tmp_path / "python"
    target = tmp_path / "python3.12"
    target.touch()
    monkeypatch.setattr(
        os_sandbox.os.path,
        "realpath",
        lambda path: str(target) if os.fspath(path) == str(alias) else os.fspath(path),
    )
    monkeypatch.setattr(os_sandbox.os.path, "exists", lambda _path: True)

    filters = os_sandbox._sbpl_path_filters((str(alias),))

    assert f"(literal {json.dumps(str(alias))})" in filters
    assert f"(literal {json.dumps(str(target))})" in filters


def test_sbpl_ancestor_filters_are_literal_and_not_global(tmp_path):
    runtime = tmp_path / "Frameworks" / "Python.framework" / "Python"
    runtime.parent.mkdir(parents = True)
    runtime.touch()

    filters = os_sandbox._sbpl_ancestor_filters((str(runtime),))

    assert f"(literal {json.dumps(str(runtime.parent))})" in filters
    assert f"(literal {json.dumps(str(tmp_path))})" in filters
    assert all(filter_.startswith("(literal ") for filter_ in filters)


def test_profile_is_deny_default_with_narrow_filesystem_and_unix_socket_rules(
    monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    private_tmp = tmp_path / "private"
    runtime = tmp_path / "runtime"
    workdir.mkdir()
    private_tmp.mkdir()
    runtime.mkdir()
    devices = []
    for name in ("null", "zero", "random", "urandom"):
        device = tmp_path / name
        device.touch()
        devices.append(str(device))
    denied = []
    for name in ("open", "osascript", "security", "launchctl", "sandbox-exec"):
        executable = tmp_path / name
        executable.touch()
        denied.append(str(executable))
    monkeypatch.setattr(os_sandbox, "_MACOS_READ_ROOTS", ())
    monkeypatch.setattr(os_sandbox, "_MACOS_DEVICES", tuple(devices))
    monkeypatch.setattr(os_sandbox, "_MACOS_DENIED_EXECUTABLES", tuple(denied))

    profile = os_sandbox._macos_seatbelt_profile(
        workdir = str(workdir),
        private_tmp = str(private_tmp),
        runtime_paths = (str(runtime),),
    )
    encoded_tmp = json.dumps(os.path.realpath(private_tmp))

    assert profile.startswith("(version 1)\n(deny default)\n")
    metadata_line = next(
        line for line in profile.splitlines() if line.startswith("(allow file-read-metadata")
    )
    assert metadata_line != "(allow file-read-metadata)"
    assert f"(literal {json.dumps(str(runtime.parent))})" in metadata_line
    deny_line = next(line for line in profile.splitlines() if line.startswith("(deny process-exec"))
    assert all(json.dumps(os.path.realpath(path)) in deny_line for path in denied)
    assert f"(allow network-bind (local unix-socket (subpath {encoded_tmp})))" in profile
    assert f"(allow network-outbound (remote unix-socket (subpath {encoded_tmp})))" in profile
    assert "AF_INET" not in profile
    assert "network-inbound" not in profile
    assert "(allow network*)" not in profile
    assert "(allow network-outbound)" not in profile


def test_prepare_uses_a_private_environment_and_removes_it_on_cleanup(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    runtime = tmp_path / "runtime"
    workdir.mkdir()
    runtime.mkdir()
    backend = os_sandbox.MacOSSeatbeltBackend()
    backend._sandbox_exec = "/usr/bin/sandbox-exec"
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: (str(runtime),))
    prepared = backend.prepare(
        _spec(
            workdir,
            env = {
                "KEEP": "yes",
                "DYLD_INSERT_LIBRARIES": "/host/inject.dylib",
                "DISPLAY": ":0",
                "SSH_AUTH_SOCK": "/host/agent.sock",
                "XPC_SERVICE_NAME": "host.service",
            },
        )
    )
    private_tmp = Path(prepared.cleanup_paths[0])

    assert private_tmp.is_dir()
    assert prepared.env["KEEP"] == "yes"
    assert prepared.env["HOME"] == os.path.realpath(workdir)
    assert {prepared.env[name] for name in ("TMPDIR", "TMP", "TEMP", "XDG_RUNTIME_DIR")} == {
        str(private_tmp)
    }
    assert not any(
        name in prepared.env
        for name in ("DYLD_INSERT_LIBRARIES", "DISPLAY", "SSH_AUTH_SOCK", "XPC_SERVICE_NAME")
    )

    prepared.cleanup()

    assert not private_tmp.exists()


def test_prepare_failure_removes_its_private_temp(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    private_tmp = tmp_path / "seatbelt-private"
    private_tmp.mkdir()
    backend = os_sandbox.MacOSSeatbeltBackend()
    backend._sandbox_exec = "/usr/bin/sandbox-exec"
    monkeypatch.setattr(os_sandbox.tempfile, "mkdtemp", lambda **_k: str(private_tmp))
    monkeypatch.setattr(os_sandbox, "_runtime_read_paths", lambda: ())
    monkeypatch.setattr(
        os_sandbox,
        "_macos_seatbelt_profile",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("profile failed")),
    )

    with pytest.raises(RuntimeError, match = "profile failed"):
        backend.prepare(_spec(workdir))

    assert not private_tmp.exists()


@pytest.fixture(scope = "module")
def live_seatbelt_backend():
    if sys.platform != "darwin":
        pytest.skip("native Seatbelt tests run only on Darwin")
    backend = os_sandbox.MacOSSeatbeltBackend()
    capability = backend.probe()
    assert capability.available is True, capability.reason
    assert capability.qualified is False
    assert capability.protection_state == "preview"
    assert "detached_descendant_cleanup_unverified" in capability.limitations
    return backend


def _run_native(
    backend: os_sandbox.MacOSSeatbeltBackend,
    workdir: Path,
    code: str,
    *,
    timeout: int = 20,
) -> subprocess.CompletedProcess[str]:
    prepared = backend.prepare(_spec(workdir, sys.executable, "-I", "-c", code))
    private_tmp = Path(prepared.cleanup_paths[0])
    try:
        completed = subprocess.run(
            prepared.argv,
            cwd = prepared.workdir,
            env = prepared.env,
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            timeout = timeout,
            close_fds = prepared.close_fds,
            preexec_fn = prepared.preexec_fn,
        )
    finally:
        prepared.cleanup()
    assert not private_tmp.exists()
    assert completed.returncode == 0, (
        f"Seatbelt command failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def test_live_workdir_boundary_symlinks_and_native_extension(live_seatbelt_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    inside = workdir / "input.txt"
    inside.write_text("inside", encoding = "utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding = "utf-8")
    outside_write = tmp_path / "outside-write.txt"
    home_secret = Path.home() / f".unsloth-seatbelt-secret-{os.getpid()}"
    home_secret.write_text("home-secret", encoding = "utf-8")
    repo_file = Path(__file__).resolve()
    (workdir / "escape-read").symlink_to(outside)
    (workdir / "escape-write").symlink_to(outside_write)
    code = f"""
import _ssl, os

def denied_read(path):
    try:
        open(path, 'rb').read(1)
    except OSError:
        return
    raise AssertionError('read escaped workdir: ' + path)

def denied_write(path):
    try:
        open(path, 'wb').write(b'escape')
    except OSError:
        return
    raise AssertionError('write escaped workdir: ' + path)

assert open('input.txt', encoding='utf-8').read() == 'inside'
open('output.txt', 'w', encoding='utf-8').write('written')
denied_read({str(outside)!r})
denied_read({str(home_secret)!r})
denied_read({str(repo_file)!r})
denied_write({str(outside_write)!r})
denied_read('escape-read')
denied_write('escape-write')
assert _ssl.__file__
with open(_ssl.__file__, 'rb') as stream:
    assert stream.read(1)
print('SEATBELT_FILESYSTEM_OK')
"""

    try:
        completed = _run_native(live_seatbelt_backend, workdir, code)
    finally:
        home_secret.unlink(missing_ok = True)

    assert "SEATBELT_FILESYSTEM_OK" in completed.stdout
    assert (workdir / "output.txt").read_text(encoding = "utf-8") == "written"
    assert outside.read_text(encoding = "utf-8") == "secret"
    assert not outside_write.exists()


def test_live_private_temp_is_fresh(live_seatbelt_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    marker = "seatbelt-private-temp-marker"
    _run_native(
        live_seatbelt_backend,
        workdir,
        f"import os; open(os.path.join(os.environ['TMPDIR'], {marker!r}), 'w').write('x')",
    )
    completed = _run_native(
        live_seatbelt_backend,
        workdir,
        f"import os; assert not os.path.exists(os.path.join(os.environ['TMPDIR'], {marker!r})); print('FRESH')",
    )
    assert completed.stdout.strip() == "FRESH"


def _listen(family, address):
    server = socket.socket(family)
    server.bind(address)
    server.listen(1)
    return server, server.getsockname()


def test_live_ip_dns_and_host_unix_are_denied_but_private_unix_works(
    live_seatbelt_backend, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    ipv4, ipv4_address = _listen(socket.AF_INET, ("127.0.0.1", 0))
    ipv6, ipv6_address = _listen(socket.AF_INET6, ("::1", 0))
    host_unix = socket.socket(socket.AF_UNIX)
    with tempfile.TemporaryDirectory(prefix = "us-host-", dir = "/tmp") as host_root:
        host_socket_path = os.path.join(host_root, "host.sock")
        host_unix.bind(host_socket_path)
        host_unix.listen(1)
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
    raise AssertionError('host endpoint reachable: ' + repr(address))

denied(socket.AF_INET, {ipv4_address!r})
denied(socket.AF_INET6, {ipv6_address!r})
denied(socket.AF_UNIX, {host_socket_path!r})
try:
    socket.getaddrinfo('example.com', 443)
except OSError:
    pass
else:
    raise AssertionError('DNS resolution escaped Seatbelt')
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
print('SEATBELT_NETWORK_OK')
"""
        try:
            completed = _run_native(live_seatbelt_backend, workdir, code)
        finally:
            ipv4.close()
            ipv6.close()
            host_unix.close()

    assert "SEATBELT_NETWORK_OK" in completed.stdout


def test_live_multiprocessing_and_resource_sharer(live_seatbelt_backend, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import multiprocessing as mp
import os
from multiprocessing.resource_sharer import DupFd, stop

def child(connection):
    connection.send('child-ok')
    connection.close()

context = mp.get_context('fork')
parent, child_connection = context.Pipe()
process = context.Process(target=child, args=(child_connection,))
process.start()
child_connection.close()
assert parent.recv() == 'child-ok'
process.join(10)
assert process.exitcode == 0
parent.close()
read_fd, write_fd = os.pipe()
shared_fd = DupFd(read_fd).detach()
os.write(write_fd, b'R')
assert os.read(shared_fd, 1) == b'R'
for fd in (read_fd, write_fd, shared_fd):
    os.close(fd)
stop()
print('SEATBELT_MULTIPROCESSING_OK')
"""

    completed = _run_native(live_seatbelt_backend, workdir, code)

    assert "SEATBELT_MULTIPROCESSING_OK" in completed.stdout


def test_live_pytorch_transfer_when_installed(live_seatbelt_backend, tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed; standard-library resource sharing remains covered")
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = """
import torch
import torch.multiprocessing as mp

def child(queue):
    queue.put(torch.arange(4))

context = mp.get_context('fork')
queue = context.Queue()
process = context.Process(target=child, args=(queue,))
process.start()
tensor = queue.get(timeout=10)
process.join(10)
assert process.exitcode == 0
assert torch.equal(tensor, torch.arange(4))
print('SEATBELT_PYTORCH_OK')
"""

    completed = _run_native(live_seatbelt_backend, workdir, code, timeout = 30)

    assert "SEATBELT_PYTORCH_OK" in completed.stdout


def test_live_real_python_and_terminal_stream_and_record_limitations(
    live_seatbelt_backend, monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(inference_tools, "_check_code_safety", lambda _code: None)
    monkeypatch.setattr(inference_tools, "_find_blocked_commands", lambda _command: set())
    chunks: list[str] = []
    records: list[os_sandbox.ToolExecutionRecord] = []

    python_result = inference_tools._python_exec(
        "print('python-seatbelt-stream')",
        timeout = 20,
        output_callback = chunks.append,
        launch_record_callback = records.append,
    )
    terminal_result = inference_tools._bash_exec(
        "printf 'terminal-seatbelt-stream\\n'",
        timeout = 20,
        output_callback = chunks.append,
        launch_record_callback = records.append,
    )

    assert python_result.strip() == "python-seatbelt-stream"
    assert terminal_result.strip() == "terminal-seatbelt-stream"
    assert "python-seatbelt-stream\n" in chunks
    assert "terminal-seatbelt-stream\n" in chunks
    assert len(records) == 2
    for record in records:
        assert record.backend == "macos-seatbelt"
        assert record.os_isolation is True
        assert record.limitations == live_seatbelt_backend.limitations


@pytest.mark.parametrize("termination", ["python-timeout", "terminal-cancellation"])
def test_live_timeout_and_cancellation_kill_an_ordinary_process_group(
    termination, live_seatbelt_backend, monkeypatch, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    marker = workdir / "ordinary-child-survived"
    monkeypatch.setattr(inference_tools, "_get_workdir", lambda _session: str(workdir))
    monkeypatch.setattr(inference_tools, "_check_code_safety", lambda _code: None)
    monkeypatch.setattr(inference_tools, "_harden_parent_against_proc_env_leak", lambda: True)
    if termination == "python-timeout":
        child = f"import time; time.sleep(2); open({str(marker)!r}, 'w', encoding='utf-8').write('escaped')"
        code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "time.sleep(30)\n"
        )
        result = inference_tools._python_exec(code, timeout = 1)
        assert result == "Execution timed out after 1 seconds."
    else:
        cancel = threading.Event()
        timer = threading.Timer(0.5, cancel.set)
        timer.start()
        try:
            command = f"(sleep 2; printf escaped > {str(marker)!r}) & sleep 30"
            result = inference_tools._bash_exec(command, cancel_event = cancel, timeout = 20)
        finally:
            timer.cancel()
        assert result == "Execution cancelled."

    time.sleep(2.5)
    assert not marker.exists()


def test_live_detached_descendant_limitation_is_demonstrated_and_cleaned(
    live_seatbelt_backend, tmp_path
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    pid_file = workdir / "detached.pid"
    code = f"""
import os, time

first = os.fork()
if first == 0:
    os.setsid()
    second = os.fork()
    if second == 0:
        for fd in (0, 1, 2):
            try:
                os.close(fd)
            except OSError:
                pass
        open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))
        time.sleep(60)
        os._exit(0)
    os._exit(0)
os.waitpid(first, 0)
"""

    _run_native(live_seatbelt_backend, workdir, code)
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "detached Seatbelt diagnostic did not publish its PID"
    pid = int(pid_file.read_text(encoding = "utf-8"))
    try:
        os.kill(pid, 0)
        assert "detached_descendant_cleanup_unverified" in live_seatbelt_backend.limitations
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        cleaned = False
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                cleaned = True
                break
            time.sleep(0.05)
        assert cleaned, f"detached Seatbelt diagnostic PID {pid} was not cleaned up"


def test_live_twenty_seatbelt_startup_samples(live_seatbelt_backend, tmp_path, record_property):
    workdir = tmp_path / "work"
    workdir.mkdir()
    samples: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        completed = _run_native(live_seatbelt_backend, workdir, "print('started')")
        samples.append((time.perf_counter() - started) * 1000)
        assert completed.stdout.strip() == "started"
    ordered = sorted(samples)
    median = statistics.median(samples)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    record_property("macos_seatbelt_startup_samples_ms", json.dumps(samples))
    record_property("macos_seatbelt_startup_median_ms", f"{median:.3f}")
    record_property("macos_seatbelt_startup_p95_ms", f"{p95:.3f}")
    print(f"Seatbelt startup over 20 launches: median={median:.3f}ms p95={p95:.3f}ms")
