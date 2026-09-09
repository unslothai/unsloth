# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Bounded control channel for the shipped, pinned SRT Node helper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import site
import shutil
import subprocess
import sys
import time
import threading

RUNTIME = Path(__file__).with_name("srt_runtime")
MAX_CONTROL = 16_384
MAX_REQUEST = 262_144


class SrtError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        code = "probe_failed",
        stage = "launch",
        dependency = None,
        details = None,
    ):
        super().__init__(message)
        from .srt_diagnostics import ProbeReason
        self.diagnostic = ProbeReason(code, stage, dependency, details)


def installation_identity() -> str:
    digest = hashlib.sha256()
    for name in (
        "bridge.mjs",
        "integrity.json",
        "package-lock.json",
        "installed-runtime-settings.json",
        "windows-read-owner.mjs",
        "windows-read-lease.mjs",
        "windows-read-recover.mjs",
    ):
        path = RUNTIME / name
        digest.update(name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"missing")
    for name in (
        "srt_adapter.py",
        "srt_probe.py",
        "srt_diagnostics.py",
        "srt_windows_owner.py",
        "srt_windows_read_lease.py",
    ):
        try:
            digest.update(Path(__file__).with_name(name).read_bytes())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def node_executable() -> str:
    # Resolve once to an absolute executable. The workload never supplies this path.
    selected = shutil.which("node")
    if not selected:
        raise SrtError(
            "Node.js >=20.11 is required for the installed Studio SRT helper",
            code = "dependency_missing",
            stage = "dependency",
            dependency = "node",
        )
    return os.path.realpath(selected)


def read_roots(executable: str) -> list[str]:
    roots = [
        os.path.dirname(os.path.abspath(executable)),
        sys.prefix,
        sys.base_prefix,
        str(Path(__file__).with_name("sandbox_site")),
        *site.getsitepackages(),
    ]
    return sorted({os.path.realpath(root) for root in roots if os.path.exists(root)})


def request_for(
    argv,
    cwd,
    env,
    timeout,
    *,
    operation = "run",
    additional_read_roots = (),
) -> dict:
    try:
        if sys.platform != "win32":
            raise SrtError(
                "Studio SRT is Windows-only", code = "operation_unsupported", stage = "policy"
            )
        request = _request_for(
            argv,
            cwd,
            env,
            timeout,
            operation = operation,
            additional_read_roots = additional_read_roots,
        )
        return request
    except SrtError as exc:
        if exc.diagnostic.code == "probe_failed":
            from .srt_diagnostics import ProbeReason
            exc.diagnostic = ProbeReason("policy_invalid", "policy")
        raise


def _request_for(
    argv,
    cwd,
    env,
    timeout,
    *,
    operation = "run",
    additional_read_roots = (),
) -> dict:
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
        or timeout > 86_400
    ):
        raise SrtError("SRT tool timeout must be positive and at most 24 hours")
    executable = (
        argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0], path = env.get("PATH", ""))
    )
    if not executable:
        raise SrtError(
            "The selected tool executable is unavailable",
            code = "dependency_missing",
            stage = "dependency",
            dependency = "selected_interpreter",
        )
    # Keep the venv's lexical executable: resolving its symlink to the base
    # Python changes pyvenv.cfg discovery and silently drops selected packages.
    executable = os.path.abspath(executable)
    roots = read_roots(executable)
    if sys.platform == "win32":
        # Use the tool's selected shell PATH; the host PATH may point to WSL.
        bash = shutil.which("bash", path = env.get("PATH", ""))
        if bash and "git" in bash.lower():
            roots.append(str(Path(bash).parent.parent))
    # Trust-store aliases can cross a restored runtime symlink. Binding the
    # alias again would traverse a read-only mount; expose its target instead.
    roots = sorted(set([*roots, *(os.path.realpath(path) for path in additional_read_roots)]))
    if sys.platform == "win32":
        # SRT applies inheriting ACEs. Avoid traversing the selected environment
        # again for nested site-packages during every grant and reset.
        minimal = []
        selected_roots = []
        # Standard users already have read/execute access to machine runtimes.
        # Their ACLs belong to administrators; stamping them at tool launch
        # fails for an ordinary Studio user (notably Program Files/Git).
        system_roots = [
            os.path.normcase(os.path.realpath(value))
            for name in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")
            if (value := os.environ.get(name))
        ]
        for root in sorted(roots, key = len):
            folded = os.path.normcase(os.path.realpath(root))
            if any(
                folded == parent or folded.startswith(parent + os.sep) for parent in system_roots
            ):
                continue
            if not any(
                folded == parent or folded.startswith(parent + os.sep) for parent in minimal
            ):
                minimal.append(folded)
                selected_roots.append(os.path.realpath(root))
        roots = selected_roots
    if any(os.path.realpath(root) == os.path.sep for root in roots):
        raise SrtError("Root filesystem read grant is forbidden")
    denied = []
    roots = [
        root
        for root in roots
        if not any(root == mount or root.startswith(mount + os.sep) for mount in denied)
    ]
    request = {
        "v": 1,
        "operation": operation,
        "executable": executable,
        "argv": list(argv[1:]),
        "cwd": os.path.realpath(cwd),
        "env": dict(env),
        "readRoots": roots,
        "writeRoots": [os.path.realpath(cwd)],
        "denyReadRoots": denied,
        "privateUnixSockets": sys.platform == "linux",
        "timeoutMs": None if timeout is None else max(1, int(timeout * 1000)),
    }
    if sys.platform == "win32":
        settings_path = RUNTIME / "installed-runtime-settings.json"
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding = "utf-8"))
                ports = settings["windowsProxyPortRange"]
                if (
                    set(settings) != {"windowsProxyPortRange"}
                    or not isinstance(ports, list)
                    or len(ports) != 2
                    or any(type(port) is not int for port in ports)
                    or not 1024 <= ports[0] < ports[1] <= 65535
                    or ports[1] - ports[0] > 99
                ):
                    raise ValueError("invalid port range")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SrtError("Invalid installed SRT Windows port settings; rerun setup") from exc
            request["windowsProxyPortRange"] = ports
    return request


def spawn(
    request: dict,
    *,
    cancel_event = None,
    launch_deadline = None,
    **kwargs,
):
    """Start one helper and require its bounded private control acknowledgement."""
    if not (RUNTIME / "bridge.mjs").is_file():
        raise SrtError("SRT helper is missing", code = "runtime_missing", stage = "installation")
    if sys.platform == "win32":
        if os.environ.get("UNSLOTH_STUDIO_SRT_READ_LEASE", "1") == "1":
            from .srt_windows_read_lease import acquire
            read_transport = acquire(request, cancel_event, launch_deadline)
            return _spawn_windows(
                request,
                cancel_event = cancel_event,
                launch_deadline = launch_deadline,
                read_transport = read_transport,
                **kwargs,
            )
        return _spawn_windows(
            request, cancel_event = cancel_event, launch_deadline = launch_deadline, **kwargs
        )
    raise SrtError("Studio SRT is Windows-only", code = "operation_unsupported", stage = "launch")


def _spawn_windows(
    request,
    *,
    cancel_event = None,
    launch_deadline = None,
    read_transport = None,
    **kwargs,
):
    """Authenticate a per-launch helper connection without Windows pass_fds."""
    token = secrets.token_hex(32)
    proc = None
    connection = None
    writer = None
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.1)
        message = dict(request, controlSocket = {"port": listener.getsockname()[1], "token": token})
        if read_transport is not None:
            message["readLeaseTransport"] = read_transport
        encoded = json.dumps(message, ensure_ascii = True, separators = (",", ":")).encode() + b"\n"
        if len(encoded) > MAX_REQUEST:
            raise SrtError(
                "SRT launch request exceeds the protocol bound",
                code = "policy_oversized",
                stage = "policy",
            )
        # Upstream Windows session ACL setup can take tens of seconds on a cold
        # host. The payload timeout starts after spawn; cancellation stays live.
        deadline = launch_deadline if launch_deadline is not None else time.monotonic() + 120

        def check_wait():
            if cancel_event is not None and cancel_event.is_set():
                raise SrtError("SRT launch cancelled before acknowledgement")
            if time.monotonic() >= deadline:
                raise SrtError(
                    "SRT helper launch acknowledgement timed out",
                    code = "probe_timeout",
                    stage = "launch",
                )

        try:
            check_wait()
            options = dict(kwargs)
            options.pop("preexec_fn", None)
            options.pop("pass_fds", None)
            # The SRT native launcher owns its sandbox job. Suspending Node here
            # would prevent the acknowledgement needed before tools resumes it.
            options["creationflags"] = options.get("creationflags", 0) & ~0x00000004
            options.update(stdin = subprocess.PIPE, close_fds = True)
            options["env"] = {
                key: value
                for key, value in os.environ.items()
                if key.upper()
                in {
                    "SYSTEMROOT",
                    "WINDIR",
                    "PATH",
                    "TEMP",
                    "TMP",
                    "USERPROFILE",
                    "LOCALAPPDATA",
                    "APPDATA",
                    "PROGRAMDATA",
                    "PATHEXT",
                }
            }
            proc = subprocess.Popen(
                [node_executable(), str(RUNTIME / "bridge.mjs"), "--control-socket"], **options
            )
            if os.name == "nt":
                from .tools import _windows_job_capture

                # Node has no request yet, so the broker cannot precede its job.
                # SRT's runner needs permission to enter its own native job.
                proc._unsloth_job = _windows_job_capture(proc, allow_breakaway = True)
                if proc._unsloth_job is None:
                    raise SrtError("Cannot establish SRT helper process ownership")

            def send_request():
                stream = proc.stdin
                try:
                    offset = 0
                    while offset < len(encoded):
                        offset += os.write(stream.fileno(), encoded[offset:])
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        stream.close()
                    except OSError:
                        pass

            writer = threading.Thread(target = send_request, daemon = True)
            writer.start()
            pending = b""
            authenticated = False
            ready = False
            while True:
                check_wait()
                if connection is None:
                    try:
                        connection, _ = listener.accept()
                        connection.settimeout(0.1)
                    except socket.timeout:
                        if proc.poll() is not None:
                            raise SrtError("SRT helper exited before launch acknowledgement")
                        continue
                try:
                    chunk = connection.recv(MAX_CONTROL + 1)
                except socket.timeout:
                    continue
                if not chunk:
                    raise SrtError("SRT helper closed its control channel before launch")
                pending += chunk
                if len(pending) > MAX_CONTROL:
                    raise SrtError("SRT control response exceeds the protocol bound")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        raise SrtError("Malformed SRT control response") from exc
                    if not isinstance(event, dict) or event.get("v") != 1:
                        raise SrtError("Unknown SRT control protocol")
                    kind = event.get("event")
                    if not authenticated:
                        if kind != "hello" or not secrets.compare_digest(
                            str(event.get("token", "")), token
                        ):
                            raise SrtError("SRT control authentication failed")
                        authenticated = True
                    elif kind == "error":
                        raise SrtError(
                            "SRT setup failed",
                            code = event.get("code"),
                            stage = event.get("stage"),
                            dependency = event.get("dependency"),
                            details = event,
                        )
                    elif kind == "ready" and not ready and event.get("version") == "0.0.75":
                        ready = True
                    elif (
                        kind == "spawned"
                        and ready
                        and type(event.get("pid")) is int
                        and event["pid"] > 0
                    ):
                        writer.join(timeout = 1)
                        if writer.is_alive():
                            raise SrtError("SRT request writer did not finish")
                        proc.stdin = None
                        proc._srt_control_socket = connection
                        proc._srt_control_pending = pending
                        proc._srt_launcher_pid = event["pid"]
                        connection = None
                        return proc
                    else:
                        raise SrtError("Unexpected SRT launch control event")
        except Exception:
            if proc is not None:
                _stop_failed_helper(proc, connection)
            raise
        finally:
            if connection is not None:
                connection.close()
            if writer is not None:
                writer.join(timeout = 1)


def _stop_failed_helper(proc, connection = None):
    """Let the native launcher close its workload job/group before forcing exit."""
    if connection is not None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()
    else:
        proc.terminate()
    try:
        try:
            proc.wait(timeout = 3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout = 5)
    finally:
        job = getattr(proc, "_unsloth_job", None)
        if job is not None:
            job.terminate()


def release_control(proc) -> None:
    connection = getattr(proc, "_srt_control_socket", None)
    if connection is not None:
        proc._srt_control_socket = None
        connection.close()
    descriptor = getattr(proc, "_srt_control_fd", None)
    if descriptor is not None:
        proc._srt_control_fd = None
        os.close(descriptor)


def verify_success(proc) -> None:
    """Require the trusted completion receipt as well as successful process exit."""
    if proc.poll() != 0:
        raise SrtError("SRT workload success has no completion receipt")
    receipt = completion_receipt(proc)
    if (
        receipt.get("code") != 0
        or receipt.get("signal") is not None
        or receipt.get("reason") != "completed"
    ):
        raise SrtError("SRT did not attest successful completion")


def completion_receipt(proc) -> dict:
    """Read the bounded private receipt; a failure receipt is never success proof."""
    descriptor = getattr(proc, "_srt_control_fd", None)
    connection = getattr(proc, "_srt_control_socket", None)
    if (descriptor is None and connection is None) or proc.poll() is None:
        raise SrtError("SRT workload success has no completion receipt")
    data = getattr(proc, "_srt_control_pending", b"")
    if connection is not None:
        connection.setblocking(False)
    else:
        os.set_blocking(descriptor, False)
    while True:
        try:
            chunk = (
                connection.recv(MAX_CONTROL + 1)
                if connection is not None
                else os.read(descriptor, MAX_CONTROL + 1)
            )
        except BlockingIOError as exc:
            raise SrtError("SRT control channel remained open after helper exit") from exc
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_CONTROL:
            raise SrtError("SRT completion receipt exceeds the protocol bound")
    try:
        lines = data.splitlines()
        if len(lines) != 1:
            raise SrtError("SRT completion receipt is missing or duplicated")
        receipt = json.loads(lines[0])
        if not isinstance(receipt, dict) or receipt.get("v") != 1 or receipt.get("event") != "exit":
            raise SrtError("SRT completion receipt is invalid")
        return receipt
    except (ValueError, UnicodeError) as exc:
        raise SrtError("SRT completion receipt is malformed") from exc
