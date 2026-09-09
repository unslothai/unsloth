# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Bounded control channel for the shipped, pinned SRT Node helper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import secrets
import socket
import site
import shutil
import subprocess
import stat
import sys
import sysconfig
import time
import threading
from .srt_seccomp import install as _install_srt_seccomp

RUNTIME = Path(__file__).with_name("srt_runtime")
MAX_CONTROL = 16_384
MAX_REQUEST = 262_144


def validate_roots(roots: list[str], workdir: str) -> list[str]:
    """Reject host IPC and return nested mounts for SRT to mask after runtime restores."""
    canonical = sorted(set(os.path.realpath(root) for root in [*roots, workdir]), key = len)
    scan_roots = []
    for root in canonical:
        if not any(root == parent or root.startswith(parent + os.sep) for parent in scan_roots):
            scan_roots.append(root)
    try:
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding = "utf-8").splitlines()
    except OSError as exc:
        raise SrtError("Cannot inspect nested runtime mounts") from exc
    denied_mounts = []
    for line in mount_lines:
        fields = line.split()
        if len(fields) < 6:
            raise SrtError("Malformed runtime mount information")
        mount = fields[4]
        for escaped, character in (
            (r"\040", " "),
            (r"\011", "\t"),
            (r"\012", "\n"),
            (r"\134", "\\"),
        ):
            mount = mount.replace(escaped, character)
        if any(mount.startswith(root + os.sep) for root in scan_roots):
            if mount.startswith(workdir + os.sep):
                raise SrtError("Workdir contains a nested host mount")
            denied_mounts.append(mount)
    count = 0
    deadline = time.monotonic() + 30

    def check_bound(path):
        nonlocal count
        count += 1
        if count > 500_000:
            raise SrtError(f"Runtime filesystem scan exceeded 500000 entries at: {path}")
        if time.monotonic() > deadline:
            raise SrtError(
                f"Runtime filesystem scan exceeded 30 seconds after {count} entries at: {path}"
            )

    def masked(path):
        return any(path == mount or path.startswith(mount + os.sep) for mount in denied_mounts)

    def in_workdir(path):
        return path == workdir or path.startswith(workdir + os.sep)

    pending = list(scan_roots)
    while pending:
        path = pending.pop()
        if masked(path):
            continue
        check_bound(path)
        try:
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    for entry in entries:
                        if masked(entry.path):
                            continue
                        check_bound(entry.path)
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks = False):
                            pending.append(entry.path)
                        elif not entry.is_file(follow_symlinks = False):
                            raise SrtError(
                                f"Runtime or workdir contains a host IPC/device entry: {entry.path}"
                            )
                        elif in_workdir(entry.path) and os.lstat(entry.path).st_nlink > 1:
                            raise SrtError(
                                "Workdir contains a hardlinked file with unrelated host authority"
                            )
            elif not stat.S_ISREG(info.st_mode):
                raise SrtError(f"Runtime or workdir contains a host IPC/device entry: {path}")
            elif info.st_nlink > 1 and in_workdir(path):
                raise SrtError("Workdir contains a hardlinked file with unrelated host authority")
        except OSError as exc:
            raise SrtError(f"Cannot validate runtime entry: {path}") from exc
    return sorted(set(denied_mounts))


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
        "srt_seccomp.py",
        "srt_network.py",
        "srt_diagnostics.py",
        "srt_nested.py",
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


def socat_executable() -> str:
    selected = shutil.which("socat")
    if not selected:
        raise SrtError(
            "HTTPS allowlists require socat installed before Studio starts",
            code = "dependency_missing",
            stage = "dependency",
            dependency = "socat",
        )
    return os.path.abspath(selected)


def read_roots(executable: str) -> list[str]:
    if sys.platform in ("win32", "darwin"):
        roots = [
            os.path.dirname(os.path.abspath(executable)),
            sys.prefix,
            sys.base_prefix,
            str(Path(__file__).with_name("sandbox_site")),
            *site.getsitepackages(),
        ]
        return sorted({os.path.realpath(root) for root in roots if os.path.exists(root)})
    roots = {
        "/usr/bin",
        "/usr/sbin",
        "/usr/lib64",
        "/usr/libexec",
        "/usr/share/zoneinfo",
        "/usr/share/fonts",
        "/usr/share/fontconfig",
        "/usr/share/locale",
        "/usr/local/bin",
        "/bin",
        "/sbin",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/alternatives",
        "/etc/localtime",
        "/etc/nsswitch.conf",
        str(Path(__file__).with_name("sandbox_site")),
        os.path.dirname(os.path.realpath(executable)),
        *(value for key, value in sysconfig.get_paths().items() if key != "data"),
        *site.getsitepackages(),
    }
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        roots.update((f"/lib/{multiarch}", f"/usr/lib/{multiarch}"))
    # Restore the loader/library closure without admitting unrelated SDK and
    # interpreter trees installed beside the selected runtime on build hosts.
    for directory in ("/lib", "/usr/lib", "/usr/local/lib"):
        if not os.path.isdir(directory):
            continue
        with os.scandir(directory) as entries:
            roots.update(
                entry.path
                for entry in entries
                if (entry.name.endswith(".so") or ".so." in entry.name) and entry.is_file()
            )
    roots.update(
        prefix for prefix in (sys.prefix, sys.base_prefix) if prefix not in ("/usr", "/usr/local")
    )
    resolved = sorted(
        {
            spelling
            for root in roots
            if os.path.exists(root)
            for spelling in (os.path.abspath(root), os.path.realpath(root))
        }
    )
    if "/" in resolved:
        raise SrtError("The selected runtime would require granting the filesystem root")
    return resolved


def request_for(
    argv,
    cwd,
    env,
    timeout,
    *,
    operation = "run",
    additional_read_roots = (),
    isolation_variant = "standard",
) -> dict:
    try:
        if isolation_variant not in ("standard", "nested") or (
            isolation_variant == "nested" and sys.platform != "linux"
        ):
            raise SrtError(
                "Unsupported SRT isolation variant", code = "policy_invalid", stage = "policy"
            )
        request = _request_for(
            argv,
            cwd,
            env,
            timeout,
            operation = operation,
            additional_read_roots = additional_read_roots,
        )
        if isolation_variant != "standard":
            request["isolationVariant"] = isolation_variant
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
    denied = validate_roots(roots, os.path.realpath(cwd)) if sys.platform == "linux" else []
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
    if sys.platform not in ("linux", "darwin"):
        raise SrtError(
            "SRT launch is unavailable on this platform",
            code = "operation_unsupported",
            stage = "launch",
        )
    node = node_executable()
    read_fd, write_fd = os.pipe()
    proc = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise SrtError("SRT launch cancelled before execution")
        message = dict(request, controlFd = write_fd)
        encoded = json.dumps(message, ensure_ascii = True, separators = (",", ":")).encode() + b"\n"
        if len(encoded) > MAX_REQUEST:
            raise SrtError(
                "SRT launch request exceeds the protocol bound",
                code = "policy_oversized",
                stage = "policy",
            )
        options = dict(kwargs)
        parent_preexec = options.get("preexec_fn")
        if parent_preexec is None:
            from .tools import _sandbox_launcher_preexec
            parent_preexec = _sandbox_launcher_preexec

        def guarded_preexec():
            parent_preexec()
            if sys.platform == "linux":
                _install_srt_seccomp()

        options["preexec_fn"] = guarded_preexec
        options.update(stdin = subprocess.PIPE, close_fds = True, pass_fds = (write_fd,))
        # Loader settings from a user's selected Python environment must not affect Node.
        options["env"] = {"PATH": os.defpath, "HOME": request["cwd"], "LANG": "C.UTF-8"}
        proc = subprocess.Popen([node, str(RUNTIME / "bridge.mjs"), str(write_fd)], **options)
        os.close(write_fd)
        write_fd = -1
        timeout_ms = request.get("timeoutMs")
        deadline = time.monotonic() + (30 if timeout_ms is None else min(30, timeout_ms / 1000))
        if launch_deadline is not None:
            deadline = min(deadline, launch_deadline)
        input_fd = proc.stdin.fileno()
        os.set_blocking(input_fd, False)
        offset = 0
        while offset < len(encoded):
            if cancel_event is not None and cancel_event.is_set():
                raise SrtError("SRT launch cancelled before acknowledgement")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SrtError("SRT helper request timed out", code = "probe_timeout", stage = "launch")
            _, writable, _ = select.select([], [input_fd], [], min(0.1, remaining))
            if writable:
                try:
                    offset += os.write(input_fd, encoded[offset:])
                except BlockingIOError:
                    pass
        proc.stdin.close()
        proc.stdin = None
        pending = b""
        ready = False
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise SrtError("SRT launch cancelled before acknowledgement")
            readable, _, _ = select.select(
                [read_fd], [], [], min(0.1, max(0, deadline - time.monotonic()))
            )
            if not readable:
                continue
            chunk = os.read(read_fd, MAX_CONTROL + 1)
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
                if event.get("event") == "error":
                    raise SrtError(
                        "SRT setup failed",
                        code = event.get("code"),
                        stage = event.get("stage"),
                        dependency = event.get("dependency"),
                        details = event,
                    )
                if event.get("event") == "ready":
                    if ready or event.get("version") != "0.0.75":
                        raise SrtError("SRT dependency identity mismatch")
                    if event.get("isolationVariant", "standard") != request.get(
                        "isolationVariant", "standard"
                    ):
                        raise SrtError("SRT isolation variant acknowledgement mismatch")
                    ready = True
                elif event.get("event") == "spawned":
                    if not ready or type(event.get("pid")) is not int or event["pid"] <= 0:
                        raise SrtError("SRT sent an invalid launch acknowledgement")
                    # Keep the control pipe open until helper exit; it carries exit/error receipts.
                    proc._srt_control_fd = read_fd
                    proc._srt_control_pending = pending
                    if sys.platform == "darwin":
                        proc._srt_launcher_pid = event["pid"]
                    read_fd = -1
                    return proc
                elif event.get("event") == "exit":
                    raise SrtError("SRT exited before launching the workload")
                else:
                    raise SrtError("Unexpected SRT launch control event")
        raise SrtError(
            "SRT helper launch acknowledgement timed out", code = "probe_timeout", stage = "launch"
        )
    except Exception:
        if proc is not None:
            _stop_failed_helper(proc)
        raise
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


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
