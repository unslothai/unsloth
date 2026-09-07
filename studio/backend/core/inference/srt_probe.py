# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native checks for the selected Python and SRT no-network launch profile."""

import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import srt_adapter

_lock = threading.Lock()
_cache = {}
_network_cache = None

_PROBE = r"""
import json, os, pathlib, socket, subprocess, sys
config = json.loads(sys.argv[1])
for name in [config['sentinel'], config['escape']]:
    try:
        pathlib.Path(name).read_bytes()
    except (OSError, PermissionError):
        pass
    else:
        raise RuntimeError('unrelated host file is readable')
pathlib.Path('private.txt').write_text('private write')
assert pathlib.Path('private.txt').read_text() == 'private write'
for family, address in [(socket.AF_INET, ('127.0.0.1', config['port']))]:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(.5)
    try:
        sock.connect(address)
    except OSError:
        pass
    else:
        raise RuntimeError('host listener is reachable')
    finally:
        sock.close()
try:
    socket.getaddrinfo('example.com', 443)
except OSError:
    pass
else:
    raise RuntimeError('system DNS resolved an external host')
for address in (config['host_socket'], config['abstract_socket']):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix:
        unix.settimeout(.5)
        try:
            unix.connect(address)
        except OSError:
            pass
        else:
            raise RuntimeError('host Unix socket escaped namespace/filesystem boundary')
left,right = socket.socketpair()
left.sendall(b'private')
assert right.recv(7) == b'private'
left.close(); right.close()
import multiprocessing.reduction, tempfile
with tempfile.TemporaryFile(dir=os.getcwd()) as shared:
    shared.write(b'shared'); shared.seek(0)
    duplicate = multiprocessing.reduction.DupFd(shared.fileno()).detach()
    assert os.read(duplicate,6) == b'shared'
    os.close(duplicate)
child = subprocess.check_output([sys.executable, '-I', '-S', '-c', 'print(42)'], text=True)
assert child.strip() == '42'
shell = subprocess.check_output([config['shell'], '--noprofile', '--norc', '-c', 'printf shell-ok'], text=True)
assert shell == 'shell-ok'
print('UNSLOTH_SRT_NATIVE_PROBE_OK')
"""


def probe(
    *,
    force = False,
    execution_kind = None,
    selected_executable = None,
):
    if execution_kind == "python" and selected_executable == sys.executable:
        execution_kind, selected_executable = None, None
    identity = (
        srt_adapter.installation_identity(),
        os.path.abspath(sys.executable),
        os.path.realpath(sys.executable),
        sys.prefix,
        execution_kind,
        selected_executable,
    )
    with _lock:
        cached = _cache.get(identity)
        if not force and cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        try:
            result = _native_probe(
                execution_kind = execution_kind, selected_executable = selected_executable
            )
        except Exception as exc:
            result = (False, f"SRT native probe failed: {str(exc)[:1500]}")
        if len(_cache) >= 8:
            _cache.clear()
        _cache[identity] = (time.monotonic(), result)
        return result


def _native_probe(*, execution_kind = None, selected_executable = None):
    if sys.platform in ("win32", "darwin"):
        return _supported_platform_probe(
            execution_kind = execution_kind, selected_executable = selected_executable
        )
    if sys.platform != "linux":
        return False, "SRT strict profile is unavailable on this platform"
    # Without working host controls a refused call inside the sandbox proves nothing.
    socket.getaddrinfo("example.com", 443)
    with (
        tempfile.TemporaryDirectory(prefix = "unsloth-srt-probe-") as directory,
        socket.socket() as listener,
        socket.socket(socket.AF_UNIX) as host_unix,
        socket.socket(socket.AF_UNIX) as host_abstract,
    ):
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        sentinel = root / "unrelated.txt"
        sentinel.write_text("benign-srt-confidentiality-control")
        escape = work / "escape"
        escape.symlink_to(sentinel)
        unix_path = str(root / "host.sock")
        abstract_path = "\0" + root.name
        for endpoint, address in ((host_unix, unix_path), (host_abstract, abstract_path)):
            endpoint.bind(address)
            endpoint.listen()
            with socket.socket(socket.AF_UNIX) as client:
                client.connect(address)
                accepted, _ = endpoint.accept()
                accepted.close()
        assert sentinel.read_text() == "benign-srt-confidentiality-control"
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout = 1):
            accepted, _ = listener.accept()
            accepted.close()
        from .tools import _build_safe_env, _sandbox_launcher_preexec

        env = _build_safe_env(str(work))
        shell = (selected_executable if execution_kind == "terminal" else None) or shutil.which(
            "bash", path = env["PATH"]
        )
        if not shell:
            return False, "Selected bash executable is unavailable"
        python = (selected_executable if execution_kind == "python" else None) or sys.executable
        args = {
            "sentinel": str(sentinel),
            "escape": str(escape),
            "port": port,
            "shell": shell,
            "host_socket": unix_path,
            "abstract_socket": abstract_path,
        }
        request = srt_adapter.request_for(
            [python, "-I", "-S", "-c", _PROBE, json.dumps(args)],
            str(work),
            env,
            30,
            operation = "probe",
        )
        proc = srt_adapter.spawn(
            request,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            cwd = str(work),
            preexec_fn = _sandbox_launcher_preexec,
        )
        try:
            output, _ = proc.communicate(timeout = 35)
            if proc.returncode == 0:
                srt_adapter.verify_success(proc)
        except Exception:
            proc.kill()
            proc.wait(timeout = 5)
            raise
        finally:
            srt_adapter.release_control(proc)
        if proc.returncode != 0 or output.strip() != b"UNSLOTH_SRT_NATIVE_PROBE_OK":
            return False, "SRT selected-runtime probe refused: " + output.decode(errors = "replace")[
                -1500:
            ]
        return (
            True,
            "Selected Python, shell children, private IPC/resource sharing, read confinement, DNS and host network checks passed.",
        )


def _supported_platform_probe(*, execution_kind = None, selected_executable = None):
    """Check the upstream platform contract without requiring Linux-only isolation."""
    from .tools import _build_safe_env
    with tempfile.TemporaryDirectory(prefix = "unsloth-srt-probe-") as directory:
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        sentinel = root / "write-denied.txt"
        sentinel.write_text("unchanged")
        env = _build_safe_env(str(work))
        shell = (selected_executable if execution_kind == "terminal" else None) or shutil.which(
            "bash", path = env.get("PATH")
        )
        if not shell:
            return False, "Selected bash executable is unavailable"
        python = (selected_executable if execution_kind == "python" else None) or sys.executable
        code = """
import pathlib, subprocess, sys
pathlib.Path('private.txt').write_text('workdir write')
assert pathlib.Path('private.txt').read_text() == 'workdir write'
try:
    pathlib.Path(sys.argv[1]).write_text('unexpected write')
except OSError:
    pass
else:
    raise RuntimeError('SRT denied path remained writable')
assert subprocess.check_output([sys.argv[2], '--noprofile', '--norc', '-c', 'printf shell-ok'], text=True) == 'shell-ok'
print('UNSLOTH_SRT_SUPPORTED_PROBE_OK')
"""
        request = srt_adapter.request_for(
            [python, "-I", "-S", "-c", code, str(sentinel), shell],
            str(work),
            env,
            30,
            operation = "probe",
        )
        request["denyWriteRoots"] = [str(sentinel)]
        proc = srt_adapter.spawn(
            request, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, cwd = str(work)
        )
        try:
            output, _ = proc.communicate(timeout = 35)
            if proc.returncode == 0:
                srt_adapter.verify_success(proc)
        except Exception:
            proc.kill()
            proc.wait(timeout = 5)
            raise
        finally:
            srt_adapter.release_control(proc)
        if (
            proc.returncode != 0
            or b"UNSLOTH_SRT_SUPPORTED_PROBE_OK" not in output
            or sentinel.read_text() != "unchanged"
        ):
            return False, "SRT platform probe refused: " + output.decode(errors = "replace")[-1500:]
        return (
            True,
            "Selected Python, Terminal child, workdir write and filesystem deny checks passed.",
        )


_NETWORK_PROBE = r"""
import socket, sys
port = int(sys.argv[1])
def request(host):
    with socket.create_connection(('127.0.0.1',3128),timeout=3) as s:
        s.sendall(('CONNECT '+host+':'+str(port)+' HTTP/1.1\r\nHost: '+host+'\r\n\r\n').encode())
        return s.recv(1024).split(b'\r\n',1)[0]
assert b'200' in request('srt-probe.invalid')
assert b'403' in request('not-allowed.invalid')
with socket.create_connection(('127.0.0.1',1080),timeout=3) as s:
    assert s.recv(2) == b'\x05\xff'
try:
    socket.create_connection(('127.0.0.1',port),timeout=.5)
except OSError:
    pass
else:
    raise RuntimeError('direct host network path bypassed proxy')
try:
    socket.getaddrinfo('example.com',443)
except OSError:
    pass
else:
    raise RuntimeError('system DNS escaped allowlist profile')
left,right=socket.socketpair()
left.sendall(b'private'); assert right.recv(7)==b'private'
left.close(); right.close()
print('UNSLOTH_SRT_NETWORK_PROBE_OK')
"""


def probe_network(*, force = False):
    """Measure namespace transport using only controlled local host listeners."""
    global _network_cache
    try:
        socat = srt_adapter.socat_executable()
    except srt_adapter.SrtError:
        return False
    identity = (srt_adapter.installation_identity(), socat, sys.executable, sys.prefix)
    with _lock:
        if (
            not force
            and _network_cache
            and _network_cache[0] == identity
            and time.monotonic() - _network_cache[1] < 60
        ):
            return _network_cache[2]
        result = False
        try:
            from .network_proxy import AllowlistProxy, NetworkAllowlist
            from .srt_network import SrtNetworkTransport
            from .tools import _build_safe_env, _sandbox_launcher_preexec

            with (
                tempfile.TemporaryDirectory(prefix = "unsloth-srt-netprobe-") as work,
                socket.socket() as origin,
            ):
                origin.bind(("127.0.0.1", 0))
                origin.listen()
                port = origin.getsockname()[1]
                with socket.create_connection(("127.0.0.1", port), timeout = 1):
                    accepted, _ = origin.accept()
                    accepted.close()
                proxy = AllowlistProxy(
                    NetworkAllowlist.from_entries(["srt-probe.invalid"]),
                    resolver = lambda host, port: ["127.0.0.1"],
                    allowed_ports = [port],
                    require_public = False,
                    connect_timeout = 2,
                )
                with SrtNetworkTransport(proxy, lifetime_seconds = 60) as transport:
                    env = _build_safe_env(work)
                    env.update(transport.environment)
                    request = srt_adapter.request_for(
                        [sys.executable, "-I", "-S", "-c", _NETWORK_PROBE, str(port)],
                        work,
                        env,
                        30,
                        operation = "probe",
                    )
                    request["network"] = {
                        "httpSocketPath": transport.http_socket_path,
                        "socksSocketPath": transport.socks_socket_path,
                        "socatPath": socat,
                    }
                    proc = srt_adapter.spawn(
                        request,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        cwd = work,
                        preexec_fn = _sandbox_launcher_preexec,
                    )
                    try:
                        output, _ = proc.communicate(timeout = 35)
                        if proc.returncode == 0:
                            srt_adapter.verify_success(proc)
                            result = output.strip() == b"UNSLOTH_SRT_NETWORK_PROBE_OK"
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.wait(timeout = 5)
                        srt_adapter.release_control(proc)
        except Exception:
            result = False
        _network_cache = (identity, time.monotonic(), result)
        return result
