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
from .srt_diagnostics import ProbeReason, boundary_identity

_lock = threading.Lock()
_cache = {}
_network_cache = None

_PROBE = r"""
import json, os, pathlib, socket, subprocess, sys
config = json.loads(sys.argv[1])
denied_files = [config['sentinel'], config['escape']]
if config.get('nested'):
    denied_files.append('/proc/' + str(config['host_pid']) + '/root' + config['sentinel'])
    caps = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
    for name in ('CapEff', 'CapPrm', 'CapBnd'):
        if int(caps[name].strip(), 16):
            raise RuntimeError('nested runtime retained capabilities')
for name in denied_files:
    try:
        pathlib.Path(name).read_bytes()
    except (OSError, PermissionError):
        pass
    else:
        raise RuntimeError('unrelated host file is readable')
pathlib.Path('private.txt').write_text('private write', encoding='utf-8')
assert pathlib.Path('private.txt').read_text(encoding='utf-8') == 'private write'
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
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
    udp.settimeout(.5)
    try:
        udp.sendto(b'network-control', ('127.0.0.1', config['udp_port']))
        udp.recvfrom(128)
    except OSError:
        pass
    else:
        raise RuntimeError('host UDP listener is reachable')
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
    isolation_variant = "standard",
):
    if isolation_variant not in ("standard", "nested"):
        return False, ProbeReason("policy_invalid", "policy")
    if execution_kind == "python" and selected_executable == sys.executable:
        execution_kind, selected_executable = None, None
    identity = (
        srt_adapter.installation_identity(),
        os.path.abspath(sys.executable),
        os.path.realpath(sys.executable),
        sys.prefix,
        execution_kind,
        selected_executable,
        boundary_identity(),
        isolation_variant,
    )
    with _lock:
        cached = _cache.get(identity)
        if not force and cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        try:
            result = _native_probe(
                execution_kind = execution_kind,
                selected_executable = selected_executable,
                **(
                    {"isolation_variant": isolation_variant}
                    if isolation_variant != "standard"
                    else {}
                ),
            )
        except subprocess.TimeoutExpired:
            result = (False, ProbeReason("probe_timeout", "probe"))
        except srt_adapter.SrtError as exc:
            result = (False, exc.diagnostic)
        except Exception:
            result = (False, ProbeReason("probe_failed", "probe"))
        if len(_cache) >= 8:
            _cache.clear()
        _cache[identity] = (time.monotonic(), result)
        return result


def _failed_linux_launch_reason(*, isolation_variant = "standard"):
    """Distinguish namespace setup from a failed protection assertion.

    Only a fixed benign executable is used, never the failed tool command or
    stderr pattern matching. This check does not qualify any alternate variant.
    """
    try:
        result = subprocess.run(
            [
                "/usr/bin/bwrap",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-net",
                "--cap-drop",
                "ALL",
                *(
                    ["--bind", "/proc", "/proc"]
                    if isolation_variant == "nested"
                    else ["--proc", "/proc"]
                ),
                "--",
                "/bin/true",
            ],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 3,
            close_fds = True,
            env = {"PATH": os.defpath},
        )
    except FileNotFoundError:
        return ProbeReason("dependency_missing", "dependency", "bubblewrap")
    except subprocess.TimeoutExpired:
        return ProbeReason("probe_timeout", "launch")
    except OSError:
        return ProbeReason("probe_failed", "launch")
    return (
        ProbeReason("operation_unsupported", "launch")
        if result.returncode
        else ProbeReason("enforcement_failed", "enforcement")
    )


def _native_probe(
    *,
    execution_kind = None,
    selected_executable = None,
    isolation_variant = "standard",
):
    if isolation_variant == "nested" and sys.platform != "linux":
        return False, ProbeReason("operation_unsupported", "probe")
    if sys.platform in ("win32", "darwin"):
        return _supported_platform_probe(
            execution_kind = execution_kind, selected_executable = selected_executable
        )
    if sys.platform != "linux":
        return False, ProbeReason("operation_unsupported", "probe")
    # Positive controls use owned local endpoints, independent of public DNS.
    with (
        tempfile.TemporaryDirectory(prefix = "unsloth-srt-probe-") as directory,
        socket.socket() as listener,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as host_udp,
        socket.socket(socket.AF_UNIX) as host_unix,
        socket.socket(socket.AF_UNIX) as host_abstract,
    ):
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        sentinel = root / "unrelated.txt"
        sentinel.write_text("benign-srt-confidentiality-control", encoding = "utf-8")
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
        assert sentinel.read_text(encoding = "utf-8") == "benign-srt-confidentiality-control"
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout = 1):
            accepted, _ = listener.accept()
            accepted.close()
        host_udp.bind(("127.0.0.1", 0))
        host_udp.settimeout(0.5)
        udp_address = host_udp.getsockname()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(0.5)
            client.sendto(b"network-control", udp_address)
            packet, address = host_udp.recvfrom(128)
            host_udp.sendto(packet, address)
            assert client.recvfrom(128)[0] == b"network-control"

        udp_stop = threading.Event()

        def echo_udp():
            while not udp_stop.is_set():
                try:
                    packet, address = host_udp.recvfrom(128)
                    host_udp.sendto(packet, address)
                except socket.timeout:
                    continue

        # Keep the positive endpoint live throughout the isolated negative check.
        udp_worker = threading.Thread(target = echo_udp, daemon = True)
        from .tools import _build_safe_env, _sandbox_launcher_preexec

        env = _build_safe_env(str(work))
        shell = (selected_executable if execution_kind == "terminal" else None) or shutil.which(
            "bash", path = env["PATH"]
        )
        if not shell:
            return False, ProbeReason("dependency_missing", "dependency", "selected_shell")
        python = (selected_executable if execution_kind == "python" else None) or sys.executable
        args = {
            "sentinel": str(sentinel),
            "escape": str(escape),
            "port": port,
            "udp_port": udp_address[1],
            "shell": shell,
            "host_socket": unix_path,
            "abstract_socket": abstract_path,
            "nested": isolation_variant == "nested",
            "host_pid": os.getpid(),
        }
        request = srt_adapter.request_for(
            [python, "-I", "-S", "-c", _PROBE, json.dumps(args)],
            str(work),
            env,
            30,
            operation = "probe",
            **({"isolation_variant": isolation_variant} if isolation_variant != "standard" else {}),
        )
        udp_worker.start()
        try:
            proc = srt_adapter.spawn(
                request,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                cwd = str(work),
                preexec_fn = _sandbox_launcher_preexec,
            )
        except BaseException:
            udp_stop.set()
            udp_worker.join(timeout = 2)
            raise
        try:
            output, _ = proc.communicate(timeout = 35)
            if proc.returncode == 0:
                srt_adapter.verify_success(proc)
            elif srt_adapter.completion_receipt(proc).get("reason") == "timeout":
                raise srt_adapter.SrtError(
                    "SRT probe timed out", code = "probe_timeout", stage = "enforcement"
                )
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait(timeout = 5)
            output = exc.output or b""
            raise srt_adapter.SrtError(
                "SRT probe timed out after launch", code = "probe_timeout", stage = "enforcement"
            ) from exc
        except Exception:
            proc.kill()
            proc.wait(timeout = 5)
            raise
        finally:
            udp_stop.set()
            udp_worker.join(timeout = 2)
            srt_adapter.release_control(proc)
        if proc.returncode != 0 or output.strip() != b"UNSLOTH_SRT_NATIVE_PROBE_OK":
            return False, _failed_linux_launch_reason(
                **(
                    {"isolation_variant": isolation_variant}
                    if isolation_variant != "standard"
                    else {}
                )
            )
        return (
            True,
            "Selected Python, shell children, private IPC/resource sharing, read confinement and controlled host TCP/UDP network checks passed."
            if isolation_variant == "standard"
            else "Selected runtime and local restriction diagnostics passed; outer /proc remains exposed. Platform qualification is separate.",
        )


def _supported_platform_probe(*, execution_kind = None, selected_executable = None):
    """Check the upstream platform contract without requiring Linux-only isolation."""
    from .tools import _build_safe_env
    with tempfile.TemporaryDirectory(prefix = "unsloth-srt-probe-") as directory:
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        sentinel = root / "write-denied.txt"
        sentinel.write_text("unchanged", encoding = "utf-8")
        env = _build_safe_env(str(work))
        shell = (selected_executable if execution_kind == "terminal" else None) or shutil.which(
            "bash", path = env.get("PATH")
        )
        if not shell:
            return False, ProbeReason("dependency_missing", "dependency", "selected_shell")
        python = (selected_executable if execution_kind == "python" else None) or sys.executable
        code = """
import pathlib, subprocess, sys
pathlib.Path('private.txt').write_text('workdir write', encoding='utf-8')
assert pathlib.Path('private.txt').read_text(encoding='utf-8') == 'workdir write'
try:
    pathlib.Path(sys.argv[1]).write_text('unexpected write', encoding='utf-8')
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
            elif srt_adapter.completion_receipt(proc).get("reason") == "timeout":
                raise srt_adapter.SrtError(
                    "SRT probe timed out", code = "probe_timeout", stage = "enforcement"
                )
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait(timeout = 5)
            output = exc.output or b""
            raise srt_adapter.SrtError(
                "SRT probe timed out after launch", code = "probe_timeout", stage = "enforcement"
            ) from exc
        except Exception:
            proc.kill()
            proc.wait(timeout = 5)
            raise
        finally:
            srt_adapter.release_control(proc)
        if (
            proc.returncode != 0
            or b"UNSLOTH_SRT_SUPPORTED_PROBE_OK" not in output
            or sentinel.read_text(encoding = "utf-8") != "unchanged"
        ):
            return False, ProbeReason("enforcement_failed", "enforcement")
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
    identity = (
        srt_adapter.installation_identity(),
        socat,
        sys.executable,
        sys.prefix,
        boundary_identity(),
    )
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
