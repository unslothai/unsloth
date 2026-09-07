# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed, private interpreter probe assembly, not a capability or cache result.

Only broker code can request this launch. Full qualification also requires the
host-controlled network, confidentiality and lifecycle checks; these core
observations alone must never enable the production backend.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as W
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import time

from .content_files import native_files
from .profiles import WindowsRuntimeError

PROBE_FILENAME = "qualification-probe.py"
HOST_CONTROL_FILENAME = "qualification-host-control"
MAX_PROBE_OUTPUT = 8192
CORE_CHECKS = (
    "private_workdir",
    "stdlib_native_asyncio",
    "threads_private_pipe",
    "aap_sentinel_denied",
    "runtime_write_denied",
    "host_read_denied",
    "host_write_denied",
    "process_policy_diagnostic",
    "host_registry_read_denied",
    "host_registry_write_denied",
)
NETWORK_CHECKS = ("ipv4_tcp_denied", "ipv4_udp_denied", "ipv6_tcp_denied", "ipv6_udp_denied")
HOST_CHECKS = NETWORK_CHECKS + ("host_named_pipe_denied",)
# This checklist is deliberately wider than compatibility and the registry
# handle audit. Missing measurements must keep production execution unavailable.
REQUIRED_QUALIFICATION_CHECKS = (
    CORE_CHECKS
    + HOST_CHECKS
    + (
        "dns_directed_denied",
        "dns_default_ex_denied",
        "dns_default_w_denied",
        "native_expanded_startup_gate",
        "retained_host_ipc_denied",
        "inherited_host_handles_denied",
        "running_cancellation_cleanup",
        "broker_death_cleanup",
    )
)
HOST_PIPE_PREFIX = "\\\\.\\pipe\\unsloth-python-probe-"

_DNS_CONTEXT_SOURCE = r"""
# Resolver prerequisites only: neither a valid ID nor an API failure proves
# DNS denial. Query after the native drop, without changing compartments.
dns_context = [None,None]
iphelper = ctypes.WinDLL('iphlpapi',use_last_error=True,winmode=0x800)
for index,name in enumerate(('GetDefaultCompartmentId','GetCurrentThreadCompartmentId')):
    try:
        query = getattr(iphelper,name)
    except AttributeError:
        continue
    query.argtypes = []
    query.restype = W.DWORD
    dns_context[index] = query()
"""

_HOST_PIPE_SOURCE = r"""
for access in (0x80000000,0x40000000,0xc0000000):
    handle = kernel.CreateFileW(HOST_PIPE_PATH,access,0,None,3,0,None)
    if handle != ctypes.c_void_p(-1).value:
        assert kernel.CloseHandle(handle)
        raise AssertionError('Host named pipe was accessible')
    assert ctypes.get_last_error() == 5,('Host pipe denial failed',access,ctypes.get_last_error())
checks.append('host_named_pipe_denied')
"""

_REGISTRY_SOURCE = r"""
import ctypes
from ctypes import wintypes as W
registry = ctypes.WinDLL('advapi32',use_last_error=True,winmode=0x800)
registry.RegOpenKeyExW.argtypes = [W.HKEY,W.LPCWSTR,W.DWORD,W.DWORD,ctypes.POINTER(W.HKEY)]
registry.RegOpenKeyExW.restype = W.LONG
registry.RegCloseKey.argtypes = [W.HKEY]
registry.RegCloseKey.restype = W.LONG
users = W.HKEY(ctypes.c_int32(0x80000003).value)
for access,name in ((1,'host_registry_read_denied'),(2,'host_registry_write_denied')):
    key = W.HKEY()
    result = registry.RegOpenKeyExW(users,HOST_REGISTRY_PATH,0,access|0x100,ctypes.byref(key))
    if result == 0:
        assert registry.RegCloseKey(key) == 0
    assert result == 5,('Host registry access was not denied',name,result)
    checks.append(name)
"""

_CORE_SOURCE = (
    r"""
import asyncio, bz2, ctypes, hashlib, json, lzma, multiprocessing, os
import sqlite3, ssl, sys, tempfile, threading, zlib
from pathlib import Path

checks = []
root = Path(__file__).parent
assert Path.cwd() == root and Path(tempfile.gettempdir()) == root
marker = root / 'probe-workdir-write'
assert not marker.exists()
marker.write_bytes(b'private probe workdir')
assert marker.read_bytes() == b'private probe workdir'
marker.unlink()
checks.append('private_workdir')

assert bz2.decompress(bz2.compress(b'core')) == b'core'
assert lzma.decompress(lzma.compress(b'core')) == b'core'
assert zlib.decompress(zlib.compress(b'core')) == b'core'
assert hashlib.sha256(b'core').hexdigest()
with sqlite3.connect(':memory:') as database:
    assert database.execute('select 17').fetchone() == (17,)
assert ssl.OPENSSL_VERSION and ctypes.sizeof(ctypes.c_void_p) == 8
assert asyncio.run(asyncio.sleep(0, result=23)) == 23
checks.append('stdlib_native_asyncio')

reader, writer = os.pipe()
failures = []
def send():
    try:
        assert os.write(writer,b'private thread IPC') == 18
    except BaseException as error:
        failures.append(str(error))
thread = threading.Thread(target=send)
try:
    thread.start()
    assert os.read(reader,18) == b'private thread IPC'
    thread.join(5)
    assert not thread.is_alive() and not failures
finally:
    os.close(reader)
    os.close(writer)
checks.append('threads_private_pipe')

from ctypes import wintypes as W
kernel = ctypes.WinDLL('kernel32',use_last_error=True,winmode=0x800)
kernel.CreateFileW.argtypes = [W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
kernel.CreateFileW.restype = W.HANDLE
kernel.CloseHandle.argtypes = [W.HANDLE]
kernel.CloseHandle.restype = W.BOOL
for path,access,name,errors in (
    (root/'startup-aap-control',0x80000000,'aap_sentinel_denied',(5,)),
    (ssl.__file__,0x40000000,'runtime_write_denied',(5,32)),
    (root.parent/HOST_CONTROL_FILENAME,0x80000000,'host_read_denied',(5,)),
    (root.parent/HOST_CONTROL_FILENAME,0x40000000,'host_write_denied',(5,)),
):
    handle = kernel.CreateFileW(str(path),access,7,None,3,0,None)
    if handle != ctypes.c_void_p(-1).value:
        assert kernel.CloseHandle(handle)
        raise AssertionError('Native filesystem denial failed: '+name)
    assert ctypes.get_last_error() in errors,(name,ctypes.get_last_error())
    checks.append(name)

try:
    multiprocessing.Process(target=lambda: None).start()
except RuntimeError as error:
    assert getattr(error,'code',None) == 'WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED'
else:
    raise AssertionError('Python worker policy missing')
checks.append('process_policy_diagnostic')
"""
    + _REGISTRY_SOURCE
    + r"""
if NETWORK_PORTS:
    import socket
    controls = ((socket.AF_INET,socket.SOCK_STREAM,'127.0.0.1','ipv4_tcp_denied'),
        (socket.AF_INET,socket.SOCK_DGRAM,'127.0.0.1','ipv4_udp_denied'),
        (socket.AF_INET6,socket.SOCK_STREAM,'::1','ipv6_tcp_denied'),
        (socket.AF_INET6,socket.SOCK_DGRAM,'::1','ipv6_udp_denied'))
    for (family,kind,address,name),port in zip(controls,NETWORK_PORTS):
        connection = None
        try:
            connection = socket.socket(family,kind)
            connection.settimeout(1)
            if kind == socket.SOCK_STREAM:
                connection.connect((address,port))
            else:
                connection.sendto(b'UNSLOTH_PRIVATE_NETWORK_PROBE',(address,port))
        except OSError as error:
            assert error.winerror == 10013,(name,repr(error))
        else:
            raise AssertionError('Network operation was not denied: '+name)
        finally:
            if connection is not None:
                connection.close()
        checks.append(name)
"""
    + "if NETWORK_PORTS:\n"
    + "\n".join("    " + line for line in _HOST_PIPE_SOURCE.splitlines())
    + "\n"
    + _DNS_CONTEXT_SOURCE
    + "\n_result={'probe_nonce':PROBE_NONCE,'version':list(sys.version_info[:3]),'checks':checks,'dns_context':dns_context}\n"
    + "if NETWORK_PORTS: _result['dns_queries']=perform_dns_probe()\n"
    + "print(json.dumps(_result,sort_keys=True))\n"
)


def validate_network_ports(value):
    if (
        type(value) is not tuple
        or len(value) not in (0, 4)
        or any(type(port) is not int or not 0 < port < 65536 for port in value)
    ):
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Invalid fixed network ports.")
    return value


def probe_source(nonce, network_ports = ()):
    if type(nonce) is not bytes or len(nonce) != 32:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Invalid fixed probe binding.")
    ports = validate_network_ports(network_ports)
    from .dns_probe import payload_source

    return (
        "PROBE_NONCE = "
        + json.dumps(nonce.hex())
        + "\n"
        + "NETWORK_PORTS = "
        + json.dumps(ports)
        + "\n"
        + "HOST_CONTROL_FILENAME = "
        + json.dumps(HOST_CONTROL_FILENAME)
        + "\n"
        + "HOST_REGISTRY_PATH = "
        + json.dumps(native_files().owner + "\\Software")
        + "\n"
        + "HOST_PIPE_PATH = "
        + json.dumps(HOST_PIPE_PREFIX + nonce.hex())
        + "\n"
        + (payload_source(nonce) if ports else "")
        + _CORE_SOURCE
    )


def _verify_host_registry():
    """Nonmutating positive control, only in the bounded preparation worker."""
    path = native_files().owner + "\\Software"
    registry = ctypes.WinDLL("advapi32", use_last_error = True, winmode = 0x800)
    registry.RegOpenKeyExW.argtypes = [W.HKEY, W.LPCWSTR, W.DWORD, W.DWORD, ctypes.POINTER(W.HKEY)]
    registry.RegOpenKeyExW.restype = W.LONG
    registry.RegCloseKey.argtypes = [W.HKEY]
    registry.RegCloseKey.restype = W.LONG
    users = W.HKEY(ctypes.c_int32(0x80000003).value)
    for access in (1, 2):
        key = W.HKEY()
        result = registry.RegOpenKeyExW(users, path, 0, access | 0x100, ctypes.byref(key))
        if result != 0:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_PROBE_FAILED", f"Host registry positive control failed: {result}."
            )
        result = registry.RegCloseKey(key)
        if result != 0:
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_CLEANUP_FAILED", f"Host registry control close failed: {result}."
            )


def prepare_probe_files(owner):
    """Fixed preparation worker only; no external workdir or supplied source."""
    from .launch import _overlap, _validate_paths

    _verify_host_registry()
    owner.workdir = Path(owner.identity.private_temp)
    owner.script = owner.workdir / PROBE_FILENAME
    api = native_files()
    api.create(owner.script, probe_source(owner.nonce, owner.probe_ports).encode("utf-8"))
    host_control = Path(owner.identity.profile_folder) / HOST_CONTROL_FILENAME
    api.create(host_control, owner.nonce)
    handle = owner.file_pins.file(host_control)
    if api.read(handle, 32) != owner.nonce:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Host file control failed.")
    from ..os_sandbox import ToolLaunchPlan

    owner.spec = ToolLaunchPlan(
        argv = (owner.probe_executable, "-u", str(owner.script)),
        workdir = str(owner.workdir),
        env = {},
        execution_kind = "python",
    )
    core = owner.published.core
    owner.workdir, owner.script = _validate_paths(
        owner.spec, owner.published.store_root, (core.runtime.prefix, core.runtime.base_prefix)
    )
    if any(_overlap(owner.workdir, item.source.path) for item in owner.published.spec().files):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROBE_INVALID", "Probe workdir overlaps runtime sources."
        )


def _host_pipe_api():
    from .native_io import pipe_api

    api = pipe_api()
    api.CreateNamedPipeW.argtypes = [W.LPCWSTR, *([W.DWORD] * 6), ctypes.c_void_p]
    api.CreateNamedPipeW.restype = W.HANDLE
    api.CreateFileW.argtypes = [
        W.LPCWSTR,
        W.DWORD,
        W.DWORD,
        ctypes.c_void_p,
        W.DWORD,
        W.DWORD,
        W.HANDLE,
    ]
    api.CreateFileW.restype = W.HANDLE
    api.ConnectNamedPipe.argtypes = [W.HANDLE, ctypes.c_void_p]
    api.ConnectNamedPipe.restype = W.BOOL
    return api


def _pipe_control_check(success, operation):
    if not success:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROBE_FAILED", f"Host named-pipe {operation} control failed."
        )


def _prepare_host_pipe(owner):
    api = _host_pipe_api()
    server = api.CreateNamedPipeW(
        HOST_PIPE_PREFIX + owner.nonce.hex(), 3 | 0x80000, 8, 1, 4096, 4096, 0, None
    )
    _pipe_control_check(server not in (None, ctypes.c_void_p(-1).value), "creation")
    # Same checked-close owner as the payload's control handles. Never inherited.
    owner.handles.add(server)
    return server


def _verify_host_pipe(owner, server):
    """Positive control on the very instance kept alive during the denial probe."""
    api = _host_pipe_api()
    client = api.CreateFileW(HOST_PIPE_PREFIX + owner.nonce.hex(), 0xC0000000, 0, None, 3, 0, None)
    _pipe_control_check(client not in (None, ctypes.c_void_p(-1).value), "open")
    owner.handles.add(client)
    # Client already connected; no waiting for an untrusted peer or reconnection.
    connected = api.ConnectNamedPipe(server, None)
    _pipe_control_check(connected or ctypes.get_last_error() == 535, "connection")
    for writer, reader, content in (
        (client, server, owner.nonce),
        (server, client, owner.nonce[::-1]),
    ):
        written, available, read = W.DWORD(), W.DWORD(), W.DWORD()
        data = ctypes.create_string_buffer(content)
        _pipe_control_check(
            api.WriteFile(writer, data, len(content), ctypes.byref(written), None)
            and written.value == len(content),
            "write",
        )
        _pipe_control_check(
            api.PeekNamedPipe(reader, None, 0, None, ctypes.byref(available), None)
            and available.value == len(content),
            "available bytes",
        )
        received = ctypes.create_string_buffer(len(content))
        _pipe_control_check(
            api.ReadFile(reader, received, len(content), ctypes.byref(read), None)
            and read.value == len(content)
            and received.raw == content,
            "read",
        )


def prepare_python_probe(
    selected_executable,
    store_root,
    *,
    timeout = 90,
    cancel = None,
    network_ports = (),
):
    """Prepare fixed core checks through the real gate, without claiming qualification."""
    from .launch import _PythonLaunch, _prepare_owned_launch, _remaining, _retry_pending_cleanup

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Invalid fixed probe deadline.")
    deadline = time.monotonic() + timeout
    _remaining(deadline, cancel)
    _retry_pending_cleanup(deadline, cancel)
    owner = _PythonLaunch(None, None, None, None, deadline, cancel)
    owner.probe_executable = selected_executable
    owner.probe_ports = validate_network_ports(network_ports)
    return _prepare_owned_launch(owner, selected_executable, store_root)


@dataclass(frozen = True)
class DnsContextObservations:
    """Resolver prerequisites, never DNS-denial or qualification evidence."""

    default_compartment: int | None
    thread_compartment: int | None

    def __post_init__(self):
        if any(
            part is not None and (type(part) is not int or not 0 <= part <= 0xFFFFFFFF)
            for part in (self.default_compartment, self.thread_compartment)
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_PROBE_INVALID", "Invalid DNS context observations."
            )

    def require_usable(self):
        """Reject unavailable setup; success still requires live DNS enforcement."""
        if (
            self.default_compartment is None
            or self.thread_compartment is None
            or self.default_compartment == 0
            or self.thread_compartment == 0
            or self.default_compartment != self.thread_compartment
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_DNS_CONTEXT_UNAVAILABLE",
                "Windows could not establish matching resolver compartments after the LPAC drop "
                f"(default={self.default_compartment}, thread={self.thread_compartment}). "
                "DNS isolation remains unverified; no capabilities were added.",
            )


def _parse_dns_context(value):
    if type(value) is not list or len(value) != 2:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROBE_INVALID", "Invalid DNS context observations."
        )
    return DnsContextObservations(*value)


@dataclass(frozen = True)
class ProbeObservations:
    """Fixed-probe measurements, not a runtime qualification or tool record."""

    content_digest: str
    runtime_digest: str
    artifact_digest: str
    profile_digest: str
    version: tuple[int, int, int]
    checks: tuple[str, ...]
    elapsed_seconds: float
    dns_context: DnsContextObservations
    qualification_complete: bool = False
    catalog_binding_digest: str = ""


def _parse_probe_output(data, nonce, version, *, network):
    from .preparation import _json

    if type(data) is not bytes or not 0 < len(data) <= MAX_PROBE_OUTPUT:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Invalid probe output size.")
    expected = CORE_CHECKS + (HOST_CHECKS if network else ())
    value = _json(data)
    if (
        type(value) is not dict
        or set(value)
        not in (
            {"probe_nonce", "version", "checks", "dns_context"},
            {"probe_nonce", "version", "checks", "dns_context", "dns_queries"}
            if network
            else set(),
        )
        or value["probe_nonce"] != nonce.hex()
        or type(value["version"]) is not list
        or any(type(part) is not int for part in value["version"])
        or value["version"] != list(version)
        or value["checks"] != list(expected)
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROBE_INVALID", "Fixed probe observations differ."
        )
    return expected, _parse_dns_context(value["dns_context"])


def _collect_probe_output(owner, process):
    """Read only a fixed probe's untouched native stdout under its launch deadline."""
    from .launch import _remaining
    from .protocol import _pipe_api

    api = _pipe_api()
    handle = owner.stdout.buffer.raw._handle
    data = bytearray()
    while True:
        _remaining(owner.deadline, owner.cancel)
        available = W.DWORD()
        if not api.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            if ctypes.get_last_error() != 109:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PROBE_FAILED", "Probe stdout query failed."
                )
            if process.poll() is not None:
                break
        if available.value:
            if available.value > MAX_PROBE_OUTPUT - len(data):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PROBE_INVALID", "Probe output exceeded its bound."
                )
            buffer = ctypes.create_string_buffer(available.value)
            count = W.DWORD()
            if (
                not api.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None)
                or not count.value
            ):
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PROBE_FAILED", "Probe stdout read failed."
                )
            data.extend(buffer.raw[: count.value])
        elif owner.cancel is not None:
            owner.cancel.wait(0.01)
        else:
            time.sleep(0.01)
    if process.returncode != 0:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROBE_FAILED",
            f"Fixed probe exited with {process.returncode}: {bytes(data[-1000:]).decode('utf-8', errors = 'replace')}",
        )
    return bytes(data)


def run_python_probe(
    selected_executable,
    store_root,
    *,
    timeout = 90,
    cancel = None,
):
    """Run fixed host-controlled checks; return only after every owner cleaned up."""
    from ..windows_lpac import _probe_network_endpoints
    from ..os_sandbox import spawn_prepared_launch
    from .launch import _remaining
    from .profiles import PYTHON_PROFILE
    from contextlib import ExitStack
    from .dns_probe import dns_probe

    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PROBE_INVALID", "Invalid fixed probe deadline.")
    start = time.monotonic()
    deadline = start + timeout
    _remaining(deadline, cancel)
    with _probe_network_endpoints() as endpoints, ExitStack() as controls:
        prepared = prepare_python_probe(
            selected_executable,
            store_root,
            timeout = _remaining(deadline, cancel),
            cancel = cancel,
            network_ports = tuple(address[1] for _, _, address in endpoints),
        )
        owner = prepared.spawn_callback.__self__
        try:
            dns = controls.enter_context(
                dns_probe(deadline = deadline, cancel = cancel, nonce = owner.nonce)
            )
            server = _prepare_host_pipe(owner)
            process = spawn_prepared_launch(
                prepared,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                close_fds = True,
                creationflags = subprocess.CREATE_NO_WINDOW,
                cwd = prepared.workdir,
                env = prepared.env,
            )
            data = _collect_probe_output(owner, process)
            catalog_binding_digest = owner.catalog.binding_digest
            checks, dns_context = _parse_probe_output(
                data, owner.nonce, owner.published.core.runtime.version, network = True
            )
            _verify_host_pipe(owner, server)
            queries = json.loads(data).get("dns_queries")
            checks += dns.verify(queries)
            from .protocol import LaunchBinding

            expected_binding = LaunchBinding(
                process.pid,
                owner.nonce,
                bytes.fromhex(PYTHON_PROFILE.digest),
                bytes.fromhex(owner.published.content_digest),
            )
            if owner.startup_binding != expected_binding:
                raise WindowsRuntimeError(
                    "WINDOWS_SANDBOX_PROBE_INVALID", "Native startup evidence is missing."
                )
            checks += ("native_expanded_startup_gate",)
        finally:
            try:
                owner.cleanup()
            except Exception as error:
                failure = WindowsRuntimeError(
                    "WINDOWS_SANDBOX_CLEANUP_FAILED", "Fixed probe cleanup failed."
                )
                failure.retained_launch = owner
                raise failure from error
    _remaining(deadline, cancel)
    return ProbeObservations(
        owner.published.content_digest,
        owner.published.core.digest,
        owner.published.artifacts.digest,
        PYTHON_PROFILE.digest,
        owner.published.core.runtime.version,
        checks,
        time.monotonic() - start,
        dns_context,
        catalog_binding_digest = catalog_binding_digest,
    )
