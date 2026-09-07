# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Owned DNS positive controls and strict post-drop denial verification.

The responder binds only loopback UDP 53. No resolver, firewall, or host policy
is changed. Native host queries run in a disposable process to bound blocking
DNS APIs; the restricted worker receives the same fixed ctypes implementation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import sys
import threading
import time

from .profiles import WindowsRuntimeError

DNS_CHECKS = ("dns_directed_denied", "dns_default_ex_denied", "dns_default_w_denied")
_DNS_SOURCE = r"""
import ctypes as _dns_c
from ctypes import wintypes as _dns_w

class _DnsAddress(_dns_c.Structure):
    _fields_ = [('sockaddr', _dns_c.c_ubyte * 32), ('data', _dns_w.DWORD * 8)]
class _DnsAddresses(_dns_c.Structure):
    _fields_ = [('maxcount', _dns_w.DWORD), ('count', _dns_w.DWORD), ('tag', _dns_w.DWORD),
                ('family', _dns_w.WORD), ('reserved_word', _dns_w.WORD), ('flags', _dns_w.DWORD),
                ('match', _dns_w.DWORD), ('reserved1', _dns_w.DWORD), ('reserved2', _dns_w.DWORD),
                ('addresses', _DnsAddress * 1)]
class _DnsRequest(_dns_c.Structure):
    _fields_ = [('version', _dns_w.ULONG), ('name', _dns_w.LPCWSTR), ('query_type', _dns_w.WORD),
                ('options', _dns_c.c_ulonglong), ('servers', _dns_c.c_void_p), ('interface', _dns_w.ULONG),
                ('callback', _dns_c.c_void_p), ('context', _dns_c.c_void_p)]
class _DnsResult(_dns_c.Structure):
    _fields_ = [('version', _dns_w.ULONG), ('status', _dns_w.LONG), ('options', _dns_c.c_ulonglong),
                ('records', _dns_c.c_void_p), ('reserved', _dns_c.c_void_p)]

def perform_dns_probe():
    if (_dns_c.sizeof(_DnsAddresses), _dns_c.sizeof(_DnsRequest), _dns_c.sizeof(_DnsResult)) != (96, 64, 32):
        raise RuntimeError('Unsupported DNS ABI')
    dns = _dns_c.WinDLL('dnsapi.dll', use_last_error=True, winmode=0x800)
    dns.DnsRecordListFree.argtypes = [_dns_c.c_void_p, _dns_c.c_int]
    dns.DnsRecordListFree.restype = None
    dns.DnsQueryEx.argtypes = [_dns_c.POINTER(_DnsRequest), _dns_c.POINTER(_DnsResult), _dns_c.c_void_p]
    dns.DnsQueryEx.restype = _dns_w.LONG
    dns.DnsQuery_W.argtypes = [_dns_w.LPCWSTR, _dns_w.WORD, _dns_w.DWORD, _dns_c.c_void_p,
                              _dns_c.POINTER(_dns_c.c_void_p), _dns_c.c_void_p]
    dns.DnsQuery_W.restype = _dns_w.LONG
    output = {}
    for label in ('dns_directed_denied', 'dns_default_ex_denied', 'dns_default_w_denied'):
        # BYPASS_CACHE, NO_HOSTS_FILE, WIRE_ONLY; never accept a cache-only control.
        options = 0x08 | 0x100 | 0x40
        if label == 'dns_default_w_denied':
            records = _dns_c.c_void_p()
            status = dns.DnsQuery_W('example.com.', 1, options, None, _dns_c.byref(records), None)
            output[label] = {'status': int(status), 'has_records': bool(records.value)}
            if records.value:
                dns.DnsRecordListFree(records, 1)
            continue
        request = _DnsRequest(version=1, name='example.com.', query_type=1, options=options)
        servers = _DnsAddresses(maxcount=_dns_c.sizeof(_DnsAddresses), count=1, family=2)
        servers.addresses[0].sockaddr[0] = 2
        servers.addresses[0].sockaddr[4] = 127
        servers.addresses[0].sockaddr[7] = 1
        if label == 'dns_directed_denied':
            request.name = DNS_PROBE_NAME
            request.servers = _dns_c.addressof(servers)
        result = _DnsResult(version=1)
        status = dns.DnsQueryEx(_dns_c.byref(request), _dns_c.byref(result), None)
        output[label] = {'status': int(status), 'query_status': int(result.status),
                         'has_records': bool(result.records)}
        if result.records:
            dns.DnsRecordListFree(result.records, 1)
    return output
"""


def _error(message, code = "WINDOWS_SANDBOX_DNS_UNQUALIFIED"):
    return WindowsRuntimeError(code, message)


def _check(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise _error("DNS qualification cancelled.", "WINDOWS_SANDBOX_CANCELLED")
    if time.monotonic() >= deadline:
        raise _error("DNS qualification exceeded its deadline.", "WINDOWS_SANDBOX_STARTUP_TIMEOUT")


def _question(packet):
    if len(packet) < 12:
        raise ValueError("Short DNS packet")
    _, flags, questions, _, _, _ = struct.unpack("!6H", packet[:12])
    if flags & 0xF800 or questions != 1:
        raise ValueError("Unexpected DNS question header")
    labels, position = [], 12
    while True:
        if position >= len(packet):
            raise ValueError("Truncated DNS name")
        length = packet[position]
        position += 1
        if length == 0:
            break
        if length > 63 or position + length > len(packet):
            raise ValueError("Invalid or compressed DNS label")
        labels.append(packet[position : position + length].decode("ascii").lower())
        position += length
    if position + 4 > len(packet) or packet[position : position + 4] != b"\x00\x01\x00\x01":
        raise ValueError("Expected an IN A query")
    name = ".".join(labels) + "."
    if len(name) > 254:
        raise ValueError("Oversized DNS name")
    return name, position + 4


def _response(packet, end):
    # Copy only the checked question. EDNS additional records are not reflected.
    header = packet[:2] + struct.pack("!5H", 0x8180, 1, 1, 0, 0)
    answer = b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 0, 4) + b"\x7f\x00\x00\x01"
    return header + packet[12:end] + answer


def _names(nonce):
    if type(nonce) is not bytes or len(nonce) != 32:
        raise _error("DNS probe nonce must contain exactly 32 bytes.")
    encoded = nonce.hex()
    prefix = f"lpac-{encoded[:32]}.{encoded[32:]}"
    return prefix + "-host.invalid.", prefix + "-sandbox.invalid."


def payload_source(nonce):
    """Derive fixed probe code from the broker's existing 32-byte challenge."""
    return _source(_names(nonce)[1])


def _source(name):
    return "DNS_PROBE_NAME = " + repr(name) + "\n" + _DNS_SOURCE


def _host_results(source, deadline, cancel):
    from .preparation import _run_worker

    _check(deadline, cancel)
    executable = Path(sys.base_prefix) / "python.exe"
    if not executable.is_file():
        raise _error("The broker's base Python executable is unavailable.")
    command = source + "\nimport json\nprint(json.dumps(perform_dns_probe()), flush=True)\n"
    # A venv python.exe may redirect to a child. Use the matching base runtime
    # and the existing creation-time Job owner, which forbids child processes.
    stdout, _ = _run_worker(
        [str(executable), "-I", "-S", "-c", command],
        os.environ.copy(),
        str(executable.parent),
        deadline = deadline - 0.1,
        cancel = cancel,
    )
    if len(stdout) > 8192:
        raise _error("Host DNS control returned oversized results.")
    try:
        return json.loads(stdout)
    except (ValueError, UnicodeError) as error:
        raise _error("Host DNS control returned invalid results.") from error


def _validate_results(results, *, denied):
    if type(results) is not dict or set(results) != set(DNS_CHECKS):
        raise _error("Missing or unexpected DNS observations.")
    for name in DNS_CHECKS:
        row = results[name]
        expected_keys = {"status", "has_records"}
        if name != "dns_default_w_denied":
            expected_keys.add("query_status")
        if type(row) is not dict or set(row) != expected_keys:
            raise _error(f"Invalid DNS observation: {name}.")
        status = row["status"]
        if type(status) is not int or type(row["has_records"]) is not bool:
            raise _error(f"Invalid DNS status: {name}.")
        if "query_status" in row and (
            type(row["query_status"]) is not int or row["query_status"] != status
        ):
            raise _error(f"DNS API and query status disagree: {name}.")
        if denied:
            if status not in (5, 10013) or row["has_records"]:
                raise _error(
                    f"DNS query was not denied by access control: {name}, status={status}."
                )
        elif status != 0 or not row["has_records"]:
            raise _error(f"Host DNS positive control failed: {name}, status={status}.")


@dataclass
class DnsProbe:
    payload_source: str
    _server: object
    _deadline: float
    _cancel: object

    def verify(self, results):
        _check(self._deadline, self._cancel)
        _validate_results(results, denied = True)
        # Let the responder consume any already queued packet before checking.
        until = time.monotonic() + 0.1
        while time.monotonic() < until:
            _check(self._deadline, self._cancel)
            time.sleep(min(0.01, until - time.monotonic()))
        self._server.require_healthy()
        if self._server.sandbox_seen.is_set():
            raise _error("The owned DNS responder received a restricted-worker query.")
        return DNS_CHECKS


class _Responder:
    def __init__(self, host_name, sandbox_name):
        self.host_name, self.sandbox_name = host_name, sandbox_name
        self.host_seen, self.sandbox_seen = threading.Event(), threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.socket.bind(("127.0.0.1", 53))
            self.socket.settimeout(0.05)
        except BaseException:
            self.socket.close()
            raise
        self.thread = threading.Thread(target = self._run, name = "lpac-owned-dns", daemon = True)
        try:
            self.thread.start()
        except BaseException:
            self.socket.close()
            raise

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    packet, peer = self.socket.recvfrom(4096)
                except socket.timeout:
                    continue
                try:
                    name, end = _question(packet)
                except (ValueError, UnicodeError):
                    continue
                if peer[0] != "127.0.0.1" or name not in (self.host_name, self.sandbox_name):
                    continue
                (self.host_seen if name == self.host_name else self.sandbox_seen).set()
                self.socket.sendto(_response(packet, end), peer)
        except BaseException as error:
            self.error = error

    def require_healthy(self):
        if self.error is not None or not self.thread.is_alive():
            raise _error("The owned DNS responder stopped unexpectedly.")

    def close(self, deadline):
        self.stop.set()
        self.thread.join(timeout = max(0.001, deadline - time.monotonic()))
        self.socket.close()
        if self.thread.is_alive():
            raise _error("Owned DNS responder did not stop.", "WINDOWS_SANDBOX_CLEANUP_FAILED")


@contextmanager
def dns_probe(
    *,
    deadline,
    cancel = None,
    nonce = None,
):
    """Yield fixed payload source only after directed and default host controls pass."""
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise _error("Invalid DNS qualification deadline.")
    _check(deadline, cancel)
    if os.name != "nt":
        raise _error("Native DNS qualification requires Windows.")
    if nonce is None:
        nonce = secrets.token_bytes(32)
    host_name, sandbox_name = _names(nonce)
    try:
        server = _Responder(host_name, sandbox_name)
    except OSError as error:
        raise _error("Owned DNS loopback UDP port 53 is unavailable.") from error
    try:
        operation_deadline = deadline - 0.25
        results = _host_results(_source(host_name), operation_deadline, cancel)
        _validate_results(results, denied = False)
        server.require_healthy()
        if not server.host_seen.is_set():
            raise _error("Host DNS query did not reach the owned responder.")
        yield DnsProbe(_source(sandbox_name), server, operation_deadline, cancel)
    finally:
        server.close(deadline)
