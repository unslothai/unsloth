# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DNS packet controls, native resolver positives, and fail-closed observations."""

import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import dns_probe as dns
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


def _results(status, records = False):
    return {
        name: dict(
            status = status,
            has_records = records,
            **({} if name.endswith("_w_denied") else {"query_status": status}),
        )
        for name in dns.DNS_CHECKS
    }


def _packet(name):
    labels = b"".join(bytes([len(label)]) + label.encode() for label in name.rstrip(".").split("."))
    return struct.pack("!6H", 123, 0x100, 1, 0, 0, 0) + labels + b"\0\0\1\0\1"


def test_packet_response_binds_id_exact_question_and_zero_ttl_answer():
    packet = _packet("lpac-nonce-host.invalid.")
    name, end = dns._question(packet)
    assert name == "lpac-nonce-host.invalid." and end == len(packet)
    response = dns._response(packet, end)
    assert struct.unpack("!6H", response[:12]) == (123, 0x8180, 1, 1, 0, 0)
    assert response[12:end] == packet[12:]
    assert response[end:] == b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 0, 4) + socket.inet_aton(
        "127.0.0.1"
    )


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"x" * 11,
        struct.pack("!6H", 1, 0x8000, 1, 0, 0, 0) + b"\0\0\1\0\1",
        struct.pack("!6H", 1, 0, 2, 0, 0, 0) + b"\0\0\1\0\1",
        struct.pack("!6H", 1, 0, 1, 0, 0, 0) + b"\xc0\x0c\0\1\0\1",
        struct.pack("!6H", 1, 0, 1, 0, 0, 0) + b"\x03ab",
        _packet("valid.invalid.")[:-1],
        _packet("valid.invalid.")[:-4] + b"\0\x1c\0\1",
    ],
)
def test_malformed_or_non_a_question_is_not_answered(packet):
    with pytest.raises(ValueError):
        dns._question(packet)


@pytest.mark.parametrize("status", [5, 10013])
def test_exact_native_access_denial_is_accepted(status):
    dns._validate_results(_results(status), denied = True)


@pytest.mark.parametrize("status", [0, 87, 10022, 10050, 10107, 9003, 9501, 997])
def test_success_and_prerequisite_errors_never_count_as_denial(status):
    with pytest.raises(WindowsRuntimeError, match = "DNS_UNQUALIFIED"):
        dns._validate_results(_results(status), denied = True)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "records", "boolean", "query_mismatch", "wrong_row"]
)
def test_invalid_observations_fail_closed(mutation):
    results = _results(5)
    row = results[dns.DNS_CHECKS[0]]
    if mutation == "missing":
        results.pop(dns.DNS_CHECKS[0])
    elif mutation == "extra":
        results["other"] = row
    elif mutation == "records":
        row["has_records"] = True
    elif mutation == "boolean":
        row["status"] = True
    elif mutation == "query_mismatch":
        row["query_status"] = 87
    else:
        results[dns.DNS_CHECKS[0]] = None
    with pytest.raises(WindowsRuntimeError, match = "DNS_UNQUALIFIED"):
        dns._validate_results(results, denied = True)


@pytest.mark.parametrize("status,records", [(0, False), (5, False), (9003, False), (10107, False)])
def test_host_requires_actual_success_with_records(status, records):
    with pytest.raises(WindowsRuntimeError, match = "Host DNS positive"):
        dns._validate_results(_results(status, records), denied = False)


@pytest.mark.skipif(sys.platform != "win32", reason = "Owned Windows DNS UDP53 responder")
def test_native_udp_responder_observes_only_owned_names_and_releases_port():
    server = dns._Responder("lpac-owned-host.invalid.", "lpac-owned-sandbox.invalid.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            client.sendto(_packet(server.host_name), ("127.0.0.1", 53))
            assert client.recvfrom(4096)[0].endswith(socket.inet_aton("127.0.0.1"))
            assert server.host_seen.is_set() and not server.sandbox_seen.is_set()
            client.sendto(_packet(server.sandbox_name), ("127.0.0.1", 53))
            client.recvfrom(4096)
        control = dns.DnsProbe("", server, time.monotonic() + 2, None)
        with pytest.raises(WindowsRuntimeError, match = "received a restricted-worker"):
            control.verify(_results(5))
    finally:
        server.close(time.monotonic() + 2)
    assert not server.thread.is_alive() and server.socket.fileno() == -1
    next_server = dns._Responder(server.host_name, server.sandbox_name)
    next_server.close(time.monotonic() + 2)


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows DNS controls")
def test_native_host_positive_control_reaches_owned_server_and_default_resolvers():
    with dns.dns_probe(deadline = time.monotonic() + 15) as control:
        assert "perform_dns_probe" in control.payload_source
        assert "-sandbox.invalid." in control.payload_source
        assert control._server.host_seen.is_set()
        # This verifies the verifier only, not actual LPAC denial. Parent native
        # integration must supply actual restricted-worker observations.
        assert control.verify(_results(5)) == dns.DNS_CHECKS
        server = control._server
    assert not server.thread.is_alive() and server.socket.fileno() == -1


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows exclusive UDP port ownership")
def test_busy_dns_port_refuses_before_host_process(monkeypatch):
    server = dns._Responder("host.invalid.", "sandbox.invalid.")
    monkeypatch.setattr(dns, "_host_results", lambda *args: pytest.fail("Host query started"))
    try:
        with pytest.raises(WindowsRuntimeError, match = "port 53 is unavailable"):
            with dns.dns_probe(deadline = time.monotonic() + 2):
                pytest.fail("Yielded DNS payload despite unavailable control")
    finally:
        server.close(time.monotonic() + 2)


def test_cancelled_probe_does_not_bind_or_spawn(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(dns, "_Responder", lambda *args: pytest.fail("Socket created"))
    with pytest.raises(WindowsRuntimeError, match = "CANCELLED"):
        with dns.dns_probe(deadline = time.monotonic() + 2, cancel = cancel):
            pytest.fail("Yielded cancelled probe")


@pytest.mark.parametrize("cancelled", [False, True])
def test_blocked_native_control_is_killed_and_reaped_under_deadline(monkeypatch, cancelled):
    processes = []
    from core.inference.windows_sandbox import native_compat

    original = native_compat.WindowsLpacProcess

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(native_compat, "WindowsLpacProcess", spawn)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    if cancelled:
        timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(
            WindowsRuntimeError, match = "CANCELLED" if cancelled else "PREPARATION_TIMEOUT"
        ):
            dns._host_results(
                "def perform_dns_probe():\n import time\n time.sleep(60)", start + 1, cancel
            )
    finally:
        if cancelled:
            timer.cancel()
            timer.join()
    assert time.monotonic() - start < 1.5
    assert len(processes) == 1 and processes[0].poll() is not None
    assert not processes[0]._handle and not processes[0]._thread_handle
    assert not processes[0]._unsloth_job._handle


@pytest.mark.parametrize("deadline", [None, True, float("nan"), float("inf")])
def test_invalid_deadlines_fail_before_resource_creation(deadline):
    with pytest.raises(WindowsRuntimeError, match = "Invalid DNS"):
        with dns.dns_probe(deadline = deadline):
            pytest.fail("Invalid deadline accepted")


@pytest.mark.parametrize("nonce", [b"", b"x" * 31, b"x" * 33, "x" * 32, bytearray(32), None])
def test_payload_source_rejects_invalid_nonce(nonce):
    with pytest.raises(WindowsRuntimeError, match = "exactly 32 bytes"):
        dns.payload_source(nonce)


def test_payload_source_binds_entire_nonce_with_valid_dns_labels():
    nonce = bytes(range(32))
    host, sandbox = dns._names(nonce)
    assert dns.payload_source(nonce) == dns._source(sandbox)
    assert all(len(label) <= 63 for label in sandbox.split("."))
    assert dns._question(_packet(sandbox))[0] == sandbox
    changed = nonce[:-1] + bytes([nonce[-1] ^ 1])
    assert dns.payload_source(nonce) != dns.payload_source(changed)
    assert host != sandbox


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows DNS controls")
def test_context_and_worker_payload_share_exact_supplied_nonce():
    nonce = b"n" * 32
    with dns.dns_probe(deadline = time.monotonic() + 15, nonce = nonce) as control:
        assert control.payload_source == dns.payload_source(nonce)
        assert control._server.host_name == dns._names(nonce)[0]


@pytest.mark.skipif(sys.platform != "win32", reason = "Native DNS responder cleanup")
def test_host_prerequisite_failure_closes_server_and_never_yields(monkeypatch):
    servers = []
    original = dns._Responder

    def create(*args):
        server = original(*args)
        servers.append(server)
        return server

    monkeypatch.setattr(dns, "_Responder", create)
    monkeypatch.setattr(dns, "_host_results", lambda *args: _results(10107))
    with pytest.raises(WindowsRuntimeError, match = "Host DNS positive control failed"):
        with dns.dns_probe(deadline = time.monotonic() + 2):
            pytest.fail("Prerequisite failure yielded payload")
    assert len(servers) == 1 and not servers[0].thread.is_alive()
    assert servers[0].socket.fileno() == -1


@pytest.mark.skipif(sys.platform != "win32", reason = "Native DNS responder cleanup")
def test_cancel_during_verification_closes_owned_responder():
    cancel = threading.Event()
    with pytest.raises(WindowsRuntimeError, match = "CANCELLED"):
        with dns.dns_probe(deadline = time.monotonic() + 15, cancel = cancel) as control:
            server = control._server
            cancel.set()
            control.verify(_results(5))
    assert not server.thread.is_alive() and server.socket.fileno() == -1


@pytest.mark.skipif(sys.platform != "win32", reason = "Native DNS host Job")
def test_host_control_uses_base_python_and_cannot_spawn_redirector_child(monkeypatch):
    monkeypatch.setattr(dns.sys, "executable", r"C:\missing\venv\Scripts\python.exe")
    source = """def perform_dns_probe():
 import subprocess,sys
 try:
  child=subprocess.Popen([sys.executable,'-I','-S','-c','pass'])
 except OSError:
  return {'child_denied':True,'executable':sys.executable}
 child.wait()
 return {'child_denied':False}
"""
    result = dns._host_results(source, time.monotonic() + 5, None)
    assert result["child_denied"] is True
    assert os.path.samefile(result["executable"], Path(sys.base_prefix) / "python.exe")
