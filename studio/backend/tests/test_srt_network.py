# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native Unix-socket transport tests; all upstream traffic stays on loopback."""

import base64
import os
from pathlib import Path
import socket
import tempfile
import threading
import time

import pytest

from core.inference.network_proxy import AllowlistProxy, NetworkAllowlist
from core.inference.srt_network import SrtNetworkTransport
from .test_network_proxy import _EchoUpstream, _client_hello

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason = "private Unix socket authority requires POSIX ownership"
)


def _proxy(**options):
    return AllowlistProxy(NetworkAllowlist.from_entries(["upstream.test"]), **options)


def _connect(path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(path)
    return client


def _request(
    transport,
    method = "CONNECT",
    host = "upstream.test:443",
    authenticated = False,
):
    client = _connect(transport.http_socket_path)
    auth = ""
    if authenticated:
        credential = base64.b64encode(
            f"sandbox:{transport.proxy.credential.token}".encode()
        ).decode()
        auth = f"Proxy-Authorization: Basic {credential}\r\n"
    client.sendall(f"{method} {host} HTTP/1.1\r\nHost: {host}\r\n{auth}\r\n".encode())
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = client.recv(4096)
        if not chunk:
            break
        response += chunk
    return client, response


@pytest.mark.parametrize(
    "method,host,authenticated,status",
    [
        ("GET", "upstream.test:443", False, b"405"),
        ("GET", "upstream.test:443", True, b"405"),
        ("CONNECT", "other.test:443", True, b"403"),
        ("CONNECT", "upstream.test:80", True, b"403"),
    ],
)
def test_http_uds_retains_authenticated_https_only_policy(method, host, authenticated, status):
    with SrtNetworkTransport(_proxy()) as transport:
        client, response = _request(transport, method, host, authenticated)
        client.close()
        assert status in response.split(b"\r\n", 1)[0]


def test_http_uds_retains_private_address_denial():
    with SrtNetworkTransport(_proxy(resolver = lambda host, port: ["127.0.0.1"])) as transport:
        client, response = _request(transport)
        client.close()
        assert b"403" in response.split(b"\r\n", 1)[0]


def test_allowed_tls_bytes_cross_uds_and_close_ends_tunnel():
    upstream = _EchoUpstream()
    transport = SrtNetworkTransport(
        _proxy(
            resolver = lambda host, port: ["127.0.0.1"],
            allowed_ports = {upstream.port},
            require_public = False,
        )
    )
    try:
        transport.start()
        client, response = _request(transport, host = f"upstream.test:{upstream.port}")
        assert b"200" in response.split(b"\r\n", 1)[0]
        hello = _client_hello("upstream.test")
        client.sendall(hello)
        received = b""
        while len(received) < len(hello):
            received += client.recv(len(hello) - len(received))
        assert received == hello
        directory = Path(transport.http_socket_path).parent
        transport.close()
        assert not directory.exists()
        assert not transport.proxy._thread.is_alive()
        assert not transport._refuser.is_alive()
        assert not transport._watchdog.is_alive()
        assert not client.recv(1)
        client.close()
    finally:
        transport.close()
        upstream.close()


def test_socks_refuses_without_input_or_outbound_resolution():
    def never_resolve(*args):
        pytest.fail("SOCKS must never resolve or connect")

    with SrtNetworkTransport(_proxy(resolver = never_resolve)) as transport:
        for _ in range(20):
            with _connect(transport.socks_socket_path) as client:
                assert client.recv(2) == b"\x05\xff"
                assert client.recv(1) == b""


def test_private_paths_and_credential_environment():
    with SrtNetworkTransport(_proxy(), lifetime_seconds = None) as transport:
        directory = Path(transport.http_socket_path).parent
        if os.name == "posix":
            assert directory.stat().st_mode & 0o777 == 0o700
            assert Path(transport.http_socket_path).stat().st_mode & 0o777 == 0o600
        token = transport.proxy.credential.token
        assert token not in transport.http_socket_path + transport.socks_socket_path
        assert all(token not in value for value in transport.environment.values())
        assert "127.0.0.1:3128" in transport.environment["HTTPS_PROXY"]
        assert transport._watchdog is None


def test_lifetime_closes_listeners_and_removes_paths():
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds = 0.1).start()
    directory = Path(transport.http_socket_path).parent
    deadline = time.monotonic() + 3
    while directory.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not directory.exists()
    transport.close()


def test_partial_start_failure_closes_owned_http_listener(monkeypatch):
    transport = SrtNetworkTransport(_proxy())
    original = transport._listener

    def fail_second(path):
        if path.endswith("socks.sock"):
            raise OSError("controlled second bind failure")
        return original(path)

    monkeypatch.setattr(transport, "_listener", fail_second)
    with pytest.raises(OSError, match = "second bind"):
        transport.start()
    assert not Path(transport.http_socket_path).parent.exists()
    assert not transport.proxy._thread.is_alive()
    transport.close()


@pytest.mark.parametrize("lifetime", [0, -1, float("inf"), float("nan")])
def test_invalid_lifetime_is_rejected(lifetime):
    with pytest.raises(ValueError):
        SrtNetworkTransport(_proxy(), lifetime_seconds = lifetime)


def test_tcp_cannot_opt_into_private_socket_authority():
    proxy = _proxy()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    with pytest.raises(ValueError, match = "POSIX Unix"):
        proxy.serve_private_unix_listener(listener)
    assert listener.fileno() == -1
    assert proxy._private_unix_authority is False
    proxy.close()


@pytest.mark.parametrize("parent_mode,socket_mode", [(0o755, 0o600), (0o700, 0o666)])
def test_unsafe_unix_path_cannot_opt_into_socket_authority(parent_mode, socket_mode):
    with tempfile.TemporaryDirectory(prefix = "srt-test-") as directory:
        os.chmod(directory, parent_mode)
        path = str(Path(directory) / "p.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        os.chmod(path, socket_mode)
        listener.listen(1)
        proxy = _proxy()
        with pytest.raises(ValueError, match = "0700"):
            proxy.serve_private_unix_listener(listener)
        assert listener.fileno() == -1
        assert proxy._private_unix_authority is False
        proxy.close()


def test_ordinary_unix_listener_still_requires_credential():
    with tempfile.TemporaryDirectory(prefix = "srt-test-") as directory:
        path = str(Path(directory) / "p.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(1)
        proxy = _proxy()
        proxy.serve_listener(listener)
        try:
            with _connect(path) as client:
                client.sendall(
                    b"CONNECT upstream.test:443 HTTP/1.1\r\nHost: upstream.test:443\r\n\r\n"
                )
                assert b"407" in client.recv(4096).split(b"\r\n", 1)[0]
        finally:
            proxy.close()


def test_concurrent_close_waits_for_owned_cleanup(monkeypatch):
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds = None).start()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = transport.proxy.close

    def delayed_close():
        entered.set()
        assert release.wait(2)
        original()

    monkeypatch.setattr(transport.proxy, "close", delayed_close)
    first = threading.Thread(target = transport.close)
    second = threading.Thread(target = lambda: (transport.close(), finished.set()))
    first.start()
    assert entered.wait(1)
    second.start()
    try:
        assert not finished.wait(0.05)
    finally:
        release.set()
        first.join(3)
        second.join(3)
    assert finished.is_set()
    assert not Path(transport.http_socket_path).parent.exists()


def test_proxy_cleanup_debt_does_not_skip_transport_cleanup(monkeypatch):
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds = None).start()
    original = transport.proxy.close

    def fail_after_close():
        original()
        raise RuntimeError("controlled proxy cleanup debt")

    monkeypatch.setattr(transport.proxy, "close", fail_after_close)
    with pytest.raises(RuntimeError, match = "SRT proxy cleanup failed"):
        transport.close()
    assert not Path(transport.http_socket_path).parent.exists()
    assert not transport._refuser.is_alive()
    with pytest.raises(RuntimeError, match = "transport cleanup failed"):
        transport.close()
