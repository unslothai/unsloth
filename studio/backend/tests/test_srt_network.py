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
from core.inference.srt_network import TlsTrustSnapshot
from .test_network_proxy import _EchoUpstream, _client_hello

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="private Unix socket authority requires POSIX ownership"
)


def test_trust_snapshot_copies_hashed_symlink_bytes_without_sibling_secrets(tmp_path, monkeypatch):
    from core.inference import srt_network

    store = tmp_path / "store"
    store.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    cert = external / "certificate.pem"
    cert.write_bytes(b"exact certificate bytes\n")
    (external / "secret.key").write_bytes(b"private key")
    (store / "abcdef01.0").symlink_to(cert)
    (store / "abcdef01.r0").write_bytes(b"exact CRL bytes")
    (store / "unrelated.secret").write_bytes(b"secret")
    (store / "12345678.0").mkdir()
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (None, str(store)))
    with TlsTrustSnapshot({}, parent_dir=str(tmp_path)) as trust:
        copied = Path(trust.environment["SSL_CERT_DIR"])
        assert copied.stat().st_mode & 0o777 == 0o700
        assert sorted(p.name for p in copied.iterdir()) == ["abcdef01.0", "abcdef01.r0"]
        assert (copied / "abcdef01.0").read_bytes() == cert.read_bytes()
        assert not (copied / "abcdef01.0").is_symlink()
        assert trust.read_roots == (str(copied),)
        assert "SSL_CERT_FILE" not in trust.environment
        assert "REQUESTS_CA_BUNDLE" not in trust.environment
    assert not copied.exists()
    trust.close()


@pytest.mark.parametrize("limit_kind", ["entries", "bytes"])
def test_trust_snapshot_limits_fail_closed_and_remove_partial_copy(
    tmp_path, monkeypatch, limit_kind
):
    from core.inference import srt_network

    store = tmp_path / "store"
    store.mkdir()
    (store / "abcdef01.0").write_bytes(b"certificate")
    (store / "abcdef01.1").write_bytes(b"certificate")
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (None, str(store)))
    monkeypatch.setattr(
        srt_network,
        "MAX_CAPATH_ENTRIES" if limit_kind == "entries" else "MAX_TRUST_SNAPSHOT_BYTES",
        1,
    )
    with pytest.raises(ValueError, match="limit"):
        TlsTrustSnapshot({}, parent_dir=str(tmp_path)).start()
    assert not list(tmp_path.glob("srt-trust-*"))


@pytest.mark.parametrize("disabled", ["", "/does/not/exist"])
def test_trust_snapshot_preserves_explicit_disabled_stores(tmp_path, monkeypatch, disabled):
    from core.inference import srt_network

    bundle = tmp_path / "bundle.pem"
    bundle.write_bytes(b"default bundle")
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (str(bundle), str(tmp_path)))
    base = {"SSL_CERT_FILE": disabled, "SSL_CERT_DIR": disabled}
    with TlsTrustSnapshot(base) as trust:
        assert trust.environment == base
        assert trust.read_roots == ()


def test_trust_snapshot_canonical_cafile_and_custom_capath(tmp_path, monkeypatch):
    from core.inference import srt_network

    bundle = tmp_path / "bundle.pem"
    bundle.write_bytes(b"bundle")
    alias = tmp_path / "bundle-link.pem"
    alias.symlink_to(bundle)
    store = tmp_path / "custom-store"
    store.mkdir()
    (store / "deadbeef.12").write_bytes(b"custom trust")
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (None, None))
    with TlsTrustSnapshot({"SSL_CERT_FILE": str(alias), "SSL_CERT_DIR": str(store)}) as trust:
        assert trust.environment["SSL_CERT_FILE"] == str(bundle)
        assert trust.environment["REQUESTS_CA_BUNDLE"] == str(bundle)
        assert (
            Path(trust.environment["SSL_CERT_DIR"], "deadbeef.12").read_bytes() == b"custom trust"
        )
        assert trust.read_roots == (str(bundle), trust.environment["SSL_CERT_DIR"])


def test_trust_snapshot_parent_environment_and_exception_cleanup(tmp_path, monkeypatch):
    from core.inference import srt_network

    store = tmp_path / "selected-store"
    store.mkdir()
    (store / "abcdef01.0").write_bytes(b"selected trust")
    monkeypatch.setenv("SSL_CERT_DIR", str(store))
    monkeypatch.setenv("SSL_CERT_FILE", "")
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (None, None))
    original = dict(os.environ)
    with pytest.raises(RuntimeError, match="launch failed"):
        with TlsTrustSnapshot(parent_dir=str(tmp_path)) as trust:
            copied = Path(trust.environment["SSL_CERT_DIR"])
            assert copied.joinpath("abcdef01.0").read_bytes() == b"selected trust"
            assert trust.environment["SSL_CERT_FILE"] == ""
            raise RuntimeError("launch failed")
    assert dict(os.environ) == original
    assert not copied.exists()


def test_trust_snapshot_requests_bundle_exposes_only_canonical_file(tmp_path, monkeypatch):
    from core.inference import srt_network

    bundle = tmp_path / "requests.pem"
    bundle.write_bytes(b"operator selected bundle")
    alias = tmp_path / "requests-link.pem"
    alias.symlink_to(bundle)
    (tmp_path / "secret.key").write_bytes(b"private sibling")
    monkeypatch.setattr(srt_network, "_openssl_default_paths", lambda: (None, None))
    with TlsTrustSnapshot({"REQUESTS_CA_BUNDLE": str(alias)}) as trust:
        assert trust.environment == {"REQUESTS_CA_BUNDLE": str(bundle)}
        assert trust.read_roots == (str(bundle),)


def _proxy(**options):
    return AllowlistProxy(NetworkAllowlist.from_entries(["upstream.test"]), **options)


def _connect(path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(path)
    return client


def _request(
    transport,
    method="CONNECT",
    host="upstream.test:443",
    authenticated=False,
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
    with SrtNetworkTransport(_proxy(resolver=lambda host, port: ["127.0.0.1"])) as transport:
        client, response = _request(transport)
        client.close()
        assert b"403" in response.split(b"\r\n", 1)[0]


def test_allowed_tls_bytes_cross_uds_and_close_ends_tunnel():
    upstream = _EchoUpstream()
    transport = SrtNetworkTransport(
        _proxy(
            resolver=lambda host, port: ["127.0.0.1"],
            allowed_ports={upstream.port},
            require_public=False,
        )
    )
    try:
        transport.start()
        client, response = _request(transport, host=f"upstream.test:{upstream.port}")
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

    with SrtNetworkTransport(_proxy(resolver=never_resolve)) as transport:
        for _ in range(20):
            with _connect(transport.socks_socket_path) as client:
                assert client.recv(2) == b"\x05\xff"
                assert client.recv(1) == b""


def test_private_paths_and_credential_environment():
    with SrtNetworkTransport(_proxy(), lifetime_seconds=None) as transport:
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
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds=0.1).start()
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
    with pytest.raises(OSError, match="second bind"):
        transport.start()
    assert not Path(transport.http_socket_path).parent.exists()
    assert not transport.proxy._thread.is_alive()
    transport.close()


@pytest.mark.parametrize("lifetime", [0, -1, float("inf"), float("nan")])
def test_invalid_lifetime_is_rejected(lifetime):
    with pytest.raises(ValueError):
        SrtNetworkTransport(_proxy(), lifetime_seconds=lifetime)


def test_tcp_cannot_opt_into_private_socket_authority():
    proxy = _proxy()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    with pytest.raises(ValueError, match="POSIX Unix"):
        proxy.serve_private_unix_listener(listener)
    assert listener.fileno() == -1
    assert proxy._private_unix_authority is False
    proxy.close()


@pytest.mark.parametrize("parent_mode,socket_mode", [(0o755, 0o600), (0o700, 0o666)])
def test_unsafe_unix_path_cannot_opt_into_socket_authority(parent_mode, socket_mode):
    with tempfile.TemporaryDirectory(prefix="srt-test-") as directory:
        os.chmod(directory, parent_mode)
        path = str(Path(directory) / "p.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        os.chmod(path, socket_mode)
        listener.listen(1)
        proxy = _proxy()
        with pytest.raises(ValueError, match="0700"):
            proxy.serve_private_unix_listener(listener)
        assert listener.fileno() == -1
        assert proxy._private_unix_authority is False
        proxy.close()


def test_ordinary_unix_listener_still_requires_credential():
    with tempfile.TemporaryDirectory(prefix="srt-test-") as directory:
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
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds=None).start()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = transport.proxy.close

    def delayed_close():
        entered.set()
        assert release.wait(2)
        original()

    monkeypatch.setattr(transport.proxy, "close", delayed_close)
    first = threading.Thread(target=transport.close)
    second = threading.Thread(target=lambda: (transport.close(), finished.set()))
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
    transport = SrtNetworkTransport(_proxy(), lifetime_seconds=None).start()
    original = transport.proxy.close

    def fail_after_close():
        original()
        raise RuntimeError("controlled proxy cleanup debt")

    monkeypatch.setattr(transport.proxy, "close", fail_after_close)
    with pytest.raises(RuntimeError, match="SRT proxy cleanup failed"):
        transport.close()
    assert not Path(transport.http_socket_path).parent.exists()
    assert not transport._refuser.is_alive()
    with pytest.raises(RuntimeError, match="transport cleanup failed"):
        transport.close()
