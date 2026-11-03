# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The network guard blocks unasked-for traffic but keeps the configured test server reachable."""

from __future__ import annotations

import errno
import socket
import threading

import pytest


def _own_routable_address(hostname: str) -> str | None:
    """This host's own non-loopback IPv4, or None if it does not have a usable one."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return None
    for info in infos:
        address = info[4][0]
        if not address.startswith("127.") and address != "0.0.0.0":
            return address
    return None


@pytest.fixture
def offbox_server(monkeypatch):
    """The routable address stands in for a remote server; the name must be configured before resolving."""
    hostname = socket.gethostname()
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", f"http://{hostname}")

    address = _own_routable_address(hostname)
    if address is None:
        pytest.skip("host has no non-loopback IPv4 to stand in for a remote server")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((address, 0))
    except OSError:
        server.close()
        pytest.skip(f"cannot bind {address} on this runner")
    server.listen(8)
    port = server.getsockname()[1]

    def _accept_quietly():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.close()

    threading.Thread(target = _accept_quietly, daemon = True).start()
    try:
        yield hostname, address, port
    finally:
        server.close()


def test_a_server_configured_by_name_is_reachable(monkeypatch, offbox_server):
    """Allowing the hostname alone is not enough: create_connection dials the resolved numeric address."""
    hostname, _address, port = offbox_server
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", f"http://{hostname}:{port}")

    socket.create_connection((hostname, port), timeout = 10).close()


def test_a_server_configured_by_address_is_reachable(monkeypatch, offbox_server):
    """The same endpoint written as an address, which skips the resolver entirely."""
    _hostname, address, port = offbox_server
    monkeypatch.setenv("STUDIO_TEST_URL", f"http://{address}:{port}")

    socket.create_connection((address, port), timeout = 10).close()


def test_resolving_an_address_literal_does_not_make_it_dialable():
    """Resolving an address literal must not make it dialable; the connect must stay refused."""
    socket.getaddrinfo("169.254.169.254", 80, socket.AF_INET, socket.SOCK_STREAM)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match = "outbound network blocked"):
            sock.connect(("169.254.169.254", 80))
    finally:
        sock.close()


def test_an_unconfigured_name_fails_at_resolution():
    """Blocked names fail the way an unresolvable name does, which callers already handle."""
    with pytest.raises(socket.gaierror, match = "name resolution blocked"):
        socket.getaddrinfo("huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)


def test_a_byte_hostname_is_read_rather_than_waved_through():
    """A bytes hostname is still a hostname: unmatched, it fell through to allow instead of blocking."""
    with pytest.raises(socket.gaierror, match = "name resolution blocked"):
        socket.getaddrinfo(b"huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)


def test_a_byte_hostname_for_a_configured_server_still_works(monkeypatch, offbox_server):
    """Reading the byte form must mean reading it, not refusing it."""
    hostname, _address, port = offbox_server
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", f"http://{hostname}:{port}")

    infos = socket.getaddrinfo(hostname.encode(), port, socket.AF_INET, socket.SOCK_STREAM)
    assert infos


def test_connect_ex_reports_the_block_the_way_it_reports_a_failure():
    """connect_ex must report a block as an errno, not an exception, as run.py's port probe expects."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        assert sock.connect_ex(("169.254.169.254", 80)) == errno.ENETUNREACH
    finally:
        sock.close()


def test_a_fixture_can_ask_for_the_traffic_it_needs(
    monkeypatch, allow_outbound_network, offbox_server, forget_resolved_servers
):
    """Fixtures lift the guard by dialing a local address, so no DNS lookup can stall the run."""
    _hostname, address, port = offbox_server
    monkeypatch.delenv("UNSLOTH_E2E_BASE_URL", raising = False)
    forget_resolved_servers()

    with pytest.raises(OSError, match = "outbound network blocked"):
        socket.create_connection((address, port), timeout = 10)

    with allow_outbound_network():
        socket.create_connection((address, port), timeout = 10).close()

    with pytest.raises(OSError, match = "outbound network blocked"):
        socket.create_connection((address, port), timeout = 10)


def test_the_proxy_bypass_covers_the_local_server_too(monkeypatch, no_proxy_bypass_value):
    """The proxy bypass must cover loopback too, or a proxied request to the local server is refused."""
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", "http://studio.example.internal:8000")
    bypass = no_proxy_bypass_value("corp.example, 10.0.0.1").split(",")

    assert "127.0.0.1" in bypass
    assert "localhost" in bypass
    assert "studio.example.internal" in bypass, "the configured server must still be bypassed"
    assert bypass[:2] == ["corp.example", "10.0.0.1"], "an existing NO_PROXY must survive"
    assert len(bypass) == len(set(bypass)), "entries must not be duplicated on re-entry"


def test_neither_no_proxy_spelling_loses_what_the_other_carried(no_proxy_bypass_value):
    """Both NO_PROXY spellings must be merged, not overwritten, or the bypass stops applying to the host."""
    bypass = no_proxy_bypass_value("only-in-uppercase.example", "").split(",")
    assert "only-in-uppercase.example" in bypass

    both = no_proxy_bypass_value("upper.example", "lower.example").split(",")
    assert "upper.example" in both and "lower.example" in both
    assert len(both) == len(set(both))


def test_loopback_stays_open():
    """The guard must not disturb the in-process servers most of the suite runs on."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        socket.create_connection(server.getsockname(), timeout = 10).close()
    finally:
        server.close()
