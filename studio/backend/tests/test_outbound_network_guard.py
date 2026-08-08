# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: the suite's outbound-network guard blocks the Hub without blocking the suite.

``conftest._no_outbound_network`` exists so no test depends on huggingface.co being up and
fast. It has to hold two things apart that look alike from inside a socket call: traffic
nobody asked for, which must fail immediately, and the server an integration run was
deliberately pointed at, which must stay reachable.

The ways it got that wrong were only visible end to end, so this covers it there:
resolution and connection are separate hooks, and a rule enforced on one of them says
nothing about the other. CPU-only, and every connection here is to this same machine.
"""

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
    """Yield ``(hostname, address, port)`` for a listener on this host's routable address.

    That address stands in for a remote server: it is off loopback, so the guard treats
    it exactly as it would treat somebody else's machine, while no packet leaves the box.

    The name is configured before it is resolved, because the guard blocks resolution of
    anything unconfigured -- including, correctly, this machine's own name.

    Skips where the stand-in is not available: a runner with only loopback configured, or
    one whose hostname does not resolve, cannot express "a server that is not local".
    """
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
    """The regression: allowing the hostname alone is not enough.

    ``socket.create_connection`` resolves first and then dials the numeric result, so a
    rule that only knows the name refuses the connect that follows it -- the destination
    is by then an address that matches nothing. A configured endpoint was unusable.
    """
    hostname, _address, port = offbox_server
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", f"http://{hostname}:{port}")

    socket.create_connection((hostname, port), timeout = 10).close()


def test_a_server_configured_by_address_is_reachable(monkeypatch, offbox_server):
    """The same endpoint written as an address, which skips the resolver entirely."""
    _hostname, address, port = offbox_server
    monkeypatch.setenv("STUDIO_TEST_URL", f"http://{address}:{port}")

    socket.create_connection((address, port), timeout = 10).close()


def test_resolving_an_address_literal_does_not_make_it_dialable():
    """The literal exemption must not become a way through.

    Literals are exempt from the resolution rule because the SSRF tests resolve private
    ones on purpose to prove they get rejected. That exemption is about the lookup only:
    the connect has to stay refused, or those tests would start reaching the address
    they are asserting is unreachable.
    """
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
    """The regression: bytes are a hostname too.

    ``socket`` takes a name as ``str`` or ``bytes``. Compared as-is, the byte form
    matched no rule and fell through to whatever the non-string case did -- which was
    to allow it. That made ``getaddrinfo(b"huggingface.co", 443)`` a way straight out:
    real resolution, and the address it returned dialable afterwards.
    """
    with pytest.raises(socket.gaierror, match = "name resolution blocked"):
        socket.getaddrinfo(b"huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)


def test_a_byte_hostname_for_a_configured_server_still_works(monkeypatch, offbox_server):
    """Reading the byte form must mean reading it, not refusing it."""
    hostname, _address, port = offbox_server
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", f"http://{hostname}:{port}")

    infos = socket.getaddrinfo(hostname.encode(), port, socket.AF_INET, socket.SOCK_STREAM)
    assert infos


def test_connect_ex_reports_the_block_the_way_it_reports_a_failure():
    """connect_ex answers with an errno; callers branch on it rather than catching.

    ``run.py`` probes a port exactly that way. Raising here would send code that only
    handles a non-zero result down a path an ordinary failed connect_ex never takes.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        assert sock.connect_ex(("169.254.169.254", 80)) == errno.ENETUNREACH
    finally:
        sock.close()


def test_a_fixture_can_ask_for_the_traffic_it_needs(allow_outbound_network):
    """The escape hatch for fixtures built before the per-test guard exists.

    Checked at the resolver, which needs no listener and opens no connection. Inside
    the block the guard must not be what refuses: a runner with no DNS at all is free
    to refuse for its own reasons, and saying so is the point of reading the message.
    """
    with pytest.raises(socket.gaierror, match = "name resolution blocked"):
        socket.getaddrinfo("huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)

    with allow_outbound_network():
        try:
            assert socket.getaddrinfo("huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)
        except socket.gaierror as lifted:
            assert "name resolution blocked" not in str(lifted), (
                "allow_outbound() did not lift the guard: the lookup inside the block "
                "was still refused by the guard rather than by the resolver"
            )

    with pytest.raises(socket.gaierror, match = "name resolution blocked"):
        socket.getaddrinfo("huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)


def test_the_proxy_bypass_covers_the_local_server_too(monkeypatch, no_proxy_bypass_value):
    """A proxy swallows loopback requests as readily as remote ones.

    With ``HTTP_PROXY`` set and no loopback entry in ``NO_PROXY``, a request to the
    managed ``studio_server`` is sent to the proxy instead. The guard then refuses the
    proxy, correctly, and the server on 127.0.0.1 is unreachable through no fault of
    its own. Naming only the configured external servers left that case out.
    """
    monkeypatch.setenv("UNSLOTH_E2E_BASE_URL", "http://studio.example.internal:8000")
    bypass = no_proxy_bypass_value("corp.example, 10.0.0.1").split(",")

    assert "127.0.0.1" in bypass
    assert "localhost" in bypass
    assert "studio.example.internal" in bypass, "the configured server must still be bypassed"
    assert bypass[:2] == ["corp.example", "10.0.0.1"], "an existing NO_PROXY must survive"
    assert len(bypass) == len(set(bypass)), "entries must not be duplicated on re-entry"


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
