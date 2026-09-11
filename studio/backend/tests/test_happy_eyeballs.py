# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A black-holed address family must cost one stagger, not one timeout per address.

The black hole is real, not mocked: 100::/64 is the RFC 6666 discard prefix, so a
connect() to it hangs exactly as it does behind a VPN that carries no IPv6.
"""

from __future__ import annotations

import ast
import errno
import socket
import threading
import time

import pytest

from utils import happy_eyeballs as he


DISCARD = [f"100::{i + 1}" for i in range(8)]


@pytest.fixture
def listener():
    """A real acceptor, so a winning connect is a genuine TCP handshake."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(32)
    accepted = []

    def _accept():
        while True:
            try:
                accepted.append(srv.accept()[0])
            except OSError:
                return

    threading.Thread(target = _accept, daemon = True).start()
    yield srv.getsockname()[1]
    srv.close()
    for sock in accepted:
        sock.close()


def _resolver(port, *, aaaa = 8, with_a = True):
    def _getaddrinfo(host, _port, family = 0, type = 0, proto = 0, flags = 0):
        infos = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (addr, port, 0, 0))
            for addr in DISCARD[:aaaa]
        ]
        if with_a:
            infos.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)))
        return infos

    return _getaddrinfo


def test_families_are_interleaved_not_walked_in_resolver_order():
    infos = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("100::1", 443, 0, 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("100::2", 443, 0, 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("100::3", 443, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("5.6.7.8", 443)),
    ]
    assert [i[4][0] for i in he._interleave(infos)] == [
        "100::1", "1.2.3.4", "100::2", "5.6.7.8", "100::3",
    ], "the second family must get a turn before the first family is exhausted"


def test_eight_black_holed_aaaa_cost_one_stagger(listener, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(listener))

    start = time.monotonic()
    sock = he.happy_eyeballs_connection(("hub.invalid", listener), 10)
    elapsed = time.monotonic() - start
    try:
        assert sock.getpeername()[0] == "127.0.0.1"
    finally:
        sock.close()

    # The stdlib pays 8 x 10s here; 3s leaves slack for a loaded CI box.
    assert elapsed < 3.0, f"took {elapsed:.1f}s; the AAAA records were walked in order"


def test_the_timeout_is_the_whole_connect_not_each_address(monkeypatch):
    """With no A record to win, six black-holed AAAA must still fail at the budget, not
    at six times it."""
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(443, aaaa = 6, with_a = False))

    start = time.monotonic()
    with pytest.raises(OSError):
        he.happy_eyeballs_connection(("hub.invalid", 443), 3)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, (
        f"took {elapsed:.1f}s for a 3s budget; the timeout is still applied per address"
    )


def test_a_single_address_keeps_stdlib_semantics(listener, monkeypatch):
    calls = []
    real = he._original_create_connection

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(he, "_original_create_connection", _spy)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", listener))],
    )

    sock = he.happy_eyeballs_connection(("one.invalid", listener), 5)
    sock.close()
    assert calls, "a single-address host did not delegate to the stdlib"


def test_a_refused_port_still_raises_immediately(monkeypatch):
    """Racing must not turn a fast, definite failure into a wait for the deadline."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 1)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 1, 0, 0)),
        ],
    )

    start = time.monotonic()
    with pytest.raises(OSError) as excinfo:
        he.happy_eyeballs_connection(("refused.invalid", 1), 5)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"a refused connect waited {elapsed:.1f}s instead of failing"
    assert excinfo.value.errno in (errno.ECONNREFUSED, errno.EADDRNOTAVAIL)


def test_the_winning_socket_is_blocking_with_the_callers_timeout(listener, monkeypatch):
    """Attempts run non-blocking; what the caller gets back must not."""
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(listener, aaaa = 2))

    sock = he.happy_eyeballs_connection(("hub.invalid", listener), 7)
    try:
        assert sock.gettimeout() == 7
    finally:
        sock.close()


def test_the_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv(he._ENV, "0")
    assert he.happy_eyeballs_enabled() is False
    monkeypatch.setenv(he._ENV, "1")
    assert he.happy_eyeballs_enabled() is True
    monkeypatch.delenv(he._ENV, raising = False)
    assert he.happy_eyeballs_enabled() is True, "it must be on by default"


def test_activation_installs_the_connector_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(he, "_activated", False)
    monkeypatch.setattr(socket, "create_connection", he._original_create_connection)

    assert he.activate_happy_eyeballs() is True
    assert socket.create_connection is he.happy_eyeballs_connection
    assert he.activate_happy_eyeballs() is True


def test_every_network_entry_point_activates_it():
    """Injection does not survive a spawn. Mirrors test_native_tls_entrypoints.py."""
    import pathlib

    backend = pathlib.Path(he.__file__).resolve().parent.parent
    for rel in (
        "main.py",
        "core/data_recipe/jobs/worker.py",
        "core/training/worker.py",
        "core/training/diffusion_training_service.py",
        "core/inference/worker.py",
    ):
        src = (backend / rel).read_text(encoding = "utf-8")
        assert "activate_native_tls()" in src, f"{rel} moved; update this guard"
        assert "activate_happy_eyeballs()" in src, (
            f"{rel} activates native TLS but not happy eyeballs"
        )
        # Parsed, not just grepped: an activation inside a function can be misindented
        # and still grep clean.
        ast.parse(src)
