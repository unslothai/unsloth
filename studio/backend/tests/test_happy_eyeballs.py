# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A black-holed address family must cost one stagger, not one timeout per address.

Where the clock is the thing under test, the black hole is real: 100::/64 is the RFC
6666 discard prefix, so a connect() to it hangs exactly as it does behind a VPN that
carries no IPv6. Where the thing under test is which attempt opens, and when, the socket
is scripted instead, because a discard address is refused outright rather than silent on
a host with no IPv6 route at all.
"""

from __future__ import annotations

import ast
import errno
import selectors
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from utils import happy_eyeballs as he

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Imported rather than restated. The first version of this guard copied the list and
# dropped three of them -- including hub/workers/hf_download.py, the Hub downloader this
# whole change exists for -- while its docstring claimed to mirror that file.
from test_native_tls_entrypoints import _ENTRYPOINTS as _NATIVE_TLS_ENTRYPOINTS  # noqa: E402

# Activates inside the spawned child function rather than at import, so the native TLS
# guard (which checks module-level calls by AST) does not list it.
_EXTRA_ENTRYPOINTS = ("core/training/diffusion_training_service.py",)


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


def _resolver(
    port,
    *,
    aaaa = 8,
    with_a = True,
):
    def _getaddrinfo(
        host,
        _port,
        family = 0,
        type = 0,
        proto = 0,
        flags = 0,
    ):
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
        "100::1",
        "1.2.3.4",
        "100::2",
        "5.6.7.8",
        "100::3",
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


def test_the_whole_walk_costs_one_timeout_plus_staggers_not_one_timeout_each(monkeypatch):
    """With no A record to win, six black-holed AAAA must fail at one budget plus the
    staggers that opened them, not at six budgets."""
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(443, aaaa = 6, with_a = False))

    start = time.monotonic()
    with pytest.raises(OSError):
        he.happy_eyeballs_connection(("hub.invalid", 443), 3)
    elapsed = time.monotonic() - start

    # 3s + five staggers = 4.25s; 6s leaves slack for a loaded CI box and is still well
    # under the 18s the stdlib pays.
    assert (
        elapsed < 6.0
    ), f"took {elapsed:.1f}s for a 3s budget; the timeout is still applied per address"


def test_a_reachable_address_late_in_the_list_is_still_dialled(monkeypatch):
    """The stdlib hands every resolved address the caller's whole timeout, so a host
    whose only reachable address sits late in the list connects, however slowly. Sharing
    one budget of ``timeout`` across the race took that away: once ``timeout / delay``
    attempts had opened, the deadline had passed and the addresses behind them were never
    tried at all. The budget now carries one stagger per extra address, so every attempt
    still gets the whole timeout.

    One family, because interleaving is what saves a host whose families differ. A host
    with eight dead A records and one live one has no second family to cut in.

    Scripted rather than dialled: the case is about which attempts open and when.
    """
    monkeypatch.setenv(he._DELAY_ENV, "0.05")
    good = "127.0.0.1"
    dialled = []

    class _Stub:
        def __init__(self, family, *_args, **_kwargs):
            self.family = family
            self.peer = None

        def setblocking(self, _flag):
            pass

        def settimeout(self, _value):
            pass

        def connect_ex(self, sa):
            dialled.append(sa[0])
            self.peer = sa[0]
            # Every black hole stays in flight forever; the one live address answers.
            return 0 if sa[0] == good else errno.EINPROGRESS

        def close(self):
            pass

    class _NeverReady:
        """No black hole ever becomes writable, so only the clock moves the loop on."""

        def register(self, *_args, **_kwargs):
            pass

        def unregister(self, *_args, **_kwargs):
            pass

        def select(self, timeout):
            if timeout:
                time.sleep(timeout)
            return []

        def close(self):
            pass

    # 192.0.2.0/24 is TEST-NET-1. The live address is ninth, due 0.4s in, where a shared
    # 0.2s budget had already run out.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"192.0.2.{i + 1}", 443)) for i in range(8)
        ]
        + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (good, 443))],
    )
    monkeypatch.setattr(socket, "socket", _Stub)
    monkeypatch.setattr(selectors, "DefaultSelector", _NeverReady)

    sock = he.happy_eyeballs_connection(("hub.invalid", 443), 0.2)

    assert good in dialled, (
        "the only live address was never dialled; the stdlib would have reached it with "
        "a whole timeout of its own"
    )
    assert sock.peer == good, "the race returned a black hole instead of the winner"


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


def test_the_last_failure_is_raised_not_the_first(monkeypatch):
    """``create_connection(all_errors = False)`` raises the LAST error, and callers read
    which one it is: utils.utils.hf_tcp_reachable treats ECONNREFUSED as proof the
    endpoint answered, so handing it an earlier family's ENETUNREACH instead declares the
    Hub unreachable and switches the offline guard on."""
    scripted = {"100::1": errno.ENETUNREACH, "127.0.0.1": errno.ECONNREFUSED}

    class _Stub:
        def __init__(self, family, *_args, **_kwargs):
            self.family = family

        def setblocking(self, _flag):
            pass

        def connect_ex(self, sa):
            return scripted[sa[0]]

        def close(self):
            pass

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("100::1", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    monkeypatch.setattr(socket, "socket", _Stub)

    with pytest.raises(OSError) as excinfo:
        he.happy_eyeballs_connection(("hub.invalid", 443), 5)

    assert excinfo.value.errno == errno.ECONNREFUSED, (
        "raised the first family's failure instead of the last; hf_tcp_reachable reads "
        "that as an unreachable Hub"
    )


def test_a_deadline_reached_with_attempts_in_flight_raises_a_timeout(monkeypatch):
    """An address still in flight when the budget runs out is a timeout, not whatever an
    address that failed earlier happened to report.

    Scripted rather than dialled: whether a discard-prefix address hangs or is refused
    outright depends on whether the host has an IPv6 route at all.
    """

    class _Stub:
        def __init__(self, family, *_args, **_kwargs):
            self.family = family

        def setblocking(self, _flag):
            pass

        def connect_ex(self, sa):
            return errno.ECONNREFUSED if sa[0] == "127.0.0.1" else errno.EINPROGRESS

        def close(self):
            pass

    class _NeverReady:
        """Nothing ever becomes writable, so the in-flight attempt outlives the budget."""

        def register(self, *_args, **_kwargs):
            pass

        def unregister(self, *_args, **_kwargs):
            pass

        def select(self, timeout):
            if timeout:
                time.sleep(min(timeout, 0.05))
            return []

        def close(self):
            pass

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("100::1", 443, 0, 0)),
        ],
    )
    monkeypatch.setattr(socket, "socket", _Stub)
    monkeypatch.setattr(selectors, "DefaultSelector", _NeverReady)

    with pytest.raises(socket.timeout):
        he.happy_eyeballs_connection(("hub.invalid", 443), 0.3)


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
    """Injection does not survive a spawn, so a worker that activates native TLS without
    this one still pays one timeout per resolved address."""
    import pathlib

    backend = pathlib.Path(he.__file__).resolve().parent.parent
    for rel in tuple(_NATIVE_TLS_ENTRYPOINTS) + _EXTRA_ENTRYPOINTS:
        src = (backend / rel).read_text(encoding = "utf-8")
        assert "activate_native_tls()" in src, f"{rel} moved; update this guard"
        assert (
            "activate_happy_eyeballs()" in src
        ), f"{rel} activates native TLS but not happy eyeballs"
        # Parsed, not just grepped: an activation inside a function can be misindented
        # and still grep clean.
        ast.parse(src)
