# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A black-holed address family must cost one stagger, not one timeout per address.

100::/64 is the RFC 6666 discard prefix; on a host with no IPv6 route it is refused
outright, so ordering cases script the socket instead.
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

from test_native_tls_entrypoints import _ENTRYPOINTS as _NATIVE_TLS_ENTRYPOINTS  # noqa: E402

# Activate inside a spawned child function or inline script, so the native TLS guard does not list them.
_EXTRA_ENTRYPOINTS = (
    "core/training/diffusion_training_service.py",
    "utils/models/model_config.py",
)


DISCARD = [f"100::{i + 1}" for i in range(8)]


@pytest.fixture
def listener():
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

    # The stdlib pays 8 x 10s.
    assert elapsed < 3.0, f"took {elapsed:.1f}s; the AAAA records were walked in order"


def test_the_whole_walk_costs_one_timeout_plus_staggers_not_one_timeout_each(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(443, aaaa = 6, with_a = False))

    start = time.monotonic()
    with pytest.raises(OSError):
        he.happy_eyeballs_connection(("hub.invalid", 443), 3)
    elapsed = time.monotonic() - start

    # 3s + five staggers = 4.25s; the stdlib pays 18s.
    assert (
        elapsed < 6.0
    ), f"took {elapsed:.1f}s for a 3s budget; the timeout is still applied per address"


def test_a_reachable_address_late_in_the_list_is_still_dialled(monkeypatch):
    """Every attempt keeps the whole timeout, so a live address opened past
    ``timeout / delay`` staggers is still dialled, as the stdlib would."""
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
            return 0 if sa[0] == good else errno.EINPROGRESS

        def close(self):
            pass

    class _NeverReady:
        def register(self, *_args, **_kwargs):
            pass

        def unregister(self, *_args, **_kwargs):
            pass

        def select(self, timeout):
            if timeout:
                time.sleep(timeout + 0.03)  # a loaded runner oversleeps every stagger
            return []

        def close(self):
            pass

    # TEST-NET-1 black holes; the live address is ninth, due 0.4s in.
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


def test_a_single_address_resolves_once_and_keeps_the_callers_timeout(listener, monkeypatch):
    lookups = []

    def _one(*a, **k):
        lookups.append(a[0])
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", listener))]

    monkeypatch.setattr(socket, "getaddrinfo", _one)
    sock = he.happy_eyeballs_connection(("one.invalid", listener), 5)
    try:
        assert sock.gettimeout() == 5
    finally:
        sock.close()
    assert lookups == ["one.invalid"], "the host was resolved more than once"


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
        he.happy_eyeballs_connection(("refused.invalid", 1), 30)
    elapsed = time.monotonic() - start

    # Windows retries a reset SYN for ~2s; the stdlib pays that per address too.
    assert elapsed < 10.0, f"a refused connect waited {elapsed:.1f}s instead of failing"
    assert excinfo.value.errno in (errno.ECONNREFUSED, errno.EADDRNOTAVAIL)


def test_the_last_failure_is_raised_not_the_first(monkeypatch):
    """hf_tcp_reachable reads ECONNREFUSED as "the Hub answered"."""
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


def _mixed_list_connector(monkeypatch, order):
    """What the connector raises for "hole" (never answers) and "refused" in *order*."""
    monkeypatch.setenv(he._DELAY_ENV, "0.05")
    addr = {"hole": ("192.0.2.1", 443), "refused": ("127.0.0.1", 1)}

    class _Stub:
        def __init__(self, family, *_args, **_kwargs):
            self.family = family

        def setblocking(self, _flag):
            pass

        def connect_ex(self, sa):
            return errno.ECONNREFUSED if sa == addr["refused"] else errno.EINPROGRESS

        def close(self):
            pass

    class _NeverReady:
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

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", addr[k]) for k in order],
    )
    monkeypatch.setattr(socket, "socket", _Stub)
    monkeypatch.setattr(selectors, "DefaultSelector", _NeverReady)

    with pytest.raises(OSError) as excinfo:
        he.happy_eyeballs_connection(("hub.invalid", 443), 0.2)
    return excinfo.value


def test_a_black_hole_then_a_refused_address_raises_refused_as_the_stdlib_does(monkeypatch):
    exc = _mixed_list_connector(monkeypatch, ("hole", "refused"))
    assert isinstance(
        exc, ConnectionRefusedError
    ), f"raised {type(exc).__name__}; the stdlib raises the last address's refusal"


def test_a_refused_address_then_a_black_hole_raises_a_timeout_as_the_stdlib_does(monkeypatch):
    exc = _mixed_list_connector(monkeypatch, ("refused", "hole"))
    assert isinstance(
        exc, socket.timeout
    ), f"raised {type(exc).__name__}; the stdlib raises the last address's timeout"


def test_the_winning_socket_is_blocking_with_the_callers_timeout(listener, monkeypatch):
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
    import pathlib
    backend = pathlib.Path(he.__file__).resolve().parent.parent
    for rel in tuple(_NATIVE_TLS_ENTRYPOINTS) + _EXTRA_ENTRYPOINTS:
        src = (backend / rel).read_text(encoding = "utf-8")
        assert "activate_native_tls()" in src, f"{rel} moved; update this guard"
        assert (
            "activate_happy_eyeballs()" in src
        ), f"{rel} activates native TLS but not happy eyeballs"
        ast.parse(src)


def test_the_config_probe_child_activates_it(monkeypatch):
    import subprocess

    from utils.transformers_version import _PROBE_CONFIG_SCRIPT

    monkeypatch.delenv(he._ENV, raising = False)
    head = _PROBE_CONFIG_SCRIPT.partition("target_dir, model_name")[0]
    out = subprocess.run(
        [sys.executable, "-c", head + "import socket\nprint(socket.create_connection.__name__)\n"],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert out.stdout.strip() == "happy_eyeballs_connection", out.stderr


def test_a_fixed_source_port_walks_like_the_stdlib(monkeypatch):
    calls = []
    monkeypatch.setattr(he, "_walk", lambda *a, **k: calls.append(a) or "sock")
    monkeypatch.setattr(socket, "getaddrinfo", _resolver(443, aaaa = 2))

    assert he.happy_eyeballs_connection(("hub.invalid", 443), 5, ("0.0.0.0", 40000)) == "sock"
    assert calls, "a fixed source port was raced; the second bind would fail with EADDRINUSE"


def test_the_sequential_walk_raises_what_the_stdlib_raises(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 1))],
    )
    with pytest.raises(ConnectionRefusedError):
        he.happy_eyeballs_connection(("refused.invalid", 1), 5)
    if sys.version_info >= (3, 11):
        with pytest.raises(ExceptionGroup):  # noqa: F821
            he.happy_eyeballs_connection(("refused.invalid", 1), 5, all_errors = True)
