# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LAN announcements: well-formed and small, malformed ones ignored, expiry, size bound, loopback
multicast when the host allows it."""

from __future__ import annotations

import importlib
import json
import socket
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _discovery():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import studio

    if str(REPO / "studio") not in list(studio.__path__):
        studio.__path__.append(str(REPO / "studio"))
    return importlib.import_module("studio.cluster_link.discovery")


d = _discovery()
FP = "ab" * 32


def _ann(node_id = "n1", port = 50070):
    return d.encode_announcement(node_id, port, FP)


def _raw(**over):
    return json.dumps({"v": 1, "node_id": "n", "agent_port": 1, "fp": FP, **over}).encode()


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(cond, timeout = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not cond():
        time.sleep(0.05)
    return cond()


def test_an_announcement_is_small_and_round_trips():
    data = d.encode_announcement("node-1", 50070, FP)
    assert len(data) < 512
    assert d.decode_announcement(data) == {"node_id": "node-1", "agent_port": 50070, "fp": FP}
    assert set(json.loads(data)) == {"v", "node_id", "agent_port", "fp"}


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xff\xfe",
        b"[]",
        b"not json",
        _raw(v = 2),
        _raw(node_id = ""),
        # Control bytes in an id would reach terminals and logs (an OSC 52 clipboard write fits in 12).
        _raw(node_id = "\x1b]52;c;QUJD\x07"),
        _raw(node_id = "a" * 65),
        _raw(agent_port = 0),
        _raw(agent_port = True),
        _raw(agent_port = 70000),
        _raw(fp = "zz" * 32),
        _raw(fp = "ab"),
        b"x" * 600,
    ],
    ids = "empty not_utf8 not_object not_json future_version empty_id control_bytes_id long_id port_zero "
    "port_bool port_too_big fp_not_hex fp_short oversized".split(),
)
def test_malformed_announcements_are_ignored(payload):
    listener = d.Listener()
    assert d.decode_announcement(payload) is None
    assert listener.observe(payload, "192.168.1.5") is False
    assert listener.latest() == {}


def test_sightings_expire_and_are_bounded(monkeypatch):
    listener = d.Listener(interval = 1.0)
    now = 1000.0
    assert listener.observe(_ann(), "192.168.1.5", now = now)
    assert listener.latest(now = now + 2.0)["n1"].host == "192.168.1.5"
    assert listener.latest(now = now + 4.0) == {}
    monkeypatch.setattr(d, "MAX_SIGHTINGS", 3)
    for i in range(5):
        listener.observe(_ann(f"x{i}"), "10.0.0.1", now = now)
    assert len(listener.latest(now = now)) == 3
    assert listener.observe(_ann("late"), "10.0.0.2", now = now + 10)
    assert "late" in listener.latest(now = now + 10)


def test_a_newer_announcement_replaces_the_address():
    listener = d.Listener()
    listener.observe(_ann(), "192.168.1.5", now = 1.0)
    listener.observe(_ann(port = 50071), "192.168.1.9", now = 2.0)
    seen = listener.latest(now = 2.0)["n1"]
    assert (seen.host, seen.agent_port) == ("192.168.1.9", 50071)


def test_a_spoofed_address_does_not_hide_the_real_one():
    listener = d.Listener()
    listener.observe(_ann(), "192.168.1.5", now = 1.0)
    listener.observe(_ann(), "192.168.1.66", now = 2.0)
    assert [s.host for s in listener.candidates("n1", now = 2.0)] == ["192.168.1.66", "192.168.1.5"]


def test_hosts_per_node_are_bounded():
    listener = d.Listener()
    for i in range(10):
        listener.observe(_ann(), f"10.0.0.{i}", now = float(i))
    hosts = [s.host for s in listener.candidates("n1", now = 9.0)]
    assert len(hosts) == d.MAX_HOSTS_PER_NODE and hosts[0] == "10.0.0.9"


def test_the_accept_filter_keeps_a_flood_from_crowding_out_paired_nodes():
    listener = d.Listener(accept = lambda node_id, fp: node_id == "paired")
    for i in range(d.MAX_SIGHTINGS + 10):
        assert not listener.observe(_ann(f"x{i}"), "10.0.0.1", now = 1.0)
    assert listener.observe(_ann("paired"), "10.0.0.2", now = 1.0)
    assert list(listener.latest(now = 1.0)) == ["paired"]


def test_announcer_reaches_a_listener_over_multicast(monkeypatch):
    port = _free_udp_port()
    monkeypatch.setattr(d, "DISCOVERY_PORT", port)
    listener = d.Listener(interval = 0.2, port = port).start()
    if listener.error:
        pytest.skip(f"no multicast here: {listener.error}")
    announcer = d.Announcer("node-mc", 50070, FP, interval = 0.2)
    try:
        if not announcer.send_once():
            pytest.skip("multicast send not possible here")
        if not _wait_for(lambda: announcer.send_once() and "node-mc" in listener.latest(), 3.0):
            pytest.skip("multicast loopback not delivered on this host")
        assert listener.latest()["node-mc"].agent_port == 50070
    finally:
        listener.stop()
        announcer.stop()


def test_start_and_stop_are_clean():
    announcer = d.Announcer("n", 50070, FP, interval = 0.05).start()
    listener = d.Listener(interval = 0.05, port = _free_udp_port()).start()
    time.sleep(0.15)
    announcer.stop()
    listener.stop()
    assert announcer._thread is None and listener._thread is None


def test_the_listener_reopens_after_a_socket_error(monkeypatch):
    # An interface reset makes recvfrom fail: the listener must come back, not stay deaf until restart.
    monkeypatch.setattr(d, "REOPEN_S", 0.1)
    port = _free_udp_port()
    listener = d.Listener(port = port).start()
    try:
        first = listener._sock
        first.close()  # the next recvfrom raises OSError
        assert _wait_for(lambda: listener._sock is not None and listener._sock is not first)
        assert listener._thread is not None and listener._thread.is_alive()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
            tx.sendto(_ann("node-1"), ("127.0.0.1", port))
        assert _wait_for(lambda: "node-1" in listener.latest())
    finally:
        listener.stop()


class _FakeSock:
    """Records multicast options and sends; stands in for socket.socket in the per-interface tests."""

    made = []

    def __init__(self, *args, **kwargs):
        self.opts, self.sent, self.closed = [], [], False
        _FakeSock.made.append(self)

    def setsockopt(self, level, name, value):
        self.opts.append((name, value))

    def sendto(self, data, dest):
        self.sent.append(dest)

    bind = settimeout = lambda self, arg: None

    def close(self):
        self.closed = True


@pytest.fixture
def ifaces(monkeypatch):
    """Fake socket module; the returned list is what `lan_ipv4_addresses` reports (mutable)."""
    fake = types.SimpleNamespace(
        **{k: getattr(socket, k) for k in dir(socket) if not k.startswith("__")}
    )
    fake.socket, fake.has_ipv6, _FakeSock.made = _FakeSock, False, []
    monkeypatch.setattr(d, "socket", fake)
    found = []
    monkeypatch.setattr(d, "lan_ipv4_addresses", lambda: list(found))
    monkeypatch.setattr(d, "REJOIN_S", 0.0)
    return found


def _multicast_ifs(sock):
    return [socket.inet_ntoa(v) for n, v in sock.opts if n == socket.IP_MULTICAST_IF]


def _joins(sock):
    return sorted(socket.inet_ntoa(v[4:8]) for n, v in sock.opts if n == socket.IP_ADD_MEMBERSHIP)


@pytest.mark.parametrize(
    "lan", [["192.168.1.5", "10.8.0.2"], []], ids = ["every_lan_interface", "default_route"]
)
def test_beacons_and_group_joins_cover_every_lan_interface(ifaces, lan):
    # A host with a Docker bridge or a VPN: the default route alone can be the wrong network.
    ifaces.extend(lan)
    assert d.Announcer("n", 50070, FP).send_once()
    v4 = [s for s in _FakeSock.made if s.sent]
    assert len(v4) == max(1, len(lan))
    assert sorted(i for s in v4 for i in _multicast_ifs(s)) == sorted(lan)
    assert all(s.sent == [(d.DISCOVERY_GROUP_V4, d.DISCOVERY_PORT)] for s in v4)
    assert all(s.closed for s in _FakeSock.made)
    assert _joins(d.Listener(port = 1)._open()) == (sorted(lan) or ["0.0.0.0"])


def test_the_listener_rejoins_when_interfaces_change(ifaces):
    ifaces.append("192.168.1.5")
    listener = d.Listener(port = 1)
    listener._open()
    assert not listener._interfaces_changed()
    ifaces.append("10.8.0.2")
    assert listener._interfaces_changed()


def test_the_listener_retries_a_membership_that_failed(ifaces, monkeypatch):
    # One interface refuses the join, another takes it; with the list unchanged only a retry recovers.
    ifaces.extend(["192.168.1.5", "10.8.0.2"])
    refuse, plain = {"10.8.0.2"}, _FakeSock.setsockopt

    def setsockopt(self, level, name, value):
        if name == socket.IP_ADD_MEMBERSHIP and socket.inet_ntoa(value[4:8]) in refuse:
            raise OSError("No such device")
        plain(self, level, name, value)

    monkeypatch.setattr(_FakeSock, "setsockopt", setsockopt)
    listener = d.Listener(port = 1)
    sock = listener._open()
    assert _joins(sock) == ["192.168.1.5"]
    assert not listener._interfaces_changed(sock)  # still refused: kept for the next pass
    refuse.clear()
    assert not listener._interfaces_changed(sock)
    assert _joins(sock) == ["10.8.0.2", "192.168.1.5"]
    assert not listener._interfaces_changed(sock)
    assert _joins(sock) == ["10.8.0.2", "192.168.1.5"]  # joined once, not again


def test_lan_addresses_skip_loopback_public_and_down_interfaces(monkeypatch):
    def nic(addr, family = socket.AF_INET):
        return types.SimpleNamespace(family = family, address = addr)

    addrs = {
        "lo": [nic("127.0.0.1")],
        "eth0": [nic("192.168.1.5"), nic("fe80::1", socket.AF_INET6)],
        "wan": [nic("8.8.8.8")],
        "down0": [nic("10.0.0.9")],
        "docker0": [nic("172.17.0.1")],
    }
    stats = {name: types.SimpleNamespace(isup = name != "down0") for name in addrs}
    fake = types.SimpleNamespace(net_if_addrs = lambda: addrs, net_if_stats = lambda: stats)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert d.lan_ipv4_addresses() == ["192.168.1.5", "172.17.0.1"]
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert d.lan_ipv4_addresses() == []
