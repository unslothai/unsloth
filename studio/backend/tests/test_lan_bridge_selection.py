# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A virtual switch address (libvirt virbr0) can be chosen explicitly but is never bound by Automatic (#17017)."""

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import lan_access  # noqa: E402

_BRIDGE = "192.168.122.1"
_NIC = "192.168.1.24"


@pytest.fixture
def interfaces(monkeypatch):
    psutil = pytest.importorskip("psutil")
    up = {"eth0": True, "virbr0": True, "virbr1": False}
    addrs = {
        "eth0": [SimpleNamespace(family = socket.AF_INET, address = _NIC)],
        "virbr0": [SimpleNamespace(family = socket.AF_INET, address = _BRIDGE)],
        "virbr1": [SimpleNamespace(family = socket.AF_INET, address = "192.168.100.1")],
    }
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: addrs)
    monkeypatch.setattr(
        psutil, "net_if_stats", lambda: {n: SimpleNamespace(isup = v) for n, v in up.items()}
    )
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: None)
    return up


@pytest.fixture
def binds(monkeypatch):
    attempted = []

    def _bind(address, port):
        attempted.append(address)
        raise OSError("not binding in a unit test")

    monkeypatch.setattr(lan_access, "_bind_listener", _bind)
    return attempted


def _start(selected):
    with pytest.raises(RuntimeError) as exc:
        lan_access.start_lan_listener(object(), object(), 8888, (), selected)
    return str(exc.value)


def test_bridge_is_host_only_and_not_detected(interfaces):
    assert _BRIDGE not in lan_access.detect_lan_addresses()
    assert lan_access.detect_host_only_addresses() == [_BRIDGE]


def test_a_chosen_bridge_address_is_bound(interfaces, binds):
    assert _start((_BRIDGE,)) == "bind_failed"
    assert binds == [_BRIDGE]


def test_automatic_never_binds_the_bridge(interfaces, binds):
    _start(None)
    assert _NIC in binds and _BRIDGE not in binds


def test_a_saved_bridge_without_a_guest_yet_still_binds(interfaces, binds):
    assert "192.168.100.1" not in lan_access.detect_host_only_addresses()
    assert _start(("192.168.100.1",)) == "bind_failed"
    assert binds == ["192.168.100.1"]


def test_a_chosen_address_on_no_interface_is_unavailable(interfaces, binds):
    assert _start(("192.168.200.1",)) == "selected_address_unavailable"
    assert binds == []


def test_wsl_nat_offers_no_bridge(interfaces, monkeypatch):
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: "nat")
    assert lan_access.detect_host_only_addresses() == []


def test_no_psutil_offers_no_bridge(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: None)
    assert lan_access.detect_host_only_addresses() == []
