# SPDX-License-Identifier: AGPL-3.0-only

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import lan_access  # noqa: E402


def test_active_virtual_bridge_is_selectable(monkeypatch):
    import psutil

    entry = SimpleNamespace(
        family = socket.AF_INET,
        address = "192.168.100.1",
    )
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"virbr0": [entry]},
    )
    monkeypatch.setattr(
        psutil,
        "net_if_stats",
        lambda: {"virbr0": SimpleNamespace(isup = True)},
    )

    assert lan_access._selectable_interface_addresses() == ["192.168.100.1"]


def test_down_virtual_bridge_is_not_selectable(monkeypatch):
    import psutil

    entry = SimpleNamespace(
        family = socket.AF_INET,
        address = "192.168.100.1",
    )
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"virbr0": [entry]},
    )
    monkeypatch.setattr(
        psutil,
        "net_if_stats",
        lambda: {"virbr0": SimpleNamespace(isup = False)},
    )

    assert lan_access._selectable_interface_addresses() == []


def test_automatic_discovery_still_excludes_virtual_bridge(monkeypatch):
    import psutil

    entry = SimpleNamespace(
        family = socket.AF_INET,
        address = "192.168.100.1",
    )
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"virbr0": [entry]},
    )
    monkeypatch.setattr(
        psutil,
        "net_if_stats",
        lambda: {"virbr0": SimpleNamespace(isup = True)},
    )

    monkeypatch.setattr(
        lan_access,
        "_wsl_networking_mode",
        lambda: None,
    )

    class FakeSocket:
        def connect(self, address):
            raise OSError("No default route")

        def close(self):
            pass

    monkeypatch.setattr(
        lan_access.socket,
        "socket",
        lambda *args, **kwargs: FakeSocket(),
    )

    assert lan_access.detect_lan_addresses() == []


def test_selected_bridge_reaches_bind_attempt(monkeypatch):
    attempted = []

    monkeypatch.setattr(
        lan_access,
        "detect_lan_addresses",
        lambda: [],
    )
    monkeypatch.setattr(
        lan_access,
        "_selectable_interface_addresses",
        lambda: ["192.168.100.1"],
    )

    def fake_bind(address, port):
        attempted.append(address)
        raise OSError("mock bind failure")

    monkeypatch.setattr(lan_access, "_bind_listener", fake_bind)

    with pytest.raises(RuntimeError, match = "bind_failed"):
        lan_access.start_lan_listener(
            object(),
            object(),
            8888,
            (),
            ("192.168.100.1",),
        )

    assert attempted == ["192.168.100.1"]


def test_unavailable_selected_bridge_never_binds(monkeypatch):
    attempted = []

    monkeypatch.setattr(
        lan_access,
        "detect_lan_addresses",
        lambda: [],
    )
    monkeypatch.setattr(
        lan_access,
        "_selectable_interface_addresses",
        lambda: [],
    )

    def fake_bind(address, port):
        attempted.append(address)
        raise OSError("unexpected bind attempt")

    monkeypatch.setattr(lan_access, "_bind_listener", fake_bind)

    with pytest.raises(RuntimeError, match = "selected_address_unavailable"):
        lan_access.start_lan_listener(
            object(),
            object(),
            8888,
            (),
            ("192.168.100.1",),
        )

    assert attempted == []


@pytest.mark.parametrize("mode", ["nat", "unknown"])
def test_wsl_unreachable_interfaces_are_not_selectable(monkeypatch, mode):
    monkeypatch.setattr(
        lan_access,
        "_wsl_networking_mode",
        lambda: mode,
    )

    assert lan_access._selectable_interface_addresses() == []
