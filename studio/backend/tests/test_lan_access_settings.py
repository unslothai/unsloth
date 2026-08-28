# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import asyncio
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

# module scope, not the fixture: postponed annotations resolve against module globals
from fastapi import FastAPI, HTTPException, Request

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import lan_access  # noqa: E402
import routes.settings as routes  # noqa: E402
import utils.host_policy as host_policy  # noqa: E402
import utils.lan_access_settings as lan_settings  # noqa: E402
from storage import studio_db  # noqa: E402


def _app(**over):
    state = SimpleNamespace(
        lan_access_port = 8888,
        lan_access_ready = True,
        lan_access_is_colab = False,
        lan_access_launch_managed = False,
        lan_access_wildcard_bind = False,
        lan_access_wildcard_ip_versions = (),
        lan_access_secure_launch = False,
        lan_access_frontend_served = True,
        lan_access_loop = None,
        server_url = "http://192.168.1.24:8888",
    )
    for key, value in over.items():
        setattr(state, key, value)
    return SimpleNamespace(state = state)


@pytest.fixture(autouse = True)
def stored_settings(monkeypatch):
    """Keep every test off the real listener state, password store and database."""
    stored: dict = {}
    monkeypatch.setattr(lan_settings, "_admin_password_ready", lambda: True)
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback: stored.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", stored.update)
    yield stored
    lan_access.stop_lan_listener()
    lan_access.clear_lan_listener_error()
    # drain watchers are daemon threads that outlive their test, so the shared
    # counter has to be reset or a later decrement lands on the next test's state
    lan_access._pending_drains = 0
    host_policy._reset_loopback_default_state()


# ── persisted preference ──


def test_auto_start_persistence_is_strict_and_fail_closed(monkeypatch, stored_settings):
    assert lan_settings.get_lan_access_auto_start() is False
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = "yes"
    assert lan_settings.get_lan_access_auto_start() is False
    assert lan_settings.set_lan_access_auto_start(True) is True
    assert lan_settings.get_lan_access_auto_start() is True
    with pytest.raises(ValueError):
        routes.LanAccessAutoStartPayload(enabled = "true")
    with pytest.raises(ValueError):
        lan_settings.set_lan_access_auto_start("true")
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda *args: (_ for _ in ()).throw(OSError())
    )
    assert lan_settings.get_lan_access_auto_start() is False


# ── launch policy ──


@pytest.mark.parametrize(
    "bind_host,launch_managed,wildcard,ip_versions",
    [
        ("127.0.0.1", False, False, ()),
        ("localhost", False, False, ()),
        ("::1", False, False, ()),
        ("0.0.0.0", True, True, (4,)),
        ("::", True, True, (6,)),
        ("::0", True, True, (6,)),
        ("0:0:0:0:0:0:0:0", True, True, (6,)),
        ("0", True, True, (4,)),
        ("::ffff:0.0.0.0", True, True, (4,)),
        ("192.168.1.24", True, False, ()),
    ],
)
def test_configure_reads_launch_ownership_from_the_bind_host(
    bind_host, launch_managed, wildcard, ip_versions
):
    state = SimpleNamespace()
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = bind_host,
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    assert state.lan_access_launch_managed is launch_managed
    assert state.lan_access_wildcard_bind is wildcard
    assert state.lan_access_wildcard_ip_versions == ip_versions
    assert state.lan_access_bind_host == bind_host
    assert state.lan_access_ready is False


def test_an_ephemeral_hostname_launch_keeps_its_loopback_policy(monkeypatch):
    monkeypatch.setattr(
        lan_settings.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ],
    )
    state = SimpleNamespace()

    lan_settings.configure_lan_access(
        state,
        port = 0,
        bind_host = "loopback.test",
        secure = False,
        is_colab = False,
        frontend_served = True,
    )

    assert state.lan_access_launch_addresses == ("127.0.0.1",)
    assert state.lan_access_launch_managed is False


def _transport_request(
    *,
    server = ("192.168.1.24", 8888),
    client = ("192.168.1.90", 54321),
    state = None,
    headers = None,
):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "root_path": "",
            "query_string": b"",
            "headers": [
                (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
            ],
            "server": server,
            "client": client,
            "app": SimpleNamespace(state = state or SimpleNamespace()),
        }
    )


def _listener(
    address = "192.168.1.24",
    *,
    running = True,
    port = 8888,
):
    return {"running": running, "port": port, "addresses": [address]}


@pytest.mark.parametrize(
    "server,client,listener,expected",
    [
        (("192.168.1.24", 8888), ("10.0.0.90", 54321), _listener(), True),
        (("fd00::24", 8888), ("fe80::90", 54321), _listener("fd00::24"), True),
        (("::ffff:192.168.1.24", 8888), ("::ffff:192.168.1.90", 54321), _listener(), True),
        (("192.168.1.24", 8889), ("192.168.1.90", 54321), _listener(), False),
        (("192.168.1.25", 8888), ("192.168.1.90", 54321), _listener(), False),
        (("192.168.1.24", 8888), ("192.168.1.90", 54321), _listener(running = False), False),
        (("64.227.100.5", 8888), ("192.168.1.90", 54321), _listener("64.227.100.5"), False),
        (("192.168.1.24", 8888), ("8.8.8.8", 54321), _listener(), False),
        (("192.168.1.24", 8888), ("127.0.0.1", 54321), _listener(), False),
        (("127.0.0.1", 8888), ("192.168.1.90", 54321), _listener("127.0.0.1"), False),
        (("192.168.1.24", 8888), None, None, False),
    ],
)
def test_private_lan_live_listener_matrix(monkeypatch, server, client, listener, expected):
    monkeypatch.setattr(lan_access, "lan_listener_status", lambda: listener)
    assert (
        lan_settings.request_on_lan_access(_transport_request(server = server, client = client))
        is expected
    )


def test_private_lan_ignores_forwarding_headers(monkeypatch):
    monkeypatch.setattr(lan_access, "lan_listener_status", lambda: _listener())
    headers = dict.fromkeys(("host", "forwarded", "x-forwarded-for", "origin"), "192.168.1.90")
    assert not lan_settings.request_on_lan_access(
        _transport_request(
            server = ("127.0.0.1", 8888),
            client = ("127.0.0.1", 54321),
            headers = headers,
        )
    )


@pytest.mark.parametrize(
    "bind_host,resolved,server,secure,is_colab,expected",
    [
        ("192.168.1.24", None, "192.168.1.24", False, False, True),
        ("0.0.0.0", None, "192.168.1.24", False, False, True),
        ("::", None, "fd00::24", False, False, True),
        ("::0", None, "fd00::24", False, False, True),
        ("0", None, "192.168.1.24", False, False, True),
        ("::ffff:0.0.0.0", None, "192.168.1.24", False, False, True),
        ("studio.lan", "192.168.1.24", "192.168.1.24", False, False, True),
        ("studio.lan", "127.0.0.1", "127.0.0.1", False, False, False),
        ("studio.lan", "64.227.100.5", "64.227.100.5", False, False, False),
        ("studio.lan", "error", "192.168.1.24", False, False, False),
        ("192.168.1.24", None, "192.168.1.24", True, False, False),
        ("192.168.1.24", None, "192.168.1.24", False, True, False),
    ],
)
def test_private_lan_launch_managed_matrix(
    monkeypatch, bind_host, resolved, server, secure, is_colab, expected
):
    def resolve(_host, port, *_args, **_kwargs):
        if resolved == "error":
            raise OSError("dns unavailable")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved, port))]

    if resolved is not None:
        monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": False, "port": None, "addresses": []},
    )
    state = SimpleNamespace()
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = bind_host,
        secure = secure,
        is_colab = is_colab,
        frontend_served = True,
    )
    assert (
        lan_settings.request_on_lan_access(
            _transport_request(
                server = (server, 8888),
                client = ("192.168.1.90", 54321),
                state = state,
            )
        )
        is expected
    )


# ── status ──


def test_a_loopback_launch_offers_a_startable_off_state():
    status = lan_settings.lan_access_status(_app())
    assert status["state"] == "off"
    assert status["block_reason"] is None
    assert status["can_start"] is True and status["can_stop"] is False
    assert status["urls"] == []
    assert status["keyless_lan_eligible"] is False


def test_lan_status_uses_one_fail_closed_keyless_state(monkeypatch):
    import utils.keyless_api_access as keyless

    monkeypatch.setattr(keyless, "get_keyless_api_access_settings", lambda: ("inference", True))
    status = lan_settings.lan_access_status(_app())
    assert (status["keyless_scope"], status["keyless_tools"]) == ("inference", True)

    def unreadable():
        raise OSError("settings unavailable")

    monkeypatch.setattr(keyless, "get_keyless_api_access_settings", unreadable)
    status = lan_settings.lan_access_status(_app())
    assert (status["keyless_scope"], status["keyless_tools"]) == ("off", False)


@pytest.mark.parametrize(
    "state_over,expected_block",
    [
        ({"lan_access_ready": False}, "server_starting"),
        ({"lan_access_is_colab": True}, "colab"),
        ({"lan_access_launch_managed": True}, "launch_managed"),
        ({"lan_access_secure_launch": True}, "secure_launch"),
    ],
)
def test_every_block_reason_closes_both_controls(state_over, expected_block):
    status = lan_settings.lan_access_status(_app(**state_over))
    assert status["block_reason"] == expected_block
    assert status["can_start"] is False and status["can_stop"] is False


def test_a_pending_admin_password_blocks_exposing_the_server(monkeypatch):
    monkeypatch.setattr(lan_settings, "_admin_password_ready", lambda: False)
    status = lan_settings.lan_access_status(_app())
    assert status["block_reason"] == "admin_password_change_required"
    assert status["can_start"] is False


@pytest.mark.parametrize(
    "bind_host,wildcard,expected_urls",
    [
        ("0.0.0.0", True, ["http://10.1.1.144:8888"]),
        ("::", True, ["http://[fd00::144]:8888"]),
        ("::0", True, ["http://[fd00::144]:8888"]),
        ("0:0:0:0:0:0:0:0", True, ["http://[fd00::144]:8888"]),
        ("0", True, ["http://10.1.1.144:8888"]),
        ("::ffff:0.0.0.0", True, ["http://10.1.1.144:8888"]),
        ("10.1.1.144", False, ["http://203.0.113.9:8888"]),
    ],
)
def test_status_carries_the_bind_host_so_a_block_can_name_it(
    monkeypatch, bind_host, wildcard, expected_urls
):
    """The launch-managed block covers a wildcard and a single-address bind
    alike, so the address itself has to reach the settings UI."""
    monkeypatch.setattr(
        lan_access,
        "detect_lan_addresses",
        lambda ip_version = 4: ["fd00::144"] if ip_version == 6 else ["10.1.1.144"],
    )
    state = SimpleNamespace(server_url = "http://203.0.113.9:8888")
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = bind_host,
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    state.lan_access_ready = True
    status = lan_settings.lan_access_status(SimpleNamespace(state = state))
    assert status["bind_host"] == bind_host
    assert status["wildcard_bind"] is wildcard
    assert status["urls"] == expected_urls
    assert status["block_reason"] == "launch_managed"


def test_a_dual_stack_wildcard_launch_reports_both_address_families(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 8888)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 8888, 0, 0)),
        ],
    )
    detected_versions = []

    def _detect(ip_version = 4):
        detected_versions.append(ip_version)
        return ["10.1.1.144"] if ip_version == 4 else ["fd00::144"]

    monkeypatch.setattr(lan_access, "detect_lan_addresses", _detect)
    state = SimpleNamespace(server_url = "http://203.0.113.9:8888")
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = "dual-wildcard.test",
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    state.lan_access_ready = True

    status = lan_settings.lan_access_status(SimpleNamespace(state = state))
    assert state.lan_access_wildcard_ip_versions == (4, 6)
    assert status["urls"] == ["http://10.1.1.144:8888", "http://[fd00::144]:8888"]
    assert detected_versions == [4, 6]


def test_the_response_model_carries_the_bind_host_to_the_client():
    """The status dict reaches the client through LanAccessResponse, which drops
    any field it does not declare."""
    state = SimpleNamespace(server_url = "http://10.1.1.144:8888")
    lan_settings.configure_lan_access(
        state,
        port = 8888,
        bind_host = "10.1.1.144",
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    state.lan_access_ready = True
    request = SimpleNamespace(app = SimpleNamespace(state = state))
    served = routes._lan_access_response(request).model_dump()
    assert served["bind_host"] == "10.1.1.144"
    assert served["wildcard_bind"] is False


def test_a_specific_host_launch_reports_the_address_it_was_given():
    status = lan_settings.lan_access_status(_app(lan_access_launch_managed = True))
    assert status["state"] == "online" and status["managed_by"] == "launch"
    assert status["urls"] == ["http://192.168.1.24:8888"]
    assert status["keyless_lan_eligible"] is True
    assert status["can_stop"] is False


def test_a_wildcard_launch_refreshes_lan_addresses_instead_of_showing_the_public_address(
    monkeypatch,
):
    """The status refreshes every LAN address instead of relying on server_url's
    single direct base, and never leaks an obsolete public-address value."""
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda _ip_version = 4: ["192.168.1.24"])
    state = _app(
        lan_access_launch_managed = True,
        lan_access_wildcard_bind = True,
        server_url = "http://203.0.113.9:8888",
    )
    status = lan_settings.lan_access_status(state)
    assert status["urls"] == ["http://192.168.1.24:8888"]
    assert status["public_urls"] == []

    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda _ip_version = 4: ["10.0.0.7"])
    assert lan_settings.lan_access_status(state)["urls"] == ["http://10.0.0.7:8888"]


def test_listener_urls_bracket_ipv6_literals():
    assert lan_settings._listener_urls(["192.168.1.24", "fd00::24"], 8888) == [
        "http://192.168.1.24:8888",
        "http://[fd00::24]:8888",
    ]


def test_a_publicly_routable_bind_is_flagged_so_the_ui_can_warn(monkeypatch):
    """A VPS carries its public IPv4 on the NIC, so "LAN access" can reach the
    internet. Binding it is the operator's call; hiding it is not."""
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {
            "running": True,
            "addresses": ["192.168.1.24", "64.227.100.5"],
            "port": 8888,
            "error": None,
        },
    )
    status = lan_settings.lan_access_status(_app())
    assert status["urls"] == ["http://192.168.1.24:8888", "http://64.227.100.5:8888"]
    assert status["public_urls"] == ["http://64.227.100.5:8888"]
    assert status["keyless_lan_eligible"] is True


@pytest.mark.parametrize(
    "address,public",
    [
        ("192.168.1.24", False),
        ("10.0.0.7", False),
        ("172.16.3.9", False),
        # CGNAT, which is where a Tailscale address lands
        ("100.64.1.1", False),
        ("64.227.100.5", True),
        ("8.8.8.8", True),
        ("fd00::24", False),
        ("2001:4860:4860::8844", True),
        ("not-an-ip", False),
    ],
)
def test_public_address_classification(address, public):
    assert lan_access.is_public_address(address) is public


def test_a_private_bind_carries_no_public_warning(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "addresses": ["192.168.1.24"], "port": 8888, "error": None},
    )
    status = lan_settings.lan_access_status(_app())
    assert status["public_urls"] == []
    assert status["keyless_lan_eligible"] is True


def test_a_cgnat_bind_is_not_reported_as_keyless_lan(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "addresses": ["100.64.0.10"], "port": 8888, "error": None},
    )
    status = lan_settings.lan_access_status(_app())
    assert status["state"] == "online"
    assert status["public_urls"] == []
    assert status["keyless_lan_eligible"] is False


def test_api_only_launches_advertise_that_the_web_ui_is_not_served():
    assert lan_settings.lan_access_status(_app())["serves_web_ui"] is True
    served = lan_settings.lan_access_status(_app(lan_access_frontend_served = False))
    assert served["serves_web_ui"] is False


def test_a_failed_start_is_reported_as_an_error_state(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": False, "addresses": [], "port": None, "error": "no_lan_address"},
    )
    status = lan_settings.lan_access_status(_app())
    assert status["state"] == "error" and status["error"] == "no_lan_address"
    # the failure is retryable: nothing about it blocks a second attempt
    assert status["can_start"] is True


# ── address detection ──


def test_detection_drops_addresses_no_other_device_can_open(monkeypatch):
    monkeypatch.setattr(
        lan_access,
        "_interface_addresses",
        lambda _ip_version = 4: [
            "127.0.0.1",
            "169.254.10.1",
            "224.0.0.1",
            "0.0.0.0",
            "not-an-ip",
            "10.0.0.7",
            "10.0.0.7",
        ],
    )
    addresses = lan_access.detect_lan_addresses()
    assert "10.0.0.7" in addresses
    assert addresses.count("10.0.0.7") == 1
    for rejected in ("127.0.0.1", "169.254.10.1", "224.0.0.1", "0.0.0.0", "not-an-ip"):
        assert rejected not in addresses


@pytest.mark.allow_network
def test_the_default_route_address_leads_so_it_becomes_the_shown_url(monkeypatch):
    routed = _require_lan_address()
    monkeypatch.setattr(lan_access, "_interface_addresses", lambda _ip_version = 4: ["203.0.113.9"])
    assert lan_access.detect_lan_addresses() == [routed, "203.0.113.9"]


def test_detection_survives_a_host_that_resolves_to_nothing(monkeypatch):
    """The hostname fallback only runs without psutil, and a Linux host mapping its
    name to 127.0.1.1 is exactly why it cannot be the source of truth."""

    def _boom(*_args, **_kwargs):
        raise OSError("no such host")

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(lan_access.socket, "getaddrinfo", _boom)
    # under the suite's outbound-network guard the UDP probe is refused too, so
    # a failed lookup must contribute nothing rather than raise
    assert lan_access.detect_lan_addresses() == []


@pytest.mark.allow_network
def test_detection_enumerates_interfaces_rather_than_the_default_route(monkeypatch):
    """An isolated LAN has no route to 8.8.8.8 and a multihomed host has more than
    one interface, so neither the probe nor the hostname finds every address."""
    routed = _require_lan_address()
    assert routed in lan_access._interface_addresses()

    # with the probe unavailable, enumeration alone still finds the address
    def _no_route(*_args, **_kwargs):
        raise OSError("network is unreachable")

    monkeypatch.setattr(lan_access.socket.socket, "connect", _no_route)
    assert routed in lan_access.detect_lan_addresses()


def test_interface_enumeration_skips_adapters_that_are_down(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        net_if_stats = lambda: {
            "en0": types.SimpleNamespace(isup = True),
            "en1": types.SimpleNamespace(isup = False),
        },
        net_if_addrs = lambda: {
            "en0": [types.SimpleNamespace(family = socket.AF_INET, address = "10.0.0.7")],
            "en1": [types.SimpleNamespace(family = socket.AF_INET, address = "10.9.9.9")],
            "en2": [types.SimpleNamespace(family = socket.AF_INET6, address = "fe80::1")],
        },
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert lan_access._interface_addresses() == ["10.0.0.7"]
    assert lan_access._interface_addresses(6) == ["fe80::1"]


def test_ipv6_detection_keeps_only_reachable_unscoped_addresses(monkeypatch):
    class _NoRouteSocket:
        def connect(self, _address):
            raise OSError("network is unreachable")

        def close(self):
            pass

    monkeypatch.setattr(lan_access.socket, "socket", lambda *_args, **_kwargs: _NoRouteSocket())
    monkeypatch.setattr(
        lan_access,
        "_interface_addresses",
        lambda _ip_version = 4: [
            "::1",
            "fe80::1%en0",
            "ff02::1",
            "::",
            "fd00::24",
            "2001:4860:4860::8844",
            "fd00::24",
        ],
    )

    assert lan_access.detect_lan_addresses(6) == ["fd00::24", "2001:4860:4860::8844"]


def test_interface_enumeration_skips_windows_host_only_switches(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        net_if_stats = lambda: {
            name: types.SimpleNamespace(isup = True)
            for name in (
                "Wi-Fi",
                "vEthernet (Default Switch)",
                "vEthernet (WSL (Hyper-V firewall))",
                "vEthernet (External LAN)",
            )
        },
        net_if_addrs = lambda: {
            "Wi-Fi": [types.SimpleNamespace(family = socket.AF_INET, address = "192.168.1.20")],
            "vEthernet (Default Switch)": [
                types.SimpleNamespace(family = socket.AF_INET, address = "172.31.32.1")
            ],
            "vEthernet (WSL (Hyper-V firewall))": [
                types.SimpleNamespace(family = socket.AF_INET, address = "172.25.32.1")
            ],
            # An external Hyper-V switch can replace the physical adapter and is
            # reachable from the LAN, so do not reject every vEthernet interface.
            "vEthernet (External LAN)": [
                types.SimpleNamespace(family = socket.AF_INET, address = "192.168.1.21")
            ],
        },
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert lan_access._interface_addresses() == ["192.168.1.20", "192.168.1.21"]


def test_wsl_nat_address_is_not_advertised(monkeypatch):
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: "nat")
    monkeypatch.setattr(lan_access, "_interface_addresses", lambda _ip_version = 4: ["172.25.35.232"])
    assert lan_access.detect_lan_addresses() == []


def test_wsl_unknown_networking_mode_fails_closed(monkeypatch):
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: "unknown")
    monkeypatch.setattr(lan_access, "_interface_addresses", lambda _ip_version = 4: ["172.25.35.232"])
    assert lan_access.detect_lan_addresses() == []


def test_wsl_mirrored_networking_keeps_reachable_addresses(monkeypatch):
    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: "mirrored")
    monkeypatch.setattr(lan_access, "_interface_addresses", lambda _ip_version = 4: ["192.168.1.20"])
    assert lan_access.detect_lan_addresses() == ["192.168.1.20"]


def test_wsl_mode_is_cached_while_interface_addresses_still_refresh(monkeypatch):
    class _NoRouteSocket:
        def connect(self, _address):
            raise OSError("network is unreachable")

        def close(self):
            pass

    mode_calls = []
    interface_addresses = iter([["192.168.1.20"], ["192.168.1.21"]])
    monkeypatch.setattr(lan_access.sys, "platform", "linux")
    monkeypatch.setattr(lan_access.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setattr(lan_access, "_wsl_mode_cache", None)
    monkeypatch.setattr(lan_access.socket, "socket", lambda *_args, **_kwargs: _NoRouteSocket())
    monkeypatch.setattr(
        lan_access,
        "_interface_addresses",
        lambda _ip_version = 4: next(interface_addresses),
    )
    monkeypatch.setattr(
        lan_access.subprocess,
        "run",
        lambda *_args, **_kwargs: mode_calls.append(True) or SimpleNamespace(stdout = "mirrored\n"),
    )

    assert lan_access.detect_lan_addresses() == ["192.168.1.20"]
    assert lan_access.detect_lan_addresses() == ["192.168.1.21"]
    assert len(mode_calls) == 1


def test_wsl_mode_cache_retries_a_timeout_after_expiry(monkeypatch):
    now = [100.0]
    mode_calls = []

    def _run(*_args, **_kwargs):
        mode_calls.append(True)
        if len(mode_calls) == 1:
            raise lan_access.subprocess.TimeoutExpired(["wslinfo"], 1)
        return SimpleNamespace(stdout = "mirrored\n")

    monkeypatch.setattr(lan_access.sys, "platform", "linux")
    monkeypatch.setattr(lan_access.platform, "release", lambda: "microsoft-standard-WSL2")
    monkeypatch.setattr(lan_access, "_wsl_mode_cache", None)
    monkeypatch.setattr(lan_access.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(lan_access.subprocess, "run", _run)

    assert lan_access._wsl_networking_mode() == "unknown"
    now[0] += lan_access._WSL_MODE_CACHE_TTL - 1
    assert lan_access._wsl_networking_mode() == "unknown"
    assert len(mode_calls) == 1
    now[0] += 2
    assert lan_access._wsl_networking_mode() == "mirrored"
    assert len(mode_calls) == 2


def test_concurrent_wsl_mode_checks_share_one_probe(monkeypatch):
    probe_started = threading.Event()
    release_probe = threading.Event()
    mode_calls = []
    results = []

    def _run(*_args, **_kwargs):
        mode_calls.append(True)
        probe_started.set()
        assert release_probe.wait(2)
        return SimpleNamespace(stdout = "mirrored\n")

    monkeypatch.setattr(lan_access.sys, "platform", "linux")
    monkeypatch.setattr(lan_access.platform, "release", lambda: "microsoft-standard-WSL2")
    monkeypatch.setattr(lan_access, "_wsl_mode_cache", None)
    monkeypatch.setattr(lan_access.subprocess, "run", _run)
    threads = [
        threading.Thread(target = lambda: results.append(lan_access._wsl_networking_mode()))
        for _ in range(2)
    ]

    threads[0].start()
    assert probe_started.wait(2)
    threads[1].start()
    release_probe.set()
    for thread in threads:
        thread.join(2)

    assert results == ["mirrored", "mirrored"]
    assert len(mode_calls) == 1


# ── live listener ──


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str) -> int:
    with urllib.request.urlopen(url, timeout = 5) as response:
        return response.status


def _free_port_is_bindable(port: int) -> bool:
    """True when the LAN address can still take ``port``, so nothing leaked a socket."""
    try:
        _bind_listener_probe = lan_access._bind_listener(_require_lan_address(), port)
    except OSError:
        return False
    _bind_listener_probe.close()
    return True


def _wait_for_trust(expected: bool, timeout: float = 5.0) -> bool:
    """The flag clears on a drain thread once the stopped listener's requests end."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if host_policy.remote_connector_active() is expected:
            return True
        time.sleep(0.02)
    return host_policy.remote_connector_active() is expected


def _require_lan_address() -> str:
    """The address these tests bind, or a skip on a host with no network address."""
    addresses = lan_access.detect_lan_addresses()
    if not addresses:
        pytest.skip("this host has no LAN address to bind")
    return addresses[0]


def _refused(url: str) -> bool:
    try:
        _get(url)
    except (urllib.error.URLError, OSError):
        return True
    return False


@pytest.fixture
def live_server():
    """A loopback uvicorn server on its own loop, the shape run.py starts."""
    import uvicorn

    app = FastAPI()

    @app.get("/where")
    def where(request: Request):
        return {"lan": lan_access.request_on_lan_listener(request.scope)}

    @app.post("/stop-lan")
    def stop_lan():
        started = time.monotonic()
        lan_access.stop_lan_listener()
        return {"seconds": time.monotonic() - started}

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host = "127.0.0.1", port = port, log_level = "warning"))
    box = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        box["loop"] = loop
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target = _run, daemon = True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "primary server never bound"
    yield SimpleNamespace(app = app, loop = box["loop"], port = port)
    lan_access.stop_lan_listener()
    server.should_exit = True
    thread.join(timeout = 10)


@pytest.mark.allow_network
def test_the_listener_adds_and_removes_network_reach_without_a_restart(live_server):
    address = _require_lan_address()
    lan_url = f"http://{address}:{live_server.port}/where"
    local_url = f"http://127.0.0.1:{live_server.port}/where"

    assert _get(local_url) == 200
    assert _refused(lan_url), "the port was already on the network before starting"

    bound = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert address in bound
    assert _get(lan_url) == 200
    assert _get(local_url) == 200, "loopback must keep serving while the LAN listener runs"
    assert lan_access.lan_listener_status()["running"] is True

    # a second start is a no-op rather than a second bind on the same address
    again = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert again == bound

    lan_access.stop_lan_listener()
    assert _refused(lan_url), "the LAN socket outlived stop"
    assert _get(local_url) == 200
    assert lan_access.lan_listener_status() == {
        "running": False,
        "addresses": [],
        "port": None,
        "error": None,
    }

    # the same address rebinds, so a stop does not strand the port until restart
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert _get(lan_url) == 200


@pytest.mark.allow_network
def test_only_traffic_on_a_lan_socket_is_reported_as_lan(live_server):
    address = _require_lan_address()
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    def _lan_flag(host: str) -> bool:
        with urllib.request.urlopen(
            f"http://{host}:{live_server.port}/where", timeout = 5
        ) as response:
            return json.loads(response.read())["lan"]

    assert _lan_flag(address) is True
    assert _lan_flag("127.0.0.1") is False, "loopback traffic must never read as LAN"


def test_a_start_with_no_usable_address_fails_without_leaving_state(monkeypatch):
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: [])
    with pytest.raises(RuntimeError, match = "no_lan_address"):
        lan_access.start_lan_listener(object(), object(), 8888)
    status = lan_access.lan_listener_status()
    assert status["running"] is False and status["error"] == "no_lan_address"
    lan_access.clear_lan_listener_error()
    assert lan_access.lan_listener_status()["error"] is None


def test_a_port_already_taken_on_every_address_fails_closed(monkeypatch):
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: ["10.0.0.7"])

    def _refuse(*_args, **_kwargs):
        raise OSError("address in use")

    monkeypatch.setattr(lan_access, "_bind_listener", _refuse)
    with pytest.raises(RuntimeError, match = "bind_failed"):
        lan_access.start_lan_listener(object(), object(), 8888)
    assert lan_access.lan_listener_status()["running"] is False


@pytest.mark.allow_network
def test_an_unbindable_address_does_not_sink_the_ones_that_work(live_server, monkeypatch):
    address = _require_lan_address()
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda: [address, "10.255.255.254"])
    bound = lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert bound == (address,)
    assert _get(f"http://{address}:{live_server.port}/where") == 200


@pytest.mark.allow_network
def test_stop_from_inside_the_event_loop_does_not_wait_on_itself(live_server):
    """/api/shutdown tears down from a task on the serving loop; blocking there deadlocks."""
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    async def _stop_from_loop():
        started = time.monotonic()
        lan_access.stop_lan_listener()
        return time.monotonic() - started

    elapsed = asyncio.run_coroutine_threadsafe(_stop_from_loop(), live_server.loop).result(
        timeout = 5
    )
    assert elapsed < lan_access._STOP_TIMEOUT
    # the gate closes at once, but ownership and the trust flag stay until uvicorn
    # closes the sockets, which it cannot do while _graceful_shutdown holds the loop
    assert lan_access._bound_addresses == ()
    assert lan_access.lan_listener_status()["running"] is True
    assert host_policy.remote_connector_active() is True


@pytest.mark.allow_network
def test_a_listener_that_fails_to_serve_gives_up_without_waiting_it_out(live_server, monkeypatch):
    _require_lan_address()

    class _Broken:
        started = False
        should_exit = False

        async def serve(self, sockets = None):
            raise RuntimeError("listener exploded")

    monkeypatch.setattr(lan_access.uvicorn, "Server", lambda _config: _Broken())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "listener_start_failed"):
        lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)
    assert time.monotonic() - started < lan_access._START_TIMEOUT
    status = lan_access.lan_listener_status()
    assert status["running"] is False and status["error"] == "listener_start_failed"
    # the sockets it opened were released, so the port is free again
    assert _free_port_is_bindable(live_server.port)


@pytest.mark.allow_network
def test_a_stop_from_a_lan_client_does_not_wait_out_its_own_response(live_server):
    """Settings > Stop pressed on the phone. uvicorn closes the listening sockets at
    the top of its shutdown and only then drains in-flight responses, so a stop that
    waited for serve() to return would be waiting for the very response it is."""
    address = _require_lan_address()
    lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    request = urllib.request.Request(f"http://{address}:{live_server.port}/stop-lan", method = "POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout = 30) as response:
        handler_seconds = json.loads(response.read())["seconds"]
    assert handler_seconds < lan_access._STOP_TIMEOUT, "the stop waited out its own response"
    assert time.monotonic() - started < lan_access._STOP_TIMEOUT
    assert lan_access.lan_listener_status()["running"] is False
    assert _refused(f"http://{address}:{live_server.port}/where")


def test_a_stop_that_cannot_confirm_the_port_keeps_the_host_marked_reachable(monkeypatch):
    """The sockets may still be accepting, so the loopback-only trust gates must
    not reopen just because the teardown wait ran out."""

    class _Server:
        should_exit = False

    lingering = socket.socket()
    lingering.bind(("127.0.0.1", 0))
    loop = asyncio.new_event_loop()
    running = threading.Event()

    def _spin():
        loop.call_soon(running.set)
        loop.run_forever()

    thread = threading.Thread(target = _spin, daemon = True)
    thread.start()
    assert running.wait(5)
    monkeypatch.setattr(lan_access, "_STOP_TIMEOUT", 0.2)
    monkeypatch.setattr(lan_access, "_server", _Server())
    monkeypatch.setattr(lan_access, "_serve_loop", loop)
    monkeypatch.setattr(lan_access, "_sockets", (lingering,))
    monkeypatch.setattr(lan_access, "_bound_addresses", ("10.0.0.7",))
    host_policy.set_lan_connector_active(True)

    try:
        app = _app()
        status = lan_settings.stop_lan_access(app)
        assert status["error"] == "stop_timed_out"
        assert host_policy.remote_connector_active() is True
        # ownership is retained, so the stop stays offered and a retry re-waits
        assert status["can_stop"] is True
        assert lan_access.lan_listener_status()["running"] is True

        # a second stop must not report success while the port may still accept
        retried = lan_settings.stop_lan_access(app)
        assert retried["error"] == "stop_timed_out"
        assert host_policy.remote_connector_active() is True

        # once the socket really closes, the retry settles and clears the error
        lingering.close()
        settled = lan_settings.stop_lan_access(app)
        assert settled["state"] == "off" and settled["error"] is None
        assert _wait_for_trust(False)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout = 5)
        loop.close()
        lingering.close()
        host_policy.set_lan_connector_active(False)


@pytest.mark.allow_network
def test_a_schedule_that_cannot_reach_the_loop_releases_the_bound_sockets(live_server, monkeypatch):
    """The loop can close between _server_loop validating it and the schedule call;
    the sockets are bound by then, so that path has to run the start cleanup too."""
    _require_lan_address()

    def _closed(*_args, **_kwargs):
        raise RuntimeError("Event loop is closed")

    monkeypatch.setattr(lan_access.asyncio, "run_coroutine_threadsafe", _closed)
    with pytest.raises(RuntimeError, match = "listener_start_failed"):
        lan_access.start_lan_listener(live_server.app, live_server.loop, live_server.port)

    status = lan_access.lan_listener_status()
    assert status["running"] is False and status["error"] == "listener_start_failed"
    assert host_policy.remote_connector_active() is False, "trust left on with no listener"
    # the port is free, so a later start is not locked out by the leaked sockets
    assert _free_port_is_bindable(live_server.port)


@pytest.mark.allow_network
def test_a_stop_on_the_serving_loop_never_blocks_on_a_start_holding_the_lock(live_server):
    """A start holds _lock while waiting for this loop to run serve(). /api/shutdown
    reaches stop_lan_listener from that same loop, so blocking on the lock would
    leave the two waiting each other out for the whole start timeout."""
    _require_lan_address()
    holding = threading.Event()
    release = threading.Event()

    def _hold():
        with lan_access._lock:
            holding.set()
            release.wait(10)

    holder = threading.Thread(target = _hold, daemon = True)
    holder.start()
    assert holding.wait(5)
    try:

        async def _stop_from_loop():
            started = time.monotonic()
            return lan_access.stop_lan_listener(), time.monotonic() - started

        stopped, elapsed = asyncio.run_coroutine_threadsafe(
            _stop_from_loop(), live_server.loop
        ).result(timeout = 10)
        assert elapsed < 1, "the loop-side stop waited on the lock"
        assert stopped is False, "an unconfirmed stop must not report success"
    finally:
        release.set()
        holder.join(timeout = 10)


def test_the_server_thread_releases_the_listener_once_its_loop_ends():
    """An embedded host calls run_server again in-process, and a stale _server would
    report the old addresses as online and skip binding a new listener."""
    source = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    run_body = source[
        source.index("    def _run():") : source.index("    thread = Thread(target = _run")
    ]
    assert "loop.close()" in run_body
    assert "_close_lan_listener()" in run_body


def test_stop_releases_the_sockets_itself_when_the_serving_loop_is_gone(monkeypatch):
    """An abnormal exit leaves nobody to run uvicorn's shutdown, so stop must close them."""

    class _Server:
        should_exit = False

    stranded = socket.socket()
    stranded.bind(("127.0.0.1", 0))
    dead = asyncio.new_event_loop()
    dead.close()
    monkeypatch.setattr(lan_access, "_server", _Server())
    monkeypatch.setattr(lan_access, "_serve_loop", dead)
    monkeypatch.setattr(lan_access, "_sockets", (stranded,))
    monkeypatch.setattr(lan_access, "_bound_addresses", ("10.0.0.7",))

    started = time.monotonic()
    lan_access.stop_lan_listener()
    assert time.monotonic() - started < 1
    assert stranded.fileno() == -1, "the socket outlived the loop that owned it"
    assert lan_access.lan_listener_status()["running"] is False


# ── settings orchestration ──


def _configured(live_server):
    """The live app carrying the launch policy run.py would have published on it."""
    lan_settings.configure_lan_access(
        live_server.app.state,
        port = live_server.port,
        bind_host = "127.0.0.1",
        secure = False,
        is_colab = False,
        frontend_served = True,
    )
    live_server.app.state.lan_access_ready = True
    live_server.app.state.lan_access_loop = live_server.loop
    return live_server.app


@pytest.mark.allow_network
def test_start_and_stop_track_the_beyond_loopback_trust_flag(live_server):
    app = _configured(live_server)
    assert host_policy.remote_connector_active() is False

    status = lan_settings.start_lan_access(app)
    assert status["state"] == "online" and status["managed_by"] == "settings"
    assert status["urls"] and status["urls"][0].endswith(f":{live_server.port}")
    assert status["can_stop"] is True and status["can_start"] is False
    assert host_policy.remote_connector_active() is True

    # a repeated start is answered with the running listener, not a second bind
    assert lan_settings.start_lan_access(app)["urls"] == status["urls"]

    stopped = lan_settings.stop_lan_access(app)
    assert stopped["state"] == "off" and stopped["managed_by"] is None
    assert stopped["can_start"] is True
    assert _wait_for_trust(False)


@pytest.mark.allow_network
def test_auto_start_brings_the_listener_up_at_boot(live_server, stored_settings):
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = True
    app = _configured(live_server)
    assert lan_settings.maybe_auto_start_lan_access(app) is True
    assert lan_settings.lan_access_status(app)["state"] == "online"
    # stopping now must not silently clear the preference
    lan_settings.stop_lan_access(app)
    assert lan_settings.get_lan_access_auto_start() is True


def test_trust_outlives_the_sockets_until_accepted_requests_drain(monkeypatch):
    """Stop closes the listening sockets, but uvicorn then drains the connections it
    already accepted, and a request that arrived on the LAN stays a remote caller."""
    connections = {object()}
    server = SimpleNamespace(server_state = SimpleNamespace(connections = connections))
    monkeypatch.setattr(lan_access, "_DRAIN_TIMEOUT", 5.0)

    with lan_access._lock:
        lan_access._arm_drain_watcher(server)
        lan_access._sync_lan_trust()
    time.sleep(0.2)
    assert host_policy.remote_connector_active() is True, "cleared while a request ran"

    connections.clear()
    assert _wait_for_trust(False)


def test_a_repeated_stop_does_not_clear_trust_a_pending_drain_still_owns(monkeypatch):
    """The second stop finds no server and would otherwise release the flag out from
    under requests the first stop left draining."""
    connections = {object()}
    server = SimpleNamespace(server_state = SimpleNamespace(connections = connections))
    monkeypatch.setattr(lan_access, "_DRAIN_TIMEOUT", 5.0)

    with lan_access._lock:
        lan_access._arm_drain_watcher(server)
        lan_access._sync_lan_trust()
    assert host_policy.remote_connector_active() is True

    # the idempotent stop path: no server, but a drain still owns the flag
    assert lan_access.stop_lan_listener() is True
    assert host_policy.remote_connector_active() is True

    connections.clear()
    assert _wait_for_trust(False)


def test_a_drain_that_finishes_after_a_restart_leaves_the_new_listener_trusted(monkeypatch):
    server = SimpleNamespace(server_state = SimpleNamespace(connections = set()))
    monkeypatch.setattr(lan_access, "_server", SimpleNamespace(should_exit = False))
    host_policy.set_lan_connector_active(True)
    lan_access._clear_trust_after_drain(server)
    assert host_policy.remote_connector_active() is True
    host_policy.set_lan_connector_active(False)


def test_the_trust_flag_moves_with_the_listener_under_one_lock():
    """A start and a stop can run in separate FastAPI worker threads, so the flag
    cannot be published by the callers: it has to change where the state does."""
    settings_source = (_BACKEND / "utils" / "lan_access_settings.py").read_text(encoding = "utf-8")
    assert "set_lan_connector_active" not in settings_source

    listener_source = (_BACKEND / "lan_access.py").read_text(encoding = "utf-8")
    tree = ast.parse(listener_source)
    holders = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        if "set_lan_connector_active" in body:
            holders.add(node.name)
    # only the two helpers that run under _lock may touch it
    assert holders == {"start_lan_listener", "_sync_lan_trust"}, holders


def test_start_refuses_while_a_block_is_in_force():
    with pytest.raises(RuntimeError, match = "server_starting"):
        lan_settings.start_lan_access(_app(lan_access_ready = False))
    with pytest.raises(RuntimeError, match = "launch_managed"):
        lan_settings.stop_lan_access(_app(lan_access_launch_managed = True))


def test_start_refuses_a_port_the_server_never_published():
    with pytest.raises(RuntimeError, match = "server_port_unavailable"):
        lan_settings.start_lan_access(_app(lan_access_port = None))


def test_start_refuses_once_the_server_loop_is_gone(monkeypatch):
    monkeypatch.setattr(
        lan_access, "start_lan_listener", lambda *_: pytest.fail("bound without a live loop")
    )
    with pytest.raises(RuntimeError, match = "server_not_running"):
        lan_settings.start_lan_access(_app(lan_access_loop = None))
    # a loop that exists but is no longer serving would never run the listener either
    idle = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match = "server_not_running"):
            lan_settings.start_lan_access(_app(lan_access_loop = idle))
    finally:
        idle.close()
    with pytest.raises(RuntimeError, match = "server_not_running"):
        lan_settings.start_lan_access(_app(lan_access_loop = idle))


def test_auto_start_stays_quiet_when_the_launch_forbids_it(stored_settings):
    stored_settings[lan_settings.LAN_ACCESS_AUTO_START_KEY] = True
    assert lan_settings.maybe_auto_start_lan_access(_app(lan_access_is_colab = True)) is False


def test_auto_start_is_skipped_entirely_when_the_preference_is_off(monkeypatch):
    monkeypatch.setattr(
        lan_settings, "start_lan_access", lambda _app: pytest.fail("started despite the preference")
    )
    assert lan_settings.maybe_auto_start_lan_access(_app()) is False


# ── routes and wiring ──


def test_colab_auto_start_setting_is_read_only(monkeypatch):
    monkeypatch.setattr(routes, "set_lan_access_auto_start", lambda *_: pytest.fail("persisted"))
    request = SimpleNamespace(app = _app(lan_access_is_colab = True))
    payload = routes.LanAccessAutoStartPayload(enabled = True)
    with pytest.raises(HTTPException) as exc:
        routes.update_lan_access_auto_start(request, payload, "admin", None)
    assert exc.value.status_code == 409


def test_a_blocked_start_answers_409_rather_than_500(monkeypatch):
    monkeypatch.setattr(
        routes, "start_lan_access", lambda _app: (_ for _ in ()).throw(RuntimeError("colab"))
    )
    with pytest.raises(HTTPException) as exc:
        routes.start_lan_access_route(SimpleNamespace(app = _app()), "admin", None)
    assert exc.value.status_code == 409 and exc.value.detail == "colab"


def test_management_rejects_api_keys():
    with pytest.raises(HTTPException) as exc:
        routes._require_ui_session(True)
    assert exc.value.status_code == 403
    # every /lan-access handler must carry the UI-session gate
    tree = ast.parse(Path(routes.__file__).read_text(encoding = "utf-8"))
    gated = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        routed = [ast.unparse(d) for d in node.decorator_list if "router." in ast.unparse(d)]
        if not any("'/lan-access" in d for d in routed):
            continue
        args = node.args.args + node.args.kwonlyargs
        gated[node.name] = any(a.arg == "_ui_session" for a in args)
    assert len(gated) == 4, f"expected 4 lan-access handlers, found {sorted(gated)}"
    assert all(
        gated.values()
    ), f"ungated lan-access handlers: {sorted(k for k, v in gated.items() if not v)}"


def test_the_desktop_frontend_gate_admits_the_lan_listener():
    """A desktop backend mounts its SPA behind a remote-only gate; LAN must open it."""
    import main

    scope = {"type": "http", "headers": [], "server": ("10.0.0.7", 8888)}
    state = SimpleNamespace(cloudflare_url = None)
    assert main._is_remote_frontend_request(scope, state) is False
    original = lan_access._bound_addresses
    lan_access._bound_addresses = ("10.0.0.7",)
    try:
        assert main._is_remote_frontend_request(scope, state) is True
        loopback = {"type": "http", "headers": [], "server": ("127.0.0.1", 8888)}
        assert main._is_remote_frontend_request(loopback, state) is False
    finally:
        lan_access._bound_addresses = original


def test_the_desktop_assets_mount_admits_the_lan_listener():
    """The SPA is mounted twice on a desktop api-only launch: /assets through
    _TunnelOnlyFrontend, and the routes through _frontend_request_allowed."""
    import main

    served = []

    async def _spa(_scope, _receive, send):
        served.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def _drive(scope):
        statuses = []

        async def _send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        await main._TunnelOnlyFrontend(_spa, SimpleNamespace(cloudflare_url = None))(
            scope, None, _send
        )
        return statuses[0]

    lan = {"type": "http", "headers": [], "server": ("10.0.0.7", 8888)}
    loopback = {"type": "http", "headers": [], "server": ("127.0.0.1", 8888)}
    assert asyncio.run(_drive(lan)) == 404

    original = lan_access._bound_addresses
    lan_access._bound_addresses = ("10.0.0.7",)
    try:
        assert asyncio.run(_drive(lan)) == 200
        assert asyncio.run(_drive(loopback)) == 404, "loopback keeps the api-only surface"
    finally:
        lan_access._bound_addresses = original
    assert served

    # the SPA routes share the decision through a closure the mount does not expose
    source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "return not tunnel_only or _is_remote_frontend_request(request.scope, app.state)" in (
        source
    )


def test_run_py_wires_the_listener_into_the_server_lifecycle():
    source = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    # without the loop the settings route has nothing to schedule the listener on
    assert "app.state.lan_access_loop = loop" in source
    assert "close_lan_listener_lifecycle" in source
    # readiness is what unblocks the whole feature, and it has to land before auto-start
    ready = source.index("app.state.lan_access_ready = True")
    assert ready < source.index("maybe_auto_start_lan_access(app)")
