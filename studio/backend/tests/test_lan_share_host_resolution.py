# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for #8868: a wildcard bind (``-H 0.0.0.0``) must not hand a
device on the LAN the machine's public WAN IP.

``_resolve_external_ip()`` (used for the reachability probe and the
Cloudflare line) can return a public address from ``ifconfig.me`` or the GCE
metadata server. ``_network_share_host_for_bind()`` is the LAN-only answer --
no third-party network call -- and is what the "another device on your
network" banner line and ``app.state.server_url`` must use instead.
"""

import logging
import socket
import urllib.request

import pytest

import lan_access
import run
import startup_banner
from run import _direct_server_url, _network_share_host_for_bind, _resolve_lan_ip
from startup_banner import print_studio_access_banner

PUBLIC_IP = "104.32.48.18"
LAN_IP = "192.168.1.50"
LAN_IPV6 = "fd00::50"
# What the route lookup reports where detect_lan_addresses declines to advertise
# anything: WSL's NAT-side address is real and routable from the Windows host,
# it is just not an address to hand a phone.
ROUTE_ONLY_IP = "172.29.1.5"


class _FakeSocket:
    """Stand-in for the UDP route lookup: no packet ever leaves the host."""

    def connect(self, addr):
        pass

    def getsockname(self):
        return (LAN_IP, 0)

    def close(self):
        pass


@pytest.fixture
def public_and_lan(monkeypatch):
    """ifconfig.me answers with a public IP; the LAN socket trick answers separately."""

    def _urlopen(req, *args, **kwargs):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return PUBLIC_IP.encode()

        url = req if isinstance(req, str) else req.full_url
        if "metadata.google" in url:
            raise OSError("not on GCE")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket())
    monkeypatch.delenv(run.DISABLE_PUBLIC_CHECK_ENV, raising = False)


# ── resolution ───────────────────────────────────────────────────────


def test_resolve_lan_ip_never_calls_the_network(public_and_lan, monkeypatch):
    # Only the UDP-socket trick backs this, never urlopen.
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("LAN resolution made an HTTP request"),
    )
    assert _resolve_lan_ip() == LAN_IP


def test_external_ip_prefers_the_public_service(public_and_lan):
    assert run._resolve_external_ip() == PUBLIC_IP


def test_network_share_host_is_lan_not_public(monkeypatch):
    monkeypatch.setattr(
        run,
        "_resolve_lan_ip",
        lambda ip_version = 4: LAN_IP if ip_version == 4 else LAN_IPV6,
    )
    assert _network_share_host_for_bind("0.0.0.0") == LAN_IP
    assert _network_share_host_for_bind("::") == LAN_IPV6
    assert _network_share_host_for_bind("::0") == LAN_IPV6


def test_network_share_host_is_unchanged_for_a_specific_bind(public_and_lan):
    # A non-wildcard bind already names its own address; nothing to resolve.
    assert _network_share_host_for_bind("192.168.1.7") == "192.168.1.7"


def test_external_ip_falls_back_to_the_raw_route_address(monkeypatch):
    """The LAN detector's filtering is right for an address we advertise and wrong
    as the last resort for "where am I": _resolve_external_ip keeps its own route
    lookup, so a WSL/NAT host with no public answer still reports a real address."""
    monkeypatch.setattr(lan_access, "detect_lan_addresses", lambda _ip_version = 4: [])
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    class _RouteSocket(_FakeSocket):
        def getsockname(self):
            return (ROUTE_ONLY_IP, 0)

    monkeypatch.setattr(socket, "socket", lambda *a, **k: _RouteSocket())
    assert run._resolve_external_ip() == ROUTE_ONLY_IP
    # The sharing resolver still declines, which is the whole point of the split.
    assert _network_share_host_for_bind("0.0.0.0") == "0.0.0.0"


# ── the direct server URL ────────────────────────────────────────────


def test_direct_server_url_is_the_lan_address(monkeypatch):
    monkeypatch.setattr(run, "_resolve_lan_ip", lambda ip_version = 4: LAN_IP)
    assert _direct_server_url("0.0.0.0", 8888) == f"http://{LAN_IP}:8888"


def test_direct_server_url_is_unset_when_no_lan_address_is_detectable(monkeypatch):
    """Never publish the wildcard: the frontend prefers any non-null server_url
    over the origin the client actually reached, so http://0.0.0.0:8888 would be
    what the API panel, the desktop agent command and a copied preview link name."""
    monkeypatch.setattr(run, "_resolve_lan_ip", lambda ip_version = 4: "0.0.0.0")
    assert _direct_server_url("0.0.0.0", 8888) is None
    monkeypatch.setattr(run, "_resolve_lan_ip", lambda ip_version = 6: "::")
    assert _direct_server_url("::", 8888) is None


# ── the uvicorn startup log line ─────────────────────────────────────


@pytest.fixture
def uvicorn_log_filters():
    """The rewrite installs a filter on the uvicorn loggers and never removes it."""
    loggers = [logging.getLogger(name) for name in ("uvicorn", "uvicorn.error")]
    saved = [list(log.filters) for log in loggers]
    yield loggers[1]
    for log, filters in zip(loggers, saved):
        log.filters[:] = filters


def _rewritten_startup_line(logger, bind_host: str) -> str:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
        ("http", bind_host, 8888),
        None,
    )
    for log_filter in logger.filters:
        log_filter.filter(record)
    return record.getMessage()


def test_startup_log_line_names_the_lan_address(uvicorn_log_filters, monkeypatch):
    """#8868 as reported: the line reads as a claim about where this machine
    answers, so a wildcard bind must not be rewritten to the public WAN IP."""
    monkeypatch.setattr(run, "_resolve_lan_ip", lambda ip_version = 4: LAN_IP)
    run._install_uvicorn_startup_log_rewrite("0.0.0.0")
    line = _rewritten_startup_line(uvicorn_log_filters, "0.0.0.0")
    assert f"http://{LAN_IP}:8888" in line
    assert PUBLIC_IP not in line


def test_startup_log_line_keeps_the_wildcard_when_no_lan_address_resolves(
    uvicorn_log_filters, monkeypatch
):
    # The issue's own "expected behavior": say 0.0.0.0 rather than invent an address.
    monkeypatch.setattr(run, "_resolve_lan_ip", lambda ip_version = 4: "0.0.0.0")
    run._install_uvicorn_startup_log_rewrite("0.0.0.0")
    assert "http://0.0.0.0:8888" in _rewritten_startup_line(uvicorn_log_filters, "0.0.0.0")


# ── the banner line itself ──────────────────────────────────────────


def test_banner_network_line_shows_lan_ip_not_public_ip(capsys):
    print_studio_access_banner(
        port = 8888,
        bind_host = "0.0.0.0",
        display_host = PUBLIC_IP,
        network_host = LAN_IP,
    )
    out = capsys.readouterr().out
    assert f"http://{LAN_IP}:8888" in out
    assert PUBLIC_IP not in out


def test_banner_network_line_brackets_ipv6_for_a_wildcard_alias(capsys):
    print_studio_access_banner(
        port = 8888,
        bind_host = "::0",
        display_host = PUBLIC_IP,
        network_host = LAN_IPV6,
    )
    out = capsys.readouterr().out
    assert f"http://[{LAN_IPV6}]:8888" in out
    assert PUBLIC_IP not in out


def test_banner_falls_back_to_display_host_when_network_host_is_unset(capsys):
    # Back-compat for a caller that only ever had one address to give.
    print_studio_access_banner(
        port = 8888,
        bind_host = "0.0.0.0",
        display_host = "203.0.113.9",
    )
    assert "http://203.0.113.9:8888" in capsys.readouterr().out
