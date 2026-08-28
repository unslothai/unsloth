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

import socket
import urllib.request

import pytest

import run
import startup_banner
from run import _network_share_host_for_bind, _resolve_lan_ip
from startup_banner import print_studio_access_banner

PUBLIC_IP = "104.32.48.18"
LAN_IP = "192.168.1.50"


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


def test_resolve_lan_ip_never_calls_the_network(public_and_lan):
    # Only the UDP-socket trick backs this, never urlopen.
    assert _resolve_lan_ip() == LAN_IP


def test_external_ip_prefers_the_public_service(public_and_lan):
    assert run._resolve_external_ip() == PUBLIC_IP


def test_network_share_host_is_lan_not_public(public_and_lan):
    assert _network_share_host_for_bind("0.0.0.0") == LAN_IP
    assert _network_share_host_for_bind("::") == LAN_IP


def test_network_share_host_is_unchanged_for_a_specific_bind(public_and_lan):
    # A non-wildcard bind already names its own address; nothing to resolve.
    assert _network_share_host_for_bind("192.168.1.7") == "192.168.1.7"


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


def test_banner_falls_back_to_display_host_when_network_host_is_unset(capsys):
    # Back-compat for a caller that only ever had one address to give.
    print_studio_access_banner(
        port = 8888,
        bind_host = "0.0.0.0",
        display_host = "203.0.113.9",
    )
    assert "http://203.0.113.9:8888" in capsys.readouterr().out
