# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for the opt-in startup probes (#7307 Problem 8).

Startup used to ask ifconfig.me for the machine's public IP and then hand that
address to check-host.net, which asks its own nodes to connect back to the bound
port. Both told a third party that this host is running Unsloth. Both are now off
unless the user sets UNSLOTH_STUDIO_PUBLIC_IP_PROBE. These tests pin the env
parsing, that neither service is contacted on the default path, and that cloud
metadata still supplies a shareable address without any third party.
"""

import socket
import urllib.request

import pytest

import run
from run import (
    PUBLIC_IP_PROBE_ENV_VAR,
    PUBLIC_IP_PROBE_URL,
    _resolve_external_ip,
    _verify_global_reachability,
    public_ip_probe_enabled,
)

CHECK_HOST = "check-host.net"


# ── public_ip_probe_enabled ─────────────────────────────────────────


def test_disabled_when_unset():
    assert public_ip_probe_enabled(env = {}) is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", " on "])
def test_enabled_values(raw):
    assert public_ip_probe_enabled(env = {PUBLIC_IP_PROBE_ENV_VAR: raw}) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "ture", "2"])
def test_anything_else_stays_disabled(raw):
    """A typo must not silently opt the user in."""
    assert public_ip_probe_enabled(env = {PUBLIC_IP_PROBE_ENV_VAR: raw}) is False


# ── network fixture ─────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, body):
        self._body = body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSocket:
    """Stand-in for the step 3 UDP route lookup."""

    def connect(self, addr):
        pass

    def getsockname(self):
        return ("192.168.1.50", 0)

    def close(self):
        pass


@pytest.fixture
def net(monkeypatch):
    """Record every outbound URL. ``replies`` maps a URL fragment to a body;
    anything unmatched raises, so by default every metadata source fails."""
    calls, replies = [], {}

    def _urlopen(req, *args, **kwargs):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        for frag, body in replies.items():
            if frag in url:
                return _FakeResp(body)
        raise OSError("unmocked URL in this test")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket())
    return calls, replies


# ── _resolve_external_ip ────────────────────────────────────────────


def test_default_never_contacts_the_public_ip_service(monkeypatch, net):
    calls, _ = net
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)

    assert _resolve_external_ip() == "192.168.1.50"
    assert (
        PUBLIC_IP_PROBE_URL not in calls
    ), "the public-IP service must not be contacted unless the user opts in"


def test_opt_in_contacts_the_public_ip_service(monkeypatch, net):
    calls, replies = net
    replies["ifconfig.me"] = "95.216.11.2"
    monkeypatch.setenv(PUBLIC_IP_PROBE_ENV_VAR, "1")

    assert _resolve_external_ip() == "95.216.11.2"
    assert PUBLIC_IP_PROBE_URL in calls


def test_falls_back_to_lan_address_when_disabled(monkeypatch, net):
    """Disabling the probe must not break the banner: a LAN address is still
    resolved, it is simply not the public one."""
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)
    assert _resolve_external_ip() == "192.168.1.50"


@pytest.mark.parametrize(
    "frag, body, expected",
    [
        ("/computeMetadata/", "34.72.10.5", "34.72.10.5"),  # GCE
        ("publicIpAddress", "20.51.7.44", "20.51.7.44"),  # Azure
    ],
)
def test_cloud_metadata_supplies_the_public_ip(monkeypatch, net, frag, body, expected):
    """Link-local metadata keeps the shareable address working with the
    third-party lookup off, so the privacy default is not a degraded mode."""
    calls, replies = net
    replies[frag] = body
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)

    assert _resolve_external_ip() == expected
    assert PUBLIC_IP_PROBE_URL not in calls


def test_aws_imdsv2_supplies_the_public_ip(monkeypatch, net):
    calls, replies = net
    replies["/latest/api/token"] = "tok"
    replies["/latest/meta-data/public-ipv4"] = "52.14.33.9"
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)

    assert _resolve_external_ip() == "52.14.33.9"
    assert PUBLIC_IP_PROBE_URL not in calls


# ── _verify_global_reachability ─────────────────────────────────────


def test_default_never_contacts_the_reachability_service(monkeypatch, net, capsys):
    """The probe reported in #7307 Problem 8: check-host.net is told this
    machine's address and port, and asks its nodes to connect to it."""
    calls, _ = net
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)

    _verify_global_reachability("95.216.11.2", 8888)
    capsys.readouterr()

    assert not any(CHECK_HOST in url for url in calls)
    assert run._public_reachable is None, "opting out must not claim the port is private"


def test_opt_in_contacts_the_reachability_service(monkeypatch, net):
    calls, _ = net
    monkeypatch.setenv(PUBLIC_IP_PROBE_ENV_VAR, "1")

    _verify_global_reachability("95.216.11.2", 8888)

    assert any(CHECK_HOST in url for url in calls)
