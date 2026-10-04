# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK (#7307 Problem 8).

A wildcard bind asks ifconfig.me for the public IP and check-host.net whether the
port is reachable. Both stay on by default; setting the var skips both, which is
what lab and privacy-sensitive deployments asked for.
"""

import socket
import urllib.request

import pytest

import run
from run import (
    DISABLE_PUBLIC_CHECK_ENV,
    _resolve_external_ip,
    _verify_global_reachability,
    public_check_disabled,
)

IFCONFIG = "https://ifconfig.me"
CHECK_HOST = "check-host.net"


class _FakeSocket:
    """Stand-in for the step 3 UDP route lookup."""

    def connect(self, addr):
        pass

    def getsockname(self):
        return ("192.168.1.50", 0)

    def close(self):
        pass


@pytest.fixture
def calls(monkeypatch):
    """Record every outbound URL and fail it, so resolution reaches the LAN step."""
    seen = []

    def _urlopen(req, *args, **kwargs):
        seen.append(req if isinstance(req, str) else req.full_url)
        raise OSError("no network in this test")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket())
    monkeypatch.delenv(DISABLE_PUBLIC_CHECK_ENV, raising = False)
    return seen


# ── public_check_disabled ───────────────────────────────────────────


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv(DISABLE_PUBLIC_CHECK_ENV, raising = False)
    assert public_check_disabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", " 1 "])
def test_disabling_values(monkeypatch, raw):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, raw)
    assert public_check_disabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "ture"])
def test_anything_else_leaves_it_on(monkeypatch, raw):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, raw)
    assert public_check_disabled() is False


# ── the two lookups ─────────────────────────────────────────────────


def test_public_ip_lookup_runs_by_default(calls):
    assert _resolve_external_ip() == "192.168.1.50"
    assert IFCONFIG in calls


def test_public_ip_lookup_skipped_when_disabled(monkeypatch, calls):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")

    assert _resolve_external_ip() == "192.168.1.50", "the LAN address still resolves"
    assert IFCONFIG not in calls


def test_display_host_resolves_every_wildcard_alias(monkeypatch):
    monkeypatch.setattr(run, "_resolve_external_ip", lambda: "192.168.1.50")
    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: ["fd00::50"])
    for host in ("0.0.0.0", "0", "::ffff:0.0.0.0"):
        assert run._display_host_for_bind(host) == "192.168.1.50"
    for host in ("::", "::0", "0:0:0:0:0:0:0:0"):
        assert run._display_host_for_bind(host) == "fd00::50"

    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: [])
    assert run._display_host_for_bind("::") == "::"


def test_display_host_falls_back_to_ipv6_for_dual_stack_wildcard(monkeypatch):
    original_getaddrinfo = socket.getaddrinfo

    def dual_stack_wildcard(host, *args, **kwargs):
        if host == "dual-wildcard.test":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0)),
            ]
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", dual_stack_wildcard)
    monkeypatch.setattr(run, "_resolve_external_ip", lambda: "0.0.0.0")
    monkeypatch.setattr("lan_access.detect_lan_addresses", lambda _ip_version = 4: ["fd00::50"])

    assert run._display_host_for_bind("dual-wildcard.test") == "fd00::50"


def test_reachability_probe_runs_by_default(calls):
    _verify_global_reachability("95.216.11.2", 8888)
    assert any(CHECK_HOST in url for url in calls)


def test_ipv6_reachability_probe_brackets_the_host(calls):
    import urllib.parse

    _verify_global_reachability("2001:4860:4860::8844", 8888)
    request_url = next(url for url in calls if CHECK_HOST in url)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(request_url).query)
    assert query["host"] == ["[2001:4860:4860::8844]:8888"]


def test_reachability_probe_skipped_when_disabled(monkeypatch, calls, capsys):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")

    _verify_global_reachability("95.216.11.2", 8888)
    capsys.readouterr()

    assert not any(CHECK_HOST in url for url in calls)
    assert run._public_reachable is None, "skipping must not claim a reachability result"
