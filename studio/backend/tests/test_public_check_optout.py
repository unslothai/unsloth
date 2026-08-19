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


def test_reachability_probe_runs_by_default(calls):
    _verify_global_reachability("95.216.11.2", 8888)
    assert any(CHECK_HOST in url for url in calls)


def test_reachability_probe_skipped_when_disabled(monkeypatch, calls, capsys):
    monkeypatch.setenv(DISABLE_PUBLIC_CHECK_ENV, "1")

    _verify_global_reachability("95.216.11.2", 8888)
    capsys.readouterr()

    assert not any(CHECK_HOST in url for url in calls)
    assert run._public_reachable is None, "skipping must not claim a reachability result"
