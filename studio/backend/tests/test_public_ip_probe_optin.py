# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for the opt-in public-IP lookup (#7307 Problem 8).

Startup used to ask ifconfig.me for the machine's public IP unconditionally,
which discloses to a third party that this host is running Unsloth. The lookup
is now off unless the user sets UNSLOTH_STUDIO_PUBLIC_IP_PROBE. These tests pin
the env parsing and, most importantly, that no outbound request to the public-IP
service happens on the default path.
"""

import socket
import urllib.request

import pytest

from run import (
    PUBLIC_IP_PROBE_ENV_VAR,
    PUBLIC_IP_PROBE_URL,
    _resolve_external_ip,
    public_ip_probe_enabled,
)


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


# ── _resolve_external_ip ────────────────────────────────────────────


@pytest.fixture
def no_metadata_server(monkeypatch):
    """Make step 1 (GCE metadata) fail so the test reaches step 2."""
    calls = []
    real_urlopen = urllib.request.urlopen

    def _urlopen(req, *args, **kwargs):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        raise OSError("no metadata server in this test")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSocket())
    return calls, real_urlopen


class _FakeSocket:
    """Stand-in for the step 3 UDP route lookup."""

    def connect(self, addr):
        pass

    def getsockname(self):
        return ("192.168.1.50", 0)

    def close(self):
        pass


def test_default_never_contacts_the_public_ip_service(monkeypatch, no_metadata_server):
    calls, _ = no_metadata_server
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)

    assert _resolve_external_ip() == "192.168.1.50"
    assert (
        PUBLIC_IP_PROBE_URL not in calls
    ), "the public-IP service must not be contacted unless the user opts in"


def test_opt_in_contacts_the_public_ip_service(monkeypatch, no_metadata_server):
    calls, _ = no_metadata_server
    monkeypatch.setenv(PUBLIC_IP_PROBE_ENV_VAR, "1")

    # Both urlopen calls raise, so this still falls through to the LAN address;
    # what matters is that the attempt was made.
    assert _resolve_external_ip() == "192.168.1.50"
    assert PUBLIC_IP_PROBE_URL in calls


def test_falls_back_to_lan_address_when_disabled(monkeypatch, no_metadata_server):
    """Disabling the probe must not break the banner: a LAN address is still
    resolved, it is simply not the public one."""
    monkeypatch.delenv(PUBLIC_IP_PROBE_ENV_VAR, raising = False)
    assert _resolve_external_ip() == "192.168.1.50"
