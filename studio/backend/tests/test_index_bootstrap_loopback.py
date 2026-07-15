# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for bootstrap password exposure to remote clients."""

from types import SimpleNamespace


def _request(client_host, headers = None):
    """Build a minimal request; ``None`` models an unresolved socket peer."""
    client = None if client_host is None else SimpleNamespace(host = client_host, port = 0)
    return SimpleNamespace(client = client, headers = headers or {})


def test_loopback_peers_are_local():
    from main import _is_loopback_client
    for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        assert _is_loopback_client(_request(host)) is True, host


def test_non_loopback_peers_are_remote():
    from main import _is_loopback_client
    for host in ("192.168.1.10", "::ffff:192.168.1.10"):
        assert _is_loopback_client(_request(host)) is False, host


def test_absent_or_unparseable_peer_fails_safe():
    from main import _is_loopback_client
    for host in (None, "localhost"):
        assert _is_loopback_client(_request(host)) is False, host


def test_cloudflare_tunnel_clients_are_remote_despite_loopback_peer():
    from main import _is_loopback_client
    for client_ip in ("203.0.113.7", ""):
        request = _request("127.0.0.1", {"cf-connecting-ip": client_ip})
        assert _is_loopback_client(request) is False, client_ip
