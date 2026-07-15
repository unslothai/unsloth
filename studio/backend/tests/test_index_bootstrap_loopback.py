# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for bootstrap password exposure to remote clients."""

from types import SimpleNamespace


def _request(
    client_host,
    request_host = "127.0.0.1",
    headers = None,
):
    """Build a minimal request; ``None`` models an unresolved socket peer."""
    client = None if client_host is None else SimpleNamespace(host = client_host, port = 0)
    url = SimpleNamespace(hostname = request_host)
    return SimpleNamespace(client = client, headers = headers or {}, url = url)


def test_loopback_peers_are_local():
    from main import _is_local_bootstrap_request
    cases = (
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
        ("::ffff:127.0.0.1", "::ffff:127.0.0.1"),
        ("127.0.0.1", "localhost"),
    )
    for peer, host in cases:
        assert _is_local_bootstrap_request(_request(peer, host)) is True, (peer, host)


def test_non_loopback_peers_are_remote():
    from main import _is_local_bootstrap_request
    for host in ("192.168.1.10", "::ffff:192.168.1.10"):
        assert _is_local_bootstrap_request(_request(host)) is False, host


def test_absent_or_unparseable_peer_fails_safe():
    from main import _is_local_bootstrap_request
    for host in (None, "localhost"):
        assert _is_local_bootstrap_request(_request(host)) is False, host


def test_cloudflare_tunnel_clients_are_remote_despite_loopback_peer():
    from main import _is_local_bootstrap_request
    for client_ip in ("203.0.113.7", ""):
        request = _request("127.0.0.1", headers = {"cf-connecting-ip": client_ip})
        assert _is_local_bootstrap_request(request) is False, client_ip


def test_dns_rebinding_host_is_remote_despite_loopback_peer():
    from main import _is_local_bootstrap_request
    for host in ("attacker.example", "192.168.1.10", None):
        assert _is_local_bootstrap_request(_request("127.0.0.1", host)) is False, host


def test_unparseable_request_host_fails_safe():
    """A Host that makes ``request.url.hostname`` raise must fall to remote."""
    from main import _is_local_bootstrap_request

    class _RaisingURL:
        @property
        def hostname(self):
            raise ValueError("malformed host")

    request = SimpleNamespace(client = SimpleNamespace(host = "127.0.0.1", port = 0), headers = {}, url = _RaisingURL())
    assert _is_local_bootstrap_request(request) is False
