# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for bootstrap password exposure to remote clients."""

from types import SimpleNamespace


def _request(
    client_host,
    request_host = "127.0.0.1",
    headers = None,
):
    """Build a minimal request; ``None`` models an unresolved peer / absent Host."""
    client = None if client_host is None else SimpleNamespace(host = client_host, port = 0)
    hdrs = {}
    if request_host is not None:
        hdrs["host"] = request_host
    hdrs.update(headers or {})
    return SimpleNamespace(
        client = client, headers = hdrs, url = SimpleNamespace(hostname = request_host)
    )


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

    # ::1%eth0 is a scope-id'd address, which ipaddress treats as loopback on
    # 3.9+; it must not count as a direct local peer.
    for host in ("192.168.1.10", "::ffff:192.168.1.10", "::1%eth0"):
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

    request = SimpleNamespace(
        client = SimpleNamespace(host = "127.0.0.1", port = 0), headers = {}, url = _RaisingURL()
    )
    assert _is_local_bootstrap_request(request) is False


def test_reverse_proxy_forwarded_headers_are_remote():
    """A loopback proxy relaying a remote client (non-Cloudflare headers) is remote."""
    from main import _is_local_bootstrap_request
    for header in ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip"):
        request = _request("127.0.0.1", "localhost", headers = {header: "203.0.113.7"})
        assert _is_local_bootstrap_request(request) is False, header


def test_malformed_or_absent_host_is_remote():
    """A malformed/absent/scope-id Host must not fall back to the loopback server address."""
    from main import _is_local_bootstrap_request

    # incl. bracket smuggling: [::1]evil / unclosed [::1 must not reduce to ::1
    for host in (
        "e_vil",
        "[malformed",
        "",
        None,
        "[::1%25eth0]:8888",
        "[::1]attacker",
        "[::1]evil.com",
        "[::1",
        "[::1]x",
    ):
        assert _is_local_bootstrap_request(_request("127.0.0.1", host)) is False, host


def test_colab_allows_notebook_proxy_but_not_shareable_tunnel(monkeypatch):
    """Colab autofills its single-user proxy, but not a public Cloudflare link."""
    import main

    monkeypatch.setattr(main, "_IS_COLAB", True)
    # In-notebook proxy: same-origin, no tunnel header, injects off-loopback too.
    assert main._should_inject_bootstrap(_request("10.0.0.2", "colab.proxy")) is True
    # Shareable Cloudflare link marks visitors with cf-connecting-ip; withhold.
    tunnel = _request(
        "127.0.0.1", "localhost", headers = {"cf-connecting-ip": "203.0.113.7"}
    )
    assert main._should_inject_bootstrap(tunnel) is False


def test_non_colab_gate_requires_local_client(monkeypatch):
    """Outside Colab the gate injects only for a direct loopback client."""
    import main

    monkeypatch.setattr(main, "_IS_COLAB", False)
    assert main._should_inject_bootstrap(_request("127.0.0.1", "localhost")) is True
    assert main._should_inject_bootstrap(_request("192.168.1.10", "localhost")) is False
