# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the bootstrap-pw LAN leak on a wildcard bind.

``_is_loopback_client`` is the second half of the gate on ``_inject_bootstrap``
(alongside ``_is_same_origin_request``): on ``-H 0.0.0.0`` the seeded admin
password must not be auto-filled in-page for a non-loopback (LAN) peer.
"""

from types import SimpleNamespace


def _request(client_host):
    """A minimal request whose ``.client`` mimics Starlette's (host, port) peer.

    ``client_host is None`` models a peer uvicorn could not resolve (e.g. a
    Unix-domain-socket bind), which must fail safe.
    """
    client = None if client_host is None else SimpleNamespace(host = client_host, port = 0)
    return SimpleNamespace(client = client)


def test_loopback_peers_are_local():
    from main import _is_loopback_client
    # IPv4 loopback covers the whole 127.0.0.0/8 range, plus IPv6 ::1 and the
    # IPv4-mapped-IPv6 form a dual-stack socket reports for an IPv4 connection.
    for host in ("127.0.0.1", "127.0.0.5", "127.255.255.254", "::1", "::ffff:127.0.0.1"):
        assert _is_loopback_client(_request(host)) is True, host


def test_non_loopback_peers_are_remote():
    from main import _is_loopback_client
    for host in ("192.168.1.10", "10.0.0.5", "169.254.1.1", "0.0.0.0", "::ffff:192.168.1.10"):
        assert _is_loopback_client(_request(host)) is False, host


def test_absent_or_unparseable_peer_fails_safe():
    from main import _is_loopback_client
    for host in (None, "", "localhost", "not-an-ip", "fe80::1%eth0"):
        assert _is_loopback_client(_request(host)) is False, host
