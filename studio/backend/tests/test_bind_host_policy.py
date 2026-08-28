# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import socket

import pytest

from utils import host_policy
from utils.host_policy import (
    is_wildcard_host,
    normalize_wildcard_bind_host,
    wildcard_loopback_host,
)


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "::0",
        "0:0:0:0:0:0:0:0",
        "0",
        "00",
        "0.0",
        "0.0.0",
        "::ffff:0.0.0.0",
        "::ffff:0:0",
    ],
)
def test_unspecified_bind_aliases_are_wildcards(host):
    assert is_wildcard_host(host) is True


@pytest.mark.parametrize("host", ["", "127.0.0.1", "localhost", "::1", "192.168.1.24", "fd00::5"])
def test_specific_bind_hosts_are_not_wildcards(host):
    assert is_wildcard_host(host) is False


@pytest.mark.parametrize(
    "host,expected",
    [
        ("0.0.0.0", "127.0.0.1"),
        ("::0", "::1"),
        ("::ffff:0.0.0.0", "127.0.0.1"),
    ],
)
def test_wildcard_loopback_matches_the_effective_address_family(host, expected):
    assert wildcard_loopback_host(host) == expected


@pytest.mark.parametrize(
    "host,expected",
    [
        ("0", "0.0.0.0"),
        ("::0", "::"),
        ("::ffff:0.0.0.0", "0.0.0.0"),
        ("192.168.1.24", "192.168.1.24"),
    ],
)
def test_effective_wildcards_are_normalized_before_binding(host, expected):
    assert normalize_wildcard_bind_host(host) == expected


def test_a_resolved_ipv6_wildcard_uses_ipv6_loopback(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0))],
    )

    assert is_wildcard_host("wildcard.test") is True
    assert normalize_wildcard_bind_host("wildcard.test") == "::"
    assert wildcard_loopback_host("wildcard.test") == "::1"


def test_run_server_rejects_an_empty_bind_before_startup():
    from run import run_server
    with pytest.raises(SystemExit, match = "--host cannot be empty"):
        run_server(host = "")
