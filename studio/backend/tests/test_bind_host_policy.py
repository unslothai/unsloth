# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import socket

import pytest

from utils import host_policy
from utils.host_policy import (
    is_wildcard_host,
    normalize_wildcard_bind_host,
    resolved_bind_address_count,
    wildcard_ip_versions,
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
        ("::ffff:127.0.0.1", "127.0.0.1"),
        ("192.168.1.24", "192.168.1.24"),
    ],
)
def test_effective_wildcards_are_normalized_before_binding(host, expected):
    assert normalize_wildcard_bind_host(host) == expected


def test_a_mapped_wildcard_is_bindable_through_asyncio_after_normalization():
    async def bind():
        server = await asyncio.start_server(
            lambda _reader, _writer: None,
            host = normalize_wildcard_bind_host("::ffff:0.0.0.0"),
            port = 0,
        )
        try:
            return server.sockets[0].family
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(bind()) == socket.AF_INET


def test_a_mapped_specific_bind_is_bindable_through_asyncio_after_normalization():
    async def bind():
        server = await asyncio.start_server(
            lambda _reader, _writer: None,
            host = normalize_wildcard_bind_host("::ffff:127.0.0.1"),
            port = 0,
        )
        try:
            return server.sockets[0].family
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(bind()) == socket.AF_INET


def test_a_resolved_mapped_bind_is_bindable_through_asyncio_after_normalization(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            host_policy.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0))
            ],
        )
        normalized_host = normalize_wildcard_bind_host("mapped.test")

    async def bind():
        server = await asyncio.start_server(
            lambda _reader, _writer: None,
            host = normalized_host,
            port = 0,
        )
        try:
            return server.sockets[0].family
        finally:
            server.close()
            await server.wait_closed()

    assert normalized_host == "127.0.0.1"
    assert asyncio.run(bind()) == socket.AF_INET


def test_ambiguous_resolved_mapped_binds_are_rejected(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:192.168.1.24", 0, 0, 0)),
        ],
    )

    with pytest.raises(ValueError, match = "resolves to ambiguous IPv4-mapped addresses"):
        normalize_wildcard_bind_host("ambiguous-mapped.test")


def test_a_resolved_ipv6_wildcard_uses_ipv6_loopback(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0))],
    )

    assert is_wildcard_host("wildcard.test") is True
    assert normalize_wildcard_bind_host("wildcard.test") == "::"
    assert wildcard_ip_versions("wildcard.test") == (6,)
    assert wildcard_loopback_host("wildcard.test") == "::1"


def test_a_dual_stack_wildcard_hostname_keeps_both_address_families(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0)),
        ],
    )

    assert is_wildcard_host("dual-wildcard.test") is True
    assert normalize_wildcard_bind_host("dual-wildcard.test") == "dual-wildcard.test"
    assert resolved_bind_address_count("dual-wildcard.test") == 2
    assert wildcard_ip_versions("dual-wildcard.test") == (4, 6)
    assert wildcard_loopback_host("dual-wildcard.test") == "127.0.0.1"


def test_a_mixed_family_wildcard_hostname_is_rejected(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::24", 0, 0, 0)),
        ],
    )

    with pytest.raises(ValueError, match = "mixes wildcard and specific address families"):
        normalize_wildcard_bind_host("mixed-wildcard.test")


def test_scoped_ipv6_endpoints_count_as_distinct_binds(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 0, 0, 2)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", 0, 0, 3)),
        ],
    )

    assert resolved_bind_address_count("scoped.test") == 2


def test_run_server_rejects_an_empty_bind_before_startup():
    from run import run_server
    with pytest.raises(SystemExit, match = "--host cannot be empty"):
        run_server(host = "")


def test_run_server_rejects_a_mixed_family_wildcard_before_startup(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::24", 0, 0, 0)),
        ],
    )
    from run import run_server
    with pytest.raises(SystemExit, match = "mixes wildcard and specific address families"):
        run_server(host = "mixed-wildcard.test")


def test_run_server_rejects_an_ephemeral_multi_address_bind(monkeypatch):
    monkeypatch.setattr(
        host_policy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0)),
        ],
    )
    from run import run_server
    with pytest.raises(SystemExit, match = "--port 0 cannot be used"):
        run_server(host = "dual-wildcard.test", port = 0)
