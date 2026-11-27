# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A same-host reverse proxy reaches the backend from loopback. When it marks the request as relayed,
keyless `full` and `inference` must not read that as the local browser. With nginx's default `proxy_pass` it
names nobody and Host is the upstream address, which this check cannot tell from a direct client."""

import types

import pytest
from starlette.requests import Request

from utils import keyless_api_access as kl


def _request(
    peer,
    host,
    headers = (),
):
    raw = [(b"host", host.encode())] + [(k.encode(), v.encode()) for k, v in headers]
    state = types.SimpleNamespace(bind_host = "127.0.0.1", secure = False, cloudflare_url = None)
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/accounts",
            "root_path": "",
            "headers": raw,
            "client": (peer, 50000),
            "server": ("127.0.0.1", 8888),
            "scheme": "http",
            "query_string": b"",
            "app": types.SimpleNamespace(state = state),
        }
    )


_RELAY_HEADERS = [
    ("x-forwarded-for", "203.0.113.9"),
    ("x-real-ip", "203.0.113.9"),
    ("forwarded", "for=203.0.113.9"),
    ("x-forwarded-host", "studio.example.com"),
    ("via", "1.1 proxy"),
    ("x-forwarded-proto", "https"),
    ("x-forwarded-server", "proxy.example.com"),
]


@pytest.mark.parametrize("scope", ["full", "inference"])
@pytest.mark.parametrize("header", _RELAY_HEADERS)
def test_keyless_refuses_a_loopback_request_relayed_by_a_proxy(header, scope):
    assert (
        kl.keyless_transport_allowed(_request("127.0.0.1", "127.0.0.1:8888", [header]), scope)
        is False
    )


@pytest.mark.parametrize("scope", ["full", "inference"])
def test_keyless_still_serves_a_direct_loopback_client(scope):
    assert kl.keyless_transport_allowed(_request("127.0.0.1", "127.0.0.1:8888"), scope) is True
