# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behind a trusted proxy the login limiter keys on the hop the proxy appended. The leftmost
X-Forwarded-For entry is whatever the client sent, so keying on it gives every guess a fresh bucket."""

import asyncio

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from routes import auth as auth_routes


@pytest.fixture(autouse = True)
def _clean_limiter(monkeypatch):
    for table in (auth_routes._LOGIN_BUCKETS, auth_routes._LOGIN_IP_BUCKETS):
        table.clear()
    for shard in auth_routes._LOGIN_IP_OVERFLOW:
        shard.clear()
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")
    yield
    for table in (auth_routes._LOGIN_BUCKETS, auth_routes._LOGIN_IP_BUCKETS):
        table.clear()
    for shard in auth_routes._LOGIN_IP_OVERFLOW:
        shard.clear()


class _Request:
    def __init__(
        self,
        peer,
        headers = None,
    ):
        self.client = type("Client", (), {"host": peer})()
        self.headers = Headers(headers or {})


def test_a_spoofed_leftmost_forwarded_for_does_not_pick_a_fresh_bucket(monkeypatch):
    from models.auth import AuthLoginRequest

    monkeypatch.setattr(auth_routes.policy, "installation_is_multi_user", lambda: False)
    monkeypatch.setattr(auth_routes.storage, "get_user_and_secret", lambda username: None)
    monkeypatch.setattr(auth_routes.hashing, "equalize_login_work", lambda password: None)
    statuses = []
    for guess in range(auth_routes._LOGIN_MAX_FAILS + 3):
        # The proxy appends the real client; everything left of it is the caller's own text.
        request = _Request("127.0.0.1", {"x-forwarded-for": f"10.9.{guess}.1, 198.51.100.7"})
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                auth_routes.login(
                    AuthLoginRequest(username = "nobody", password = "wrong-password"), request
                )
            )
        statuses.append(exc.value.status_code)
    assert statuses[: auth_routes._LOGIN_MAX_FAILS] == [401] * auth_routes._LOGIN_MAX_FAILS
    assert statuses[-1] == 429


def test_forwarded_header_takes_the_proxy_appended_element():
    request = _Request("127.0.0.1", {"forwarded": "for=10.9.9.9, for=198.51.100.42;proto=https"})
    assert auth_routes._client_ip(request) == "198.51.100.42"


def test_a_single_proxy_hop_still_names_the_client():
    request = _Request("127.0.0.1", {"x-forwarded-for": "198.51.100.7"})
    assert auth_routes._client_ip(request) == "198.51.100.7"


@pytest.mark.parametrize("name", [b"x-forwarded-for", b"forwarded"])
def test_a_proxy_appended_header_field_is_the_one_keyed_on(name):
    def hop(addr):
        return addr.encode() if name == b"x-forwarded-for" else b"for=" + addr.encode()

    request = _Request("10.0.0.5")
    for spoofed in ("198.51.100.1", "198.51.100.2"):
        request.headers = Headers(raw = [(name, hop(spoofed)), (name, hop("203.0.113.9"))])
        assert auth_routes._client_ip(request) == "203.0.113.9"


def test_shared_client_ip_reads_a_proxy_appended_header_field(monkeypatch):
    from utils import client_ip as client_ip_mod

    monkeypatch.setattr(client_ip_mod, "_trust_forwarded_for", lambda: True)
    request = _Request("10.0.0.5")
    request.headers = Headers(
        raw = [(b"x-forwarded-for", b"198.51.100.1"), (b"x-forwarded-for", b"203.0.113.9")]
    )
    assert client_ip_mod.client_ip(request) == "203.0.113.9"
