# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A public tunnel closes keyless access on every transport, not just loopback."""

from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from auth import storage
from utils import host_policy
from utils.keyless_api_access import _reset_scope_cache, keyless_request_allowed, set_keyless_api_access


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    monkeypatch.setattr(host_policy, "_remote_connector_active", False, raising = False)
    monkeypatch.setattr(host_policy, "_lan_connector_active", True, raising = False)
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME, password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


LAN = {"server": ("192.168.1.24", 8888), "client": ("192.168.1.90", 51000)}
LOOPBACK = {"server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 51000)}


def request_for(endpoints, *, cloudflare_url = None):
    state = SimpleNamespace(
        bind_host = "0.0.0.0", secure = False, remote_access_is_colab = False,
        lan_access_is_colab = False, lan_access_secure_launch = False,
        cloudflare_url = cloudflare_url,
    )
    return Request({
        "type": "http", "method": "POST", "path": "/v1/chat/completions", "root_path": "",
        "query_string": b"", "scheme": "http", "headers": [],
        "server": endpoints["server"], "client": endpoints["client"],
        "app": SimpleNamespace(state = state),
    })


@pytest.fixture(autouse = True)
def live_lan_listener(monkeypatch):
    import lan_access
    monkeypatch.setattr(
        lan_access, "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"], "error": None},
    )


def test_a_published_tunnel_closes_the_lan_listener_too(monkeypatch):
    """The settings card promises keyless stops while a tunnel is up. Hold it to that.

    Checked after the LAN branch the exposure rule only reached loopback, so the LAN
    listener kept serving keyless with a tunnel published. That is safe for the tunnel
    Studio manages, which targets ``http://localhost:<port>`` and so arrives as a
    loopback peer, but not for a cloudflared or ngrok an admin points at the LAN
    address themselves: there the peer is the LAN interface and admission held.
    """
    set_keyless_api_access("inference")
    assert keyless_request_allowed(request_for(LAN)) is True

    for endpoints in (LAN, LOOPBACK):
        assert keyless_request_allowed(
            request_for(endpoints, cloudflare_url = "https://demo.trycloudflare.com")
        ) is False
        monkeypatch.setattr(host_policy, "_remote_connector_active", True, raising = False)
        assert keyless_request_allowed(request_for(endpoints)) is False
        monkeypatch.setattr(host_policy, "_remote_connector_active", False, raising = False)

    # stopping the tunnel restores it, so this is exposure and not a latch
    assert keyless_request_allowed(request_for(LAN)) is True


def test_full_scope_keeps_refusing_a_tunnel_on_every_transport(monkeypatch):
    set_keyless_api_access("full")
    for endpoints in (LAN, LOOPBACK):
        assert keyless_request_allowed(
            request_for(endpoints, cloudflare_url = "https://demo.trycloudflare.com")
        ) is False
    # and full was never honoured over LAN regardless
    assert keyless_request_allowed(request_for(LAN)) is False


def test_an_unreadable_tunnel_state_fails_closed_on_lan(monkeypatch):
    """`_public_tunnel_active` reports True when it cannot tell. That must reach LAN now."""
    def explode():
        raise RuntimeError("connector registry unavailable")

    monkeypatch.setattr(host_policy, "tunnel_connector_active", explode, raising = False)
    set_keyless_api_access("inference")
    assert keyless_request_allowed(request_for(LAN)) is False
    assert keyless_request_allowed(request_for(LOOPBACK)) is False
