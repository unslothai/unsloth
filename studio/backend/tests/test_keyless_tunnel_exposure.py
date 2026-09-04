# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Why a published tunnel closes keyless on loopback but not on the LAN listener.

The asymmetry looks like an oversight and is not, so it is worth a test: two
readers in a row have taken it for one. A tunnel makes a loopback peer ambiguous,
because ``CloudflareTunnel`` targets ``http://localhost:<port>`` on purpose, so
tunnelled internet traffic arrives on loopback and is indistinguishable from a
real local client. Nothing makes a LAN peer ambiguous the same way: the managed
tunnel never presents a LAN address, so the LAN branch is still deciding on
authoritative socket state.

Closing LAN as well would cost a legitimate LAN client its access for no gain,
and it would not touch the case it looks like it addresses either. An
externally run cloudflared or ngrok sets neither ``app_state.cloudflare_url``
nor ``_remote_connector_active``, so Studio cannot see it at all.
"""

from __future__ import annotations

import secrets
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from auth import storage
from utils import host_policy
from utils.keyless_api_access import (
    _reset_scope_cache,
    keyless_request_allowed,
    set_keyless_api_access,
)


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
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


@pytest.fixture(autouse = True)
def live_lan_listener(monkeypatch):
    import lan_access
    monkeypatch.setattr(
        lan_access,
        "lan_listener_status",
        lambda: {"running": True, "port": 8888, "addresses": ["192.168.1.24"], "error": None},
    )


LAN = {"server": ("192.168.1.24", 8888), "client": ("192.168.1.90", 51000)}
LOOPBACK = {"server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 51000)}


def request_for(endpoints, *, cloudflare_url = None):
    state = SimpleNamespace(
        bind_host = "0.0.0.0",
        secure = False,
        remote_access_is_colab = False,
        lan_access_is_colab = False,
        lan_access_secure_launch = False,
        cloudflare_url = cloudflare_url,
    )
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "root_path": "",
            "query_string": b"",
            "scheme": "http",
            "headers": [],
            "server": endpoints["server"],
            "client": endpoints["client"],
            "app": SimpleNamespace(state = state),
        }
    )


def test_a_published_tunnel_closes_loopback_and_leaves_the_lan_listener_alone(monkeypatch):
    set_keyless_api_access("inference")
    assert keyless_request_allowed(request_for(LOOPBACK)) is True
    assert keyless_request_allowed(request_for(LAN)) is True

    for signal in ("cloudflare_url", "connector"):
        if signal == "cloudflare_url":
            loopback = request_for(LOOPBACK, cloudflare_url = "https://demo.trycloudflare.com")
            lan = request_for(LAN, cloudflare_url = "https://demo.trycloudflare.com")
        else:
            monkeypatch.setattr(host_policy, "_remote_connector_active", True, raising = False)
            loopback, lan = request_for(LOOPBACK), request_for(LAN)
        # loopback is ambiguous while a tunnel terminates on it
        assert keyless_request_allowed(loopback) is False, signal
        # the LAN socket is not: the managed tunnel never arrives with a LAN address
        assert keyless_request_allowed(lan) is True, signal
        monkeypatch.setattr(host_policy, "_remote_connector_active", False, raising = False)


def test_full_scope_ignores_the_lan_listener_whatever_the_tunnel_state():
    """`full` is loopback-only, so the asymmetry above cannot widen it."""
    set_keyless_api_access("full")
    assert keyless_request_allowed(request_for(LAN)) is False
    assert (
        keyless_request_allowed(request_for(LAN, cloudflare_url = "https://demo.trycloudflare.com"))
        is False
    )
    assert (
        keyless_request_allowed(
            request_for(LOOPBACK, cloudflare_url = "https://demo.trycloudflare.com")
        )
        is False
    )
