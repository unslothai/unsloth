# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Under client_secret_basic the token request must not also carry client_id."""

import asyncio
import base64
import types
from urllib.parse import parse_qs

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from core.inference import mcp_client

URL = "https://mcp.notion.com/mcp"
CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"


def _auth(tmp_path, monkeypatch, auth_method):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    auth = mcp_client._oauth(URL)
    auth.context.client_info = OAuthClientInformationFull(
        client_id = CLIENT_ID,
        client_secret = CLIENT_SECRET,
        token_endpoint_auth_method = auth_method,
        redirect_uris = auth.context.client_metadata.redirect_uris,
    )
    return auth


def _form(request):
    return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}


def _basic_header():
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()


def test_basic_auth_omits_client_id_from_body(tmp_path, monkeypatch):
    auth = _auth(tmp_path, monkeypatch, "client_secret_basic")
    request = asyncio.run(auth._exchange_token_authorization_code("auth-code", "verifier"))
    form = _form(request)

    assert "client_id" not in form
    assert "client_secret" not in form
    assert request.headers["Authorization"] == _basic_header()
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "auth-code"
    assert form["code_verifier"] == "verifier"


def test_post_auth_keeps_client_id_in_body(tmp_path, monkeypatch):
    auth = _auth(tmp_path, monkeypatch, "client_secret_post")
    request = asyncio.run(auth._exchange_token_authorization_code("auth-code", "verifier"))
    form = _form(request)

    assert form["client_id"] == CLIENT_ID
    assert form["client_secret"] == CLIENT_SECRET
    assert "Authorization" not in request.headers


def test_basic_auth_omits_client_id_from_refresh(tmp_path, monkeypatch):
    auth = _auth(tmp_path, monkeypatch, "client_secret_basic")
    auth.context.current_tokens = OAuthToken(access_token = "at", refresh_token = "rt")
    request = asyncio.run(auth._refresh_token())
    form = _form(request)

    assert "client_id" not in form
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt"
    assert request.headers["Authorization"] == _basic_header()


def test_public_client_keeps_client_id_in_body(tmp_path, monkeypatch):
    """A public client authenticates with client_id alone, so stripping it would break it."""
    auth = _auth(tmp_path, monkeypatch, "none")
    auth.context.client_info.client_secret = None
    request = asyncio.run(auth._exchange_token_authorization_code("auth-code", "verifier"))
    form = _form(request)

    assert form["client_id"] == CLIENT_ID
    assert "Authorization" not in request.headers


def test_unpatchable_context_does_not_break_oauth():
    """If a future SDK makes the context unwritable, callers still get a working provider."""

    class Frozen:
        def prepare_token_auth(
            self,
            data,
            headers = None,
        ):
            return data, headers or {}

        def __setattr__(self, name, value):
            raise AttributeError("frozen")

    auth = types.SimpleNamespace(context = Frozen())
    mcp_client._strip_client_id_under_basic_auth(auth)
