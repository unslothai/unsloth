# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A later connection must refresh at the server's advertised token endpoint, not at <origin>/token."""

import asyncio
import time

from mcp.client.auth import oauth2
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken

from core.inference import mcp_client

URL = "https://mcp.example.com/mcp"
TOKEN_ENDPOINT = "https://mcp.example.com/api/auth/oauth2/token"
http = getattr(oauth2, "httpx2", None) or oauth2.httpx


def _metadata():
    return OAuthMetadata(
        issuer = "https://mcp.example.com/api/auth",
        authorization_endpoint = "https://mcp.example.com/api/auth/oauth2/authorize",
        token_endpoint = TOKEN_ENDPOINT,
    )


async def _sign_in():
    auth = mcp_client._oauth(URL)
    await auth.token_storage_adapter.set_client_info(
        OAuthClientInformationFull(
            client_id = "client",
            redirect_uris = ["http://localhost:1/callback"],
            token_endpoint_auth_method = "none",
        )
    )
    auth.context.oauth_metadata = _metadata()
    await auth.token_storage_adapter.set_tokens(
        OAuthToken(access_token = "old", refresh_token = "r0", expires_in = 1)
    )


async def _first_request_of_next_connection():
    gen = mcp_client._oauth(URL).async_auth_flow(http.Request("POST", URL))
    try:
        return await gen.asend(None)
    finally:
        await gen.aclose()


def _studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)


def test_refresh_uses_the_discovered_token_endpoint(tmp_path, monkeypatch):
    _studio(tmp_path, monkeypatch)
    asyncio.run(_sign_in())
    time.sleep(1.1)

    request = asyncio.run(_first_request_of_next_connection())

    assert str(request.url) == TOKEN_ENDPOINT
    assert b"grant_type=refresh_token" in request.content


def test_clearing_tokens_forgets_the_endpoint(tmp_path, monkeypatch):
    _studio(tmp_path, monkeypatch)
    asyncio.run(_sign_in())
    asyncio.run(mcp_client.clear_oauth_tokens_async(URL))

    async def loaded():
        auth = mcp_client._oauth(URL)
        await auth._initialize()
        return auth.context.oauth_metadata

    assert asyncio.run(loaded()) is None
