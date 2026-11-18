# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel connections to one OAuth MCP server must share one refresh and keep a live refresh token."""

import asyncio
import json
import threading
import time
from urllib.parse import parse_qs

from mcp.client.auth import oauth2
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from core.inference import mcp_client

URL = "https://mcp.example.com/mcp"
http = getattr(oauth2, "httpx2", None) or oauth2.httpx


class _RotatingServer:
    """Notion's token endpoint: a refresh accepts the current or the previous token and rotates it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = "r0"
        self.previous = None
        self.access_tokens = set()
        self.refreshes = 0

    def handle(self, request):
        if request.url.path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            with self.lock:
                if form.get("refresh_token") not in (self.current, self.previous):
                    return http.Response(400, json = {"error": "invalid_grant"}, request = request)
                self.refreshes += 1
                self.previous, self.current = form["refresh_token"], f"r{self.refreshes}"
                access = f"a{self.refreshes}"
                self.access_tokens.add(access)
                body = {
                    "access_token": access,
                    "refresh_token": self.current,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            time.sleep(0.2)
            return http.Response(200, content = json.dumps(body).encode(), request = request)
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        return http.Response(200 if token in self.access_tokens else 401, request = request)


async def _connect(server):
    auth = mcp_client._oauth(URL)
    gen = auth.async_auth_flow(http.Request("POST", URL))
    response = None
    try:
        while True:
            request = await gen.asend(response)
            response = server.handle(request)
            if request.url.path == "/mcp":
                return response.status_code
    finally:
        await gen.aclose()


def _run_parallel_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)

    async def seed():
        adapter = mcp_client._oauth(URL).token_storage_adapter
        await adapter.set_client_info(
            OAuthClientInformationFull(
                client_id = "client",
                redirect_uris = ["http://localhost:1/callback"],
                token_endpoint_auth_method = "none",
            )
        )
        await adapter.set_tokens(OAuthToken(access_token = "stale", refresh_token = "r0", expires_in = 1))

    asyncio.run(seed())
    time.sleep(1.1)

    server = _RotatingServer()
    statuses = []

    def worker():
        statuses.append(asyncio.run(_connect(server)))

    threads = [threading.Thread(target = worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert statuses == [200, 200, 200]
    assert server.refreshes == 1
    stored = asyncio.run(mcp_client._oauth(URL).token_storage_adapter.get_tokens())
    assert stored.refresh_token in (server.current, server.previous)


def test_parallel_connections_refresh_once(tmp_path, monkeypatch):
    _run_parallel_refresh(tmp_path, monkeypatch)


def test_racing_store_init_still_refreshes_once(tmp_path, monkeypatch):
    original = mcp_client._oauth_store

    def fresh_store():
        mcp_client._oauth_token_store = None
        return original()

    monkeypatch.setattr(mcp_client, "_oauth_store", fresh_store)
    _run_parallel_refresh(tmp_path, monkeypatch)
