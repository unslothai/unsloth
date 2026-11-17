# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel connections to one OAuth MCP server must share one refresh, not race a rotating token."""

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
    def __init__(self):
        self.lock = threading.Lock()
        self.refresh_token = "r0"
        self.access_token = None
        self.refreshes = 0

    def handle(self, request):
        if request.url.path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            with self.lock:
                if form.get("refresh_token") != self.refresh_token:
                    return http.Response(400, json = {"error": "invalid_grant"}, request = request)
                self.refreshes += 1
                self.refresh_token = f"r{self.refreshes}"
                self.access_token = f"a{self.refreshes}"
                body = {
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }
            time.sleep(0.2)
            return http.Response(200, content = json.dumps(body).encode(), request = request)
        ok = request.headers.get("Authorization") == f"Bearer {self.access_token}"
        return http.Response(200 if ok else 401, request = request)


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


def test_parallel_connections_refresh_once(tmp_path, monkeypatch):
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
