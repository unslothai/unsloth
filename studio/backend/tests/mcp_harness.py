# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve a Studio MCP server behind the real gate in front of a fake Studio, so tool tests drive the whole path (gate, caller, forwarder) without a GPU or the real routes. Built fresh per test: at the fastmcp floor an http_app's session manager runs once per instance."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.applications import Starlette
from starlette.routing import Mount

from studio_mcp import gate
from studio_mcp.gate import StudioMcpGate

TEST_TOKEN = "sk-unsloth-test"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
SENTINEL_ROOTS = ["/srv/mcp-sentinel-host", "C:\\mcp-sentinel"]


def fake_studio(routes: dict[tuple[str, str], Callable[..., Any]]) -> FastAPI:
    """A Studio stand-in. Each handler gets ``(request, body)`` and returns a Response or JSON; every call is recorded in ``app.state.calls`` as ``(method, path, headers, body)``."""
    app = FastAPI()
    app.state.calls = []

    def bind(method: str, path: str, handler: Callable[..., Any]) -> None:
        async def endpoint(request: Request) -> Response:
            body = await request.body()
            app.state.calls.append((method, request.url.path, dict(request.headers), body))
            result = handler(request, body)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, Response) else JSONResponse(result)

        app.add_api_route(path, endpoint, methods = [method])

    for (method, path), handler in routes.items():
        bind(method, path, handler)
    return app


def _accept_test_key(token: str) -> tuple[Optional[dict], str]:
    if token == TEST_TOKEN:
        return {"account_id": "owner", "username": "unsloth", "role": "owner"}, ""
    return None, "Invalid or expired API key"


def served(
    mcp: Any,
    studio_app: Any,
    *,
    monkeypatch: Any,
    enabled: bool = True,
    validate: Callable[[str], tuple[Optional[dict], str]] = _accept_test_key,
) -> Starlette:
    monkeypatch.setattr(gate, "is_mcp_enabled", lambda: enabled)
    monkeypatch.setattr(gate, "_validate_key", validate)
    mcp_app = mcp.http_app(path = "/", stateless_http = True)
    app = Starlette(
        routes = [Mount("/mcp", StudioMcpGate(mcp_app)), Mount("/", studio_app)],
        lifespan = mcp_app.lifespan,
    )
    app.state.server_port = 8888
    app.state.cloudflare_url = None
    return app


def call_tool(
    client: Any,
    name: str,
    args: Optional[dict] = None,
    token: Optional[str] = TEST_TOKEN,
    headers: Optional[dict] = None,
) -> dict:
    """``tools/call`` over the streamable HTTP transport; returns the JSON-RPC ``result`` (or ``error``)."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    }
    sent = {**MCP_HEADERS, **(headers or {})}
    if token is not None:
        sent["Authorization"] = f"Bearer {token}"
    response = client.post("/mcp/", json = body, headers = sent)
    assert response.status_code == 200, response.text
    data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    message = json.loads(data[-1])
    return message.get("result", message.get("error"))


def poison(payload: Any) -> Any:
    """Suffix every string with both sentinel host paths, so a leak shows up wherever it comes from."""
    if isinstance(payload, str):
        return f"{payload} {SENTINEL_ROOTS[0]}/x {SENTINEL_ROOTS[1]}\\x"
    if isinstance(payload, dict):
        return {key: poison(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [poison(value) for value in payload]
    return payload
