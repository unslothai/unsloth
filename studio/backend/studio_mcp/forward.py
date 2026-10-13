# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools reach Unsloth Studio through its own HTTP routes, in-process, as the API-key caller. Calling the route functions directly would skip the auth dependency, so the call would run as the owner and see host paths; going through the app means each call is authenticated, account-scoped and redacted exactly as the same request from the agent would be.

The in-process client is deliberately remote: a TEST-NET peer and an ``.invalid`` Host fail every loopback, LAN and keyless check, so a forwarded call can never use keyless access, whatever Unsloth Studio's keyless scope. Only the caller's key and, where a route reads it, the Hub token header are sent; nothing from the inbound request is copied."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx

from studio_mcp.caller import Caller

# RFC 5737 TEST-NET-1 and an RFC 6761 .invalid name: neither can be loopback, private or resolvable.
FORWARD_CLIENT = ("192.0.2.1", 0)
FORWARD_BASE_URL = "http://unsloth-mcp.invalid"
HF_TOKEN_HEADER = "X-Unsloth-HF-Token"

# The route families MCP tools use. Anything else, notably auth, sandbox and the local-person settings, is a programming error.
ALLOWED_PREFIXES = (
    "/api/inference/",
    "/v1/",
    "/api/train/",
    "/api/models/",
    "/api/export/",
    "/api/data-recipe/",
    "/api/hub/datasets/",
    "/api/hub/gguf-variants",
    "/api/settings/embedding-model",
)


def checked_path(url: httpx.URL) -> str:
    path = url.path
    segments = path.split("/")
    if not path.startswith(ALLOWED_PREFIXES) or any(s in (".", "..") for s in segments):
        raise RuntimeError(f"Unsloth Studio MCP does not forward to {path!r}")
    return path


def _headers(caller: Caller, hub_header: bool, content_type: Optional[str]) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {caller.token}"}
    if hub_header and caller.hf_token:
        headers[HF_TOKEN_HEADER] = caller.hf_token
    if content_type:
        headers["Content-Type"] = content_type
    return headers


async def forward(
    caller: Caller,
    method: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json_body: Any = None,
    content: Optional[bytes] = None,
    content_type: Optional[str] = None,
    files: Any = None,
    data: Optional[dict[str, Any]] = None,
    hub_header: bool = False,
) -> httpx.Response:
    """Send one request to Unsloth Studio as ``caller`` and return the buffered response. ``content`` must be bytes, so httpx sets Content-Length: the upload routes answer a streamed body with 411."""
    if content is not None and not isinstance(content, (bytes, bytearray)):
        raise TypeError("forward() uploads need bytes")
    # The route binds the account ContextVar in whatever task runs it; its own task keeps that out of the tool.
    body = {"params": params, "json": json_body, "content": content, "files": files, "data": data}
    task = asyncio.create_task(
        _send(caller, method, path, body, _headers(caller, hub_header, content_type))
    )
    try:
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise


async def _send(
    caller: Caller, method: str, path: str, body: dict[str, Any], headers: dict[str, str]
) -> httpx.Response:
    transport = httpx.ASGITransport(app = caller.studio_app, client = FORWARD_CLIENT)
    async with httpx.AsyncClient(
        transport = transport, base_url = FORWARD_BASE_URL, timeout = None
    ) as client:
        request = client.build_request(method, path, headers = headers, **body)
        checked_path(request.url)
        return await client.send(request)


def parse_json(body: bytes) -> Any:
    """Route JSON. ``/load`` and ``/unload`` pad a slow answer with leading spaces to keep tunnels open."""
    return json.loads(body.strip())


def ndjson_last(body: bytes) -> Any:
    """The last JSON line of an NDJSON stream, which carries the result or the error."""
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty NDJSON body")
    return json.loads(lines[-1])
