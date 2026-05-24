# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"

_oauth_token_store = None


def parse_server_headers(server: dict) -> Optional[dict]:
    raw = server.get("headers_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _oauth_store():
    global _oauth_token_store
    if _oauth_token_store is None:
        from key_value.aio._utils.sanitization import AlwaysHashStrategy
        from key_value.aio.stores.filetree import FileTreeStore
        from utils.paths.storage_roots import ensure_dir, studio_root

        # Hash keys/collections — fastmcp uses raw URLs like https://x.com as
        # keys and FileTreeStore would treat the "://" as nested directories.
        _oauth_token_store = FileTreeStore(
            data_directory = ensure_dir(studio_root() / "mcp-oauth-tokens"),
            key_sanitization_strategy = AlwaysHashStrategy(),
            collection_sanitization_strategy = AlwaysHashStrategy(),
        )
    return _oauth_token_store


def _client(url: str, headers: Optional[dict], use_oauth: bool = False):
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    from fastmcp.mcp_config import infer_transport_type_from_url

    auth = None
    if use_oauth:
        from fastmcp.client.auth import OAuth

        auth = OAuth(mcp_url = url, token_storage = _oauth_store())

    transport_cls = (
        SSETransport
        if infer_transport_type_from_url(url) == "sse"
        else StreamableHttpTransport
    )
    return Client(transport_cls(url = url, headers = headers or None, auth = auth))


async def list_tools_async(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 5.0,
    use_oauth: bool = False,
) -> list[dict]:
    async def _fetch() -> list[dict]:
        async with _client(url, headers, use_oauth) as client:
            tools = await client.list_tools()
        return [t.model_dump(exclude_none = True) for t in tools]

    return await asyncio.wait_for(_fetch(), timeout = timeout)


def _flatten_result(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    body = "\n".join(parts)
    if not body:
        structured = getattr(result, "structured_content", None)
        body = str(structured) if structured is not None else ""

    if getattr(result, "is_error", False):
        # "Error: " prefix triggers tool_call_parser's TOOL_ERROR_PREFIXES nudge.
        return f"Error: {body}" if body else "Error: tool returned no content"
    return body


def call_tool_sync(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout: Optional[float] = 300.0,
    use_oauth: bool = False,
) -> str:
    async def _call() -> Any:
        async with _client(url, headers, use_oauth) as client:
            return await client.call_tool(name, args)

    try:
        result = asyncio.run(asyncio.wait_for(_call(), timeout = timeout))
    except asyncio.TimeoutError:
        return f"Error: MCP tool '{name}' timed out after {timeout:g}s"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result)
