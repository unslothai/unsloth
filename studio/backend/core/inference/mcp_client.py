# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"

_oauth_token_store = None


def is_stdio(address: str) -> bool:
    """A non-HTTP address is a local stdio command, e.g.
    'npx -y @modelcontextprotocol/server-filesystem /path'."""
    return not address.strip().lower().startswith(("http://", "https://"))


def parse_stdio_command(address: str) -> list[str]:
    """Split a stdio command line into argv. Shared by route validation and the
    transport so both agree on quoting (notably Windows backslash paths)."""
    return shlex.split(address, posix = sys.platform != "win32")


def stdio_mcp_enabled() -> bool:
    """stdio MCP servers spawn local processes as the backend user (and bypass
    the python/terminal sandbox), so they are only allowed when the backend
    host is the user's own machine. The Tauri desktop app sets
    UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 (see main.py); advanced localhost /
    self-hosted users can opt in with the same variable. It stays off for
    Colab and any network (0.0.0.0) bind."""
    return os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") == "1"


# Probe timeouts for discovering a server's tool list. OAuth needs minutes for
# first-connect/expired-token browser sign-in; stdio allows for first-run
# package download (e.g. `npx -y ...`); HTTP fails fast.
_HTTP_PROBE_TIMEOUT = 8.0
_OAUTH_PROBE_TIMEOUT = 305.0
_STDIO_PROBE_TIMEOUT = 60.0


def probe_timeout(address: str, use_oauth: bool) -> float:
    if use_oauth:
        return _OAUTH_PROBE_TIMEOUT
    return _STDIO_PROBE_TIMEOUT if is_stdio(address) else _HTTP_PROBE_TIMEOUT


def parse_server_headers(server: dict) -> Optional[dict]:
    """Parsed headers_json. For stdio servers this dict is the process
    environment instead of HTTP headers (see _client)."""
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


async def clear_oauth_tokens_async(url: str) -> None:
    """Drop any persisted OAuth tokens for ``url``. fastmcp keys tokens by
    MCP URL, so on server delete / URL change / OAuth disable we have to
    clear the old credentials explicitly. Otherwise re-registering the
    same URL would silently reuse the old account's token. The entire
    body runs inside the protected block -- store / OAuth construction
    failing must not make the delete / update route 500."""
    try:
        from fastmcp.client.auth import OAuth

        auth = OAuth(mcp_url = url, token_storage = _oauth_store())
        await auth.token_storage_adapter.clear()
    except Exception as exc:  # noqa: BLE001
        # Cleanup is best-effort; the row delete still wins.
        logger.warning("Failed to clear OAuth tokens for %s: %s", url, exc)


def _client(url: str, headers: Optional[dict], use_oauth: bool = False):
    from fastmcp import Client

    if is_stdio(url):
        from fastmcp.client.transports import StdioTransport

        parts = parse_stdio_command(url)
        # stdio env vars ride the (HTTP-only) headers field. The MCP SDK merges
        # them over its default safe env (PATH etc.), so pass them through as-is.
        return Client(
            StdioTransport(command = parts[0], args = parts[1:], env = headers or None)
        )

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
    cancel_event = None,
) -> str:
    """Synchronously call an MCP tool.

    ``cancel_event``: optional ``threading.Event``. When set, the in-flight
    HTTP call is cancelled and the function returns a cancellation Error.
    Polled in parallel with the tool call via ``asyncio.wait`` so a /cancel
    POST from the UI interrupts even mid-network-read.
    """

    async def _call() -> Any:
        async with _client(url, headers, use_oauth) as client:
            return await client.call_tool(name, args)

    async def _watch_cancel() -> None:
        # 50 ms cadence keeps cancellation responsive without busy-looping;
        # matches the cadence routes/inference.py uses for cancel watchers.
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.05)

    async def _race() -> Any:
        # Check cancellation before spawning the call task so a pre-set
        # event short-circuits before opening the transport / HTTP
        # connection (reviewer-reproduced race).
        if cancel_event is not None and cancel_event.is_set():
            raise _MCPCancelled
        call_task = asyncio.create_task(_call())
        if cancel_event is None:
            return await asyncio.wait_for(call_task, timeout = timeout)
        watch_task = asyncio.create_task(_watch_cancel())
        try:
            done, pending = await asyncio.wait(
                {call_task, watch_task},
                timeout = timeout,
                return_when = asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (call_task, watch_task):
                if not t.done():
                    t.cancel()
        if not done:
            raise asyncio.TimeoutError
        if call_task in done:
            return call_task.result()
        raise _MCPCancelled

    try:
        result = asyncio.run(_race())
    except _MCPCancelled:
        return f"Error: MCP tool '{name}' cancelled"
    except asyncio.TimeoutError:
        return f"Error: MCP tool '{name}' timed out after {timeout:g}s"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result)


class _MCPCancelled(Exception):
    """Internal sentinel raised when cancel_event fires before the tool returns."""
