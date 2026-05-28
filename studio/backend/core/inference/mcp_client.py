# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"

PROBE_TIMEOUT_SECONDS = 8.0
# When OAuth probes need to open a browser, wait long enough for the user to
# sign in. Matches fastmcp's default OAuth callback_timeout (300 s) + slack.
OAUTH_PROBE_TIMEOUT_SECONDS = 305.0

# A failed probe isn't cached (a recovered server must come back), but it's
# recorded so a down server isn't re-probed -- and the chat send re-hung for
# the full timeout -- on every message. Cool off for this long after a failure;
# much longer for OAuth, whose probe can hang up to OAUTH_PROBE_TIMEOUT_SECONDS,
# so that hang doesn't recur every minute.
FAILED_PROBE_COOLOFF_SECONDS = 60.0
OAUTH_FAILED_PROBE_COOLOFF_SECONDS = 300.0

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


# Discovered-tool cache, keyed by MCP server id. get_enabled_mcp_tools()
# probes a server only on a cache miss, keeping MCP discovery off the chat
# send's critical path -- tool schemas are stable within a session. The
# /refresh route warms it; a URL/header/OAuth change or a delete evicts it.
# Successful probes are cached indefinitely.
_tool_cache: dict[str, list[dict]] = {}

# server_id -> monotonic time before which a failed server must not be
# re-probed (see record_probe_failure). Cleared on a successful probe or
# eviction.
_probe_cooloff_until: dict[str, float] = {}

# MCP server fields whose change invalidates a server's discovered tools: the
# endpoint/auth used to probe it (url, headers, oauth) or whether it's used at
# all (is_enabled). A rename does not. The update route's eviction and
# get_enabled_mcp_tools' mid-probe guard both key off this so they can't drift.
TOOL_CACHE_INVALIDATING_FIELDS = frozenset(
    {"url", "headers_json", "use_oauth", "is_enabled"}
)


def get_cached_tools(server_id: str) -> Optional[list[dict]]:
    return _tool_cache.get(server_id)


def cache_tools(server_id: str, tools: list[dict]) -> None:
    _tool_cache[server_id] = tools
    _probe_cooloff_until.pop(server_id, None)


def record_probe_failure(server_id: str, use_oauth: bool = False) -> None:
    cooloff = (
        OAUTH_FAILED_PROBE_COOLOFF_SECONDS
        if use_oauth
        else FAILED_PROBE_COOLOFF_SECONDS
    )
    _probe_cooloff_until[server_id] = time.monotonic() + cooloff


def in_failure_cooloff(server_id: str) -> bool:
    return _probe_cooloff_until.get(server_id, 0.0) > time.monotonic()


def invalidate_tool_cache(server_id: Optional[str] = None) -> None:
    """Evict one server's cached tools, or every entry when server_id is None."""
    if server_id is None:
        _tool_cache.clear()
        _probe_cooloff_until.clear()
    else:
        _tool_cache.pop(server_id, None)
        _probe_cooloff_until.pop(server_id, None)


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
