# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel Search MCP as an opt-in web search provider.

Talks to Parallel's free authless Streamable HTTP MCP endpoint
(``https://search.parallel.ai/mcp``) with the same ``tools/call`` flow an MCP
client uses: ``initialize`` handshake, ``notifications/initialized``, then
``tools/call`` for ``web_search`` / ``web_fetch``. No auth by default; an
optional user Bearer key raises the rate limits.

Uses ``httpx`` when importable (a declared backend dependency), falling back
to stdlib ``urllib`` so a minimal install still works. Responses may be plain
JSON or SSE (``text/event-stream``); both are parsed.
"""

from __future__ import annotations

import json
import time
import urllib.request
import uuid
from typing import Any

from loggers import get_logger

logger = get_logger(__name__)

PARALLEL_SEARCH_MCP_URL = "https://search.parallel.ai/mcp"
PARALLEL_PROVIDER_ID = "parallel"
DUCK_PROVIDER_ID = "duckduckgo"

USER_AGENT = "unsloth-studio/1.0"
_MCP_PROTOCOL_VERSION = "2025-06-18"
_PARALLEL_API_KEY_MAX_LEN = 500
_CONNECT_TIMEOUT = 10.0
_SNIPPET_MAX_CHARS = 600


def web_search_provider() -> str:
    """The configured search provider id, defaulting to DuckDuckGo.

    Never raises: a settings read must never break a tool call.
    """
    try:
        from storage.studio_db import list_chat_settings
        if list_chat_settings().get("webSearchProvider") == PARALLEL_PROVIDER_ID:
            return PARALLEL_PROVIDER_ID
    except Exception:  # noqa: BLE001
        pass
    return DUCK_PROVIDER_ID


def parallel_api_key() -> str | None:
    """The user's optional Parallel Bearer key, stripped, or None.

    Never raises and never logs the key.
    """
    try:
        from storage.studio_db import list_chat_settings
        raw = list_chat_settings().get("parallelSearchApiKey")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, str):
        return None
    key = raw.strip()
    if not key:
        return None
    return key[:_PARALLEL_API_KEY_MAX_LEN]


def _headers(api_key: str | None, session_id: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    return headers


def _parse_mcp_body(text: str) -> dict[str, Any]:
    """Parse a Streamable HTTP response: plain JSON or an SSE stream."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    payload: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
    return payload


def _post_json(
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float | None,
    session_id: str | None = None,
    deadline: float | None = None,
) -> tuple[dict[str, Any], str | None]:
    """POST one JSON-RPC message; returns ``(body, session_id)``."""
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Parallel search timed out")
        timeout = remaining if timeout is None else min(timeout, remaining)
    body = json.dumps(payload).encode("utf-8")
    try:
        import httpx  # noqa: PLC0415
        with httpx.Client(timeout = timeout) as client:
            response = client.post(
                PARALLEL_SEARCH_MCP_URL,
                content = body,
                headers = _headers(api_key, session_id),
            )
            response.raise_for_status()
            return (
                _parse_mcp_body(response.text),
                response.headers.get("Mcp-Session-Id") or session_id,
            )
    except ImportError:
        pass
    request = urllib.request.Request(
        PARALLEL_SEARCH_MCP_URL,
        data = body,
        headers = _headers(api_key, session_id),
        method = "POST",
    )
    with urllib.request.urlopen(request, timeout = timeout) as response:
        raw = response.read().decode("utf-8", "replace")
        session = response.headers.get("Mcp-Session-Id") or session_id
    return _parse_mcp_body(raw), session


def _call_tool(
    tool: str,
    arguments: dict[str, Any],
    api_key: str | None,
    timeout: float | None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Full MCP round-trip for one tool call: handshake then ``tools/call``."""
    request_id = 1
    body, session = _post_json(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "unsloth-studio", "version": "1.0"},
            },
        },
        api_key,
        timeout,
        deadline = deadline,
    )
    if body.get("error"):
        raise RuntimeError(f"Parallel MCP initialize failed: {body['error']}")
    try:
        _post_json(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            api_key,
            timeout,
            session_id = session,
            deadline = deadline,
        )
    except Exception as exc:  # noqa: BLE001 - some servers 4xx the notification
        logger.debug("parallel mcp initialized notification ignored (%s)", type(exc).__name__)
    result, _ = _post_json(
        {
            "jsonrpc": "2.0",
            "id": request_id + 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
        api_key,
        timeout,
        session_id = session,
        deadline = deadline,
    )
    if result.get("error"):
        raise RuntimeError(f"Parallel {tool} failed: {result['error']}")
    inner = result.get("result")
    if not isinstance(inner, dict):
        raise RuntimeError(f"Parallel {tool} returned an unexpected response")
    if inner.get("isError"):
        texts = _content_texts(inner)
        raise RuntimeError(f"Parallel {tool} errored: {' | '.join(texts)[:300]}")
    return inner


def _content_texts(result: dict[str, Any]) -> list[str]:
    texts = []
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
    return texts


def _structured_payload(result: dict[str, Any]) -> dict[str, Any]:
    for text in _content_texts(result):
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean(text: Any, limit: int = _SNIPPET_MAX_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        raise TimeoutError("Search cancelled.")


def parallel_web_search(
    query: str,
    max_results: int = 5,
    timeout: int | float | None = 30,
    cancel_event = None,
    website_policy: dict | None = None,
    api_key: str | None = None,
) -> str:
    """Search via Parallel's MCP ``web_search``; same text format as ddgs."""
    from .web_access_policy import check_url_access, scope_search_query

    _check_cancel(cancel_event)
    deadline = None if timeout is None else time.monotonic() + timeout
    effective_query = scope_search_query(query, website_policy)
    result = _call_tool(
        "web_search",
        {
            "objective": query.strip(),
            "search_queries": [effective_query],
            "session_id": uuid.uuid4().hex,
        },
        api_key,
        _CONNECT_TIMEOUT if timeout is None else min(timeout, 60),
        deadline = deadline,
    )
    _check_cancel(cancel_event)
    payload = _structured_payload(result)
    items = payload.get("results")
    if not isinstance(items, list) or not items:
        return "No results found."
    parts = []
    for item in items:
        if len(parts) >= max_results:
            break
        if not isinstance(item, dict):
            continue
        href = str(item.get("url") or "").strip()
        allowed, _reason, _hostname = check_url_access(href, website_policy)
        if not allowed:
            continue
        title = _clean(item.get("title"))
        excerpts = item.get("excerpts")
        snippet = ""
        if isinstance(excerpts, list) and excerpts:
            snippet = _clean(" ".join(str(e) for e in excerpts if e))
        parts.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
    if not parts:
        return "No results found within the website access limits."
    text = "\n\n---\n\n".join(parts)
    text += (
        "\n\n---\n\nIMPORTANT: These are only short snippets. "
        "To get the full page content, call web_search with "
        'the url parameter (e.g. {"url": "<URL>"}).'
    )
    return text


def parallel_web_fetch(
    url: str,
    timeout: int | float | None = 30,
    cancel_event = None,
    website_policy: dict | None = None,
    api_key: str | None = None,
    objective: str | None = None,
    max_chars: int | None = None,
) -> str:
    """Fetch one page via Parallel's MCP ``web_fetch``; readable text."""
    from .web_access_policy import check_url_access

    _check_cancel(cancel_event)
    allowed, reason, _hostname = check_url_access(url, website_policy)
    if not allowed:
        return reason
    deadline = None if timeout is None else time.monotonic() + timeout
    arguments: dict[str, Any] = {"urls": [url.strip()], "session_id": uuid.uuid4().hex}
    if objective:
        arguments["objective"] = str(objective)[:200]
    result = _call_tool(
        "web_fetch",
        arguments,
        api_key,
        _CONNECT_TIMEOUT if timeout is None else min(timeout, 60),
        deadline = deadline,
    )
    _check_cancel(cancel_event)
    payload = _structured_payload(result)
    items = payload.get("results")
    texts: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            excerpts = item.get("excerpts")
            if isinstance(excerpts, list):
                texts.extend(str(e) for e in excerpts if e)
            for key in ("content", "text", "markdown"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value)
    if not texts:
        texts = _content_texts(result)
    body = "\n\n".join(t.strip() for t in texts if str(t).strip()).strip()
    if not body:
        return "(page returned no readable text)"
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars] + f"\n\n... (truncated, {len(body)} chars total)"
    return body
