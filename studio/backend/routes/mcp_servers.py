# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import uuid
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import get_current_subject
from core.inference.mcp_client import (
    OAUTH_PROBE_TIMEOUT_SECONDS,
    PROBE_TIMEOUT_SECONDS,
    TOOL_CACHE_INVALIDATING_FIELDS,
    cache_tools,
    clear_oauth_tokens_async,
    invalidate_tool_cache,
    list_tools_async,
    parse_server_headers,
)
from models.mcp_servers import (
    McpServerCreate,
    McpServerProbeResult,
    McpServerResponse,
    McpServerTestRequest,
    McpServerUpdate,
)
from storage import mcp_servers_db

logger = structlog.get_logger(__name__)

router = APIRouter()


def _validate_url(url: str) -> str:
    trimmed = (url or "").strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "url must not be empty")
    parsed = urlparse(trimmed)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code = 400,
            detail = "url must start with http:// or https://",
        )
    if not parsed.netloc:
        raise HTTPException(status_code = 400, detail = "url is missing a host")
    return trimmed


def _normalize_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Trim header names, drop empties, coerce values to str. None if nothing left."""
    if not headers:
        return None
    out: dict[str, str] = {}
    for raw_key, value in headers.items():
        key = str(raw_key).strip()
        if key:
            out[key] = str(value)
    return out or None


def _row_to_response(row: dict) -> McpServerResponse:
    return McpServerResponse(
        id = row["id"],
        display_name = row["display_name"],
        url = row["url"],
        headers = parse_server_headers(row) or {},
        is_enabled = bool(row["is_enabled"]),
        use_oauth = bool(row.get("use_oauth")),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


@router.get("/", response_model = list[McpServerResponse])
async def list_mcp_servers(
    current_subject: str = Depends(get_current_subject),
):
    return [_row_to_response(row) for row in mcp_servers_db.list_servers()]


@router.post("/", response_model = McpServerResponse, status_code = 201)
async def create_mcp_server(
    payload: McpServerCreate,
    current_subject: str = Depends(get_current_subject),
):
    display_name = (payload.display_name or "").strip()
    if not display_name:
        raise HTTPException(status_code = 400, detail = "display_name must not be empty")
    url = _validate_url(payload.url)
    headers = _normalize_headers(payload.headers)

    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = payload.is_enabled,
        use_oauth = payload.use_oauth,
    )
    return _row_to_response(mcp_servers_db.get_server(server_id))


def _changes_from_payload(payload: McpServerUpdate) -> dict:
    sent = payload.model_fields_set
    changes: dict = {}

    if "display_name" in sent:
        name = (payload.display_name or "").strip()
        if not name:
            raise HTTPException(
                status_code = 400, detail = "display_name must not be empty"
            )
        changes["display_name"] = name
    if "url" in sent:
        changes["url"] = _validate_url(payload.url or "")
    if "headers" in sent:
        headers = _normalize_headers(payload.headers)
        changes["headers_json"] = json.dumps(headers) if headers else None
    if "is_enabled" in sent:
        if payload.is_enabled is None:
            raise HTTPException(
                status_code = 400, detail = "is_enabled must be true or false"
            )
        changes["is_enabled"] = payload.is_enabled
    if "use_oauth" in sent:
        if payload.use_oauth is None:
            raise HTTPException(
                status_code = 400, detail = "use_oauth must be true or false"
            )
        changes["use_oauth"] = payload.use_oauth
    return changes


@router.put("/{server_id}", response_model = McpServerResponse)
async def update_mcp_server(
    server_id: str,
    payload: McpServerUpdate,
    current_subject: str = Depends(get_current_subject),
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    changes = _changes_from_payload(payload)
    if not changes:
        raise HTTPException(status_code = 400, detail = "No fields to update")
    # Clear persisted OAuth tokens when the URL changes or OAuth is
    # disabled; fastmcp keys tokens by URL and would otherwise let a
    # re-pointed server silently inherit the old account's credentials.
    if bool(old.get("use_oauth")) and (
        ("url" in changes and changes["url"] != old["url"])
        or changes.get("use_oauth") is False
    ):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.update_server(server_id, changes)
    # A new endpoint/auth makes cached tools wrong and disabling makes them
    # unreachable, so drop them and let the next send re-probe; a rename
    # leaves them valid.
    if changes.keys() & TOOL_CACHE_INVALIDATING_FIELDS:
        invalidate_tool_cache(server_id)
    return _row_to_response(mcp_servers_db.get_server(server_id))


@router.delete("/{server_id}", status_code = 204)
async def delete_mcp_server(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if old.get("use_oauth"):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.delete_server(server_id)
    invalidate_tool_cache(server_id)


@router.post("/{server_id}/refresh", response_model = McpServerProbeResult)
async def refresh_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
):
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")

    use_oauth = bool(server.get("use_oauth"))
    try:
        tools = await list_tools_async(
            url = server["url"],
            headers = parse_server_headers(server),
            timeout = OAUTH_PROBE_TIMEOUT_SECONDS
            if use_oauth
            else PROBE_TIMEOUT_SECONDS,
            use_oauth = use_oauth,
        )
    except Exception as exc:  # noqa: BLE001 — surface transport+timeout errors to UI
        logger.warning("MCP refresh failed", server_id = server_id, error = str(exc))
        return McpServerProbeResult(ok = False, error = str(exc))

    # Warm the chat-path cache so the next send skips re-probing.
    cache_tools(server_id, tools)
    return McpServerProbeResult(ok = True, tool_count = len(tools))


@router.post("/test", response_model = McpServerProbeResult)
async def test_mcp_server(
    payload: McpServerTestRequest,
    current_subject: str = Depends(get_current_subject),
):
    # URL/header validation must surface as 400 like create/update so the
    # frontend's create-form pre-flight gets the same error semantics as
    # the actual save call. Only catch transport/timeout errors below.
    url = _validate_url(payload.url)
    headers = _normalize_headers(payload.headers)
    try:
        tools = await list_tools_async(
            url = url,
            headers = headers,
            timeout = OAUTH_PROBE_TIMEOUT_SECONDS
            if payload.use_oauth
            else PROBE_TIMEOUT_SECONDS,
            use_oauth = payload.use_oauth,
        )
    except Exception as exc:  # noqa: BLE001
        return McpServerProbeResult(ok = False, error = str(exc))

    return McpServerProbeResult(ok = True, tool_count = len(tools))
