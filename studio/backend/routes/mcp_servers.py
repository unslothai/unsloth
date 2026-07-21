# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import uuid
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import get_current_subject
from core.inference.mcp_client import (
    TOOL_CACHE_INVALIDATING_FIELDS,
    cache_tools,
    clear_oauth_tokens_async,
    close_stdio_sessions,
    invalidate_tool_cache,
    is_stdio,
    list_tools_async,
    parse_server_headers,
    parse_stdio_command,
    probe_timeout,
    record_probe_failure,
    stdio_mcp_enabled,
)
from core.inference.mcp_config_import import parse_mcp_config
from models.mcp_servers import (
    McpServerCreate,
    McpServerImportRequest,
    McpServerImportResult,
    McpServerProbeResult,
    McpServerResponse,
    McpServerTestRequest,
    McpServerUpdate,
)
from storage import mcp_servers_db
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)

router = APIRouter()


def _looks_like_command(value: str) -> bool:
    """Whitespace is a one-way signal: a URL can't hold an unencoded space, so
    a value with whitespace is definitely a command. No whitespace proves
    nothing (a lone token may be a single-arg command or a scheme-less URL)."""
    return any(ch.isspace() for ch in value)


def _validate_url(url: str) -> str:
    trimmed = (url or "").strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "url must not be empty")
    # When stdio is enabled, a non-HTTP value is a local command (reuses this
    # field so stdio servers ride existing CRUD/storage).
    if stdio_mcp_enabled() and is_stdio(trimmed):
        try:
            parts = parse_stdio_command(trimmed)
        except ValueError as exc:
            raise log_and_http_error(
                exc,
                400,
                "Invalid command. Check quoting and try again.",
                event = "mcp_servers.invalid_command",
                log = logger,
            )
        if not parts or not parts[0].strip():
            raise HTTPException(status_code = 400, detail = "command must not be empty")
        if "://" in parts[0]:
            # A URL-scheme first token is a mistyped URL, not a command. Reject
            # cleanly instead of exec-ing it (mirrors the frontend check).
            raise HTTPException(
                status_code = 400,
                detail = "Enter an http(s):// URL, or a local command whose "
                "first token is an executable (not a URL).",
            )
        return trimmed
    parsed = urlparse(trimmed)
    if parsed.scheme not in ("http", "https"):
        if _looks_like_command(trimmed):
            detail = (
                "Local commands aren't enabled on this server. To allow them, "
                "set UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 and restart Unsloth, or use "
                "an http:// or https:// URL instead."
            )
        else:
            detail = (
                "MCP server address must start with http:// or https:// "
                "(for example https://example.com/mcp)."
            )
        raise HTTPException(status_code = 400, detail = detail)
    if not parsed.netloc:
        raise HTTPException(status_code = 400, detail = "url is missing a host")
    return trimmed


def _normalize_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Trim header names, drop empties, coerce values to str; None if empty."""
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
async def list_mcp_servers(current_subject: str = Depends(get_current_subject)):
    return [_row_to_response(row) for row in mcp_servers_db.list_servers()]


@router.post("/", response_model = McpServerResponse, status_code = 201)
async def create_mcp_server(
    payload: McpServerCreate, current_subject: str = Depends(get_current_subject)
):
    display_name = (payload.display_name or "").strip()
    if not display_name:
        raise HTTPException(status_code = 400, detail = "display_name must not be empty")
    url = _validate_url(payload.url)
    headers = _normalize_headers(payload.headers)
    # OAuth is HTTP-only; force it off for stdio commands so a stale flag can't
    # push the probe onto the 305s OAuth timeout. Backend enforces this.
    use_oauth = payload.use_oauth and not is_stdio(url)

    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = payload.is_enabled,
        use_oauth = use_oauth,
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
    # stdio is OAuth-less: drop a stale OAuth flag when switching to a command.
    if "url" in changes and is_stdio(changes["url"]):
        changes["use_oauth"] = False
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
    # headers == HTTP headers (remote) or env vars (stdio). On a transport-type
    # switch with no new headers, drop the old ones so env secrets aren't
    # re-sent as HTTP headers (or vice versa).
    if (
        "url" in changes
        and is_stdio(changes["url"]) != is_stdio(old["url"])
        and "headers_json" not in changes
    ):
        changes["headers_json"] = None
    # Clear persisted OAuth tokens when the URL changes or OAuth is disabled;
    # fastmcp keys tokens by URL and would otherwise let a re-pointed server
    # silently inherit the old account's credentials.
    if bool(old.get("use_oauth")) and (
        ("url" in changes and changes["url"] != old["url"])
        or changes.get("use_oauth") is False
    ):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.update_server(server_id, changes)
    # A new endpoint/auth makes cached tools wrong and disabling makes them unreachable, so drop
    # them and let the next send re-probe; a rename leaves them valid. Live stdio sessions for the
    # old endpoint close too. Gate on a real value change, not mere presence: the edit dialog
    # resends url/headers/oauth unchanged on a rename, which must not drop the session.
    if any(
        changes[k] != old.get(k)
        for k in changes.keys() & TOOL_CACHE_INVALIDATING_FIELDS
    ):
        invalidate_tool_cache(server_id)
        # Narrow to this row's env: another server row sharing the command but
        # with a different env keeps its live sessions.
        await asyncio.to_thread(
            close_stdio_sessions, old["url"], parse_server_headers(old)
        )
    return _row_to_response(mcp_servers_db.get_server(server_id))


@router.delete("/{server_id}", status_code = 204)
async def delete_mcp_server(
    server_id: str, current_subject: str = Depends(get_current_subject)
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if old.get("use_oauth"):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.delete_server(server_id)
    invalidate_tool_cache(server_id)
    await asyncio.to_thread(close_stdio_sessions, old["url"], parse_server_headers(old))


@router.post("/{server_id}/refresh", response_model = McpServerProbeResult)
async def refresh_mcp_server_tools(
    server_id: str, current_subject: str = Depends(get_current_subject)
):
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    # Refresh uses the stored address, so re-check the stdio gate here too: a
    # stdio row from a desktop DB must not spawn on a hosted/network host.
    if is_stdio(server["url"]) and not stdio_mcp_enabled():
        raise HTTPException(
            status_code = 400, detail = "stdio MCP servers are disabled on this host"
        )

    use_oauth = bool(server.get("use_oauth"))
    try:
        tools = await list_tools_async(
            url = server["url"],
            headers = parse_server_headers(server),
            timeout = probe_timeout(server["url"], use_oauth),
            use_oauth = use_oauth,
        )
    except Exception as exc:  # noqa: BLE001 — surface transport+timeout errors to UI
        logger.error(
            "mcp_servers.refresh_failed",
            server_id = server_id,
            error = str(exc),
            exc_info = True,
        )
        current = mcp_servers_db.get_server(server_id)
        if current is not None and not any(
            current.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
        ):
            # Start the cool-off so the next chat send doesn't immediately re-hang
            # on this server's timeout. If the row changed while the probe was
            # awaiting, the failure belongs to the old config and must not park
            # the newly edited server.
            record_probe_failure(server_id, use_oauth)
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    # Warm the chat-path cache so the next send skips re-probing.
    current = mcp_servers_db.get_server(server_id)
    if current is not None and not any(
        current.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
    ):
        cache_tools(server_id, tools)
    return McpServerProbeResult(ok = True, tool_count = len(tools))


@router.post("/import", response_model = McpServerImportResult)
async def import_mcp_servers(
    payload: McpServerImportRequest, current_subject: str = Depends(get_current_subject)
):
    """Bulk-register servers from a standard mcpServers JSON config (issue
    #5936). Each entry rides the existing create path: _validate_url applies
    the same stdio gate (a stdio entry becomes a per-entry error when stdio is
    off; http still imports), and entries whose url already exists are skipped
    so re-importing the same file is idempotent. One bad entry never 400s the
    whole batch -- failures are reported per entry."""
    entries, errors = parse_mcp_config(payload.config)
    created: list[McpServerResponse] = []
    skipped: list[str] = []
    seen_urls = {row["url"] for row in mcp_servers_db.list_servers()}

    for entry in entries:
        try:
            url = _validate_url(entry.url)
        except HTTPException as exc:
            errors.append(f"{entry.display_name}: {exc.detail}")
            continue
        if url in seen_urls:
            skipped.append(entry.display_name)
            continue
        headers = _normalize_headers(entry.headers)
        server_id = uuid.uuid4().hex[:16]
        mcp_servers_db.create_server(
            id = server_id,
            display_name = entry.display_name,
            url = url,
            headers_json = json.dumps(headers) if headers else None,
            is_enabled = entry.is_enabled,
            use_oauth = entry.use_oauth and not is_stdio(url),
        )
        seen_urls.add(url)
        created.append(_row_to_response(mcp_servers_db.get_server(server_id)))

    return McpServerImportResult(created = created, skipped = skipped, errors = errors)


@router.post("/test", response_model = McpServerProbeResult)
async def test_mcp_server(
    payload: McpServerTestRequest, current_subject: str = Depends(get_current_subject)
):
    # URL/header validation must surface as 400 like create/update so the
    # frontend's create-form pre-flight gets the same error semantics as the
    # save call. Only catch transport/timeout errors below.
    url = _validate_url(payload.url)
    headers = _normalize_headers(payload.headers)
    try:
        tools = await list_tools_async(
            url = url,
            headers = headers,
            timeout = probe_timeout(url, payload.use_oauth),
            use_oauth = payload.use_oauth,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "mcp_servers.test_failed",
            error = str(exc),
            exc_info = True,
        )
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    return McpServerProbeResult(ok = True, tool_count = len(tools))
