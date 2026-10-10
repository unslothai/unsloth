# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import uuid
from typing import Annotated, Optional
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException
from integrations.blender import service as blender
from models.mcp_servers import BlenderSettings, BlenderSetup, McpBuiltinResponse

from auth.authentication import (
    authenticated_via_api_key,
    get_current_subject,
    request_admitted_without_credential,
    require_ui_session_for_local_commands,
)
from core.inference.mcp_client import (
    TOOL_CACHE_INVALIDATING_FIELDS,
    UI_RESOURCE_SCHEME,
    cache_tools,
    call_tool_structured_sync,
    get_cached_tools,
    in_failure_cooloff,
    clear_oauth_tokens_async,
    close_mcp_sessions,
    invalidate_tool_cache,
    is_stdio,
    join_stdio_command,
    list_tools_async,
    oauth_client_kwargs,
    parse_server_headers,
    parse_stdio_command,
    probe_timeout,
    read_resource_sync,
    record_probe_failure,
    serialize_mcp_server_mutation,
    stdio_mcp_disabled_reason,
    stdio_mcp_enabled,
    tool_visible_to,
)
from core.inference.mcp_config_import import parse_mcp_config
from core.inference.mcp_image import image_input_mappings, image_mapping
from models.mcp_servers import (
    BlenderTest,
    McpServerCreate,
    McpServerImportRequest,
    McpServerImportResult,
    McpServerProbeResult,
    McpServerResponse,
    McpServerTestRequest,
    McpServerUpdate,
    McpStdioCommand,
    McpStdioDecodeRequest,
    McpStdioEncodeResponse,
    McpUiResourceResponse,
    McpUiToolCallRequest,
    McpUiToolCallResult,
)
from storage import mcp_servers_db
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)


router = APIRouter(dependencies = [Depends(get_current_subject)])

# Annotated, not a Depends default: direct test calls would read a Depends object as truthy.
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]
WithoutCredential = Annotated[bool, Depends(request_admitted_without_credential)]


def _looks_like_command(value: str) -> bool:
    """Whitespace is a one-way signal: a URL can't hold an unencoded space, so a value with whitespace is
    definitely a command. No whitespace proves nothing (a lone token may be a single-arg command or a
    scheme-less URL)."""
    return any(ch.isspace() for ch in value)


def _normalize_stdio_command(url: str) -> str:
    raw = url or ""
    trimmed = raw.strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    # On Windows only space/tab delimit arguments.
    normalized = raw.lstrip().rstrip(" \t") if sys.platform == "win32" else trimmed
    try:
        parts = parse_stdio_command(normalized)
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
    if any("\x00" in part for part in parts):
        raise HTTPException(
            status_code = 400,
            detail = "command and arguments must not contain NUL characters",
        )
    if "://" in parts[0]:
        raise HTTPException(
            status_code = 400,
            detail = "Enter an http(s):// URL, or a local command whose "
            "first token is an executable (not a URL).",
        )
    return normalized


def _validate_url(url: str) -> str:
    raw = url or ""
    trimmed = raw.strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "url must not be empty")
    # Syntax only: persisting and running a command stay behind the stdio gate.
    if stdio_mcp_enabled() and is_stdio(trimmed):
        return _normalize_stdio_command(raw)
    parsed = urlparse(trimmed)
    if parsed.scheme not in ("http", "https"):
        if _looks_like_command(trimmed):
            detail = stdio_mcp_disabled_reason()
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
            normalized_value = str(value)
            if "\x00" in key or "\x00" in normalized_value:
                raise HTTPException(
                    status_code = 400,
                    detail = "headers and environment variables must not contain NUL characters",
                )
            if "=" in key:
                raise HTTPException(
                    status_code = 400,
                    detail = "header and environment variable names must not contain '='",
                )
            out[key] = normalized_value
    return out or None


def _image_mappings_active(row: dict) -> bool:
    from core.inference.tools import _enabled_mcp_servers

    if not image_input_mappings(row) or not _enabled_mcp_servers([row]):
        return False
    if is_stdio(row["url"]) and not stdio_mcp_enabled():
        return False
    tools = get_cached_tools(row["id"])
    return tools is None or any(
        image_mapping(row, tool) for tool in tools if tool_visible_to(tool, "model")
    )


def _oauth_client(
    client_id: str | None, client_secret: str | None
) -> tuple[str | None, str | None]:
    client_id = (client_id or "").strip() or None
    if client_secret and not client_id:
        raise HTTPException(status_code = 400, detail = "oauth_client_secret requires oauth_client_id")
    return client_id, client_secret or None


def _row_to_response(row: dict, *, include_headers: bool = True) -> McpServerResponse:
    return McpServerResponse(
        id = row["id"],
        builtin_id = row.get("builtin_id"),
        display_name = row["display_name"],
        url = row["url"],
        headers = (parse_server_headers(row) or {}) if include_headers else {},
        is_enabled = bool(row["is_enabled"]),
        use_oauth = bool(row.get("use_oauth")),
        oauth_client_id = row.get("oauth_client_id"),
        has_oauth_client_secret = bool(row.get("oauth_client_secret")),
        image_input_mappings = image_input_mappings(row),
        image_mappings_active = _image_mappings_active(row),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


def _blender_row():
    return next(
        (row for row in mcp_servers_db.list_servers() if row.get("builtin_id") == "blender"), None
    )


def _require_managed_access(
    via_api_key,
    no_credential,
    *,
    executes = False,
):
    require_ui_session_for_local_commands(via_api_key or no_credential)
    if executes and not stdio_mcp_enabled():
        raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())


@router.get("/builtins", response_model = list[McpBuiltinResponse])
def list_builtins(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    if via_api_key or no_credential:
        item = blender.catalog_item()
        item.available = False
        item.unavailable_reason = (
            "An authenticated Unsloth Studio UI session is required for Blender MCP."
        )
        return [item]
    return [blender.catalog_item(_blender_row())]


@router.post("/builtins/blender/test", response_model = McpServerProbeResult)
@serialize_mcp_server_mutation
async def test_blender(
    payload: BlenderTest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    _require_managed_access(via_api_key, no_credential, executes = True)
    row = _blender_row()
    config = json.loads(row.get("builtin_config_json") or "{}") if row else {}
    if not (config.get("consent") or payload.consent):
        raise HTTPException(
            status_code = 400, detail = "Explicit consent is required before testing Blender MCP."
        )
    settings = BlenderSettings(port = payload.port, blender_path = payload.blender_path)
    on_tools = None
    if row and blender.settings_for(row) == settings:
        on_tools = lambda tools: cache_tools(row["id"], tools)
    return await blender.probe(settings, on_tools = on_tools)


@router.put("/builtins/blender", response_model = McpBuiltinResponse)
@serialize_mcp_server_mutation
async def setup_blender(
    payload: BlenderSetup,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    _require_managed_access(via_api_key, no_credential, executes = payload.is_enabled)
    old = _blender_row()
    config = json.loads(old.get("builtin_config_json") or "{}") if old else {}
    if payload.is_enabled and not (config.get("consent") or payload.consent):
        raise HTTPException(
            status_code = 400, detail = "Explicit consent is required before enabling Blender MCP."
        )
    settings = BlenderSettings(port = payload.port, blender_path = payload.blender_path)
    config = {**settings.model_dump(), "consent": bool(config.get("consent") or payload.consent)}
    server_id = old["id"] if old else uuid.uuid4().hex[:16]
    if old:
        mcp_servers_db.update_server(
            server_id, {"builtin_config_json": json.dumps(config), "is_enabled": False}
        )
    else:
        mcp_servers_db.create_server(
            server_id,
            "Blender",
            "",
            is_enabled = False,
            builtin_id = "blender",
            builtin_config_json = json.dumps(config),
        )
    invalidate_tool_cache(server_id)
    if old:
        await asyncio.to_thread(close_mcp_sessions, old["url"], parse_server_headers(old))
    if payload.is_enabled:
        result = await blender.probe(
            settings, check_bridge = False, on_tools = lambda tools: cache_tools(server_id, tools)
        )
        if not result.ok:
            raise HTTPException(status_code = 400, detail = result.error)
        mcp_servers_db.update_server(server_id, {"is_enabled": True})
    return blender.catalog_item(mcp_servers_db.get_server(server_id))


@router.post("/stdio/decode", response_model = McpStdioCommand)
def decode_stdio_command(
    payload: McpStdioDecodeRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    require_ui_session_for_local_commands(via_api_key)
    if not is_stdio(payload.url.strip()):
        raise HTTPException(status_code = 400, detail = "HTTP(S) MCP servers do not have arguments")
    url = _normalize_stdio_command(payload.url)
    parts = parse_stdio_command(url)
    return McpStdioCommand(command = parts[0], arguments = parts[1:])


@router.post("/stdio/encode", response_model = McpStdioEncodeResponse)
def encode_stdio_command(
    payload: McpStdioCommand,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    require_ui_session_for_local_commands(via_api_key)
    command = payload.command.strip()
    if not command:
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    if "://" in command:
        raise HTTPException(
            status_code = 400,
            detail = "command must be a local executable, not a URL",
        )
    url = join_stdio_command([command, *payload.arguments])
    _normalize_stdio_command(url)
    return McpStdioEncodeResponse(url = url)


# FastAPI offloads sync reads; mutations stay on-loop to preserve atomic sequences.
@router.get("/", response_model = list[McpServerResponse])
def list_mcp_servers(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    rows = mcp_servers_db.list_servers()
    if via_api_key or no_credential:
        # url/headers hold argv/env secrets; blanking url allows bogus commands on update.
        rows = [row for row in rows if not is_stdio(row["url"])]
    return [_row_to_response(row, include_headers = not no_credential) for row in rows]


@router.get("/research-tools")
async def list_research_search_tools(
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    from core.inference.tools import mcp_search_tools
    tools = await mcp_search_tools(include_stdio = not (via_api_key or no_credential))
    return [
        {key: tool[key] for key in ("serverId", "serverName", "tool", "description")}
        for tool in tools
    ]


@router.post("/", response_model = McpServerResponse, status_code = 201)
async def create_mcp_server(
    payload: McpServerCreate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    display_name = (payload.display_name or "").strip()
    if not display_name:
        raise HTTPException(status_code = 400, detail = "display_name must not be empty")
    url = _validate_url(payload.url)
    if is_stdio(url):
        require_ui_session_for_local_commands(via_api_key)
    headers = _normalize_headers(payload.headers)
    # OAuth is HTTP-only; a stale flag would push stdio probes onto the 305s OAuth timeout.
    use_oauth = payload.use_oauth and not is_stdio(url)
    client_id, client_secret = (
        _oauth_client(payload.oauth_client_id, payload.oauth_client_secret)
        if use_oauth
        else (None, None)
    )

    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = payload.is_enabled,
        use_oauth = use_oauth,
        image_input_mappings_json = _mappings_json(payload.image_input_mappings),
        oauth_client_id = client_id,
        oauth_client_secret = client_secret,
    )
    return _row_to_response(mcp_servers_db.get_server(server_id))


def _mappings_json(mappings) -> str:
    return json.dumps([mapping.model_dump() for mapping in mappings or []])


def _changes_from_payload(payload: McpServerUpdate) -> dict:
    sent = payload.model_fields_set
    changes: dict = {}

    if "display_name" in sent:
        name = (payload.display_name or "").strip()
        if not name:
            raise HTTPException(status_code = 400, detail = "display_name must not be empty")
        changes["display_name"] = name
    if "url" in sent:
        changes["url"] = _validate_url(payload.url or "")
    if "headers" in sent:
        headers = _normalize_headers(payload.headers)
        changes["headers_json"] = json.dumps(headers) if headers else None
    if "is_enabled" in sent:
        if payload.is_enabled is None:
            raise HTTPException(status_code = 400, detail = "is_enabled must be true or false")
        changes["is_enabled"] = payload.is_enabled
    if "use_oauth" in sent:
        if payload.use_oauth is None:
            raise HTTPException(status_code = 400, detail = "use_oauth must be true or false")
        changes["use_oauth"] = payload.use_oauth
    if "image_input_mappings" in sent:
        changes["image_input_mappings_json"] = _mappings_json(payload.image_input_mappings)
    if "oauth_client_id" in sent:
        changes["oauth_client_id"] = (payload.oauth_client_id or "").strip() or None
    if "oauth_client_secret" in sent:
        changes["oauth_client_secret"] = payload.oauth_client_secret or None
    if "url" in changes and is_stdio(changes["url"]):
        changes["use_oauth"] = False
    if changes.get("use_oauth") is False:
        changes["oauth_client_id"] = changes["oauth_client_secret"] = None
    return changes


@router.put("/{server_id}", response_model = McpServerResponse)
@serialize_mcp_server_mutation
async def update_mcp_server(
    server_id: str,
    payload: McpServerUpdate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
    no_credential: WithoutCredential = False,
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    changes = _changes_from_payload(payload)
    if old.get("builtin_id"):
        _require_managed_access(via_api_key, no_credential)
        if payload.model_fields_set != {"is_enabled"} or payload.is_enabled is not False:
            raise HTTPException(
                status_code = 400,
                detail = "Use the managed integration setup to configure or enable this server.",
            )
    client_id = changes.get("oauth_client_id", old.get("oauth_client_id"))
    # A secret belongs to one client at one origin: a new client ID or URL drops it unless replaced.
    if "oauth_client_secret" not in changes and (
        client_id != old.get("oauth_client_id") or changes.get("url", old["url"]) != old["url"]
    ):
        changes["oauth_client_secret"] = None
    _oauth_client(client_id, changes.get("oauth_client_secret", old.get("oauth_client_secret")))
    if not changes:
        raise HTTPException(status_code = 400, detail = "No fields to update")
    # Check both directions, before any side effect, so a refusal changes nothing.
    if is_stdio(old["url"]) or is_stdio(changes.get("url", old["url"])):
        require_ui_session_for_local_commands(via_api_key)
    # On a transport switch drop old headers so env secrets aren't sent as HTTP headers.
    if (
        "url" in changes
        and is_stdio(changes["url"]) != is_stdio(old["url"])
        and "headers_json" not in changes
    ):
        changes["headers_json"] = None
    if bool(old.get("use_oauth")) and (
        ("url" in changes and changes["url"] != old["url"])
        or changes.get("use_oauth") is False
        or any(
            changes.get(k, old.get(k)) != old.get(k)
            for k in ("oauth_client_id", "oauth_client_secret")
        )
    ):
        await clear_oauth_tokens_async(old["url"])
        # That await hands the loop to other requests.
        current = mcp_servers_db.get_server(server_id)
        if current is not None and (
            is_stdio(current["url"]) or is_stdio(changes.get("url", current["url"]))
        ):
            require_ui_session_for_local_commands(via_api_key)
    invalidates_tools = any(
        changes[k] != old.get(k) for k in changes.keys() & TOOL_CACHE_INVALIDATING_FIELDS
    )
    mcp_servers_db.update_server(server_id, changes)
    if invalidates_tools:
        invalidate_tool_cache(server_id)
    if invalidates_tools:
        # Narrowed by env: another row sharing the command with a different env keeps its sessions.
        await asyncio.to_thread(close_mcp_sessions, old["url"], parse_server_headers(old))
    return _row_to_response(mcp_servers_db.get_server(server_id), include_headers = not no_credential)


@router.delete("/{server_id}", status_code = 204)
@serialize_mcp_server_mutation
async def delete_mcp_server(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    old = mcp_servers_db.get_server(server_id)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if old.get("builtin_id"):
        raise HTTPException(
            status_code = 400, detail = "Managed integrations cannot be deleted; disable them instead."
        )
    # Same rule as update: an API key cannot touch a stdio row.
    if is_stdio(old["url"]):
        require_ui_session_for_local_commands(via_api_key)
    if old.get("use_oauth"):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.delete_server(server_id)
    invalidate_tool_cache(server_id)
    await asyncio.to_thread(close_mcp_sessions, old["url"], parse_server_headers(old))


@router.get("/{server_id}/tools")
def list_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    """Cached tool names and input schemas, for choosing an image input mapping."""
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
    tools = get_cached_tools(server_id)
    if tools is None:
        raise HTTPException(status_code = 409, detail = "Refresh this server's tools first")
    return [
        {"name": tool["name"], "inputSchema": tool.get("inputSchema") or tool.get("input_schema")}
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and tool_visible_to(tool, "model")
    ]


@router.post("/{server_id}/refresh", response_model = McpServerProbeResult)
async def refresh_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if server.get("builtin_id"):
        raise HTTPException(
            status_code = 400,
            detail = "Use the managed integration Test action to check Blender readiness.",
        )
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
        if not stdio_mcp_enabled():
            raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())

    use_oauth = bool(server.get("use_oauth"))
    try:
        tools = await list_tools_async(
            url = server["url"],
            headers = parse_server_headers(server),
            timeout = probe_timeout(server["url"], use_oauth),
            use_oauth = use_oauth,
            **oauth_client_kwargs(server),
        )
    except Exception as exc:  # noqa: BLE001 - surface transport+timeout errors to UI
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
            # If the row changed mid-probe, the failure belongs to the old config.
            record_probe_failure(server_id, use_oauth)
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    current = mcp_servers_db.get_server(server_id)
    if current is not None and not any(
        current.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
    ):
        cache_tools(server_id, tools)
    return McpServerProbeResult(ok = True, tool_count = len(tools))


@router.post("/import", response_model = McpServerImportResult)
async def import_mcp_servers(
    payload: McpServerImportRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
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
            if is_stdio(url):
                require_ui_session_for_local_commands(via_api_key)
            headers = _normalize_headers(entry.headers)
        except HTTPException as exc:
            errors.append(f"{entry.display_name}: {exc.detail}")
            continue
        if url in seen_urls:
            skipped.append(entry.display_name)
            continue
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
    payload: McpServerTestRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    # Validation errors must surface as 400 like create/update.
    url = _validate_url(payload.url)
    # Gate before list_tools_async: after it the process has already started.
    if is_stdio(url):
        require_ui_session_for_local_commands(via_api_key)
    headers = _normalize_headers(payload.headers)
    use_oauth = payload.use_oauth and not is_stdio(url)
    client_id, client_secret = _oauth_client(payload.oauth_client_id, payload.oauth_client_secret)
    if use_oauth and payload.server_id and client_id and not client_secret:
        stored = mcp_servers_db.get_server(payload.server_id) or {}
        if stored.get("url") == url and stored.get("oauth_client_id") == client_id:
            client_secret = stored.get("oauth_client_secret")
    try:
        tools = await list_tools_async(
            url = url,
            headers = headers,
            timeout = probe_timeout(url, use_oauth),
            use_oauth = use_oauth,
            **oauth_client_kwargs(
                {"oauth_client_id": client_id, "oauth_client_secret": client_secret}
                if use_oauth
                else {}
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "mcp_servers.test_failed",
            error = str(exc),
            exc_info = True,
        )
        return McpServerProbeResult(ok = False, error = safe_curated_detail(exc))

    return McpServerProbeResult(ok = True, tool_count = len(tools))


_UI_TIMEOUT = 60.0
UI_TOOL_APPROVAL_REQUIRED = "approval_required"


def _ui_server_or_404(server_id: str, via_api_key: bool) -> dict:
    """Re-read per request: a stale widget must not keep a removed server reachable."""
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if not server.get("is_enabled"):
        raise HTTPException(status_code = 400, detail = "MCP server is disabled")
    if is_stdio(server["url"]):
        require_ui_session_for_local_commands(via_api_key)
        if not stdio_mcp_enabled():
            raise HTTPException(status_code = 400, detail = stdio_mcp_disabled_reason())
    return server


def _row_still_matches(server_id: str, server: dict) -> bool:
    current = mcp_servers_db.get_server(server_id)
    return current is not None and all(
        current.get(k) == server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
    )


# One discovery per server at a time: reopening a chat mounts every widget at once, each probing a cold cache.
_discovery_locks: dict = {}


async def _warm_tool_cache(server: dict) -> None:
    """Rediscover once on a cold cache: a chat reopened after a restart never ran the chat path, and widget calls read the cache."""
    server_id = server["id"]
    async with _discovery_locks.setdefault(server_id, asyncio.Lock()):
        tools = get_cached_tools(server_id)
        if tools is None and not in_failure_cooloff(server_id):
            use_oauth = bool(server.get("use_oauth"))
            url = server["url"]
            try:
                tools = await list_tools_async(
                    url = url,
                    headers = parse_server_headers(server),
                    timeout = probe_timeout(url, use_oauth),
                    use_oauth = use_oauth,
                    **oauth_client_kwargs(server),
                )
            except Exception:  # noqa: BLE001 - a probe failure reads as "nothing declared"
                tools = None
            # A row edited mid-probe: the old endpoint's answer must neither authorize a read nor be cached.
            if not _row_still_matches(server_id, server):
                tools = None
            elif tools is None:
                record_probe_failure(server_id, use_oauth)
            else:
                cache_tools(server_id, tools)


def _ui_call_kwargs(server_id: str, server: dict, thread_id, session_id) -> dict:
    from core.inference.tools import mcp_session_scope
    return {
        "url": server["url"],
        "headers": parse_server_headers(server),
        "timeout": _UI_TIMEOUT,
        "use_oauth": bool(server.get("use_oauth")),
        **oauth_client_kwargs(server),
        # execute_tool's key, so a widget reaches the chat's own stdio subprocess.
        "scope": mcp_session_scope(session_id, thread_id),
        "config_check": lambda: _row_still_matches(server_id, server),
    }


@router.get("/{server_id}/ui-resource", response_model = McpUiResourceResponse)
async def read_mcp_ui_resource(
    server_id: str,
    uri: str,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    server = _ui_server_or_404(server_id, via_api_key)
    uri = (uri or "").strip()
    # Only ui:// resources: a filesystem server maps file:// onto the host.
    if not uri.startswith(UI_RESOURCE_SCHEME):
        raise HTTPException(status_code = 400, detail = "uri must be a ui:// resource")
    from core.inference.tools import (
        _STUDIO_CREDENTIAL_BLOCKED,
        _mcp_arguments_reference_studio_credential,
    )

    if _mcp_arguments_reference_studio_credential({"uri": uri}):
        raise HTTPException(status_code = 403, detail = _STUDIO_CREDENTIAL_BLOCKED)
    await _warm_tool_cache(server)
    try:
        contents = await asyncio.to_thread(
            read_resource_sync, uri = uri, **_ui_call_kwargs(server_id, server, thread_id, session_id)
        )
    except Exception as exc:  # noqa: BLE001
        raise log_and_http_error(
            exc,
            502,
            "Could not load this MCP app's interface.",
            event = "mcp_servers.ui_resource_failed",
            log = logger,
        )
    return McpUiResourceResponse(**contents)


@router.post("/{server_id}/ui-tool-call", response_model = McpUiToolCallResult)
async def call_mcp_ui_tool(
    server_id: str,
    payload: McpUiToolCallRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    """Widget is untrusted: server_id comes from the host frame, the tool must be discovered with "app" visibility, and it passes the confirm gate."""
    from core.inference.tools import (
        _STUDIO_CREDENTIAL_BLOCKED,
        MCP_TOOL_PREFIX,
        _mcp_arguments_reference_studio_credential,
        is_potentially_unsafe_tool_call,
        mcp_tool_definition,
    )
    from state.tool_policy import get_tool_policy

    if get_tool_policy() is False:
        raise HTTPException(status_code = 403, detail = "Tools are disabled on this server")
    server = _ui_server_or_404(server_id, via_api_key)
    tool_name = (payload.tool_name or "").strip()
    if not tool_name:
        raise HTTPException(status_code = 400, detail = "tool_name must not be empty")
    await _warm_tool_cache(server)
    tool = mcp_tool_definition(server_id, tool_name)
    if tool is None:
        raise HTTPException(
            status_code = 404, detail = f"MCP server has no discovered tool named '{tool_name}'"
        )
    if not tool_visible_to(tool, "app"):
        raise HTTPException(
            status_code = 403, detail = f"Tool '{tool_name}' is not callable by an MCP app"
        )
    arguments = payload.arguments or {}
    if _mcp_arguments_reference_studio_credential(arguments):
        raise HTTPException(status_code = 403, detail = _STUDIO_CREDENTIAL_BLOCKED)
    mode = payload.permission_mode
    # Unknown modes ask; "auto" asks only for what the model's call would be asked for.
    needs_approval = mode not in ("off", "full") and (
        mode != "auto"
        or is_potentially_unsafe_tool_call(f"{MCP_TOOL_PREFIX}{server_id}__{tool_name}", arguments)
    )
    if needs_approval and not payload.approved:
        raise HTTPException(status_code = 409, detail = UI_TOOL_APPROVAL_REQUIRED)
    try:
        result = await asyncio.to_thread(
            call_tool_structured_sync,
            name = tool_name,
            args = arguments,
            **_ui_call_kwargs(server_id, server, payload.thread_id, payload.session_id),
        )
    except Exception as exc:  # noqa: BLE001
        raise log_and_http_error(
            exc,
            502,
            "The MCP app's tool call failed.",
            event = "mcp_servers.ui_tool_call_failed",
            log = logger,
        )
    return McpUiToolCallResult(**result)
