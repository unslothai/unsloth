# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import uuid
from typing import Annotated
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import (
    authenticated_via_api_key,
    get_current_subject,
    request_admitted_without_credential,
    require_ui_session_for_local_commands,
)
from core.inference.mcp_client import (
    TOOL_CACHE_INVALIDATING_FIELDS,
    cache_tools,
    clear_oauth_tokens_async,
    close_stdio_sessions,
    invalidate_tool_cache,
    is_stdio,
    join_stdio_command,
    list_tools_async,
    oauth_client_kwargs,
    parse_server_headers,
    parse_stdio_command,
    probe_timeout,
    record_probe_failure,
    serialize_mcp_server_mutation,
    stdio_mcp_disabled_reason,
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
    McpStdioCommand,
    McpStdioDecodeRequest,
    McpStdioEncodeResponse,
)
from storage import mcp_servers_db
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)

router = APIRouter()

# Only a UI session may define a local command; API keys keep http(s) MCP.
# Annotated, not a Depends default: these routes are also called directly by the
# tests, where a Depends object is truthy and would read as "API key".
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]
WithoutCredential = Annotated[bool, Depends(request_admitted_without_credential)]


def _looks_like_command(value: str) -> bool:
    """Whitespace is a one-way signal: a URL can't hold an unencoded space, so
    a value with whitespace is definitely a command. No whitespace proves
    nothing (a lone token may be a single-arg command or a scheme-less URL)."""
    return any(ch.isspace() for ch in value)


def _normalize_stdio_command(url: str) -> str:
    raw = url or ""
    trimmed = raw.strip()
    if not trimmed:
        raise HTTPException(status_code = 400, detail = "command must not be empty")
    # Leading whitespace is executable-field padding. At the other end, only
    # space/tab delimit arguments on Windows. POSIX quoting protects whitespace.
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
    # Non-HTTP values reuse the URL field for local commands. Syntax validation
    # is policy-free, but persistence and execution stay behind the stdio gate.
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


OAUTH_CLIENT_ID_FIELD = "oauth_client_id"
OAUTH_CLIENT_SECRET_FIELD = "oauth_client_secret"


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


def _oauth_credentials(
    client_id: str | None, client_secret: str | None
) -> tuple[str | None, str | None]:
    normalized_id = (client_id or "").strip() or None
    normalized_secret = client_secret or None
    if normalized_secret and not normalized_id:
        raise HTTPException(
            status_code = 400,
            detail = "oauth_client_secret requires oauth_client_id",
        )
    return normalized_id, normalized_secret


def _row_to_response(row: dict, *, include_headers: bool = True) -> McpServerResponse:
    return McpServerResponse(
        id = row["id"],
        display_name = row["display_name"],
        url = row["url"],
        headers = (parse_server_headers(row) or {}) if include_headers else {},
        is_enabled = bool(row["is_enabled"]),
        use_oauth = bool(row.get("use_oauth")),
        oauth_client_id = row.get("oauth_client_id"),
        has_oauth_client_secret = bool(row.get(mcp_servers_db.HAS_OAUTH_CLIENT_SECRET_KEY)),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


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
    # Metadata-only endpoint: never load stored OAuth secrets here.
    rows = mcp_servers_db.list_servers(include_secrets = False)
    if via_api_key or no_credential:
        # Drop the row, not just its fields: `url` is the argv (carries
        # credentials), `headers` is the subprocess env, and a blanked url would
        # round-trip into update as a bogus command.
        rows = [row for row in rows if not is_stdio(row["url"])]
    return [_row_to_response(row, include_headers = not no_credential) for row in rows]


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
    # OAuth is HTTP-only; force it off for stdio commands so a stale flag can't
    # push the probe onto the 305s OAuth timeout. Backend enforces this.
    use_oauth = payload.use_oauth and not is_stdio(url)
    oauth_client_id, oauth_client_secret = _oauth_credentials(
        payload.oauth_client_id,
        payload.oauth_client_secret,
    )

    server_id = uuid.uuid4().hex[:16]
    mcp_servers_db.create_server(
        id = server_id,
        display_name = display_name,
        url = url,
        headers_json = json.dumps(headers) if headers else None,
        is_enabled = payload.is_enabled,
        use_oauth = use_oauth,
        oauth_client_id = oauth_client_id if use_oauth else None,
        oauth_client_secret = oauth_client_secret if use_oauth else None,
    )
    return _row_to_response(mcp_servers_db.get_server(server_id, include_secret = False))


def _changes_from_payload(
    payload: McpServerUpdate, stored_oauth_client_id: str | None = None
) -> dict:
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
    if "oauth_client_id" in sent:
        changes["oauth_client_id"], _ = _oauth_credentials(
            payload.oauth_client_id,
            payload.oauth_client_secret if "oauth_client_secret" in sent else None,
        )
        if changes["oauth_client_id"] is None:
            changes["oauth_client_secret"] = None
    if "oauth_client_secret" in sent:
        effective_id = changes.get(
            "oauth_client_id",
            payload.oauth_client_id or stored_oauth_client_id,
        )
        _, changes["oauth_client_secret"] = _oauth_credentials(
            effective_id,
            payload.oauth_client_secret,
        )
    # Apply transport/auth normalization last so later credential fields cannot
    # reintroduce stale OAuth configuration.
    if changes.get("use_oauth") is False:
        changes["oauth_client_id"] = None
        changes["oauth_client_secret"] = None
    # stdio is OAuth-less: drop a stale OAuth flag when switching to a command.
    if "url" in changes and is_stdio(changes["url"]):
        changes["use_oauth"] = False
        changes["oauth_client_id"] = None
        changes["oauth_client_secret"] = None
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
    old = mcp_servers_db.get_server(server_id, include_secret = False)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    changes = _changes_from_payload(
        payload,
        stored_oauth_client_id = old.get("oauth_client_id"),
    )
    if (
        "oauth_client_id" in changes
        and changes["oauth_client_id"] != old.get("oauth_client_id")
        and "oauth_client_secret" not in changes
    ):
        # A secret belongs to one registered client ID. Never silently pair an
        # existing secret with a newly entered ID.
        changes["oauth_client_secret"] = None
    if "url" in changes and changes["url"] != old["url"] and "oauth_client_secret" not in changes:
        # OAuth metadata at the new endpoint controls where credentials are
        # submitted. Never carry a confidential client secret across origins.
        changes["oauth_client_secret"] = None
    if not changes:
        raise HTTPException(status_code = 400, detail = "No fields to update")
    # Both directions, so an API key can neither repoint an http row at a command
    # nor edit a stdio row's env/name/enabled flag. Before every side effect, so a
    # refusal leaves the row, its OAuth tokens, cache and sessions untouched.
    if is_stdio(old["url"]) or is_stdio(changes.get("url", old["url"])):
        require_ui_session_for_local_commands(via_api_key)
    # headers == HTTP headers (remote) or env vars (stdio). On a transport-type
    # switch with no new headers, drop the old ones so env secrets aren't
    # re-sent as HTTP headers (or vice versa).
    if (
        "url" in changes
        and is_stdio(changes["url"]) != is_stdio(old["url"])
        and "headers_json" not in changes
    ):
        changes["headers_json"] = None
    # `old` is a masked read, so it never carries the stored secret value and no
    # comparison against it can see one being removed. Decide from presence
    # instead: the secret changed when a replacement was supplied, or when an
    # existing one is being cleared. A blank field on an unchanged row -- what
    # the edit dialog sends on a rename -- is neither.
    secret_changed = OAUTH_CLIENT_SECRET_FIELD in changes and (
        bool(changes[OAUTH_CLIENT_SECRET_FIELD])
        or bool(old.get(mcp_servers_db.HAS_OAUTH_CLIENT_SECRET_KEY))
    )
    # Clear persisted OAuth tokens when the URL changes or OAuth is disabled;
    # fastmcp keys tokens by URL and would otherwise let a re-pointed server
    # silently inherit the old account's credentials.
    oauth_config_changed = secret_changed or (
        OAUTH_CLIENT_ID_FIELD in changes
        and changes[OAUTH_CLIENT_ID_FIELD] != old.get(OAUTH_CLIENT_ID_FIELD)
    )
    if bool(old.get("use_oauth")) and (
        ("url" in changes and changes["url"] != old["url"])
        or changes.get("use_oauth") is False
        or oauth_config_changed
    ):
        await clear_oauth_tokens_async(old["url"])
        # That await hands the loop to other requests, so re-read and re-gate
        # before writing: a UI conversion to stdio landing in the window would
        # otherwise let an API key's headers become the command's env.
        # Metadata-only re-gate: it reads url alone, so don't load the secret.
        current = mcp_servers_db.get_server(server_id, include_secret = False)
        if current is not None and (
            is_stdio(current["url"]) or is_stdio(changes.get("url", current["url"]))
        ):
            require_ui_session_for_local_commands(via_api_key)
    # A new endpoint/auth makes cached tools wrong and disabling makes them unreachable, so drop
    # them and let the next send re-probe; a rename leaves them valid. Live stdio sessions for the
    # old endpoint close too. Gate on a real value change, not mere presence: the edit dialog
    # resends url/headers/oauth unchanged on a rename, which must not drop the session.
    invalidates_tools = secret_changed or any(
        changes[k] != old.get(k)
        for k in changes.keys() & TOOL_CACHE_INVALIDATING_FIELDS
        # Handled by `secret_changed`: `old` cannot supply a value to compare.
        if k != OAUTH_CLIENT_SECRET_FIELD
    )
    mcp_servers_db.update_server(server_id, changes)
    if invalidates_tools:
        invalidate_tool_cache(server_id)
    if invalidates_tools:
        # Narrow to this row's env: another server row sharing the command but
        # with a different env keeps its live sessions.
        await asyncio.to_thread(close_stdio_sessions, old["url"], parse_server_headers(old))
    return _row_to_response(
        mcp_servers_db.get_server(server_id, include_secret = False),
        include_headers = not no_credential,
    )


@router.delete("/{server_id}", status_code = 204)
@serialize_mcp_server_mutation
async def delete_mcp_server(server_id: str, current_subject: str = Depends(get_current_subject)):
    old = mcp_servers_db.get_server(server_id, include_secret = False)
    if not old:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    if old.get("use_oauth"):
        await clear_oauth_tokens_async(old["url"])
    mcp_servers_db.delete_server(server_id)
    invalidate_tool_cache(server_id)
    await asyncio.to_thread(close_stdio_sessions, old["url"], parse_server_headers(old))


@router.post("/{server_id}/refresh", response_model = McpServerProbeResult)
async def refresh_mcp_server_tools(
    server_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    server = mcp_servers_db.get_server(server_id)
    if not server:
        raise HTTPException(status_code = 404, detail = "MCP server not found")
    # Refresh uses the stored address, so re-check the stdio gate here too: a
    # stdio row from a desktop DB must not spawn on a hosted/network host.
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
    seen_urls = {row["url"] for row in mcp_servers_db.list_servers(include_secrets = False)}

    for entry in entries:
        try:
            url = _validate_url(entry.url)
            # Per entry, so an API-key import of a mixed config still creates its
            # http entries and reports the stdio ones.
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
        created.append(_row_to_response(mcp_servers_db.get_server(server_id, include_secret = False)))

    return McpServerImportResult(created = created, skipped = skipped, errors = errors)


@router.post("/test", response_model = McpServerProbeResult)
async def test_mcp_server(
    payload: McpServerTestRequest,
    current_subject: str = Depends(get_current_subject),
    via_api_key: ViaApiKey = False,
):
    # URL/header validation must surface as 400 like create/update so the
    # frontend's create-form pre-flight gets the same error semantics as the
    # save call. Only catch transport/timeout errors below.
    url = _validate_url(payload.url)
    # Caller-supplied and unstored, so the gate has to land before
    # list_tools_async -- after it the process has already started.
    if is_stdio(url):
        require_ui_session_for_local_commands(via_api_key)
    headers = _normalize_headers(payload.headers)
    use_oauth = payload.use_oauth and not is_stdio(url)
    oauth_client_id, oauth_client_secret = _oauth_credentials(
        payload.oauth_client_id,
        payload.oauth_client_secret,
    )
    if use_oauth and payload.server_id and oauth_client_id and not oauth_client_secret:
        stored = mcp_servers_db.get_server(payload.server_id)
        if (
            stored
            and bool(stored.get("use_oauth"))
            and stored.get("url") == url
            and stored.get("oauth_client_id") == oauth_client_id
        ):
            oauth_client_secret = stored.get("oauth_client_secret")
    try:
        tools = await list_tools_async(
            url = url,
            headers = headers,
            timeout = probe_timeout(url, use_oauth),
            use_oauth = use_oauth,
            **oauth_client_kwargs(
                {
                    "oauth_client_id": oauth_client_id,
                    "oauth_client_secret": oauth_client_secret,
                }
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
