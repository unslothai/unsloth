# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parse a standard ``mcpServers`` JSON config (Claude Desktop / Cursor / Cline
/ VS Code) into entries the existing MCP storage understands. See issue #5936.

A stdio entry (``command`` + ``args`` + ``env``) is joined into the single
command string the ``url`` field already stores; a remote entry (``url`` +
``headers``) maps straight through. Parsing never raises on a single bad entry:
it returns ``(entries, errors)`` so one malformed server can't sink the import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.inference.mcp_client import join_stdio_command

_SCALAR = (str, int, float, bool)
_UNSUPPORTED_STDIO_FIELDS = ("cwd", "envFile")
_UNSUPPORTED_TIMEOUT_FIELDS = ("timeout", "timeoutMs", "timeoutSeconds")
_HTTP_REMOTE_TYPES = ("http", "streamableHttp")


@dataclass
class ParsedMcpEntry:
    display_name: str
    url: str  # joined command (stdio) or http(s) url (remote)
    headers: Optional[dict[str, str]]  # env vars (stdio) or http headers (remote)
    is_stdio: bool
    is_enabled: bool = True
    use_oauth: bool = False


def _coerce_str_dict(value: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()}


def _has_variable_reference(value: object) -> bool:
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, list):
        return any(_has_variable_reference(item) for item in value)
    if isinstance(value, dict):
        return any(_has_variable_reference(item) for item in value.values())
    return False


def _has_null_value(value: object) -> bool:
    return isinstance(value, dict) and any(item is None for item in value.values())


def _enabled_from_spec(label: str, spec: dict) -> tuple[Optional[bool], Optional[str]]:
    disabled = spec.get("disabled")
    if disabled is None:
        return True, None
    if not isinstance(disabled, bool):
        return None, f"{label}: 'disabled' must be true or false."
    return not disabled, None


def _parse_entry(name: str, spec: object) -> tuple[Optional[ParsedMcpEntry], Optional[str]]:
    label = str(name).strip()
    if not label:
        return None, "Server entry has an empty name."
    if not isinstance(spec, dict):
        return None, f"{label}: entry must be an object."
    if _has_variable_reference(spec):
        return None, f"{label}: VS Code variable references are not supported by import."

    is_enabled, error = _enabled_from_spec(label, spec)
    if error:
        return None, error

    has_command = bool(spec.get("command"))
    has_url = bool(spec.get("url"))
    if has_command and has_url:
        return None, f"{label}: entry has both 'command' and 'url'; use one."
    if not has_command and not has_url:
        return None, f"{label}: entry needs a 'command' (stdio) or 'url' (remote)."

    if has_command:
        command = spec["command"]
        if not isinstance(command, str):
            return None, f"{label}: 'command' must be a string."
        entry_type = spec.get("type")
        if entry_type is not None and entry_type != "stdio":
            return None, f"{label}: stdio entry has unsupported type {entry_type!r}."
        sandbox_enabled = spec.get("sandboxEnabled")
        if sandbox_enabled is not None and not isinstance(sandbox_enabled, bool):
            return None, f"{label}: 'sandboxEnabled' must be true or false."
        if sandbox_enabled:
            return None, f"{label}: sandboxed stdio servers cannot be preserved by import."
        unsupported = [field for field in _UNSUPPORTED_STDIO_FIELDS if spec.get(field) is not None]
        if unsupported:
            return None, f"{label}: import cannot preserve {', '.join(unsupported)}."
        if spec.get("oauth") is not None:
            return None, f"{label}: 'oauth' is only supported for remote servers."
        args = spec.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, _SCALAR) for a in args):
            return None, f"{label}: 'args' must be a list of strings."
        env = spec.get("env")
        if env is not None and not isinstance(env, dict):
            return None, f"{label}: 'env' must be an object."
        if _has_null_value(env):
            return None, f"{label}: null environment values are not supported by import."
        url = join_stdio_command([command, *(str(a) for a in args)])
        headers = _coerce_str_dict(env) if env else None
        return ParsedMcpEntry(label, url, headers, True, is_enabled = is_enabled), None

    url = spec["url"]
    if not isinstance(url, str):
        return None, f"{label}: 'url' must be a string."
    url = url.strip()
    entry_type = spec.get("type")
    if entry_type is not None and entry_type not in (*_HTTP_REMOTE_TYPES, "sse"):
        return None, f"{label}: remote entry has unsupported type {entry_type!r}."
    unsupported_timeout = [
        field for field in _UNSUPPORTED_TIMEOUT_FIELDS if spec.get(field) is not None
    ]
    if unsupported_timeout:
        return None, f"{label}: import cannot preserve {', '.join(unsupported_timeout)}."
    url_infers_sse = url.rstrip("/").endswith("/sse")
    if entry_type == "sse" and not url_infers_sse:
        return None, f"{label}: explicit SSE transport cannot be preserved for this URL."
    if entry_type in _HTTP_REMOTE_TYPES and url_infers_sse:
        return None, f"{label}: explicit HTTP transport cannot be preserved for this URL."
    oauth_raw = spec.get("oauth")
    if oauth_raw is not None and not isinstance(oauth_raw, dict):
        return None, f"{label}: 'oauth' must be an object."
    headers_raw = spec.get("headers")
    if headers_raw is not None and not isinstance(headers_raw, dict):
        return None, f"{label}: 'headers' must be an object."
    if _has_null_value(headers_raw):
        return None, f"{label}: null header values are not supported by import."
    headers = _coerce_str_dict(headers_raw) if headers_raw else None
    return ParsedMcpEntry(
        label,
        url,
        headers,
        False,
        is_enabled = is_enabled,
        use_oauth = oauth_raw is not None,
    ), None


def parse_mcp_config(config: object) -> tuple[list[ParsedMcpEntry], list[str]]:
    """Parse a Claude-Desktop/Cursor/Cline/VS Code config. Accepts the
    ``mcpServers`` key (primary) or ``servers`` (VS Code alias). Returns
    ``(entries, errors)``; a bad entry adds an error rather than raising."""
    if not isinstance(config, dict):
        return [], ["Config must be a JSON object."]
    servers_key = "mcpServers" if "mcpServers" in config else "servers"
    servers = config.get(servers_key)
    if servers is None:
        return [], ["Config has no 'mcpServers' (or 'servers') object."]
    if not isinstance(servers, dict):
        return [], [f"'{servers_key}' must be an object mapping name -> server."]

    entries: list[ParsedMcpEntry] = []
    errors: list[str] = []
    for name, spec in servers.items():
        entry, error = _parse_entry(name, spec)
        if error:
            errors.append(error)
        elif entry:
            entries.append(entry)
    return entries, errors
