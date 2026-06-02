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


@dataclass
class ParsedMcpEntry:
    display_name: str
    url: str  # joined command (stdio) or http(s) url (remote)
    headers: Optional[dict[str, str]]  # env vars (stdio) or http headers (remote)
    is_stdio: bool


def _coerce_str_dict(value: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()}


def _parse_entry(
    name: str, spec: object
) -> tuple[Optional[ParsedMcpEntry], Optional[str]]:
    label = str(name).strip()
    if not label:
        return None, "Server entry has an empty name."
    if not isinstance(spec, dict):
        return None, f"{label}: entry must be an object."

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
        args = spec.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, _SCALAR) for a in args):
            return None, f"{label}: 'args' must be a list of strings."
        env = spec.get("env")
        if env is not None and not isinstance(env, dict):
            return None, f"{label}: 'env' must be an object."
        url = join_stdio_command([command, *(str(a) for a in args)])
        headers = _coerce_str_dict(env) if env else None
        return ParsedMcpEntry(label, url, headers, True), None

    url = spec["url"]
    if not isinstance(url, str):
        return None, f"{label}: 'url' must be a string."
    headers_raw = spec.get("headers")
    if headers_raw is not None and not isinstance(headers_raw, dict):
        return None, f"{label}: 'headers' must be an object."
    headers = _coerce_str_dict(headers_raw) if headers_raw else None
    return ParsedMcpEntry(label, url.strip(), headers, False), None


def parse_mcp_config(config: object) -> tuple[list[ParsedMcpEntry], list[str]]:
    """Parse a Claude-Desktop/Cursor/Cline/VS Code config. Accepts the
    ``mcpServers`` key (primary) or ``servers`` (VS Code alias). Returns
    ``(entries, errors)``; a bad entry adds an error rather than raising."""
    if not isinstance(config, dict):
        return [], ["Config must be a JSON object."]
    servers = config.get("mcpServers")
    if servers is None:
        servers = config.get("servers")
    if servers is None:
        return [], ["Config has no 'mcpServers' (or 'servers') object."]
    if not isinstance(servers, dict):
        return [], ["'mcpServers' must be an object mapping name -> server."]

    entries: list[ParsedMcpEntry] = []
    errors: list[str] = []
    for name, spec in servers.items():
        entry, error = _parse_entry(name, spec)
        if error:
            errors.append(error)
        elif entry:
            entries.append(entry)
    return entries, errors
