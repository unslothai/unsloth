# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-managed MCP for desktop apps that serve MCP themselves: loopback HTTP from an add-on, or a stdio launcher the app ships."""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from integrations.installed import is_installed
from core.inference.mcp_client import (
    join_stdio_command,
    list_tools_async,
    stdio_mcp_disabled_reason,
    stdio_mcp_enabled,
)
from models.mcp_servers import EditorSettings, McpBuiltinResponse, McpServerProbeResult


@dataclass(frozen = True)
class Editor:
    display_name: str
    default_port: int
    not_ready: str
    path: str = "/mcp"
    platforms: tuple[str, ...] = ()


EDITORS = {
    # MCP for Unity (CoplayDev): the Unity package starts this server; /api/instances lists connected editors.
    "unity": Editor(
        "Unity",
        8080,
        "No Unity Editor is connected. Open Window > MCP for Unity in your project and start the server.",
    ),
    # Epic's Unreal MCP plugin (UE 5.8+, no auth). A project folder switches to ChiR24's EngineWire token.
    "unreal": Editor(
        "Unreal Engine",
        8000,
        "Unreal Editor is not answering. Open your project with the Unreal MCP plugin enabled.",
    ),
    "godot": Editor(
        "Godot",
        9080,
        "Godot is not answering. Open the project with the Godot MCP Native plugin enabled and its server started.",
    ),
    "figma": Editor(
        "Figma",
        3845,
        "Figma is not answering. In the Figma desktop app, open a design file, switch to Dev Mode and enable the desktop MCP server.",
        platforms = ("win32", "darwin"),
    ),
    "roblox": Editor(
        "Roblox Studio",
        0,
        "Roblox Studio is not answering. Open Studio and turn on Assistant > Manage MCP Servers > Enable Studio as MCP server.",
        platforms = ("win32", "darwin"),
    ),
}


def roblox_command() -> list[str]:
    if sys.platform == "win32":
        return ["cmd.exe", "/c", os.path.expandvars(r"%LOCALAPPDATA%\Roblox\mcp.bat")]
    return ["/Applications/RobloxStudio.app/Contents/MacOS/StudioMCP"]


def normalize(builtin_id: str, settings: EditorSettings) -> EditorSettings:
    return EditorSettings(
        port = settings.port or EDITORS[builtin_id].default_port or None,
        project_path = settings.project_path.strip(),
    )


def settings_for(builtin_id: str, row = None) -> EditorSettings:
    config = json.loads(row.get("builtin_config_json") or "{}") if row else {}
    return normalize(
        builtin_id,
        EditorSettings(port = config.get("port"), project_path = config.get("project_path", "")),
    )


def unreal_token_path(project_path: str) -> Optional[Path]:
    if not project_path.strip():
        return None
    root = Path(project_path.strip()).expanduser()
    if root.suffix.lower() == ".uproject":
        root = root.parent
    return root / "Saved" / "MCP" / "capability-token"


def _unreal_token(project_path: str) -> Optional[str]:
    path = unreal_token_path(project_path)
    try:
        token = path.read_text(encoding = "utf-8").strip() if path else ""
    except (OSError, UnicodeError):
        return None
    return token or None


def _endpoint(builtin_id: str, settings: EditorSettings) -> tuple[str, dict[str, str]]:
    if builtin_id == "roblox":
        return join_stdio_command(roblox_command()), {}
    headers = {}
    if builtin_id == "unreal":
        # Read per resolve: the editor writes the token, so a restart or rotation needs no Studio change.
        token = _unreal_token(settings.project_path)
        if token:
            headers["X-MCP-Capability-Token"] = token
    return f"http://127.0.0.1:{settings.port}{EDITORS[builtin_id].path}", headers


def resolve_server(row: dict) -> dict:
    builtin_id = row.get("builtin_id")
    if builtin_id not in EDITORS:
        raise ValueError("Unknown managed MCP integration")
    url, headers = _endpoint(builtin_id, settings_for(builtin_id, row))
    return {
        **row,
        "url": url,
        "headers_json": json.dumps(headers) if headers else None,
        "use_oauth": False,
    }


def unavailable_reason(builtin_id: str = "") -> Optional[str]:
    platforms = EDITORS[builtin_id].platforms if builtin_id in EDITORS else ()
    if platforms and sys.platform not in platforms:
        return f"{EDITORS[builtin_id].display_name} MCP runs on Windows and macOS only."
    # Editor tools run code on this computer, so they follow the local-command gate as Blender does.
    return None if stdio_mcp_enabled() else stdio_mcp_disabled_reason()


def catalog_item(builtin_id: str, row = None) -> McpBuiltinResponse:
    reason = unavailable_reason(builtin_id)
    settings = settings_for(builtin_id, row)
    return McpBuiltinResponse(
        builtin_id = builtin_id,
        display_name = EDITORS[builtin_id].display_name,
        port = settings.port or 0,
        project_path = settings.project_path,
        installed = is_installed(builtin_id),
        server_id = row["id"] if row else None,
        is_enabled = bool(row and row["is_enabled"]),
        available = reason is None,
        unavailable_reason = reason,
    )


async def _unity_connected(port: int) -> bool:
    async with httpx.AsyncClient(trust_env = False, timeout = 5) as client:
        response = await client.get(f"http://127.0.0.1:{port}/api/instances")
        response.raise_for_status()
        return bool(response.json().get("instances"))


async def probe(
    builtin_id: str,
    settings: EditorSettings,
    *,
    check_editor: bool = True,
    on_tools = None,
) -> McpServerProbeResult:
    editor = EDITORS[builtin_id]
    reason = unavailable_reason(builtin_id)
    if reason:
        return McpServerProbeResult(ok = False, error = reason)
    url, headers = _endpoint(builtin_id, settings)
    if builtin_id == "unreal" and settings.project_path and not headers:
        path = unreal_token_path(settings.project_path)
        return McpServerProbeResult(
            ok = False,
            error = f"No EngineWire capability token at {path}. Open the project in Unreal with the plugin enabled, or clear the project folder to use Epic's Unreal MCP plugin.",
        )
    try:
        tools = await list_tools_async(url, headers = headers or None, timeout = 15)
    except Exception:
        where = f" (port {settings.port})" if settings.port else ""
        return McpServerProbeResult(ok = False, error = f"{editor.not_ready}{where}")
    result = McpServerProbeResult(ok = True, tool_count = len(tools))
    if check_editor:
        result.editor_ready = True
        if builtin_id == "unity":
            try:
                result.editor_ready = await asyncio.wait_for(
                    _unity_connected(settings.port), timeout = 5
                )
            except Exception:
                result.editor_ready = False
            if not result.editor_ready:
                result.editor_error = editor.not_ready
    if on_tools is not None:
        on_tools(tools)
    return result
