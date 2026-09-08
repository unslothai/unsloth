# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys

from core.inference.mcp_client import (
    join_stdio_command,
    list_tools_async,
    stdio_mcp_disabled_reason,
    stdio_mcp_enabled,
)
from integrations.blender import MIN_BLENDER_VERSION, MIN_PYTHON_VERSION, launch_command
from integrations.blender.runtime import ensure_runtime
from models.mcp_servers import BlenderSettings, McpBuiltinResponse, McpServerProbeResult


def unavailable_reason():
    if sys.version_info < MIN_PYTHON_VERSION:
        return "Blender MCP requires Python 3.10 or newer."
    if not stdio_mcp_enabled():
        return stdio_mcp_disabled_reason()
    return None


def settings_for(row):
    config = json.loads(row.get("builtin_config_json") or "{}") if row else {}
    return BlenderSettings(
        port = config.get("port", 9876), blender_path = config.get("blender_path", "")
    )


def resolve_server(row: dict) -> dict:
    if row.get("builtin_id") != "blender":
        raise ValueError("Unknown managed MCP integration")
    settings = settings_for(row)
    env = {
        "BLENDER_MCP_HOST": "127.0.0.1",
        "BLENDER_MCP_PORT": str(settings.port),
        "BLENDER_PATH": settings.blender_path or "blender",
    }
    return {
        **row,
        "url": join_stdio_command(launch_command()),
        "headers_json": json.dumps(env),
        "use_oauth": False,
    }


def catalog_item(row = None):
    reason = unavailable_reason()
    return McpBuiltinResponse(
        **settings_for(row).model_dump(),
        server_id = row["id"] if row else None,
        is_enabled = bool(row and row["is_enabled"]),
        available = reason is None,
        unavailable_reason = reason,
        min_blender_version = MIN_BLENDER_VERSION,
    )


async def _bridge_version(port):
    reader, writer = await asyncio.open_connection("127.0.0.1", port, limit = 16384)
    try:
        request = {
            "type": "execute",
            "code": "import bpy\nresult = {'version': list(bpy.app.version)}",
            "strict_json": True,
        }
        writer.write((json.dumps(request) + "\0").encode())
        await writer.drain()
        response = json.loads((await reader.readuntil(b"\0"))[:-1])
        version = response.get("result", {}).get("version")
        if (
            response.get("status") != "ok"
            or not isinstance(version, list)
            or len(version) != 3
            or any(type(v) is not int for v in version)
        ):
            raise ValueError("Invalid Blender bridge version response")
        if tuple(version) < tuple(map(int, MIN_BLENDER_VERSION.split("."))):
            raise ValueError(f"Blender {MIN_BLENDER_VERSION} or newer is required")
    finally:
        writer.close()
        await writer.wait_closed()


async def probe(
    settings: BlenderSettings,
    *,
    check_bridge: bool = True,
    on_tools = None,
) -> McpServerProbeResult:
    reason = unavailable_reason()
    if reason:
        return McpServerProbeResult(ok = False, error = reason)
    try:
        await asyncio.to_thread(ensure_runtime)
    except Exception:
        return McpServerProbeResult(
            ok = False,
            error = "Could not set up Blender MCP. Check the backend's internet connection and cache permissions, then retry. Downloads must match the pinned checksum.",
        )
    row = resolve_server(
        {"builtin_id": "blender", "builtin_config_json": settings.model_dump_json()}
    )
    env = json.loads(row["headers_json"])
    try:
        tools = await list_tools_async(row["url"], headers = env, timeout = 15)
    except Exception:
        return McpServerProbeResult(
            ok = False,
            error = "Could not connect to Blender MCP. Check the Studio backend installation and retry.",
        )
    result = McpServerProbeResult(ok = True, tool_count = len(tools))
    if check_bridge:
        try:
            await asyncio.wait_for(_bridge_version(settings.port), timeout = 5)
            result.blender_ready = True
        except Exception:
            result.blender_ready = False
            result.blender_error = f"Blender is not connected on port {settings.port}. Enable its MCP add-on, then retry."
    if on_tools is not None:
        on_tools(tools)
    return result
