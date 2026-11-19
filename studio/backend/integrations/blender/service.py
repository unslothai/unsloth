# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import functools
import json
import os
import shutil
import stat
import sys
from typing import Optional

from core.inference.mcp_client import (
    join_stdio_command,
    list_tools_async,
    stdio_mcp_disabled_reason,
    stdio_mcp_enabled,
)
from integrations.blender import MIN_BLENDER_VERSION, MIN_PYTHON_VERSION, launch_command
from integrations.blender.runtime import ensure_runtime
from integrations.installed import is_installed, note_connected
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


def catalog_item(row = None, detect: bool = True):
    reason = unavailable_reason()
    settings = settings_for(row)
    return McpBuiltinResponse(
        **settings.model_dump(),
        installed = is_installed("blender", settings.blender_path) if detect else None,
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
            "code": "import bpy\nresult = {'version': list(bpy.app.version), 'binary_path': bpy.app.binary_path}",
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
        binary = response.get("result", {}).get("binary_path")
        return binary if isinstance(binary, str) and binary else None
    finally:
        writer.close()
        await writer.wait_closed()


def _not_plantable(st) -> bool:
    """Owned by this user or root; writable by nobody else (a private group or a sticky directory is fine)."""
    if st.st_uid not in (0, os.getuid()):
        return False
    if stat.S_ISDIR(st.st_mode) and st.st_mode & stat.S_ISVTX:
        return True
    return not (st.st_mode & 0o002) and not (st.st_mode & 0o020 and not _private_group(st.st_gid))


@functools.lru_cache(maxsize = 8)
def _private_group(gid: int) -> bool:
    """No other account is in the group (a primary gid is often shared: macOS staff)."""
    try:
        import grp, pwd  # noqa: E401, PLC0415 (POSIX only)
        others = set(grp.getgrgid(gid).gr_mem) - {pwd.getpwuid(os.getuid()).pw_name}
        return not others and not any(
            p.pw_gid == gid and p.pw_uid != os.getuid() for p in pwd.getpwall()
        )
    except (ImportError, KeyError, OSError):
        return False


def _parents(path: str):
    while True:
        yield path
        up = os.path.dirname(path)
        if up == path:
            return
        path = up


def _own_blender(path) -> Optional[str]:
    """Anything on 127.0.0.1 can answer the bridge port and Studio later runs this path: the canonical path of an
    own blender* executable, else None (a symlink or a writable parent could be swapped after the check)."""
    if not hasattr(os, "getuid") or not isinstance(path, str) or not os.path.isabs(path):
        return None  # no POSIX owner to check (Windows): never adopt a reported path
    try:
        real = os.path.realpath(path, strict = True)
        st = os.stat(real)
    except (OSError, ValueError):
        return None
    if not os.path.basename(real).lower().startswith("blender"):
        return None
    if not stat.S_ISREG(st.st_mode) or not os.access(real, os.X_OK):
        return None
    try:
        chain = [st] + [os.stat(d) for d in _parents(os.path.dirname(real))]
    except OSError:
        return None
    if not all(_not_plantable(item) for item in chain):
        return None
    return real


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
            error = "Could not connect to Blender MCP. Check the Unsloth backend installation and retry.",
        )
    result = McpServerProbeResult(ok = True, tool_count = len(tools))
    if check_bridge:
        try:
            binary = await asyncio.wait_for(_bridge_version(settings.port), timeout = 5)
            result.blender_ready = True
            note_connected("blender")
            # An unpacked Linux tarball is not on PATH: report the running Blender for the CLI tools.
            own = _own_blender(binary)
            if not settings.blender_path and not shutil.which("blender") and own:
                result.blender_path = own
        except Exception:
            result.blender_ready = False
            result.blender_error = f"Blender is not connected on port {settings.port}. Enable its MCP add-on, then retry."
    if on_tools is not None:
        on_tools(tools)
    return result
