# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ASGI gate in front of the Studio MCP app. ``/mcp`` is always mounted, so the gate is what keeps it hidden: while the owner's switch is off every request gets the same 404 as an unknown path, before anything about the caller is read."""

from __future__ import annotations

import json
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import URL

from utils.mcp_access import is_mcp_enabled


async def send_json(
    send: Any,
    status: int,
    body: dict,
    headers: tuple = (),
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), *headers],
        }
    )
    await send({"type": "http.response.body", "body": json.dumps(body).encode()})


async def deny_all(scope: dict, receive: Any, send: Any) -> None:
    """Stands in for the MCP app when no static token is configured, so turning the switch on never opens it."""
    await send_json(
        send, 401, {"detail": "MCP bearer token required"}, ((b"www-authenticate", b"Bearer"),)
    )


def _mount_relative_path(scope: dict) -> str:
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


class StudioMcpGate:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 4404})
            return

        # The decisions mount only matches with the trailing slash, so /mcp/decisions lands here.
        if _mount_relative_path(scope).rstrip("/") == "/decisions":
            target = URL(scope = {**scope, "path": scope["path"].rstrip("/") + "/"})
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", str(target).encode("latin-1"))],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        if not await run_in_threadpool(is_mcp_enabled):
            await send_json(send, 404, {"detail": "Not Found"})
            return

        await self.app(scope, receive, send)
