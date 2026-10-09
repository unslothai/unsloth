# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ASGI gate in front of the Studio MCP app. ``/mcp`` is always mounted, so the gate is what keeps it hidden: while the owner's switch is off every request gets the same 404 as an unknown path, before anything about the caller is read. Once on, a browser page from a foreign Origin is refused, and only a valid Studio API key gets through; keyless access, UI sessions and workflow keys never do, whatever the keyless scope."""

from __future__ import annotations

import hmac
import json
import os
from typing import Any, Optional

from fastapi.security.utils import get_authorization_scheme_param
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import URL
from starlette.requests import Request

from studio_mcp.caller import STATE_KEY, Caller
from utils.mcp_access import is_mcp_enabled

LEGACY_TOKEN_ENV = "UNSLOTH_STUDIO_MCP_TOKEN"
HF_TOKEN_HEADER = b"x-unsloth-hf-token"
HF_TOKEN_MAX_LENGTH = 512

NEED_KEY = "Studio MCP needs a Studio API key (sk-unsloth-…). Create one in Settings > API."
ONE_HEADER = "Send one Authorization header"
INVALID_KEY = "Invalid or expired API key"
WORKFLOW_KEY = "Workflow keys cannot use Studio MCP"
RETIRED_TOKEN = "The MCP static token is no longer supported; use a Studio API key (sk-unsloth-…)"
# The mcp SDK refuses bodies over 4 MiB from 1.29 on, as a bare 413 before any tool runs. Holding
# every version to it here keeps the limit the same everywhere and says what to send instead.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
TOO_LARGE = (
    "Studio MCP requests are limited to 4 MiB. Send larger media as a Studio id "
    "(gallery_id, input_id, clip_id or voice_id) or, from the Studio computer, as a file path."
)


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


async def _unauthorized(send: Any, detail: str) -> None:
    # No resource_metadata: advertising OAuth would send agents into a login flow Studio does not have.
    await send_json(send, 401, {"detail": detail}, ((b"www-authenticate", b"Bearer"),))


def _mount_relative_path(scope: dict) -> str:
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _validate_key(token: str) -> tuple[Optional[dict], str]:
    """The key's account record, or None and why. Read-only: no account binding and no last-used write; the forwarded call does both."""
    from auth import storage

    try:
        verified = storage.validate_api_key_account(token, touch = False)
        if verified is None:
            return None, INVALID_KEY
        # Recipe and Deep Research keys carry narrower authority than every MCP tool.
        if storage.is_internal_api_key(token):
            return None, WORKFLOW_KEY
    except Exception:
        return None, INVALID_KEY
    return verified[0], ""


async def _bounded_receive(scope: dict, receive: Any) -> Optional[Any]:
    """A receive that replays the request body, or None when the body is over MAX_REQUEST_BYTES."""
    for name, value in scope.get("headers") or []:
        if name.lower() == b"content-length":
            try:
                if int(value) > MAX_REQUEST_BYTES:
                    return None
            except ValueError:
                return None
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            # The client went away; let the app see the disconnect.
            pending = [message]
            break
        body = message.get("body", b"")
        size += len(body)
        if size > MAX_REQUEST_BYTES:
            return None
        chunks.append(body)
        if not message.get("more_body", False):
            pending = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]
            break

    async def replay() -> dict:
        return pending.pop(0) if pending else await receive()

    return replay


def _is_legacy_token(token: str) -> bool:
    legacy = os.environ.get(LEGACY_TOKEN_ENV, "")
    if not legacy.strip():
        return False
    return hmac.compare_digest(token.encode("utf-8"), legacy.encode("utf-8"))


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

        headers = scope.get("headers") or []
        origins = [value for name, value in headers if name.lower() == b"origin"]
        if origins:
            from utils.origin_policy import mcp_origin_allowed
            outer = URL(scope = scope)
            if len(origins) > 1 or not mcp_origin_allowed(
                origins[0].decode("latin-1"),
                request_scheme = outer.scheme,
                request_netloc = outer.netloc,
                app_state = getattr(scope.get("app"), "state", None),
            ):
                await send_json(send, 403, {"detail": "Origin not allowed for Studio MCP"})
                return

        authorization = [value for name, value in headers if name.lower() == b"authorization"]
        if len(authorization) > 1:
            await _unauthorized(send, ONE_HEADER)
            return
        from utils.keyless_api_access import APPROVED_DUMMY_BEARERS

        scheme, token = get_authorization_scheme_param(
            authorization[0].decode("latin-1") if authorization else ""
        )
        if scheme.lower() != "bearer" or not token or token in APPROVED_DUMMY_BEARERS:
            await _unauthorized(send, NEED_KEY)
            return
        if _is_legacy_token(token):
            await _unauthorized(send, RETIRED_TOKEN)
            return
        from auth.storage import API_KEY_PREFIX

        if not token.startswith(API_KEY_PREFIX):
            await _unauthorized(send, NEED_KEY)
            return
        record, reason = await run_in_threadpool(_validate_key, token)
        if record is None:
            await _unauthorized(send, reason)
            return

        hf_values = [value for name, value in headers if name.lower() == HF_TOKEN_HEADER]
        hf_token = hf_values[0].decode("latin-1").strip() if len(hf_values) == 1 else ""
        if len(hf_values) > 1 or len(hf_token) > HF_TOKEN_MAX_LENGTH:
            await send_json(
                send, 400, {"detail": "Send one X-Unsloth-HF-Token of at most 512 characters"}
            )
            return

        bounded = await _bounded_receive(scope, receive)
        if bounded is None:
            await send_json(send, 413, {"detail": TOO_LARGE})
            return

        from utils.client_ip import is_direct_local_request

        outer = URL(scope = scope)
        scope.setdefault("state", {})[STATE_KEY] = Caller(
            token = token,
            account_id = record["account_id"],
            direct_local = is_direct_local_request(Request(scope)),
            public_base = f"{outer.scheme}://{outer.netloc}",
            studio_app = scope.get("app"),
            hf_token = hf_token or None,
        )
        await self.app(scope, bounded, send)
