# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How a flow run reaches Unsloth Studio: in-process through its own routes, as a short-lived internal API key
minted for the run's account, exactly as the MCP server forwards (see ``studio_mcp/forward.py``). The route
families are Flows' own list: MCP's plus the Library and Flows assets, so MCP's list is never widened for this."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Optional

import httpx

from studio_mcp.caller import Caller
from studio_mcp.forward import ALLOWED_PREFIXES as MCP_PREFIXES
from studio_mcp.forward import FORWARD_BASE_URL, FORWARD_CLIENT, HF_TOKEN_HEADER
from utils.account_context import AccountContext

FLOW_KEY_NAME = "flows workflow"
FLOW_KEY_LIFETIME = timedelta(hours = 24)

FLOW_PREFIXES = tuple(
    dict.fromkeys(
        (
            *MCP_PREFIXES,
            "/api/library/",
            "/api/flows/assets",
            "/api/hub/",
        )
    )
)


FLOW_EXACT = ("/api/library",)


def checked_path(url: httpx.URL) -> str:
    path = url.path
    segments = path.split("/")
    allowed = path in FLOW_EXACT or path.startswith(FLOW_PREFIXES)
    if not allowed or any(s in (".", "..") for s in segments):
        raise RuntimeError(f"Flows do not forward to {path!r}")
    return path


def mint_run_key(account: AccountContext) -> tuple[str, int]:
    """A 24 h internal key for the run's account; the run revokes it when it ends."""
    from auth import storage

    expires_at = (datetime.now(timezone.utc) + FLOW_KEY_LIFETIME).isoformat()
    token, row = storage.create_api_key(
        username = account.username,
        name = FLOW_KEY_NAME,
        expires_at = expires_at,
        internal = True,
        account_id = None if account.is_owner else account.account_id,
    )
    return token, int(row["id"])


def revoke_run_key(key_id: Optional[int]) -> None:
    if key_id is None:
        return
    try:
        from auth import storage
        storage.revoke_internal_api_key(key_id)
    except Exception:
        pass


def make_caller(
    token: str,
    account: AccountContext,
    app: Any,
    hf_token: Optional[str] = None,
) -> Caller:
    return Caller(
        token = token,
        account_id = account.account_id,
        direct_local = False,
        public_base = FORWARD_BASE_URL,
        studio_app = app,
        hf_token = hf_token,
    )


def _headers(caller: Caller, content_type: Optional[str], extra: Optional[dict]) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {caller.token}"}
    if caller.hf_token:
        headers[HF_TOKEN_HEADER] = caller.hf_token
    if content_type:
        headers["Content-Type"] = content_type
    if extra:
        headers.update(extra)
    return headers


def _client(caller: Caller, timeout: Optional[float]) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app = caller.studio_app, client = FORWARD_CLIENT)
    return httpx.AsyncClient(transport = transport, base_url = FORWARD_BASE_URL, timeout = timeout)


async def forward(
    caller: Caller,
    method: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json_body: Any = None,
    content: Optional[bytes] = None,
    content_type: Optional[str] = None,
    files: Any = None,
    data: Optional[dict[str, Any]] = None,
    timeout: Optional[float] = None,
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    """One buffered request to Unsloth Studio as the run's key; its own task keeps the route's account binding out of the node."""

    async def send() -> httpx.Response:
        async with _client(caller, timeout) as client:
            request = client.build_request(
                method,
                path,
                headers = _headers(caller, content_type, headers),
                params = params,
                json = json_body,
                content = content,
                files = files,
                data = data,
            )
            checked_path(request.url)
            return await client.send(request)

    task = asyncio.create_task(send())
    try:
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise


async def stream_sse(
    caller: Caller,
    path: str,
    json_body: Any,
    *,
    headers: Optional[dict[str, str]] = None,
) -> AsyncIterator[dict]:
    """POST and yield each SSE ``data:`` JSON object; raises RouteError on a non-2xx answer."""
    async with _client(caller, None) as client:
        request = client.build_request(
            "POST", path, headers = _headers(caller, "application/json", headers), json = json_body
        )
        checked_path(request.url)
        response = await client.send(request, stream = True)
        try:
            if response.status_code >= 400:
                body = await response.aread()
                raise RouteError(response.status_code, route_detail(body))
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    yield json.loads(payload)
                except ValueError:
                    continue
        finally:
            await response.aclose()


class RouteError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def route_detail(body: bytes) -> str:
    try:
        data = json.loads(body.strip() or b"{}")
    except ValueError:
        return body[:500].decode("utf-8", "replace") or "Unsloth Studio refused the request"
    if isinstance(data, dict):
        detail = data.get("detail")
        if detail is None and isinstance(data.get("error"), dict):
            detail = data["error"].get("message")
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg") if isinstance(d, dict) else d) for d in detail[:5])
        if detail:
            return str(detail)
    return "Unsloth Studio refused the request"


def parse_body(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise RouteError(response.status_code, route_detail(response.content))
    ctype = response.headers.get("content-type", "")
    if "json" in ctype or response.content.lstrip()[:1] in (b"{", b"["):
        try:
            return json.loads(response.content.strip() or b"null")
        except ValueError:
            pass
    return response.content
