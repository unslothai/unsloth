# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat's "Unsloth Studio" MCP row: the /mcp tools the CHAT profile offers, answered in process.

Nothing here passes the /mcp gate, so the owner's Agent access switch does not apply and nothing new listens on
the network. Each call mints an internal API key for the chat's own account and binds it as the tools' caller, so
every forwarded route is account-scoped and redacted exactly as for an agent's key. The key is deleted when the
call ends, however it ends."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.inference.generation_ticket import GenerationTicket, generation_ticket
from loggers import get_logger

logger = get_logger(__name__)

STUDIO_TOOLS_MCP_PATH = "/mcp/studio-tools"
# The row's URL. Never dialled: mcp_client answers it in process, whatever port Unsloth Studio runs on.
STUDIO_TOOLS_MCP_URL = "http://127.0.0.1" + STUDIO_TOOLS_MCP_PATH

_KEY_MAX_S = 6 * 3600
_KEY_GRACE_S = 300.0
_DEFAULT_WAIT_S = 3600.0
_POLL_S = 0.25
_STOP_WAIT_S = 30.0

_studio_app: Any = None
_mcp: Any = None
_mcp_lock = threading.Lock()
_root_path: contextvars.ContextVar[str] = contextvars.ContextVar(
    "studio_tools_root_path", default = ""
)


class ChatRootPathMiddleware:
    """Remembers each HTTP request's root_path for the chat's tool calls, which run off that request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        token = _root_path.set((scope.get("root_path") or "").rstrip("/"))
        try:
            await self.app(scope, receive, send)
        finally:
            _root_path.reset(token)


def remember_studio_app(app: Any) -> None:
    global _studio_app
    _studio_app = app


def keyless_caller(account: Any) -> bool:
    # Keyless API access binds the owner with no credential; minting it the owner's key would widen its scope.
    return account.via_api_key and account.credential_gen is None


def chat_tools_mcp() -> Any:
    """The chat's own instance, separate from /mcp's, so the external server is untouched."""
    global _mcp
    with _mcp_lock:
        if _mcp is None:
            from mcp_server import create_studio_mcp
            from studio_mcp.profile import CHAT
            _mcp = create_studio_mcp(CHAT)
        return _mcp


def _mint(account: Any, limit: float) -> tuple[str, dict]:
    from auth import storage

    # The stored username, not the context's: the owner's context may carry the default name.
    record = storage.get_account_by_id(account.account_id)
    if record is None:
        raise RuntimeError("the account is missing or disabled")
    expires = datetime.now(timezone.utc) + timedelta(seconds = min(limit + _KEY_GRACE_S, _KEY_MAX_S))
    return storage.create_api_key(
        username = record.username,
        name = storage.STUDIO_CHAT_TOOLS_KEY_NAME,
        expires_at = expires.isoformat(),
        internal = True,
        expect_gen = account.credential_gen,
        # Pinned to this account, as Deep Research does: the username could name a recreated namesake.
        account_id = None if record.is_owner else record.account_id,
    )


def _revoke(key_id: int) -> None:
    from auth import storage
    try:
        storage.delete_internal_api_key(key_id)
    except Exception:  # noqa: BLE001 - the key still expires on its own
        logger.exception("Could not revoke the Unsloth Studio chat tools key %s", key_id)


async def _call(
    caller: Any,
    name: str,
    args: dict,
    ticket: Optional[GenerationTicket] = None,
    config_check: Any = None,
) -> Any:
    from fastmcp import Client

    from studio_mcp.caller import bind_caller, reset_caller

    # Bound before connecting: the in-memory server task is started from this task and copies its context.
    token = bind_caller(caller)
    try:
        with generation_ticket(ticket or GenerationTicket()):
            async with Client(chat_tools_mcp()) as client:
                # As for a remote row: an edit or delete of the row while connecting stops the call.
                if config_check is not None and not await asyncio.to_thread(config_check):
                    raise RuntimeError("MCP server was updated or removed while connecting")
                return await client.call_tool(name, args, raise_on_error = False)
    finally:
        reset_caller(token)


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]
    return value


def _without_nulls(result: Any) -> Any:
    """The model pays for every ``"x":null`` the typed outputs carry, so its JSON text goes without them."""
    content = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None) if getattr(block, "type", None) == "text" else None
        if isinstance(text, str) and text[:1] in "{[":
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                block = block.model_copy(
                    update = {
                        "text": json.dumps(
                            _drop_nulls(parsed), ensure_ascii = False, separators = (",", ":")
                        )
                    }
                )
        content.append(block)
    if content:
        result.content = content
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, (dict, list)):
        result.structured_content = _drop_nulls(structured)
    return result


def _stop(future: Any, ticket: Any) -> None:
    # Cancelling the route alone leaves an image render running on the GPU; stop it and let the route unwind.
    if ticket.stop():
        concurrent.futures.wait([future], timeout = _STOP_WAIT_S)
    future.cancel()


def call(
    name: str,
    args: dict,
    *,
    timeout: Optional[float],
    cancel_event: Any = None,
    ui_resource_uri: Optional[str] = None,
    config_check: Any = None,
) -> str:
    """Run one tool as the current account on the server loop. Never raises: failures are "Error: ..." text."""
    from core.inference.mcp_client import _flatten_result
    from core.inference.media_chat_tools import server_loop
    from studio_mcp.caller import Caller
    from studio_mcp.profile import CHAT
    from utils.account_context import arun_as, current_account

    if cancel_event is not None and cancel_event.is_set():
        return f"Error: MCP tool '{name}' cancelled"
    account = current_account()
    if keyless_caller(account):
        return f"Error: MCP tool '{name}' failed: Unsloth Studio tools need an API key or a signed-in session."
    root_path = _root_path.get()
    app = _studio_app
    loop = server_loop()
    if app is None or loop is None:
        return (
            f"Error: MCP tool '{name}' failed: Unsloth Studio tools are not ready yet. Try again."
        )
    limit = _DEFAULT_WAIT_S if timeout is None or timeout >= 9999 else max(1.0, float(timeout))
    limit = min(limit, _KEY_MAX_S - _KEY_GRACE_S)
    try:
        key, row = _mint(account, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not mint an Unsloth Studio chat tools key: %s", exc)
        return f"Error: MCP tool '{name}' failed: Unsloth Studio could not authorize this call."
    try:
        # Minting can outlast an edit or delete of the row.
        if config_check is not None and not config_check():
            return f"Error: MCP tool '{name}' failed: MCP server was updated or removed"
        # A Stop during the mint must not still start the tool.
        if cancel_event is not None and cancel_event.is_set():
            return f"Error: MCP tool '{name}' cancelled"
        caller = Caller(
            token = key,
            account_id = account.account_id,
            direct_local = False,
            public_base = root_path,
            studio_app = app,
            hf_token = None,
            profile = CHAT,
        )
        # Empty context: another request's fastmcp HTTP request on this thread would outrank the bound caller.
        ticket = GenerationTicket()
        future = contextvars.Context().run(
            asyncio.run_coroutine_threadsafe,
            arun_as(account, _call(caller, name, args, ticket, config_check)),
            loop,
        )
        deadline = time.monotonic() + limit
        while not future.done():
            if cancel_event is not None and cancel_event.is_set():
                _stop(future, ticket)
                return f"Error: MCP tool '{name}' cancelled"
            if time.monotonic() >= deadline:
                _stop(future, ticket)
                return f"Error: MCP tool '{name}' timed out after {limit:g}s"
            concurrent.futures.wait([future], timeout = _POLL_S)
        try:
            result = future.result()
        except concurrent.futures.CancelledError:
            return f"Error: MCP tool '{name}' cancelled"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unsloth Studio chat tool %s failed", name)
            return f"Error: MCP tool '{name}' failed: {exc}"
        return _flatten_result(_without_nulls(result), ui_resource_uri)
    finally:
        # Here, not on the loop: a cancelled or timed-out call still loses its key at once.
        _revoke(row["id"])
