# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who is calling an MCP tool. The gate validates the key and records the caller in the request scope; tools read it back here, because fastmcp's own header accessor drops ``Authorization``. The chat's in-process bridge has no HTTP request, so it binds its caller in a ContextVar instead."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional

STATE_KEY = "unsloth_mcp"


@dataclass(frozen = True)
class Caller:
    token: str = field(repr = False)
    account_id: str
    # The outer /mcp request was a direct loopback connection, so host file paths are the caller's own.
    direct_local: bool
    # scheme://host[:port] of the outer request, for URLs handed back to the agent.
    public_base: str
    # The outer Studio app, so forwarded calls never need to import main.
    studio_app: Any = field(repr = False)
    hf_token: Optional[str] = field(default = None, repr = False)
    # The token ends with this tool call (the chat bridge's per-call key), so nothing may keep it for later work.
    call_scoped: bool = False


def caller_from_scope(scope: dict) -> Optional[Caller]:
    caller = (scope.get("state") or {}).get(STATE_KEY)
    return caller if isinstance(caller, Caller) else None


_bound: ContextVar[Optional[Caller]] = ContextVar("unsloth_mcp_bound_caller", default = None)


CHAT_TOOLS_SWITCH_REFUSED = (
    "An Unsloth chat tool cannot load or switch models: it would replace the model serving this chat. "
    "Leave model unset to use the loaded one, or load the model from the Unsloth UI first."
)


def chat_tools_call() -> bool:
    """True inside a call from the chat's Unsloth tools bridge (never for an /mcp request)."""
    caller = _bound.get()
    return caller is not None and caller.call_scoped


def bind_caller(caller: Caller) -> Token:
    return _bound.set(caller)


def reset_caller(token: Token) -> None:
    _bound.reset(token)


def current_caller() -> Caller:
    from fastmcp.server.dependencies import get_http_request

    try:
        request = get_http_request()
    except RuntimeError:
        caller = _bound.get()
    else:
        # An HTTP request always answers for itself, so /mcp never sees a bound caller.
        caller = caller_from_scope(request.scope)
    if caller is None:
        raise RuntimeError("Unsloth Studio MCP tools need a caller admitted by the /mcp gate.")
    return caller
