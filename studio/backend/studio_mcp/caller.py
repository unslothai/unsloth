# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who is calling an MCP tool. The gate validates the key and records the caller in the request scope; tools read it back here, because fastmcp's own header accessor drops ``Authorization``."""

from __future__ import annotations

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


def caller_from_scope(scope: dict) -> Optional[Caller]:
    caller = (scope.get("state") or {}).get(STATE_KEY)
    return caller if isinstance(caller, Caller) else None


def current_caller() -> Caller:
    from fastmcp.server.dependencies import get_http_request

    caller = caller_from_scope(get_http_request().scope)
    if caller is None:
        raise RuntimeError("Studio MCP tools need a caller admitted by the /mcp gate.")
    return caller
