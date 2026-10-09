# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Studio MCP tools, one module per area, each with a ``register_*(mcp)``."""

from __future__ import annotations

from typing import Any, Optional

from mcp.types import ToolAnnotations

from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward

# Every tool acts on this Studio only. MCP assumes destructive unless told otherwise, so plain tools say so.
READ_ONLY = ToolAnnotations(readOnlyHint = True, openWorldHint = False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint = False, destructiveHint = True, openWorldHint = False)
WRITES = ToolAnnotations(readOnlyHint = False, destructiveHint = False, openWorldHint = False)


async def route_json(
    method: str,
    path: str,
    *,
    caller: Optional[Caller] = None,
    hints: Optional[dict[int, str]] = None,
    **kwargs: Any,
) -> Any:
    """Forward one call as the MCP caller and return its JSON, or raise a ToolError with ``hints`` guidance."""
    response = await forward(caller or current_caller(), method, path, **kwargs)
    return raise_for_route(response, hints = hints)


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None
