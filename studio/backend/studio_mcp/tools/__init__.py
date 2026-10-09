# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Unsloth Studio MCP tools, one module per area, each with a ``register_*(mcp)``."""

from __future__ import annotations

import re
from typing import Any, Optional

from mcp.types import ToolAnnotations

from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward

# Every tool acts on this Studio only. MCP assumes destructive unless told otherwise, so plain tools say so.
READ_ONLY = ToolAnnotations(readOnlyHint = True, openWorldHint = False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint = False, destructiveHint = True, openWorldHint = False)
WRITES = ToolAnnotations(readOnlyHint = False, destructiveHint = False, openWorldHint = False)
EXPORT_STATUS = "/api/export/status"


async def route_json(
    method: str,
    path: str,
    *,
    caller: Optional[Caller] = None,
    hints: Optional[dict[int, str]] = None,
    hint_if: Optional[tuple[str, dict[int, str]]] = None,
    **kwargs: Any,
) -> Any:
    """Forward one call as the MCP caller and return its JSON, or raise a ToolError with ``hints`` guidance."""
    response = await forward(caller or current_caller(), method, path, **kwargs)
    # hint_if is (marker, hints): hints that apply only when the failed answer's text names the marker.
    if hint_if is not None and response.status_code >= 400 and hint_if[0] in response.text:
        hints = {**(hints or {}), **hint_if[1]}
    return raise_for_route(response, hints = hints)


async def try_json(caller: Caller, path: str, **kwargs: Any) -> Any:
    # Best effort: the JSON of a 200 answer, else None; never raises.
    try:
        response = await forward(caller, "GET", path, **kwargs)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def opt_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def strings(values: Any) -> list[str]:
    return [v for v in values if isinstance(v, str)] if isinstance(values, list) else []


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def leaf_name(path: str) -> str:
    # The last segment of a POSIX or Windows path, ignoring trailing separators.
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def present(**values: Any) -> dict[str, Any]:
    # The given values that are not None, in order: the optional fields of a route body.
    return {key: value for key, value in values.items() if value is not None}
