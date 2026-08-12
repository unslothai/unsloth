# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP helper endpoints for data recipe."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends

from auth.authentication import (
    authenticated_via_api_key,
    require_ui_session_for_local_commands,
)
from core.data_recipe.service import build_mcp_providers, recipe_has_stdio_mcp
from loggers import get_logger
from models.data_recipe import (
    McpToolsListRequest,
    McpToolsListResponse,
    McpToolsProviderResult,
)
from utils.utils import safe_error_detail

logger = get_logger(__name__)
router = APIRouter()

# A stdio provider is a command this host would run, so only a UI session may
# supply one. Annotated, not a Depends default, so a direct call gets False.
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]


@router.post("/mcp/tools", response_model = McpToolsListResponse)
def list_mcp_tools(
    payload: McpToolsListRequest, via_api_key: ViaApiKey = False
) -> McpToolsListResponse:
    # Caller-supplied and probed immediately, so gate before any is built.
    if recipe_has_stdio_mcp({"mcp_providers": payload.mcp_providers}):
        require_ui_session_for_local_commands(via_api_key)
    try:
        from data_designer.engine.mcp import io as mcp_io
    except ImportError as exc:
        logger.error(
            "data_recipe.mcp.dependencies_unavailable",
            error = str(exc),
            exc_info = True,
        )
        return McpToolsListResponse(
            providers = [
                McpToolsProviderResult(
                    name = "",
                    error = "MCP dependencies unavailable.",
                )
            ]
        )

    providers: list[McpToolsProviderResult] = []
    tool_to_providers: dict[str, list[str]] = defaultdict(list)

    from core.inference.mcp_client import stdio_mcp_enabled

    for provider_payload in payload.mcp_providers:
        provider_name = str(provider_payload.get("name", "")).strip()
        if provider_payload.get("provider_type") == "stdio" and not stdio_mcp_enabled():
            providers.append(
                McpToolsProviderResult(
                    name = provider_name,
                    error = "Local (stdio) MCP servers are disabled on this host.",
                )
            )
            continue
        built = build_mcp_providers({"mcp_providers": [provider_payload]})
        if len(built) != 1:
            providers.append(
                McpToolsProviderResult(
                    name = provider_name,
                    error = "Unsupported MCP provider config.",
                )
            )
            continue

        provider = built[0]
        try:
            tools = mcp_io.list_tools(provider, timeout_sec = payload.timeout_sec)
            tool_names = sorted({tool.name for tool in tools if getattr(tool, "name", "")})
            for tool_name in tool_names:
                tool_to_providers[tool_name].append(provider.name)
            providers.append(
                McpToolsProviderResult(
                    name = provider.name,
                    tools = tool_names,
                )
            )
        except Exception as exc:
            logger.error(
                "data_recipe.mcp.list_tools_failed",
                error = str(exc),
                exc_info = True,
            )
            providers.append(
                McpToolsProviderResult(
                    name = provider.name or provider_name,
                    error = safe_error_detail(exc, fallback = "Failed to load tools."),
                )
            )

    duplicate_tools = {
        tool_name: provider_names
        for tool_name, provider_names in sorted(tool_to_providers.items())
        if len(provider_names) > 1
    }

    return McpToolsListResponse(
        providers = providers,
        duplicate_tools = duplicate_tools,
    )
