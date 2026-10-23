# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hide and refuse the MCP tools the owner turned off in Settings > MCP, and the prompts that read them. The list is installation-wide and read as the owner, so it works with or without a request."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastmcp.exceptions import PromptError
from fastmcp.server.middleware import Middleware

from studio_mcp.errors import tool_error
from studio_mcp.prompts import PROMPT_REQUIRES
from utils import mcp_access

TAG = "tool_disabled"


def disabled_message(name: str) -> str:
    return f"{name} is turned off in Unsloth Studio. The owner can turn it on in Settings > MCP."


async def _disabled() -> Optional[frozenset[str]]:
    # sqlite, so never on the event loop.
    return await asyncio.to_thread(mcp_access.disabled_tools)


def _off(name: str, disabled: Optional[frozenset[str]]) -> bool:
    return disabled is None or name in disabled


def _prompt_off(name: str, disabled: Optional[frozenset[str]]) -> list[str]:
    return sorted(tool for tool in PROMPT_REQUIRES.get(name, ()) if _off(tool, disabled))


class StudioToolAccess(Middleware):
    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        disabled = await _disabled()
        return [tool for tool in tools if not _off(tool.name, disabled)]

    async def on_call_tool(self, context, call_next):
        from mcp_server import tool_names

        name = context.message.name
        # An unregistered name falls through to fastmcp's own "Unknown tool".
        if name in tool_names() and _off(name, await _disabled()):
            raise tool_error(disabled_message(name), tag = TAG)
        return await call_next(context)

    async def on_list_prompts(self, context, call_next):
        prompts = await call_next(context)
        disabled = await _disabled()
        return [prompt for prompt in prompts if not _prompt_off(prompt.name, disabled)]

    async def on_get_prompt(self, context, call_next):
        name = context.message.name
        off = _prompt_off(name, await _disabled())
        if off:
            raise PromptError(
                f"[{TAG}] The {name} prompt needs {', '.join(off)}, turned off in Unsloth Studio. "
                "The owner can turn them on in Settings > MCP."
            )
        return await call_next(context)
