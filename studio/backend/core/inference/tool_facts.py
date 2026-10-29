# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the chat tool loop may assume about a tool, keyed by the function name the model calls."""

from __future__ import annotations

from typing import Any, Mapping

from core.inference.browser_agent import BROWSER_TOOL_NAMES
from core.inference.media_chat_tools import MEDIA_CHAT_TOOLS

# Never deduplicated: an identical call is new work (the page as it is now, another image).
REPEATABLE = BROWSER_TOOL_NAMES | MEDIA_CHAT_TOOLS
# Reading an image is not new work that licenses a rerun.
_BUILT_IN_READS = frozenset({"view_image"})
# MCP functions their server marked readOnlyHint. Trusting the hint only makes deduplication
# stricter: a read never counts as the new work that lets an identical call run again.
_mcp_reads: set[str] = set()


# A tool result carrying one of these tags is blocked until the user acts, so a reply that tells
# the user what to do is the answer, not a stall to push past.
AWAITS_USER = ("[needs_user_load",)
# Blocked on the user only when the hint says so: an agent with load_model can act on it itself.
AWAITS_USER_WHEN_ASKED = ("[not_loaded",)
ASKS_THE_USER = "Ask the user to "


def awaits_user(result: str) -> bool:
    if any(tag in result for tag in AWAITS_USER):
        return True
    return ASKS_THE_USER in result and any(tag in result for tag in AWAITS_USER_WHEN_ASKED)


def note_mcp_tool(name: str, tool: Mapping[str, Any]) -> None:
    annotations = tool.get("annotations")
    if isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True:
        _mcp_reads.add(name)
    else:
        _mcp_reads.discard(name)


def repeatable(name: Any) -> bool:
    return name in REPEATABLE


def reads_only(name: Any) -> bool:
    return name in _BUILT_IN_READS or name in _mcp_reads
