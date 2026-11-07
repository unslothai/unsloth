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
# Unsloth's own tools whose answer changes while a job runs: generate_video says to keep calling get_job.
_STUDIO_POLLS = frozenset({"get_job"})
_mcp_polls: set[str] = set()
# readOnlyHint MCP functions; trusting the hint only makes deduplication stricter.
_mcp_reads: set[str] = set()


# Results blocked until the user acts: telling the user what to do is the answer, not a stall.
AWAITS_USER = ("[needs_user_load",)
# Blocked on the user only when the hint says so: an agent with load_model or unload_model can act on it itself.
AWAITS_USER_WHEN_ASKED = ("[not_loaded", "[multi_model_always")
ASKS_THE_USER = "Ask the user to "


def awaits_user(result: str) -> bool:
    if any(tag in result for tag in AWAITS_USER):
        return True
    return ASKS_THE_USER in result and any(tag in result for tag in AWAITS_USER_WHEN_ASKED)


def note_mcp_tool(
    name: str,
    tool: Mapping[str, Any],
    *,
    studio: bool = False,
) -> None:
    if studio and tool.get("name") in _STUDIO_POLLS:
        _mcp_polls.add(name)
    else:
        _mcp_polls.discard(name)
    annotations = tool.get("annotations")
    if isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True:
        _mcp_reads.add(name)
    else:
        _mcp_reads.discard(name)


def repeatable(name: Any) -> bool:
    return name in REPEATABLE or name in _mcp_polls


def reads_only(name: Any) -> bool:
    return name in _BUILT_IN_READS or name in _mcp_reads
