# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Bracket-tag, rehearsal, and thinking-block-strip logic adapted from forge
# (https://github.com/antoinezambelli/forge), Copyright (c) 2025-2026
# Antoine Zambelli, used under the MIT License.

"""
Backend-neutral tool-call parser shared by GGUF and safetensors.

Handles four serializations a local model may emit when bypassing
native function calling:

* ``<tool_call>{json}</tool_call>``
* ``<function=name><parameter=k>v</parameter></function>``
* ``[TOOL_CALLS]name{json}`` (Mistral / Devstral fallback)
* ``name[ARGS]{json}`` (reasoning-model rehearsal)

Closing tags are optional. ``<think>...</think>`` and
``[THINK]...[/THINK]`` blocks are stripped before parsing so calls
emitted after a reasoning preamble are still found.
"""

from core import tool_healing as _tool_healing


_TOOL_ALL_PATS = _tool_healing._TOOL_ALL_PATS


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
    enabled_tool_names=None,
) -> list[dict]:
    return _tool_healing.parse_tool_calls_from_text(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
        enabled_tool_names = enabled_tool_names,
    )


def strip_tool_markup(text: str, *, final: bool = False, enabled_tool_names=None) -> str:
    return _tool_healing.strip_tool_call_markup(
        text, final = final, enabled_tool_names = enabled_tool_names
    )


# Prefixes the streaming buffer watches to gate in-progress text. Bracket-tag forms
# (Mistral [TOOL_CALLS], rehearsal [ARGS]) keep that markup buffered until parsed.
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<|tool_call>",
    "<function=",
    "[TOOL_CALLS]",
    "[ARGS]",
)


# Nudges + error prefixes shared by the GGUF and safetensors loops.
TOOL_ERROR_PREFIXES = (
    "Error",
    "Search failed",
    "Execution error",
    "Blocked:",
    "Exit code",
    "Failed to fetch",
    "Failed to resolve",
    "No query provided",
)

DUPLICATE_CALL_NUDGE = (
    "You already made this exact call. Do not repeat the same tool "
    "call. Try a different approach: fetch a URL from previous "
    "results, use Python to process data you already have, or "
    "provide your final answer now."
)

RENDER_HTML_REPEAT_NUDGE = (
    "Error: render_html was already called for this response. Do not call "
    "render_html again in this response unless the user asks for changes. "
    "Provide the final answer now."
)

TOOL_ERROR_NUDGE = (
    "\n\nThe tool call encountered an issue. Please try a different "
    "approach or rephrase your request."
)

BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you "
    "have found so far, provide your final answer now. Do not call "
    "any more tools."
)

# The exact-args dup guard misses paraphrased re-searches, so also cap executed
# KB searches per turn, then nudge.
RAG_MAX_SEARCHES_PER_TURN = 3
RAG_SEARCH_CAP_NUDGE = (
    "You have already searched the knowledge base several times this turn. "
    "Do not search again. Answer the question using the passages already "
    "retrieved above; if they do not contain the answer, say so plainly."
)


def has_tool_signal(text: str) -> bool:
    """Return True if ``text`` contains any tool-call XML signal."""
    return any(s in text for s in TOOL_XML_SIGNALS)
