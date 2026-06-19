# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call XML parser shared by GGUF and safetensors.
Tolerates missing closing tags in either ``<tool_call>{json}</tool_call>``
or ``<function=name><parameter=k>v...`` shape.
"""

from core.tool_healing import (
    _TC_END_TAG_RE,
    _TC_FUNC_CLOSE_RE,
    _TC_FUNC_START_RE,
    _TC_GEMMA_END_TAG_RE,
    _TC_GEMMA_START_RE,
    _TC_JSON_START_RE,
    _TC_PARAM_CLOSE_RE,
    _TC_PARAM_START_RE,
    _TOOL_ALL_PATS,
    _TOOL_CLOSED_PATS,
    _FUNC_CLOSE_TAG,
    _PARAM_CLOSE_TAG,
    _balanced_brace_end,
    _gemma_arguments_to_json,
    _inside_open_parameter,
    parse_tool_calls_from_text,
    strip_tool_call_markup as strip_tool_markup,
)


# Prefixes the streaming buffer watches for to gate in-progress text.
TOOL_XML_SIGNALS = ("<tool_call>", "<|tool_call>", "<function=")


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
