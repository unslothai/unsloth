# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Backend-neutral tool-call XML parser shared by GGUF and safetensors.
Tolerates missing closing tags in either ``<tool_call>{json}</tool_call>``
or ``<function=name><parameter=k>v...`` shape.
"""

import re

from core import tool_healing as _tool_healing


_TOOL_ALL_PATS = _tool_healing._TOOL_ALL_PATS


def parse_tool_calls_from_text(
    content: str,
    *,
    id_offset: int = 0,
    allow_incomplete: bool = True,
) -> list[dict]:
    return _tool_healing.parse_tool_calls_from_text(
        content,
        id_offset = id_offset,
        allow_incomplete = allow_incomplete,
    )


def strip_tool_markup(text: str, *, final: bool = False) -> str:
    return _tool_healing.strip_tool_call_markup(text, final = final)


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


# ── Plan-without-action re-prompt (shared by the GGUF and safetensors loops) ──
# Forward-looking intent: the model says what it *will* do, not a final answer.
INTENT_SIGNAL = re.compile(
    r"(?i)("
    # Direct intent ("I'll", "Let me"); lookahead drops negated forms
    # ("I will not") so a refusal does not re-prompt.
    r"\b(i['\u2019](ll|m going to|m gonna)|i am (going to|gonna)|i will|i shall|let me|allow me)\b(?!\s+(?:not|never)\b)"
    r"|"
    # Step/plan framing: "First ...", "Step 1:", "Here's my plan"
    r"\b(?:first\b|step \d+:?|here['\u2019]?s (?:my |the |a )?(?:plan|approach))"
    r"|"
    r"\b(?:now i|next i)\b"
    r")"
)
MAX_ACT_REPROMPTS = 1
REPROMPT_MAX_CHARS = 2000


def is_short_intent_without_action(text: str) -> bool:
    stripped = text.strip()
    return 0 < len(stripped) < REPROMPT_MAX_CHARS and INTENT_SIGNAL.search(stripped) is not None


def reprompt_to_act_message(tool_hint: str) -> str:
    """The user message appended when re-prompting a plan-without-action turn."""
    return (
        "You have access to enabled tools. If a tool is needed to satisfy "
        "the user's request or complete the action you described, call "
        f"{tool_hint} now. If no tool is needed, provide the final answer "
        "and follow the user's requested format."
    )


def has_tool_signal(text: str) -> bool:
    """Return True if ``text`` contains any tool-call XML signal."""
    return any(s in text for s in TOOL_XML_SIGNALS)
