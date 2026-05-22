# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the plan-without-action re-prompt guard.

The re-prompt path in ``LlamaCppEngine.chat_stream`` exists to nudge a
model that described what it *will* do (forward-looking language)
without actually calling a tool. Before the guard added in this PR, the
heuristic only checked ``len(content) < _REPROMPT_MAX_CHARS`` and the
intent regex, which over-fired on long-but-complete responses that
happened to contain phrases like "first" or "let me". Specifically, a
correct Python game answer of the form ::

    First, let me set up pygame.
    ```python
    import pygame; ...
    ```

would still match (length < 2000, intent signal present) and the next
synthetic user turn ("STOP. Do NOT write code or explain.") wiped the
visible code from the conversation.

The new ``_HAS_ANSWER_ARTIFACT`` regex blocks the re-prompt whenever
the response already contains a real answer artifact: a closed code
fence, an HTML page, a complete SVG, or a numbered list of items.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

from core.inference.llama_cpp import (  # noqa: E402
    _HAS_ANSWER_ARTIFACT,
    _INTENT_SIGNAL,
)


# ── _INTENT_SIGNAL still matches plan-only stalls ──────────────────


def test_intent_signal_matches_plan_only_phrases():
    """Original behaviour is preserved: intent regex still matches the
    plan-without-action phrases that motivated the re-prompt."""
    plan_only_samples = [
        "I'll search the web for that.",
        "I will look that up.",
        "I am going to search.",
        "Let me search the web for the answer.",
        "First, I need to look up the date.",
        "Step 1: I'll search for the song list.",
        "Now I need to call the tool.",
        "Here's my plan: search for X.",
    ]
    for s in plan_only_samples:
        assert _INTENT_SIGNAL.search(s), f"_INTENT_SIGNAL should match {s!r}"


def test_intent_signal_ignores_direct_answers():
    """Direct, complete answers do not match the intent regex."""
    direct_samples = [
        "4",
        "Hello!",
        "The answer is 42.",
        "The capital of France is Paris.",
    ]
    for s in direct_samples:
        assert not _INTENT_SIGNAL.search(s), f"_INTENT_SIGNAL must not match {s!r}"


# ── _HAS_ANSWER_ARTIFACT recognises substantive content ────────────


def test_artifact_regex_detects_closed_code_fence():
    """Closed Python code fence is an answer artifact."""
    text = "First, let me set up pygame.\n```python\nimport pygame\npygame.init()\n```"
    assert _HAS_ANSWER_ARTIFACT.search(
        text
    ), "Closed code fence must be detected as an answer artifact"


def test_artifact_regex_detects_html_page():
    """HTML pages (doctype or <html> root) are answer artifacts."""
    text_a = "<!doctype html><html><body><script>fetch('...')</script></body></html>"
    text_b = "Sure, here is the dashboard:\n<html><body>...</body></html>"
    assert _HAS_ANSWER_ARTIFACT.search(text_a)
    assert _HAS_ANSWER_ARTIFACT.search(text_b)


def test_artifact_regex_detects_complete_svg():
    """A complete <svg>...</svg> is an answer artifact."""
    text = (
        "Here is the sloth SVG:\n"
        "<svg width='200' height='100'>"
        "<circle cx='50' cy='50' r='30'/>"
        "<ellipse cx='100' cy='50' rx='40' ry='20'/>"
        "</svg>"
    )
    assert _HAS_ANSWER_ARTIFACT.search(text)


def test_artifact_regex_detects_numbered_list():
    """A list of 2+ numbered items is an answer artifact."""
    text = (
        "Let me list these:\n"
        "1. Animals — Maroon 5\n"
        "2. Take Me to Church — Hozier\n"
        "3. Love Me Like You Do — Ellie Goulding\n"
    )
    assert _HAS_ANSWER_ARTIFACT.search(text)


def test_artifact_regex_ignores_open_code_fence():
    """An UNCLOSED code fence is not yet a complete artifact."""
    text = "Let me set up pygame.\n```python\nimport pygame"
    assert not _HAS_ANSWER_ARTIFACT.search(
        text
    ), "Open code fence must not satisfy the artifact guard"


def test_artifact_regex_ignores_plain_text():
    """Plain conversational text contains no artifact."""
    text = "First, I will search for the songs that charted #3 in 2015."
    assert not _HAS_ANSWER_ARTIFACT.search(text)


# ── End-to-end guard semantics on realistic responses ──────────────


def _would_reprompt(content: str) -> bool:
    """Return True if the re-prompt block at llama_cpp.py would fire."""
    from core.inference.llama_cpp import _REPROMPT_MAX_CHARS

    stripped = content.strip()
    return bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not _HAS_ANSWER_ARTIFACT.search(stripped)
    )


def test_no_reprompt_on_complete_python_game():
    """Response with intent phrasing + complete code does NOT re-prompt."""
    content = (
        "First, let me set up pygame.\n"
        "```python\n"
        "import pygame\n"
        "pygame.init()\n"
        "screen = pygame.display.set_mode((640, 480))\n"
        "while True:\n"
        "    for e in pygame.event.get():\n"
        "        if e.type == pygame.QUIT: break\n"
        "```"
    )
    assert not _would_reprompt(
        content
    ), "Re-prompt must not fire after a complete code block was produced"


def test_no_reprompt_on_complete_svg():
    """Response with intent phrasing + complete SVG does NOT re-prompt."""
    content = (
        "Let me draw a cute sloth:\n"
        "<svg width='100' height='100'>"
        "<circle cx='50' cy='50' r='30' fill='brown'/>"
        "<circle cx='40' cy='45' r='3' fill='black'/>"
        "<circle cx='60' cy='45' r='3' fill='black'/>"
        "<path d='M40 60 Q50 70 60 60' stroke='black' fill='none'/>"
        "</svg>"
    )
    assert not _would_reprompt(content)


def test_no_reprompt_on_numbered_list_answer():
    """Response with intent + numbered list (Billboard-style) does NOT re-prompt."""
    content = (
        "Here's my list of #3 hits:\n"
        "1. Animals — Maroon 5\n"
        "2. Take Me to Church — Hozier\n"
        "3. Drag Me Down — One Direction\n"
    )
    assert not _would_reprompt(content)


def test_reprompts_on_plan_only_stall():
    """Response that is purely a plan and no artifact STILL re-prompts."""
    content = "I'll search the web for the answer."
    assert _would_reprompt(content), "Plan-only stalls must still trigger the re-prompt"


def test_reprompts_on_intent_with_open_fence():
    """Open code fence is not a complete artifact, so we still re-prompt."""
    content = "First, let me write the code.\n```python\nimport"
    assert _would_reprompt(content)
