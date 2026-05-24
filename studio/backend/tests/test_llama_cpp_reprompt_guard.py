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

The guard recognises completed code fences (any markdown info string,
indented closing fence allowed), complete HTML documents, and complete
SVGs as answer artifacts. A numbered list is an artifact only when the
response does NOT also contain plan framing ("Here's my plan", a tool-
action verb following intent phrasing, etc.), so plan-only stalls of
the form ``Here's my plan:\\n1. search\\n2. summarise`` still re-prompt.
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
    _NUMBERED_LIST_ARTIFACT,
    _PLAN_LIST_FRAMING,
    _has_answer_artifact,
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


# ── Code fence artifact detection ──────────────────────────────────


def test_artifact_regex_detects_closed_code_fence():
    """Closed Python code fence is an answer artifact."""
    text = "First, let me set up pygame.\n```python\nimport pygame\npygame.init()\n```"
    assert _has_answer_artifact(text)


def test_artifact_regex_detects_non_alpha_info_strings():
    """Common languages with digits / symbols in the fence info string
    (python3, c++, c#, objective-c, ts-node, bash-session) must all be
    recognised as complete code answers."""
    samples = [
        "First, let me write it.\n```python3\nprint('hi')\n```",
        "First, let me write it.\n```c++\nint main() { return 0; }\n```",
        'First, let me write it.\n```c#\nConsole.WriteLine("hi");\n```',
        'First, let me write it.\n```objective-c\nNSLog(@"hi");\n```',
        "First, let me write it.\n```ts-node\nconsole.log('hi')\n```",
        "First, let me script it.\n```bash-session\n$ echo hi\n```",
        "First, let me show it.\n```python linenums=\"1\"\nprint('hi')\n```",
    ]
    for text in samples:
        assert _has_answer_artifact(text), text
        assert not _would_reprompt(text), text


def test_artifact_regex_detects_indented_close_fence():
    """A closing fence indented under a list / blockquote still counts.
    Common when the model nests code in markdown structure."""
    text = "First, let me show:\n```python\nx = 1\n  ```"
    assert _has_answer_artifact(text)


def test_artifact_regex_detects_tilde_code_fence():
    """CommonMark also allows ``~~~`` fences. Models emit them when the
    body itself contains backticks, e.g. shell or markdown."""
    samples = [
        "First, let me write it.\n~~~python\nprint('hi')\n~~~",
        "First, let me show:\n~~~\nplain block\n~~~",
        "Sure, here is the script.\n~~~bash\necho hi\n~~~",
    ]
    for text in samples:
        assert _has_answer_artifact(text), text
        assert not _would_reprompt(text), text


def test_artifact_regex_ignores_open_code_fence():
    """An UNCLOSED code fence is not yet a complete artifact."""
    text = "Let me set up pygame.\n```python\nimport pygame"
    assert not _has_answer_artifact(text)


def test_artifact_regex_ignores_plain_text():
    """Plain conversational text contains no artifact."""
    text = "First, I will search for the songs that charted #3 in 2015."
    assert not _has_answer_artifact(text)


# ── HTML artifact detection ────────────────────────────────────────


def test_artifact_regex_detects_html_page():
    """Complete HTML pages (doctype optional, </html> required) match."""
    text_a = "<!doctype html><html><body><script>fetch('...')</script></body></html>"
    text_b = "Sure, here is the dashboard:\n<html><body>...</body></html>"
    assert _has_answer_artifact(text_a)
    assert _has_answer_artifact(text_b)


def test_artifact_regex_ignores_incomplete_html_mention():
    """A plan-only mention of <html> / <!doctype> without </html> close
    must NOT be treated as a completed answer. Pre-fix the guard matched
    bare ``<!doctype\\b`` and ``<html\\b`` and suppressed the re-prompt
    even though the model never emitted a complete page."""
    samples = [
        "First, I'll create an <html> skeleton, then add CSS and JavaScript.",
        "First, I'll write a complete <!doctype html> page with a button.",
        "Let me design a <html> structure for the dashboard.",
    ]
    for s in samples:
        assert not _has_answer_artifact(s), s


# ── SVG artifact detection ─────────────────────────────────────────


def test_artifact_regex_detects_complete_svg():
    """A complete <svg>...</svg> is an answer artifact."""
    text = (
        "Here is the sloth SVG:\n"
        "<svg width='200' height='100'>"
        "<circle cx='50' cy='50' r='30'/>"
        "<ellipse cx='100' cy='50' rx='40' ry='20'/>"
        "</svg>"
    )
    assert _has_answer_artifact(text)


def test_artifact_regex_ignores_incomplete_svg():
    text = "Let me draw a sloth: <svg width='200'><circle"
    assert not _has_answer_artifact(text)


# ── Numbered list semantics ────────────────────────────────────────


def test_numbered_list_artifact_regex_matches_two_items():
    """The raw numbered-list pattern still recognises a 2+ item list.
    The artifact decision combines this with plan-framing checks via
    ``_has_answer_artifact``."""
    text = (
        "Let me list these:\n"
        "1. Animals - Maroon 5\n"
        "2. Take Me to Church - Hozier\n"
        "3. Love Me Like You Do - Ellie Goulding\n"
    )
    assert _NUMBERED_LIST_ARTIFACT.search(text)


def test_numbered_list_without_plan_framing_is_artifact():
    """A plain numbered answer (no intent / plan markers) counts as a
    completed artifact."""
    text = (
        "1. Animals - Maroon 5\n"
        "2. Take Me to Church - Hozier\n"
        "3. Drag Me Down - One Direction\n"
    )
    assert _has_answer_artifact(text)
    assert not _PLAN_LIST_FRAMING.search(text)


def test_numbered_list_with_plan_framing_is_NOT_artifact():
    """A numbered list paired with explicit plan framing must NOT count
    as a completed artifact. The list IS the plan, not the answer."""
    samples = [
        # "Here's my plan" / "plan:" / "approach:".
        "Here's my plan:\n1. Search the web\n2. Summarise the result.",
        "Here is the plan:\n1. Look up the date.\n2. Compare versions.",
        "My approach:\n1. Search\n2. Verify\n3. Answer.",
        # Intent phrase + tool-action verb in close proximity.
        "First, I'll do these:\n1. search for the song list\n2. cross-check the chart",
        "Let me look up the values:\n1. fetch the data\n2. compare to baseline",
    ]
    for s in samples:
        assert _PLAN_LIST_FRAMING.search(s), s
        assert not _has_answer_artifact(s), s


# ── End-to-end guard semantics on realistic responses ──────────────


def _would_reprompt(content: str) -> bool:
    """Return True if the re-prompt block at llama_cpp.py would fire."""
    from core.inference.llama_cpp import _REPROMPT_MAX_CHARS

    stripped = content.strip()
    return bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not _has_answer_artifact(stripped)
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
    assert not _would_reprompt(content)


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
    """A list answer without plan framing does NOT re-prompt."""
    content = (
        "Here's my list of #3 hits:\n"
        "1. Animals - Maroon 5\n"
        "2. Take Me to Church - Hozier\n"
        "3. Drag Me Down - One Direction\n"
    )
    assert not _would_reprompt(content)


def test_reprompts_on_plan_only_stall():
    """Response that is purely a plan and no artifact STILL re-prompts."""
    content = "I'll search the web for the answer."
    assert _would_reprompt(content)


def test_reprompts_on_intent_with_open_fence():
    """Open code fence is not a complete artifact, so we still re-prompt."""
    content = "First, let me write the code.\n```python\nimport"
    assert _would_reprompt(content)


def test_reprompts_on_numbered_plan_only_stall():
    """Numbered plan ("Here's my plan: 1. search 2. summarise") STILL
    re-prompts. Pre-fix the numbered-list artifact branch suppressed
    the tool-call nudge, which contradicted the PR's stated invariant."""
    content = (
        "Here's my plan:\n"
        "1. Search the web for the current Billboard Hot 100 2015 data.\n"
        "2. Use python to categorise the matching songs."
    )
    assert _would_reprompt(content)


def test_reprompts_on_intent_with_numbered_action_plan():
    """Numbered list where each item is an action (search, fetch, ...)
    paired with intent phrasing is treated as a plan, not an answer."""
    content = (
        "First, I'll do these:\n"
        "1. Search the web\n"
        "2. Compare the sources\n"
        "3. Answer concisely"
    )
    assert _would_reprompt(content)


def test_reprompts_on_incomplete_html_intent():
    """A plan-only mention of <html> without close STILL re-prompts."""
    content = "First, I'll create an <html> skeleton, then add CSS."
    assert _would_reprompt(content)


def test_reprompts_on_plan_colon_intent():
    """Bare ``Plan:`` / ``Approach:`` at the start of a structured reply
    is now an intent signal so the plan stall re-prompts. Pre-fix the
    response slipped past ``_INTENT_SIGNAL`` entirely."""
    samples = [
        "Plan:\n1. search the docs\n2. summarise",
        "Approach:\n1. fetch the data\n2. compare",
        "Plan: search the docs then summarise",
    ]
    for s in samples:
        assert _INTENT_SIGNAL.search(s), s
        assert _would_reprompt(s), s


def test_reprompts_on_plan_with_extended_action_verbs():
    """The plan-framing verb whitelist also covers think / respond /
    answer / analy[sz]e / explore / outline / gather / query / reason
    so plan stalls phrased with those verbs still re-prompt."""
    samples = [
        "Here is what I will do:\n1. think it through\n2. respond clearly",
        "First, let me reason about this:\n1. weigh options\n2. answer concisely",
        "Now I will analyse this:\n1. break it down\n2. summarise findings",
    ]
    for s in samples:
        assert _would_reprompt(s), s


# ── Cross-platform line endings ────────────────────────────────────


def test_artifact_regex_handles_crlf_code_fence():
    """Windows / CRLF-converted content still detects a closed fence."""
    content = "First, let me code.\r\n```python\r\nimport sys\r\nprint('hi')\r\n```"
    assert _has_answer_artifact(content)


def test_artifact_regex_handles_mixed_lf_crlf():
    """Mixed line endings (real-world: paste-and-edit on Windows)."""
    content = "Here's the code:\r\n```python\nimport sys\r\n```"
    assert _has_answer_artifact(content)


def test_no_reprompt_on_crlf_complete_python_game():
    """End-to-end CRLF: complete fence -> no re-prompt."""
    content = (
        "First, let me set up pygame.\r\n"
        "```python\r\n"
        "import pygame\r\n"
        "pygame.init()\r\n"
        "while True:\r\n"
        "    for e in pygame.event.get():\r\n"
        "        if e.type == pygame.QUIT: break\r\n"
        "```"
    )
    assert not _would_reprompt(content)


# ── ReDoS guards ───────────────────────────────────────────────────


def test_no_backtrack_on_crlf_spam():
    """10K of `\\r\\n` repeats must complete fast.

    The numbered-list alternative previously used greedy `\\s*` which
    O(n^2)-backtracked through embedded `\\r\\n` characters (~630 ms on
    10 KB). The current `[ \\t]*` indent restriction plus length-bounded
    `[\\s\\S]{...}?` runs keep every alternative linear."""
    import time

    payload = "\r\n" * 5000
    t0 = time.time()
    _has_answer_artifact(payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 50, f"guard took {elapsed_ms:.1f}ms on 10KB CRLF spam"


def test_no_backtrack_on_open_html_spam():
    """Many `<html ` openings without `</html>` close must still complete
    quickly. Bounded `[\\s\\S]{0,4000}?` between the open and close caps
    the scan per occurrence."""
    import time

    payload = "<html " * 200  # ~1200 chars, under _REPROMPT_MAX_CHARS
    t0 = time.time()
    _has_answer_artifact(payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 50, f"guard took {elapsed_ms:.1f}ms on <html spam"


def test_no_backtrack_on_doctype_html_alternation_worst_case():
    """The HTML branch is the slowest path because the inner
    ``[\\s\\S]{0,4000}?</html>`` is retried at every ``<html\\b`` anchor.
    With ``<!doctype html><html foo `` repeated under the 2000-char
    gate the worst observed measurement was about 7 ms; assert a
    generous budget so future quantifier changes that drop the inner
    ``{0,4000}`` bound fail loudly."""
    import time

    payload = ("<!doctype html><html foo " * 60)[:1999]
    t0 = time.time()
    _has_answer_artifact(payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 50, f"guard took {elapsed_ms:.1f}ms on doctype/html alt"


def test_no_backtrack_on_tilde_fence_spam():
    """Open ``~~~`` fences without close must terminate quickly."""
    import time

    payload = "~~~a\n" * 400  # ~2000 chars, near _REPROMPT_MAX_CHARS
    t0 = time.time()
    _has_answer_artifact(payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 50, f"guard took {elapsed_ms:.1f}ms on ~~~ spam"
