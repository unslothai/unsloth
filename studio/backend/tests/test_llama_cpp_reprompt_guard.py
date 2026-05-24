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

# Inject minimal stand-ins ONLY when the real modules are unavailable.
# Using ``setdefault`` with a non-package ``ModuleType`` would otherwise
# poison ``sys.modules`` for any later test that does
# ``from loggers.handlers import ...`` (Python would raise "loggers is
# not a package" because the stub has no ``__path__``).
try:  # noqa: E402
    import loggers  # type: ignore  # real backend package
except ModuleNotFoundError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.__path__ = []  # type: ignore[attr-defined]
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules["loggers"] = _loggers_stub

try:  # noqa: E402
    import structlog  # type: ignore
except ModuleNotFoundError:
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.__path__ = []  # type: ignore[attr-defined]
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    sys.modules["structlog"] = _structlog_stub

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
    """A numbered list paired with explicit plan framing (intent phrase
    followed by a freshness-gated tool-action verb such as
    ``search the web`` / ``fetch the latest`` / ``query the internet``)
    must NOT count as a completed artifact. The list IS the plan, not
    the answer. Bare ``search the docs`` / ``compare versions`` /
    ``verify input`` are intentionally NOT plan framing because real
    answer lists use them."""
    samples = [
        "Here's my plan:\n1. Search the web for the answer.\n2. then summarise.",
        "First, I'll do these:\n1. fetch the latest chart\n2. cross-check",
        "Let me look up the values: fetch the current data first.",
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


def test_plan_framing_requires_apostrophe_in_ill():
    """The ``i['’]ll`` plan-framing alternative requires an apostrophe so
    the regex does not match the word "ill" (sick). Without this, a
    numbered list near "ill" plus a freshness-gated lookup verb would
    be misclassified as a plan and trigger a spurious re-prompt."""
    samples = [
        ("She is ill. Here is the list:\n1. Apple\n2. Orange\n3. Banana", False),
        ("I'll search the web for X:\n1. step\n2. step", True),
        ("I will search the latest docs:\n1. step\n2. step", True),
    ]
    for content, expected in samples:
        got = _would_reprompt(content)
        assert got == expected, f"{content!r} expected reprompt={expected} got {got}"


def test_reprompts_on_all_intent_form_numbered_action_plans():
    """``_PLAN_LIST_FRAMING`` must mirror every intent form that
    ``_INTENT_SIGNAL`` accepts so numbered action plans phrased with
    ``Allow me``, ``I'm going to``, ``I'm gonna``, ``I am gonna``,
    ``I shall``, ``Now I``, ``Next I`` also re-prompt instead of being
    silently classified as completed answers. Each sample pairs the
    intent form with a freshness-gated lookup verb so the cross-check
    against _TOOL_ACTION_VERBS succeeds."""
    samples = [
        "Allow me to do this:\n1. search the web for X\n2. fetch the latest result",
        "I'm going to do this:\n1. search the latest docs\n2. fetch the current result",
        "I'm gonna do this:\n1. search the web for X\n2. fetch the latest result",
        "I am gonna do this:\n1. search the latest docs\n2. fetch the current result",
        "I shall do this:\n1. search the web for X\n2. fetch the latest result",
        "Now I will do these:\n1. search the web\n2. summarise",
        "Next I will do these:\n1. fetch the latest chart\n2. compare",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_no_reprompt_on_plan_titled_final_answer_without_actions():
    """A final answer naturally titled ``Plan:`` / ``My plan:`` /
    ``Approach:`` must NOT wipe. Bare ``Plan:`` / ``Approach:`` is
    deliberately NOT an intent signal in _INTENT_SIGNAL because it
    too often appears as a normal answer heading (lesson plan, meal
    plan, business plan, project plan, ...)."""
    samples = [
        "Plan:\n1. Warm-up: Students review fractions.\n2. Group practice.\n3. Assessment.",
        "My plan:\n1. Breakfast: oatmeal and fruit.\n2. Lunch: rice bowl.\n3. Dinner: lentil soup.",
        "The plan:\n1. Bring umbrellas.\n2. Pack snacks.\n3. Drive carefully.",
    ]
    for s in samples:
        assert not _would_reprompt(s), s


def test_no_reprompt_on_bare_plan_header_action_stall():
    """Bare ``Plan:`` / ``Approach:`` headers paired with tool-action
    verbs are NOT classified as plan stalls. Adding them as intent
    markers caused false positives on legitimate plan answers; we
    accept the smaller false negative (action plans titled only with
    ``Plan:`` slip through) in exchange for not wiping valid answers.
    Plan stalls that use an explicit first-person intent phrase ("I'll
    search...", "First, I'll fetch...") are still caught."""
    samples = [
        "Plan:\n1. search the docs\n2. summarise the result",
        "My plan:\n1. fetch the data\n2. verify the rows",
        "The approach:\n1. look up the value\n2. compare versions",
    ]
    for s in samples:
        assert not _would_reprompt(s), s


def test_no_reprompt_on_here_is_the_plan_prose_answer():
    """``Here is the plan you asked for. ...`` and similar prose
    answers without action verbs must NOT wipe. The action-verb
    lookahead on the ``Here is the plan`` intent branch filters them."""
    samples = [
        "Here is the plan you asked for. It is two pages long and covers Q4 goals.",
        "Here are my steps in plain English. Step one is patience.",
        "Here is a plan for the dinner party. Welcome, eat, dance.",
    ]
    for s in samples:
        assert not _would_reprompt(s), s


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


def test_no_backtrack_on_plan_framing_long_preamble():
    """``_PLAN_LIST_FRAMING`` scans up to _REPROMPT_MAX_CHARS chars between
    the intent phrase and a tool-action verb. A pathological 2000-char
    payload with many false intent triggers must still complete fast."""
    import time

    payload = (
        "Here's my plan:\n" + ("long preamble text. " * 90) + "\nsearch the web for X"
    )
    t0 = time.time()
    _has_answer_artifact(payload)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 50, f"guard took {elapsed_ms:.1f}ms on long-preamble plan"


# ── Closing-fence-must-end-line edge cases ────────────────────────


def test_artifact_regex_rejects_backtick_close_with_trailing_text():
    """``\\n```not actually closed`` must NOT match a closed fence.

    The closing fence must end the line (only trailing whitespace
    before a newline or end-of-string). Otherwise an unclosed fence
    where a later line begins with three backticks plus prose is
    treated as a complete artifact and the re-prompt is wrongly
    suppressed."""
    samples = [
        "First, let me write it.\n```python\nprint('hi')\n```not actually closed",
        "First, let me show:\n```python\nprint('hi')\n```more text after",
    ]
    for s in samples:
        assert not _has_answer_artifact(s), s
        assert _would_reprompt(s), s


def test_artifact_regex_rejects_tilde_close_with_trailing_text():
    """Same rule for tilde fences."""
    text = "First, let me write it.\n~~~python\nprint('hi')\n~~~not actually closed"
    assert not _has_answer_artifact(text)
    assert _would_reprompt(text)


# ── Freshness-gated find / check / verify lookup plans ────────────


def test_reprompts_on_numbered_lookup_plan_with_freshness_verbs():
    """``find the current``, ``check the latest``, ``verify today's`` in a
    numbered plan are tool-lookup framing and STILL re-prompt."""
    samples = [
        "Here's my plan:\n1. Find the current Billboard chart.\n2. Summarise.",
        "First, I'll do these:\n1. Check the latest release notes.\n2. Answer.",
        "Let me proceed:\n1. Verify today's USD/EUR rate.\n2. Cite the source.",
        "I'll do this:\n1. Find the up-to-date docs.\n2. Quote the change.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_no_reprompt_on_numbered_answer_with_bare_find_or_check():
    """Bare ``find`` / ``check`` / ``verify`` without a freshness word
    stay valid answer verbs ("find the bug", "check the answer")."""
    samples = [
        (
            "Here's how I'd debug this:\n"
            "1. Find the failing test.\n"
            "2. Check the stack trace."
        ),
        (
            "First, here are common steps:\n"
            "1. Verify your input.\n"
            "2. Check each assertion."
        ),
    ]
    for s in samples:
        assert _has_answer_artifact(s), s
        assert not _would_reprompt(s), s


# ── CommonMark fences with 4+ delimiters ──────────────────────────


def test_artifact_regex_detects_four_or_more_backticks():
    """CommonMark allows opening fences of 3+ backticks. Models use
    4+ delimiters when the body itself contains a triple fence."""
    samples = [
        "First, let me show.\n````python\nprint('``` inside')\n````",
        "Let me show.\n`````markdown\n```python\nprint(1)\n```\n`````",
    ]
    for text in samples:
        assert _has_answer_artifact(text), text
        assert not _would_reprompt(text), text


def test_artifact_regex_detects_four_or_more_tildes():
    """Same 3+ delimiter rule for tilde fences."""
    text = "First, let me show.\n~~~~python\nprint('hi')\n~~~~"
    assert _has_answer_artifact(text)
    assert not _would_reprompt(text)


# ── Query / consult online sources ────────────────────────────────


def test_reprompts_on_numbered_plan_with_query_consult_synonyms():
    """``query the web`` / ``consult online sources`` are tool-lookup
    synonyms and STILL re-prompt as numbered tool plans."""
    samples = [
        "Here's my plan:\n1. Query the web for today's USD/EUR rate.\n2. Summarize.",
        "Here's my plan:\n1. Consult online sources for the latest release.\n2. Answer.",
        "First, I'll do this:\n1. Query the internet for the current chart.\n2. Summarize.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


# ── Delayed numbered tool action ──────────────────────────────────


def test_reprompts_on_numbered_plan_when_action_after_long_first_item():
    """Plans where the explicit tool action appears beyond the first 80
    chars (long preamble or long item 1) must STILL re-prompt. The
    framing scan needs to cover the whole short candidate, not just the
    nearest 80 chars."""
    samples = [
        (
            "Here's my plan:\n"
            "1. Review the question and identify exactly what current data is "
            "needed before using external sources.\n"
            "2. Search the web for today's USD/EUR rate.\n"
            "3. Answer with a citation."
        ),
        (
            "Here's my plan:\n"
            "1. Clarify the requirements and identify the exact data source "
            "that contains the current numbers.\n"
            "2. Search the web for the current Billboard chart.\n"
            "3. Summarise the answer."
        ),
        (
            "First, I'll explain the process before acting so the user can "
            "follow along safely and so I can avoid using stale information.\n"
            "1. Search the web for the current Billboard chart.\n"
            "2. Summarise the answer."
        ),
    ]
    for s in samples:
        assert _would_reprompt(s), s


# ── Reasoning-only visible-output path ────────────────────────────


def test_reasoning_only_visible_artifact_suppresses_reprompt():
    """When content_accum is empty AND there are no content tokens, the
    backend yields reasoning_accum as plain content. In that case the
    reasoning text IS the user-visible answer and a complete artifact
    inside it should suppress the re-prompt."""
    from core.inference.llama_cpp import _REPROMPT_MAX_CHARS

    content_accum = ""
    reasoning_accum = (
        "First, let me set up pygame.\n"
        "```python\n"
        "import pygame\n"
        "pygame.init()\n"
        "```"
    )
    has_content_tokens = False

    visible = content_accum.strip()
    reasoning = reasoning_accum.strip()
    stripped = visible if visible else reasoning
    artifact_text = (
        visible if visible else (reasoning if not has_content_tokens else "")
    )
    would_reprompt = bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not (artifact_text and _has_answer_artifact(artifact_text))
    )
    assert not would_reprompt


def test_no_reprompt_on_binary_search_algorithm_answer():
    """A final answer that uses ``search`` as an ordinary algorithm verb
    (binary search, linear search, depth-first search, etc.) must NOT
    re-prompt. The lookup gating on ``search`` requires a freshness or
    web/internet target, so ``Search the left half`` stays an answer."""
    samples = [
        (
            "First, use binary search:\n"
            "1. Search the left half.\n"
            "2. Search the right half."
        ),
        (
            "First, here are the debugging steps:\n"
            "1. Search the project for the failing function.\n"
            "2. Check the stack trace.\n"
            "3. Verify your fix with tests."
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_reprompts_on_numbered_plan_with_google_synonym():
    """``google the current X`` reads as an external lookup and STILL
    re-prompts as a numbered tool plan."""
    samples = [
        "Here's my plan:\n1. Google the current Billboard chart.\n2. Summarise.",
        "First, I'll do this:\n1. Investigate the current exchange rate.\n2. Cite source.",
        "Here's my approach:\n1. Research the latest release notes.\n2. Summarise.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_artifact_regex_rejects_shorter_commonmark_closing_fence():
    """Four-or-more delimiter opening fence cannot be closed by fewer
    delimiters. The opener cannot backtrack to three delimiters and
    consume the rest as info-string text."""
    samples = [
        "First, let me show.\n````python\nprint('hi')\n```",
        "First, let me show.\n~~~~python\nprint('hi')\n~~~",
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_artifact_regex_accepts_longer_commonmark_closing_fence():
    """CommonMark allows the closing fence to have MORE delimiters than
    the opener. A 3-backtick opener with a 4-backtick close, or a
    3-tilde opener with a 4-tilde close, is still a complete artifact."""
    samples = [
        "First, let me show.\n```python\nprint('hi')\n````",
        "First, let me show.\n````python\nprint('``` inside')\n`````",
        "First, let me show.\n~~~python\nprint('hi')\n~~~~",
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_reprompts_on_explicit_plan_header_numbered_list():
    """``Here's my plan`` / ``Here's my approach`` is a strong stand-alone
    plan signal. The following numbered list is the plan itself, not a
    final answer, even when no narrow tool-action verb appears."""
    samples = [
        "Here's my plan:\n1. Analyze the request.\n2. Draft the answer.",
        "Here's my plan:\n1. Create the Python file.\n2. Add the game loop.\n3. Test.",
        "Here's my approach:\n1. Outline.\n2. Write.\n3. Review.",
        "Here's the plan:\n1. Define the variables.\n2. Return the result.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_reprompts_on_numbered_plan_with_python_tool_wording():
    """``use python (tool) to ...`` / ``use the python tool`` / ``use the
    search tool`` in a numbered plan still re-prompts."""
    samples = [
        "Here's my plan:\n1. Use Python to calculate the answer.\n2. Return.",
        "First, I'll do this:\n1. Use the python tool to parse the file.\n2. Summarize.",
        "Here's my plan:\n1. Use the search tool.\n2. Summarize.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_no_reprompt_on_lesson_plan_answer_without_explicit_header():
    """A final answer with a ``Plan:`` heading (no ``Here's my``
    possessive) and no tool framing must STILL count as an answer.
    Common cases: lesson plan, workout plan, meal plan."""
    samples = [
        (
            "Plan:\n"
            "1. Warm up for 5 minutes.\n"
            "2. Run for 20 minutes.\n"
            "3. Cool down with stretching."
        ),
        (
            "My weekly plan:\n"
            "1. Monday: rest.\n"
            "2. Tuesday: jog.\n"
            "3. Wednesday: swim."
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_reprompts_on_direct_intent_numbered_local_action_plan():
    """Direct first-person intent (``I'll do this``, ``Let me do this``,
    etc.) followed by a numbered list of work/tool actions is a plan
    stall, not a final answer. Even when no narrow lookup verb appears
    in the items, the model is announcing actions it has not yet
    taken."""
    samples = [
        (
            "First, I'll do this:\n"
            "1. Load the uploaded CSV.\n"
            "2. Compute the total revenue.\n"
            "3. Return the answer."
        ),
        (
            "Let me do this:\n"
            "1. Parse the pasted JSON.\n"
            "2. Calculate the average.\n"
            "3. Explain the result."
        ),
        (
            "First, I'll create a Python game:\n"
            "1. Set up pygame.\n"
            "2. Add the game loop."
        ),
        (
            "First, I'll do these:\n"
            "1. Create the Python file.\n"
            "2. Add the game loop.\n"
            "3. Test it."
        ),
    ]
    for content in samples:
        assert _would_reprompt(content), content


def test_no_reprompt_on_let_me_explain_numbered_answer():
    """``Let me explain`` / ``Let me show`` followed by a numbered
    answer must NOT be misclassified as a plan stall. The verb after
    the intent phrase is not in the work/tool whitelist."""
    samples = [
        (
            "Let me explain in steps:\n"
            "1. Apples are red.\n"
            "2. Bananas are yellow.\n"
            "3. Cherries are red."
        ),
        (
            "Let me show the matches:\n"
            "1. Maroon 5 - Animals.\n"
            "2. Hozier - Take Me to Church."
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_same_line_open_fence_with_numbered_body_still_reprompts():
    """An OPEN code fence on the same line as preceding prose ("First,
    let me write it. ``\\u00e0``text\\n...") still gates the numbered-list
    fallback. The unclosed-fence helper now uses ``search`` so inline
    openers are tracked, not just openers at column 0."""
    content = (
        "First, let me write it. ```text\n" "1. Install dependencies\n" "2. Run the app"
    )
    assert not _has_answer_artifact(content)
    assert _would_reprompt(content)


def test_reprompts_on_first_step_numbered_compute_plan():
    """Bare ``First, [verb]`` / ``Step N: [verb]`` followed by a numbered
    list is a plan stall when the verb implies compute / tool work
    (analyze, parse, calculate, create, etc.). Distinct from
    ``First, use binary search:`` (verb ``use`` not in whitelist)."""
    samples = [
        (
            "First, analyze the uploaded CSV:\n"
            "1. Load the rows.\n"
            "2. Compute the average revenue."
        ),
        (
            "First, parse the pasted JSON:\n"
            "1. Load the object.\n"
            "2. Calculate the total."
        ),
        (
            "First, create the Python game:\n"
            "1. Set up pygame.\n"
            "2. Add the game loop."
        ),
        (
            "Step 1: analyze the uploaded CSV:\n"
            "1. Load rows.\n"
            "2. Compute the total."
        ),
        ("I'll look that up:\n" "1. Search the docs.\n" "2. Summarize the result."),
    ]
    for content in samples:
        assert _would_reprompt(content), content


def test_reprompts_on_incomplete_html_with_inner_numbered_list():
    """Partial markup (open <html> with no </html>) plus a numbered
    list must NOT be treated as a final answer; the markup is still
    being streamed."""
    samples = [
        (
            "First, I'll draft a page.\n"
            "<html><body>\n"
            "1. Section one.\n"
            "2. Section two.\n"
        ),
        ("Let me design a chart.\n" "<svg width='100'>\n" "1. circle.\n" "2. rect."),
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_complete_html_with_trailing_prose_tag_still_counts():
    """A complete <html> answer followed by prose that mentions <html>
    or <svg> tags (explanatory text) stays a complete artifact. The
    unbalanced-tag count is skipped once a real artifact exists so
    common explanatory prose does not falsely wipe valid answers."""
    samples = [
        "Here is the page:\n<html><body>1</body></html>\nUse the <html> tag for the root.",
        "Here is the SVG: <svg width='10'><circle/></svg> Place it inside an <html> page.",
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_reprompts_on_empty_html_or_svg_skeleton_mention():
    """``<html></html>`` / ``<svg></svg>`` with no body content is a
    plan-only mention, not a substantive answer."""
    samples = [
        "First, I'll create an <html></html> skeleton, then add CSS.",
        "First, I'll draft a <svg></svg> icon, then add shapes.",
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_no_reprompt_on_code_fence_containing_markup_literal():
    """A closed code fence whose body contains literal ``<html>``,
    ``<svg>``, ``<body>`` strings is still a complete code answer.
    The unclosed-markup cross-check operates on text with closed
    fences stripped out so code literals do not falsely trip it."""
    samples = [
        (
            "First, let me write the scraper.\n"
            "```python\n"
            "html = '<html><body>'\n"
            "svg = \"<svg width='100'>\"\n"
            "print(html, svg)\n"
            "```"
        ),
        (
            "First, let me write the parser.\n"
            "```javascript\n"
            "const open = '<html>';\n"
            "const fragment = '<svg width=\"10\">';\n"
            "console.log(open, fragment);\n"
            "```"
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_reprompts_on_direct_first_person_read_check_open_plan():
    """Direct first-person intent + open/read/check/review/inspect verbs
    + numbered list is a tool stall. The broader verb set applies to
    first-person intent only; bare ``First, ...`` and ``Step N: ...``
    keep their narrower verb whitelist."""
    samples = [
        (
            "Let me read the uploaded file:\n"
            "1. Identify the columns.\n"
            "2. Return the total."
        ),
        ("I will check the docs:\n" "1. Gather relevant sections.\n" "2. Answer."),
        (
            "First, I'll review the repository:\n"
            "1. Open the relevant file.\n"
            "2. Read the implementation.\n"
            "3. Suggest a fix."
        ),
        (
            "Let me examine the log file:\n"
            "1. Open the log.\n"
            "2. Read the errors.\n"
            "3. Summarize."
        ),
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_no_reprompt_on_html_with_inner_svg_or_self_closing_tag():
    """Complete <html> answers that contain nested SVG / self-closing
    tags are still complete pages. The unbalanced-count cross-check is
    skipped when a real artifact already exists."""
    samples = [
        "<html><body><svg width='10'/></body></html>",
        "<html><body>"
        + "<script>const s = '<svg width=10>';</script>"
        + "</body></html>",
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_no_reprompt_on_complete_artifact_with_prose_tag_mention():
    """Complete code/markup artifacts followed by ordinary prose that
    mentions ``<html>`` or ``<svg>`` tags are not mid-stream output."""
    samples = [
        "<html><body>hi</body></html>\nUse the <html> tag as the root.",
        (
            "First, here is the SVG: <svg width='10'><circle/></svg>\n"
            "Put it inside an <html> page if needed."
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_no_reprompt_on_html_containing_backtick_literal():
    """A complete <html> answer whose body contains a JS string with
    literal backticks is still a complete page. The unclosed-fence
    cross-check operates on text with closed markup stripped out."""
    content = (
        "First, here is the page.\n"
        "<html><body><script>const fence = '```';</script></body></html>"
    )
    assert _has_answer_artifact(content)
    assert not _would_reprompt(content)


def test_empty_markup_before_real_artifact_still_counts_real_artifact():
    """An empty <html></html> / <svg></svg> skeleton that PRECEDES a
    real complete artifact must not hide it. _looks_like_real_artifact
    iterates every match."""
    samples = [
        (
            "First, the minimal skeleton is <html></html>. "
            "Here is the full page: <html><body><h1>Hello</h1></body></html>"
        ),
        (
            "First, the icon skeleton is <svg></svg>. "
            "Here is the full SVG: "
            "<svg width='10'><circle cx='5' cy='5' r='4'/></svg>"
        ),
    ]
    for content in samples:
        assert _has_answer_artifact(content), content
        assert not _would_reprompt(content), content


def test_doctype_empty_html_skeleton_still_reprompts():
    """``<!doctype html><html></html>`` is an empty skeleton even with
    a doctype prefix; the artifact check must reject it."""
    content = (
        "First, I'll create a <!doctype html><html></html> skeleton, then add CSS."
    )
    assert not _has_answer_artifact(content)
    assert _would_reprompt(content)


def test_reprompts_on_bare_intent_colon_numbered_plan():
    """Bare ``I'll:`` / ``Let me:`` immediately followed by a numbered
    list of work verbs is a tool stall regardless of whether the verbs
    sit before or in the list."""
    samples = [
        "I'll:\n1. Open the URL.\n2. Read the page.\n3. Summarize the answer.",
        "First, I'll:\n1. Create the Python file.\n2. Build the game loop.\n3. Test it.",
        "Let me:\n1. Parse the JSON.\n2. Calculate the average.",
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_reprompts_on_intent_plus_numbered_action_items():
    """``First, I'll:\\n1. Load CSV\\n2. Compute total`` is a plan stall:
    the work verbs are in the list ITEMS even though no work verb
    appears between the intent phrase and the list. Verbs are taken
    from the narrow _LOCAL_ACTION_VERBS set (load, parse, calculate,
    compute, analyze, run, execute, fetch, download, query, inspect,
    extract). Bare ``read`` / ``search`` / ``check`` are NOT in the
    set because they appear in non-plan answers too."""
    samples = [
        "First, I'll:\n1. Load the uploaded CSV.\n2. Compute the total revenue.",
        "Let me:\n1. Parse the JSON.\n2. Calculate the average.",
        "Now I:\n1. Inspect the file.\n2. Analyze the rows.",
        "I'll:\n1. Fetch the latest data.\n2. Compute the totals.",
    ]
    for content in samples:
        assert _would_reprompt(content), content


def test_reprompts_on_numbered_compare_or_review_lookup_plan():
    """Freshness-gated ``compare`` / ``review`` lookups read as tool
    plans and STILL re-prompt as numbered plans."""
    samples = [
        "Here's my plan:\n1. Compare the latest release sources.\n2. Summarise.",
        "First, I'll do this:\n1. Review the current documentation.\n2. Answer.",
    ]
    for s in samples:
        assert _would_reprompt(s), s


def test_no_reprompt_on_first_use_binary_search_answer():
    """``First, use binary search:`` is an ordinary algorithm answer.
    ``use`` is not in the direct-numbered-plan verb whitelist so the
    following list stays an answer."""
    content = (
        "First, use binary search:\n"
        "1. Search the left half.\n"
        "2. Search the right half."
    )
    assert _has_answer_artifact(content)
    assert not _would_reprompt(content)


def test_reprompts_when_later_fence_is_open_after_closed_fence():
    """A response with a complete code fence followed by a SECOND,
    unclosed fence is still mid-stream and must re-prompt. The
    `_has_unclosed_code_fence` cross-check must short-circuit even
    after `_HAS_ANSWER_ARTIFACT` finds the first complete fence."""
    content = (
        "First, let me provide two files:\n"
        "```python\n"
        "print('main')\n"
        "```\n"
        "```python\n"
        "print('utils')"
    )
    assert not _has_answer_artifact(content)
    assert _would_reprompt(content)


def test_open_fence_with_inner_numbered_list_still_reprompts():
    """A response that opens a code fence and emits numbered lines INSIDE
    must NOT count those lines as a completed numbered-list answer."""
    samples = [
        (
            "First, let me write it.\n"
            "```text\n"
            "1. Install dependencies\n"
            "2. Run the app"
        ),
        ("Let me draft a checklist.\n" "````markdown\n" "1. step one\n" "2. step two"),
    ]
    for content in samples:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content


def test_hidden_reasoning_artifact_still_reprompts():
    """When content tokens were emitted but content_accum is empty (a
    streaming oddity) and reasoning hides a complete artifact, the user
    sees nothing, so the re-prompt MUST still fire."""
    from core.inference.llama_cpp import _REPROMPT_MAX_CHARS

    content_accum = ""
    reasoning_accum = (
        "First, let me draft it.\n" "```python\n" "print('hidden answer')\n" "```"
    )
    has_content_tokens = True  # content existed but was stripped

    visible = content_accum.strip()
    reasoning = reasoning_accum.strip()
    stripped = visible if visible else reasoning
    artifact_text = (
        visible if visible else (reasoning if not has_content_tokens else "")
    )
    would_reprompt = bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not (artifact_text and _has_answer_artifact(artifact_text))
    )
    assert would_reprompt
