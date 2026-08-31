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
indented or blockquoted closing fence), complete HTML documents, and
complete SVGs. Those are the shapes a plan cannot contain. A numbered
list is deliberately not one of them, so plan-only stalls of the form
``Here's my plan:\\n1. search\\n2. summarise`` still re-prompt.
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
    _has_answer_artifact,
)
from core.inference.tool_call_parser import INTENT_SIGNAL as _INTENT_SIGNAL  # noqa: E402


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


# ── Blockquoted fences ─────────────────────────────────────────────


def test_artifact_regex_detects_blockquoted_code_fence():
    """A fence nested in a blockquote closes with ``> ``` ``.
    ``_has_unclosed_code_fence`` already reads those delimiters as
    balanced, so the closed-fence patterns must agree or a complete
    answer is classified as mid-stream and wiped."""
    samples = [
        "First, let me show it.\n> ```python\n> x = 1\n> ```",
        "Let me quote it.\n> ~~~bash\n> echo hi\n> ~~~",
        # The marker is stripped before the column is judged, so a quoted
        # fence keeps the multi-token info strings a column-0 fence allows.
        'First, let me show it.\n> ```python linenums="1"\n> x = 1\n> ```',
        "First, let me show it.\n>> ```python title=demo.py\n>> x = 1\n>> ```",
    ]
    for text in samples:
        assert _has_answer_artifact(text), text
        assert not _would_reprompt(text), text


def test_blockquoted_open_fence_still_reprompts():
    """Stripping the marker must not make an unfinished quoted block
    look complete."""
    text = 'First, let me show it.\n> ```python linenums="1"\n> x = 1'
    assert not _has_answer_artifact(text)
    assert _would_reprompt(text)


def test_quoted_prose_mention_of_backticks_after_code():
    """A blockquoted prose line is still prose: it must not reopen a
    fence that the code block already closed."""
    text = "First, let me show it.\n```python\nx = 1\n```\n> Use ``` for markdown."
    assert _has_answer_artifact(text)
    assert not _would_reprompt(text)


def test_quoted_closer_does_not_close_an_unquoted_fence():
    """A ``>`` closer belongs to a quoted container. Inside an unquoted
    ```markdown block a literal ``> ``` `` line is content, so the block
    is still open and the turn is still mid-stream."""
    text = "First, let me show it.\n```markdown\nExample:\n> ```"
    assert not _has_answer_artifact(text)
    assert _would_reprompt(text)
    # ... and the same output, actually closed, is a complete answer.
    assert _has_answer_artifact(text + "\n```")


def test_balanced_inline_fence_span_is_not_an_opener():
    """``The marker is ```python``` `` is a balanced inline span, not a
    fence. Scanning one delimiter per line read the first run as an
    opener and left the answer looking unfinished."""
    text = "First, let me show it.\n```python\nx = 1\n```\nThe marker is ```python```."
    assert _has_answer_artifact(text)
    assert not _would_reprompt(text)


def test_markup_closing_tag_tolerates_whitespace():
    """`</html >` and `</svg >` are legal per the HTML spec, and must
    count both as artifacts and as closes in the balance check."""
    samples = [
        "First, let me show it.\n<html><body>hi</body></html >",
        'First, let me draw it.\n<svg width="10"><circle r="3"/></svg >',
    ]
    for text in samples:
        assert _has_answer_artifact(text), text
        assert not _would_reprompt(text), text
    # An empty skeleton stays a plan-only mention with the same spacing.
    assert _would_reprompt("First, I'll search the web for current data, then <svg></svg >.")


def test_blockquote_marker_does_not_close_an_open_fence_early():
    """The ``>`` tolerance is for the closing delimiter only; prose that
    merely quotes a fence must not close a block that is still open."""
    text = "First, let me write it.\n```python\nimport sys\n> ``` is the delimiter"
    assert not _has_answer_artifact(text)
    assert _would_reprompt(text)


# ── Numbered lists are not artifacts ───────────────────────────────


def test_numbered_list_is_not_an_answer_artifact():
    """A numbered list is never an answer artifact.

    ``1. ... 2. ...`` is a plan as often as it is an answer and nothing
    in the text separates them, so the guard covers only content a plan
    cannot contain: a closed fence, a complete page, a complete SVG.
    A list answer with no intent phrasing never reached the re-prompt
    anyway; one with intent phrasing behaves as it did before the guard.
    """
    plan_stalls = [
        "Here's my plan:\n1. Search the web for the chart.\n2. Summarise.",
        "I will:\n1. review code.\n2. summarize.",
        "Step 1:\n1. Analyze the request.\n2. Draft the answer.",
    ]
    for content in plan_stalls:
        assert not _has_answer_artifact(content), content
        assert _would_reprompt(content), content

    answers_without_intent = [
        "Here's my list of #3 hits:\n1. Animals - Maroon 5\n2. Take Me to Church",
        "Plan:\n1. Warm up.\n2. Run.\n3. Cool down.",
    ]
    for content in answers_without_intent:
        assert not _would_reprompt(content), content


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
    """The ``i['’]ll`` intent alternative requires an apostrophe so the
    regex does not match the word "ill" (sick)."""
    samples = [
        ("She is ill. Here is the list:\n1. Apple\n2. Orange\n3. Banana", False),
        ("I'll search the web for X:\n1. step\n2. step", True),
        ("I will search the latest docs:\n1. step\n2. step", True),
    ]
    for content, expected in samples:
        got = _would_reprompt(content)
        assert got == expected, f"{content!r} expected reprompt={expected} got {got}"


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


# ── Delayed numbered tool action ──────────────────────────────────


# ── Reasoning-only visible-output path ────────────────────────────


def test_reasoning_only_visible_artifact_suppresses_reprompt():
    """When content_accum is empty AND there are no content tokens, the
    backend yields reasoning_accum as plain content. In that case the
    reasoning text IS the user-visible answer and a complete artifact
    inside it should suppress the re-prompt."""
    from core.inference.llama_cpp import _REPROMPT_MAX_CHARS

    content_accum = ""
    reasoning_accum = "First, let me set up pygame.\n```python\nimport pygame\npygame.init()\n```"
    has_content_tokens = False

    visible = content_accum.strip()
    reasoning = reasoning_accum.strip()
    stripped = visible if visible else reasoning
    artifact_text = visible if visible else (reasoning if not has_content_tokens else "")
    would_reprompt = bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not (artifact_text and _has_answer_artifact(artifact_text))
    )
    assert not would_reprompt


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
        ("My weekly plan:\n1. Monday: rest.\n2. Tuesday: jog.\n3. Wednesday: swim."),
    ]
    for content in samples:
        assert not _would_reprompt(content), content


def test_same_line_open_fence_with_numbered_body_still_reprompts():
    """An OPEN code fence on the same line as preceding prose ("First,
    let me write it. ``\\u00e0``text\\n...") still gates the numbered-list
    fallback. The unclosed-fence helper now uses ``search`` so inline
    openers are tracked, not just openers at column 0."""
    content = "First, let me write it. ```text\n1. Install dependencies\n2. Run the app"
    assert not _has_answer_artifact(content)
    assert _would_reprompt(content)


def test_reprompts_on_first_step_numbered_compute_plan():
    """``First, [action verb]`` / ``Step N:`` framing is an intent signal,
    so the plan that follows it re-prompts."""
    samples = [
        (
            "First, analyze the uploaded CSV:\n"
            "1. Load the rows.\n"
            "2. Compute the average revenue."
        ),
        ("Step 1: analyze the uploaded CSV:\n1. Load rows.\n2. Compute the total."),
        ("I'll look that up:\n1. Search the docs.\n2. Summarize the result."),
    ]
    for content in samples:
        assert _would_reprompt(content), content


def test_reprompts_on_incomplete_html_with_inner_numbered_list():
    """Partial markup (open <html> with no </html>) plus a numbered
    list must NOT be treated as a final answer; the markup is still
    being streamed."""
    samples = [
        ("First, I'll draft a page.\n<html><body>\n1. Section one.\n2. Section two.\n"),
        ("Let me design a chart.\n<svg width='100'>\n1. circle.\n2. rect."),
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


def test_no_reprompt_on_bare_i_need_to_clarification():
    """Bare ``I need to`` clarification or prose answers must NOT
    trigger the re-prompt. The phrase is too common in plain answers."""
    samples = [
        "I need to know your operating system before giving the install command.",
        "I need to be clear: the answer is Paris.",
        'The sentence is: "I need to leave early today."',
    ]
    for content in samples:
        assert not _would_reprompt(content), content


def test_no_reprompt_on_inline_backtick_python_prose_after_code():
    """``Use ```python to start a Python fence.`` is prose after a
    completed answer; it must NOT be treated as an unclosed fence."""
    content = (
        "Here is the snippet:\n"
        "```python\n"
        "print(1)\n"
        "```\n"
        "Use ```python to start a Python block in your reply."
    )
    assert _has_answer_artifact(content)
    assert not _would_reprompt(content)


def test_no_reprompt_on_prose_mention_of_triple_backticks_after_code():
    """Closed code fence followed by prose that describes triple-
    backtick syntax (with leading space after the ticks) must NOT be
    treated as an unclosed fence."""
    content = (
        "Here is the snippet:\n"
        "```python\n"
        "print(1)\n"
        "```\n"
        "Use ``` to start a markdown code fence in your reply."
    )
    assert _has_answer_artifact(content)
    assert not _would_reprompt(content)


def test_no_reprompt_on_html_with_inner_svg_or_self_closing_tag():
    """Complete <html> answers that contain nested SVG / self-closing
    tags are still complete pages. The unbalanced-count cross-check is
    skipped when a real artifact already exists."""
    samples = [
        "<html><body><svg width='10'/></body></html>",
        "<html><body>" + "<script>const s = '<svg width=10>';</script>" + "</body></html>",
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
    content = "First, I'll create a <!doctype html><html></html> skeleton, then add CSS."
    assert not _has_answer_artifact(content)
    assert _would_reprompt(content)


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
        ("First, let me write it.\n```text\n1. Install dependencies\n2. Run the app"),
        ("Let me draft a checklist.\n````markdown\n1. step one\n2. step two"),
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
    reasoning_accum = "First, let me draft it.\n```python\nprint('hidden answer')\n```"
    has_content_tokens = True  # content existed but was stripped

    visible = content_accum.strip()
    reasoning = reasoning_accum.strip()
    stripped = visible if visible else reasoning
    artifact_text = visible if visible else (reasoning if not has_content_tokens else "")
    would_reprompt = bool(
        0 < len(stripped) < _REPROMPT_MAX_CHARS
        and _INTENT_SIGNAL.search(stripped)
        and not (artifact_text and _has_answer_artifact(artifact_text))
    )
    assert would_reprompt
