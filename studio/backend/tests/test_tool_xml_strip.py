# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the leaked-tool-call-XML stripper at routes/inference.py.

The model's chat template instructs it to emit tool calls in an XML
shape like ::

    <tool_call>
    <function=web_search>
    <parameter=query>
    "Billboard Hot 100" "2015" peak 3
    </parameter>
    </function>
    </tool_call>

The speculative-buffer state machine in `core/inference/llama_cpp.py`
detects this and routes the bytes to `tool_calls_acc` instead of
visible content. But the detection only fires once 32 chars have been
buffered, so for models that hop straight into `<tool_call>` mid-stream
after the BUFFERING -> STREAMING transition, the OPENING tags can leak
into `content_accum` before DRAINING engages. The closing `</tool_call>`
then disappears into the silent buffer, leaving an UNTERMINATED
fragment in the user-visible content.

`_TOOL_XML_RE` is the post-stream cleaner that strips this. The original
pattern only matched well-formed `<tool_call>...</tool_call>` pairs;
this test suite pins the relaxed pattern that also strips orphan
openings.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Studio's routes module needs a lot of stubbing to import in isolation;
# pull the regex out directly via the source so this test is hermetic.
import re as _re

_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text()
_m = _re.search(r"_TOOL_XML_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _m, "could not extract _TOOL_XML_RE source"
_ns = {"_re": _re}
exec(f"_TOOL_XML_RE = _re.compile({_m.group(1)})", _ns)
_TOOL_XML_RE = _ns["_TOOL_XML_RE"]


# ── Well-formed tool_call XML is stripped (no regression) ─────────


def test_strips_well_formed_tool_call():
    text = (
        "Let me search.\n"
        "<tool_call>\n"
        "<function=web_search>\n"
        "<parameter=query>\nBillboard 2015\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "Here are the songs:"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "<function=" not in cleaned
    assert "</tool_call>" not in cleaned
    assert "</function>" not in cleaned
    assert "Here are the songs:" in cleaned, "non-XML content must survive"
    assert "Let me search." in cleaned


def test_strips_function_only_well_formed():
    text = "Setup.\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\nDone."
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<function=" not in cleaned
    assert "Setup." in cleaned
    assert "Done." in cleaned


# ── Orphan / unterminated openings (THIS PR) ──────────────────────


def test_strips_orphan_tool_call_no_close():
    """Pre-fix this would survive the strip (regex required close tag)."""
    text = (
        "Reasoning.\n</think>"
        "<tool_call>\n"
        "<function=web_search>\n"
        "<parameter=query>\nBillboard 2015\n</parameter>\n"
        "</function"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned, "orphan <tool_call> must be stripped"
    assert "<function=" not in cleaned
    assert "Reasoning." in cleaned, "leading prose must survive"


def test_strips_orphan_function_no_close():
    text = (
        "I'll call python:\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<function=" not in cleaned
    assert "I'll call python:" in cleaned


def test_strips_orphan_only_opening_tag():
    """Just `<tool_call>` with nothing after it."""
    text = "Search starting.\n<tool_call>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "Search starting." in cleaned


def test_strips_multiple_orphans():
    """Two orphan openings in the same content."""
    text = (
        "First call:\n<tool_call>\n<function=python>\n<parameter=code>\nx=1\n"
        "Second call:\n<function=web_search>\n<parameter=query>\nhi\n"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "<function=" not in cleaned


# ── Mixed: well-formed first, orphan second ───────────────────────


def test_strips_orphan_closing_tag():
    """The opposite case: opening got DRAINED, only the close leaked.

    Real example from the 2026-05-22 sweep on Qwen3.6-27B Q8:
    `...the table rows directly.\\n</parameter>\\n</function>\\n</tool_call><think>...`
    """
    text = "...the table rows directly.\n</parameter>\n</function>\n</tool_call><think>Continuing</think>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</tool_call>" not in cleaned
    assert "</function>" not in cleaned
    # Mid-string </parameter> is intentionally NOT stripped here -- a user
    # could legitimately have <parameter>...</parameter> XML in a code
    # sample. Only the end-of-buffer orphan (test below) is stripped.


# ── Tail-only </parameter> orphan (gdpval sweep, 7/192 trials) ────


def test_strips_tail_only_parameter_orphan():
    """`</parameter>\\n\\n` at end-of-buffer with no enclosing tags.

    Real example from the 2026-05-22 gdpval sweep on
    Qwen3.5-27B Q8 / worldbank seed00 -- the OUTER `</function>\\n</tool_call>`
    was truncated by EOS (max_tokens=4096) and the INNER `<parameter=...>`
    open got DRAINED, leaving just `</parameter>\\n\\n` as the visible tail.
    """
    text = "and the text is not readable.\n</parameter>\n\n"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</parameter>" not in cleaned, "tail orphan </parameter> must be stripped"
    assert "and the text is not readable." in cleaned


def test_strips_tail_only_parameter_orphan_single_newline():
    text = "Global Economic Prospects\n</parameter>\n"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</parameter>" not in cleaned
    assert "Global Economic Prospects" in cleaned


def test_strips_tail_only_parameter_orphan_no_trailing_ws():
    text = "Final answer.</parameter>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</parameter>" not in cleaned
    assert "Final answer." in cleaned


def test_preserves_mid_string_parameter_in_code_sample():
    """A user-supplied XML code sample with </parameter> mid-text must survive.

    Stripping every </parameter> would corrupt documentation pages, model
    chat-template examples, or any prose discussing tool-call XML shapes.
    """
    text = (
        "Here is the Qwen tool-call format:\n"
        "```xml\n"
        "<tool_call><function=foo><parameter=arg>value</parameter></function></tool_call>\n"
        "```\n"
        "Note the closing </parameter> sits inside <function>."
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    # The fenced sample DOES get its <tool_call>...</tool_call> stripped
    # (legacy behaviour, well-formed pattern). What we need is the
    # SECOND `</parameter>` (in prose, mid-text, NOT end-of-buffer) to
    # survive.
    assert "Note the closing </parameter> sits inside" in cleaned


def test_strips_well_formed_then_orphan():
    text = (
        "Round one:\n<tool_call>\n<function=python>\n<parameter=code>\n1\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "Now round two:\n<tool_call>\n<function=web_search>\n<parameter=query>\n"
        "what is X\n</parameter>\n</function"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "<function=" not in cleaned
    assert "Round one:" in cleaned
    assert "Now round two:" in cleaned


# ── No regression on plain text ──────────────────────────────────


def test_preserves_plain_text():
    """Text without XML must be unchanged."""
    text = "1. Animals — Maroon 5\n2. Take Me to Church — Hozier"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_code_fences():
    """Code fences are not tool_call XML and must be untouched."""
    text = "```python\nimport sys\nprint(sys.version)\n```"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_html_in_prose():
    """Inline HTML mentions (not <tool_call> / <function=) survive."""
    text = "Use the <html> tag for documents."
    assert _TOOL_XML_RE.sub("", text) == text


# ── Real-world leak samples from the 2026-05-22 sweep ─────────────


REAL_LEAKS = [
    # Qwen3.5-35B-A3B UD-Q4_K_XL billboard seed 22 -- orphan opening
    'rectly.\n\nLet me try searching for Wikipedia pages that might have weekly chart data for 2015.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"Billboard Hot 100" "2015" "weekly" "chart" "position" "3"\n</parameter>\n</function',
    # Qwen3.6-27B UD-Q2_K_XL billboard seed 14 -- orphan opening
    'arch `site:wikipedia.org "peaked at number 3" "2015" Billboard`\nI\'ll do a quick web search.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"peaked at number 3" Billboard Hot 100 2015 list\n</parameter>\n</function',
    # Qwen3.6-27B UD-Q2_K_XL billboard seed 15 -- orphan opening
    'rd Hot 100 top-ten singles in 2015".\nI\'ll use web_search to find this exact Wikipedia page.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"List of Billboard Hot 100 top-ten singles in 2015" wikipedia\n</parameter>\n</function',
    # Qwen3.6-27B Q8_0 billboard seed 02 -- orphan close
    "the table rows directly.\n</parameter>\n</function>\n</tool_call><think>The user wants me to list and categorize all songs that charted #3 on the Billboard Hot 100 in 2015. I have been trying to get this data",
    # Qwen3.6-35B-A3B Q8_0 billboard seed 21 -- orphan close
    "parse it more carefully.\n</parameter>\n</function>\n</tool_call><think>The user wants a list of songs that charted #3 on the Billboard Hot 100 in 2015, categorized.",
]


@pytest.mark.parametrize(
    "leak", REAL_LEAKS, ids = [f"sweep_sample_{i}" for i in range(len(REAL_LEAKS))]
)
def test_real_world_sweep_leaks_get_stripped(leak):
    """Each of these triggered the XML-leak bug in the 2026-05-22 sweep.

    Pre-fix `_TOOL_XML_RE` left `<tool_call>` / `<function=` in the output
    because the closing `</tool_call>` was lost to the speculative-buffer
    DRAINING path. Post-fix the regex strips from the opening tag all
    the way to end-of-string when no close tag is found.
    """
    cleaned = _TOOL_XML_RE.sub("", leak)
    assert "<tool_call>" not in cleaned, f"leak survived: {cleaned!r}"
    assert "<function=" not in cleaned, f"leak survived: {cleaned!r}"


# ── Real-world tail-only </parameter> orphans (gdpval sweep) ──────


# All 7 of these are end-anchored: `</parameter>` followed only by
# whitespace to end-of-buffer. The opening `<tool_call><function=...>
# <parameter=...>` was DRAINED but `</parameter>...EOS` truncation
# left a bare close visible.
GDPVAL_PARAMETER_LEAKS = [
    # Qwen3.5-27B Q8_0 / worldbank seed00
    "the page contains image data and the text is not readable.\n</parameter>\n\n",
    # Qwen3.5-27B Q8_0 / worldbank seed42 (preceded by mojibake)
    "...some mojibake content here...\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / coppa seed07
    "blocked, while others may still be in effect. The law is currently under further review by the Ninth Circuit.\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / police_training seed00
    "comprehensive training report\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / worldbank seed00
    "Global Economic Prospects\nJune 2025\nGlobal Economic Prospects\n</parameter>\n",
    # Qwen3.6-27B Q8_0 / overpass seed07
    "Let me create a comprehensive query and instructions document.\n</parameter>\n\n",
]


@pytest.mark.parametrize(
    "leak",
    GDPVAL_PARAMETER_LEAKS,
    ids = [f"gdpval_param_orphan_{i}" for i in range(len(GDPVAL_PARAMETER_LEAKS))],
)
def test_gdpval_parameter_orphans_get_stripped(leak):
    """The 6 distinct tail-only </parameter> orphans from the
    2026-05-22 gdpval sweep (7 trial hits across 6 patterns).

    Pre-fix these leaked `</parameter>\\n\\n` visible to the user.
    """
    cleaned = _TOOL_XML_RE.sub("", leak)
    assert "</parameter>" not in cleaned, f"leak survived: {cleaned!r}"


# ── Performance regression guards ────────────────────────────────


def test_no_catastrophic_backtracking_on_open_bracket_spam():
    """A long run of '<' chars with no close tag must not blow up.

    Each '<' could conceivably start a `<tool_call>` or `<function=...>`
    match attempt; we need the engine to fail fast (literal mismatch on
    char 2) rather than backtracking.
    """
    import time

    adv = "<" * (1024 * 256) + "X"  # 256 KB of '<'
    t0 = time.perf_counter()
    _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    # generous bound: 256KB should complete in well under 100ms on any
    # reasonable hardware. Real numbers seen: ~5ms.
    assert elapsed < 0.5, f"regex took {elapsed*1000:.0f}ms on 256KB '<' spam"


def test_no_catastrophic_backtracking_on_orphan_opening_spam():
    """1000 unclosed `<tool_call>X` openings -- non-greedy could regress here.

    Each opening matches via the `\\Z` branch, but only one (the last
    one before EOF). The engine should not retry every prior opening.
    """
    import time

    adv = "<tool_call>X" * 1000
    t0 = time.perf_counter()
    cleaned = _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"regex took {elapsed*1000:.0f}ms on 1000x orphan opens"
    # All openings must be stripped (greedy first alt consumes them all).
    assert "<tool_call>" not in cleaned
