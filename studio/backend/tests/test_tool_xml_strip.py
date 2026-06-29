# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `_TOOL_XML_RE` (routes/inference.py) -- strips tool-call XML that
leaks past the speculative buffer in core/inference/llama_cpp.py when the
open/close pair is split across the visible/DRAIN boundary.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Extract the regex from source (routes module needs heavy stubbing to import).
import re as _re

_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text()
_m = _re.search(r"_TOOL_XML_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _m, "could not extract _TOOL_XML_RE source"
# _strip_tool_xml_for_display now delegates to _strip_tool_xml, which runs the
# parser's Mistral [TOOL_CALLS] balanced strip before _TOOL_XML_RE. Provide both
# into the exec namespace so the extracted helpers resolve.
from core.inference.tool_call_parser import _strip_mistral_closed_calls

_ns = {"_re": _re, "_strip_mistral_closed_calls": _strip_mistral_closed_calls}
exec(f"_TOOL_XML_RE = _re.compile({_m.group(1)})", _ns)
_TOOL_XML_RE = _ns["_TOOL_XML_RE"]

_xml_helper = _re.search(
    r"def _strip_tool_xml\(text: str\) -> str:\n(?:    .+\n)+",
    _src,
)
assert _xml_helper, "could not extract _strip_tool_xml source"
assert "_strip_mistral_closed_calls" in _xml_helper.group(
    0
), "extracted _strip_tool_xml no longer runs the Mistral balanced strip"
exec(_xml_helper.group(0), _ns)

_helper = _re.search(
    r"def _strip_tool_xml_for_display\(text: str, \*, auto_heal_tool_calls: bool\) -> str:\n"
    r"(?:    .+\n)+",
    _src,
)
assert _helper, "could not extract _strip_tool_xml_for_display source"
assert "_strip_tool_xml(" in _helper.group(0), "display helper no longer delegates"
exec(_helper.group(0), _ns)
_strip_tool_xml_for_display = _ns["_strip_tool_xml_for_display"]


# ── Well-formed pairs ─────────────────────────────────────────────


def test_route_display_strip_respects_disabled_auto_heal_contract():
    text = 'literal <tool_call>{"name":"web_search"}</tool_call> survives'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<tool_call>" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


def test_route_display_strip_removes_mistral_tool_calls_with_nested_json():
    # _TOOL_XML_RE has no [TOOL_CALLS] arm; the display helper must delegate to
    # _strip_tool_xml so the Mistral balanced-brace strip runs (a non-greedy
    # \{.*?\} would truncate the nested JSON at the first }).
    text = 'ok [TOOL_CALLS]web_search{"filters":{"date":"2024"},"query":"cats"} tail'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "web_search" not in out, out
    assert out == "ok  tail"


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


def test_strips_function_attribute_form():
    # MiniCPM-5 / MiniMax-M2 emit the attribute form ``<function name="...">``;
    # the parser handles it (_parse_function_xml) but it previously leaked past
    # the route strip into the UI. A dotted/hyphenated name must also strip.
    text = (
        'Sure.\n<function name="get_weather">\n'
        "<parameter=city>\nSydney\n</parameter>\n</function>\nDone."
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<function name=" not in cleaned
    assert "</function>" not in cleaned
    assert "Sure." in cleaned and "Done." in cleaned

    dotted = 'A <function name="srv.list-issues">x</function> B'
    assert _TOOL_XML_RE.sub("", dotted) == "A  B"

    # Auto-Heal-disabled display contract still preserves literal markup.
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<function name=" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


# ── Orphan openings ───────────────────────────────────────────────


def test_strips_orphan_tool_call_no_close():
    text = (
        "Reasoning.\n</think>"
        "<tool_call>\n"
        "<function=web_search>\n"
        "<parameter=query>\nBillboard 2015\n</parameter>\n"
        "</function"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "<function=" not in cleaned
    assert "Reasoning." in cleaned


def test_strips_orphan_function_no_close():
    text = "I'll call python:\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<function=" not in cleaned
    assert "I'll call python:" in cleaned


def test_strips_orphan_only_opening_tag():
    cleaned = _TOOL_XML_RE.sub("", "Search starting.\n<tool_call>")
    assert "<tool_call>" not in cleaned
    assert "Search starting." in cleaned


def test_strips_multiple_orphans():
    text = (
        "First call:\n<tool_call>\n<function=python>\n<parameter=code>\nx=1\n"
        "Second call:\n<function=web_search>\n<parameter=query>\nhi\n"
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "<tool_call>" not in cleaned
    assert "<function=" not in cleaned


# ── Orphan closes ─────────────────────────────────────────────────


def test_strips_orphan_closing_tag():
    # Real shape from Qwen3.6-27B Q8 sweep (open got DRAINED, close leaked).
    text = "...the table rows directly.\n</parameter>\n</function>\n</tool_call><think>Continuing</think>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</tool_call>" not in cleaned
    assert "</function>" not in cleaned
    # Mid-string </parameter> intentionally preserved (see preserve test).


def test_strips_gemma_native_orphan_closing_tag():
    cleaned = _TOOL_XML_RE.sub("", "Tool call drained.<tool_call|>Visible tail.")

    assert "<tool_call|>" not in cleaned
    assert "Tool call drained." in cleaned
    assert "Visible tail." in cleaned


# ── Tail-only </parameter> (PR #5735 follow-up) ───────────────────


def test_strips_tail_only_parameter_orphan():
    # Outer </function></tool_call> truncated by EOS, inner <parameter=...> DRAINED.
    cleaned = _TOOL_XML_RE.sub("", "and the text is not readable.\n</parameter>\n\n")
    assert "</parameter>" not in cleaned
    assert "and the text is not readable." in cleaned


def test_strips_tail_only_parameter_orphan_single_newline():
    cleaned = _TOOL_XML_RE.sub("", "Global Economic Prospects\n</parameter>\n")
    assert "</parameter>" not in cleaned
    assert "Global Economic Prospects" in cleaned


def test_strips_tail_only_parameter_orphan_no_trailing_ws():
    cleaned = _TOOL_XML_RE.sub("", "Final answer.</parameter>")
    assert "</parameter>" not in cleaned
    assert "Final answer." in cleaned


def test_preserves_mid_string_parameter_in_code_sample():
    # Tail-anchor on `</parameter>` so doc/example prose survives.
    text = (
        "Here is the Qwen tool-call format:\n"
        "```xml\n"
        "<tool_call><function=foo><parameter=arg>value</parameter></function></tool_call>\n"
        "```\n"
        "Note the closing </parameter> sits inside <function>."
    )
    cleaned = _TOOL_XML_RE.sub("", text)
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


# ── Preservation (no false positives) ────────────────────────────


def test_preserves_plain_text():
    text = "1. Animals — Maroon 5\n2. Take Me to Church — Hozier"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_code_fences():
    text = "```python\nimport sys\nprint(sys.version)\n```"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_html_in_prose():
    text = "Use the <html> tag for documents."
    assert _TOOL_XML_RE.sub("", text) == text


# ── Real-world leak samples from the 2026-05-22 sweep ────────────


REAL_LEAKS = [
    # Qwen3.5-35B-A3B UD-Q4_K_XL billboard s22 -- orphan open
    'rectly.\n\nLet me try searching for Wikipedia pages that might have weekly chart data for 2015.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"Billboard Hot 100" "2015" "weekly" "chart" "position" "3"\n</parameter>\n</function',
    # Qwen3.6-27B UD-Q2_K_XL billboard s14 -- orphan open
    'arch `site:wikipedia.org "peaked at number 3" "2015" Billboard`\nI\'ll do a quick web search.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"peaked at number 3" Billboard Hot 100 2015 list\n</parameter>\n</function',
    # Qwen3.6-27B UD-Q2_K_XL billboard s15 -- orphan open
    'rd Hot 100 top-ten singles in 2015".\nI\'ll use web_search to find this exact Wikipedia page.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"List of Billboard Hot 100 top-ten singles in 2015" wikipedia\n</parameter>\n</function',
    # Qwen3.6-27B Q8_0 billboard s02 -- orphan close
    "the table rows directly.\n</parameter>\n</function>\n</tool_call><think>The user wants me to list and categorize all songs that charted #3 on the Billboard Hot 100 in 2015. I have been trying to get this data",
    # Qwen3.6-35B-A3B Q8_0 billboard s21 -- orphan close
    "parse it more carefully.\n</parameter>\n</function>\n</tool_call><think>The user wants a list of songs that charted #3 on the Billboard Hot 100 in 2015, categorized.",
]


@pytest.mark.parametrize(
    "leak", REAL_LEAKS, ids = [f"sweep_sample_{i}" for i in range(len(REAL_LEAKS))]
)
def test_real_world_sweep_leaks_get_stripped(leak):
    cleaned = _TOOL_XML_RE.sub("", leak)
    assert "<tool_call>" not in cleaned, f"leak survived: {cleaned!r}"
    assert "<function=" not in cleaned, f"leak survived: {cleaned!r}"


# ── Real-world tail-only </parameter> from gdpval sweep ──────────


# All end-anchored: outer </function></tool_call> truncated by EOS, inner
# <parameter=...> open DRAINED, leaving bare </parameter> tail.
GDPVAL_PARAMETER_LEAKS = [
    # Qwen3.5-27B Q8_0 / worldbank s00
    "the page contains image data and the text is not readable.\n</parameter>\n\n",
    # Qwen3.5-27B Q8_0 / worldbank s42 (preceded by mojibake)
    "...some mojibake content here...\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / coppa s07
    "blocked, while others may still be in effect. The law is currently under further review by the Ninth Circuit.\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / police_training s00
    "comprehensive training report\n</parameter>\n\n",
    # Qwen3.5-27B UD-Q4_K_XL / worldbank s00
    "Global Economic Prospects\nJune 2025\nGlobal Economic Prospects\n</parameter>\n",
    # Qwen3.6-27B Q8_0 / overpass s07
    "Let me create a comprehensive query and instructions document.\n</parameter>\n\n",
]


@pytest.mark.parametrize(
    "leak",
    GDPVAL_PARAMETER_LEAKS,
    ids = [f"gdpval_param_orphan_{i}" for i in range(len(GDPVAL_PARAMETER_LEAKS))],
)
def test_gdpval_parameter_orphans_get_stripped(leak):
    cleaned = _TOOL_XML_RE.sub("", leak)
    assert "</parameter>" not in cleaned, f"leak survived: {cleaned!r}"


# ── Backtracking guards ──────────────────────────────────────────


def test_no_catastrophic_backtracking_on_open_bracket_spam():
    # 256KB of '<' must fail fast (literal mismatch char 2), not backtrack.
    import time

    adv = "<" * (1024 * 256) + "X"
    t0 = time.perf_counter()
    _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"regex took {elapsed*1000:.0f}ms on 256KB '<' spam"


def test_no_catastrophic_backtracking_on_orphan_opening_spam():
    # 1000 unclosed openings: first alt must consume them all greedily.
    import time

    adv = "<tool_call>X" * 1000
    t0 = time.perf_counter()
    cleaned = _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"regex took {elapsed*1000:.0f}ms on 1000x orphan opens"
    assert "<tool_call>" not in cleaned


# ── Llama-3 <|python_tag|> arm bounds on REAL sentinels only ──────


def test_python_tag_strip_consumes_literal_sentinel_in_arg():
    # A <|python_tag|> tool call whose JSON argument carries a literal <|...|>
    # token (here <|cite|>) must be stripped whole. The old `<(?!\|)` arm stopped
    # at any `<|`, leaking the call tail (e.g. `<|cite|> here"}}`) into display.
    text = '<|python_tag|>{"name": "send", "parameters": {"text": "use <|cite|> here"}}'
    cleaned = _TOOL_XML_RE.sub("", text)
    assert cleaned == "", f"python_tag call leaked at literal sentinel: {cleaned!r}"


@pytest.mark.parametrize(
    "sentinel",
    [
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
    ],
)
def test_python_tag_strip_stops_at_real_sentinel(sentinel):
    # A genuine Llama control sentinel still bounds the strip so following
    # assistant text is preserved (the arm must not swallow past it).
    text = f'<|python_tag|>{{"name": "x", "parameters": {{}}}}{sentinel}visible answer'
    cleaned = _TOOL_XML_RE.sub("", text)
    assert (
        cleaned == f"{sentinel}visible answer"
    ), f"strip did not stop at real sentinel {sentinel!r}: {cleaned!r}"


def test_python_tag_strip_restarts_on_second_python_tag():
    # A second <|python_tag|> opens a new tool-call region, so the whole pair is
    # stripped (the arm bounds the first, then the next match consumes the rest).
    text = '<|python_tag|>{"name": "a"}<|python_tag|>{"name": "b"}'
    cleaned = _TOOL_XML_RE.sub("", text)
    assert cleaned == "", f"second python_tag region leaked: {cleaned!r}"
