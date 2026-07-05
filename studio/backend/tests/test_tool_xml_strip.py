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
# The lazy ``(.*?)\n\)`` could grab a shorter expression if an arm is ever wrapped;
# pin the DeepSeek + bare-Kimi arms so a silent truncation fails loudly here.
assert "_DS_OPEN_SRC" in _m.group(1) and "tool_call_begin" in _m.group(
    1
), "extracted _TOOL_XML_RE is missing expected arms (extraction truncated?)"
# The regex reuses the parser's shared DeepSeek opener alternation; provide it so the extracted
# ``_re.compile`` expression resolves the same source.
from core.inference.tool_call_parser import _DEEPSEEK_OPEN_RE_SRC as _DS_OPEN_SRC
from core.inference.tool_call_parser import (
    _strip_function_xml_calls,
    _strip_gemma_wrapperless_calls,
    _strip_glm_calls,
    _strip_mistral_closed_calls,
)

from typing import Optional as _Optional

_ns = {
    "_re": _re,
    "_DS_OPEN_SRC": _DS_OPEN_SRC,
    "Optional": _Optional,
    "_strip_mistral_closed_calls": _strip_mistral_closed_calls,
    "_strip_gemma_wrapperless_calls": _strip_gemma_wrapperless_calls,
    "_strip_glm_calls": _strip_glm_calls,
    "_strip_function_xml_calls": _strip_function_xml_calls,
}
exec(f"_TOOL_XML_RE = _re.compile({_m.group(1)})", _ns)
_TOOL_XML_RE = _ns["_TOOL_XML_RE"]

# Signatures may span multiple lines and now carry the enabled_tool_names gate; match
# the whole (possibly multi-line) signature up to ``-> str:`` then the indented body.
_xml_helper = _re.search(
    r"def _strip_tool_xml\((?:.|\n)*?\) -> str:\n(?:    .+\n)+",
    _src,
)
assert _xml_helper, "could not extract _strip_tool_xml source"
assert "_strip_mistral_closed_calls" in _xml_helper.group(
    0
), "extracted _strip_tool_xml no longer runs the Mistral balanced strip"
exec(_xml_helper.group(0), _ns)
_strip_tool_xml = _ns["_strip_tool_xml"]

_helper = _re.search(
    r"def _strip_tool_xml_for_display\((?:.|\n)*?\) -> str:\n(?:    .+\n)+",
    _src,
)
assert _helper, "could not extract _strip_tool_xml_for_display source"
# After the V1 fix the display helper delegates to _strip_tool_xml; confirm the
# extracted body actually reached that call rather than truncating early.
assert "_strip_tool_xml(" in _helper.group(0), "display helper no longer delegates"
exec(_helper.group(0), _ns)
_strip_tool_xml_for_display = _ns["_strip_tool_xml_for_display"]

_gate_src = _re.search(
    r"def _gemma_strip_gate\((?:.|\n)*?\) -> set:\n(?:    .+\n)+",
    _src,
)
assert _gate_src, "could not extract _gemma_strip_gate source"
exec(_gate_src.group(0), _ns)
_gemma_strip_gate = _ns["_gemma_strip_gate"]


# ── Well-formed pairs ─────────────────────────────────────────────


def test_route_display_strip_respects_disabled_auto_heal_contract():
    text = 'literal <tool_call>{"name":"web_search"}</tool_call> survives'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<tool_call>" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


def test_route_display_strip_removes_mistral_tool_calls_with_nested_json():
    # _TOOL_XML_RE has no [TOOL_CALLS] arm, so the helper delegates to _strip_tool_xml for the Mistral
    # balanced-brace strip (a non-greedy \{.*?\} would truncate nested JSON).
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
    # Attribute form ``<function name="...">`` (MiniCPM-5 / MiniMax-M2) must strip from the route too
    # (it previously leaked into the UI); a dotted/hyphenated name also strips.
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


# ── DeepSeek opener variants + bare Kimi (parse/strip symmetry) ──


def test_strips_deepseek_space_opener_variant():
    # The space-separated opener is parsed by the parser, so the display strip
    # must remove it too (the shared opener alternation is reused here).
    text = (
        "pre <｜tool calls begin｜><｜tool▁call▁begin｜>get_x<｜tool▁sep｜>"
        '{"a":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜> post'
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "tool" not in cleaned.replace("post", "").replace("pre", "")
    assert cleaned == "pre  post"


def test_strips_deepseek_escaped_underscore_opener_variant():
    text = (
        "pre <｜tool\\_calls\\_begin｜><｜tool▁call▁begin｜>get_y<｜tool▁sep｜>"
        '{"a":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜> post'
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert cleaned == "pre  post"


def test_strips_bare_kimi_call_without_section_wrapper():
    # Kimi can emit a bare <|tool_call_begin|>...<|tool_call_end|> with no
    # section wrapper; the parser accepts it, so the strip must cover it.
    text = (
        "pre <|tool_call_begin|>functions.get_w:0<|tool_call_argument_begin|>"
        '{"a":1}<|tool_call_end|> post'
    )
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "tool_call_begin" not in cleaned
    assert cleaned == "pre  post"


# ── Llama-3 <|python_tag|> arm bounds on REAL sentinels only ──────


# Llama-3 <|python_tag|> arm bounds on REAL sentinels only
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


def test_glm_call_with_literal_close_tag_in_arg_value_is_stripped_whole():
    # GLM 4.x emits <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value> ...</tool_call>.
    text = (
        "<tool_call>web_search\n<arg_key>query</arg_key>\n"
        "<arg_value>find </tool_call> here</arg_value>\n</tool_call> done"
    )
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "</arg_value>" not in out
    assert "<arg_key>" not in out
    assert out.strip() == "done"


def test_glm_normal_and_qwen_calls_still_stripped_by_route():
    # Regression: a normal GLM call (no literal close tag) and a Qwen
    # <tool_call>{json}</tool_call> are still stripped; trailing prose is kept.
    glm = "<tool_call>get_time\n<arg_key>tz</arg_key>\n<arg_value>UTC</arg_value>\n</tool_call> ok"
    assert _strip_tool_xml_for_display(glm, auto_heal_tool_calls = True).strip() == "ok"
    qwen = '<tool_call>{"name":"web_search","arguments":{"q":"x"}}</tool_call> after'
    assert _strip_tool_xml_for_display(qwen, auto_heal_tool_calls = True).strip() == "after"


def test_route_strip_removes_param_alias_close_tag():
    # The parser accepts the <param name="...">...</param> attribute-form alias of
    # <parameter=...>; the route tail cleanup must strip an orphan </param> close too.
    assert _strip_tool_xml_for_display("answer </param>", auto_heal_tool_calls = True) == "answer "
    assert (
        _strip_tool_xml_for_display("answer </parameter>", auto_heal_tool_calls = True) == "answer "
    )


def test_route_strip_uses_guarded_function_scan_for_literal_nested_markup():
    # A literal <function=...></function> in a value must not truncate the strip: the route runs the
    # parser's guarded function-XML scan before the regex, matching the core strip.
    text = "<function=python><parameter=code><function=evil></function></parameter></function> tail"
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True).strip() == "tail"


def test_route_strip_gates_wrapperless_gemma_by_enabled_tools():
    # The route strip must gate the markerless Gemma call:NAME{...} form on the enabled tool names,
    # like the parser/loop, so a disabled/example name in prose is preserved in ...
    prose = "To document syntax you write call:foo{query:example}. That shows the format."
    assert "call:foo{query:example}" in _strip_tool_xml(prose, {"web_search"})
    # An enabled name is still a real call and stripped.
    assert "call:web_search" not in _strip_tool_xml(
        "Answer. call:web_search{query:x}", {"web_search"}
    )
    # No gate (legacy) strips every closed call.
    assert "call:foo" not in _strip_tool_xml(prose)


def test_gemma_strip_gate_empty_tools_preserves_prose():
    # With NO tools enabled the gate must return an EMPTY set (strip nothing), not None: None falls
    # back to strip-all and deletes an answer that documents the call:NAME{...} syntax.
    assert _gemma_strip_gate([]) == set()
    assert _gemma_strip_gate(None) == set()
    assert _gemma_strip_gate([{"function": {"name": "web_search"}}]) == {"web_search"}
    prose = "To document syntax you write call:foo{query:example}. That shows the format."
    assert "call:foo{query:example}" in _strip_tool_xml(prose, _gemma_strip_gate([]))
    assert "call:foo{query:example}" in _strip_tool_xml(prose, _gemma_strip_gate(None))
    # An enabled tool's real call is still stripped.
    assert "call:web_search" not in _strip_tool_xml(
        "Answer. call:web_search{query:x}",
        _gemma_strip_gate([{"function": {"name": "web_search"}}]),
    )


def test_strip_keeps_prose_after_closed_function_call_with_literal_close():
    # The call ends at its first non-data close: prose after it survives the
    # strip even when it mentions a literal </function>.
    from core.inference.tool_call_parser import strip_tool_markup
    text = (
        "<function=web_search><parameter=query>cats</parameter></function>"
        " Done. The tag </function> closes a call."
    )
    assert strip_tool_markup(text, final = True) == "Done. The tag </function> closes a call."


def test_final_strip_keeps_prose_mentioning_bare_markers():
    # A false-alarm marker in a normal answer must not lose everything after
    # it; only text that looks like that family's call start drops.
    from core.inference.tool_call_parser import strip_tool_markup
    for text in (
        "See [TOOL_CALLS] docs for details. More prose after.",
        "<|python_tag|> is the Llama marker. Explanation continues.",
        "The <|tool_call> opener wraps Gemma calls.",
    ):
        assert strip_tool_markup(text, final = True) == text
    # A bare marker at end-of-text is a fragment and still drops.
    assert strip_tool_markup("Answer text [TOOL_CALLS]", final = True) == "Answer text"


def test_final_strip_still_drops_truncated_marker_calls():
    from core.inference.tool_call_parser import strip_tool_markup
    for text in (
        '[TOOL_CALLS][{"name":"web_search","argu',
        '[TOOL_CALLS]web_search[ARGS]{"q":"x',
        '<|python_tag|>{"name":"web_search","par',
        '<|python_tag|>foo.call(items=["a',
        "<|tool_call>call:web_search{query:tru",
    ):
        assert strip_tool_markup(text, final = True) == ""


def test_chained_bare_json_strip_consumes_all_calls():
    # The loops keep this text as next-turn history: a leftover executed call
    # would be replayed alongside the structured tool_calls.
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    enabled = {"web_search", "python"}
    chained = (
        '{"name":"web_search","parameters":{"q":"first"}};'
        '{"name":"python","parameters":{"code":"x"}}'
    )
    assert strip_leading_bare_json_call(chained, enabled_tool_names = enabled) == ""
    assert (
        strip_leading_bare_json_call(chained + " trailing prose", enabled_tool_names = enabled)
        == "trailing prose"
    )
    # The chain stops at a non-call answer object, which stays visible.
    call_then_answer = (
        '{"name":"web_search","parameters":{"q":"x"}};{"name":"web_search","result":"data"}'
    )
    assert (
        strip_leading_bare_json_call(call_then_answer, enabled_tool_names = enabled)
        == '{"name":"web_search","result":"data"}'
    )
