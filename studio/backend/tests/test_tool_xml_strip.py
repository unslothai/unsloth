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
# The display helper uses the closed-only variant before the last think block; keep it in scope.
_mc = _re.search(r"_TOOL_XML_CLOSED_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _mc, "could not extract _TOOL_XML_CLOSED_RE source"
exec(f"_TOOL_XML_CLOSED_RE = _re.compile({_mc.group(1)})", _ns)
_TOOL_XML_CLOSED_RE = _ns["_TOOL_XML_CLOSED_RE"]

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

# Extract the gate helper and display strip up to the next top-level ``logger =``.
_helper = _re.search(
    r"def _display_tool_name_gate\(.*?(?=\nlogger = get_logger)",
    _src,
    _re.DOTALL,
)
assert _helper, "could not extract display strip helper source"
# The extracted block spans _display_tool_name_gate through _strip_tool_xml (defined before
# ``logger =``); confirm the shared _strip_tool_xml delegate is present.
assert "_strip_tool_xml(" in _helper.group(0), "display helper no longer delegates"
exec(_helper.group(0), _ns)
_strip_tool_xml_for_display = _ns["_strip_tool_xml_for_display"]
_display_tool_name_gate = _ns["_display_tool_name_gate"]

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


def test_route_display_strip_preserves_rehearsal_inside_think():
    # A rehearsed bracket call inside think is reasoning: the block is preserved while a real
    # call outside it still strips.
    text = '<think>plan: search[ARGS]{"q":"x"}</think> answer [TOOL_CALLS]web_search{"q":"y"} tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert '<think>plan: search[ARGS]{"q":"x"}</think>' in out
    assert "[TOOL_CALLS]web_search" not in out
    assert "answer" in out and "tail" in out


def test_route_display_strip_keeps_bare_args_before_think_block():
    # A bare ``foo[ARGS]`` before a think block is prose: EOS-anchored tail arms run only on
    # the last segment (earlier segments use the closed-only regex).
    text = "Please pass foo[ARGS] <think>pause</think> to the template."
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True) == text


def test_route_display_strip_removes_complete_call_before_think_block():
    # A complete bracket call before a think block still strips (balanced scan runs on every segment).
    text = 'before search[ARGS]{"q":"x"} <think>pause</think> after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "search[ARGS]" not in out
    assert "<think>pause</think>" in out
    assert "before" in out and "after" in out


def test_route_display_strip_removes_closed_xml_before_think_block():
    # A closed <tool_call> before a think block is removed in the non-last segment.
    text = 'pre <tool_call>{"name":"x"}</tool_call> <think>p</think> tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out
    assert "<think>p</think>" in out
    assert "pre" in out and "tail" in out


def test_all_route_cleanup_sites_use_protected_display_helper():
    # Every route cleanup site must use _strip_tool_xml_for_display (think-preserving,
    # balanced); raw _TOOL_XML_RE.sub corrupted think rehearsal and trailing prose. The only
    # legitimate raw sub lives inside the helper itself.
    raw_sub_lines = [
        (i, line)
        for i, line in enumerate(_src.splitlines(), 1)
        if "_TOOL_XML_RE.sub(" in line and not line.lstrip().startswith("#")
    ]
    assert len(raw_sub_lines) == 1, (
        "raw _TOOL_XML_RE.sub must appear only inside _strip_tool_xml_for_display; "
        f"found extra call sites: {raw_sub_lines!r}"
    )


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


def test_strips_complete_bracket_tag_keeps_trailing_prose():
    # A complete Mistral call strips only its balanced JSON, leaving following prose intact.
    cleaned = _TOOL_XML_RE.sub("", '[TOOL_CALLS]web_search{"q":"x"} and then prose')
    assert "[TOOL_CALLS]" not in cleaned
    assert "and then prose" in cleaned


def test_strips_unclosed_bracket_tail():
    # Close brace lost to EOS: the truncated tail strips to the end instead of leaking.
    cleaned = _TOOL_XML_RE.sub("", 'here [TOOL_CALLS]web_search{"query":"weather"')
    assert "[TOOL_CALLS]" not in cleaned
    assert cleaned.strip() == "here"


def test_strips_unclosed_rehearsal_tail():
    cleaned = _TOOL_XML_RE.sub("", 'text python[ARGS]{"code":"print(1)"')
    assert "[ARGS]" not in cleaned
    assert cleaned.strip() == "text"


def test_strips_hyphenated_mcp_bracket_name():
    cleaned = _TOOL_XML_RE.sub("", 'x [TOOL_CALLS]mcp__srv__list-issues{"q":"x"}')
    assert "list-issues" not in cleaned
    assert cleaned.strip() == "x"


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


# ── Two-level-nested bracket JSON (balanced-scan strip) ──────────


def test_route_strip_two_level_nested_bracket_keeps_trailing_prose():
    # Two-level-nested args must be removed whole so the trailing prose survives.
    text = 'before [TOOL_CALLS]search{"f":{"g":{"h":1}}} after'
    cleaned = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert cleaned == "before  after"
    assert "[TOOL_CALLS]" not in cleaned


def test_route_strip_two_level_nested_rehearsal_keeps_trailing_prose():
    text = 'note python[ARGS]{"a":{"b":{"c":1}}} done'
    cleaned = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert cleaned == "note  done"
    assert "[ARGS]" not in cleaned


def test_route_strip_removes_call_with_literal_think_in_argument():
    # A literal <think> inside a call argument strips with the call, not as reasoning.
    text = (
        '<tool_call>{"name":"write","arguments":'
        '{"text":"compare <think> and </think> tags"}}</tool_call>'
    )
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out and '"name"' not in out


def test_route_strip_removes_truncated_mistral_array():
    # A canonical array truncated by EOS is stripped by the route fallback like other orphans.
    text = 'before [TOOL_CALLS] [{"name":"a","arguments":{"x":1}}'  # missing ]
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "{" not in out
    assert "before" in out


def test_route_strip_keeps_prose_mentioning_args_marker():
    # ``foo[ARGS] in a sentence`` is prose; the rehearsal arm must not truncate the line.
    text = "Please pass foo[ARGS] to the template and continue reading."
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert out == text


def test_route_strip_handles_mistral_v11_call_id_args_shape():
    # v11 [CALL_ID]/[ARGS] shape (Mistral Small 3.2) must strip whole.
    text = 'before [TOOL_CALLS]web_search[CALL_ID]abc123[ARGS]{"q":"x"} after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "[CALL_ID]" not in out and "[ARGS]" not in out
    assert "before" in out and "after" in out


# ── Mistral [/TOOL_CALLS] closer + literal <think> inside a call ───────────────

from core.tool_healing import strip_tool_call_markup as _strip_tool_call_markup


def test_core_strip_removes_orphan_tool_calls_closer_array_form():
    # The bare v11 [/TOOL_CALLS] closer left by the balanced scan must not leak as content.
    text = '[TOOL_CALLS] [{"name":"x","arguments":{}}][/TOOL_CALLS]'
    assert _strip_tool_call_markup(text, final = True) == ""


def test_core_strip_removes_orphan_tool_calls_closer_named_form_keeps_tail():
    text = '[TOOL_CALLS]web_search{"q":"x"}[/TOOL_CALLS] tail'
    assert _strip_tool_call_markup(text, final = True) == "tail"


def test_core_strip_removes_call_with_literal_think_in_argument():
    # An unclosed literal <think> inside call arguments strips with the call (argument data).
    text = 'before <tool_call>{"name":"write","arguments":{"text":"literal <think> marker"}}</tool_call> after'
    assert _strip_tool_call_markup(text, final = True) == "before  after"


def test_route_display_strip_removes_orphan_tool_calls_closer_array_form():
    text = '[TOOL_CALLS] [{"name":"x","arguments":{}}][/TOOL_CALLS]'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert out.strip() == ""


def test_route_display_strip_removes_orphan_tool_calls_closer_named_form_keeps_tail():
    text = '[TOOL_CALLS]web_search{"q":"x"}[/TOOL_CALLS] tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[/TOOL_CALLS]" not in out
    assert out.strip() == "tail"


def test_incomplete_xml_call_with_literal_think_in_arg_is_stripped():
    # An incomplete <tool_call> holding a literal <think> strips to EOS, not as a reasoning
    # block (the unclosed tail _tool_call_markup_spans previously missed).
    from core.tool_healing import parse_tool_calls_from_text as _parse
    from core.tool_healing import strip_tool_call_markup as _strip

    text = 'before <tool_call>{"name":"write","arguments":{"text":"literal <think> marker"}} after'
    assert [c["function"]["name"] for c in _parse(text)] == ["write"]
    assert _strip(text, final = True) == "before"

    # A real reasoning block with no tool call is still preserved verbatim.
    assert (
        _strip("answer <think>real</think> done", final = True) == "answer <think>real</think> done"
    )

    # A complete call followed by a real reasoning block: call stripped, block kept.
    mixed = '<tool_call>{"name":"a","arguments":{}}</tool_call> mid <think>r</think> end'
    assert _strip(mixed, final = True) == "mid <think>r</think> end"


# ── enabled-tool gate for the ambiguous bare-rehearsal strip (#5704) ──


def test_display_tool_name_gate_returns_active_names_or_none():
    # Empty / no tools -> None (unrestricted; keep the legacy strip-all behavior).
    assert _display_tool_name_gate([]) is None
    assert _display_tool_name_gate(None) is None
    # OpenAI-shaped tool dicts -> set of function names, malformed entries dropped.
    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "run_python"}},
        {"type": "function"},  # no name
        {"nope": 1},  # no function
    ]
    assert _display_tool_name_gate(tools) == {"web_search", "run_python"}


def test_route_display_strip_keeps_inactive_rehearsal_when_gated():
    # P1 #5704: an inactive ``foo[ARGS]{...}`` is prose; the gated strip leaves the sentence intact.
    gate = {"web_search"}
    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert (
        _strip_tool_xml_for_display(text, auto_heal_tool_calls = True, enabled_tool_names = gate)
        == text
    )
    # A bare marker with no JSON body is likewise prose when inactive.
    assert (
        _strip_tool_xml_for_display(
            "use foo[ARGS] here", auto_heal_tool_calls = True, enabled_tool_names = gate
        )
        == "use foo[ARGS] here"
    )


def test_route_display_strip_removes_active_rehearsal_when_gated():
    # Mirror case: an active tool name is a real rehearsal and still strips.
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        'web_search[ARGS]{"query":"x"} done', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "web_search[ARGS]" not in out
    assert out.strip() == "done"


def test_route_display_strip_ungated_strips_all_rehearsal_unchanged():
    # Backwards-compat: with no gate (None) the bare rehearsal strips as before.
    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True).strip() == "is just syntax."
    assert (
        _strip_tool_xml_for_display(
            text, auto_heal_tool_calls = True, enabled_tool_names = None
        ).strip()
        == "is just syntax."
    )


def test_route_display_strip_control_token_stripped_regardless_of_gate():
    # [TOOL_CALLS] is a control token: stripped even when its NAME is not in the gate.
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        '[TOOL_CALLS]foo[ARGS]{"x":1} keep', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "[TOOL_CALLS]" not in out and "foo[ARGS]" not in out
    assert out.strip() == "keep"


def test_core_strip_gates_bare_rehearsal_on_enabled_tools():
    # P1 (#5704): the shared strip gate mirrors the parse gate -- inactive names are prose
    # and preserved, active names strip, ``None`` keeps legacy strip-all.
    from core.tool_healing import strip_tool_call_markup as _strip

    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert _strip(text, final = True, enabled_tool_names = {"web_search"}) == text
    assert (
        _strip('web_search[ARGS]{"q":1} done', final = True, enabled_tool_names = {"web_search"})
        == "done"
    )
    assert _strip(text, final = True).strip() == "is just syntax."
    assert _strip(text, final = True, enabled_tool_names = None).strip() == "is just syntax."


def test_route_display_strip_gate_preserves_inactive_history_rehearsal():
    # The GGUF history sanitiser passes the gate, so a documented inactive shape survives in
    # the replayed prompt context.
    gate = _display_tool_name_gate([{"function": {"name": "web_search"}}])
    text = 'To call it write foo[ARGS]{"x":1} in your reply.'
    assert 'foo[ARGS]{"x":1}' in _strip_tool_xml_for_display(
        text, auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    # An ACTIVE name is still stripped as a real rehearsed call.
    assert "web_search[ARGS]" not in _strip_tool_xml_for_display(
        'Result web_search[ARGS]{"q":"x"} done', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    # No gate (legacy) strips every NAME[ARGS]{...}.
    assert "foo[ARGS]" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


def test_gguf_history_sanitizer_forwards_enabled_tool_names_gate():
    # Wiring guard: the GGUF history strip must forward the display gate like the live strip.
    block = _re.search(
        r"Strip stale tool-call XML from conversation history.*?\.strip\(\)",
        _src,
        _re.DOTALL,
    )
    assert block, "could not locate GGUF history sanitizer block"
    assert "enabled_tool_names" in block.group(
        0
    ), "GGUF history sanitizer must pass enabled_tool_names to _strip_tool_xml_for_display"


def test_route_history_and_passthrough_forward_the_display_gate():
    # The safetensors/Anthropic history sanitisers and the Anthropic non-stream passthrough
    # must forward the gate so inactive examples survive in replayed prompt / final text.
    blocks = {
        "safetensors history": r"Strip stale tool-call XML from prior assistant turns.*?\.strip\(\)",
        "anthropic history": r"Strip stale tool-call XML via the protected display helper.*?\.strip\(\)",
        "anthropic passthrough": r"gated on the declared tools so an\n.*?\.strip\(\)",
    }
    for label, pat in blocks.items():
        m = _re.search(pat, _src, _re.DOTALL)
        assert m, f"could not locate {label} strip block"
        assert "enabled_tool_names" in m.group(
            0
        ), f"{label} must forward enabled_tool_names to _strip_tool_xml_for_display"


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


@pytest.mark.parametrize(
    "text",
    [
        # Prose that merely names a Kimi/DeepSeek marker (no real call follows) must
        # survive: the call-shaped lookahead fires only on a real call or a bare EOF
        # fragment, so an answer discussing the protocol is never truncated.
        "See <|tool_call_begin|> in the docs. More prose after it.",
        "The <|tool_calls_section_begin|> marker opens a batch. Read on.",
        "DeepSeek uses <｜tool▁calls▁begin｜> to start a call block, then continues.",
    ],
)
def test_deepseek_kimi_false_alarm_prose_is_kept(text):
    # Regression for the route arm truncating a prose answer that references a marker
    # without a following call (parser _TOOL_ALL_PATS already had this lookahead).
    assert _TOOL_XML_RE.sub("", text) == text


def test_deepseek_kimi_real_calls_still_strip_after_false_alarm_fix():
    # The lookahead must not weaken real-call stripping: closed, truncated, and bare
    # EOF-fragment forms all still get removed.
    closed = (
        "answer <|tool_call_begin|>functions.get_w:0<|tool_call_argument_begin|>"
        '{"a":1}<|tool_call_end|> tail'
    )
    assert _TOOL_XML_RE.sub("", closed) == "answer  tail"
    eof_fragment = "prefix <|tool_call_begin|>"
    assert _TOOL_XML_RE.sub("", eof_fragment) == "prefix "
    deepseek = (
        "reply <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_x<｜tool▁sep｜>"
        '{"a":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    assert _TOOL_XML_RE.sub("", deepseek) == "reply "


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
