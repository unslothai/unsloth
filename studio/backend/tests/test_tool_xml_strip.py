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
_ns = {"_re": _re}
exec(f"_TOOL_XML_RE = _re.compile({_m.group(1)})", _ns)
_TOOL_XML_RE = _ns["_TOOL_XML_RE"]
# The display helper uses the closed-only variant on segments before the last <think>
# block, so the extracted helper needs it in scope to run.
_mc = _re.search(r"_TOOL_XML_CLOSED_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _mc, "could not extract _TOOL_XML_CLOSED_RE source"
exec(f"_TOOL_XML_CLOSED_RE = _re.compile({_mc.group(1)})", _ns)
_TOOL_XML_CLOSED_RE = _ns["_TOOL_XML_CLOSED_RE"]
# Extract the gate helper and the display strip, capturing up to the next top-level
# ``logger =`` rather than pinning the multi-line signature.
_helper = _re.search(
    r"def _display_tool_name_gate\(.*?(?=\nlogger = get_logger)",
    _src,
    _re.DOTALL,
)
assert _helper, "could not extract display strip helper source"
exec(_helper.group(0), _ns)
_strip_tool_xml_for_display = _ns["_strip_tool_xml_for_display"]
_display_tool_name_gate = _ns["_display_tool_name_gate"]


# ── Well-formed pairs ─────────────────────────────────────────────


def test_route_display_strip_respects_disabled_auto_heal_contract():
    text = 'literal <tool_call>{"name":"web_search"}</tool_call> survives'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<tool_call>" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


def test_route_display_strip_preserves_rehearsal_inside_think():
    # A rehearsed bracket call inside <think> is reasoning, not markup to strip; the
    # route display strip must preserve the block (consistent with
    # strip_tool_call_markup) while still stripping a real call outside it.
    text = '<think>plan: search[ARGS]{"q":"x"}</think> answer [TOOL_CALLS]web_search{"q":"y"} tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert '<think>plan: search[ARGS]{"q":"x"}</think>' in out
    assert "[TOOL_CALLS]web_search" not in out
    assert "answer" in out and "tail" in out


def test_route_display_strip_keeps_bare_args_before_think_block():
    # A bare ``foo[ARGS]`` (no JSON body) before a <think> block is prose, not a
    # truncated call: the open-ended tail arms are anchored to true EOS, so they
    # must run only on the LAST segment. Earlier segments use the closed-only regex,
    # matching strip_tool_call_markup. Previously the route treated the segment
    # boundary as EOS and stripped ``foo[ARGS]``.
    text = "Please pass foo[ARGS] <think>pause</think> to the template."
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True) == text


def test_route_display_strip_removes_complete_call_before_think_block():
    # A COMPLETE bracket call before a <think> block is still stripped (the balanced
    # scan runs on every segment), so the closed-only non-last regex does not let a
    # real call leak.
    text = 'before search[ARGS]{"q":"x"} <think>pause</think> after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "search[ARGS]" not in out
    assert "<think>pause</think>" in out
    assert "before" in out and "after" in out


def test_route_display_strip_removes_closed_xml_before_think_block():
    # A closed <tool_call>...</tool_call> before a <think> block is removed in the
    # non-last segment via the closed-only regex.
    text = 'pre <tool_call>{"name":"x"}</tool_call> <think>p</think> tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out
    assert "<think>p</think>" in out
    assert "pre" in out and "tail" in out


def test_all_route_cleanup_sites_use_protected_display_helper():
    # Every route content-cleanup site must go through _strip_tool_xml_for_display
    # (think-preserving, balanced) -- including the Anthropic stream / non-stream /
    # passthrough paths, which previously called raw _TOOL_XML_RE.sub and so
    # corrupted <think> rehearsal and dropped trailing prose after nested calls.
    # The only legitimate raw _TOOL_XML_RE.sub lives INSIDE the helper itself.
    raw_sub_lines = [
        (i, line)
        for i, line in enumerate(_src.splitlines(), 1)
        if "_TOOL_XML_RE.sub(" in line and not line.lstrip().startswith("#")
    ]
    assert len(raw_sub_lines) == 1, (
        "raw _TOOL_XML_RE.sub must appear only inside _strip_tool_xml_for_display; "
        f"found extra call sites: {raw_sub_lines!r}"
    )


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
    # A complete Mistral [TOOL_CALLS] call strips only its balanced JSON,
    # leaving following prose intact.
    cleaned = _TOOL_XML_RE.sub("", '[TOOL_CALLS]web_search{"q":"x"} and then prose')
    assert "[TOOL_CALLS]" not in cleaned
    assert "and then prose" in cleaned


def test_strips_unclosed_bracket_tail():
    # Close brace lost to EOS: the truncated tail is stripped up to the end
    # instead of leaking the raw bracket marker to the UI.
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
    # A [TOOL_CALLS] call with two-level-nested JSON args must be removed whole
    # so the trailing prose survives. A one-level regex either left the markup
    # or let the catch-all eat everything to EOS.
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
    # A literal <think>...</think> inside a tool-call argument must be stripped with
    # the call, not preserved as reasoning (which would split the call span).
    text = (
        '<tool_call>{"name":"write","arguments":'
        '{"text":"compare <think> and </think> tags"}}</tool_call>'
    )
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out and '"name"' not in out


def test_route_strip_removes_truncated_mistral_array():
    # A canonical array truncated by EOS (no closing ``]``) cannot be removed by the
    # balanced scan; the route fallback must strip its tail like other orphans.
    text = 'before [TOOL_CALLS] [{"name":"a","arguments":{"x":1}}'  # missing ]
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "{" not in out
    assert "before" in out


def test_route_strip_keeps_prose_mentioning_args_marker():
    # ``foo[ARGS] in a sentence`` is prose (no JSON body); the rehearsal arm must
    # not truncate the rest of the line.
    text = "Please pass foo[ARGS] to the template and continue reading."
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert out == text


def test_route_strip_handles_mistral_v11_call_id_args_shape():
    # [TOOL_CALLS]name[CALL_ID]id[ARGS]{json} (Mistral Small 3.2) -- arms aligned
    # with the parser regexes must strip it whole.
    text = 'before [TOOL_CALLS]web_search[CALL_ID]abc123[ARGS]{"q":"x"} after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "[CALL_ID]" not in out and "[ARGS]" not in out
    assert "before" in out and "after" in out


# ── Mistral [/TOOL_CALLS] closer + literal <think> inside a call ───────────────

from core.tool_healing import strip_tool_call_markup as _strip_tool_call_markup


def test_core_strip_removes_orphan_tool_calls_closer_array_form():
    # The Mistral v11 wrapper closes a call with ``[/TOOL_CALLS]``. The balanced scan
    # removes the call body but leaves the bare closer; it must not leak as content.
    text = '[TOOL_CALLS] [{"name":"x","arguments":{}}][/TOOL_CALLS]'
    assert _strip_tool_call_markup(text, final = True) == ""


def test_core_strip_removes_orphan_tool_calls_closer_named_form_keeps_tail():
    text = '[TOOL_CALLS]web_search{"q":"x"}[/TOOL_CALLS] tail'
    assert _strip_tool_call_markup(text, final = True) == "tail"


def test_core_strip_removes_call_with_literal_think_in_argument():
    # An unclosed literal ``<think>`` inside a call's arguments must be stripped WITH the
    # call (argument data), not preserved as a reasoning block, despite the greedy match.
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
    # An INCOMPLETE <tool_call> (run via allow_incomplete) whose argument holds a literal
    # <think> must be stripped to EOS, not treated as a reasoning block; covers the
    # unclosed tail _tool_call_markup_spans previously missed.
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
    # P1 #5704: an inactive ``foo[ARGS]{...}`` is prose, not a call, so the gated display
    # strip must leave it (and its sentence) intact, matching the loop-level gate.
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
    # The mirror case: an ACTIVE tool name IS a real rehearsal call and must still be
    # stripped from the visible text even with the gate in play.
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        'web_search[ARGS]{"query":"x"} done', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "web_search[ARGS]" not in out
    assert out.strip() == "done"


def test_route_display_strip_ungated_strips_all_rehearsal_unchanged():
    # Backwards-compat: with no gate (enabled_tool_names=None, the default and the
    # history-sanitize / non-loop call sites) the bare rehearsal is stripped as before.
    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True).strip() == "is just syntax."
    assert (
        _strip_tool_xml_for_display(
            text, auto_heal_tool_calls = True, enabled_tool_names = None
        ).strip()
        == "is just syntax."
    )


def test_route_display_strip_control_token_stripped_regardless_of_gate():
    # ``[TOOL_CALLS]`` is a control token, not an ambiguous bare identifier: it is
    # stripped even when its NAME is not in the active-tool gate.
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        '[TOOL_CALLS]foo[ARGS]{"x":1} keep', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "[TOOL_CALLS]" not in out and "foo[ARGS]" not in out
    assert out.strip() == "keep"


def test_core_strip_gates_bare_rehearsal_on_enabled_tools():
    # P1 (#5704): the shared strip_tool_call_markup gate mirrors the parse gate --
    # an inactive ``foo[ARGS]{...}`` is prose and preserved with its trailing
    # sentence; an active tool name is a real call and stripped. ``None`` keeps the
    # legacy strip-all behavior so existing callers are unchanged.
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
    # The GGUF assistant-history sanitiser passes the enabled-tool-name gate, so a
    # prior turn that documented an INACTIVE ``foo[ARGS]{...}`` shape survives in the
    # replayed prompt context (the loop treats it as prose). Without the gate every
    # ``NAME[ARGS]{...}`` is stripped, corrupting the history.
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
    # Guard the wiring: the GGUF assistant-history strip must forward the display gate
    # (like the live-response strip), else a documented inactive rehearsal is deleted
    # from the replayed prompt context.
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
    # The safetensors and Anthropic assistant-history sanitisers and the Anthropic
    # non-stream passthrough must forward an enabled-tool-name gate to
    # _strip_tool_xml_for_display, like the GGUF history sanitiser and the live strips,
    # so an inactive NAME[ARGS]{...} example is preserved in the replayed prompt / final
    # text instead of deleted.
    blocks = {
        "safetensors history": r"Strip stale tool-call XML from prior assistant turns.*?\.strip\(\)",
        "anthropic history": r"Strip stale tool-call XML via the protected display helper.*?\.strip\(\)",
        "anthropic passthrough": r"Gate on the declared tools, like.*?\.strip\(\)",
    }
    for label, pat in blocks.items():
        m = _re.search(pat, _src, _re.DOTALL)
        assert m, f"could not locate {label} strip block"
        assert "enabled_tool_names" in m.group(
            0
        ), f"{label} must forward enabled_tool_names to _strip_tool_xml_for_display"
