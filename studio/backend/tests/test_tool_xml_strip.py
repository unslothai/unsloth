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

import re as _re

_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
_m = _re.search(r"_TOOL_XML_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _m, "could not extract _TOOL_XML_RE source"
# The lazy ``(.*?)`` could grab a shorter expression if an arm is ever wrapped, so pin the DeepSeek and bare-Kimi arms.
assert "_DS_OPEN_SRC" in _m.group(1) and "tool_call_begin" in _m.group(
    1
), "extracted _TOOL_XML_RE is missing expected arms (extraction truncated?)"
# The regex reuses the parser's shared DeepSeek alternation, so provide it or the expression differs.
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
_mc = _re.search(r"_TOOL_XML_CLOSED_RE = _re\.compile\((.*?)\n\)", _src, _re.DOTALL)
assert _mc, "could not extract _TOOL_XML_CLOSED_RE source"
exec(f"_TOOL_XML_CLOSED_RE = _re.compile({_mc.group(1)})", _ns)
_TOOL_XML_CLOSED_RE = _ns["_TOOL_XML_CLOSED_RE"]

# Signatures may span lines and carry the enabled_tool_names gate, so match to ``-> str:`` then the body.
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
    r"def _display_tool_name_gate\(.*?(?=\nlogger = get_logger)",
    _src,
    _re.DOTALL,
)
assert _helper, "could not extract display strip helper source"
# The extracted block spans _display_tool_name_gate through _strip_tool_xml; confirm the shared delegate is present.
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




def test_route_display_strip_respects_disabled_auto_heal_contract():
    text = 'literal <tool_call>{"name":"web_search"}</tool_call> survives'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<tool_call>" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)


def test_route_display_strip_preserves_rehearsal_inside_think():
    # A rehearsed bracket call inside think is reasoning, so the block is preserved and a real call strips.
    text = '<think>plan: search[ARGS]{"q":"x"}</think> answer [TOOL_CALLS]web_search{"q":"y"} tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert '<think>plan: search[ARGS]{"q":"x"}</think>' in out
    assert "[TOOL_CALLS]web_search" not in out
    assert "answer" in out and "tail" in out


def test_route_display_strip_keeps_bare_args_before_think_block():
    # A bare ``foo[ARGS]`` before a think block is prose: EOS-anchored tail arms run only on the last segment.
    text = "Please pass foo[ARGS] <think>pause</think> to the template."
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True) == text


def test_route_display_strip_removes_complete_call_before_think_block():
    text = 'before search[ARGS]{"q":"x"} <think>pause</think> after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "search[ARGS]" not in out
    assert "<think>pause</think>" in out
    assert "before" in out and "after" in out


def test_route_display_strip_removes_closed_xml_before_think_block():
    text = 'pre <tool_call>{"name":"x"}</tool_call> <think>p</think> tail'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out
    assert "<think>p</think>" in out
    assert "pre" in out and "tail" in out


def test_all_route_cleanup_sites_use_protected_display_helper():
    # Every route cleanup must use _strip_tool_xml_for_display: the raw sub corrupted rehearsal and prose.
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
    # _TOOL_XML_RE has no [TOOL_CALLS] arm, so it delegates the balanced-brace strip; non-greedy truncates nested JSON.
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
    # The attribute form must strip from the route too, where it previously leaked into the UI.
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

    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = False) == text
    assert "<function name=" not in _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)




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




def test_strips_orphan_closing_tag():
    text = "...the table rows directly.\n</parameter>\n</function>\n</tool_call><think>Continuing</think>"
    cleaned = _TOOL_XML_RE.sub("", text)
    assert "</tool_call>" not in cleaned
    assert "</function>" not in cleaned


def test_strips_gemma_native_orphan_closing_tag():
    cleaned = _TOOL_XML_RE.sub("", "Tool call drained.<tool_call|>Visible tail.")

    assert "<tool_call|>" not in cleaned
    assert "Tool call drained." in cleaned
    assert "Visible tail." in cleaned




def test_strips_tail_only_parameter_orphan():
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
    cleaned = _TOOL_XML_RE.sub("", '[TOOL_CALLS]web_search{"q":"x"} and then prose')
    assert "[TOOL_CALLS]" not in cleaned
    assert "and then prose" in cleaned


def test_strips_unclosed_bracket_tail():
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




def test_preserves_plain_text():
    text = "1. Animals — Maroon 5\n2. Take Me to Church — Hozier"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_code_fences():
    text = "```python\nimport sys\nprint(sys.version)\n```"
    assert _TOOL_XML_RE.sub("", text) == text


def test_preserves_html_in_prose():
    text = "Use the <html> tag for documents."
    assert _TOOL_XML_RE.sub("", text) == text




REAL_LEAKS = [
    'rectly.\n\nLet me try searching for Wikipedia pages that might have weekly chart data for 2015.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"Billboard Hot 100" "2015" "weekly" "chart" "position" "3"\n</parameter>\n</function',
    'arch `site:wikipedia.org "peaked at number 3" "2015" Billboard`\nI\'ll do a quick web search.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"peaked at number 3" Billboard Hot 100 2015 list\n</parameter>\n</function',
    'rd Hot 100 top-ten singles in 2015".\nI\'ll use web_search to find this exact Wikipedia page.\n</think><tool_call>\n<function=web_search>\n<parameter=query>\n"List of Billboard Hot 100 top-ten singles in 2015" wikipedia\n</parameter>\n</function',
    "the table rows directly.\n</parameter>\n</function>\n</tool_call><think>The user wants me to list and categorize all songs that charted #3 on the Billboard Hot 100 in 2015. I have been trying to get this data",
    "parse it more carefully.\n</parameter>\n</function>\n</tool_call><think>The user wants a list of songs that charted #3 on the Billboard Hot 100 in 2015, categorized.",
]


@pytest.mark.parametrize(
    "leak", REAL_LEAKS, ids = [f"sweep_sample_{i}" for i in range(len(REAL_LEAKS))]
)
def test_real_world_sweep_leaks_get_stripped(leak):
    cleaned = _TOOL_XML_RE.sub("", leak)
    assert "<tool_call>" not in cleaned, f"leak survived: {cleaned!r}"
    assert "<function=" not in cleaned, f"leak survived: {cleaned!r}"




# All end-anchored: the outer close was truncated by EOS and the inner open DRAINED, leaving a bare </parameter> tail.
GDPVAL_PARAMETER_LEAKS = [
    "the page contains image data and the text is not readable.\n</parameter>\n\n",
    "...some mojibake content here...\n</parameter>\n\n",
    "blocked, while others may still be in effect. The law is currently under further review by the Ninth Circuit.\n</parameter>\n\n",
    "comprehensive training report\n</parameter>\n\n",
    "Global Economic Prospects\nJune 2025\nGlobal Economic Prospects\n</parameter>\n",
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




def test_no_catastrophic_backtracking_on_open_bracket_spam():
    # 256KB of '<' must fail fast on a literal mismatch, not backtrack.
    import time

    adv = "<" * (1024 * 256) + "X"
    t0 = time.perf_counter()
    _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"regex took {elapsed*1000:.0f}ms on 256KB '<' spam"


def test_no_catastrophic_backtracking_on_orphan_opening_spam():
    import time

    adv = "<tool_call>X" * 1000
    t0 = time.perf_counter()
    cleaned = _TOOL_XML_RE.sub("", adv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"regex took {elapsed*1000:.0f}ms on 1000x orphan opens"
    assert "<tool_call>" not in cleaned




def test_route_strip_two_level_nested_bracket_keeps_trailing_prose():
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
    text = (
        '<tool_call>{"name":"write","arguments":'
        '{"text":"compare <think> and </think> tags"}}</tool_call>'
    )
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "<tool_call>" not in out and '"name"' not in out


def test_route_strip_removes_truncated_mistral_array():
    text = 'before [TOOL_CALLS] [{"name":"a","arguments":{"x":1}}'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "{" not in out
    assert "before" in out


def test_route_strip_keeps_prose_mentioning_args_marker():
    text = "Please pass foo[ARGS] to the template and continue reading."
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert out == text


def test_route_strip_handles_mistral_v11_call_id_args_shape():
    text = 'before [TOOL_CALLS]web_search[CALL_ID]abc123[ARGS]{"q":"x"} after'
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "[TOOL_CALLS]" not in out and "[CALL_ID]" not in out and "[ARGS]" not in out
    assert "before" in out and "after" in out



from core.tool_healing import strip_tool_call_markup as _strip_tool_call_markup


def test_core_strip_removes_orphan_tool_calls_closer_array_form():
    text = '[TOOL_CALLS] [{"name":"x","arguments":{}}][/TOOL_CALLS]'
    assert _strip_tool_call_markup(text, final = True) == ""


def test_core_strip_removes_orphan_tool_calls_closer_named_form_keeps_tail():
    text = '[TOOL_CALLS]web_search{"q":"x"}[/TOOL_CALLS] tail'
    assert _strip_tool_call_markup(text, final = True) == "tail"


def test_core_strip_removes_call_with_literal_think_in_argument():
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

    assert (
        _strip("answer <think>real</think> done", final = True) == "answer <think>real</think> done"
    )

    mixed = '<tool_call>{"name":"a","arguments":{}}</tool_call> mid <think>r</think> end'
    assert _strip(mixed, final = True) == "mid <think>r</think> end"




def test_display_tool_name_gate_returns_active_names_or_none():
    assert _display_tool_name_gate([]) is None
    assert _display_tool_name_gate(None) is None
    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "run_python"}},
        {"type": "function"},
        {"nope": 1},
    ]
    assert _display_tool_name_gate(tools) == {"web_search", "run_python"}


def test_route_display_strip_keeps_inactive_rehearsal_when_gated():
    gate = {"web_search"}
    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert (
        _strip_tool_xml_for_display(text, auto_heal_tool_calls = True, enabled_tool_names = gate)
        == text
    )
    assert (
        _strip_tool_xml_for_display(
            "use foo[ARGS] here", auto_heal_tool_calls = True, enabled_tool_names = gate
        )
        == "use foo[ARGS] here"
    )


def test_route_display_strip_removes_active_rehearsal_when_gated():
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        'web_search[ARGS]{"query":"x"} done', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "web_search[ARGS]" not in out
    assert out.strip() == "done"


def test_route_display_strip_ungated_strips_all_rehearsal_unchanged():
    text = 'foo[ARGS]{"x":1} is just syntax.'
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True).strip() == "is just syntax."
    assert (
        _strip_tool_xml_for_display(
            text, auto_heal_tool_calls = True, enabled_tool_names = None
        ).strip()
        == "is just syntax."
    )


def test_route_display_strip_control_token_stripped_regardless_of_gate():
    gate = {"web_search"}
    out = _strip_tool_xml_for_display(
        '[TOOL_CALLS]foo[ARGS]{"x":1} keep', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "[TOOL_CALLS]" not in out and "foo[ARGS]" not in out
    assert out.strip() == "keep"


def test_core_strip_gates_bare_rehearsal_on_enabled_tools():
    # The strip gate mirrors the parse gate: inactive names are prose, active strip, ``None`` strips all.
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
    # The GGUF history sanitiser passes the gate, so a documented inactive shape survives the replay.
    gate = _display_tool_name_gate([{"function": {"name": "web_search"}}])
    text = 'To call it write foo[ARGS]{"x":1} in your reply.'
    assert 'foo[ARGS]{"x":1}' in _strip_tool_xml_for_display(
        text, auto_heal_tool_calls = True, enabled_tool_names = gate
    )
    assert "web_search[ARGS]" not in _strip_tool_xml_for_display(
        'Result web_search[ARGS]{"q":"x"} done', auto_heal_tool_calls = True, enabled_tool_names = gate
    )
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
    # The other sanitisers and the Anthropic non-stream passthrough must forward the gate so examples survive.
    blocks = {
        "safetensors history": r"Strip stale tool-call XML from prior assistant turns.*?\.strip\(\)",
        "anthropic history": r"Strip stale tool-call XML via the protected display helper.*?\.strip\(\)",
        "anthropic passthrough": r"if not healing_active:.*?\.strip\(\)",
    }
    for label, pat in blocks.items():
        m = _re.search(pat, _src, _re.DOTALL)
        assert m, f"could not locate {label} strip block"
        assert "enabled_tool_names" in m.group(
            0
        ), f"{label} must forward enabled_tool_names to _strip_tool_xml_for_display"


# ── DeepSeek opener variants + bare Kimi (parse/strip symmetry) ──


def test_strips_deepseek_space_opener_variant():
    # The space-separated opener is parsed by the parser, so the display strip must remove it too.
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
    # Kimi can emit a bare call with no section wrapper, and the parser accepts it, so the strip must cover it.
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
        # The call-shaped lookahead fires only on a real call or a bare EOF fragment, never on protocol prose.
        "See <|tool_call_begin|> in the docs. More prose after it.",
        "The <|tool_calls_section_begin|> marker opens a batch. Read on.",
        "DeepSeek uses <｜tool▁calls▁begin｜> to start a call block, then continues.",
    ],
)
def test_deepseek_kimi_false_alarm_prose_is_kept(text):
    # Regression for the route arm truncating a prose answer that references a marker with no following call.
    assert _TOOL_XML_RE.sub("", text) == text


def test_deepseek_kimi_real_calls_still_strip_after_false_alarm_fix():
    # The lookahead must not weaken real-call stripping: closed, truncated and EOF forms all still go.
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




def test_python_tag_strip_consumes_literal_sentinel_in_arg():
    # The old `<(?!\
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
    # A genuine Llama control sentinel still bounds the strip, so following assistant text is preserved.
    text = f'<|python_tag|>{{"name": "x", "parameters": {{}}}}{sentinel}visible answer'
    cleaned = _TOOL_XML_RE.sub("", text)
    assert (
        cleaned == f"{sentinel}visible answer"
    ), f"strip did not stop at real sentinel {sentinel!r}: {cleaned!r}"


def test_python_tag_strip_restarts_on_second_python_tag():
    # A second <
    text = '<|python_tag|>{"name": "a"}<|python_tag|>{"name": "b"}'
    cleaned = _TOOL_XML_RE.sub("", text)
    assert cleaned == "", f"second python_tag region leaked: {cleaned!r}"


def test_glm_call_with_literal_close_tag_in_arg_value_is_stripped_whole():
    text = (
        "<tool_call>web_search\n<arg_key>query</arg_key>\n"
        "<arg_value>find </tool_call> here</arg_value>\n</tool_call> done"
    )
    out = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True)
    assert "</arg_value>" not in out
    assert "<arg_key>" not in out
    assert out.strip() == "done"


def test_glm_normal_and_qwen_calls_still_stripped_by_route():
    # A normal GLM call and a Qwen <tool_call>{json}</tool_call> are still stripped, with trailing prose kept.
    glm = "<tool_call>get_time\n<arg_key>tz</arg_key>\n<arg_value>UTC</arg_value>\n</tool_call> ok"
    assert _strip_tool_xml_for_display(glm, auto_heal_tool_calls = True).strip() == "ok"
    qwen = '<tool_call>{"name":"web_search","arguments":{"q":"x"}}</tool_call> after'
    assert _strip_tool_xml_for_display(qwen, auto_heal_tool_calls = True).strip() == "after"


def test_route_strip_removes_param_alias_close_tag():
    # The parser accepts the <param name="..."> alias, so the route tail must strip an orphan </param>.
    assert _strip_tool_xml_for_display("answer </param>", auto_heal_tool_calls = True) == "answer "
    assert (
        _strip_tool_xml_for_display("answer </parameter>", auto_heal_tool_calls = True) == "answer "
    )


def test_route_strip_uses_guarded_function_scan_for_literal_nested_markup():
    # The route runs the guarded function-XML scan first, so a literal <function=...> value does not truncate.
    text = "<function=python><parameter=code><function=evil></function></parameter></function> tail"
    assert _strip_tool_xml_for_display(text, auto_heal_tool_calls = True).strip() == "tail"


def test_route_strip_gates_wrapperless_gemma_by_enabled_tools():
    # The route strip gates the markerless Gemma form on enabled names, like the parser, sparing prose.
    prose = "To document syntax you write call:foo{query:example}. That shows the format."
    assert "call:foo{query:example}" in _strip_tool_xml(prose, {"web_search"})
    assert "call:web_search" not in _strip_tool_xml(
        "Answer.\ncall:web_search{query:x}", {"web_search"}
    )
    assert "call:web_search" in _strip_tool_xml("Answer. call:web_search{query:x}", {"web_search"})
    assert "call:foo" not in _strip_tool_xml(
        "To document syntax you write\ncall:foo{query:example}"
    )


def test_gemma_strip_gate_empty_tools_preserves_prose():
    # With NO tools enabled the gate returns an EMPTY set, not None: None strips all and deletes a syntax answer.
    assert _gemma_strip_gate([]) == set()
    assert _gemma_strip_gate(None) == set()
    assert _gemma_strip_gate([{"function": {"name": "web_search"}}]) == {"web_search"}
    prose = "To document syntax you write call:foo{query:example}. That shows the format."
    assert "call:foo{query:example}" in _strip_tool_xml(prose, _gemma_strip_gate([]))
    assert "call:foo{query:example}" in _strip_tool_xml(prose, _gemma_strip_gate(None))
    assert "call:web_search" not in _strip_tool_xml(
        "Answer.\ncall:web_search{query:x}",
        _gemma_strip_gate([{"function": {"name": "web_search"}}]),
    )


def test_strip_keeps_prose_after_closed_function_call_with_literal_close():
    # The call ends at its first non-data close, so prose after it survives even when it mentions a literal </function>.
    from core.inference.tool_call_parser import strip_tool_markup
    text = (
        "<function=web_search><parameter=query>cats</parameter></function>"
        " Done. The tag </function> closes a call."
    )
    assert strip_tool_markup(text, final = True) == "Done. The tag </function> closes a call."


def test_final_strip_keeps_prose_mentioning_bare_markers():
    # A false-alarm marker in a normal answer must not lose everything after it.
    from core.inference.tool_call_parser import strip_tool_markup
    for text in (
        "See [TOOL_CALLS] docs for details. More prose after.",
        "<|python_tag|> is the Llama marker. Explanation continues.",
        "The <|tool_call> opener wraps Gemma calls.",
    ):
        assert strip_tool_markup(text, final = True) == text
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
    # The loops keep this text as next-turn history, so a leftover call would be replayed beside tool_calls.
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
    call_then_answer = (
        '{"name":"web_search","parameters":{"q":"x"}};{"name":"web_search","result":"data"}'
    )
    assert (
        strip_leading_bare_json_call(call_then_answer, enabled_tool_names = enabled)
        == '{"name":"web_search","result":"data"}'
    )
