# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM2 / LFM2.5 ``<|tool_call_start|>[name(arg=...)]<|tool_call_end|>`` calls left in content by the
final answer pass (tool_choice "none") must not reach the client (#18802)."""

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from core.inference.tool_call_parser import (  # noqa: E402
    StreamingMarkupStripper,
    strip_tool_markup,
)
from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
)

ENABLED = {"web_search", "web_fetch"}
CALL = "<|tool_call_start|>[web_fetch(url='https://example.com/lfm')]<|tool_call_end|>"


def _streamed(tokens, strip):
    shown, prev, cum = "", "", ""
    for token in tokens:
        cum += token
        clean = strip.strip(cum)
        if len(clean) > len(prev):
            shown += clean[len(prev) :]
            prev = clean
    return shown


def test_closed_and_truncated_calls_are_stripped():
    assert strip_tool_markup(CALL, final = True, enabled_tool_names = ENABLED) == ""
    text = "Let me check.\n" + CALL + "\nIt is a MoE model."
    assert (
        strip_tool_markup(text, final = True, enabled_tool_names = ENABLED)
        == "Let me check.\n\nIt is a MoE model."
    )
    truncated = "Let me check. <|tool_call_start|>[web_search(query='lfm"
    assert strip_tool_markup(truncated, final = True, enabled_tool_names = ENABLED) == "Let me check."


def test_prose_mention_of_the_marker_is_kept():
    text = "LFM wraps calls in <|tool_call_start|> and <|tool_call_end|> tokens."
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text
    assert _streamed(list(text), StreamingMarkupStripper(ENABLED, seg_final = False)) == text


def test_open_call_is_held_while_streaming():
    tokens = ["Here is what I found.\n", "<|tool_call_start|>", "[web", "_fetch(url='https://ex"]
    tokens += ["ample.com/lfm')]", "<|tool_call_end|>", "\nSummary: a MoE model."]
    shown = _streamed(tokens, StreamingMarkupStripper(ENABLED, seg_final = False))
    assert shown == "Here is what I found.\n\nSummary: a MoE model."


def test_route_display_strip_hides_the_call():
    from routes.inference import _strip_tool_xml_for_display

    text = "Answer.\n" + CALL
    shown = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
    assert "tool_call_start" not in shown and "web_fetch" not in shown
    assert "Answer." in shown


def test_a_call_in_the_final_answer_pass_never_reaches_the_client(monkeypatch):
    from routes.inference import _StreamingDisplayStrip

    tokens = [
        "\n\n",
        "<|tool_call_start|>",
        "[web_search(",
        "query='LFM2.5-24B-A2B",
        " Liquid AI')]",
    ]
    tokens += ["<|tool_call_end|>"]
    streams = [
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "LFM2.5-24B-A2B"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": token}) for token in tokens] + [_done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5-24B-A2B."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert payloads[-1].get("tool_choice") == "none"
    assert calls == [("web_search", {"query": "LFM2.5-24B-A2B"})]
    strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
    shown, prev = "", ""
    for event in events:
        if event.get("type") == "tool_start" or (
            event.get("type") == "status" and not event.get("text")
        ):
            prev = ""
        elif event.get("type") == "content":
            clean = strip.strip(event.get("text", ""))
            shown += clean[len(prev) :]
            prev = clean
    assert "tool_call_start" not in shown and "tool_call_end" not in shown, shown
    assert "web_search(" not in shown, shown
