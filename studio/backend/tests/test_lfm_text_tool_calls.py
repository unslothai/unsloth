# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM2 / LFM2.5 ``<|tool_call_start|>[name(arg=...)]<|tool_call_end|>`` calls left in content by the
final answer pass (tool_choice "none") must not reach the client (#18802)."""

import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from core.inference.tool_call_parser import (  # noqa: E402
    StreamingMarkupStripper,
    parse_tool_calls_from_text,
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
        [_sse({"content": "LFM2.5-24B-A2B is a sparse MoE model."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5-24B-A2B."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert payloads[1].get("tool_choice") == "none" and payloads[1].get("tools")
    # The call-only final answer is re-asked once with no tools in the prompt.
    assert len(payloads) == 3 and "tools" not in payloads[2] and "tool_choice" not in payloads[2]
    # Noted on the trailing tool result, not a new user turn (Ministral alternation).
    retry_messages = payloads[2]["messages"]
    assert len(retry_messages) == len(payloads[1]["messages"])
    assert retry_messages[-1]["role"] == "tool"
    assert "Tools are no longer available" in json.dumps(retry_messages[-1])
    assert "Tools are no longer available" not in json.dumps(payloads[1]["messages"])
    assert calls == [("web_search", {"query": "LFM2.5-24B-A2B"})]
    shown = _route_shown(events)
    assert "tool_call_start" not in shown and "tool_call_end" not in shown, shown
    assert "web_search(" not in shown, shown
    assert shown.strip() == "LFM2.5-24B-A2B is a sparse MoE model.", shown


def test_call_only_final_answer_is_retried_once(monkeypatch):
    first = _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "q"})},
                }
            ]
        }
    )
    call_only = [_sse({"content": CALL}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [[first, _done()], call_only, call_only])
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3 and "tools" not in payloads[2]
    shown = _route_shown(events)
    assert "tool call limit" in shown and "tool_call_start" not in shown, shown


def test_reasoning_then_call_final_answer_is_retried(monkeypatch):
    first = _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "q"})},
                }
            ]
        }
    )
    thought_then_call = [
        _sse({"reasoning_content": "Need more detail."}),
        _sse({"content": CALL}),
        _done(),
    ]
    answer = [_sse({"content": "A sparse MoE model."}), _done()]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [[first, _done()], thought_then_call, answer]
    )
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3 and "tools" not in payloads[2]
    final = [e["text"] for e in events if e.get("type") == "content"][-1]
    assert final.endswith("A sparse MoE model.") and "tool_call_start" not in final, final


def _route_shown(events):
    from routes.inference import _StreamingDisplayStrip

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
    return shown


@pytest.mark.parametrize(
    "parts",
    [
        ["<|tool_call_start|>", "[web_search(", "query='x')]", "<|tool_call_end|>"],
        ["<tool_call>", '{"name": "web_search", ', '"arguments": {"query": "x"}}', "</tool_call>"],
        ["[TOOL_CALLS]", '[{"name": "web_search", ', '"arguments": {"query": "x"}}]'],
    ],
    ids = ["lfm", "qwen", "mistral"],
)
def test_split_call_prefix_does_not_garble_the_retry(monkeypatch, parts):
    first = _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "q"})},
                }
            ]
        }
    )
    split_call = [_sse({"content": part}) for part in parts] + [_done()]
    answer = [_sse({"content": "A sparse MoE model."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [[first, _done()], split_call, answer])
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3 and "tools" not in payloads[2]
    assert _route_shown(events).endswith("A sparse MoE model."), _route_shown(events)


def _names_args(text, **kw):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in parse_tool_calls_from_text(text, **kw)
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (CALL, [("web_fetch", {"url": "https://example.com/lfm"})]),
        (
            "Checking. <|tool_call_start|>[web_search(query='a', n=3), web_fetch(url=\"u\")]<|tool_call_end|>",
            [("web_search", {"query": "a", "n": 3}), ("web_fetch", {"url": "u"})],
        ),
        (
            '<|tool_call_start|>[{"name": "web_search", "arguments": {"query": "x"}}]<|tool_call_end|>',
            [("web_search", {"query": "x"})],
        ),
        (
            "<|tool_call_start|>[f(a=true, b=null)]<|tool_call_end|>",
            [("f", {"a": True, "b": None})],
        ),
        ("<|tool_call_start|>[web_search(query='cut')]", [("web_search", {"query": "cut"})]),
        ('<|tool_call_start|>[__import__("os").system("x")]<|tool_call_end|>', []),
        ("<|tool_call_start|>[web_search('positional')]<|tool_call_end|>", []),
        ("LFM wraps calls in <|tool_call_start|> and <|tool_call_end|> tokens.", []),
        (
            '<|tool_call_start|>[ {"name": "web_search", "arguments": {"query": "x"}}]<|tool_call_end|>',
            [("web_search", {"query": "x"})],
        ),
        (
            "<|tool_call_start|>[edit_file(path='x', edits=[{'old_string': 'a', 'replace_all': true}])]<|tool_call_end|>",
            [("edit_file", {"path": "x", "edits": [{"old_string": "a", "replace_all": True}]})],
        ),
        (
            "<|tool_call_start|>[mcp__srv__create-issue(title='Bug')]<|tool_call_end|>",
            [("mcp__srv__create-issue", {"title": "Bug"})],
        ),
        (
            "<|tool_call_start|>[web_search(query='a-b(c)'), web.fetch(url='u, x(y)')]<|tool_call_end|>",
            [("web_search", {"query": "a-b(c)"}), ("web.fetch", {"url": "u, x(y)"})],
        ),
        ("<|tool_call_start|>[f(a=[g(x=1)])]<|tool_call_end|>", []),
        ("<|tool_call_start|>[f(a=[{**x}])]<|tool_call_end|>", []),
        ("<|tool_call_start|>[f(a=[open('x')])]<|tool_call_end|>", []),
        ("<|tool_call_start|>[web_search(query={1, 2})]<|tool_call_end|>", []),
        ("<|tool_call_start|>[web_search(query=b'x')]<|tool_call_end|>", []),
        (
            '<tool_call>{"name": "web_search", "arguments": {"query": "' + CALL + '"}}</tool_call>',
            [("web_search", {"query": CALL})],
        ),
    ],
)
def test_lfm_calls_parse_as_literals(text, expected):
    assert _names_args(text) == expected


def test_strict_mode_refuses_an_unclosed_lfm_call():
    assert _names_args("<|tool_call_start|>[web_search(query='cut')]", allow_incomplete = False) == []


def test_a_raw_lfm_call_in_a_tool_round_is_executed(monkeypatch):
    raw = "<|tool_call_start|>[web_search(query='LFM2.5-24B-A2B')]<|tool_call_end|>"
    streams = [
        [_sse({"content": raw}), _done()],
        [_sse({"content": "A sparse MoE model."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert calls == [("web_search", {"query": "LFM2.5-24B-A2B"})]
    assert _route_shown(events).strip() == "A sparse MoE model."


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            '<|tool_call_start|>[{"name": "python", "arguments": {"code": "print(1)"}}]<|tool_call_end|>',
            [("python", {"code": "print(1)"})],
        ),
        (
            '<|tool_call_start|>[terminal(command="ls -la")]<|tool_call_end|>',
            [("terminal", {"command": "ls -la"})],
        ),
        (
            'Example: {"name": "python", "arguments": {"code": "rm"}} then ' + CALL,
            [("web_fetch", {"url": "https://example.com/lfm"})],
        ),
    ],
)
def test_lfm_execution_calls_keep_their_arguments(text, expected):
    assert _names_args(text, enabled_tool_names = {"python", "terminal", "web_fetch"}) == expected


def test_an_lfm_call_quoted_in_a_rehearsal_is_its_data():
    text = (
        'web_search[ARGS]{"query": "<|tool_call_start|>[terminal(command=\'echo x\')]'
        '<|tool_call_end|>"}'
    )
    assert [n for n, _ in _names_args(text, enabled_tool_names = {"web_search", "terminal"})] == [
        "web_search"
    ]
