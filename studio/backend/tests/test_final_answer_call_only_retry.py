# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A final answer that is only a tool call (#18802) is re-asked once without tools, else explained."""

import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
)

ENABLED = {"web_search", "web_fetch"}
CALL = "<|tool_call_start|>[web_fetch(url='https://example.com/lfm')]<|tool_call_end|>"


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
    assert "its tools ran out" in shown and "tool_call_start" not in shown, shown


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


def test_a_call_only_answer_with_no_budget_left_says_so(monkeypatch):
    def usage(n):
        return (
            "data: "
            + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": n}})
            + "\n"
        )

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
    streams = [[first, usage(48), _done()], [_sse({"content": CALL}), usage(24), _done()]]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tokens = 64,
    )
    assert len(payloads) == 2
    shown = _route_shown(events)
    assert "its tools ran out" in shown and "tool_call_start" not in shown, shown


def test_safetensors_loop_runs_lfm_calls_and_explains_a_call_only_final_turn():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    calls = []

    def _gen(_messages):
        yield CALL

    events = list(
        run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "Research LFM2.5."}],
            tools = [{"type": "function", "function": {"name": "web_fetch"}}],
            execute_tool = lambda name, arguments, **_: calls.append((name, arguments)) or "RESULT",
            max_tool_iterations = 1,
        )
    )
    assert calls == [("web_fetch", {"url": "https://example.com/lfm"})]
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert (
        contents and "its tools ran out" in contents[-1] and "tool_call_start" not in contents[-1]
    )


def test_a_bare_json_call_from_the_retry_still_gets_the_notice(monkeypatch):
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
    bare = '{"name": "web_search", "parameters": {"query": "x"}}'
    streams = [
        [first, _done()],
        [_sse({"content": CALL}), _done()],
        [_sse({"content": bare}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3
    shown = _route_shown(events)
    assert "its tools ran out" in shown and '"parameters"' not in shown, shown


def test_the_limit_notice_survives_a_split_call_from_the_retry(monkeypatch):
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
    parts = ["<tool_call>", '{"name": "web_search", "arguments": {"query": "x"}}', "</tool_call>"]
    streams = [
        [first, _done()],
        [_sse({"content": CALL}), _done()],
        [_sse({"content": part}) for part in parts] + [_done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3
    shown = _route_shown(events)
    assert "its tools ran out" in shown and "<tool_call>" not in shown, shown


def test_safetensors_loop_explains_a_reasoning_then_call_final_turn():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    def _gen(_messages):
        yield "<think>Need more detail.</think>" + CALL

    events = list(
        run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "Research LFM2.5."}],
            tools = [{"type": "function", "function": {"name": "web_fetch"}}],
            execute_tool = lambda name, arguments, **_: "RESULT",
            max_tool_iterations = 1,
        )
    )
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert (
        contents
        and contents[-1].endswith("if it hit that limit.")
        and "tool_call_start" not in contents[-1]
    ), contents


def _search_call():
    return _sse(
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


def test_the_retry_keeps_a_trailing_image_turn_image_only(monkeypatch):
    from core.inference.mcp_images import SENTINEL, is_synthetic_image_turn

    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    view = _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "view_image",
                        "arguments": json.dumps({"path": "chart.png"}),
                    },
                }
            ]
        }
    )
    qwen_call = (
        '<tool_call>\n{"name": "view_image", "arguments": {"path": "more.png"}}\n</tool_call>'
    )

    def _pair(i):
        calls = [("call_n%d" % i, "new%d.png" % i), ("call_d%d" % i, "chart.png")]
        return _sse(
            {
                "tool_calls": [
                    {
                        "index": k,
                        "id": cid,
                        "type": "function",
                        "function": {"name": "view_image", "arguments": json.dumps({"path": path})},
                    }
                    for k, (cid, path) in enumerate(calls)
                ]
            }
        )

    # The repeated call trips the duplicate guard, so synthesis starts right after the image turn.
    streams = [
        *[[_pair(i), _done()] for i in range(3)],
        [_sse({"content": qwen_call}), _done()],
        [_sse({"content": "A bar chart."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    backend._is_vision = backend._mmproj_accepts_image = True
    _record_tool_calls(
        monkeypatch, "chart\n" + SENTINEL + json.dumps([{"data": png, "mimeType": "image/png"}])
    )
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Describe the chart."}],
        [{"type": "function", "function": {"name": "view_image"}}],
        max_tool_iterations = 10,
    )
    final = next(i for i, p in enumerate(payloads) if p.get("tool_choice") == "none")
    assert is_synthetic_image_turn(payloads[final]["messages"][-1]), payloads[final]["messages"][-1]
    assert "tools" not in payloads[final + 1]
    # Ministral's alternation fix only recognises an image-only synthetic turn.
    assert is_synthetic_image_turn(payloads[final + 1]["messages"][-1]), payloads[final + 1][
        "messages"
    ][-1]
    assert _route_shown(events).strip() == "A bar chart."


def test_a_reasoning_only_final_answer_is_not_called_a_tool_call(monkeypatch):
    streams = [
        [_search_call(), _done()],
        [_sse({"content": "[THINK]Weighing the results.[/THINK]"}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 2, "no retry without a call in the answer"
    assert "its tools ran out" not in _route_shown(events)


def test_a_wrapperless_gemma_call_is_a_call_only_answer(monkeypatch):
    gemma = [_sse({"content": 'call:web_search{query:"next"}'}), _done()]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [[_search_call(), _done()], gemma, gemma]
    )
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3 and "tools" not in payloads[2]
    shown = _route_shown(events)
    assert "its tools ran out" in shown and "call:web_search" not in shown, shown


def test_a_marker_quoted_in_raw_reasoning_is_not_a_call(monkeypatch):
    think = [
        _sse({"content": "[THINK]I should use <tool_call> if more search is needed.[/THINK]"}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [[_search_call(), _done()], think])
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 2
    assert "its tools ran out" not in _route_shown(events)


def test_safetensors_loop_explains_a_call_after_prefilled_reasoning():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    def _gen(_messages):
        yield "Need more sources.</think>" + CALL

    events = list(
        run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "Research LFM2.5."}],
            tools = [{"type": "function", "function": {"name": "web_fetch"}}],
            execute_tool = lambda name, arguments, **_: "RESULT",
            max_tool_iterations = 1,
            reasoning_prefilled = True,
        )
    )
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert contents and contents[-1].endswith("if it hit that limit."), contents


def test_inline_reasoning_then_call_final_answer_is_retried(monkeypatch):
    inline = '<think>Need more sources.</think><tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
    streams = [
        [_search_call(), _done()],
        [_sse({"content": inline}), _done()],
        [_sse({"content": "LFM2.5 is a sparse MoE model."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5."}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(payloads) == 3 and "tools" not in payloads[2]
    assert "LFM2.5 is a sparse MoE model." in _route_shown(events)


def test_safetensors_notice_extends_the_streamed_prefix():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    def _gen(_messages):
        yield "\n\n"
        yield "\n\n" + CALL

    events = list(
        run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "Research LFM2.5."}],
            tools = [{"type": "function", "function": {"name": "web_fetch"}}],
            execute_tool = lambda name, arguments, **_: "RESULT",
            max_tool_iterations = 1,
        )
    )
    shown = _route_shown(events)
    assert "The model asked for another tool" in shown, repr(shown)


def test_safetensors_notice_stays_after_a_prefilled_reasoning_close():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    def _gen(_messages):
        yield "Need more sources.</think>"
        yield "Need more sources.</think>" + CALL

    events = list(
        run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "Research LFM2.5."}],
            tools = [{"type": "function", "function": {"name": "web_fetch"}}],
            execute_tool = lambda name, arguments, **_: "RESULT",
            max_tool_iterations = 1,
            reasoning_prefilled = True,
        )
    )
    contents = [e["text"] for e in events if e["type"] == "content"]
    # The prefilled extractor needs the closer in the snapshot that carries the notice.
    last = contents[-1]
    assert "</think>" in last and last.index("</think>") < last.index("The model asked"), contents
