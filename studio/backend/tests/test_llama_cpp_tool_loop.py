# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused tests for the GGUF llama.cpp agentic tool loop.

These tests drive ``LlamaCppBackend.generate_chat_completion_with_tools``
with fake llama-server SSE streams. They require no model, subprocess, GPU,
or network access.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import (
    _MAX_REPROMPTS,
    _PROVISIONAL_ARGS_MIN_CHARS,
    LlamaCppBackend,
)
from state import tool_approvals
from state.tool_approvals import TOOL_REJECTED_MESSAGE, resolve_tool_decision


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _make_backend(monkeypatch, streams: list[list[str]], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    return backend


def _tool_names(payload: dict) -> list[str]:
    return [
        (tool.get("function") or {}).get("name")
        for tool in payload.get("tools", [])
        if (tool.get("function") or {}).get("name")
    ]


def _patch_monotonic(monkeypatch, values: list[float]) -> None:
    import core.inference.llama_cpp as llama_cpp_mod

    it = iter(values)
    last = values[-1]

    def fake_monotonic() -> float:
        nonlocal last
        try:
            last = next(it)
        except StopIteration:
            pass
        return last

    monkeypatch.setattr(llama_cpp_mod.time, "monotonic", fake_monotonic)


def _structured_tool_call(tool_name: str, arguments: dict, call_id: str) -> list[str]:
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]


def test_structured_tool_call_after_visible_preface_is_executed(monkeypatch):
    """llama-server may emit content first and then native delta.tool_calls.

    Unsloth must not drop that tool call after it has streamed the preface.
    """

    tool_call_id = "call_render_late"
    first_stream = [
        _sse({"content": "Here is the canvas.\n\n"}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps(
                                {
                                    "code": "<html><body><div>red</div></body></html>",
                                    "title": "Simple Red Square",
                                }
                            ),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    second_stream = [
        _sse({"content": "Done."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, second_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML canvas: Simple Red Square."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
        )
    )

    content_events = [e for e in events if e.get("type") == "content"]
    assert content_events[0]["text"] == "Here is the canvas.\n\n"

    first_content_index = next(
        i for i, event in enumerate(events) if event.get("type") == "content"
    )
    actual_tool_start_index = next(
        i
        for i, event in enumerate(events)
        if event.get("type") == "tool_start" and event.get("arguments", {}).get("code")
    )
    assert first_content_index < actual_tool_start_index

    assert calls == [
        (
            "render_html",
            {
                "code": "<html><body><div>red</div></body></html>",
                "title": "Simple Red Square",
            },
        )
    ]
    assert any(e.get("type") == "tool_end" and e.get("tool_name") == "render_html" for e in events)

    # The second llama-server request should include the assistant preface
    # plus the structured tool call, preserving OpenAI-compatible ordering.
    assert len(payloads) == 2
    assistant_messages = [m for m in payloads[1]["messages"] if m.get("role") == "assistant"]
    assert assistant_messages[-1]["content"] == "Here is the canvas.\n\n"
    assert assistant_messages[-1]["tool_calls"][0]["id"] == tool_call_id
    assert assistant_messages[-1]["tool_calls"][0]["function"]["name"] == "render_html"


def test_streamed_reasoning_answer_emits_backend_summary(monkeypatch):
    stream = [
        _sse({"reasoning_content": "I am thinking."}),
        _sse({"reasoning_content": " Still thinking."}),
        _sse({"content": "Final answer."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)
    _patch_monotonic(monkeypatch, [100.0, 110.0, 172.0, 172.0])

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "answer"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    content_texts = [e["text"] for e in events if e["type"] == "content"]
    # Reasoning streams live during BUFFERING instead of arriving as one block:
    # each reasoning delta is emitted immediately, wrapped in <think>.
    assert content_texts[0] == "<think>I am thinking."
    assert content_texts[1] == "<think>I am thinking. Still thinking."
    # The final event closes the block and appends the answer.
    assert content_texts[-1] == "<think>I am thinking. Still thinking.</think>Final answer."

    summary_index = next(
        i for i, event in enumerate(events) if event["type"] == "reasoning_summary"
    )
    final_content_index = max(i for i, event in enumerate(events) if event["type"] == "content")
    assert summary_index < final_content_index
    assert events[summary_index]["duration_ms"] == 62000


def test_reasoning_streams_incrementally_with_tools(monkeypatch):
    # Regression (DeepSeek "thinking doesn't stream"): with a tool/pill active the
    # tool-loop generator must stream reasoning token-by-token like the no-tool
    # path, not accumulate it and dump one buffered <think> block.
    stream = [
        _sse({"reasoning_content": "Step one."}),
        _sse({"reasoning_content": " Step two."}),
        _sse({"reasoning_content": " Step three."}),
        _sse({"content": "Done."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "think then answer"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    reasoning_stage = [
        e["text"]
        for e in events
        if e["type"] == "content"
        and e["text"].startswith("<think>")
        and "</think>" not in e["text"]
    ]
    # One live emission per reasoning delta -- not a single dump.
    assert reasoning_stage == [
        "<think>Step one.",
        "<think>Step one. Step two.",
        "<think>Step one. Step two. Step three.",
    ]
    final = [e["text"] for e in events if e["type"] == "content"][-1]
    assert final == "<think>Step one. Step two. Step three.</think>Done."


def test_reasoning_only_reply_matches_no_tool_path_with_tools(monkeypatch):
    # A reasoning-only turn (whole answer in reasoning_content, no content, no
    # tool) with a tool active streams the reasoning live, then resolves to the
    # bare reasoning text -- identical to the no-tool generate_chat_completion
    # path -- so the non-streaming drain still returns it as `content`, not an
    # empty answer.
    stream = [
        _sse({"reasoning_content": "The capital of France is Paris."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)
    _patch_monotonic(monkeypatch, [1.0, 5.0, 5.0])

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "just think"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    content_texts = [e["text"] for e in events if e["type"] == "content"]
    # Reasoning streamed live during BUFFERING (the fix).
    assert content_texts[0] == "<think>The capital of France is Paris."
    # Resolves to bare reasoning, matching the no-tool sibling.
    assert content_texts[-1] == "The capital of France is Paris."


def test_reasoning_before_structured_tool_closes_think_block(monkeypatch):
    # Regression: reasoning streamed live during BUFFERING must be closed with
    # </think> before a structured tool_call drains, so consumers without a
    # reasoning extractor (Anthropic /v1/messages) never receive an unclosed
    # <think>. Mirrors the is_match (XML tool signal) path.
    tool_stream = [
        _sse({"reasoning_content": "Let me search."}),
        *_structured_tool_call("web_search", {"query": "weather"}, "call_1"),
    ]
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    tool_start_index = next(i for i, e in enumerate(events) if e["type"] == "tool_start")
    content_before_tool = [e["text"] for e in events[:tool_start_index] if e["type"] == "content"]
    # Reasoning streamed live, then closed before the tool -- balanced block.
    assert content_before_tool[0] == "<think>Let me search."
    assert content_before_tool[-1] == "<think>Let me search.</think>"


def _replay_route_reasoning_extractor(cumulatives: list[str]) -> tuple[str, str]:
    """Replay the route's cumulative suffix-diff + reasoning extractor (the
    shared core of routes/inference.py gguf_stream_chunks and the tool-loop
    consumer) over content snapshots. Returns (visible, reasoning)."""
    from routes.inference import _ResponsesReasoningExtractor

    extractor = _ResponsesReasoningExtractor(parse_think_markers = True)
    prev_text = ""
    visible: list[str] = []
    reasoning: list[str] = []
    for cumulative in cumulatives:
        new_text = cumulative[len(prev_text) :]
        prev_text = cumulative
        if not new_text:
            continue
        reasoning_delta, visible_delta = extractor.feed(new_text)
        if reasoning_delta:
            reasoning.append(reasoning_delta)
        if visible_delta:
            visible.append(visible_delta)
    final_reasoning, final_visible = extractor.finish()
    if final_reasoning:
        reasoning.append(final_reasoning)
    if final_visible:
        visible.append(final_visible)
    return "".join(visible), "".join(reasoning)


def test_reasoning_only_route_output_matches_no_tool_path(monkeypatch):
    # Parity contract: a reasoning-only reply must reach the client identically
    # whether tools are on or off. Both generators stream <think> live then
    # resolve to the bare reasoning text; the route's suffix-diff + extractor
    # must therefore produce the same (visible, reasoning) split for both.
    stream = [
        _sse({"reasoning_content": "The capital"}),
        _sse({"reasoning_content": " of France is Paris."}),
        _done(),
    ]

    tool_backend = _make_backend(monkeypatch, [list(stream)], [])
    _patch_monotonic(monkeypatch, [1.0, 2.0, 2.0])
    tool_cumulatives = [
        e["text"]
        for e in tool_backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "capital of France?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
        if e.get("type") == "content"
    ]

    no_tool_backend = _make_backend(monkeypatch, [list(stream)], [])
    no_tool_cumulatives = [
        y
        for y in no_tool_backend.generate_chat_completion(
            messages = [{"role": "user", "content": "capital of France?"}],
        )
        if isinstance(y, str)
    ]

    # Both paths stream the reasoning live with the same leading shape. (Raw
    # yield lists aren't compared verbatim: the tool path emits a pre-existing
    # duplicate trailing event that the route's suffix-diff dedupes.)
    assert tool_cumulatives[:3] == no_tool_cumulatives[:3]
    # The contract that matters: identical route-level output.
    tool_out = _replay_route_reasoning_extractor(tool_cumulatives)
    no_tool_out = _replay_route_reasoning_extractor(no_tool_cumulatives)
    assert tool_out == no_tool_out
    # Pin the shared contract so a change to either path shows up here.
    _visible, reasoning = tool_out
    assert reasoning == "The capital of France is Paris."


def test_reasoning_before_bare_json_tool_closes_think_block(monkeypatch):
    # _drain_silently sibling of the structured-tool close: a bare-JSON tool call
    # with a live reasoning prefix must also close </think> before draining, and
    # must never leak the drained call text as content.
    tool_stream = [
        _sse({"reasoning_content": "Searching now."}),
        _sse({"content": '{"name":"web_search","arguments":{"query":"weather"}}'}),
        _done(),
    ]
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    tool_start_index = next(i for i, e in enumerate(events) if e["type"] == "tool_start")
    content_before_tool = [e["text"] for e in events[:tool_start_index] if e["type"] == "content"]
    assert content_before_tool[0] == "<think>Searching now."
    assert content_before_tool[-1] == "<think>Searching now.</think>"
    # The bare-JSON call text was drained, never surfaced as content.
    assert not any('"name"' in t for t in content_before_tool)


def test_consumed_tool_final_pass_emits_latest_reasoning_summary(monkeypatch):
    tool_stream = [
        _sse({"reasoning_content": "Need a render."}),
        _sse(
            {
                "content": '<tool_call>{"name":"render_html","arguments":{"code":"<html>ok</html>"}}</tool_call>'
            }
        ),
        _done(),
    ]
    final_stream = [
        _sse({"reasoning_content": "Now synthesize."}),
        _sse({"content": "Final from tool."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)
    _patch_monotonic(monkeypatch, [200.0, 201.0, 203.0, 300.0, 400.0, 405.0, 405.0])

    def fake_execute_tool(name, arguments, **_kwargs):
        return "Rendered HTML canvas: Done."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "render then answer"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            max_tool_iterations = 1,
        )
    )

    summaries = [event for event in events if event["type"] == "reasoning_summary"]
    assert [event["duration_ms"] for event in summaries] == [2000, 5000]
    final_summary_index = events.index(summaries[-1])
    final_content_index = next(
        i
        for i, event in enumerate(events)
        if event.get("type") == "content" and "Final from tool." in event.get("text", "")
    )
    assert final_summary_index < final_content_index


def test_repeat_render_html_nudge_is_not_user_visible_error(monkeypatch):
    """A repeated render_html call is an internal no-op, not a visible card."""

    first_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps(
                                {
                                    "code": "<html><body>first</body></html>",
                                    "title": "First",
                                }
                            ),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    repeat_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_repeat",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps(
                                {
                                    "code": "<html><body>repeat</body></html>",
                                    "title": "Repeat",
                                }
                            ),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Short note."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, repeat_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML canvas: First."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
        {"type": "function", "function": {"name": "web_search"}},
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 2,
        )
    )

    assert calls == [
        (
            "render_html",
            {"code": "<html><body>first</body></html>", "title": "First"},
        )
    ]
    assert _tool_names(payloads[1]) == ["web_search"]

    actual_tool_starts = [
        event
        for event in events
        if event.get("type") == "tool_start" and event.get("arguments", {}).get("code")
    ]
    tool_ends = [
        event
        for event in events
        if event.get("type") == "tool_end" and event.get("tool_name") == "render_html"
    ]
    assert len(actual_tool_starts) == 1
    assert len(tool_ends) == 1

    assert len(payloads) == 3
    render_tool_messages = [
        message
        for message in payloads[2]["messages"]
        if message.get("role") == "tool" and message.get("name") == "render_html"
    ]
    assert len(render_tool_messages) == 1
    internal_nudges = [
        message
        for message in payloads[2]["messages"]
        if message.get("role") == "user"
        and "Do not call render_html again" in message.get("content", "")
    ]
    assert len(internal_nudges) == 1


def test_render_html_success_drops_tool_schema_before_final_pass(monkeypatch):
    first_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps({"code": "<html>ok</html>"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        return "Rendered HTML canvas: Done."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Render this."}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            max_tool_iterations = 3,
        )
    )

    assert len(payloads) == 2
    assert "tools" not in payloads[1]
    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)
    final_user_messages = [
        m.get("content", "") for m in payloads[1]["messages"] if m.get("role") == "user"
    ]
    assert not any("used all available tool calls" in message for message in final_user_messages)


def test_non_consecutive_duplicate_web_search_is_internal_noop(monkeypatch):
    first_search = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    python_call = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_python",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('ok')"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    duplicate_search = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_2",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from gathered data."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, python_call, duplicate_search, final_stream],
        payloads,
    )

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return f"ok:{name}"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "python"}},
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search gpus in 2026 prices and use python"}],
            tools = tools,
            max_tool_iterations = 3,
        )
    )

    assert calls == [
        ("web_search", {"query": "gpu prices 2026"}),
        ("python", {"code": "print('ok')"}),
    ]
    assert [
        event.get("tool_name")
        for event in events
        if event.get("type") == "tool_start" and event.get("tool_name")
    ] == ["web_search", "python"]
    assert [
        event.get("tool_name")
        for event in events
        if event.get("type") == "tool_end" and event.get("tool_name")
    ] == ["web_search", "python"]
    assert not [
        event
        for event in events
        if event.get("tool_call_id") == "call_search_2"
        and event.get("type") in {"tool_start", "tool_end"}
    ]
    assert len(payloads) == 4
    assert _tool_names(payloads[3]) == ["web_search", "python"]
    duplicate_nudges = [
        message
        for message in payloads[3]["messages"]
        if message.get("role") == "user"
        and "already completed successfully" in message.get("content", "")
    ]
    assert len(duplicate_nudges) == 1


def test_duplicate_web_search_noop_allows_distinct_followup_tool(monkeypatch):
    first_search = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    duplicate_search = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_2",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    python_call = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_python",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('ok')"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from gathered data."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, duplicate_search, python_call, final_stream],
        payloads,
    )

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return f"ok:{name}"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "python"}},
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search gpus in 2026 prices and use python"}],
            tools = tools,
            max_tool_iterations = 4,
        )
    )

    assert calls == [
        ("web_search", {"query": "gpu prices 2026"}),
        ("python", {"code": "print('ok')"}),
    ]
    assert [
        event.get("tool_name")
        for event in events
        if event.get("type") == "tool_start" and event.get("tool_name")
    ] == ["web_search", "python"]
    assert [
        event.get("tool_name")
        for event in events
        if event.get("type") == "tool_end" and event.get("tool_name")
    ] == ["web_search", "python"]
    assert not [
        event
        for event in events
        if event.get("tool_call_id") == "call_search_2"
        and event.get("type") in {"tool_start", "tool_end"}
    ]
    assert len(payloads) == 4
    assert _tool_names(payloads[2]) == ["web_search", "python"]
    duplicate_nudges = [
        message
        for message in payloads[2]["messages"]
        if message.get("role") == "user"
        and "already completed successfully" in message.get("content", "")
    ]
    assert len(duplicate_nudges) == 1


def test_repeated_duplicate_noop_transitions_to_final_pass(monkeypatch):
    first_search = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    duplicate_one = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_2",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    duplicate_two = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_3",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from first search."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, duplicate_one, duplicate_two, final_stream],
        payloads,
    )
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search gpus"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 10,
        )
    )

    assert calls == [("web_search", {"query": "gpu prices 2026"})]
    assert [event.get("tool_call_id") for event in events if event.get("type") == "tool_end"] == [
        "call_search_1"
    ]
    assert len(payloads) == 4
    assert "tools" not in payloads[-1]
    assert any(
        event.get("type") == "content" and event.get("text") == "Final answer from first search."
        for event in events
    )


def test_same_turn_duplicate_web_search_is_internal_noop(monkeypatch):
    same_turn_duplicates = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_search_2",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "gpu prices 2026"}),
                        },
                    },
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [same_turn_duplicates, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search gpus"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [("web_search", {"query": "gpu prices 2026"})]
    assert [event.get("tool_call_id") for event in events if event.get("type") == "tool_end"] == [
        "call_search_1"
    ]
    assert not [
        event
        for event in events
        if event.get("tool_call_id") == "call_search_2"
        and event.get("type") in {"tool_start", "tool_end"}
    ]


def test_same_turn_duplicate_does_not_drop_later_parallel_call(monkeypatch):
    # One batch: search(a), search(a) [duplicate], search(b). The duplicate is an
    # internal no-op, but the distinct search(b) after it must still run, and the
    # no-op nudge must land after the tool results rather than splitting them.
    batch = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": json.dumps({"query": "a"})},
                    },
                    {
                        "index": 1,
                        "id": "call_a2",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": json.dumps({"query": "a"})},
                    },
                    {
                        "index": 2,
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": json.dumps({"query": "b"})},
                    },
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [batch, final_stream], payloads)

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append(arguments)
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 3,
        )
    )

    # Both distinct calls ran; the duplicate did not (old `break` dropped search(b)).
    assert calls == [{"query": "a"}, {"query": "b"}]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a1",
        "call_b",
    ]

    # The next generation's conversation must be well-formed: the assistant lists
    # only the executed calls (no orphan for the duplicate), the two tool results
    # follow contiguously, and the no-op nudge lands after them, never between.
    conv = payloads[1]["messages"]
    asst = next(m for m in conv if m["role"] == "assistant" and m.get("tool_calls"))
    assert [tc.get("id") for tc in asst["tool_calls"]] == ["call_a1", "call_b"]
    after = conv[conv.index(asst) + 1 :]
    assert [m["role"] for m in after[:2]] == ["tool", "tool"]
    assert [m.get("tool_call_id") for m in after[:2]] == ["call_a1", "call_b"]
    assert after[2]["role"] == "user"  # deferred duplicate nudge, after the results
    assert after[2]["content"].startswith(
        "One earlier request to call tool 'web_search' in this batch was not executed"
    )
    assert "previous tool request" not in after[2]["content"].lower()


def test_same_turn_repeated_render_html_does_not_emit_second_provisional_start(monkeypatch):
    same_turn_render_calls = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_html_1",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps({"code": "<html>one</html>"}),
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_html_2",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps({"code": "<html>two</html>"}),
                        },
                    },
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [same_turn_render_calls, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML canvas: One."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "render html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [("render_html", {"code": "<html>one</html>"})]
    assert [
        event.get("tool_call_id")
        for event in events
        if event.get("type") == "tool_start" and not event.get("arguments")
    ] == ["call_html_1"]
    assert not [
        event
        for event in events
        if event.get("tool_call_id") == "call_html_2"
        and event.get("type") in {"tool_start", "tool_end"}
    ]
    assert len(payloads) == 2
    assert "tools" not in payloads[1]
    render_nudges = [
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "user"
        and "Do not call render_html again" in message.get("content", "")
    ]
    assert len(render_nudges) == 1


def test_disabled_tool_call_is_internal_noop(monkeypatch):
    disabled_python = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_python_disabled",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print(1)"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [disabled_python, final_stream], payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run python"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert not [event for event in events if event.get("type") in {"tool_start", "tool_end"}]
    assert len(payloads) == 2
    disabled_nudges = [
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "user" and "not enabled" in message.get("content", "")
    ]
    assert len(disabled_nudges) == 1


def test_render_html_success_does_not_reprompt_render_html_intent(monkeypatch):
    """After render_html succeeds, do not force another render_html call.

    The post-tool model pass can say it will use render_html again without
    emitting a tool call. That should be accepted as a final model mistake,
    not turned into repeated internal re-prompts after the canvas already
    exists.
    """

    first_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps(
                                {
                                    "code": "<html><body>first</body></html>",
                                    "title": "First",
                                }
                            ),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    post_tool_stream = [
        _sse({"content": "I will now use render_html again."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, post_tool_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML canvas: First."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
        )
    )

    assert len(payloads) == 2
    assert len(calls) == 1
    assert any(
        event.get("type") == "content" and event.get("text") == "I will now use render_html again."
        for event in events
    )


def test_internal_reprompt_attempts_do_not_duplicate_visible_text(monkeypatch):
    """No-tool re-prompt attempts should not concatenate into the UI."""

    # One initial response plus one stream per re-prompt; derive the count from the shared cap.
    streams = [[_sse({"content": "I will use render_html now."}), _done()]]
    streams += [
        [_sse({"content": "Understood. I will use render_html now."}), _done()]
        for _ in range(_MAX_REPROMPTS)
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == _MAX_REPROMPTS + 1


def test_forced_reprompt_plain_final_answer_is_visible(monkeypatch):
    """A hidden forced re-prompt may fall back to a plain final answer."""

    streams = [
        [_sse({"content": "I will use render_html now."}), _done()],
        [
            _sse({"content": "No tool is needed. Final answer: use a red square."}),
            _done(),
        ],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "render_html",
                        "description": "Render HTML.",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                }
            ],
            max_tool_iterations = 1,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == [
        "I will use render_html now.",
        "No tool is needed. Final answer: use a red square.",
    ]
    assert len(payloads) == 2


def test_internal_reprompt_disabled_when_auto_heal_disabled(monkeypatch):
    streams = [[_sse({"content": "I will use render_html now."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
            auto_heal_tool_calls = False,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == 1


def test_internal_reprompt_disabled_when_nudge_tool_calls_false(monkeypatch):
    # Explicit nudge_tool_calls=False disables the plan-without-action
    # re-prompt even with Auto-Heal on (None keeps the default-on behavior).
    streams = [[_sse({"content": "I will use render_html now."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
            auto_heal_tool_calls = True,
            nudge_tool_calls = False,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == 1


def test_auto_heal_disabled_parses_well_formed_xml_when_tools_enabled(monkeypatch):
    streams = [
        [
            _sse(
                {
                    "content": '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                }
            ),
            _done(),
        ],
        [_sse({"content": "done"}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            auto_heal_tool_calls = False,
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "x"})]
    assert not any(
        event.get("type") == "content" and "<tool_call>" in event.get("text", "")
        for event in events
    )


def test_textual_mistral_marker_not_leaked_when_inline_with_preface(monkeypatch):
    # Textual Mistral ``[TOOL_CALLS]`` inline with visible preface: the DRAINING flush must use the
    # shared parser patterns (which know ``[TOOL_CALLS]``); the legacy set leaked the marker to clients.
    streams = [
        [_sse({"content": 'Let me search. [TOOL_CALLS]web_search{"query":"cats"}'}), _done()],
        [_sse({"content": "done"}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("[TOOL_CALLS]" not in t for t in content_texts), content_texts
    assert any("Let me search." in t for t in content_texts)


def test_textual_explicit_id_reuses_provisional_card(monkeypatch):
    # A textual Mistral-style call with an explicit ``id`` must reconcile onto the
    # open provisional TEXT card (keyed "call_0"), not spawn a duplicate under the
    # explicit id (which the parser keeps for execution).
    big_query = "cats " * 80  # push the drained call past the provisional floor
    call = "[TOOL_CALLS]" + json.dumps(
        [{"name": "web_search", "arguments": {"query": big_query}, "id": "explicit-42"}]
    )
    assert len(call) > 256
    # Small chunks so the provisional card opens mid-generation (a single-shot
    # delta parses instantly and never shows a provisional to exercise).
    chunks = [call[i : i + 24] for i in range(0, len(call), 24)]
    streams = [
        [_sse({"content": c}) for c in chunks] + [_done()],
        [_sse({"content": "done"}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": big_query})]
    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    # Empty-args card = provisional open; full-args card = reconciled real start.
    provisional = [e for e in tool_starts if not e.get("arguments")]
    real = [e for e in tool_starts if e.get("arguments", {}).get("query")]
    assert len(provisional) == 1, tool_starts  # provisional actually opened
    prov_id = provisional[0]["tool_call_id"]
    # Exactly one real card, sharing the provisional id, not a duplicate under
    # the explicit "explicit-42" id.
    assert len(real) == 1, tool_starts
    assert real[0]["tool_call_id"] == prov_id
    assert real[0]["tool_name"] == "web_search"
    assert {e["tool_call_id"] for e in tool_starts} == {prov_id}
    # A single tool_end reconciles the card; no stale empty-result close.
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert [e["tool_call_id"] for e in ends] == [prov_id]
    assert ends[0]["result"] == "result"


def test_textual_llama_python_tag_marker_not_leaked(monkeypatch):
    # Same leak class for the Llama-3 built-in ``<|python_tag|>NAME.call(...)`` form.
    streams = [
        [_sse({"content": '<|python_tag|>web_search.call(query="cats")'}), _done()],
        [_sse({"content": "done"}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("<|python_tag|>" not in t for t in content_texts), content_texts


def test_reprompted_tool_call_still_streams_final_answer(monkeypatch):
    """Suppression ends once a forced re-prompt actually calls a tool."""

    streams = [
        [_sse({"content": "I will use render_html now."}), _done()],
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_forced",
                            "type": "function",
                            "function": {
                                "name": "render_html",
                                "arguments": json.dumps(
                                    {
                                        "code": "<html><body>forced</body></html>",
                                        "title": "Forced",
                                    }
                                ),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": "Final note after tool."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML canvas: Forced."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
        )
    )

    assert len(calls) == 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now.", "Final note after tool."]
    assert len(payloads) == 3


def test_confirm_tool_calls_allow_executes_gguf_tool(monkeypatch):
    streams = [
        _structured_tool_call("python", {"code": "print(1)"}, "call_py"),
        [_sse({"content": "Done."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    monkeypatch.setattr("core.inference.llama_cpp.new_approval_id", lambda: "approval-1")
    monkeypatch.setattr(
        "core.inference.llama_cpp.begin_tool_decision",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr("core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: "allow")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run python"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
            confirm_tool_calls = True,
            session_id = "sess",
        )
    )

    starts = [event for event in events if event.get("type") == "tool_start"]
    assert len(starts) == 1
    assert starts[0]["approval_id"]
    assert starts[0]["awaiting_confirmation"] is True
    assert calls == [("python", {"code": "print(1)"})]
    assert any(event.get("type") == "tool_end" and event.get("result") == "OK" for event in events)


def test_confirm_tool_calls_close_after_prompt_cleans_gguf_slot(monkeypatch):
    approval_id = "approval-close"
    streams = [_structured_tool_call("python", {"code": "print(1)"}, "call_py")]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("tool should not run")),
    )
    monkeypatch.setattr("core.inference.llama_cpp.new_approval_id", lambda: approval_id)

    with tool_approvals._lock:
        tool_approvals._pending.clear()

    gen = backend.generate_chat_completion_with_tools(
        messages = [{"role": "user", "content": "run python"}],
        tools = [{"type": "function", "function": {"name": "python"}}],
        max_tool_iterations = 1,
        confirm_tool_calls = True,
        session_id = "sess",
    )
    try:
        assert next(gen)["type"] == "status"
        start = next(gen)
        assert start["type"] == "tool_start"
        assert start["approval_id"] == approval_id
        with tool_approvals._lock:
            assert approval_id in tool_approvals._pending
    finally:
        gen.close()

    with tool_approvals._lock:
        assert approval_id not in tool_approvals._pending
    assert resolve_tool_decision(approval_id, "allow", session_id = "sess") is False


def test_confirm_tool_calls_skips_gguf_rag_autoinject(monkeypatch):
    streams = [[_sse({"content": "Done."}), _done()]]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    def fail_autoinject(*_args, **_kwargs):
        raise AssertionError("RAG autoinject must not run before approval")

    monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fail_autoinject)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "use docs"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            max_tool_iterations = 1,
            confirm_tool_calls = True,
            session_id = "sess",
            rag_scope = {"thread_id": "t1"},
        )
    )

    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)


def test_confirm_tool_calls_deny_skips_gguf_tool_and_retry_can_execute(monkeypatch):
    same_call = _structured_tool_call("python", {"code": "print(1)"}, "call_py")
    streams = [
        same_call,
        _structured_tool_call("python", {"code": "print(1)"}, "call_py_retry"),
        [_sse({"content": "Done."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    decisions = iter(["deny", "allow"])
    approvals = iter(["approval-1", "approval-2"])

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    monkeypatch.setattr("core.inference.llama_cpp.new_approval_id", lambda: next(approvals))
    monkeypatch.setattr(
        "core.inference.llama_cpp.begin_tool_decision",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.wait_tool_decision",
        lambda *_a, **_k: next(decisions),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run python"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 2,
            confirm_tool_calls = True,
            session_id = "sess",
        )
    )

    starts = [event for event in events if event.get("type") == "tool_start"]
    ends = [event for event in events if event.get("type") == "tool_end"]
    assert len(starts) == 2
    assert [event["result"] for event in ends] == [TOOL_REJECTED_MESSAGE, "OK"]
    assert calls == [("python", {"code": "print(1)"})]


def _streamed_structured_tool_call(
    tool_name: str,
    arguments: dict,
    call_id: str,
    frag: int = 24,
) -> list[str]:
    """A structured tool call whose arguments arrive token-by-token across many
    deltas (id + name on the first delta), mirroring how llama-server streams a
    large tool-call argument such as a full HTML/code file."""
    args_json = json.dumps(arguments)
    fragments = [args_json[i : i + frag] for i in range(0, len(args_json), frag)] or [""]
    chunks = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": fragments[0]},
                    }
                ]
            }
        )
    ]
    for fragment in fragments[1:]:
        chunks.append(_sse({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}))
    chunks.append(_done())
    return chunks


def test_large_python_tool_call_emits_early_provisional_start(monkeypatch):
    """Regression: a large streamed tool-call argument surfaces a provisional
    tool card BEFORE the full arguments finish, so the UI shows progress during
    generation instead of a frozen 'Generating...'. (The bug: only render_html
    surfaced early; python/terminal/etc. were silent until the call completed.)"""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    args_json = json.dumps({"code": big_code})
    assert len(args_json) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_py_big")
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write code"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    real = [e for e in tool_starts if e.get("arguments", {}).get("code")]

    # Exactly one provisional (empty args) and one real (full args), same id so
    # the frontend reconciles them into a single card.
    assert len(provisional) == 1, tool_starts
    assert provisional[0]["tool_name"] == "python"
    assert provisional[0]["tool_call_id"] == "call_py_big"
    assert provisional[0]["provenance"].get("provisional") is True
    assert len(real) == 1
    assert real[0]["tool_call_id"] == "call_py_big"
    # The provisional card appears before the real (completed) tool_start.
    assert events.index(provisional[0]) < events.index(real[0])

    assert calls == [("python", {"code": big_code})]
    assert any(e.get("type") == "tool_end" and e.get("tool_name") == "python" for e in events)


def test_auto_mode_render_html_suppresses_provisional_card_under_confirm(monkeypatch):
    """render_html is no longer unconditionally safe (a networked canvas asks), so
    with confirm_tool_calls set under permission_mode="auto" its early provisional
    card is suppressed; the real full-argument tool_start still fires and a static
    canvas runs without a prompt."""
    args = {"code": "<html>" + "x" * 80 + "</html>"}
    first_stream = _streamed_structured_tool_call("render_html", args, "call_rh")
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda name, arguments, **_k: "OK")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "make a card"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            confirm_tool_calls = True,
            permission_mode = "auto",
            max_tool_iterations = 1,
        )
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    # The confirm gate now suppresses the early provisional card for render_html.
    assert provisional == [], tool_starts
    real = [e for e in tool_starts if e.get("arguments")]
    assert real and real[0]["tool_name"] == "render_html"
    # A static canvas is classified safe, so it still runs without an approval gate.
    assert real[0].get("awaiting_confirmation") in (False, None)


def test_small_python_tool_call_has_no_provisional_start(monkeypatch):
    """A small tool-call argument finishes streaming instantly, so it keeps the
    existing behavior of a single (real) tool_start with no provisional card."""

    first_stream = _structured_tool_call("python", {"code": "print(1)"}, "call_py_small")
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "x"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    assert [e for e in tool_starts if not e.get("arguments")] == []
    assert len([e for e in tool_starts if e.get("arguments", {}).get("code")]) == 1


def _streamed_parallel_tool_calls(specs, frag: int = 24) -> list[str]:
    """Two or more structured tool calls, each streamed token-by-token across
    deltas, one index fully before the next, mirroring how llama-server streams
    several parallel tool calls whose arguments are large."""
    chunks: list[str] = []
    for index, (tool_name, arguments, call_id) in enumerate(specs):
        args_json = json.dumps(arguments)
        fragments = [args_json[i : i + frag] for i in range(0, len(args_json), frag)] or [""]
        chunks.append(
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": fragments[0]},
                        }
                    ]
                }
            )
        )
        for fragment in fragments[1:]:
            chunks.append(
                _sse({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]})
            )
    chunks.append(_done())
    return chunks


def test_parallel_large_tool_calls_each_emit_provisional_start(monkeypatch):
    """With parallel tool use enabled (the default), every streamed large tool
    call surfaces its own provisional card, not just the first one, so the UI
    shows progress for each call as its arguments stream."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    big_cmd = "echo start\n" + "\n".join(f"echo line {i}" for i in range(60))
    assert len(json.dumps({"code": big_code})) > _PROVISIONAL_ARGS_MIN_CHARS
    assert len(json.dumps({"command": big_cmd})) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_parallel_tool_calls(
        [
            ("python", {"code": big_code}, "call_py"),
            ("terminal", {"command": big_cmd}, "call_term"),
        ]
    )
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "do both"}],
            tools = [
                {"type": "function", "function": {"name": "python"}},
                {"type": "function", "function": {"name": "terminal"}},
            ],
            max_tool_iterations = 1,
        )
    )

    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert sorted(e["tool_call_id"] for e in provisional) == ["call_py", "call_term"]
    assert all(e["provenance"].get("provisional") is True for e in provisional)
    # Both calls actually executed (parallel tool use is enabled by default).
    assert sorted(name for name, _ in calls) == ["python", "terminal"]


def test_parallel_disabled_suppresses_provisional_for_later_calls(monkeypatch):
    """When parallel tool use is disabled the downstream truncates to the first
    call, so only the first streamed call may surface a provisional; a later
    call must not get a card that could never reconcile or be closed."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    big_cmd = "echo start\n" + "\n".join(f"echo line {i}" for i in range(60))

    first_stream = _streamed_parallel_tool_calls(
        [
            ("python", {"code": big_code}, "call_py"),
            ("terminal", {"command": big_cmd}, "call_term"),
        ]
    )
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "do both"}],
            tools = [
                {"type": "function", "function": {"name": "python"}},
                {"type": "function", "function": {"name": "terminal"}},
            ],
            max_tool_iterations = 1,
            disable_parallel_tool_use = True,
        )
    )

    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert [e["tool_call_id"] for e in provisional] == ["call_py"]
    # Only the first call executes when parallel use is disabled.
    assert calls == [("python", {"code": big_code})]
    # The lone provisional is closed exactly once (no dangling card).
    closing = [
        e for e in events if e.get("type") == "tool_end" and e.get("tool_call_id") == "call_py"
    ]
    assert len(closing) == 1


def test_connect_error_during_tool_call_closes_provisional_card(monkeypatch):
    """If llama-server drops mid tool-call after a provisional card is shown, the
    loop must close that card before surfacing the error so the UI never leaves a
    tool spinning forever."""
    import httpx

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    fragments = _streamed_structured_tool_call("python", {"code": big_code}, "call_py_err")
    # Drop the trailing [DONE]; raise a connection error after the fragments
    # stream (and after the provisional card has been emitted).
    fragments = fragments[:-1]

    def raising_stream():
        for chunk in fragments:
            yield chunk
        raise httpx.ConnectError("connection lost mid stream")

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [raising_stream()], payloads)

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    collected: list[dict] = []
    raised = False
    gen = backend.generate_chat_completion_with_tools(
        messages = [{"role": "user", "content": "write code"}],
        tools = [{"type": "function", "function": {"name": "python"}}],
        max_tool_iterations = 1,
    )
    try:
        for event in gen:
            collected.append(event)
    except RuntimeError as exc:
        raised = True
        assert "Lost connection" in str(exc)

    assert raised
    provisional = [e for e in collected if e.get("type") == "tool_start" and not e.get("arguments")]
    assert len(provisional) == 1
    assert provisional[0]["tool_call_id"] == "call_py_err"
    # The provisional card is closed before the error propagates.
    closing = [
        e
        for e in collected
        if e.get("type") == "tool_end" and e.get("tool_call_id") == "call_py_err"
    ]
    assert len(closing) == 1
    # The closing card is marked as an error, not an empty success, so the UI
    # renders it as failed.
    assert "Error" in (closing[0].get("result") or "")


def test_empty_tool_call_id_does_not_emit_provisional_card(monkeypatch):
    """llama.cpp can stream a tool call whose id is an empty string. A provisional
    card keyed by "" cannot reconcile with the real tool_start (the frontend mints
    its own id per event), so it must not be emitted -- otherwise the empty card
    would dangle. The real call must still execute normally."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    assert len(json.dumps({"code": big_code})) > _PROVISIONAL_ARGS_MIN_CHARS

    # Same large streamed call as the provisional test, but with an empty id.
    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "")
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write code"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    # No provisional card (empty-args tool_start) was surfaced for the empty id.
    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert provisional == []
    # The real call still executes despite the missing id.
    assert calls == [("python", {"code": big_code})]


def _streamed_content(text: str, frag: int = 4) -> list[str]:
    """Stream content token-by-token like llama-server; ``frag`` sets the chunk size."""
    chunks = [_sse({"content": text[i : i + frag]}) for i in range(0, len(text), frag)]
    chunks.append(_done())
    return chunks


def test_bare_json_tool_call_streamed_is_not_leaked_and_executes(monkeypatch):
    """A wrapper-less bare-JSON call must be held while incomplete, drained silently, and executed with nothing leaking."""

    bare_call = '{"name": "web_search", "parameters": {"query": "weather in Sydney"}}'
    first_stream = _streamed_content(bare_call)
    final_stream = [_sse({"content": "It is sunny in Sydney."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Weather: sunny, 22C."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather in Sydney?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    # The tool ran with the parsed arguments.
    assert calls == [("web_search", {"query": "weather in Sydney"})]
    assert any(
        event.get("type") == "tool_end" and event.get("tool_name") == "web_search"
        for event in events
    )

    # The bare JSON never leaked to the user-visible stream.
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all('"name"' not in t for t in content_texts), content_texts
    assert all("web_search" not in t for t in content_texts), content_texts
    # The post-tool synthesis is still streamed.
    assert any("sunny in Sydney" in t for t in content_texts), content_texts


def test_ordinary_json_with_name_key_is_shown_not_treated_as_tool_call(monkeypatch):
    """Markerless JSON with a non-enabled name is the answer, not a phantom call."""

    answer = '{"name": "Alice", "parameters": {"age": 30}}'
    first_stream = _streamed_content(answer)
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: (calls.append((n, a)) or "x"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "give me a person record"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_incomplete_bare_json_truncation_is_not_leaked(monkeypatch):
    """If generation is cut off mid bare-JSON object (no closing brace), the held
    fragment must be stripped at stream end rather than dumped to the user."""

    truncated = '{"name": "web_search", "parameters": {"query": "weather in S'
    stream = _streamed_content(truncated)
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no complete call")),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all('{"name"' not in t for t in content_texts), content_texts


def test_gguf_truncated_ordinary_json_with_name_key_is_shown_not_suppressed(monkeypatch):
    """A truncated markerless object whose "name" is NOT an enabled tool (a person
    record cut off mid-stream, ``{"name":"Alice","age":``) must still be shown. The
    end-of-stream ``_is_bare_tc`` heuristic routed any ``{...,"name",...}`` fragment
    to DRAINING (dropped); it is now gated on the enabled tool names so only a real
    truncated tool call is suppressed, ordinary JSON streams through."""

    truncated = '{"name": "Alice", "age": 30, "bio": "loves '
    stream = _streamed_content(truncated)
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: (calls.append((n, a)) or "x"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "start a person record"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_gguf_truncated_disabled_name_json_is_preserved_when_tools_active(monkeypatch):
    """A truncated JSON answer with a non-enabled name must still be shown (resolvers are gated on enabled names)."""

    truncated = '{"name": "Alice", "parameters": {"age": 30'
    stream = _streamed_content(truncated)
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: (calls.append((n, a)) or "x"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "give json"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_gguf_truncated_enabled_name_json_is_still_suppressed(monkeypatch):
    """Counterpart guard: a truncated ENABLED-tool bare call (``web_search``) cut off
    mid-JSON still must NOT leak -- the gate only spares disabled / non-tool names."""

    truncated = '{"name": "web_search", "parameters": {"query": "weather in S'
    stream = _streamed_content(truncated)
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no complete call")),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all('{"name"' not in t for t in content_texts), content_texts


def test_gguf_oversized_disabled_name_json_is_preserved(monkeypatch):
    """An oversized still-open JSON answer with a non-enabled name streams as content, not a phantom drain."""

    cap = 16384
    big = "A" * (cap + 5000)
    answer = '{"name":"Alice","parameters":{"bio":"' + big  # never closes
    first_stream = [_sse({"content": answer[i : i + 2000]}) for i in range(0, len(answer), 2000)]
    first_stream.append(_done())
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: (calls.append((n, a)) or "x"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "long json"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts[:1]


def test_gemma_wrapperless_call_streamed_is_not_leaked_and_executes(monkeypatch):
    """Gemma 4 GGUF (skip_special_tokens) streams a wrapper-less ``call:NAME{..}``
    with no XML signal. Like bare JSON, the BUFFERING scan must recognise it via
    _GEMMA_BARE_TC_RE, drain it silently, and execute the tool -- never leaking
    the ``call:`` markup to the user-visible stream."""

    gemma_call = 'call:web_search{query:"weather in Sydney"}'
    first_stream = _streamed_content(gemma_call)
    final_stream = [_sse({"content": "It is sunny in Sydney."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Weather: sunny, 22C."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather in Sydney?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "weather in Sydney"})]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("call:" not in t for t in content_texts), content_texts
    assert any("sunny in Sydney" in t for t in content_texts), content_texts


def _usage_done(usage: dict, finish_reason: str = "stop") -> str:
    """A terminal SSE chunk carrying llama-server's ``usage`` block, the way the
    real server reports it on the final chunk of a completion."""
    return (
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": usage,
            }
        )
        + "\n"
    )


def test_metadata_event_preserves_prompt_tokens_details(monkeypatch):
    """The tool loop's metadata event must carry llama-server's
    ``prompt_tokens_details`` (KV-cache hits) through ``_build_metadata_event``,
    so the route reports real ``cached_tokens`` instead of always 0 (#6570).

    This drives the *real* generator; the route-level test feeds a pre-built
    metadata event and so never exercises this code.
    """
    stream = [
        _sse({"content": "The answer is 42."}),
        _usage_done(
            {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 16},
            }
        ),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            max_tool_iterations = 1,
        )
    )

    metadata = [e for e in events if e.get("type") == "metadata"]
    assert metadata, "expected a metadata event"
    usage = metadata[-1]["usage"]
    assert usage["prompt_tokens_details"] == {"cached_tokens": 16}
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 4


def test_metadata_event_omits_prompt_tokens_details_when_absent(monkeypatch):
    """No KV-cache block from the server -> the key isn't fabricated, so the
    route falls back to its 0-default instead of reading a bogus value."""
    stream = [
        _sse({"content": "hi"}),
        _usage_done({"prompt_tokens": 5, "completion_tokens": 2}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            max_tool_iterations = 1,
        )
    )

    metadata = [e for e in events if e.get("type") == "metadata"]
    assert metadata, "expected a metadata event"
    assert "prompt_tokens_details" not in metadata[-1]["usage"]


def test_gguf_rehearsal_name_split_before_args_is_not_leaked(monkeypatch):
    """Finding 6: a rehearsal call whose name (``web_search``) and ``[ARGS]{...}``
    arrive in separate content deltas must hold the bare name in the buffer until
    ``[ARGS]`` flips it to a drain. Without _is_rehearsal_prefix the GGUF path
    streams the tool name as visible content before the call executes."""

    first_stream = [
        _sse({"content": "web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all("[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_initial_buffer_flush_holds_split_rehearsal_name(monkeypatch):
    """The first flush out of BUFFERING (prose plus a trailing active-tool-name in
    the first delta, ``[ARGS]{...}`` in the next) must apply the same trailing-name
    hold the STREAMING branch uses. The first delta has spaces so it is not a
    rehearsal prefix and falls to the initial flush, which previously emitted the
    bare name before the call drained."""

    first_stream = [
        _sse({"content": "I will use web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all("[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_rehearsal_name_after_prose_in_streaming_is_not_leaked(monkeypatch):
    """Finding 9: the BUFFERING guard only covers a rehearsal at the turn start.
    When prose has already streamed (STREAMING state) and the model then emits the
    tool name and ``[ARGS]{...}`` in later deltas, the bare name must still be held,
    not flushed as visible content before the call drains."""

    first_stream = [
        _sse({"content": "Let me think. "}),
        _sse({"content": "I will search "}),
        _sse({"content": "web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts


def test_gguf_plain_answer_ending_with_tool_name_word_is_preserved(monkeypatch):
    """End-of-stream flush: a plain answer that ENDS on a tool-name word with no
    ``[ARGS]`` following is real prose and must not be dropped by the streaming
    rehearsal hold."""

    first_stream = [
        _sse({"content": "I think "}),
        _sse({"content": "you should "}),
        _sse({"content": "web_search"}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "advise"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any(t.rstrip().endswith("web_search") for t in content_texts), content_texts


def test_gguf_long_tool_name_split_rehearsal_is_not_capped_and_executes(monkeypatch):
    """Finding 11: a realistic MCP name longer than the 32-char buffer cap split as
    NAME then [ARGS]{...} must still be held (a rehearsal prefix is self-bounding),
    so the name does not leak and the call executes."""
    name = "mcp__github__create_pull_request"
    assert len(name) >= 32, len(name)

    first_stream = [
        _sse({"content": name}),
        _sse({"content": '[ARGS]{"x":1}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "done"}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: (calls.append((n, a)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": name}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [(name, {"x": 1})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert not any(name in t for t in content_texts), content_texts


def test_gguf_streaming_keeps_bare_args_before_think_block(monkeypatch):
    """F4: the GGUF streaming strip must run its open-ended ``[ARGS]`` tail cleanup
    only on the LAST segment. A bare ``foo[ARGS]`` (no JSON body, ``foo`` not a tool)
    before a <think> block is prose, not a truncated call, so the final visible text
    must keep it verbatim instead of dropping ``foo[ARGS]`` and corrupting the
    sentence."""

    first_stream = [
        _sse({"content": "Please pass foo[ARGS] "}),
        _sse({"content": "<think>pause</think> "}),
        _sse({"content": "to the template."}),
        _done(),
    ]
    backend = _make_backend(monkeypatch, [first_stream], [])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "x"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert content_texts, events
    assert content_texts[-1] == "Please pass foo[ARGS] <think>pause</think> to the template."


def test_gguf_inactive_name_args_in_prose_is_not_drained(monkeypatch):
    """BUG A: an inactive-name ``foo[ARGS]{...}`` in a prose answer must not be treated
    as a tool call. The BUFFERING and end-of-stream safety-net ``[ARGS]`` checks gate on
    active tool names (like the safetensors loop and the mid-stream path), so ``foo``
    (``web_search`` is the only enabled tool) is neither drained/parsed into a disabled
    no-op nor forced into another generation turn."""
    first_stream = [
        _sse({"content": 'foo[ARGS]{"x":1} is just syntax.'}),
        _done(),
    ]
    backend = _make_backend(monkeypatch, [first_stream], [])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "x"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    # No tool executed for the inactive name; a spurious no-op re-prompt would exhaust the
    # single supplied stream and error.
    assert calls == [], calls
    assert not any(e.get("type") in ("tool_start", "tool_end") for e in events), events
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    # The inactive ``foo[ARGS]{...}`` is prose: the name-gated strip keeps the whole sentence.
    assert any('foo[ARGS]{"x":1} is just syntax.' in t for t in content_texts), content_texts


def test_gguf_inactive_rehearsal_before_active_call_executes_and_keeps_prose(monkeypatch):
    """BUG X (#5704): an inactive ``foo[ARGS]{...}`` before a real ``web_search[ARGS]{...}``
    in one delta must NOT swallow the real call; web_search executes while the inactive
    rehearsal stays visible as prose."""
    first_stream = [
        _sse({"content": 'foo[ARGS]{"a":1} web_search[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    # The real call runs; ``foo`` is not executed as a phantom disabled call.
    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    # The inactive rehearsal is preserved as prose; the active one is stripped.
    assert any('foo[ARGS]{"a":1}' in t for t in content_texts), content_texts
    assert all("web_search[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_rehearsal_detection_recognises_spent_one_shot_with_original_tools():
    # Rehearsal detection is fed the ORIGINAL tool list, so a spent one-shot's re-emitted
    # repeat is still detected (matching the strip gate) instead of blanking the turn.
    from core.inference.llama_cpp import _gguf_has_genuine_tool_signal
    from core.inference.tool_call_parser import TOOL_XML_SIGNALS

    repeat = 'render_html[ARGS]{"code":"<html>x</html>"}'
    active_only = [{"type": "function", "function": {"name": "web_search"}}]
    original = active_only + [{"type": "function", "function": {"name": "render_html"}}]
    assert not _gguf_has_genuine_tool_signal(repeat, TOOL_XML_SIGNALS, active_only)
    assert _gguf_has_genuine_tool_signal(repeat, TOOL_XML_SIGNALS, original)


def test_gguf_rehearsal_prefix_and_tail_hold_recognise_spent_one_shot():
    # The BUFFERING prefix check and STREAMING/flush tail-holds use the ORIGINAL tool list,
    # so a spent one-shot's split repeat is held rather than leaked as visible text.
    from core.inference.llama_cpp import _held_rehearsal_tail_len, _is_rehearsal_prefix

    active_only = [{"type": "function", "function": {"name": "web_search"}}]
    original = active_only + [{"type": "function", "function": {"name": "render_html"}}]
    assert not _is_rehearsal_prefix("render_html", active_only)
    assert _is_rehearsal_prefix("render_html", original)
    assert _held_rehearsal_tail_len("answer render_html", active_only) == 0
    assert _held_rehearsal_tail_len("answer render_html", original) == len("render_html")


def test_gguf_oversized_bare_json_not_leaked_and_executes(monkeypatch):
    """An oversized bare-JSON call drains rather than streams, and still executes via the safety net."""

    cap = 16384
    big = "A" * (cap + 5000)
    full = '{"name":"python","parameters":{"code":"' + big + '"}}'
    first_stream = [_sse({"content": full[i : i + 2000]}) for i in range(0, len(full), 2000)]
    first_stream.append(_done())
    final_stream = [_sse({"content": "done"}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "OK"),
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert not any(t.lstrip().startswith('{"name') for t in content_texts), content_texts[:1]
    assert calls and calls[0][0] == "python"
    assert len(calls[0][1].get("code", "")) > cap


def test_gguf_bare_json_call_not_replayed_in_next_turn_content(monkeypatch):
    """After a bare-JSON call executes, the kept assistant message must not carry the raw call as content."""

    import copy

    first_stream = [
        _sse({"content": '{"name":"web_search","parameters":{"query":"cats"}}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "RESULT")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert len(payloads) >= 2
    asst = [m for m in payloads[1]["messages"] if m.get("role") == "assistant"]
    assert asst and not any('"name"' in (m.get("content") or "") for m in asst), asst


def test_gguf_textual_fallback_caps_distinct_tool_calls_per_turn(monkeypatch):
    """A single textual-fallback turn that parses many DISTINCT tool calls must be
    capped at _MAX_TOOL_CALLS_PER_TURN (structured delta.tool_calls are grammar
    bounded by llama-server; text parsed from content is not). Mirrors the
    safetensors loop so one runaway turn cannot fan out into dozens of executions."""
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN

    n = _MAX_TOOL_CALLS_PER_TURN + 4
    blocks = "".join(
        '<tool_call>{"name":"t%d","arguments":{"i":%d}}</tool_call>' % (i, i) for i in range(n)
    )
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "OK"),
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(n)],
            max_tool_iterations = 1,
        )
    )

    assert len(calls) == _MAX_TOOL_CALLS_PER_TURN, [c[0] for c in calls]
    # The cap keeps the first calls in order (no reordering / drop of leading ones).
    assert [c[0] for c in calls] == [f"t{i}" for i in range(_MAX_TOOL_CALLS_PER_TURN)]


def test_gguf_textual_fallback_collapses_duplicate_tool_calls(monkeypatch):
    """Exact-duplicate textual calls in one turn collapse to a single execution."""
    blocks = '<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>' * 5
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "OK"),
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert len(calls) == 1, [c[0] for c in calls]


def test_gguf_drain_truncated_enabled_name_json_preserved_when_auto_heal_disabled(monkeypatch):
    """Auto-Heal OFF keeps a truncated enabled-name fragment visible; ON suppresses it (strip gated on auto_heal_tool_calls)."""

    trunc = '{"name":"web_search","parameters":{"query":"weather'

    def _run(auto_heal):
        stream = [_sse({"content": trunc}), _done()]
        backend = _make_backend(monkeypatch, [stream], [])
        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
        )
        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "x"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                max_tool_iterations = 1,
                auto_heal_tool_calls = auto_heal,
            )
        )
        contents = "".join(e.get("text", "") for e in events if e.get("type") == "content")
        return calls, contents

    calls_off, contents_off = _run(False)
    assert calls_off == [], calls_off
    assert "web_search" in contents_off, contents_off

    calls_on, contents_on = _run(True)
    assert calls_on == [], calls_on
    assert "web_search" not in contents_on, contents_on


def test_gguf_valid_tool_calls_respect_max_tool_iterations(monkeypatch):
    """Re-prompt slots must not extend the tool budget: stop after ``max_tool_iterations`` executed rounds."""
    # More tool-call streams than the budget: if re-prompt slots leaked into the budget (the bug) the
    # loop would run 2+3=5 rounds; honouring it stops after 2, then a tool-less final-answer pass.
    streams = [
        _structured_tool_call("web_search", {"query": f"q{i}"}, f"call_{i}") for i in range(6)
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: (calls.append((name, arguments)) or "result"),
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search repeatedly"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    # Exactly two executed tool rounds, then one final-answer pass.
    assert len(calls) == 2, calls
    assert len(payloads) == 3, len(payloads)
    # The final pass is the budget-exhausted nudge and carries no tools.
    assert _tool_names(payloads[2]) == [], _tool_names(payloads[2])
    assert any(
        m.get("role") == "user" and "used all available tool calls" in m.get("content", "")
        for m in payloads[2]["messages"]
    ), payloads[2]["messages"]


# ── Live tool-call argument streaming (tool_args events) ─────────────────────


def _python_tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": "Run python code.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]


def test_structured_tool_args_stream_to_provisional_card(monkeypatch):
    """A large structured tool call must stream its arguments as tool_args events
    to the provisional card (backlog that triggered the card, then each
    fragment), while the executed call and the model's view stay exactly what the
    accumulator built."""

    code = "print('x')\n" + ("# pad\n" * 80)
    args_json = json.dumps({"code": code})
    call_id = "call_live_args"
    split = _PROVISIONAL_ARGS_MIN_CHARS + 16
    frag1, frag2, frag3 = (
        args_json[:split],
        args_json[split : split + 40],
        args_json[split + 40 :],
    )

    def _tc_delta(fragment: str, with_header: bool) -> str:
        entry: dict = {"index": 0, "function": {"arguments": fragment}}
        if with_header:
            entry.update({"id": call_id, "type": "function"})
            entry["function"]["name"] = "python"
        return _sse({"tool_calls": [entry]})

    first_stream = [
        _tc_delta(frag1, with_header = True),
        _tc_delta(frag2, with_header = False),
        _tc_delta(frag3, with_header = False),
        _done(),
    ]
    second_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, second_stream], payloads)

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run it"}],
            tools = _python_tool_schema(),
            max_tool_iterations = 1,
        )
    )

    starts = [e for e in events if e.get("type") == "tool_start"]
    assert starts and starts[0]["tool_call_id"] == call_id

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "no tool_args events were streamed"
    assert all(e["tool_call_id"] == call_id for e in args_events)
    # First event is the backlog, the rest raw fragments; together the args JSON.
    assert args_events[0]["text"] == frag1
    assert "".join(e["text"] for e in args_events) == args_json

    # The streamed display path must not perturb execution or the model view.
    assert executed == [("python", {"code": code})]
    assistant_messages = [m for m in payloads[1]["messages"] if m.get("role") == "assistant"]
    tc = assistant_messages[-1]["tool_calls"][0]
    assert tc["id"] == call_id
    # Controller re-serializes args (normalized JSON); parsed payload unchanged.
    assert json.loads(tc["function"]["arguments"]) == {"code": code}


def test_text_tool_call_streams_args_and_reconciles_card(monkeypatch):
    """A TEXT (XML) tool call must stream its raw call text as tool_args under the
    id the stream-end parser assigns ("call_0"), so the provisional card and the
    final tool_start reconcile."""

    code = "print('hello')\n" + ("# filler\n" * 60)
    call_json = json.dumps({"name": "python", "arguments": {"code": code}})
    call_text = f"<tool_call>{call_json}</tool_call>"
    chunks = [call_text[i : i + 48] for i in range(0, len(call_text), 48)]
    first_stream = [_sse({"content": chunk}) for chunk in chunks] + [_done()]
    second_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, second_stream], payloads)

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run it"}],
            tools = _python_tool_schema(),
            max_tool_iterations = 1,
        )
    )

    starts = [e for e in events if e.get("type") == "tool_start"]
    assert starts, "no tool_start emitted"
    # Provisional card first (parser's first-call id), then the reconciling start.
    assert starts[0]["tool_call_id"] == "call_0"
    assert starts[0]["arguments"] == {}
    assert starts[-1]["tool_call_id"] == "call_0"

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "no tool_args events for the text call"
    assert all(e["tool_call_id"] == "call_0" for e in args_events)
    streamed = "".join(e["text"] for e in args_events)
    # Streamed text is the drained call (display only); it must never leak into
    # content events.
    assert '"name": "python"' in streamed
    assert executed == [("python", {"code": code})]
    content_events = [e for e in events if e.get("type") == "content"]
    assert not any("<tool_call>" in e["text"] for e in content_events)


def test_ordinary_json_answer_streams_no_tool_args(monkeypatch):
    """A large ordinary JSON answer (no enabled tool name) must not spawn a
    provisional card or tool_args events; it stays a normal content answer."""

    answer = json.dumps({"result": "fine", "data": ["x" * 40] * 12, "note": "not a tool call"})
    chunks = [answer[i : i + 64] for i in range(0, len(answer), 64)]
    stream = [_sse({"content": chunk}) for chunk in chunks] + [_done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "give me json"}],
            tools = _python_tool_schema(),
            max_tool_iterations = 1,
        )
    )

    assert not [e for e in events if e.get("type") == "tool_args"]
    assert not [e for e in events if e.get("type") == "tool_start"]
    content_events = [e for e in events if e.get("type") == "content"]
    assert content_events and answer in content_events[-1]["text"]


def test_provisional_text_card_closed_when_parse_fails(monkeypatch):
    """A >=256-char enabled-name text sniff opens a provisional card; if the
    drained text then fails to parse (auto-heal off, truncated call), the
    DRAINING false-positive path must close the card with a tool_end instead of
    leaving it spinning forever."""

    # Truncated mid-arguments and never closed: unparseable without healing.
    call_text = '<tool_call>{"name": "python", "arguments": {"code": "' + "x" * (
        _PROVISIONAL_ARGS_MIN_CHARS + 64
    )
    chunks = [call_text[i : i + 48] for i in range(0, len(call_text), 48)]
    stream = [_sse({"content": chunk}) for chunk in chunks] + [_done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run it"}],
            tools = _python_tool_schema(),
            max_tool_iterations = 1,
            auto_heal_tool_calls = False,
        )
    )

    starts = [e for e in events if e.get("type") == "tool_start"]
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert starts and starts[0]["tool_call_id"] == "call_0"
    assert executed == []  # nothing parsed, nothing ran
    assert ends, "provisional card left dangling (no tool_end)"
    assert ends[-1]["tool_call_id"] == "call_0"
