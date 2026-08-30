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
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import (
    _MAX_REPROMPTS,
    _PROVISIONAL_ARGS_MIN_CHARS,
    GgufLoadIntent,
    LlamaCppBackend,
)
from core.inference.tool_call_parser import NUDGE_TOOL_CALLS_STATUS
from state import tool_approvals
from state.tool_approvals import TOOL_REJECTED_MESSAGE, resolve_tool_decision


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _progress(*, processed: int, cached: int, time_ms: int) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}],
                "prompt_progress": {
                    "total": processed,
                    "processed": processed,
                    "cache": cached,
                    "time_ms": time_ms,
                },
            }
        )
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": reason,
                    }
                ]
            }
        )
        + "\n"
    )


def _make_backend(
    monkeypatch,
    streams: list[object],
    payloads: list[dict],
    urls: list[str] | None = None,
):
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
        if urls is not None:
            urls.append(_url)
        stream = streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend


def test_plain_stream_reports_request_scoped_live_prompt_and_generation_timings(monkeypatch):
    stream = [
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}],
                "prompt_progress": {
                    "total": 1000,
                    "processed": 1000,
                    "cache": 100,
                    "time_ms": 100,
                },
                "timings": {"prompt_ms": 100, "prompt_per_second": 9000},
            }
        )
        + "\n",
        "data: "
        + json.dumps(
            {
                "choices": [{"index": 0, "delta": {"content": "OK"}}],
                "timings": {
                    "prompt_n": 900,
                    "prompt_ms": 100,
                    "prompt_per_second": 9000,
                    "predicted_n": 4,
                    "predicted_ms": 20,
                    "predicted_per_second": 200,
                },
            }
        )
        + "\n",
        _done(),
    ]
    payloads: list[dict] = []
    samples: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "benchmark"}],
            perf_callback = samples.append,
        )
    )

    assert payloads[0]["return_progress"] is True
    assert payloads[0]["timings_per_token"] is True
    assert samples[0]["prompt_n"] == 900
    assert samples[0]["prompt_per_second"] == 9000
    assert all("prompt_ms" not in sample for sample in samples)
    assert samples[-1]["predicted_per_second"] == 200


def test_tool_stream_reports_progress_without_leaking_a_content_event(monkeypatch):
    stream = [
        _progress(processed = 512, cached = 0, time_ms = 64),
        _sse({"content": "done"}),
        _done(),
    ]
    payloads: list[dict] = []
    samples: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "benchmark"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
            perf_callback = samples.append,
        )
    )

    assert payloads[0]["return_progress"] is True
    assert payloads[0]["timings_per_token"] is True
    assert samples[0]["prompt_per_second"] == 8000
    assert [event["text"] for event in events if event["type"] == "content"] == ["done"]


def _patch_successful_respawn(
    monkeypatch,
    backend,
    port: int | None = None,
) -> list[bool]:
    calls: list[bool] = []

    def fake_respawn():
        calls.append(True)
        if port is not None:
            backend._port = port
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
    return calls


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


def test_forced_web_search_tool_choice_is_sent_until_a_tool_runs(monkeypatch):
    """#9730: a forced web_search must reach llama-server on the first turn.

    After the call executes, the follow-up is auto so the model can answer.
    """
    first_stream = _structured_tool_call(
        "web_search", {"query": "current Linux kernel version"}, "call_search"
    )
    second_stream = [_sse({"content": "The current version of the Linux kernel is 6.10."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, second_stream], payloads)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Linux kernel 6.10",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Search the web for the current version of the Linux kernel, "
                        "then answer in one sentence."
                    ),
                }
            ],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            tool_choice = {"type": "function", "function": {"name": "web_search"}},
            max_tool_iterations = 5,
            permission_mode = "off",
        )
    )

    assert payloads[0]["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == ["web_search"]
    assert payloads[1]["tool_choice"] == "auto"
    assert any(
        event.get("type") == "tool_start" and event.get("tool_name") == "web_search"
        for event in events
    )


def test_forced_tool_choice_must_exist_in_the_catalog(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [], payloads)

    with pytest.raises(ValueError, match = "Forced tool 'python' is not enabled"):
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "run python"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                tool_choice = {"type": "function", "function": {"name": "python"}},
                max_tool_iterations = 1,
            )
        )

    assert payloads == []


def test_forced_tool_choice_retries_after_other_structured_calls(monkeypatch):
    wrong_code = "print('wrong tool')\n" * 20
    streams = [
        _structured_tool_call("python", {"code": wrong_code}, "call_python"),
        _structured_tool_call("web_search", {"query": "kernel version"}, "call_search"),
        [_sse({"content": "The search completed."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Linux kernel result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search the web"}],
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "python"}},
            ],
            tool_choice = {"type": "function", "function": {"name": "web_search"}},
            max_tool_iterations = 2,
        )
    )

    assert payloads[0]["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == ["web_search"]
    assert payloads[1]["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in payloads[1]["tools"]] == ["web_search"]
    assert payloads[2]["tool_choice"] == "auto"
    assert calls == [("web_search", {"query": "kernel version"})]
    assert [event.get("tool_name") for event in events if event.get("type") == "tool_start"] == [
        "web_search"
    ]


def test_none_tool_choice_never_executes_model_tool_calls(monkeypatch):
    stream = [
        *_structured_tool_call("python", {"code": "print(1)"}, "call_python")[:-1],
        _sse({"content": "I will answer without tools."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    def fail_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    def fail_autoinject(*_args, **_kwargs):
        raise AssertionError("tool_choice=none must not autoinject retrieval")

    monkeypatch.setattr("core.inference.tools.execute_tool", fail_execute_tool)
    monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fail_autoinject)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "answer directly"}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice = "none",
            max_tool_iterations = 1,
            rag_scope = {"thread_id": "t1", "autoinject": True},
        )
    )

    assert len(payloads) == 1
    assert "tools" not in payloads[0]
    assert "tool_choice" not in payloads[0]
    assert not [event for event in events if event.get("type") in {"tool_start", "tool_end"}]
    assert any(event.get("text") == "I will answer without tools." for event in events)


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
    # same text on the visible channel. The final cumulative snapshot stays
    # append-only so route suffix extraction cannot drop that fallback.
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
    assert content_texts[-1] == (
        "<think>The capital of France is Paris.</think>The capital of France is Paris."
    )


def _assert_reasoning_only_raw_consumer_gets_one_balanced_think_block(monkeypatch, with_tools):
    stream = [
        _sse({"reasoning_content": "The capital of France is Paris."}),
        _done(),
    ]
    backend = _make_backend(monkeypatch, [stream], [])

    if with_tools:
        items = list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "capital of France?"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                max_tool_iterations = 1,
                promote_reasoning_only = False,
            )
        )
        cumulatives = [item["text"] for item in items if item.get("type") == "content"]
    else:
        items = list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "capital of France?"}],
                promote_reasoning_only = False,
            )
        )
        cumulatives = [item for item in items if isinstance(item, str)]

    assert cumulatives[-1] == "<think>The capital of France is Paris.</think>"
    assert all(
        current.startswith(previous) for previous, current in zip([""] + cumulatives, cumulatives)
    )


def test_reasoning_only_raw_consumer_without_tools_gets_one_balanced_think_block(monkeypatch):
    _assert_reasoning_only_raw_consumer_gets_one_balanced_think_block(monkeypatch, False)


def test_reasoning_only_raw_consumer_with_tools_gets_one_balanced_think_block(monkeypatch):
    _assert_reasoning_only_raw_consumer_gets_one_balanced_think_block(monkeypatch, True)


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
    # append a balanced close plus visible fallback; the route's suffix-diff +
    # extractor must therefore produce the same split for both.
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
    visible, reasoning = tool_out
    assert visible == "The capital of France is Paris."
    assert reasoning == "The capital of France is Paris."


def test_length_truncated_reasoning_stays_append_only_without_visible_promotion(monkeypatch):
    stream = [
        _sse({"reasoning_content": "The proof begins by assuming finitely many primes."}),
        _finish("length"),
        _done(),
    ]
    backend = _make_backend(monkeypatch, [stream], [])

    items = list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Prove infinitely many primes"}],
            max_tokens = 16,
        )
    )
    cumulatives = [item for item in items if isinstance(item, str)]

    assert all(
        current.startswith(previous) for previous, current in zip([""] + cumulatives, cumulatives)
    )
    assert cumulatives[-1] == ("<think>The proof begins by assuming finitely many primes.</think>")
    visible, reasoning = _replay_route_reasoning_extractor(cumulatives)
    assert visible == ""
    assert reasoning == "The proof begins by assuming finitely many primes."
    assert items[-1]["finish_reason"] == "length"


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


def test_structured_tool_call_turn_replays_pre_tool_reasoning_in_next_payload(monkeypatch):
    """llama-server sends reasoning in delta.reasoning_content, so content
    alone drops it, meaning history must carry it or iteration 2 can't see it."""
    tool_stream = [
        _sse({"reasoning_content": "I should search "}),
        _sse({"reasoning_content": "for the weather."}),
        _sse({"content": "Let me check that.\n\n"}),
    ] + _structured_tool_call("web_search", {"query": "weather"}, "call_r")
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert len(payloads) == 2
    first_assistant = next(
        m for m in payloads[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert first_assistant["content"] == "Let me check that.\n\n"
    assert first_assistant["reasoning_content"] == "I should search for the weather."


def test_mixed_execute_and_noop_batch_keeps_structured_reasoning(monkeypatch):
    calls_delta = {
        "tool_calls": [
            {
                "index": index,
                "id": f"call_mixed_{index}",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "weather"}),
                },
            }
            for index in range(2)
        ]
    }
    tool_stream = [
        _sse({"reasoning_content": "Search <|channel>thought safely<channel|>."}),
        _sse({"content": "Checking."}),
        _sse(calls_delta),
        _done(),
    ]
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "sunny"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "weather"})]
    messages = payloads[1]["messages"]
    assistant_index, assistant = next(
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    tool_index = next(
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    )
    no_op_index, result_with_feedback = next(
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "tool" and "identical call" in message.get("content", "")
    )
    assert assistant_index < tool_index == no_op_index
    assert assistant["content"] == "Checking."
    assert assistant["reasoning_content"] == "Search < |channel>thought safely< channel|>."
    assert len(assistant["tool_calls"]) == 1
    assert result_with_feedback["tool_call_id"] == "call_mixed_0"
    assert not any(message.get("role") == "user" for message in messages[1:])


def test_textual_tool_call_turn_replays_reasoning_only_trace_in_next_payload(monkeypatch):
    """A reasoning only tool turn has empty content, so without the field the
    trace left the conversation entirely."""
    tool_stream = [
        _sse({"reasoning_content": "I must search before answering."}),
        _sse(
            {
                "content": '<tool_call>{"name":"web_search","arguments":{"query":"weather"}}</tool_call>'
            }
        ),
        _done(),
    ]
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert len(payloads) == 2
    first_assistant = next(
        m for m in payloads[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert first_assistant["content"] == ""
    assert first_assistant["reasoning_content"] == "I must search before answering."


def test_tool_call_turn_without_reasoning_adds_no_reasoning_content(monkeypatch):
    """Non-reasoning models keep their history unchanged."""
    tool_stream = _structured_tool_call("web_search", {"query": "weather"}, "call_plain")
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert len(payloads) == 2
    first_assistant = next(
        m for m in payloads[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert "reasoning_content" not in first_assistant


def test_tool_call_turn_with_blank_reasoning_adds_no_reasoning_content(monkeypatch):
    """Whitespace split from a closed thinking prefill is not a trace."""
    tool_stream = [_sse({"reasoning_content": "\n\n"})] + _structured_tool_call(
        "web_search", {"query": "weather"}, "call_blank"
    )
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    first_assistant = next(
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert "reasoning_content" not in first_assistant


def test_blank_reasoning_noop_turn_adds_no_empty_assistant_message(monkeypatch):
    """A blank trace on a suppressed call must not open an empty model turn."""
    tool_stream = [
        _sse({"reasoning_content": "\n\n"}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_blank_noop",
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
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run python"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert not [
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "assistant"
        and not message.get("tool_calls")
        and not str(message.get("content") or "").strip()
    ]
    assert any(
        message.get("role") == "user" and "not enabled" in message.get("content", "")
        for message in payloads[1]["messages"]
    )


def test_noop_reasoning_continuation_separates_partial_from_inlined_trace(monkeypatch):
    """A suppressed call must not weld inlined reasoning onto a resumed partial."""

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    for partial in ("PARTIAL_TEXT", "PARTIAL_TEXT\n"):
        tool_stream = [_sse({"reasoning_content": "TRACE_ABC"})] + _structured_tool_call(
            "python", {"code": "print(1)"}, "call_continued_noop"
        )
        final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

        list(
            backend.generate_chat_completion_with_tools(
                messages = [
                    {"role": "user", "content": "continue"},
                    {"role": "assistant", "content": partial},
                ],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                continue_final_message = True,
                max_tool_iterations = 1,
            )
        )

        continued = next(
            message for message in payloads[1]["messages"] if message.get("role") == "assistant"
        )
        assert continued == {"role": "assistant", "content": "PARTIAL_TEXT\nTRACE_ABC"}


def test_noop_reasoning_without_continuation_adds_clean_assistant_turn(monkeypatch):
    """A trailing assistant turn is not merged unless continuation is requested."""
    tool_stream = [_sse({"reasoning_content": "TRACE_ABC"})] + _structured_tool_call(
        "python", {"code": "print(1)"}, "call_separate_noop"
    )
    final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    def fake_execute_tool(name, arguments, **_kwargs):
        raise AssertionError(f"unexpected tool execution: {name} {arguments}")

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "new turn"},
                {"role": "assistant", "content": "EARLIER_ANSWER"},
            ],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            continue_final_message = False,
            max_tool_iterations = 1,
        )
    )

    assistant_messages = [
        message for message in payloads[1]["messages"] if message.get("role") == "assistant"
    ]
    assert assistant_messages == [
        {"role": "assistant", "content": "EARLIER_ANSWER"},
        {"role": "assistant", "content": "TRACE_ABC"},
    ]


def test_noop_feedback_is_not_folded_into_another_tool_s_result(monkeypatch):
    """Feedback about tool A must never ride tool B's result.

    Templates label the whole block with the result's OWN tool name (gemma-4.jinja
    resolves tool_call_id -> name and wraps the body), so a note about a suppressed
    ``python`` call folded into a ``web_search`` result reads as web_search's output.
    Only a same-tool result may carry it; otherwise the user turn is the lesser loss.
    """
    tool_stream = [
        _sse({"reasoning_content": "Plan the batch."}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_mixed_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for index, (name, args) in enumerate(
                        [
                            ("web_search", {"query": "a"}),
                            ("python", {"code": "print(1)"}),  # disabled -> internal no-op
                            ("web_search", {"query": "b"}),
                        ]
                    )
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: f"RESULT_OF_{name}",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "q"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    for message in messages:
        if message.get("role") == "tool":
            assert "not executed" not in message["content"], message
            assert message["content"] == "RESULT_OF_web_search"
    assert [
        message
        for message in messages
        if message.get("role") == "user" and "not executed" in message.get("content", "")
    ]


def test_noop_feedback_for_multiple_tools_is_not_folded_by_partial_name_match(monkeypatch):
    """Every no-op tool must match the fold target, not merely one of them."""
    tool_stream = [
        _sse({"reasoning_content": "Plan the batch."}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_multi_noop_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for index, (name, args) in enumerate(
                        [
                            ("web_search", {"query": "a"}),
                            ("web_search", {"query": "a"}),  # duplicate -> internal no-op
                            ("python", {"code": "print(1)"}),  # disabled -> internal no-op
                            ("web_search", {"query": "b"}),
                        ]
                    )
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: f"RESULT_OF_{name}",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "q"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    assert all(
        message["content"] == "RESULT_OF_web_search"
        for message in messages
        if message.get("role") == "tool"
    )
    feedback = [message for message in messages if message.get("role") == "user"][1:]
    assert len(feedback) == 1
    assert "identical call" in feedback[0]["content"]
    assert "not enabled" in feedback[0]["content"]


def test_same_tool_noop_feedback_still_rides_its_own_result(monkeypatch):
    """The fold is kept where attribution is unambiguous: a duplicate names the
    same tool as the result it lands on, so no newer user turn is opened."""
    tool_stream = [
        _sse({"reasoning_content": "Plan the batch."}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_dup_{index}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(args),
                        },
                    }
                    for index, args in enumerate([{"query": "a"}, {"query": "a"}, {"query": "b"}])
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "q"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    assert not [m for m in messages[1:] if m.get("role") == "user"]
    assert any(m.get("role") == "tool" and "not executed" in m["content"] for m in messages)


def test_tool_loop_does_not_mutate_the_caller_s_messages(monkeypatch):
    """The caller's own message dicts stay untouched across a fold.

    ``conversation`` copies the list, not the dicts. Every fold target is a result this
    loop just built, so nothing caller-owned is reachable today; pinning it matters
    because /v1/messages passes a client's own history through, and a fold that moved
    to a different target would leave the hidden instruction in that client's list.
    """
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "earlier",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "tool_call_id": "earlier", "content": "cloudy"},
    ]
    before = copy.deepcopy(messages)

    tool_stream = [
        _sse({"reasoning_content": "One search is enough."}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call_own_{index}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "weather"}),
                        },
                    }
                    for index in range(2)  # the second is a duplicate -> internal no-op
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "It is sunny."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = messages,
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    assert messages == before


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
    _patch_monotonic(monkeypatch, [200.0, 201.0, 203.0, 300.0, 400.0, 405.0, 410.0])

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
    # only the executed calls (no orphan for the duplicate), and the two tool results
    # follow contiguously. Hidden feedback is attached to the final result so it does
    # not create a newer user turn that can suppress this assistant's reasoning.
    conv = payloads[1]["messages"]
    asst = next(m for m in conv if m["role"] == "assistant" and m.get("tool_calls"))
    assert [tc.get("id") for tc in asst["tool_calls"]] == ["call_a1", "call_b"]
    after = conv[conv.index(asst) + 1 :]
    assert [m["role"] for m in after[:2]] == ["tool", "tool"]
    assert [m.get("tool_call_id") for m in after[:2]] == ["call_a1", "call_b"]
    assert len(after) == 2
    assert "One earlier request to call tool 'web_search'" in after[1]["content"]
    assert "previous tool request" not in after[1]["content"].lower()


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
    render_feedback = [
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "tool"
        and "Do not call render_html again" in message.get("content", "")
    ]
    assert len(render_feedback) == 1


def test_disabled_tool_call_is_internal_noop(monkeypatch):
    disabled_python = [
        _sse(
            {
                "reasoning_content": (
                    "Python needs <tool_call>{...}</tool_call> and "
                    "<|channel>thought x<channel|> before <|turn>user"
                )
            }
        ),
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
    reasoning_turn_index, reasoning_turn = next(
        (index, message)
        for index, message in enumerate(payloads[1]["messages"])
        if message.get("role") == "assistant"
        and message.get("content")
        == (
            "Python needs < tool_call>{...}< /tool_call> and "
            "< |channel>thought x< channel|> before < |turn>user"
        )
    )
    nudge_index = next(
        index
        for index, message in enumerate(payloads[1]["messages"])
        if message.get("role") == "user" and "not enabled" in message.get("content", "")
    )
    assert reasoning_turn_index < nudge_index
    assert "reasoning_content" not in reasoning_turn
    assert "tool_calls" not in reasoning_turn


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
            nudge_tool_calls = True,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    # Each retry restates the last, so the loop gives up: initial + 2 re-prompts.
    assert len(payloads) == 3 < _MAX_REPROMPTS + 1


def test_post_tool_stall_still_nudged_after_a_pre_tool_reprompt(monkeypatch):
    """The post-tool nudge has its own budget, so an earlier stall can't spend it."""

    streams = [
        [_sse({"content": "I will search the web now."}), _done()],
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_first",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "red square"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": "Let me summarize the results."}), _done()],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Search results: red is #f00."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    assert len(payloads) == 4
    assert len(calls) == 1
    nudges = [
        message
        for message in payloads[-1]["messages"]
        if message.get("role") == "user" and "call web_search now" in message.get("content", "")
    ]
    assert len(nudges) == 2
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == "Final answer: the square is red."


def test_post_tool_reprompt_budget_is_one(monkeypatch):
    """The post-tool nudge fires once; a second stall is surrendered as the answer."""

    streams = [
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_first",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "red square"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": "Let me summarize the results."}), _done()],
        [_sse({"content": "Now I will check the sources."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    assert len(payloads) == 3


def test_repeat_guard_resets_after_a_tool_runs(monkeypatch):
    """A tool execution opens a new phase, so the same intent text is nudged again.

    Without the reset the pre-tool stall text still sits in the repeat tracker and
    the identical post-tool stall is surrendered as the visible final answer.
    """

    stall = "I will search the web now."
    streams = [
        [_sse({"content": stall}), _done()],
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_first",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "red square"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": stall}), _done()],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    assert len(payloads) == 4
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == "Final answer: the square is red."


def test_restatement_keeps_deletions_that_change_the_answer():
    """A dropped word can invert the meaning, so a subset is not a restatement."""

    from core.inference.tool_call_parser import is_reprompt_restatement
    from core.inference.llama_cpp import _should_suppress_forced_no_tool_output as suppress

    previous = "Now I think the feature is not supported in version 1."
    corrected = "Now I think the feature is supported in version 1."
    assert not is_reprompt_restatement(corrected, previous)
    assert not suppress(corrected, previous)

    stall = "I'll search for that now."
    assert is_reprompt_restatement(stall, stall)
    assert is_reprompt_restatement("Understood. " + stall, "Understood, " + stall)
    assert not is_reprompt_restatement(stall + " Tokyo.", stall)


def test_forced_turn_suppression_covers_obligation_phrasing():
    from core.inference.llama_cpp import _should_suppress_forced_no_tool_output as suppress
    for stall in (
        "I need to use render_html now",
        "Need to call web_search",
        "I will summarize the results now",
        "I have to run the search first",
        "I should call web_search now",
        "I should use render_html now",
        # Plain modals take a bare infinitive, not the need|have|ought "to" group.
        "I must call web_search now",
        "I must use render_html now",
        "I must run the search first",
        # Subjectless plans open a new sentence just as often as a new line.
        "Okay. Need to call web_search now.",
        "Understood. Going to search now.",
        # Subjectless modals, not just subjectless semi-modals.
        "Must call web_search now.",
        "Should search the web now.",
        # A missing answer is not a final answer: the plan behind it is still a stall.
        "I should call web_search because the answer is not in the provided context",
        "I must run the search since the answer is unknown so far",
        # A pivot with nothing behind it answers nothing.
        "I should call web_search, though.",
        "I need to run the search, but",
        # A purpose clause is part of the plan, not a summary of results.
        "I need to call web_search to summarize the results",
    ):
        assert suppress(stall), f"leaked {stall!r}"

    for answer in (
        "You need to install the package first.",
        "The square is red.",
        "Here is the summary of what I found.",
        "Run `pip install unsloth` to get started.",
        "I should mention that the square is red.",
        # Obligation phrasing mid-sentence is prose that happens to name a tool.
        "The API I should invoke is foo() because it supports streaming.",
        "The tool I need to use is documented here.",
        # "invoke"/"query" read as technical prose far more often than as a stall.
        "I should invoke foo() because it supports streaming.",
        "I should query the cache first for a faster path.",
        "You should call your bank about the charge.",
        # Second person is the user's obligation, not the model's plan.
        "You must call your bank about the charge.",
        "I must admit the square is red.",
        # A plan that pivots to an answer must ship the answer with it.
        "I should call web_search, but the answer is Tokyo.",
        "I need to call web_search. The answer is Tokyo.",
        "I should call web_search to confirm, but Tokyo is the capital of Japan.",
        "I must run the search, however the result is already known: 42.",
    ):
        assert not suppress(answer), f"dropped {answer!r}"


def test_forced_turn_intent_lead_in_needs_a_restatement_to_be_dropped():
    """A bare intent match is a stall only when the retry restates the nudge.

    ``INTENT_SIGNAL`` fires on lead-ins that introduce a real answer ("Now I
    have the results. ..."), so matching it alone would discard the answer.
    """
    from core.inference.llama_cpp import _should_suppress_forced_no_tool_output as suppress

    stall = "I will summarize the results now"
    answer = "Now I have the search results. The capital of Japan is Tokyo."

    # Restating the nudged text is still a stall.
    assert suppress(stall, stall)
    assert suppress("Understood. " + stall, "Understood, " + stall)
    # Progress past the nudged text keeps the answer, lead-in and all.
    assert not suppress(answer, stall)
    assert not suppress("Step 3: done. Tokyo is the capital.", stall)
    # Near-repeat is enough to stop nudging, never enough to drop the turn.
    assert not suppress(stall + ": Tokyo.", stall)
    # An obligation plan is a stall on its own, no previous text needed.
    assert suppress("I must call web_search now", answer)


def test_forced_turn_answer_with_an_intent_lead_in_survives_after_a_tool(monkeypatch):
    """The post-tool retry answers behind a lead-in; the answer must still ship.

    The nudge budget is spent, so the reply lands on the suppression branch.
    ``INTENT_SIGNAL`` matches its "Now I ..." opener, and dropping it on that
    alone left the user with the stall and no answer at all.
    """

    answer = "Now I have the results. The capital of Japan is Tokyo."
    streams = [
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_first",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "capital of Japan"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": "Let me summarize what I found."}), _done()],
        [_sse({"content": answer}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: Tokyo.",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What is the capital of Japan?"}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    assert len(payloads) == 3
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == answer


def test_forced_turn_answer_with_an_intent_lead_in_survives_pre_tool(monkeypatch):
    """Same guarantee once the pre-tool nudge budget is spent on distinct stalls."""

    answer = "Now I see the data clearly. Tokyo is the capital."
    streams = [
        [_sse({"content": text}), _done()]
        for text in (
            "I will look that up for you.",
            "Now I have the search results. The capital of Japan is Tokyo.",
            "Now I can confirm it. Japan's capital city is Tokyo.",
            answer,
        )
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
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What is the capital of Japan?"}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    # Initial turn plus the three pre-tool nudges.
    assert len(payloads) == _MAX_REPROMPTS + 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == answer


def test_forced_reprompt_plain_final_answer_is_visible(monkeypatch):
    """A hidden forced re-prompt may fall back to a plain final answer."""

    streams = [
        [_sse({"content": "I will use render_html now."}), _done()],
        [
            _sse({"reasoning_content": "I reconsidered the request."}),
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
            nudge_tool_calls = True,
        )
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == [
        "I will use render_html now.",
        (
            "<think>I reconsidered the request.</think>"
            "No tool is needed. Final answer: use a red square."
        ),
    ]
    summaries = [event for event in events if event.get("type") == "reasoning_summary"]
    assert len(summaries) == 1
    visible_answer_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "content" and "No tool is needed" in event.get("text", "")
    )
    assert visible_answer_index < events.index(summaries[0])
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
            _sse({"reasoning_content": "I should render the requested HTML."}),
            _sse(
                {
                    "content": (
                        '<tool_call>{"name":"render_html","arguments":'
                        '{"code":"<html><body>forced</body></html>",'
                        '"title":"Forced"}}</tool_call>'
                    )
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
            nudge_tool_calls = True,
        )
    )

    assert len(calls) == 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now.", "Final note after tool."]
    assert not any(event.get("type") == "reasoning_summary" for event in events)
    assert len(payloads) == 3


def _status_texts(events: list[dict]) -> list[str]:
    return [event["text"] for event in events if event.get("type") == "status"]


_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _nudge_then_search_streams() -> list[list[str]]:
    """Stall, then a re-prompted turn that finally searches, then the answer."""

    return [
        [_sse({"content": "I will search the web now."}), _done()],
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "red square"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]


def test_plan_without_action_nudge_is_announced_on_the_status_channel(monkeypatch):
    """The re-prompted turn is hidden, so without a badge the UI looks frozen."""

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, _nudge_then_search_streams(), payloads)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What colour is the square?"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    statuses = _status_texts(events)
    assert NUDGE_TOOL_CALLS_STATUS in statuses
    index = statuses.index(NUDGE_TOOL_CALLS_STATUS)
    # Blank first: the route resets its text cursor only on an empty status.
    # index > 0 matters: at 0, statuses[-1] wraps to the terminal clear.
    assert index > 0 and statuses[index - 1] == ""
    assert statuses[index + 1].startswith("Searching:")
    assert statuses[-1] == ""


def test_plan_without_action_nudge_status_clears_when_the_retry_just_answers(monkeypatch):
    streams = [
        [_sse({"content": "I will search the web now."}), _done()],
        [_sse({"content": "No search needed. Final answer: the square is red."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What colour is the square?"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    statuses = _status_texts(events)
    assert NUDGE_TOOL_CALLS_STATUS in statuses
    assert statuses[-1] == ""


def test_direct_answer_never_shows_the_nudge_status(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": "The square is red."}), _done()]],
        payloads,
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What colour is the square?"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
        )
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)


def test_clarification_request_is_not_nudged(monkeypatch):
    """#8907: the model asked what the user wants, so there is nothing to act on.

    The turn signs off with "I'll dig in", which ``INTENT_SIGNAL`` used to read as a
    plan. Nudging it regenerated the turn and showed two near-identical questions.
    """

    clarification = (
        '"balls" is pretty broad, so what would you like to know or do?\n\n'
        "- Sports: rules of a game\n"
        "- Physics: projectile motion, volume of a sphere\n\n"
        "Let me know what you're after and I'll dig in."
    )
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": clarification}), _done()]],
        payloads,
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Balls"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
        )
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    # one payload: a second would be the wasted re-prompted generation.
    assert len(payloads) == 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts and content_texts[-1] == clarification


def test_nudge_status_absent_when_nudging_is_disabled(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, _nudge_then_search_streams(), payloads)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What colour is the square?"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
            nudge_tool_calls = False,
        )
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    assert len(payloads) == 1


def test_nudge_is_off_when_the_request_flag_is_omitted(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, _nudge_then_search_streams(), payloads)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "What colour is the square?"}],
            tools = [_WEB_SEARCH_TOOL],
            max_tool_iterations = 2,
        )
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    assert len(payloads) == 1


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
            # Unset defaults to "auto", which would not prompt this safe print(1).
            permission_mode = "ask",
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
        # Unset defaults to "auto", which would not prompt this safe print(1).
        permission_mode = "ask",
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
            # "ask" gates every call so autoinject waits; unset defaults to
            # "auto", where this safe retrieval never gates.
            permission_mode = "ask",
            session_id = "sess",
            rag_scope = {"thread_id": "t1"},
        )
    )

    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)


def test_rag_autoinject_counts_as_a_prior_tool_execution(monkeypatch):
    """Autoinjected retrieval runs before the controller, so history stays empty.

    Without counting it the turn reads as pre-tool and gets the full re-prompt
    budget, repeating the expensive retrieval the post-tool cap exists to stop.
    """

    stall = "I will summarize the retrieved passages now."
    streams = [
        [_sse({"content": stall}), _done()],
        [_sse({"content": "Still working on the summary."}), _done()],
        [_sse({"content": "Final answer: the passages describe Tokyo."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)

    monkeypatch.setattr(
        "core.inference.tools.build_rag_autoinject",
        lambda *_a, **_k: {
            "events": [],
            "messages": [{"role": "user", "content": "Retrieved passage: Tokyo."}],
        },
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "summarize the docs"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            max_tool_iterations = 2,
            nudge_tool_calls = True,
            rag_scope = {"thread_id": "t1"},
        )
    )

    # Initial turn plus one retry; read as pre-tool it would spend the full budget.
    assert len(payloads) == 2, payloads
    nudges = [
        message
        for message in payloads[-1]["messages"]
        if message.get("role") == "user"
        and "call search_knowledge_base now" in message.get("content", "")
    ]
    assert len(nudges) == 1, nudges
    assert events


def test_rag_autoinject_only_resolves_matching_forced_choices(monkeypatch):
    monkeypatch.setattr(
        "core.inference.tools.build_rag_autoinject",
        lambda *_a, **_k: {
            "events": [],
            "messages": [{"role": "user", "content": "Retrieved passage: Tokyo."}],
        },
    )
    tools = [
        {"type": "function", "function": {"name": "search_knowledge_base"}},
        {"type": "function", "function": {"name": "web_search"}},
    ]

    def first_payload(tool_choice):
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [[_sse({"content": "The passage describes Tokyo."}), _done()]],
            payloads,
        )
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "summarize the docs"}],
                tools = tools,
                tool_choice = tool_choice,
                max_tool_iterations = 1,
                nudge_tool_calls = False,
                rag_scope = {"thread_id": "t1"},
            )
        )
        return payloads[0]

    matching = first_payload({"type": "function", "function": {"name": "search_knowledge_base"}})
    required = first_payload("required")
    unrelated = first_payload({"type": "function", "function": {"name": "web_search"}})

    assert matching["tool_choice"] == "auto"
    assert required["tool_choice"] == "auto"
    assert unrelated["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in unrelated["tools"]] == ["web_search"]


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
            tool_choice = {"type": "function", "function": {"name": "python"}},
            max_tool_iterations = 2,
            confirm_tool_calls = True,
            # Unset defaults to "auto", which would not prompt this safe print(1).
            permission_mode = "ask",
            session_id = "sess",
        )
    )

    starts = [event for event in events if event.get("type") == "tool_start"]
    ends = [event for event in events if event.get("type") == "tool_end"]
    assert len(starts) == 2
    assert [event["result"] for event in ends] == [TOOL_REJECTED_MESSAGE, "OK"]
    assert calls == [("python", {"code": "print(1)"})]
    assert [payload.get("tool_choice") for payload in payloads] == ["required", "auto", "auto"]


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


def test_gated_python_call_still_streams_its_arguments(monkeypatch):
    """A call awaiting approval still streams its code into the card.

    Suppressing it left the chat completely blank for as long as the model took
    to write the payload, which for a large file is minutes. Nothing runs before
    the decision either way, and the code is what the user is approving.
    """

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    assert len(json.dumps({"code": big_code})) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_gated")
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, final_stream], payloads)

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda name, arguments, **_k: "OK")
    monkeypatch.setattr("core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: "allow")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write code"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            confirm_tool_calls = True,
            permission_mode = "ask",
            max_tool_iterations = 1,
        )
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    assert len(provisional) == 1, tool_starts
    assert provisional[0]["tool_call_id"] == "call_gated"

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "gated call streamed no arguments"
    assert "total += 119" in "".join(e["text"] for e in args_events)

    # The approval prompt still fires, and it comes after the code is on screen.
    gated = [e for e in tool_starts if e.get("awaiting_confirmation")]
    assert gated, tool_starts
    assert events.index(provisional[0]) < events.index(gated[0])


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
    respawn_calls: list[bool] = []

    monkeypatch.setattr(
        backend,
        "_respawn_if_dead",
        lambda: respawn_calls.append(True) or True,
    )
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
    assert respawn_calls == []


def test_connect_error_before_tool_stream_respawns_and_retries(monkeypatch):
    """A dead server before the first tool-loop response is opened is safe to retry."""
    import httpx

    payloads: list[dict] = []
    urls: list[str] = []
    backend = _make_backend(
        monkeypatch,
        [
            httpx.ConnectError("server is down"),
            [_sse({"content": "Recovered."}), _done()],
        ],
        payloads,
        urls,
    )
    respawn_calls = _patch_successful_respawn(monkeypatch, backend, port = 49999)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hello"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    assert respawn_calls == [True]
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert urls == [
        "http://127.0.0.1:48847/v1/chat/completions",
        "http://127.0.0.1:49999/v1/chat/completions",
    ]
    assert any(e.get("type") == "content" and e.get("text") == "Recovered." for e in events)


def test_tool_loop_refits_each_preflight_path_after_context_shrinking_respawn(monkeypatch):
    """Both an ordinary iteration and final synthesis refit without repeating old drops."""
    import httpx
    for max_tool_iterations in (1, 0):
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                httpx.ConnectError("server is down"),
                [_sse({"content": "Recovered."}), _done()],
            ],
            payloads,
        )
        # Sized so each window overflows by roughly one turn-group. Compaction trims a
        # headroom margin BELOW the budget and the turn-picking estimator is coarser than
        # the exact count, so single-group steps would evict the whole history in one pass
        # and leave the second preflight nothing to refit. The property under test is that
        # BOTH preflight paths refit against the window they were given.
        backend._effective_context_length = 2000
        monkeypatch.setattr(
            backend,
            "count_chat_tokens",
            lambda candidate, *_args, **_kwargs: sum(
                len(str(message.get("content", ""))) for message in candidate
            ),
        )

        def fake_respawn():
            backend._effective_context_length = 1000
            return True

        monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [
                    {"role": "user", "content": "u" * 400},
                    {"role": "assistant", "content": "a" * 400},
                    {"role": "user", "content": "u" * 400},
                    {"role": "assistant", "content": "a" * 400},
                    {"role": "user", "content": "final"},
                ],
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = max_tool_iterations,
                context_overflow = "truncate_oldest",
            )
        )

        notices = [event for event in events if event.get("type") == "context_truncated"]
        assert [notice["dropped_messages"] for notice in notices] == [2, 2]
        assert [notice["context_length"] for notice in notices] == [2000, 1000]
        assert [payload["max_tokens"] for payload in payloads] == [2000, 1000]
        assert len(payloads[0]["messages"]) == 3
        assert len(payloads[1]["messages"]) == 1


def test_tool_loop_compacts_text_history_around_latest_audio(monkeypatch):
    """Unpriced media must not disable compaction that the text alone requires."""
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [[_sse({"content": "OK"}), _done()]], payloads)
    backend._effective_context_length = 100
    counted: list[list[dict]] = []

    def count_tokens(candidate, *_args, **_kwargs):
        counted.append(copy.deepcopy(candidate))
        return sum(len(str(message.get("content", ""))) for message in candidate)

    monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)
    audio_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "latest"},
            {
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"},
            },
        ],
    }
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "u" * 40},
                {"role": "assistant", "content": "a" * 40},
                audio_turn,
            ],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tokens = 20,
            max_tool_iterations = 1,
            context_overflow = "truncate_oldest",
        )
    )

    notices = [event for event in events if event.get("type") == "context_truncated"]
    assert counted
    assert all(
        part.get("type") != "input_audio"
        for candidate in counted
        for message in candidate
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
    assert [notice["dropped_messages"] for notice in notices] == [2]
    assert payloads[0]["messages"] == [audio_turn]


def test_tool_loop_secondary_counts_strip_media_but_payloads_keep_it(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            _structured_tool_call("python", {"code": "print('ok')"}, "call_python"),
            [_sse({"content": "Done."}), _done()],
        ],
        payloads,
    )
    counted: list[list[dict]] = []

    def count_tokens(candidate, *_args, **_kwargs):
        counted.append(copy.deepcopy(candidate))
        return 1000

    monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_args, **_kwargs: "ok",
    )
    audio_data = "A" * 100_000
    audio_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "x" * 12_000},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            },
        ],
    }

    list(
        backend.generate_chat_completion_with_tools(
            messages = [audio_turn],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tokens = 512,
            max_tool_iterations = 2,
        )
    )

    assert len(counted) >= 3
    assert all(
        part.get("type") != "input_audio"
        for candidate in counted
        for message in candidate
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
    assert len(payloads) == 2
    assert payloads[0]["messages"][0] == audio_turn
    assert payloads[1]["messages"][0] == audio_turn
    assert payloads[1]["messages"][0]["content"][1]["input_audio"]["data"] == audio_data


@pytest.mark.parametrize("with_tools", [False, True])
def test_media_compaction_recall_recount_uses_the_stripped_view(monkeypatch, with_tools):
    from core.inference import llama_cpp

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [[_sse({"content": "OK"}), _done()]], payloads)
    backend._effective_context_length = 100
    counted: list[list[dict]] = []
    recall_recounts: list[list[dict]] = []

    def count_tokens(candidate, *_args, **_kwargs):
        counted.append(copy.deepcopy(candidate))
        total = 0
        for message in candidate:
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
            else:
                total += sum(len(part.get("text", "")) for part in content)
        return total

    def fake_archive(conversation, _before, **kwargs):
        before_count = len(counted)
        kwargs["count_tokens"](conversation)
        assert len(counted) == before_count + 1
        recall_recounts.append(counted[-1])
        return {
            "conversation": conversation,
            "events": [],
            "counts": {},
            "recalled": False,
            "anchored": [],
        }

    monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(llama_cpp, "_archive_and_recall", fake_archive)
    audio_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "latest"},
            {
                "type": "input_audio",
                "input_audio": {"data": "A" * 100_000, "format": "wav"},
            },
        ],
    }
    kwargs = {
        "messages": [
            {"role": "user", "content": "u" * 40},
            {"role": "assistant", "content": "a" * 40},
            audio_turn,
        ],
        "max_tokens": 20,
        "context_overflow": "truncate_oldest",
        "thread_id": "media-recall",
    }

    if with_tools:
        list(
            backend.generate_chat_completion_with_tools(
                **kwargs,
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = 1,
            )
        )
    else:
        list(backend.generate_chat_completion(**kwargs))

    assert recall_recounts
    assert all(
        part.get("type") != "input_audio"
        for candidate in recall_recounts
        for message in candidate
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
    assert payloads[0]["messages"][-1] == audio_turn


def test_a_respawn_refit_that_misses_its_target_still_archives_and_reports(monkeypatch):
    """A rescued respawn refit archives its evictions and emits metadata."""
    import httpx

    from core.inference import llama_cpp

    # Captured once: the second pass would otherwise wrap the first pass's spy.
    real_archive = llama_cpp._archive_and_recall

    for max_tool_iterations in (1, 0):
        payloads: list[dict] = []
        archived: list[tuple[int, int]] = []
        backend = _make_backend(
            monkeypatch,
            [
                httpx.ConnectError("server is down"),
                [_sse({"content": "Recovered."}), _done()],
            ],
            payloads,
        )
        backend._effective_context_length = 2000
        monkeypatch.setattr(
            backend,
            "count_chat_tokens",
            lambda candidate, *_args, **_kwargs: sum(
                len(str(message.get("content", ""))) for message in candidate
            ),
        )

        def spy(conversation, before, **kwargs):
            archived.append((len(before), len(conversation)))
            return real_archive(conversation, before, **kwargs)

        monkeypatch.setattr(llama_cpp, "_archive_and_recall", spy)

        def fake_respawn():
            backend._effective_context_length = 1000
            return True

        monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [
                    {"role": "user", "content": "u" * 250},
                    {"role": "assistant", "content": "a" * 250},
                    {"role": "user", "content": "u" * 250},
                    {"role": "assistant", "content": "a" * 250},
                    {"role": "user", "content": "f" * 900},
                ],
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = max_tool_iterations,
                context_overflow = "truncate_oldest",
            )
        )

        notices = [event for event in events if event.get("type") == "context_truncated"]
        assert [notice["context_length"] for notice in notices] == [2000, 1000]
        refit = notices[1]
        # The rescued refusal reports what it evicted, boundary included: the client reads
        # that depth to place the compaction notice, so recording nothing would compact
        # silently. Reported is not REPLAYED, which `_sticky_compaction_boundary` still
        # declines for any `fits` false record.
        assert refit["fits"] is False
        assert refit["dropped_messages"] == 2
        assert refit["prompt_tokens_after"] == 900 < refit["prompt_tokens_before"]
        # 4, not the 2 of `dropped_messages`: the boundary counts against the REQUEST's
        # own leading messages, which the next request replays it against, while the drop
        # count is what this one fit removed.
        assert refit["boundary_messages"] == 4
        assert "boundary_anchor" in refit
        assert archived[-1] == (3, 1)
        assert len(payloads[1]["messages"]) == 1


def test_tool_loop_retries_preflight_when_counting_failed_on_the_dead_server(monkeypatch):
    import httpx

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [httpx.ConnectError("server is down"), [_sse({"content": "OK"}), _done()]],
        payloads,
    )
    backend._effective_context_length = 60
    count_calls = 0

    def count_tokens(candidate, *_args, **_kwargs):
        nonlocal count_calls
        count_calls += 1
        if count_calls == 1:
            raise httpx.ConnectError("token counter is down")
        return sum(len(str(message.get("content", ""))) for message in candidate)

    monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(backend, "_respawn_if_dead", lambda: True)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "u" * 25},
                {"role": "assistant", "content": "a" * 25},
                {"role": "user", "content": "final"},
            ],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
            context_overflow = "truncate_oldest",
        )
    )

    notices = [event for event in events if event.get("type") == "context_truncated"]
    assert count_calls >= 2
    assert [notice["dropped_messages"] for notice in notices] == [2]
    assert len(payloads[0]["messages"]) == 3
    assert len(payloads[1]["messages"]) == 1


def test_tool_loop_does_not_send_a_stale_payload_when_respawn_refit_fails(monkeypatch):
    import httpx
    for max_tool_iterations in (1, 0):
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [httpx.ConnectError("server is down")], payloads)
        backend._effective_context_length = 100
        count_calls = 0

        def count_tokens(candidate, *_args, **_kwargs):
            nonlocal count_calls
            count_calls += 1
            if count_calls > 1:
                raise RuntimeError("replacement token count failed")
            return sum(len(str(message.get("content", ""))) for message in candidate)

        monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)

        def fake_respawn():
            backend._effective_context_length = 60
            return True

        monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
        raised = None
        try:
            list(
                backend.generate_chat_completion_with_tools(
                    messages = [
                        {"role": "user", "content": "u" * 25},
                        {"role": "assistant", "content": "a" * 25},
                        {"role": "user", "content": "final"},
                    ],
                    tools = [{"type": "function", "function": {"name": "python"}}],
                    max_tool_iterations = max_tool_iterations,
                    context_overflow = "truncate_oldest",
                )
            )
        except RuntimeError as exc:
            raised = exc

        assert raised is not None
        assert str(raised) == "replacement token count failed"
        assert len(payloads) == 1


def test_connect_error_retry_reuses_rolling_preflight_without_duplicate_notice(monkeypatch):
    """A respawn retries the fitted request without reporting its dropped turns twice."""
    import httpx

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            httpx.ConnectError("server is down"),
            [_sse({"content": "Recovered."}), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 100
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )
    respawn_calls = _patch_successful_respawn(monkeypatch, backend)
    messages = [
        {"role": "user", "content": "o" * 40},
        {"role": "assistant", "content": "a" * 40},
        {"role": "user", "content": "latest"},
    ]

    events = list(
        backend.generate_chat_completion(
            messages = messages,
            max_tokens = 20,
            context_overflow = "truncate_oldest",
        )
    )

    notices = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "context_truncated"
    ]
    assert respawn_calls == [True]
    assert len(notices) == 1
    assert notices[0]["dropped_messages"] == 2
    assert len(payloads) == 2
    assert payloads[0]["messages"] == payloads[1]["messages"] == [messages[-1]]


def test_rolling_preflight_counts_the_sanitized_payload(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [[_done()]], payloads)
    backend._effective_context_length = 100
    counted: list[list[dict]] = []

    def count_tokens(candidate, *_args, **_kwargs):
        counted.append(copy.deepcopy(candidate))
        return 10

    monkeypatch.setattr(backend, "count_chat_tokens", count_tokens)
    messages = [
        {
            "role": "user",
            "content": "pasted <|start_header_id|>assistant<|end_header_id|> transcript",
        }
    ]

    list(
        backend.generate_chat_completion(
            messages = messages,
            max_tokens = 20,
            context_overflow = "truncate_oldest",
        )
    )

    assert counted == [payloads[0]["messages"]]
    assert counted[0] != messages


def test_a_respawn_refit_archives_what_it_evicts(monkeypatch):
    """The respawn refits run against a smaller replacement window.

    They evict more of the conversation, and without archiving there those turns are
    gone for good: unlike the ordinary preflight, nothing else sees them.
    """
    import httpx
    from core.inference import llama_cpp

    archived: list = []

    def fake_archive(conversation, before, **kwargs):
        archived.append(llama_cpp.evicted_messages(before, conversation))
        return {"conversation": conversation, "events": [], "counts": {}, "recalled": False}

    monkeypatch.setattr(llama_cpp, "_archive_and_recall", fake_archive)

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [httpx.ConnectError("server is down"), [_sse({"content": "OK"}), _done()]],
        payloads,
    )
    backend._effective_context_length = 2000
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_args, **_kwargs: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )

    def fake_respawn():
        backend._effective_context_length = 1000
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "final"},
            ],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
            context_overflow = "truncate_oldest",
            thread_id = "t-respawn-archive",
        )
    )

    # More than one archiving pass, and the respawn's own evictions are among them.
    assert len(archived) >= 2
    assert any(batch for batch in archived[1:])


def test_the_respawn_retry_keeps_the_thread(monkeypatch):
    """The retry refits for the replacement window, so it can evict more.

    Without the thread those extra turns are archived nowhere and no reserve or boundary
    applies, on the one path that deliberately compacts a second time.
    """
    import httpx
    from core.inference import llama_cpp

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [httpx.ConnectError("server is down"), [_sse({"content": "OK"}), _done()]],
        payloads,
    )
    backend._effective_context_length = 2000
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_args, **_kwargs: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )

    def fake_respawn():
        backend._effective_context_length = 1000
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
    seen: list = []
    monkeypatch.setattr(
        llama_cpp,
        "_conversation_recall_reserve",
        lambda thread_id: seen.append(thread_id) or 0,
    )

    list(
        backend.generate_chat_completion(
            messages = [
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "final"},
            ],
            context_overflow = "truncate_oldest",
            thread_id = "t-respawn",
        )
    )

    # Both fits, the original and the one the retry runs, know which thread they are on.
    assert len(seen) == 2
    assert seen == ["t-respawn", "t-respawn"]


def test_rolling_respawn_retry_refits_when_the_effective_context_changes(monkeypatch):
    """A smaller replacement window can evict more without repeating the first eviction."""
    import httpx

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [httpx.ConnectError("server is down"), [_sse({"content": "OK"}), _done()]],
        payloads,
    )
    # Sized so each window overflows by roughly one turn-group. Compaction trims a
    # headroom margin BELOW the budget and the turn-picking estimator is coarser than
    # the exact count, so single-group steps would evict the whole history in one pass
    # and leave the second preflight nothing to refit. The property under test is that
    # BOTH preflight paths refit against the window they were given.
    backend._effective_context_length = 2000
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_args, **_kwargs: sum(
            len(str(message.get("content", ""))) for message in candidate
        ),
    )

    def fake_respawn():
        backend._effective_context_length = 1000
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", fake_respawn)
    events = list(
        backend.generate_chat_completion(
            messages = [
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "u" * 400},
                {"role": "assistant", "content": "a" * 400},
                {"role": "user", "content": "final"},
            ],
            context_overflow = "truncate_oldest",
        )
    )

    notices = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "context_truncated"
    ]
    assert [notice["dropped_messages"] for notice in notices] == [2, 2]
    assert [notice["context_length"] for notice in notices] == [2000, 1000]
    assert [payload["max_tokens"] for payload in payloads] == [2000, 1000]
    assert len(payloads[0]["messages"]) == 3
    assert len(payloads[1]["messages"]) == 1


def test_connect_error_after_tool_result_recovers_both_generation_paths(monkeypatch):
    """Recover either post-tool generation path without rerunning the tool."""
    import httpx
    for max_tool_iterations, final_text in (
        (2, "The result is 1."),
        (1, "Final answer."),
    ):
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _structured_tool_call("python", {"code": "print(1)"}, "call_once"),
                httpx.ConnectError("server died between turns"),
                [_sse({"content": final_text}), _done()],
            ],
            payloads,
        )
        respawn_calls = _patch_successful_respawn(monkeypatch, backend)
        tool_calls: list[tuple[str, dict]] = []

        def fake_execute_tool(name, arguments, **_kwargs):
            tool_calls.append((name, arguments))
            return "1"

        monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "print one"}],
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = max_tool_iterations,
            )
        )

        assert respawn_calls == [True]
        assert tool_calls == [("python", {"code": "print(1)"})]
        assert len(payloads) == 3
        assert payloads[1] == payloads[2]
        assert any(e.get("type") == "content" and e.get("text") == final_text for e in events)


def test_connect_error_retry_is_bounded(monkeypatch):
    """A failed retry surfaces the error without another respawn attempt."""
    import httpx

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            httpx.ConnectError("server is down"),
            httpx.ConnectError("replacement is also down"),
        ],
        payloads,
    )
    respawn_calls = _patch_successful_respawn(monkeypatch, backend)

    raised = False
    try:
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "hello"}],
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = 1,
            )
        )
    except RuntimeError as exc:
        raised = True
        assert "Lost connection" in str(exc)

    assert raised
    assert respawn_calls == [True]
    assert len(payloads) == 2


def test_pre_header_transport_errors_also_respawn(monkeypatch):
    """A child that dies during prefill already accepted the socket, so it does
    not surface as ConnectError. Nothing has streamed yet, so replay is safe."""
    import httpx
    for exc in (
        httpx.RemoteProtocolError("server disconnected without sending a response"),
        httpx.ReadError("connection reset by peer"),
        httpx.WriteError("broken pipe"),
    ):
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch, [exc, [_sse({"content": "Recovered."}), _done()]], payloads
        )
        respawn_calls = _patch_successful_respawn(monkeypatch, backend)

        events = list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "hello"}],
                tools = [{"type": "function", "function": {"name": "python"}}],
                max_tool_iterations = 1,
            )
        )

        assert respawn_calls == [True], type(exc).__name__
        assert len(payloads) == 2, type(exc).__name__
        assert any(e.get("type") == "content" and e.get("text") == "Recovered." for e in events)


def test_a_not_yet_reaped_child_does_not_burn_the_retry(monkeypatch):
    """A closing server can beat its own exit status, so poll() briefly reports it
    alive. Without a grace wait _respawn_if_dead hands back the stale _healthy and the
    single retry is spent on the corpse rather than on a replacement."""
    import httpx

    class _Dying:
        # reapable only from the 4th poll, mimicking teardown lagging the socket close
        def __init__(self):
            self.polls = 0
            self.returncode = None

        def poll(self):
            self.polls += 1
            if self.polls > 3:
                self.returncode = -9
                return -9
            return None

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [], payloads)
    backend._process = _Dying()
    backend._healthy = True
    backend._respawn_lock = threading.RLock()
    backend._lock = threading.RLock()
    backend._mtp_runtime_fallback_lock = threading.Lock()
    backend._serial_load_lock = threading.RLock()
    backend._cancel_event = threading.Event()
    backend._unload_epoch = 0
    backend._mtp_runtime_fallback_in_progress = False
    backend._mtp_runtime_fallback_active = False
    backend._last_load_intent = GgufLoadIntent(
        gguf_path = "/m.gguf",
        model_identifier = "m",
    )
    backend._model_identifier = "m"
    dying = backend._process
    loads: list[dict] = []

    @contextlib.contextmanager
    def dead_until_respawned(
        _c,
        _url,
        payload,
        _ce,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        if backend._process is dying:
            raise httpx.ReadError("connection reset while shutting down")
        yield type(
            "FakeResponse",
            (),
            {"status_code": 200, "chunks": [_sse({"content": "Recovered."}), _done()]},
        )()

    def fake_load(intent):
        loads.append(intent)
        backend._process = type("Live", (), {"poll": lambda self: None, "returncode": None})()
        backend._healthy = True
        return True

    monkeypatch.setattr(backend, "_stream_with_retry", dead_until_respawned)
    monkeypatch.setattr(backend, "load_model", fake_load)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hello"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )

    assert len(loads) == 1
    assert any(e.get("type") == "content" and e.get("text") == "Recovered." for e in events)


def test_prefill_timeout_is_not_retried(monkeypatch):
    """A slow-but-alive server must not have its first-token budget spent twice."""
    import httpx
    for exc in (httpx.ReadTimeout("no first token"), httpx.PoolTimeout("pool")):
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [exc], payloads)
        respawn_calls = _patch_successful_respawn(monkeypatch, backend)

        raised = False
        try:
            list(
                backend.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "hello"}],
                    tools = [{"type": "function", "function": {"name": "python"}}],
                    max_tool_iterations = 1,
                )
            )
        except httpx.TimeoutException:
            raised = True

        assert raised, type(exc).__name__
        assert respawn_calls == [], type(exc).__name__
        assert len(payloads) == 1, type(exc).__name__


def test_mtp_crash_recovery_wins_over_respawn(monkeypatch):
    """An MTP crash reloads without MTP, so never respawn the same config on top."""
    import httpx
    for max_tool_iterations in (2, 1):
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [httpx.ConnectError("mtp crash")], payloads)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: True)
        respawn_calls = _patch_successful_respawn(monkeypatch, backend)

        raised = False
        try:
            list(
                backend.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "hello"}],
                    tools = [{"type": "function", "function": {"name": "python"}}],
                    max_tool_iterations = max_tool_iterations,
                )
            )
        except RuntimeError as exc:
            raised = True
            assert "Lost connection" in str(exc)

        assert raised
        assert respawn_calls == []
        assert len(payloads) == 1


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
        lambda n, a, **_k: calls.append((n, a)) or "x",
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
        lambda n, a, **_k: calls.append((n, a)) or "x",
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
        lambda n, a, **_k: calls.append((n, a)) or "x",
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
        lambda n, a, **_k: calls.append((n, a)) or "x",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda n, a, **_k: calls.append((n, a)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "OK",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "OK",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "OK",
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
            lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
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
    # The final pass carries no tool schemas. Its controller feedback stays in
    # the latest tool result rather than opening a newer user turn.
    assert _tool_names(payloads[2]) == [], _tool_names(payloads[2])
    assert any(
        m.get("role") == "tool" and "used all available tool calls" in m.get("content", "")
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


def test_provisional_mcp_card_carries_server_display_name(tmp_path, monkeypatch):
    """Regression: the early card for a large-argument MCP call must already
    carry the server display name. Without it the card (and a cancelled turn's
    saved history/export/search text) shows the internal uuid server id."""
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    mcp_servers_db.create_server(id = "a3f9c1d2e4b6f807", display_name = "GitHub", url = "https://a/m")

    tool_name = "mcp__a3f9c1d2e4b6f807__create_issue"
    args = {"title": "Bug", "body": "x" * 400}
    assert len(json.dumps(args)) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call(tool_name, args, "call_mcp_big")
    final_stream = [_sse({"content": "Filed."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "file a bug"}],
            tools = [{"type": "function", "function": {"name": tool_name}}],
            max_tool_iterations = 1,
        )
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    assert len(provisional) == 1, tool_starts
    assert provisional[0]["provenance"].get("provisional") is True
    assert provisional[0]["provenance"].get("mcp_server") == "GitHub"


def test_provisional_non_mcp_card_omits_mcp_server(tmp_path, monkeypatch):
    """A plain tool's provisional card gains no mcp_server key."""
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_py")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write code"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )
    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert len(provisional) == 1, events
    assert "mcp_server" not in provisional[0]["provenance"]


def _tool_call_fragment(
    index: int,
    arguments: str,
    call_id: str | None = None,
) -> str:
    delta: dict = {"index": index, "function": {"arguments": arguments}}
    if call_id is not None:
        delta["id"] = call_id
    return _sse({"tool_calls": [delta]})


def _tool_call_opening(index: int, call_id: str, name: str, arguments: str) -> str:
    return _sse(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ]
        }
    )


def test_second_structured_call_at_one_index_keeps_its_own_fragments(monkeypatch):
    """Two tool rounds in one llama-server response, both streamed at index 0.

    llama-server restarts ``delta.tool_calls[].index`` at 0 for every round
    while giving each call its own id, and the continuation fragments carrying
    the rest of the arguments arrive bare. Keying the accumulator on the index
    alone appended round two's name and argument tail to round one, leaving a
    single ``web_searchweb_search`` entry that matched no enabled tool, so
    neither call ran.
    """

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":'),
        _tool_call_fragment(0, '"first"}'),
        _tool_call_opening(0, "call_b", "web_search", '{"query":'),
        _tool_call_fragment(0, '"second"}'),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream, final_stream], payloads)

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search twice"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [
        {"name": "web_search", "arguments": {"query": "first"}},
        {"name": "web_search", "arguments": {"query": "second"}},
    ]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a",
        "call_b",
    ]

    # The replayed conversation must list both calls with their own arguments.
    asst = next(m for m in payloads[1]["messages"] if m.get("tool_calls"))
    assert [tc["id"] for tc in asst["tool_calls"]] == ["call_a", "call_b"]
    assert [tc["function"]["arguments"] for tc in asst["tool_calls"]] == [
        '{"query":"first"}',
        '{"query":"second"}',
    ]


def test_structured_fragment_naming_its_call_goes_back_to_that_call(monkeypatch):
    """Two calls at index 0 with the id repeated on every argument fragment.

    The latest-index mapping only exists to place fragments that carry no id,
    so a fragment naming the call the index opened first has to go back to it
    rather than fork a third slot. Forking left that call with truncated JSON
    and dropped the fragment for having no function name.
    """

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":'),
        _tool_call_opening(0, "call_b", "web_search", '{"query":'),
        _tool_call_fragment(0, '"first"}', call_id = "call_a"),
        _tool_call_fragment(0, '"second"}', call_id = "call_b"),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream, final_stream], payloads)

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search twice"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [
        {"name": "web_search", "arguments": {"query": "first"}},
        {"name": "web_search", "arguments": {"query": "second"}},
    ]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a",
        "call_b",
    ]


def test_structured_call_id_arriving_after_the_opening_delta_updates_that_call(monkeypatch):
    """llama-server can open a call with no id and send the real one later.

    The opening slot holds the synthetic ``call_0`` until then, and treating
    that placeholder as a rival id forked a second nameless slot: the fork was
    dropped for having no function name and the original call ran on truncated
    arguments.
    """

    stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":'},
                    }
                ]
            }
        ),
        _tool_call_fragment(0, '"late"}', call_id = "call_late"),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream, final_stream], payloads)

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [{"name": "web_search", "arguments": {"query": "late"}}]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == ["call_late"]


def test_structured_call_forked_onto_a_reused_index_executes_last(monkeypatch):
    """A first round at indices 0 and 1, then a second round back at index 0.

    The fork belongs at the end: it arrived after both first-round calls, and
    running it ahead of the index-1 call reorders side effects for stateful
    tools. Grouping every fork next to the index it reused did exactly that.
    """

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":"a"}'),
        _tool_call_opening(1, "call_b", "web_search", '{"query":"b"}'),
        _tool_call_opening(0, "call_c", "web_search", '{"query":"c"}'),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream, final_stream], payloads)

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append(arguments)
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search three times"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 2,
        )
    )

    assert calls == [{"query": "a"}, {"query": "b"}, {"query": "c"}]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a",
        "call_b",
        "call_c",
    ]
    asst = next(m for m in payloads[1]["messages"] if m.get("tool_calls"))
    assert [tc["id"] for tc in asst["tool_calls"]] == ["call_a", "call_b", "call_c"]


def test_parallel_disabled_suppresses_provisional_for_reused_index(monkeypatch):
    """A later call at reused index 0 is not the first accumulated call.

    ``parallel_tool_calls=false`` permits a provisional card only for the call
    that can execute. Checking the raw index admitted every fork at index 0,
    leaving a card for the truncated call that could only close empty.
    """

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    big_cmd = "echo start\n" + "\n".join(f"echo line {i}" for i in range(60))
    python_args = json.dumps({"code": big_code})
    terminal_args = json.dumps({"command": big_cmd})
    stream = [
        _tool_call_opening(0, "call_py", "python", python_args[:-1]),
        _tool_call_fragment(0, python_args[-1]),
        _tool_call_opening(0, "call_term", "terminal", terminal_args[:-1]),
        _tool_call_fragment(0, terminal_args[-1]),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [stream, final_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "OK"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "do the first only"}],
            tools = [
                {"type": "function", "function": {"name": "python"}},
                {"type": "function", "function": {"name": "terminal"}},
            ],
            max_tool_iterations = 1,
            disable_parallel_tool_use = True,
            permission_mode = "off",
        )
    )

    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert [e["tool_call_id"] for e in provisional] == ["call_py"]
    assert calls == [("python", {"code": big_code})]
    assert not [
        e
        for e in events
        if e.get("tool_call_id") == "call_term"
        and e.get("type") in {"tool_start", "tool_args", "tool_end"}
    ]


def test_conversation_search_budget_counts_the_tool_catalogue(monkeypatch):
    """The estimator sees the messages only; the tools array is prompt too.

    A large (MCP) catalogue can be thousands of tokens, so a budget ignoring it reports
    room the request lacks, into a tool exchange the next iteration cannot evict.
    """
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_s",
                                "function": {
                                    "name": "search_conversation",
                                    "arguments": '{"query":"the code"}',
                                },
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "It was 5150."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 4096
    # What llama-server would really return: the messages, plus a catalogue that on its
    # own fills most of the window. The estimator counts the messages and nothing else.
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: 2800
        + sum(len(str(message.get("content", ""))) for message in candidate) // 10,
    )

    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    monkeypatch.setattr("core.inference.tools.execute_tool", execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "u" * 2000},
                {"role": "assistant", "content": "a" * 2000},
                {"role": "user", "content": "u" * 2000},
                {"role": "assistant", "content": "a" * 2000},
                {"role": "user", "content": "what was the code"},
            ],
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            max_tokens = 512,
            context_overflow = "truncate_oldest",
        )
    )

    from core.inference.context_window import prompt_budget

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    # 2,800 of the 3,584-token budget is catalogue and framing the estimator cannot see,
    # so what is left is hundreds of tokens, not the thousands it would have claimed.
    assert 0 <= budget < 1000


def test_a_long_tool_run_reports_a_boundary_in_the_requests_own_terms(monkeypatch):
    """dropped_messages is summed by the client, and it counts THIS request's messages.

    A tool loop refits every iteration, so a long agent run also counts the tool
    exchanges it created, which the next request's transcript lacks. Re-applying that
    total advances the boundary past the turns actually evicted, so the boundary is
    carried separately, measured against the messages the request was sent with.
    """
    calls = 6
    streams = []
    for index in range(calls):
        streams.append(
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"c{index}",
                                "function": {
                                    "name": "python",
                                    "arguments": '{"code": "step %d"}' % index,
                                },
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ]
        )
    streams.append([_sse({"content": "done."}), _finish("stop"), _done()])

    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._effective_context_length = 4000
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: sum(
            len(str(message.get("content", ""))) for message in candidate
        )
        // 4,
    )
    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_k: "R" * 3200
    )

    branch = [
        # Unsloth always prepends one and a fit never evicts it, so counting it as the
        # front of the branch reported zero on every compaction.
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "u" * 1200},
        {"role": "assistant", "content": "a" * 1200},
        {"role": "user", "content": "u2" * 600},
        {"role": "assistant", "content": "a2" * 600},
        {"role": "user", "content": "keep going"},
    ]
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = branch,
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tokens = 400,
            max_tool_iterations = calls + 1,
            context_overflow = "truncate_oldest",
        )
    )

    notices = [
        event for event in events if event.get("type") == "context_truncated" and event.get("fits")
    ]
    assert len(notices) > 1, "the fixture must refit more than once"
    # Summed, this passes the number of evictable messages the branch ever had.
    assert sum(notice["dropped_messages"] for notice in notices) > len(branch)
    # The boundary does not: it says where the branch was cut, so it never passes what the
    # branch had to give (4; the system prompt and the latest turn are neither evictable
    # nor counted) and it only ever moves forward.
    boundaries = [notice["boundary_messages"] for notice in notices]
    assert max(boundaries) == 4
    assert boundaries == sorted(boundaries)


def test_conversation_search_budget_is_exact_when_nothing_was_truncated(monkeypatch):
    """`fit_rolling_context` returns None when it drops nothing.

    A prompt that simply FITS, after a context-length increase or on a shorter branch,
    therefore left the budget to a character estimate that cannot see the template's own
    framing. It reported room the request did not have, the recall appended a passage too
    large for the real window, and the next iteration could not evict it again because the
    current tool exchange is protected.
    """
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_s",
                                "function": {
                                    "name": "search_conversation",
                                    "arguments": '{"query":"the code"}',
                                },
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "It was 5150."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 4096
    # Most of the window is catalogue and template framing, which no character estimate
    # can see. The messages themselves are short, so the fit drops nothing at all.
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: 2800
        + sum(len(str(message.get("content", ""))) for message in candidate) // 10,
    )

    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    monkeypatch.setattr("core.inference.tools.execute_tool", execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what was the code"}],
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            max_tokens = 512,
            context_overflow = "truncate_oldest",
        )
    )

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    # 3,584 of budget against a real prompt of roughly 2,800: hundreds of tokens of room,
    # not the thousands the estimate claimed from a handful of short messages.
    assert 0 <= budget < 1000, budget


def test_the_exact_recall_budget_is_recomputed_after_an_intervening_tool(monkeypatch):
    """The exact count is absolute, so caching it for the request goes stale.

    The loop appends the assistant call and the tool result of every intervening tool to
    the conversation, so a figure taken before them understates the prompt by exactly
    those exchanges and hands the search room that is already spent.
    """
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_t",
                                "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_s",
                                "function": {
                                    "name": "search_conversation",
                                    "arguments": '{"query":"the code"}',
                                },
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "It was 5150."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 4096
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: 1000
        + sum(len(str(message.get("content", ""))) for message in candidate) // 10,
    )

    budgets: list = []

    def execute_tool(name, arguments, **kwargs):
        if name == "search_conversation":
            budgets.append(kwargs.get("conversation_budget_tokens"))
            return "an earlier turn"
        # A big result, which the loop appends before the search runs.
        return "x" * 12000

    monkeypatch.setattr("core.inference.tools.execute_tool", execute_tool)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what was the code"}],
            tools = [
                {"type": "function", "function": {"name": "terminal"}},
                {"type": "function", "function": {"name": "search_conversation"}},
            ],
            max_tokens = 512,
            context_overflow = "truncate_oldest",
        )
    )

    assert budgets and budgets[0] is not None
    # The 12,000-character tool result is roughly 1,200 tokens of the 3,584-token budget,
    # and the count taken before it cannot see them.
    assert budgets[0] < 1400, budgets


def _count_from_size(messages, *_args, **_kwargs):
    """Stand in for the tokenizer, priced the way a real chat template prices.

    Two behaviours the fake has to keep or the tests pass while the gate is blind:

    1. The size FALLS when the conversation shrinks, so re-pricing after a compaction is
       distinguishable from the attempt before it.
    2. An assistant turn's `tool_calls` cost NOTHING until a `tool` message answers them.
       Qwen3.8's template renders them only then, which is what made the first version of
       this gate useless: measured on the conversation as it stood, a 40 KB argument was
       invisible, the turn priced at 1,063 tokens against a 4,096 window, and the tool ran
       into a request that came back 400. A counter that charges for unanswered arguments
       cannot catch that regression.
    """
    answered = {
        str(message.get("tool_call_id"))
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    billed = []
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and calls:
            visible = [call for call in calls if str(call.get("id")) in answered]
            billed.append({**message, "tool_calls": visible})
        else:
            billed.append(message)
    return len(json.dumps(billed, default = str)) // 4


def test_an_unservable_tool_call_is_refused_before_it_runs(monkeypatch):
    """The write must not land on a turn llama-server is going to reject anyway.

    The model's own arguments are already in the conversation by the time the tool is
    invoked, so a whole-file `edit_file` can put the prompt over the window before the
    tool has returned anything. `tool_result_budget` clamps to zero there and the
    truncation reads that as "cut hard", so the tool used to run, the result was cut to
    its notice, and the next request was refused with the file written.
    """
    # The bulk is in the USER turn, which no receipt can replace, so running the call and
    # compacting its arguments cannot rescue this one either -- which is what makes it the
    # case that still earns a refusal.
    immovable = "please read all of this: " + "u" * 40000
    streams = [
        _structured_tool_call(
            "edit_file",
            {"path": "flappy-bird.html", "edits": [{"old_string": "", "new_string": "x" * 2000}]},
            "call_write_game",
        ),
        [_sse({"content": "Understood."}), _done()],
    ]
    backend = _make_backend(monkeypatch, streams, [])
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []

    def fake_execute_tool(name, _arguments, **_kwargs):
        executed.append(name)
        return "wrote the file"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": immovable}],
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tokens = 512,
            max_tool_iterations = 2,
        )
    )

    assert executed == [], "the side effect was spent on an unservable turn"
    refusals = [
        event
        for event in events
        if event.get("type") == "tool_end" and "Nothing was written" in str(event.get("result", ""))
    ]
    assert refusals, [e.get("type") for e in events]
    assert "edit_file" in refusals[0]["result"]


def test_compacting_an_earlier_call_lets_the_next_one_run(monkeypatch):
    """The first lever, before refusing: arguments of a call that already returned.

    They are pure replay -- the tool received them in full and the file is on disk -- so
    spending them is what keeps a thread alive that would otherwise dead-end.
    """
    earlier = "<!DOCTYPE html>" + "y" * 30000
    prior_call = {
        "id": "call_earlier",
        "type": "function",
        "function": {
            "name": "edit_file",
            "arguments": json.dumps({"path": "page.html", "old_string": "", "new_string": earlier}),
        },
    }
    history = [
        {"role": "user", "content": "Write page.html"},
        {"role": "assistant", "content": "Writing.", "tool_calls": [prior_call]},
        {
            "role": "tool",
            "tool_call_id": "call_earlier",
            "name": "edit_file",
            "content": "Wrote page.html",
        },
        {"role": "user", "content": "Now fix the title"},
    ]
    streams = [
        _structured_tool_call(
            "edit_file",
            {
                "path": "page.html",
                "old_string": "<title>a</title>",
                "new_string": "<title>b</title>",
            },
            "call_fix_title",
        ),
        [_sse({"content": "Fixed."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append(name)
        # The tool still receives real arguments, never a receipt.
        assert arguments.get("new_string") == "<title>b</title>"
        return "Edited page.html"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = history,
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tokens = 512,
            max_tool_iterations = 2,
        )
    )

    assert executed == ["edit_file"], [e.get("type") for e in events]
    assert not [
        event
        for event in events
        if event.get("type") == "tool_end" and "Nothing was written" in str(event.get("result", ""))
    ]
    # The assertions above hold with no gate at all -- an ungated loop runs every tool it
    # is handed. What distinguishes the fix is the prompt SENT after the tool returned:
    # the earlier call's 30 KB argument must have become a receipt, and only there.
    assert len(payloads) >= 2, "the loop never made a second request"
    replayed = json.dumps(payloads[-1]["messages"], default = str)
    assert earlier not in replayed, "the earlier 30 KB argument was replayed verbatim"
    assert "arguments you sent" in replayed
    assert "page.html" in replayed


def test_refusing_a_call_also_stops_it_costing_the_window(monkeypatch):
    """Observed live: an accurate refusal, then the 400 it was issued to prevent.

    The refusal is a `tool` message, and a chat template renders an assistant turn's
    `tool_calls` only once one of those answers them. So declining to run the tool is the
    very thing that makes its arguments start costing the prompt, and the generation that
    follows is rejected anyway -- with nothing written, but also nothing the user can do.
    The refused arguments are the one case with no replay value at all.
    """
    # Again the bulk is immovable: a refusal is the only outcome left, and the point here
    # is that refusing must not ALSO leave the arguments costing the window.
    immovable = "please read all of this: " + "u" * 40000
    oversized = "<!DOCTYPE html>" + "x" * 8000
    streams = [
        _structured_tool_call(
            "edit_file",
            {"path": "flappy-bird.html", "edits": [{"old_string": "", "new_string": oversized}]},
            "call_write_game",
        ),
        [_sse({"content": "I could not write that file."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, _arguments, **_kwargs: executed.append(name) or "wrote it",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": immovable}],
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tokens = 512,
            max_tool_iterations = 2,
        )
    )

    assert executed == []
    assert len(payloads) >= 2, "the loop never got to a follow-up generation"
    replayed = json.dumps(payloads[-1]["messages"], default = str)
    # The prompt that follows the refusal must not carry what was refused.
    assert oversized not in replayed
    assert "refused before it ran" in replayed
    # And must not claim a file exists to go and read.
    assert "re-read the file" not in replayed


def test_reply_room_is_reclaimed_before_generating(monkeypatch):
    """A prompt that FITS can still leave nothing to answer in.

    Observed on a 4096 window: every tool call servable, none refused, the file written,
    and the turn ended on `finish_reason: length` with the model still thinking -- the
    prompt had eaten the room its answer needed. The pre-execution gate never fired
    because nothing was ever unservable, so compaction, the exact lever for this, was
    never asked to run.
    """
    bulky = "<!DOCTYPE html>" + "z" * 30000
    prior_call = {
        "id": "call_done",
        "type": "function",
        "function": {
            "name": "edit_file",
            "arguments": json.dumps(
                {"path": "game.html", "edits": [{"old_string": "", "new_string": bulky}]}
            ),
        },
    }
    history = [
        {"role": "user", "content": "Write game.html"},
        {"role": "assistant", "content": "Writing.", "tool_calls": [prior_call]},
        {
            "role": "tool",
            "tool_call_id": "call_done",
            "name": "edit_file",
            "content": "Created game.html",
        },
        {"role": "user", "content": "Now tell me what you did"},
    ]
    payloads: list[dict] = []
    # No tool call this turn: the model just answers, so only the reply-room pass can act.
    backend = _make_backend(monkeypatch, [[_sse({"content": "Done."}), _done()]], payloads)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "should not run",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = history,
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tokens = 512,
            max_tool_iterations = 1,
        )
    )

    assert payloads, "no generation request was made"
    sent = json.dumps(payloads[0]["messages"], default = str)
    assert bulky not in sent, "the finished call's 30 KB argument was still replayed"
    assert "arguments you sent" in sent


def test_an_oversized_call_is_run_and_compacted_rather_than_refused(monkeypatch):
    """Refusing costs the same tokens as running, and leaves nothing written.

    The refusal is itself the `tool` message that makes the arguments render, so declining
    does not avoid their cost. The model then retries with a fresh oversized call and each
    round reclaims less -- 50%, then 34%, then 15% of one measured thread, ending in a
    one-character reply. Running the call needs no context at all; only the next prompt
    does, and by then the arguments describe a file on disk.
    """
    oversized = "<!DOCTYPE html>" + "x" * 24000
    streams = [
        _structured_tool_call(
            "edit_file",
            {"path": "flappy-bird.html", "edits": [{"old_string": "", "new_string": oversized}]},
            "call_write_game",
        ),
        [_sse({"content": "Wrote the game."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, streams, payloads)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append(name)
        # The tool still receives the real content -- the file must actually be written.
        assert oversized in json.dumps(arguments)
        return "Created flappy-bird.html"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tokens = 512,
            max_tool_iterations = 2,
        )
    )

    assert executed == ["edit_file"], "the call was refused instead of run"
    assert not [
        e
        for e in events
        if e.get("type") == "tool_end" and "Nothing was written" in str(e.get("result", ""))
    ]
    assert len(payloads) >= 2, "no follow-up generation was made"
    replayed = json.dumps(payloads[-1]["messages"], default = str)
    assert oversized not in replayed, "the arguments were replayed after the call ran"
    assert "arguments you sent" in replayed


_BIG_BODY = "<!DOCTYPE html>" + "x" * 9000


def _two_edits_in_one_turn():
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_big",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps(
                                {
                                    "path": "game.html",
                                    "old_string": "",
                                    "new_string": _BIG_BODY,
                                }
                            ),
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_small",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps(
                                {
                                    "path": "game.html",
                                    "old_string": "TODO",
                                    "new_string": "done",
                                }
                            ),
                        },
                    },
                ]
            }
        ),
        _done(),
    ]


def test_a_second_call_in_a_compacted_turn_is_still_visible_to_the_model(monkeypatch):
    """Compaction rebuilds the messages, which silently detaches the local handle.

    The run-then-compact rescue rewrites `conversation` in place. The loop was still
    holding the assistant message it built BEFORE that, so the next call in the same
    batch appended its `tool_call` to a dict no longer in the list while its RESULT was
    appended to the list. The model then received a `tool` message answering a call it
    could not see, which some templates reject outright and the rest render as an
    unexplained result.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [_two_edits_in_one_turn(), [_sse({"content": "Done."}), _done()]],
        payloads,
    )

    # Price the turn off the replayed JSON: the big call does not fit, the receipt does.
    def fake_count_chat_tokens(messages, *_args, **_kwargs):
        return len(json.dumps(messages, default = str)) // 2

    monkeypatch.setattr(backend, "count_chat_tokens", fake_count_chat_tokens)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Wrote game.html",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Write the game"}],
            tools = [{"type": "function", "function": {"name": "edit_file"}}],
            max_tool_iterations = 4,
        )
    )

    sent = payloads[-1]["messages"]
    announced = {
        call.get("id")
        for message in sent
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    answered = {message.get("tool_call_id") for message in sent if message.get("role") == "tool"}

    assert answered, "no tool result reached the model at all"
    assert answered <= announced, f"results with no visible call: {answered - announced}"
    # And the compaction still happened: the body is not replayed.
    assert _BIG_BODY not in json.dumps(sent)
