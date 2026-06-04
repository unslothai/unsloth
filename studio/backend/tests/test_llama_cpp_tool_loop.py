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

from core.inference.llama_cpp import LlamaCppBackend


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
    def fake_stream_with_retry(_client, _url, payload, _cancel_event, headers = None):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(response, _cancel_event):
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


def test_structured_tool_call_after_visible_preface_is_executed(monkeypatch):
    """llama-server may emit content first and then native delta.tool_calls.

    Studio must not drop that tool call after it has streamed the preface.
    """

    tool_call_id = "call_render_late"
    first_stream = [
        _sse({"content": "Here is the artifact.\n\n"}),
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
        return "Rendered HTML artifact: Simple Red Square."

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
    assert content_events[0]["text"] == "Here is the artifact.\n\n"

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
    assert any(
        e.get("type") == "tool_end" and e.get("tool_name") == "render_html"
        for e in events
    )

    # The second llama-server request should include the assistant preface
    # plus the structured tool call, preserving OpenAI-compatible ordering.
    assert len(payloads) == 2
    assistant_messages = [
        m for m in payloads[1]["messages"] if m.get("role") == "assistant"
    ]
    assert assistant_messages[-1]["content"] == "Here is the artifact.\n\n"
    assert assistant_messages[-1]["tool_calls"][0]["id"] == tool_call_id
    assert assistant_messages[-1]["tool_calls"][0]["function"]["name"] == "render_html"


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
    backend = _make_backend(
        monkeypatch, [first_stream, repeat_stream, final_stream], payloads
    )

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML artifact: First."

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
        return "Rendered HTML artifact: Done."

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
    assert any(
        event.get("type") == "content" and event.get("text") == "Done."
        for event in events
    )
    final_user_messages = [
        m.get("content", "") for m in payloads[1]["messages"] if m.get("role") == "user"
    ]
    assert not any(
        "used all available tool calls" in message for message in final_user_messages
    )


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
            messages = [
                {"role": "user", "content": "search gpus in 2026 prices and use python"}
            ],
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
            messages = [
                {"role": "user", "content": "search gpus in 2026 prices and use python"}
            ],
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
    assert [
        event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
    ] == ["call_search_1"]
    assert len(payloads) == 4
    assert "tools" not in payloads[-1]
    assert any(
        event.get("type") == "content"
        and event.get("text") == "Final answer from first search."
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
    assert [
        event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
    ] == ["call_search_1"]
    assert not [
        event
        for event in events
        if event.get("tool_call_id") == "call_search_2"
        and event.get("type") in {"tool_start", "tool_end"}
    ]


def test_same_turn_repeated_render_html_does_not_emit_second_provisional_start(
    monkeypatch,
):
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
    backend = _make_backend(
        monkeypatch, [same_turn_render_calls, final_stream], payloads
    )

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML artifact: One."

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

    assert not [
        event for event in events if event.get("type") in {"tool_start", "tool_end"}
    ]
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
    not turned into repeated internal re-prompts after the artifact already
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
        return "Rendered HTML artifact: First."

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
        event.get("type") == "content"
        and event.get("text") == "I will now use render_html again."
        for event in events
    )


def test_internal_reprompt_attempts_do_not_duplicate_visible_text(monkeypatch):
    """No-tool re-prompt attempts should not concatenate into the UI."""

    streams = [
        [_sse({"content": "I will use render_html now."}), _done()],
        [_sse({"content": "Understood. I will use render_html now."}), _done()],
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

    content_texts = [
        event.get("text", "") for event in events if event.get("type") == "content"
    ]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == 2


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

    content_texts = [
        event.get("text", "") for event in events if event.get("type") == "content"
    ]
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

    content_texts = [
        event.get("text", "") for event in events if event.get("type") == "content"
    ]
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
        return "Rendered HTML artifact: Forced."

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
    content_texts = [
        event.get("text", "") for event in events if event.get("type") == "content"
    ]
    assert content_texts == ["I will use render_html now.", "Final note after tool."]
    assert len(payloads) == 3
