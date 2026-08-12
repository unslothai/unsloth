# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the local tool loop over self-hosted OpenAI-compat providers.

Covers the provider/tool gates that decide whether the loop may run at all, and
the loop itself against a fake ExternalProviderClient: tool_calls fragments must
never reach the client, tool cards must be emitted in their place, and the
follow-up turn must carry the assistant tool_calls plus the tool results.
"""

import asyncio
import json
import time

import pytest

from core.inference.external_tool_loop import (
    local_tool_loop_supported,
    select_local_tool_names,
    stream_chat_completion_with_local_tools,
)


def _chunk(**delta) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def _finish(reason: str) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        }
    )


def _usage(prompt: int, completion: int) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


class _FakeClient:
    """Replays one canned SSE turn per call and records the request kwargs."""

    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]
        self.calls = []

    async def stream_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        for line in self.turns.pop(0):
            yield line


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {"name": "web_search", "parameters": {"type": "object", "properties": {}}},
}
PYTHON_TOOL = {
    "type": "function",
    "function": {"name": "python", "parameters": {"type": "object", "properties": {}}},
}


async def _collect(gen) -> list[str]:
    return [line async for line in gen]


def _events(lines: list[str]) -> list[dict]:
    """Parse the custom tool events out of an SSE line list."""
    parsed = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        if isinstance(data, dict) and "type" in data:
            parsed.append(data)
    return parsed


def _content(lines: list[str]) -> str:
    """Concatenate the assistant text the client would render."""
    text = ""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        for choice in data.get("choices") or []:
            text += (choice.get("delta") or {}).get("content") or ""
    return text


# ── gates ────────────────────────────────────────────────────────


def test_local_tool_loop_supported_only_for_self_hosted_providers():
    for provider_type in ("vllm", "ollama", "llama_cpp", "custom"):
        assert local_tool_loop_supported(provider_type) is True, provider_type
    # hosted providers ship their own server-side tools; the local loop must
    # never advertise a competing catalog to them.
    for provider_type in ("openai", "anthropic", "gemini", "openrouter", "kimi", None, ""):
        assert local_tool_loop_supported(provider_type) is False, provider_type


def test_select_local_tool_names_filters_and_orders():
    assert select_local_tool_names(["terminal", "web_search", "python"]) == [
        "web_search",
        "python",
        "terminal",
    ]
    # rag stays local-only and the hosted builtin names are not local tools.
    assert select_local_tool_names(["search_knowledge_base"]) == []
    assert select_local_tool_names(["render_html", "web_fetch", "code_execution"]) == []
    assert select_local_tool_names(None) == []
    assert select_local_tool_names([]) == []


# ── loop ─────────────────────────────────────────────────────────


def test_tool_call_is_executed_and_never_forwarded_raw():
    client = _FakeClient(
        [
            [
                _chunk(role = "assistant"),
                _chunk(content = "Looking that up. "),
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "web_search", "arguments": '{"query":'},
                        }
                    ]
                ),
                _chunk(tool_calls = [{"index": 0, "function": {"arguments": '"unsloth"}'}}]),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "Unsloth is a fine-tuning library."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    executed = []

    def _execute(name, arguments, **kwargs):
        executed.append((name, arguments, kwargs))
        return "Title: Unsloth\nURL: https://unsloth.ai"

    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "what is unsloth"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                session_id = "sess-1",
                thread_id = "thread-1",
                execute_tool = _execute,
            )
        )
    )

    assert len(executed) == 1
    name, arguments, kwargs = executed[0]
    assert (name, arguments) == ("web_search", {"query": "unsloth"})
    assert kwargs["timeout"] is None
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["thread_id"] == "thread-1"
    assert kwargs["disable_sandbox"] is False
    # live stdout reaches the card only if the streaming wrapper's sink is forwarded.
    assert callable(kwargs["output_callback"])

    # no raw tool_calls fragment may reach the client: the frontend would draw a
    # second card that never resolves.
    assert all('"tool_calls"' not in line for line in lines if "type" not in line)

    events = _events(lines)
    starts = [event for event in events if event["type"] == "tool_start"]
    ends = [event for event in events if event["type"] == "tool_end"]
    assert len(starts) == 1
    assert starts[0]["tool_name"] == "web_search"
    assert starts[0]["arguments"] == {"query": "unsloth"}
    assert starts[0]["awaiting_confirmation"] is False
    assert len(ends) == 1
    assert ends[0]["result"] == "Title: Unsloth\nURL: https://unsloth.ai"

    assert lines[-1] == "data: [DONE]"
    # only the last turn's finish_reason survives; a mid-loop one would end the
    # client's stream before the answer.
    finishes = [
        json.loads(line[len("data:") :].strip())["choices"][0]["finish_reason"]
        for line in lines
        if line != "data: [DONE]" and '"choices"' in line
    ]
    assert [reason for reason in finishes if reason] == ["stop"]

    # the follow-up turn replays the assistant tool_calls and the tool result.
    follow_up = client.calls[1]["messages"]
    assert follow_up[-2]["role"] == "assistant"
    assert follow_up[-2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert follow_up[-1] == {
        "role": "tool",
        "name": "web_search",
        "content": "Title: Unsloth\nURL: https://unsloth.ai",
        "tool_call_id": "call_0",
    }


def test_answer_without_tool_calls_passes_straight_through():
    client = _FakeClient([[_chunk(content = "hello"), _finish("stop"), "data: [DONE]"]])
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "hi"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda *a, **k: pytest.fail("no tool should run"),
            )
        )
    )
    assert len(client.calls) == 1
    assert client.calls[0]["tools"] == [WEB_SEARCH_TOOL]
    assert _events(lines) == []
    assert lines[-1] == "data: [DONE]"


def test_iteration_budget_drops_the_catalog_for_the_final_pass():
    tool_turn = [
        _chunk(
            tool_calls = [
                {
                    "index": 0,
                    "id": "call_0",
                    "function": {"name": "python", "arguments": '{"code":"1"}'},
                }
            ]
        ),
        _finish("tool_calls"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_turn, [_chunk(content = "done"), _finish("stop"), "data: [DONE]"]])

    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "run it"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                max_tool_iterations = 1,
                execute_tool = lambda *a, **k: "1",
            )
        )
    )
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] == [PYTHON_TOOL]
    # budget spent: the last request carries no catalog, so the model has to answer.
    assert client.calls[1]["tools"] is None
    assert lines[-1] == "data: [DONE]"


def test_usage_is_summed_into_one_trailing_chunk():
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ]
                ),
                _finish("tool_calls"),
                _usage(10, 5),
                "data: [DONE]",
            ],
            [_chunk(content = "ok"), _finish("stop"), _usage(20, 7), "data: [DONE]"],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "go"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                execute_tool = lambda *a, **k: "ok",
            )
        )
    )
    usage_chunks = [
        json.loads(line[len("data:") :].strip())
        for line in lines
        if line != "data: [DONE]" and '"usage"' in line
    ]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }


def test_provider_error_ends_the_loop_without_re_prompting():
    client = _FakeClient(
        [
            [
                'data: {"error": {"message": "boom", "type": "provider_error"}}',
                "data: [DONE]",
            ]
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "go"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                execute_tool = lambda *a, **k: pytest.fail("no tool should run"),
            )
        )
    )
    assert len(client.calls) == 1
    assert lines == ['data: {"error": {"message": "boom", "type": "provider_error"}}']


def test_bypass_permissions_disables_the_sandbox():
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "python", "arguments": '{"code":"1"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(content = "1"), _finish("stop"), "data: [DONE]"],
        ]
    )
    seen = {}

    def _execute(name, arguments, **kwargs):
        seen.update(kwargs)
        return "1"

    asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "go"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                bypass_permissions = True,
                confirm_tool_calls = True,
                execute_tool = _execute,
            )
        )
    )
    # bypass wins over the confirm gate, so the call runs unprompted with the
    # sandbox off.
    assert seen["disable_sandbox"] is True


def test_disabled_tool_call_is_suppressed_and_nudged():
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(content = "cannot"), _finish("stop"), "data: [DONE]"],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "list files"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                execute_tool = lambda *a, **k: pytest.fail("terminal is not enabled"),
            )
        )
    )
    # tool_status is only the badge text; a suppressed call must draw no card.
    card_events = {"tool_start", "tool_end", "tool_output", "tool_args"}
    assert [event["type"] for event in _events(lines) if event["type"] in card_events] == []
    # the unadvertised call becomes a hidden nudge, never a tool result.
    follow_up = client.calls[1]["messages"]
    assert follow_up[-1]["role"] == "user"
    assert "not enabled" in follow_up[-1]["content"]


# ── parity with the local loops ──────────────────────────────────


def test_ollama_reasoning_is_renamed_for_the_frontend():
    """Ollama says `reasoning`; the local loops and the UI say `reasoning_content`."""
    client = _FakeClient(
        [
            [
                _chunk(reasoning = "Thinking about it."),
                _chunk(content = "Done."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "hi"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    deltas = [
        (json.loads(line[len("data:") :].strip()).get("choices") or [{}])[0].get("delta") or {}
        for line in lines
        if line.startswith("data:") and '"choices"' in line
    ]
    assert any(d.get("reasoning_content") == "Thinking about it." for d in deltas)
    assert all("reasoning" not in d for d in deltas)


def test_reasoning_only_answer_is_promoted_to_content():
    """An always-think model puts its whole reply in `reasoning` and sends no
    content. The GGUF loop shows that as the answer; so must this one."""
    client = _FakeClient(
        [
            [
                _chunk(reasoning = "The user asked a simple question. "),
                _chunk(reasoning = "Unsloth is a fine-tuning library."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "what is unsloth"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert "Unsloth is a fine-tuning library." in _content(lines)


def test_truncated_reasoning_is_not_promoted():
    """A thought cut off by the token cap is not a final answer."""
    client = _FakeClient(
        [
            [
                _chunk(reasoning = "Unsloth is a fine-tuning library that supports"),
                _finish("length"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "explain"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert _content(lines) == ""


def test_plan_without_action_is_re_prompted_once():
    """Model describes the search instead of calling it: nudge it to act."""
    client = _FakeClient(
        [
            [
                _chunk(reasoning = "I will search the web for the 2026 box office totals."),
                _finish("stop"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "The Odyssey led 2026."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "highest grossing movie of 2026"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert len(client.calls) == 2
    retry = client.calls[1]["messages"]
    assert retry[-1]["role"] == "user"
    assert "web_search" in retry[-1]["content"]
    # the stalled turn is kept as the assistant turn the nudge answers.
    assert retry[-2]["role"] == "assistant"
    # a hidden retry looks like a hang without a badge.
    statuses = [event["content"] for event in _events(lines) if event["type"] == "tool_status"]
    assert any(status for status in statuses)
    assert _content(lines) == "The Odyssey led 2026."


def test_visible_text_is_never_re_prompted():
    """OpenAI deltas are append-only: text already streamed cannot be retracted,
    so a turn that showed something has to stand as the answer."""
    client = _FakeClient(
        [
            [
                _chunk(content = "I will search the web for that."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "highest grossing movie of 2026"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert len(client.calls) == 1
    assert _content(lines) == "I will search the web for that."


def test_tool_stdout_streams_to_the_card():
    """python/terminal output has to reach the card while the tool runs."""
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "It printed 1."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )

    def _execute(
        name,
        arguments,
        output_callback = None,
        **kwargs,
    ):
        output_callback("1\n")
        return "1"

    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "print 1"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                execute_tool = _execute,
            )
        )
    )
    outputs = [event for event in _events(lines) if event["type"] == "tool_output"]
    assert [event["text"] for event in outputs] == ["1\n"]
    assert outputs[0]["tool_call_id"] == "call_1"


def test_large_payload_opens_a_provisional_card_and_streams_args():
    """A long code payload renders while the model is still writing it."""
    code = "x = 1\n" * 60
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "python", "arguments": '{"code":"' + code},
                        }
                    ]
                ),
                _chunk(tool_calls = [{"index": 0, "function": {"arguments": '"}'}}]),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "Done."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "run it"}],
                model = "qwen3-14b",
                tools = [PYTHON_TOOL],
                execute_tool = lambda *a, **k: "ok",
            )
        )
    )
    events = _events(lines)
    starts = [event for event in events if event["type"] == "tool_start"]
    args_events = [event for event in events if event["type"] == "tool_args"]
    ends = [event for event in events if event["type"] == "tool_end"]
    assert starts[0]["provenance"].get("provisional") is True
    assert "".join(event["text"] for event in args_events).endswith('"}')
    # both starts carry the provider's call id, so the card reconciles instead of
    # opening a second one, and only the real end closes it.
    assert {event["tool_call_id"] for event in starts} == {"call_1"}
    assert len(ends) == 1
    assert ends[0]["result"] == "ok"


def test_text_form_tool_call_runs_and_never_leaks():
    """Some servers hand back the call as text. It has to execute, and the
    markup must not reach the transcript."""
    client = _FakeClient(
        [
            [
                _chunk(content = "Let me check that. "),
                _chunk(content = '<tool_call>\n{"name": "web_search", '),
                _chunk(content = '"arguments": {"query": "unsloth"}}\n</tool_call>'),
                _finish("stop"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "Unsloth is a fine-tuning library."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    executed = []
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "what is unsloth"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda name, args, **kw: executed.append((name, args)) or "result",
            )
        )
    )
    assert executed == [("web_search", {"query": "unsloth"})]
    assert _content(lines) == "Let me check that. Unsloth is a fine-tuning library."
    assert [event["type"] for event in _events(lines) if event["type"] == "tool_start"] == [
        "tool_start"
    ]
    # the replayed assistant turn carries the call structurally, not as markup.
    assert "<tool_call>" not in json.dumps(client.calls[1]["messages"])


def test_unparsed_markup_is_released_as_prose():
    """A marker that never forms a call is text the user asked for."""
    client = _FakeClient(
        [
            [
                _chunk(content = "Write it as <tool_call> and the server parses it."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "how do tool calls look"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert _content(lines) == "Write it as <tool_call> and the server parses it."


def test_partial_marker_at_a_delta_boundary_is_not_split():
    """A marker split across deltas must still be caught, and a lookalike tail
    must still be released."""
    client = _FakeClient(
        [
            [
                _chunk(content = "Prefix <tool"),
                _chunk(content = "s are handy."),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "hi"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
            )
        )
    )
    assert _content(lines) == "Prefix <tools are handy."


def test_budget_exhausted_turn_is_told_to_answer():
    """The last pass gets the local loops' nudge, not a silently empty catalog."""
    call_chunk = _chunk(
        tool_calls = [
            {
                "index": 0,
                "id": "call_0",
                "function": {"name": "web_search", "arguments": '{"query":"a"}'},
            }
        ]
    )
    client = _FakeClient(
        [
            [call_chunk, _finish("tool_calls"), "data: [DONE]"],
            [_chunk(content = "Final answer."), _finish("stop"), "data: [DONE]"],
        ]
    )
    asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "search"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                max_tool_iterations = 1,
                execute_tool = lambda *a, **k: "ok",
            )
        )
    )
    assert client.calls[1]["tools"] is None
    assert "all available tool calls" in client.calls[1]["messages"][-1]["content"]


def test_parallel_tool_calls_disabled_runs_only_the_first():
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "web_search", "arguments": '{"query":"a"}'},
                        },
                        {
                            "index": 1,
                            "id": "call_1",
                            "function": {"name": "web_search", "arguments": '{"query":"b"}'},
                        },
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(content = "Done."), _finish("stop"), "data: [DONE]"],
        ]
    )
    executed = []
    asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "search twice"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                disable_parallel_tool_use = True,
                execute_tool = lambda name, args, **kw: executed.append(args) or "ok",
            )
        )
    )
    assert executed == [{"query": "a"}]


def test_text_markup_never_leaks_when_the_call_cannot_run():
    """Budget spent, so the attempted call is dropped rather than printed."""
    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "web_search", "arguments": '{"query":"a"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [
                _chunk(content = "Here is what I found. "),
                _chunk(
                    content = (
                        "<tool_call> <function=web_search> <parameter=url> "
                        "https://example.com/ </tool_call>"
                    )
                ),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "search"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                max_tool_iterations = 1,
                execute_tool = lambda *a, **k: "ok",
            )
        )
    )
    assert _content(lines) == "Here is what I found. "


def test_text_markup_alongside_a_structured_call_never_leaks():
    """One turn carrying both forms must still show only the prose."""
    client = _FakeClient(
        [
            [
                _chunk(content = "Checking. <tool_call> <function=web_search> "),
                _chunk(content = "<parameter=url> https://example.com/ </tool_call>"),
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "web_search", "arguments": '{"query":"a"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(content = "Found it."), _finish("stop"), "data: [DONE]"],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "search"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                execute_tool = lambda *a, **k: "ok",
            )
        )
    )
    assert _content(lines) == "Checking. Found it."


def test_unhealed_text_call_is_hidden_even_when_it_cannot_run():
    """Auto-Heal off blocks execution, never display: markup still stays out."""
    client = _FakeClient(
        [
            [
                _chunk(
                    content = (
                        "<tool_call> <function=web_search> <parameter=url> "
                        "https://example.com/ </tool_call>"
                    )
                ),
                _finish("stop"),
                "data: [DONE]",
            ],
        ]
    )
    lines = asyncio.run(
        _collect(
            stream_chat_completion_with_local_tools(
                client,
                messages = [{"role": "user", "content": "search"}],
                model = "qwen3-14b",
                tools = [WEB_SEARCH_TOOL],
                auto_heal_tool_calls = False,
                execute_tool = lambda *a, **k: pytest.fail("an unhealed call must not run"),
            )
        )
    )
    assert "<tool_call>" not in _content(lines)


def test_approval_wait_flushes_card_on_a_separate_event_loop_turn():
    """The first keepalive must not coalesce with the gated card's write."""
    import core.inference.external_tool_loop as loop_module

    client = _FakeClient(
        [
            [
                _chunk(
                    tool_calls = [
                        {
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                        }
                    ]
                ),
                _finish("tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(content = "1"), _finish("stop"), "data: [DONE]"],
        ]
    )
    decided = []

    def _slow_decision(
        slot,
        approval_id,
        cancel_event = None,
        timeout = None,
    ):
        decided.append(approval_id)
        time.sleep(0.25)
        return "allow"

    async def _collect_with_turn_probe(stream):
        lines = []
        next_turn = None
        first_post_card_turn_advanced = None
        async for line in stream:
            if next_turn is not None and first_post_card_turn_advanced is None:
                first_post_card_turn_advanced = next_turn[0]
            lines.append(line)
            if '"tool_start"' in line:
                next_turn = [False]
                asyncio.get_running_loop().call_soon(next_turn.__setitem__, 0, True)
        return lines, first_post_card_turn_advanced

    original_wait = loop_module.wait_tool_decision
    original_flush_delay = loop_module.TOOL_APPROVAL_FLUSH_DELAY_S
    original_interval = loop_module.TOOL_HEARTBEAT_INTERVAL_S
    loop_module.wait_tool_decision = _slow_decision
    loop_module.TOOL_APPROVAL_FLUSH_DELAY_S = 0.01
    loop_module.TOOL_HEARTBEAT_INTERVAL_S = 0.05
    try:
        lines, first_post_card_turn_advanced = asyncio.run(
            _collect_with_turn_probe(
                stream_chat_completion_with_local_tools(
                    client,
                    messages = [{"role": "user", "content": "print 1"}],
                    model = "qwen3-14b",
                    tools = [PYTHON_TOOL],
                    session_id = "sess-1",
                    confirm_tool_calls = True,
                    permission_mode = "ask",
                    execute_tool = lambda *a, **k: "1",
                )
            )
        )
    finally:
        loop_module.wait_tool_decision = original_wait
        loop_module.TOOL_APPROVAL_FLUSH_DELAY_S = original_flush_delay
        loop_module.TOOL_HEARTBEAT_INTERVAL_S = original_interval

    assert decided, "the approval gate never ran"
    starts = [event for event in _events(lines) if event["type"] == "tool_start"]
    assert starts[0]["awaiting_confirmation"] is True
    keepalives = [line for line in lines if line.startswith(":")]
    assert keepalives, "no keepalive was written while the approval was pending"
    card_index = next(i for i, line in enumerate(lines) if '"tool_start"' in line)
    assert lines.index(keepalives[0]) > card_index
    assert first_post_card_turn_advanced is True
