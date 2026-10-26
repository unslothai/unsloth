# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drives generate_chat_completion_with_tools with fake SSE streams; needs no model or GPU."""

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
from core.inference import llama_cpp
from core.inference.llama_cpp import _should_suppress_forced_no_tool_output as suppress
import httpx


def _web_search_tool():
    """The web_search tool schema the tool-loop tests advertise."""
    return {
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


def _render_html_tools():
    """The one-entry render_html tool list the tool-loop tests advertise."""
    return [
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


def fake_execute_tool(name, arguments, **_kwargs):
    raise AssertionError(f"unexpected tool execution: {name} {arguments}")


def _tool_call_sse(
    tool_name: str,
    arguments: dict,
    call_id: str,
    index: int = 0,
) -> str:
    """One tool-call delta frame: the twelve-line literal the streams below repeat."""
    return _sse(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                }
            ]
        }
    )


def _run_tool_loop(
    backend,
    messages,
    tools,
    *,
    max_tool_iterations = 1,
    **kwargs,
):
    """Drain the tool loop into a list, the seven-line call nearly every test below repeats."""
    return list(
        backend.generate_chat_completion_with_tools(
            messages = messages,
            tools = tools,
            max_tool_iterations = max_tool_iterations,
            **kwargs,
        )
    )


_MCP_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAYAAAAGCAIAAABvrngfAAAAFElEQVR4nGM8ISfHgAqY0PhUFgIAdYgBEED+8ToAAAAASUVORK5CYII="


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


def _backend_and_payloads(monkeypatch, streams):
    """_make_backend plus the payload list every caller threads through it."""
    payloads: list[dict] = []
    return _make_backend(monkeypatch, streams, payloads), payloads


def _record_tool_calls(monkeypatch, result):
    """Stub execute_tool to record (name, arguments) pairs; a callable result receives the tool name."""
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return result(name) if callable(result) else result

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    return calls


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
    assert samples[0]["prompt_progress"]["total"] == 1000
    assert samples[1]["running_phase"] == "token_generation"
    assert all("prompt_ms" not in sample for sample in samples)
    assert samples[-1]["predicted_per_second"] == 200


def test_plain_fixed_seed_disables_slot_prompt_cache_reuse(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": "seeded"}), _done()]],
        payloads,
    )

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "repeat this"}],
            seed = 3407,
        )
    )

    assert payloads[0]["seed"] == 3407
    assert payloads[0]["cache_prompt"] is False


def test_the_uint32_random_seed_sentinel_also_keeps_cache_reuse(monkeypatch):
    """llama.h defines LLAMA_DEFAULT_SEED as 0xFFFFFFFF read as uint32, so 4294967295 is the same "pick one at random" as -1 and must keep prompt-cache reuse."""
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": "random"}), _done()]],
        payloads,
    )

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "vary this"}],
            seed = 0xFFFFFFFF,
        )
    )

    assert payloads[0]["seed"] == 0xFFFFFFFF
    assert "cache_prompt" not in payloads[0]


def test_plain_random_seed_sentinel_keeps_slot_prompt_cache_reuse(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": "random"}), _done()]],
        payloads,
    )

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "vary this"}],
            seed = -1,
        )
    )

    assert payloads[0]["seed"] == -1
    assert "cache_prompt" not in payloads[0]


def test_reuse_prompt_cache_keeps_fixed_seed_without_disabling_slot_reuse():
    """Tool-loop continuations opt out of #9979's cold-cache pin (#10698)."""
    from core.inference.llama_cpp import _apply_seeded_llama_request

    cold: dict = {}
    _apply_seeded_llama_request(cold, 3407)
    assert cold == {"seed": 3407, "cache_prompt": False}

    warm: dict = {}
    _apply_seeded_llama_request(warm, 3407, reuse_prompt_cache = True)
    assert warm == {"seed": 3407}
    assert "cache_prompt" not in warm


def test_seeded_tool_loop_keeps_prompt_cache_after_the_first_round(monkeypatch):
    """#10698: round 0 stays cold (#9979); later rounds keep the seed and the cache."""
    streams = [
        _structured_tool_call("web_search", {"query": "kernel"}, "call_search"),
        [_sse({"content": "Linux 6.10."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Linux kernel 6.10",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search then answer"}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            seed = 3407,
            max_tool_iterations = 5,
            permission_mode = "off",
        )
    )

    assert len(payloads) >= 2
    assert payloads[0]["seed"] == 3407
    assert payloads[0]["cache_prompt"] is False
    for later in payloads[1:]:
        assert later["seed"] == 3407
        assert "cache_prompt" not in later, later


def test_seeded_zero_tool_iterations_keeps_cold_cache_on_the_final_pass(monkeypatch):
    """max_tool_iterations=0 never sends an in-loop request; the final pass is round 0."""
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [[_sse({"content": "no tools this turn"}), _done()]],
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "just answer"}],
            tools = [],
            seed = 3407,
            max_tool_iterations = 0,
            permission_mode = "off",
        )
    )

    assert len(payloads) == 1
    assert payloads[0]["seed"] == 3407
    assert payloads[0]["cache_prompt"] is False


def test_tool_stream_reports_progress_without_leaking_a_content_event(monkeypatch):
    stream = [
        _progress(processed = 512, cached = 0, time_ms = 64),
        _sse({"content": "done"}),
        _done(),
    ]
    payloads: list[dict] = []
    samples: list[dict] = []
    backend = _make_backend(monkeypatch, [stream], payloads)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "benchmark"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        perf_callback = samples.append,
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
        _tool_call_sse(tool_name, arguments, call_id),
        _done(),
    ]


def test_forced_web_search_tool_choice_is_sent_until_a_tool_runs(monkeypatch):
    """A forced web_search tool_choice is sent until a tool has run; the follow-up turn reverts to auto."""
    first_stream = _structured_tool_call(
        "web_search", {"query": "current Linux kernel version"}, "call_search"
    )
    second_stream = [_sse({"content": "The current version of the Linux kernel is 6.10."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, second_stream])
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


def test_web_search_keeps_to_the_requested_domains(monkeypatch):
    queries = []

    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
            **_kwargs,
        ):
            queries.append(query)
            return [
                {
                    "title": "Coroutines and Tasks",
                    "href": "https://docs.python.org/3/library/asyncio-task.html",
                    "body": "TaskGroup",
                },
                {
                    "title": "TaskGroup?",
                    "href": "https://stackoverflow.com/q/1",
                    "body": "Off-site",
                },
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    backend, _payloads = _backend_and_payloads(
        monkeypatch,
        [
            _structured_tool_call("web_search", {"query": "asyncio TaskGroup"}, "call_search"),
            [_sse({"content": "Done."}), _done()],
        ],
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search the Python docs for asyncio.TaskGroup"}],
        [_web_search_tool()],
        max_tool_iterations = 5,
        permission_mode = "off",
        website_policy = {"allowedDomains": ["docs.python.org"], "blockedDomains": []},
    )

    [result] = [
        event["result"]
        for event in events
        if event.get("type") == "tool_end" and event.get("tool_name") == "web_search"
    ]
    assert queries == ["asyncio TaskGroup (site:docs.python.org)"]
    assert "https://docs.python.org/3/library/asyncio-task.html" in result
    assert "stackoverflow.com" not in result


def test_forced_tool_choice_must_exist_in_the_catalog(monkeypatch):
    backend, payloads = _backend_and_payloads(monkeypatch, [])

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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "Linux kernel result")

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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

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
    """A structured tool call after streamed preface content must still execute, not be dropped."""

    tool_call_id = "call_render_late"
    first_stream = [
        _sse({"content": "Here is the canvas.\n\n"}),
        _tool_call_sse(
            "render_html",
            {"code": "<html><body><div>red</div></body></html>", "title": "Simple Red Square"},
            tool_call_id,
        ),
        _done(),
    ]
    second_stream = [
        _sse({"content": "Done."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, second_stream])

    calls = _record_tool_calls(monkeypatch, "Rendered HTML canvas: Simple Red Square.")

    tools = _render_html_tools()

    events = _run_tool_loop(backend, [{"role": "user", "content": "Make a red square."}], tools)

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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])
    _patch_monotonic(monkeypatch, [100.0, 110.0, 172.0, 172.0])

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "answer"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    content_texts = [e["text"] for e in events if e["type"] == "content"]
    assert content_texts[0] == "<think>I am thinking."
    assert content_texts[1] == "<think>I am thinking. Still thinking."
    assert content_texts[-1] == "<think>I am thinking. Still thinking.</think>Final answer."

    summary_index = next(
        i for i, event in enumerate(events) if event["type"] == "reasoning_summary"
    )
    final_content_index = max(i for i, event in enumerate(events) if event["type"] == "content")
    assert summary_index < final_content_index
    assert events[summary_index]["duration_ms"] == 62000


def test_reasoning_streams_incrementally_with_tools(monkeypatch):
    stream = [
        _sse({"reasoning_content": "Step one."}),
        _sse({"reasoning_content": " Step two."}),
        _sse({"reasoning_content": " Step three."}),
        _sse({"content": "Done."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "think then answer"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    reasoning_stage = [
        e["text"]
        for e in events
        if e["type"] == "content"
        and e["text"].startswith("<think>")
        and "</think>" not in e["text"]
    ]
    assert reasoning_stage == [
        "<think>Step one.",
        "<think>Step one. Step two.",
        "<think>Step one. Step two. Step three.",
    ]
    final = [e["text"] for e in events if e["type"] == "content"][-1]
    assert final == "<think>Step one. Step two. Step three.</think>Done."


def test_reasoning_only_reply_matches_no_tool_path_with_tools(monkeypatch):
    stream = [
        _sse({"reasoning_content": "The capital of France is Paris."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])
    _patch_monotonic(monkeypatch, [1.0, 5.0, 5.0])

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "just think"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    content_texts = [e["text"] for e in events if e["type"] == "content"]
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
        items = _run_tool_loop(
            backend,
            [{"role": "user", "content": "capital of France?"}],
            [{"type": "function", "function": {"name": "web_search"}}],
            promote_reasoning_only = False,
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
    tool_stream = [
        _sse({"reasoning_content": "Let me search."}),
        *_structured_tool_call("web_search", {"query": "weather"}, "call_1"),
    ]
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    tool_start_index = next(i for i, e in enumerate(events) if e["type"] == "tool_start")
    content_before_tool = [e["text"] for e in events[:tool_start_index] if e["type"] == "content"]
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
    # Parity: a reasoning-only reply must reach the client identically with tools on or off.
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

    # Raw yields differ: the tool path emits a duplicate trailing event the route dedupes.
    assert tool_cumulatives[:3] == no_tool_cumulatives[:3]
    tool_out = _replay_route_reasoning_extractor(tool_cumulatives)
    no_tool_out = _replay_route_reasoning_extractor(no_tool_cumulatives)
    assert tool_out == no_tool_out
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
    tool_stream = [
        _sse({"reasoning_content": "Searching now."}),
        _sse({"content": '{"name":"web_search","arguments":{"query":"weather"}}'}),
        _done(),
    ]
    final_stream = [
        _sse({"content": "It is sunny."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])
    _patch_monotonic(monkeypatch, [1.0, 2.0, 3.0, 4.0, 4.0])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "sunny"
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    tool_start_index = next(i for i, e in enumerate(events) if e["type"] == "tool_start")
    content_before_tool = [e["text"] for e in events[:tool_start_index] if e["type"] == "content"]
    assert content_before_tool[0] == "<think>Searching now."
    assert content_before_tool[-1] == "<think>Searching now.</think>"
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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])
    calls = _record_tool_calls(monkeypatch, "sunny")

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
        _tool_call_sse("python", {"code": "print(1)"}, "call_blank_noop"),
        _done(),
    ]
    final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    for partial in ("PARTIAL_TEXT", "PARTIAL_TEXT\n"):
        tool_stream = [_sse({"reasoning_content": "TRACE_ABC"})] + _structured_tool_call(
            "python", {"code": "print(1)"}, "call_continued_noop"
        )
        final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
        backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    """Feedback about tool A must not fold into tool B's result, which templates label with B's name."""
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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    """Folds must not mutate caller message dicts, since /v1/messages passes client history straight in."""
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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])

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
    backend, payloads = _backend_and_payloads(monkeypatch, [tool_stream, final_stream])
    _patch_monotonic(monkeypatch, [200.0, 201.0, 203.0, 300.0, 400.0, 405.0, 410.0])

    def fake_execute_tool(name, arguments, **_kwargs):
        return "Rendered HTML canvas: Done."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "render then answer"}],
        [{"type": "function", "function": {"name": "render_html"}}],
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
        _tool_call_sse(
            "render_html",
            {"code": "<html><body>first</body></html>", "title": "First"},
            "call_first",
        ),
        _done(),
    ]
    repeat_stream = [
        _tool_call_sse(
            "render_html",
            {"code": "<html><body>repeat</body></html>", "title": "Repeat"},
            "call_repeat",
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "Short note."}), _done()]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [first_stream, repeat_stream, final_stream]
    )

    calls = _record_tool_calls(monkeypatch, "Rendered HTML canvas: First.")

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

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "Make a red square."}], tools, max_tool_iterations = 2
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
        _tool_call_sse("render_html", {"code": "<html>ok</html>"}, "call_first"),
        _done(),
    ]
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    def fake_execute_tool(name, arguments, **_kwargs):
        return "Rendered HTML canvas: Done."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Render this."}],
        [{"type": "function", "function": {"name": "render_html"}}],
        max_tool_iterations = 3,
    )

    assert len(payloads) == 2
    # The final pass may not call tools, but re-renders the same catalogue so the cached
    # prompt prefix still matches.
    assert payloads[1]["tool_choice"] == "none"
    assert payloads[1]["tools"] == payloads[0]["tools"]
    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)
    final_user_messages = [
        m.get("content", "") for m in payloads[1]["messages"] if m.get("role") == "user"
    ]
    assert not any("used all available tool calls" in message for message in final_user_messages)


def test_non_consecutive_duplicate_web_search_is_internal_noop(monkeypatch):
    first_search = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_1"),
        _done(),
    ]
    python_call = [
        _tool_call_sse("python", {"code": "print('ok')"}, "call_python"),
        _done(),
    ]
    duplicate_search = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_2"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from gathered data."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, python_call, duplicate_search, final_stream],
        payloads,
    )

    calls = _record_tool_calls(monkeypatch, lambda name: f"ok:{name}")

    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "python"}},
    ]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search gpus in 2026 prices and use python"}],
        tools,
        max_tool_iterations = 3,
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
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_1"),
        _done(),
    ]
    duplicate_search = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_2"),
        _done(),
    ]
    python_call = [
        _tool_call_sse("python", {"code": "print('ok')"}, "call_python"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from gathered data."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, duplicate_search, python_call, final_stream],
        payloads,
    )

    calls = _record_tool_calls(monkeypatch, lambda name: f"ok:{name}")

    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "python"}},
    ]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search gpus in 2026 prices and use python"}],
        tools,
        max_tool_iterations = 4,
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


@pytest.mark.parametrize("deduplicate", [True, False])
def test_rebuild_without_an_edit_runs_again_only_when_deduplication_is_off(
    monkeypatch, deduplicate
):
    # #10379: a second identical build command after no file edit is a duplicate by default.
    build = {"command": "./gradlew build"}
    streams = [
        [_tool_call_sse("terminal", build, "call_build_1"), _done()],
        [_tool_call_sse("terminal", build, "call_build_2"), _done()],
        [_sse({"content": "Built."}), _done()],
    ]
    backend = _make_backend(monkeypatch, streams, [])
    calls = _record_tool_calls(monkeypatch, lambda name: "BUILD SUCCESSFUL")

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "build it twice"}],
        [{"type": "function", "function": {"name": "terminal"}}],
        max_tool_iterations = 3,
        deduplicate_tool_calls = deduplicate,
    )

    assert calls == [("terminal", build)] * (1 if deduplicate else 2)


def test_repeated_duplicate_noop_transitions_to_final_pass(monkeypatch):
    first_search = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_1"),
        _done(),
    ]
    duplicate_one = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_2"),
        _done(),
    ]
    duplicate_two = [
        _tool_call_sse("web_search", {"query": "gpu prices 2026"}, "call_search_3"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer from first search."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [first_search, duplicate_one, duplicate_two, final_stream],
        payloads,
    )
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search gpus"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 10,
    )

    assert calls == [("web_search", {"query": "gpu prices 2026"})]
    assert [event.get("tool_call_id") for event in events if event.get("type") == "tool_end"] == [
        "call_search_1"
    ]
    assert len(payloads) == 4
    assert payloads[-1]["tool_choice"] == "none"
    assert payloads[-1]["tools"] == payloads[-2]["tools"]
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
    backend, payloads = _backend_and_payloads(monkeypatch, [same_turn_duplicates, final_stream])

    calls = _record_tool_calls(monkeypatch, "search-result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search gpus"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
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
    backend, payloads = _backend_and_payloads(monkeypatch, [batch, final_stream])

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append(arguments)
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 3,
    )

    assert calls == [{"query": "a"}, {"query": "b"}]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a1",
        "call_b",
    ]

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
    backend, payloads = _backend_and_payloads(monkeypatch, [same_turn_render_calls, final_stream])

    calls = _record_tool_calls(monkeypatch, "Rendered HTML canvas: One.")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "render html"}],
        [{"type": "function", "function": {"name": "render_html"}}],
        max_tool_iterations = 2,
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
    # No more calls, but the same catalogue keeps the cached prompt prefix.
    assert payloads[1]["tool_choice"] == "none"
    assert payloads[1]["tools"] == payloads[0]["tools"]
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
        _tool_call_sse("python", {"code": "print(1)"}, "call_python_disabled"),
        _done(),
    ]
    final_stream = [_sse({"content": "I cannot run Python here."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [disabled_python, final_stream])

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "run python"}],
        [{"type": "function", "function": {"name": "web_search"}}],
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
    """Once render_html succeeds, its intent text is not re-prompted, since the canvas already exists."""

    first_stream = [
        _tool_call_sse(
            "render_html",
            {"code": "<html><body>first</body></html>", "title": "First"},
            "call_first",
        ),
        _done(),
    ]
    post_tool_stream = [
        _sse({"content": "I will now use render_html again."}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, post_tool_stream])

    calls = _record_tool_calls(monkeypatch, "Rendered HTML canvas: First.")

    tools = _render_html_tools()

    events = _run_tool_loop(backend, [{"role": "user", "content": "Make a red square."}], tools)

    assert len(payloads) == 2
    assert len(calls) == 1
    assert any(
        event.get("type") == "content" and event.get("text") == "I will now use render_html again."
        for event in events
    )


def test_internal_reprompt_attempts_do_not_duplicate_visible_text(monkeypatch):
    """No-tool re-prompt attempts should not concatenate into the UI."""

    streams = [[_sse({"content": "I will use render_html now."}), _done()]]
    streams += [
        [_sse({"content": "Understood. I will use render_html now."}), _done()]
        for _ in range(_MAX_REPROMPTS)
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = _render_html_tools()

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "Make a red square."}], tools, nudge_tool_calls = True
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == 3 < _MAX_REPROMPTS + 1


def test_post_tool_stall_still_nudged_after_a_pre_tool_reprompt(monkeypatch):
    """The post-tool nudge has its own budget, so an earlier stall can't spend it."""

    streams = [
        [_sse({"content": "I will search the web now."}), _done()],
        [
            _tool_call_sse("web_search", {"query": "red square"}, "call_first"),
            _done(),
        ],
        [_sse({"content": "Let me summarize the results."}), _done()],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    calls = _record_tool_calls(monkeypatch, "Search results: red is #f00.")

    tools = [_web_search_tool()]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Make a red square."}],
        tools,
        max_tool_iterations = 2,
        nudge_tool_calls = True,
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
            _tool_call_sse("web_search", {"query": "red square"}, "call_first"),
            _done(),
        ],
        [_sse({"content": "Let me summarize the results."}), _done()],
        [_sse({"content": "Now I will check the sources."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    tools = [_web_search_tool()]

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 2,
            nudge_tool_calls = True,
        )
    )

    assert len(payloads) == 3


def test_browser_turns_get_a_post_tool_nudge_per_step(monkeypatch):
    """A browser task announces many steps; each tool turn earns its own nudge, not one per request."""

    def call(name, arguments, call_id):
        return [_tool_call_sse(name, arguments, call_id), _done()]

    streams = [
        call("browser_snapshot", {}, "call_1"),
        [_sse({"content": "Let me click the Code panel."}), _done()],
        call("browser_click", {"ref": "r4"}, "call_2"),
        [_sse({"content": "Let me take a screenshot to see it."}), _done()],
        call("browser_snapshot", {}, "call_3"),
        [_sse({"content": "The panel shows the CSS."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "Done.")
    tools = [
        {"type": "function", "function": {"name": name}}
        for name in ("browser_snapshot", "browser_click")
    ]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Fix the CSS."}],
        tools,
        max_tool_iterations = 10,
        nudge_tool_calls = True,
    )

    assert len(calls) == 3 and len(payloads) == 6
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == "The panel shows the CSS."


def test_repeat_guard_resets_after_a_tool_runs(monkeypatch):
    """A tool run resets the repeat tracker, so an identical post-tool stall is nudged, not accepted."""

    stall = "I will search the web now."
    streams = [
        [_sse({"content": stall}), _done()],
        [
            _tool_call_sse("web_search", {"query": "red square"}, "call_first"),
            _done(),
        ],
        [_sse({"content": stall}), _done()],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    tools = [_web_search_tool()]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Make a red square."}],
        tools,
        max_tool_iterations = 2,
        nudge_tool_calls = True,
    )

    assert len(payloads) == 4
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts[-1] == "Final answer: the square is red."


def test_restatement_keeps_deletions_that_change_the_answer():
    """A dropped word can invert the meaning, so a subset is not a restatement."""

    from core.inference.tool_call_parser import is_reprompt_restatement

    previous = "Now I think the feature is not supported in version 1."
    corrected = "Now I think the feature is supported in version 1."
    assert not is_reprompt_restatement(corrected, previous)
    assert not suppress(corrected, previous)

    stall = "I'll search for that now."
    assert is_reprompt_restatement(stall, stall)
    assert is_reprompt_restatement("Understood. " + stall, "Understood, " + stall)
    assert not is_reprompt_restatement(stall + " Tokyo.", stall)


def test_forced_turn_suppression_covers_obligation_phrasing():
    for stall in (
        "I need to use render_html now",
        "Need to call web_search",
        "I will summarize the results now",
        "I have to run the search first",
        "I should call web_search now",
        "I should use render_html now",
        "I must call web_search now",
        "I must use render_html now",
        "I must run the search first",
        "Okay. Need to call web_search now.",
        "Understood. Going to search now.",
        "Must call web_search now.",
        "Should search the web now.",
        "I should call web_search because the answer is not in the provided context",
        "I must run the search since the answer is unknown so far",
        "I should call web_search, though.",
        "I need to run the search, but",
        "I need to call web_search to summarize the results",
    ):
        assert suppress(stall), f"leaked {stall!r}"

    for answer in (
        "You need to install the package first.",
        "The square is red.",
        "Here is the summary of what I found.",
        "Run `pip install unsloth` to get started.",
        "I should mention that the square is red.",
        "The API I should invoke is foo() because it supports streaming.",
        "The tool I need to use is documented here.",
        "I should invoke foo() because it supports streaming.",
        "I should query the cache first for a faster path.",
        "You should call your bank about the charge.",
        "You must call your bank about the charge.",
        "I must admit the square is red.",
        "I should call web_search, but the answer is Tokyo.",
        "I need to call web_search. The answer is Tokyo.",
        "I should call web_search to confirm, but Tokyo is the capital of Japan.",
        "I must run the search, however the result is already known: 42.",
    ):
        assert not suppress(answer), f"dropped {answer!r}"


def test_forced_turn_intent_lead_in_needs_a_restatement_to_be_dropped():
    """INTENT_SIGNAL matches real answer lead-ins, so drop a turn only when it restates the nudge."""
    stall = "I will summarize the results now"
    answer = "Now I have the search results. The capital of Japan is Tokyo."

    assert suppress(stall, stall)
    assert suppress("Understood. " + stall, "Understood, " + stall)
    assert not suppress(answer, stall)
    assert not suppress("Step 3: done. Tokyo is the capital.", stall)
    assert not suppress(stall + ": Tokyo.", stall)
    assert suppress("I must call web_search now", answer)


def test_forced_turn_answer_with_an_intent_lead_in_survives_after_a_tool(monkeypatch):
    """After a tool, an answer behind an intent lead-in must still ship, not be dropped as a stall."""

    answer = "Now I have the results. The capital of Japan is Tokyo."
    streams = [
        [
            _tool_call_sse("web_search", {"query": "capital of Japan"}, "call_first"),
            _done(),
        ],
        [_sse({"content": "Let me summarize what I found."}), _done()],
        [_sse({"content": answer}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: Tokyo.",
    )

    tools = [_web_search_tool()]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What is the capital of Japan?"}],
        tools,
        max_tool_iterations = 2,
        nudge_tool_calls = True,
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [_web_search_tool()]

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What is the capital of Japan?"}],
        tools,
        max_tool_iterations = 2,
        nudge_tool_calls = True,
    )

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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = _render_html_tools(),
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = _render_html_tools()

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Make a red square."}],
        tools,
        auto_heal_tool_calls = False,
    )

    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now."]
    assert len(payloads) == 1


def test_internal_reprompt_disabled_when_nudge_tool_calls_false(monkeypatch):
    streams = [[_sse({"content": "I will use render_html now."}), _done()]]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = _render_html_tools()

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Make a red square."}],
        tools,
        auto_heal_tool_calls = True,
        nudge_tool_calls = False,
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        auto_heal_tool_calls = False,
    )

    assert calls == [("web_search", {"query": "x"})]
    assert not any(
        event.get("type") == "content" and "<tool_call>" in event.get("text", "")
        for event in events
    )


def test_textual_mistral_marker_not_leaked_when_inline_with_preface(monkeypatch):
    streams = [
        [_sse({"content": 'Let me search. [TOOL_CALLS]web_search{"query":"cats"}'}), _done()],
        [_sse({"content": "done"}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "cats"})]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("[TOOL_CALLS]" not in t for t in content_texts), content_texts
    assert any("Let me search." in t for t in content_texts)


def test_textual_explicit_id_reuses_provisional_card(monkeypatch):
    big_query = "cats " * 80
    call = "[TOOL_CALLS]" + json.dumps(
        [{"name": "web_search", "arguments": {"query": big_query}, "id": "explicit-42"}]
    )
    assert len(call) > 256
    chunks = [call[i : i + 24] for i in range(0, len(call), 24)]
    streams = [
        [_sse({"content": c}) for c in chunks] + [_done()],
        [_sse({"content": "done"}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": big_query.strip()})]
    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    real = [e for e in tool_starts if e.get("arguments", {}).get("query")]
    assert len(provisional) == 1, tool_starts
    prov_id = provisional[0]["tool_call_id"]
    assert len(real) == 1, tool_starts
    assert real[0]["tool_call_id"] == prov_id
    assert real[0]["tool_name"] == "web_search"
    assert {e["tool_call_id"] for e in tool_starts} == {prov_id}
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert [e["tool_call_id"] for e in ends] == [prov_id]
    assert ends[0]["result"] == "result"


def test_textual_llama_python_tag_marker_not_leaked(monkeypatch):
    streams = [
        [_sse({"content": '<|python_tag|>web_search.call(query="cats")'}), _done()],
        [_sse({"content": "done"}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    calls = _record_tool_calls(monkeypatch, "Rendered HTML canvas: Forced.")

    tools = _render_html_tools()

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "Make a red square."}], tools, nudge_tool_calls = True
    )

    assert len(calls) == 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts == ["I will use render_html now.", "Final note after tool."]
    assert not any(event.get("type") == "reasoning_summary" for event in events)
    assert len(payloads) == 3


def _status_texts(events: list[dict]) -> list[str]:
    return [event["text"] for event in events if event.get("type") == "status"]


_WEB_SEARCH_TOOL = _web_search_tool()


def _nudge_then_search_streams() -> list[list[str]]:
    """Stall, then a re-prompted turn that finally searches, then the answer."""

    return [
        [_sse({"content": "I will search the web now."}), _done()],
        [
            _tool_call_sse("web_search", {"query": "red square"}, "call_search"),
            _done(),
        ],
        [_sse({"content": "Final answer: the square is red."}), _done()],
    ]


def test_plan_without_action_nudge_is_announced_on_the_status_channel(monkeypatch):
    """The re-prompted turn is hidden, so without a badge the UI looks frozen."""

    backend, payloads = _backend_and_payloads(monkeypatch, _nudge_then_search_streams())
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What colour is the square?"}],
        [_WEB_SEARCH_TOOL],
        max_tool_iterations = 2,
        nudge_tool_calls = True,
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What colour is the square?"}],
        [_WEB_SEARCH_TOOL],
        max_tool_iterations = 2,
        nudge_tool_calls = True,
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What colour is the square?"}],
        [_WEB_SEARCH_TOOL],
        max_tool_iterations = 2,
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

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "Balls"}], [_WEB_SEARCH_TOOL], max_tool_iterations = 2
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    assert len(payloads) == 1
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert content_texts and content_texts[-1] == clarification


def test_nudge_status_absent_when_nudging_is_disabled(monkeypatch):
    backend, payloads = _backend_and_payloads(monkeypatch, _nudge_then_search_streams())
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: "Search results: red is #f00.",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What colour is the square?"}],
        [_WEB_SEARCH_TOOL],
        max_tool_iterations = 2,
        nudge_tool_calls = False,
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    assert len(payloads) == 1


def test_nudge_is_off_when_the_request_flag_is_omitted(monkeypatch):
    backend, payloads = _backend_and_payloads(monkeypatch, _nudge_then_search_streams())

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "What colour is the square?"}],
        [_WEB_SEARCH_TOOL],
        max_tool_iterations = 2,
    )

    assert NUDGE_TOOL_CALLS_STATUS not in _status_texts(events)
    assert len(payloads) == 1


def test_confirm_tool_calls_allow_executes_gguf_tool(monkeypatch):
    streams = [
        _structured_tool_call("python", {"code": "print(1)"}, "call_py"),
        [_sse({"content": "Done."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    calls = _record_tool_calls(monkeypatch, "OK")
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
            # Unset defaults to 'auto', which would not prompt this safe print(1).
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

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
        # Unset defaults to 'auto', which would not prompt this safe print(1).
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    def fail_autoinject(*_args, **_kwargs):
        raise AssertionError("RAG autoinject must not run before approval")

    monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fail_autoinject)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "use docs"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            max_tool_iterations = 1,
            confirm_tool_calls = True,
            # 'ask' gates every call so autoinject waits; 'auto' never gates this retrieval.
            permission_mode = "ask",
            session_id = "sess",
            rag_scope = {"thread_id": "t1"},
        )
    )

    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)


def test_rag_autoinject_counts_as_a_prior_tool_execution(monkeypatch):
    """Autoinjected RAG counts as a prior tool execution, so the post-tool re-prompt cap still applies."""

    stall = "I will summarize the retrieved passages now."
    streams = [
        [_sse({"content": stall}), _done()],
        [_sse({"content": "Still working on the summary."}), _done()],
        [_sse({"content": "Final answer: the passages describe Tokyo."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

    monkeypatch.setattr(
        "core.inference.tools.build_rag_autoinject",
        lambda *_a, **_k: {
            "events": [],
            "messages": [{"role": "user", "content": "Retrieved passage: Tokyo."}],
        },
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "summarize the docs"}],
        [{"type": "function", "function": {"name": "search_knowledge_base"}}],
        max_tool_iterations = 2,
        nudge_tool_calls = True,
        rag_scope = {"thread_id": "t1"},
    )

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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

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
            # Unset defaults to 'auto', which would not prompt this safe print(1).
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
    """A large streamed tool-call argument must open a provisional card before its arguments finish."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    args_json = json.dumps({"code": big_code})
    assert len(args_json) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_py_big")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "write code"}],
        [{"type": "function", "function": {"name": "python"}}],
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    real = [e for e in tool_starts if e.get("arguments", {}).get("code")]

    assert len(provisional) == 1, tool_starts
    assert provisional[0]["tool_name"] == "python"
    assert provisional[0]["tool_call_id"] == "call_py_big"
    assert provisional[0]["provenance"].get("provisional") is True
    assert len(real) == 1
    assert real[0]["tool_call_id"] == "call_py_big"
    assert events.index(provisional[0]) < events.index(real[0])

    assert calls == [("python", {"code": big_code})]
    assert any(e.get("type") == "tool_end" and e.get("tool_name") == "python" for e in events)


def test_gated_python_call_still_streams_its_arguments(monkeypatch):
    """Approval-gated calls still stream arguments into the card; the code is what the user approves."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    assert len(json.dumps({"code": big_code})) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_gated")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda name, arguments, **_k: "OK")
    monkeypatch.setattr("core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: "allow")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "write code"}],
        [{"type": "function", "function": {"name": "python"}}],
        confirm_tool_calls = True,
        permission_mode = "ask",
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    assert len(provisional) == 1, tool_starts
    assert provisional[0]["tool_call_id"] == "call_gated"

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "gated call streamed no arguments"
    assert "total += 119" in "".join(e["text"] for e in args_events)

    gated = [e for e in tool_starts if e.get("awaiting_confirmation")]
    assert gated, tool_starts
    assert events.index(provisional[0]) < events.index(gated[0])


@pytest.mark.parametrize("verdict", ["allow", "deny"])
def test_only_an_approved_call_is_marked_approved(monkeypatch, verdict):
    first_stream = _streamed_structured_tool_call("python", {"code": "print(1)"}, "call_gated")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, _payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])
    seen: list = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **kwargs: seen.append(kwargs.get("host_access_approved")) or "OK",
    )
    monkeypatch.setattr("core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: verdict)

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "run it"}],
        [{"type": "function", "function": {"name": "python"}}],
        confirm_tool_calls = True,
        permission_mode = "ask",
    )

    assert seen == ([True] if verdict == "allow" else [])


@pytest.mark.parametrize("verdict", ["allow", "deny"])
def test_an_allowed_call_tells_every_follower_before_tool_end(monkeypatch, verdict):
    """Only the client that answered clears its own prompt; other tabs learn it from tool_approved."""
    first_stream = _streamed_structured_tool_call("python", {"code": "print(1)"}, "call_gated")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, _payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda name, arguments, **_k: "OK")
    monkeypatch.setattr("core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: verdict)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "run it"}],
        [{"type": "function", "function": {"name": "python"}}],
        confirm_tool_calls = True,
        permission_mode = "ask",
    )

    gated = [e for e in events if e.get("type") == "tool_start" and e.get("awaiting_confirmation")]
    approved = [e for e in events if e.get("type") == "tool_approved"]
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert len(gated) == 1 and len(ends) == 1, events
    if verdict == "deny":
        # A refusal ends the card at once; there is no run to wait through.
        assert approved == []
        return
    assert approved == [
        {
            "type": "tool_approved",
            "tool_name": "python",
            "tool_call_id": gated[0]["tool_call_id"],
            "approval_id": gated[0]["approval_id"],
        }
    ]
    assert events.index(gated[0]) < events.index(approved[0]) < events.index(ends[0])
    # Before the running status: the route waits for the model slot on that frame, and other tabs must not.
    between = events[events.index(gated[0]) + 1 : events.index(approved[0])]
    assert [e for e in between if e.get("type") == "status"] == [], between


def test_auto_mode_render_html_suppresses_provisional_card_under_confirm(monkeypatch):
    """Confirm under permission_mode=auto hides the render_html provisional card; full tool_start fires."""
    args = {"code": "<html>" + "x" * 80 + "</html>"}
    first_stream = _streamed_structured_tool_call("render_html", args, "call_rh")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda name, arguments, **_k: "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "make a card"}],
        [{"type": "function", "function": {"name": "render_html"}}],
        confirm_tool_calls = True,
        permission_mode = "auto",
    )

    tool_starts = [e for e in events if e.get("type") == "tool_start"]
    provisional = [e for e in tool_starts if not e.get("arguments")]
    assert provisional == [], tool_starts
    real = [e for e in tool_starts if e.get("arguments")]
    assert real and real[0]["tool_name"] == "render_html"
    assert real[0].get("awaiting_confirmation") in (False, None)


def test_small_python_tool_call_has_no_provisional_start(monkeypatch):
    """A small tool-call argument finishes streaming instantly, so it keeps the
    existing behavior of a single (real) tool_start with no provisional card."""

    first_stream = _structured_tool_call("python", {"code": "print(1)"}, "call_py_small")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "python"}}],
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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "OK")

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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "OK")

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
    assert calls == [("python", {"code": big_code})]
    closing = [
        e for e in events if e.get("type") == "tool_end" and e.get("tool_call_id") == "call_py"
    ]
    assert len(closing) == 1


def test_connect_error_during_tool_call_closes_provisional_card(monkeypatch):
    """If llama-server drops mid tool-call after a provisional card is shown, the
    loop must close that card before surfacing the error so the UI never leaves a
    tool spinning forever."""
    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    fragments = _streamed_structured_tool_call("python", {"code": big_code}, "call_py_err")
    fragments = fragments[:-1]

    def raising_stream():
        for chunk in fragments:
            yield chunk
        raise httpx.ConnectError("connection lost mid stream")

    backend, payloads = _backend_and_payloads(monkeypatch, [raising_stream()])
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
    closing = [
        e
        for e in collected
        if e.get("type") == "tool_end" and e.get("tool_call_id") == "call_py_err"
    ]
    assert len(closing) == 1
    assert "Error" in (closing[0].get("result") or "")
    assert respawn_calls == []


def test_connect_error_before_tool_stream_respawns_and_retries(monkeypatch):
    """A dead server before the first tool-loop response is opened is safe to retry."""
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "hello"}],
        [{"type": "function", "function": {"name": "python"}}],
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
        # Each window overflows by about one turn-group so both preflight paths must refit.
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
    backend, payloads = _backend_and_payloads(monkeypatch, [[_sse({"content": "OK"}), _done()]])
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
    backend, payloads = _backend_and_payloads(monkeypatch, [[_sse({"content": "OK"}), _done()]])
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
        assert refit["fits"] is False
        assert refit["dropped_messages"] == 2
        assert refit["prompt_tokens_after"] == 900 < refit["prompt_tokens_before"]
        # 4, not dropped_messages' 2: the boundary counts the request's own leading messages.
        assert refit["boundary_messages"] == 4
        assert "boundary_anchor" in refit
        assert archived[-1] == (3, 1)
        assert len(payloads[1]["messages"]) == 1


def test_tool_loop_retries_preflight_when_counting_failed_on_the_dead_server(monkeypatch):
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
    for max_tool_iterations in (1, 0):
        backend, payloads = _backend_and_payloads(
            monkeypatch, [httpx.ConnectError("server is down")]
        )
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
    backend, payloads = _backend_and_payloads(monkeypatch, [[_done()]])
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
    """Respawn refits evict more turns against a smaller window, so they must archive what they evict."""
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

    assert len(archived) >= 2
    assert any(batch for batch in archived[1:])


def test_the_respawn_retry_keeps_the_thread(monkeypatch):
    """The respawn retry must keep the archiving thread, since it refits and can evict more turns."""
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

    assert len(seen) == 2
    assert seen == ["t-respawn", "t-respawn"]


def test_rolling_respawn_retry_refits_when_the_effective_context_changes(monkeypatch):
    """A smaller replacement window can evict more without repeating the first eviction."""
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [httpx.ConnectError("server is down"), [_sse({"content": "OK"}), _done()]],
        payloads,
    )
    # Each window overflows by about one turn-group so both preflight paths must refit.
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

        events = _run_tool_loop(
            backend,
            [{"role": "user", "content": "print one"}],
            [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = max_tool_iterations,
        )

        assert respawn_calls == [True]
        assert tool_calls == [("python", {"code": "print(1)"})]
        assert len(payloads) == 3
        assert payloads[1] == payloads[2]
        assert any(e.get("type") == "content" and e.get("text") == final_text for e in events)


def test_connect_error_retry_is_bounded(monkeypatch):
    """A failed retry surfaces the error without another respawn attempt."""
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

        events = _run_tool_loop(
            backend,
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "python"}}],
        )

        assert respawn_calls == [True], type(exc).__name__
        assert len(payloads) == 2, type(exc).__name__
        assert any(e.get("type") == "content" and e.get("text") == "Recovered." for e in events)


def test_a_not_yet_reaped_child_does_not_burn_the_retry(monkeypatch):
    """A closing server can beat its own exit status, so poll() briefly reports it
    alive. Without a grace wait _respawn_if_dead hands back the stale _healthy and the
    single retry is spent on the corpse rather than on a replacement."""

    class _Dying:
        # Reapable only from the 4th poll, mimicking teardown lagging the socket close.
        def __init__(self):
            self.polls = 0
            self.returncode = None

        def poll(self):
            self.polls += 1
            if self.polls > 3:
                self.returncode = -9
                return -9
            return None

    backend, payloads = _backend_and_payloads(monkeypatch, [])
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "hello"}],
        [{"type": "function", "function": {"name": "python"}}],
    )

    assert len(loads) == 1
    assert any(e.get("type") == "content" and e.get("text") == "Recovered." for e in events)


def test_prefill_timeout_is_not_retried(monkeypatch):
    """A slow-but-alive server must not have its first-token budget spent twice."""
    for exc in (httpx.ReadTimeout("no first token"), httpx.PoolTimeout("pool")):
        backend, payloads = _backend_and_payloads(monkeypatch, [exc])
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
    for max_tool_iterations in (2, 1):
        backend, payloads = _backend_and_payloads(monkeypatch, [httpx.ConnectError("mtp crash")])
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
    """An empty tool-call id must emit no provisional card, since it can never reconcile with tool_start."""

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    assert len(json.dumps({"code": big_code})) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "write code"}],
        [{"type": "function", "function": {"name": "python"}}],
    )

    provisional = [e for e in events if e.get("type") == "tool_start" and not e.get("arguments")]
    assert provisional == []
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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "Weather: sunny, 22C.")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather in Sydney?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "weather in Sydney"})]
    assert any(
        event.get("type") == "tool_end" and event.get("tool_name") == "web_search"
        for event in events
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all('"name"' not in t for t in content_texts), content_texts
    assert all("web_search" not in t for t in content_texts), content_texts
    assert any("sunny in Sydney" in t for t in content_texts), content_texts


def test_ordinary_json_with_name_key_is_shown_not_treated_as_tool_call(monkeypatch):
    """Markerless JSON with a non-enabled name is the answer, not a phantom call."""

    answer = '{"name": "Alice", "parameters": {"age": 30}}'
    first_stream = _streamed_content(answer)
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: calls.append((n, a)) or "x",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "give me a person record"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_incomplete_bare_json_truncation_is_not_leaked(monkeypatch):
    """If generation is cut off mid bare-JSON object (no closing brace), the held
    fragment must be stripped at stream end rather than dumped to the user."""

    truncated = '{"name": "web_search", "parameters": {"query": "weather in S'
    stream = _streamed_content(truncated)
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no complete call")),
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all('{"name"' not in t for t in content_texts), content_texts


def test_gguf_truncated_ordinary_json_with_name_key_is_shown_not_suppressed(monkeypatch):
    """Truncated JSON whose name is not an enabled tool must still be shown, not drained as a call."""

    truncated = '{"name": "Alice", "age": 30, "bio": "loves '
    stream = _streamed_content(truncated)
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: calls.append((n, a)) or "x",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "start a person record"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_gguf_truncated_disabled_name_json_is_preserved_when_tools_active(monkeypatch):
    """A truncated JSON answer with a non-enabled name must still be shown (resolvers are gated on enabled names)."""

    truncated = '{"name": "Alice", "parameters": {"age": 30'
    stream = _streamed_content(truncated)
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: calls.append((n, a)) or "x",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "give json"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts


def test_gguf_truncated_enabled_name_json_is_still_suppressed(monkeypatch):
    """Counterpart guard: a truncated ENABLED-tool bare call (``web_search``) cut off
    mid-JSON still must NOT leak -- the gate only spares disabled / non-tool names."""

    truncated = '{"name": "web_search", "parameters": {"query": "weather in S'
    stream = _streamed_content(truncated)
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no complete call")),
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all('{"name"' not in t for t in content_texts), content_texts


def test_gguf_oversized_disabled_name_json_is_preserved(monkeypatch):
    """An oversized still-open JSON answer with a non-enabled name streams as content, not a phantom drain."""

    cap = 16384
    big = "A" * (cap + 5000)
    answer = '{"name":"Alice","parameters":{"bio":"' + big
    first_stream = [_sse({"content": answer[i : i + 2000]}) for i in range(0, len(answer), 2000)]
    first_stream.append(_done())
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: calls.append((n, a)) or "x",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "long json"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("Alice" in t for t in content_texts), content_texts[:1]


def test_gemma_wrapperless_call_streamed_is_not_leaked_and_executes(monkeypatch):
    """Wrapper-less Gemma call:NAME{..} must be drained via _GEMMA_BARE_TC_RE and run, never leaked."""

    gemma_call = 'call:web_search{query:"weather in Sydney"}'
    first_stream = _streamed_content(gemma_call)
    final_stream = [_sse({"content": "It is sunny in Sydney."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "Weather: sunny, 22C.")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather in Sydney?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "weather in Sydney"})]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("call:" not in t for t in content_texts), content_texts
    assert any("sunny in Sydney" in t for t in content_texts), content_texts


def test_gemma_wrapperless_execution_call_opens_no_provisional_card(monkeypatch):
    """A bare call:terminal{..} is prose, so it must not open a live terminal card that never runs."""

    payload = "echo hi; " * 40
    gemma_call = f'call:terminal{{command:"{payload}"}}'
    assert len(gemma_call) >= 256
    backend = _make_backend(monkeypatch, [_streamed_content(gemma_call)], [])

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "uid=0(root)"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "summarise this page"}],
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "terminal"}},
            ],
            max_tool_iterations = 1,
        )
    )

    assert calls == []
    assert not any(e.get("type") in ("tool_start", "tool_end") for e in events), [
        e for e in events if e.get("type", "").startswith("tool")
    ]
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any("call:terminal{command:" in t for t in content_texts), content_texts


def test_gemma_wrapperless_execution_call_streams_instead_of_draining(monkeypatch):
    """The leading ``call:NAME{`` drain runs before the parser, so a name the parser will
    not promote has to keep streaming rather than hold the turn until EOS."""

    gemma_call = 'call:terminal{command:"id"}'
    tail = " is what the page suggested; I did not run it."
    backend = _make_backend(monkeypatch, [_streamed_content(gemma_call + tail)], [])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "summarise this page"}],
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "terminal"}},
            ],
            max_tool_iterations = 1,
        )
    )

    assert calls == []
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert gemma_call in content_texts[-1], content_texts
    assert any(gemma_call in t and "did not run" not in t for t in content_texts), content_texts


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
    """Keep prompt_tokens_details in _build_metadata_event so cached_tokens reports real KV-cache hits."""
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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    events = _run_tool_loop(backend, [{"role": "user", "content": "hi"}], [])

    metadata = [e for e in events if e.get("type") == "metadata"]
    assert metadata, "expected a metadata event"
    usage = metadata[-1]["usage"]
    assert usage["prompt_tokens_details"] == {"cached_tokens": 16}
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 4


def test_metadata_event_omits_context_tokens_on_a_single_pass(monkeypatch):
    stream = [
        _sse({"content": "hi"}),
        _usage_done({"prompt_tokens": 5, "completion_tokens": 2}),
        _done(),
    ]
    backend, _ = _backend_and_payloads(monkeypatch, [stream])

    events = _run_tool_loop(backend, [{"role": "user", "content": "hi"}], [_web_search_tool()])

    usage = [e for e in events if e.get("type") == "metadata"][-1]["usage"]
    assert usage["total_tokens"] == 7
    assert "context_tokens" not in usage


def test_metadata_event_omits_prompt_tokens_details_when_absent(monkeypatch):
    """No KV-cache block from the server -> the key isn't fabricated, so the
    route falls back to its 0-default instead of reading a bogus value."""
    stream = [
        _sse({"content": "hi"}),
        _usage_done({"prompt_tokens": 5, "completion_tokens": 2}),
        _done(),
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    events = _run_tool_loop(backend, [{"role": "user", "content": "hi"}], [])

    metadata = [e for e in events if e.get("type") == "metadata"]
    assert metadata, "expected a metadata event"
    assert "prompt_tokens_details" not in metadata[-1]["usage"]


def test_metadata_event_context_tokens_count_earlier_passes_once(monkeypatch):
    """A tool pass's completion is re-sent inside the next pass's prompt, so the
    context the turn leaves is the final prompt plus the final completion only;
    total_tokens keeps billing every pass's completion."""
    streams = [
        [
            _tool_call_sse("web_search", {"query": "cats"}, "call_1"),
            _usage_done({"prompt_tokens": 100, "completion_tokens": 30}, "tool_calls"),
            _done(),
        ],
        [
            _sse({"content": "Found cats."}),
            _usage_done({"prompt_tokens": 140, "completion_tokens": 10}),
            _done(),
        ],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats"}],
        [_web_search_tool()],
        max_tool_iterations = 2,
    )

    assert len(payloads) == 2
    usage = [e for e in events if e.get("type") == "metadata"][-1]["usage"]
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 180
    assert usage["context_tokens"] == 150


def test_context_usage_frame_precedes_each_tool_run(monkeypatch):
    """#12624: the meter refreshes when a pass hands off to a tool, from the usage
    llama-server already reported, before the tool runs; none on the final pass."""
    streams = [
        [
            _tool_call_sse("web_search", {"query": "cats"}, "call_1"),
            _usage_done(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
                "tool_calls",
            ),
            _done(),
        ],
        [
            _tool_call_sse("web_search", {"query": "dogs"}, "call_2"),
            _usage_done({"prompt_tokens": 140, "completion_tokens": 20}, "tool_calls"),
            _done(),
        ],
        [
            _sse({"content": "Found both."}),
            _usage_done({"prompt_tokens": 170, "completion_tokens": 10}),
            _done(),
        ],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats then dogs"}],
        [_web_search_tool()],
        max_tool_iterations = 3,
    )

    assert len(payloads) == 3
    frames = [e for e in events if e.get("type") == "context_usage"]
    assert frames == [
        {
            "type": "context_usage",
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "context_tokens": 130,
            "cached_tokens": 64,
        },
        {
            "type": "context_usage",
            "prompt_tokens": 140,
            "completion_tokens": 20,
            "context_tokens": 160,
            "cached_tokens": 0,
        },
    ]
    # Each frame lands once its pass's tool has started and before it finishes.
    order = [
        e["type"] for e in events if e.get("type") in ("context_usage", "tool_start", "tool_end")
    ]
    assert order == ["tool_start", "context_usage", "tool_end"] * 2, order
    assert [e for e in events if e.get("type") == "metadata"][-1]["usage"]["context_tokens"] == 180


def test_context_usage_frame_never_delays_the_tool(monkeypatch):
    """The frame goes out while the tool is already executing, so the tool never waits on the
    consumer: hold the generator at the frame and the tool still starts."""
    streams = [
        [
            _tool_call_sse("web_search", {"query": "cats"}, "call_1"),
            _usage_done({"prompt_tokens": 100, "completion_tokens": 30}, "tool_calls"),
            _done(),
        ],
        [
            _sse({"content": "Found cats."}),
            _usage_done({"prompt_tokens": 140, "completion_tokens": 10}),
            _done(),
        ],
    ]
    backend, _ = _backend_and_payloads(monkeypatch, streams)
    # No window: the result budget is priced through a template render the fake server lacks.
    backend._effective_context_length = None
    tool_started = threading.Event()
    release_tool = threading.Event()

    def fake_execute_tool(name, arguments, **_kwargs):
        tool_started.set()
        release_tool.wait(timeout = 10)
        return "result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    gen = backend.generate_chat_completion_with_tools(
        messages = [{"role": "user", "content": "search cats"}],
        tools = [_web_search_tool()],
        max_tool_iterations = 2,
    )
    for event in gen:
        if event.get("type") == "context_usage":
            break
    # The consumer has not pulled again, yet the tool is already running.
    assert tool_started.wait(timeout = 5)
    release_tool.set()
    rest = list(gen)
    assert [e["type"] for e in rest if e.get("type") == "tool_end"] == ["tool_end"]


def test_context_usage_frame_absent_on_a_single_pass(monkeypatch):
    stream = [
        _sse({"content": "hi"}),
        _usage_done({"prompt_tokens": 5, "completion_tokens": 2}),
        _done(),
    ]
    backend, _ = _backend_and_payloads(monkeypatch, [stream])

    events = _run_tool_loop(backend, [{"role": "user", "content": "hi"}], [_web_search_tool()])

    assert not [e for e in events if e.get("type") == "context_usage"]


def test_context_usage_event_reads_timings_when_usage_is_missing():
    """timings.prompt_n leaves out the KV-cache hit that usage.prompt_tokens includes."""
    from core.inference.llama_cpp import _context_usage_event

    assert _context_usage_event(None, {"prompt_n": 12, "cache_n": 88, "predicted_n": 7}) == {
        "type": "context_usage",
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "context_tokens": 107,
        "cached_tokens": 88,
    }
    assert _context_usage_event(None, None) is None
    assert _context_usage_event({"prompt_tokens": 0}, {}) is None


def test_gguf_rehearsal_name_split_before_args_is_not_leaked(monkeypatch):
    """Hold a bare rehearsal name until [ARGS] arrives, or the tool name streams as visible text."""

    first_stream = [
        _sse({"content": "web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all("[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_split_bare_json_chain_is_owned_before_later_call_executes(monkeypatch):
    """A blocked first object must not stream before a later chain peer arrives."""
    blocked = '{"name":"terminal","parameters":{"command":"id"}}'
    later = '{"name":"web_search","parameters":{"query":"cats"}}'
    first_stream = [
        _sse({"content": blocked}),
        _sse({"content": ";" + later}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: calls.append((name, arguments)) or "result",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search cats"}],
            tools = [
                {"type": "function", "function": {"name": "terminal"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            max_tool_iterations = 1,
        )
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert not any(blocked in text or later in text for text in content_texts), content_texts


def test_gguf_lone_blocked_bare_json_is_released_at_eof(monkeypatch):
    """A blocked object is content when no executable chain peer follows it."""
    blocked = '{"name":"terminal","parameters":{"command":"id"}}'
    backend = _make_backend(
        monkeypatch,
        [[_sse({"content": blocked}), _done()]],
        [],
    )

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: calls.append((name, arguments)) or "result",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "show the example"}],
            tools = [{"type": "function", "function": {"name": "terminal"}}],
            max_tool_iterations = 1,
        )
    )

    assert calls == []
    content_texts = [event.get("text", "") for event in events if event.get("type") == "content"]
    assert any(blocked in text for text in content_texts), content_texts


def test_gguf_initial_buffer_flush_holds_split_rehearsal_name(monkeypatch):
    """The initial BUFFERING flush needs the same trailing-name hold as STREAMING, or the name leaks."""

    first_stream = [
        _sse({"content": "I will use web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert all("web_search" not in t for t in content_texts), content_texts
    assert all("[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_rehearsal_name_after_prose_in_streaming_is_not_leaked(monkeypatch):
    """The rehearsal hold also applies in STREAMING state, not just at the turn start."""

    first_stream = [
        _sse({"content": "Let me think. "}),
        _sse({"content": "I will search "}),
        _sse({"content": "web_search"}),
        _sse({"content": '[ARGS]{"query":"cats"}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found cats."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats"}],
        [{"type": "function", "function": {"name": "web_search"}}],
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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "result",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "advise"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any(t.rstrip().endswith("web_search") for t in content_texts), content_texts


def test_gguf_long_tool_name_split_rehearsal_is_not_capped_and_executes(monkeypatch):
    """Finding 11: an explicitly marked MCP name longer than the 32-char buffer cap,
    split before [ARGS]{...}, must still be held, so the name does not leak and the
    call executes."""
    name = "mcp__github__create_pull_request"
    assert len(name) >= 32, len(name)

    first_stream = [
        _sse({"content": "[TOOL_CALLS]" + name}),
        _sse({"content": '[ARGS]{"x":1}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda n, a, **_k: calls.append((n, a)) or "result",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "go"}],
        [{"type": "function", "function": {"name": name}}],
    )

    assert calls == [(name, {"x": 1})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert not any(name in t for t in content_texts), content_texts


def test_gguf_streaming_keeps_bare_args_before_think_block(monkeypatch):
    """Run the [ARGS] tail cleanup only on the last segment; a bare foo[ARGS] before <think> is prose."""

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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert content_texts, events
    assert content_texts[-1] == "Please pass foo[ARGS] <think>pause</think> to the template."


def test_gguf_inactive_name_args_in_prose_is_not_drained(monkeypatch):
    """An inactive-name foo[ARGS]{...} in prose must not be drained or forced into another turn."""
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
    )

    assert calls == [], calls
    assert not any(e.get("type") in ("tool_start", "tool_end") for e in events), events
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search cats"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "cats"})], calls
    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert any('foo[ARGS]{"a":1}' in t for t in content_texts), content_texts
    assert all("web_search[ARGS]" not in t for t in content_texts), content_texts


def test_gguf_rehearsal_detection_recognises_spent_one_shot_with_original_tools():
    from core.inference.llama_cpp import _gguf_has_genuine_tool_signal
    from core.inference.tool_call_parser import TOOL_XML_SIGNALS

    repeat = 'render_html[ARGS]{"code":"<html>x</html>"}'
    active_only = [{"type": "function", "function": {"name": "web_search"}}]
    original = active_only + [{"type": "function", "function": {"name": "render_html"}}]
    assert not _gguf_has_genuine_tool_signal(repeat, TOOL_XML_SIGNALS, active_only)
    assert _gguf_has_genuine_tool_signal(repeat, TOOL_XML_SIGNALS, original)


def test_gguf_rehearsal_prefix_and_tail_hold_recognise_spent_one_shot():
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
    full = '{"name":"web_search","parameters":{"query":"' + big + '"}}'
    first_stream = [_sse({"content": full[i : i + 2000]}) for i in range(0, len(full), 2000)]
    first_stream.append(_done())
    final_stream = [_sse({"content": "done"}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "OK",
    )

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "run"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    content_texts = [e.get("text", "") for e in events if e.get("type") == "content"]
    assert not any(t.lstrip().startswith('{"name') for t in content_texts), content_texts[:1]
    assert calls and calls[0][0] == "web_search"
    assert len(calls[0][1].get("query", "")) > cap


def test_gguf_bare_json_call_not_replayed_in_next_turn_content(monkeypatch):
    """After a bare-JSON call executes, the kept assistant message must not carry the raw call as content."""

    import copy

    first_stream = [
        _sse({"content": '{"name":"web_search","parameters":{"query":"cats"}}'}),
        _done(),
    ]
    final_stream = [_sse({"content": "Found."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

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
    """Textual fallback calls are capped at _MAX_TOOL_CALLS_PER_TURN; llama-server bounds structured
    ones."""
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN

    n = _MAX_TOOL_CALLS_PER_TURN + 4
    blocks = "".join(
        '<tool_call>{"name":"t%d","arguments":{"i":%d}}</tool_call>' % (i, i) for i in range(n)
    )
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "OK",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(n)],
            max_tool_iterations = 2,
        )
    )

    assert len(calls) == _MAX_TOOL_CALLS_PER_TURN, [c[0] for c in calls]
    assert [c[0] for c in calls] == [f"t{i}" for i in range(_MAX_TOOL_CALLS_PER_TURN)]
    (notice,) = [
        m for m in payloads[1]["messages"] if "more tool call(s)" in (m.get("content") or "")
    ]
    assert notice["role"] == "user"
    assert notice["content"].startswith("4 more tool call(s)")
    for i in range(_MAX_TOOL_CALLS_PER_TURN, n):
        assert f"t{i} " in notice["content"]


def test_gguf_textual_fallback_over_cap_notice_is_not_folded_into_tool_result(monkeypatch):
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN

    def _call(i):
        return '<tool_call>{"name":"web_search","arguments":{"query":"q%d"}}</tool_call>' % i

    streams = [
        [_sse({"content": _call(0)}), _done()],
        [
            _sse({"content": "".join(_call(i) for i in range(_MAX_TOOL_CALLS_PER_TURN + 2))}),
            _done(),
        ],
        [_sse({"content": "done"}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "OK")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 3,
        )
    )

    last = payloads[2]["messages"][-1]
    assert last["role"] == "user"
    assert "2 more tool call(s)" in last["content"]
    assert not any(
        "more tool call(s)" in (m.get("content") or "")
        for m in payloads[2]["messages"]
        if m.get("role") == "tool"
    )


def test_gguf_textual_fallback_over_cap_on_last_turn_does_not_ask_for_retry(monkeypatch):
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN
    from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE

    blocks = "".join(
        '<tool_call>{"name":"web_search","arguments":{"query":"q%d"}}</tool_call>' % i
        for i in range(_MAX_TOOL_CALLS_PER_TURN + 2)
    )
    streams = [[_sse({"content": blocks}), _done()], [_sse({"content": "done"}), _done()]]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "OK")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    last = messages[-1]
    assert last["role"] == "tool"
    assert "2 more tool call(s)" in last["content"]
    assert BUDGET_EXHAUSTED_NUDGE in last["content"]
    assert not any("Call them again" in (m.get("content") or "") for m in messages)
    assert not any(
        a.get("role") == "user" and b.get("role") == "user" for a, b in zip(messages, messages[1:])
    )


def test_gguf_over_cap_notice_is_not_folded_into_an_unrelated_tools_result(monkeypatch):
    """Final-turn notices must not fold into an unrelated result, since it is labelled with that tool."""
    n = 12
    blocks = "".join(
        '<tool_call>{"name":"t%d","arguments":{"x":"t%d"}}</tool_call>' % (i, i) for i in range(n)
    )
    streams = [[_sse({"content": blocks}), _done()], [_sse({"content": "done"}), _done()]]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "OK")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": "t%d" % i}} for i in range(n)],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    holders = [m for m in messages if "more tool call(s)" in (m.get("content") or "")]
    assert len(holders) == 1
    holder = holders[0]
    skipped = {"t8", "t9", "t10", "t11"}
    if holder["role"] == "tool":
        assert holder.get("name") in skipped, "a note about %s must not sit inside %r's result" % (
            sorted(skipped),
            holder.get("name"),
        )


def test_gguf_over_cap_notice_for_several_tools_is_not_folded_into_one_tools_result(monkeypatch):
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN
    from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE

    blocks = "".join(
        '<tool_call>{"name":"web_search","arguments":{"query":"q%d"}}</tool_call>' % i
        for i in range(_MAX_TOOL_CALLS_PER_TURN + 1)
    )
    blocks += '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'
    streams = [[_sse({"content": blocks}), _done()], [_sse({"content": "done"}), _done()]]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "OK")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "python"}},
            ],
            max_tool_iterations = 1,
        )
    )

    messages = payloads[1]["messages"]
    (holder,) = [m for m in messages if "more tool call(s)" in (m.get("content") or "")]
    assert holder["role"] == "user"
    assert "python" in holder["content"] and '"q8"' in holder["content"]
    assert holder == messages[-1]
    assert BUDGET_EXHAUSTED_NUDGE in holder["content"]
    assert not any(
        a.get("role") == "user" and b.get("role") == "user" for a, b in zip(messages, messages[1:])
    )


def test_gguf_over_cap_does_not_ask_for_a_retry_when_the_range_check_ends_the_loop(monkeypatch):
    """Range-exit must not ask for a retry: no-op turns end the loop before the tool-cap ever trips."""

    def _call(q):
        return '<tool_call>{"name":"web_search","arguments":{"query":"%s"}}</tool_call>' % q

    # No-op turns burn the range without raising the executed-tool count; distinct keys
    # avoid the duplicate limit.
    streams = [[_sse({"content": "".join(_call("q%d" % i) for i in range(6))}), _done()]]
    streams += [[_sse({"content": _call("q%d" % i)}), _done()] for i in range(5)]
    streams.append([_sse({"content": "".join(_call("z%d" % i) for i in range(10))}), _done()])
    streams += [[_sse({"content": "done"}), _done()] for _ in range(6)]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    _record_tool_calls(monkeypatch, "OK")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            max_tool_iterations = 3,
        )
    )

    # Assert the scenario really happened rather than letting the check pass vacuously: the
    # loop ended without offering tools again, and it did produce exactly one notice.
    assert payloads[-1]["tool_choice"] == "none"
    messages = payloads[-1]["messages"]
    notices = [m for m in messages if "more tool call(s)" in (m.get("content") or "")]
    assert len(notices) == 1
    assert "Call them again" not in notices[0]["content"]


def test_gguf_textual_fallback_collapses_duplicate_tool_calls(monkeypatch):
    """Exact-duplicate textual calls in one turn collapse to a single execution."""
    blocks = '<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>' * 5
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

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


@pytest.mark.parametrize("deduplicate", [True, False])
def test_gguf_textual_fallback_keeps_identical_calls_when_deduplication_is_off(
    monkeypatch, deduplicate
):
    blocks = '<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>' * 3
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, _payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

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
            deduplicate_tool_calls = deduplicate,
        )
    )

    assert len(calls) == (1 if deduplicate else 3), [c[0] for c in calls]


@pytest.mark.parametrize(
    "name, arguments",
    [("browser_scroll", '{"direction":"down"}'), ("generate_image", '{"prompt":"a sloth"}')],
)
def test_gguf_identical_browser_calls_in_one_batch_all_run(monkeypatch, name, arguments):
    # Each acts on the page as it is then, so the second scroll is not a repeat of the first;
    # a second identical generate_image is a second image.
    blocks = f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>' * 2
    first_stream = [_sse({"content": blocks}), _done()]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, _payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: calls.append((name, arguments)) or "Done.",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "scroll"}],
            tools = [{"type": "function", "function": {"name": name}}],
            max_tool_iterations = 1,
        )
    )

    assert [called for called, _ in calls] == [name, name]


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
        events = _run_tool_loop(
            backend,
            [{"role": "user", "content": "x"}],
            [{"type": "function", "function": {"name": "web_search"}}],
            auto_heal_tool_calls = auto_heal,
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
    streams = [
        _structured_tool_call("web_search", {"query": f"q{i}"}, f"call_{i}") for i in range(6)
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)

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

    assert len(calls) == 2, calls
    assert len(payloads) == 3, len(payloads)
    # The final pass may not call tools (it keeps the schemas for the cache). Its controller feedback stays in
    # the latest tool result rather than opening a newer user turn.
    assert payloads[2]["tool_choice"] == "none", payloads[2].get("tool_choice")
    assert any(
        m.get("role") == "tool" and "used all available tool calls" in m.get("content", "")
        for m in payloads[2]["messages"]
    ), payloads[2]["messages"]


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
    """Stream a large structured call's arguments as tool_args events to its provisional card."""

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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, second_stream])

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(backend, [{"role": "user", "content": "run it"}], _python_tool_schema())

    starts = [e for e in events if e.get("type") == "tool_start"]
    assert starts and starts[0]["tool_call_id"] == call_id

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "no tool_args events were streamed"
    assert all(e["tool_call_id"] == call_id for e in args_events)
    assert args_events[0]["text"] == frag1
    assert "".join(e["text"] for e in args_events) == args_json

    assert executed == [("python", {"code": code})]
    assistant_messages = [m for m in payloads[1]["messages"] if m.get("role") == "assistant"]
    tc = assistant_messages[-1]["tool_calls"][0]
    assert tc["id"] == call_id
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
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, second_stream])

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(backend, [{"role": "user", "content": "run it"}], _python_tool_schema())

    starts = [e for e in events if e.get("type") == "tool_start"]
    assert starts, "no tool_start emitted"
    assert starts[0]["tool_call_id"] == "call_0"
    assert starts[0]["arguments"] == {}
    assert starts[-1]["tool_call_id"] == "call_0"

    args_events = [e for e in events if e.get("type") == "tool_args"]
    assert args_events, "no tool_args events for the text call"
    assert all(e["tool_call_id"] == "call_0" for e in args_events)
    streamed = "".join(e["text"] for e in args_events)
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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "give me json"}], _python_tool_schema()
    )

    assert not [e for e in events if e.get("type") == "tool_args"]
    assert not [e for e in events if e.get("type") == "tool_start"]
    content_events = [e for e in events if e.get("type") == "content"]
    assert content_events and answer in content_events[-1]["text"]


def test_provisional_text_card_closed_when_parse_fails(monkeypatch):
    """A text-sniffed provisional card must close with tool_end when its drained text fails to parse."""

    call_text = '<tool_call>{"name": "python", "arguments": {"code": "' + "x" * (
        _PROVISIONAL_ARGS_MIN_CHARS + 64
    )
    chunks = [call_text[i : i + 48] for i in range(0, len(call_text), 48)]
    stream = [_sse({"content": chunk}) for chunk in chunks] + [_done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])

    executed: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "run it"}],
        _python_tool_schema(),
        auto_heal_tool_calls = False,
    )

    starts = [e for e in events if e.get("type") == "tool_start"]
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert starts and starts[0]["tool_call_id"] == "call_0"
    assert executed == []
    assert ends, "provisional card left dangling (no tool_end)"
    assert ends[-1]["tool_call_id"] == "call_0"


def test_provisional_mcp_card_carries_server_display_name(tmp_path, monkeypatch):
    """Regression: the early card for a large-argument MCP call must already
    carry the server display name. Without it the card (and a cancelled turn's
    saved history/export/search text) shows the internal uuid server id."""
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    mcp_servers_db.create_server(id = "a3f9c1d2e4b6f807", display_name = "GitHub", url = "https://a/m")

    tool_name = "mcp__a3f9c1d2e4b6f807__create_issue"
    args = {"title": "Bug", "body": "x" * 400}
    assert len(json.dumps(args)) > _PROVISIONAL_ARGS_MIN_CHARS

    first_stream = _streamed_structured_tool_call(tool_name, args, "call_mcp_big")
    final_stream = [_sse({"content": "Filed."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "file a bug"}],
        [{"type": "function", "function": {"name": tool_name}}],
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
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())

    big_code = "total = 0\n" + "\n".join(f"total += {i}" for i in range(120))
    first_stream = _streamed_structured_tool_call("python", {"code": big_code}, "call_py")
    final_stream = [_sse({"content": "Done."}), _done()]
    backend = _make_backend(monkeypatch, [first_stream, final_stream], [])
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "write code"}],
        [{"type": "function", "function": {"name": "python"}}],
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
    """llama-server restarts delta index at 0 each round, so index alone merges two calls into one."""

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":'),
        _tool_call_fragment(0, '"first"}'),
        _tool_call_opening(0, "call_b", "web_search", '{"query":'),
        _tool_call_fragment(0, '"second"}'),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search twice"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
    )

    assert calls == [
        {"name": "web_search", "arguments": {"query": "first"}},
        {"name": "web_search", "arguments": {"query": "second"}},
    ]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == [
        "call_a",
        "call_b",
    ]

    asst = next(m for m in payloads[1]["messages"] if m.get("tool_calls"))
    assert [tc["id"] for tc in asst["tool_calls"]] == ["call_a", "call_b"]
    assert [tc["function"]["arguments"] for tc in asst["tool_calls"]] == [
        '{"query":"first"}',
        '{"query":"second"}',
    ]


def test_structured_fragment_naming_its_call_goes_back_to_that_call(monkeypatch):
    """A fragment naming a call's id must return to that call, not fork a third slot with truncated JSON."""

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":'),
        _tool_call_opening(0, "call_b", "web_search", '{"query":'),
        _tool_call_fragment(0, '"first"}', call_id = "call_a"),
        _tool_call_fragment(0, '"second"}', call_id = "call_b"),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search twice"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
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
    """A late real id must update the call opened with the placeholder call_0, not fork a nameless slot."""

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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append({"name": name, "arguments": arguments})
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
    )

    assert calls == [{"name": "web_search", "arguments": {"query": "late"}}]
    assert [e.get("tool_call_id") for e in events if e.get("type") == "tool_end"] == ["call_late"]


def test_structured_call_forked_onto_a_reused_index_executes_last(monkeypatch):
    """A fork onto a reused index runs last, so side effects of stateful tools keep their order."""

    stream = [
        _tool_call_opening(0, "call_a", "web_search", '{"query":"a"}'),
        _tool_call_opening(1, "call_b", "web_search", '{"query":"b"}'),
        _tool_call_opening(0, "call_c", "web_search", '{"query":"c"}'),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "Final answer."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])

    calls: list[dict] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append(arguments)
        return "search-result"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search three times"}],
        [{"type": "function", "function": {"name": "web_search"}}],
        max_tool_iterations = 2,
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
    """With parallel_tool_calls=false, only the first accumulated call may get a provisional card."""

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
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])

    calls = _record_tool_calls(monkeypatch, "OK")

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
    """Conversation-search budget must count the tools array, since a large MCP catalogue is prompt too."""
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
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: (
            2800 + sum(len(str(message.get("content", ""))) for message in candidate) // 10
        ),
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
    assert 0 <= budget < 1000


def test_a_long_tool_run_reports_a_boundary_in_the_requests_own_terms(monkeypatch):
    """Carry the dropped boundary in the request's own message terms; re-summing per refit overshoots."""
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

    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    backend._effective_context_length = 4000
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: (
            sum(len(str(message.get("content", ""))) for message in candidate) // 4
        ),
    )
    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_k: "R" * 3200
    )

    branch = [
        # Never evicted, so not counted as the front of the branch.
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "u" * 1200},
        {"role": "assistant", "content": "a" * 1200},
        {"role": "user", "content": "u2" * 600},
        {"role": "assistant", "content": "a2" * 600},
        {"role": "user", "content": "keep going"},
    ]
    events = _run_tool_loop(
        backend,
        branch,
        [{"type": "function", "function": {"name": "python"}}],
        max_tool_iterations = calls + 1,
        max_tokens = 400,
        context_overflow = "truncate_oldest",
    )

    notices = [
        event for event in events if event.get("type") == "context_truncated" and event.get("fits")
    ]
    assert len(notices) > 1, "the fixture must refit more than once"
    assert sum(notice["dropped_messages"] for notice in notices) > len(branch)
    # 4: the system prompt and latest turn are neither evictable nor counted.
    boundaries = [notice["boundary_messages"] for notice in notices]
    assert max(boundaries) == 4
    assert boundaries == sorted(boundaries)


def test_conversation_search_budget_is_exact_when_nothing_was_truncated(monkeypatch):
    """fit_rolling_context returns None when nothing drops, so the budget must not fall back to chars."""
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
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda candidate, *_a, **_k: (
            2800 + sum(len(str(message.get("content", ""))) for message in candidate) // 10
        ),
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
    assert 0 <= budget < 1000, budget


def test_the_exact_recall_budget_is_recomputed_after_an_intervening_tool(monkeypatch):
    """Recompute the exact recall budget after each intervening tool; a cached count goes stale."""
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
        lambda candidate, *_a, **_k: (
            1000 + sum(len(str(message.get("content", ""))) for message in candidate) // 10
        ),
    )

    budgets: list = []

    def execute_tool(name, arguments, **kwargs):
        if name == "search_conversation":
            budgets.append(kwargs.get("conversation_budget_tokens"))
            return "an earlier turn"
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
    assert budgets[0] < 1400, budgets


def _count_from_size(messages, *_args, **_kwargs):
    """Size falls as the conversation shrinks; tool_calls cost nothing until a tool message answers them."""
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
    """Refuse an unservable edit_file before it runs, since tool_result_budget clamps to zero."""
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

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": immovable}],
        [{"type": "function", "function": {"name": "edit_file"}}],
        max_tool_iterations = 2,
        max_tokens = 512,
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
    """Arguments of an already-returned call are the first lever, since dropping them frees prompt room."""
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append(name)
        assert arguments.get("new_string") == "<title>b</title>"
        return "Edited page.html"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        history,
        [{"type": "function", "function": {"name": "edit_file"}}],
        max_tool_iterations = 2,
        max_tokens = 512,
    )

    assert executed == ["edit_file"], [e.get("type") for e in events]
    assert not [
        event
        for event in events
        if event.get("type") == "tool_end" and "Nothing was written" in str(event.get("result", ""))
    ]
    assert len(payloads) >= 2, "the loop never made a second request"
    replayed = json.dumps(payloads[-1]["messages"], default = str)
    assert earlier not in replayed, "the earlier 30 KB argument was replayed verbatim"
    assert "arguments you sent" in replayed
    assert "page.html" in replayed


def test_refusing_a_call_also_stops_it_costing_the_window(monkeypatch):
    """Refused arguments must not be replayed, so refusing a call also stops them costing the window"""
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
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
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
    assert oversized not in replayed
    assert "refused before it ran" in replayed
    assert "re-read the file" not in replayed


def test_reply_room_is_reclaimed_before_generating(monkeypatch):
    """A prompt that fits can still leave no room for the reply, so compaction must run before
    generating."""
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
    """Refusing an oversized call costs the same prompt tokens as running it, so it is run and compacted."""
    oversized = "<!DOCTYPE html>" + "x" * 24000
    streams = [
        _structured_tool_call(
            "edit_file",
            {"path": "flappy-bird.html", "edits": [{"old_string": "", "new_string": oversized}]},
            "call_write_game",
        ),
        [_sse({"content": "Wrote the game."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    monkeypatch.setattr(backend, "count_chat_tokens", _count_from_size)

    executed: list[str] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        executed.append(name)
        assert oversized in json.dumps(arguments)
        return "Created flappy-bird.html"

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Create a Flappy Bird game in HTML"}],
        [{"type": "function", "function": {"name": "edit_file"}}],
        max_tool_iterations = 2,
        max_tokens = 512,
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
    """Compaction rebuilds the messages, so a call after it must not append to a stale message dict."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [_two_edits_in_one_turn(), [_sse({"content": "Done."}), _done()]],
        payloads,
    )

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
    assert _BIG_BODY not in json.dumps(sent)


def test_the_synthesized_final_pass_is_recosted_before_it_is_sent(monkeypatch):
    """The synthesized final pass must be re-costed; the iteration cap skips on_conversation_grew."""
    first_stream = _structured_tool_call("web_search", {"query": "kernel"}, "call_search")
    final_stream = [_sse({"content": "6.10."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, final_stream])
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Linux kernel 6.10. " * 400,
    )

    seen: list[list[dict]] = []

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Search for the kernel version."}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            max_tool_iterations = 1,
            permission_mode = "off",
            on_conversation_grew = lambda conversation: seen.append(copy.deepcopy(conversation)),
        )
    )

    assert len(payloads) == 2, "expected one tool round and one synthesized final pass"
    final_messages = payloads[-1]["messages"]
    assert any(
        message.get("role") == "tool" for message in final_messages
    ), "the final pass should carry the tool result this test is about"
    assert seen, "the callback never ran"
    last_seen = seen[-1]
    assert any(message.get("role") == "tool" for message in last_seen), (
        "the last re-cost ran before the tool result was appended, so the final pass "
        "was sent under stale KV accounting"
    )
    assert len(last_seen) == len(final_messages), (
        f"the final pass sends {len(final_messages)} messages but the pool was last told "
        f"about {len(last_seen)}"
    )


def _mcp_image_result() -> str:
    from core.inference import mcp_images
    image = {"data": _MCP_PNG_B64, "mimeType": "image/png"}
    return "[1 image returned]\n" + mcp_images.SENTINEL + json.dumps([image])


def _vision_backend(monkeypatch, streams, payloads, *, vision: bool):
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._is_vision = vision
    backend._mmproj_accepts_image = vision
    return backend


def _run_mcp_image_turn(monkeypatch, *, vision: bool) -> list[dict]:
    streams = [
        _structured_tool_call("mcp__fs__read_media_file", {"path": "cat.png"}, "call_mcp"),
        [_sse({"content": "A tabby cat."}), _done()],
    ]
    payloads: list[dict] = []
    backend = _vision_backend(monkeypatch, streams, payloads, vision = vision)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: _mcp_image_result(),
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "describe the image"}],
            tools = [{"type": "function", "function": {"name": "mcp__fs__read_media_file"}}],
            max_tool_iterations = 2,
        )
    )
    return payloads[1]["messages"]


def test_mcp_images_reach_a_vision_model_as_their_own_user_turn(monkeypatch):
    messages = _run_mcp_image_turn(monkeypatch, vision = True)

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "__MCP_IMAGES__" not in tool_message["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][1]["type"] == "image_url"
    assert messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_mcp_images_are_not_sent_to_a_text_only_model(monkeypatch):
    messages = _run_mcp_image_turn(monkeypatch, vision = False)

    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert "__MCP_IMAGES__" not in messages[-1]["content"]


def test_repeated_note_on_an_mcp_image_keeps_the_base64_out_of_the_tool_text(monkeypatch):
    """#11358: a note after the image array broke its parse and sent the base64 as text."""
    tool = "mcp__browser__screenshot"
    # A window-cut result: the repeat key ignores the arguments, so three distinct calls repeat.
    fitted = (
        "page\n\n... (truncated to 4 chars for the model; 99 chars total.)\n" + _mcp_image_result()
    )
    streams = [_structured_tool_call(tool, {"page": page}, f"call_{page}") for page in range(3)] + [
        [_sse({"content": "done"}), _done()]
    ]
    payloads: list[dict] = []
    backend = _vision_backend(monkeypatch, streams, payloads, vision = True)
    # Images are charged in the fit (#14968): with no projector each one costs the ~4.2k
    # admission ceiling, more than this 4096 window, so price the 6 px screenshot by size.
    backend._mmproj_projector_type = "qwen3vl_merger"
    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: fitted
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "screenshot each page"}],
            tools = [{"type": "function", "function": {"name": tool}}],
            max_tool_iterations = 4,
        )
    )

    tool_texts = [m["content"] for m in payloads[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_texts) == 3
    assert "returned exactly this 3 times" in tool_texts[-1]
    assert not any(_MCP_PNG_B64 in text for text in tool_texts)
    saved = [e["result"] for e in events if e.get("type") == "tool_end"][-1]
    assert saved.endswith("}]")
    assert saved.index("returned exactly this 3 times") < saved.index("__MCP_IMAGES__")


@pytest.mark.parametrize("wanted", [True, False])
def test_the_decode_slot_is_asked_for_and_read_only_when_a_caller_wants_it(monkeypatch, wanted):
    """Verbose alone would attach the whole prompt, so it travels with the narrowing."""
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    final["__verbose"] = {"id_slot": 3}
    stream = [_sse({"content": "done"}), "data: " + json.dumps(final) + "\n", _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream])
    seen = []

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "hi"}],
        _render_html_tools(),
        on_decode_slot = (lambda url, slot: seen.append((url, slot))) if wanted else None,
    )

    assert seen == ([(backend.base_url, 3)] if wanted else [])
    assert payloads[0].get("verbose") is (True if wanted else None)
    assert payloads[0].get("response_fields") == (["id_slot"] if wanted else None)


@pytest.mark.parametrize(
    "held",
    [
        '{"name":"terminal","arguments":{"command":"id"}}',
        'call:terminal{command:"id"}',
    ],
)
def test_a_cancel_emits_the_blocked_object_the_gguf_guard_was_holding(monkeypatch, held):
    """A completed blocked markerless call keeps the chain guard true while its suffix is
    empty, so the whole object sits in the buffer as display text the parser never promotes.
    The cancel arm returned without it and the reply vanished."""
    from core.inference.llama_cpp import _LlamaStreamCancelled

    backend = _make_backend(monkeypatch, [[_sse({"content": held})]], [])

    def _cancel_after_chunks(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks
        raise _LlamaStreamCancelled

    monkeypatch.setattr(backend, "_iter_text_cancellable", _cancel_after_chunks)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [
                {"type": "function", "function": {"name": "terminal"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            max_tool_iterations = 1,
        )
    )

    texts = [event["text"] for event in events if event["type"] == "content"]
    assert texts and texts[-1] == held
    assert not any(event["type"] == "tool_start" for event in events)


def test_only_the_tool_loop_flushes_held_text_on_cancel():
    """Only the tool loop flushes held text on cancel; the synthesized final pass holds nothing to flush."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
    arms = src.split("except _LlamaStreamCancelled:")[1:]
    flushing = [i for i, arm in enumerate(arms) if "_cancelled_hold_text()" in arm]
    assert (
        len(flushing) == 1
    ), f"exactly one cancel arm may flush held text; flushing arms: {flushing}"
    assert flushing[0] != len(arms) - 1, "the final pass must not flush the tool loop's buffers"


@pytest.mark.parametrize("edit_result", ["Edited notes.txt", "Error: failed after writing"])
def test_textual_workspace_read_edit_read_in_one_turn(monkeypatch, edit_result):
    read = '<tool_call>{"name":"terminal","arguments":{"command":"cat notes.txt"}}</tool_call>'
    edit = '<tool_call>{"name":"edit_file","arguments":{"path":"notes.txt","edits":[]}}</tool_call>'
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": read + edit + read}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["before", edit_result, "after"])

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Read, edit, and verify notes.txt"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file")
            ],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["terminal", "edit_file", "terminal"]


def test_textual_repeated_workspace_reads_do_not_crowd_out_a_later_edit(monkeypatch):
    read = '<tool_call>{"name":"terminal","arguments":{"command":"cat notes.txt"}}</tool_call>'
    edit = '<tool_call>{"name":"edit_file","arguments":{"path":"notes.txt","edits":[]}}</tool_call>'
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": read * 8 + edit + read}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["before", "Edited notes.txt", "after"])

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Read, edit, and verify notes.txt"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file")
            ],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["terminal", "edit_file", "terminal"]


def test_textual_alternating_workspace_block_does_not_replay_the_edit(monkeypatch):
    """One re-run verifies an edit. A repeating block past that applied the edit twice."""
    read = '<tool_call>{"name":"terminal","arguments":{"command":"cat notes.txt"}}</tool_call>'
    edit = '<tool_call>{"name":"edit_file","arguments":{"path":"notes.txt","edits":[]}}</tool_call>'
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": (read + edit) * 2}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["before", "Edited notes.txt", "after", "Edited notes.txt"])

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Read, edit, and verify notes.txt"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file")
            ],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["terminal", "edit_file", "terminal"]


def test_textual_alternating_workspace_block_does_not_crowd_out_a_later_tool(monkeypatch):
    """The block used to fill the 8-call cap, so the search the model asked for never ran."""
    read = '<tool_call>{"name":"terminal","arguments":{"command":"cat notes.txt"}}</tool_call>'
    edit = '<tool_call>{"name":"edit_file","arguments":{"path":"notes.txt","edits":[]}}</tool_call>'
    search = '<tool_call>{"name":"web_search","arguments":{"query":"gpu prices"}}</tool_call>'
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": (read + edit) * 4 + search}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["before", "Edited notes.txt", "after", "results"])

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Read, edit, verify, then search"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file", "web_search")
            ],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["terminal", "edit_file", "terminal", "web_search"]


def test_textual_every_independent_edit_gets_its_own_verification_rerun(monkeypatch):
    """Two edit-and-verify cycles in one turn: the test after the second edit must run."""
    test = '<tool_call>{"name":"terminal","arguments":{"command":"pytest -q"}}</tool_call>'
    edit_a = '<tool_call>{"name":"edit_file","arguments":{"path":"a.py","edits":[]}}</tool_call>'
    edit_b = '<tool_call>{"name":"edit_file","arguments":{"path":"b.py","edits":[]}}</tool_call>'
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": test + edit_a + test + edit_b + test}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["1 failed", "Edited a.py", "1 failed", "Edited b.py", "1 passed"])

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Fix the test"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file")
            ],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["terminal", "edit_file", "terminal", "edit_file", "terminal"]


def test_textual_mcp_listing_runs_again_after_an_mcp_write(monkeypatch):
    """#10348: `list, create, list` on an MCP server re-lists instead of dropping the call."""
    name = "mcp__docs__manager"
    listing = '<tool_call>{"name":"%s","arguments":{"action":"list"}}</tool_call>' % name
    create = '<tool_call>{"name":"%s","arguments":{"action":"create"}}</tool_call>' % name
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": listing + create + listing}), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    calls = []
    results = iter(["[]", "created", "[a]"])

    def execute(name, arguments, **kwargs):
        calls.append(arguments.get("action"))
        return next(results)

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Create a doc"}],
            tools = [{"type": "function", "function": {"name": name}}],
            max_tool_iterations = 3,
        )
    )
    assert calls == ["list", "create", "list"]


def _structured_batch(spec: list) -> list:
    """One assistant turn carrying `spec` as parallel structured tool_calls."""
    frames = [
        _tool_call_sse(name, args, f"call_{i}", index = i) for i, (name, args) in enumerate(spec)
    ]
    return [frames + [_done()], [_sse({"content": "Done."}), _done()]]


def _drive_structured(monkeypatch, spec, results):
    backend, _ = _backend_and_payloads(monkeypatch, _structured_batch(spec))
    calls: list[str] = []
    supply = iter(results)

    def execute(name, arguments, **kwargs):
        calls.append(name)
        return next(supply, "OK")

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Read, edit, and verify notes.txt"}],
            tools = [
                {"type": "function", "function": {"name": name}}
                for name in ("terminal", "edit_file")
            ],
            max_tool_iterations = 3,
        )
    )
    return calls


def test_structured_workspace_block_does_not_replay_the_edit(monkeypatch):
    """Structured batches skip the textual prefilter, so the controller has to catch this."""
    read = ("terminal", {"command": "cat notes.txt"})
    edit = ("edit_file", {"path": "notes.txt", "edits": []})
    calls = _drive_structured(
        monkeypatch, [read, edit, read, edit], ["v1", "Edited notes.txt", "v2", "Edited"]
    )
    assert calls == ["terminal", "edit_file", "terminal"]


def test_structured_every_independent_edit_keeps_its_verification(monkeypatch):
    read = ("terminal", {"command": "cat notes.txt"})
    edit_a = ("edit_file", {"path": "a.txt", "edits": []})
    edit_b = ("edit_file", {"path": "b.txt", "edits": []})
    calls = _drive_structured(
        monkeypatch,
        [read, edit_a, read, edit_b, read],
        ["v1", "Edited a.txt", "v2", "Edited b.txt", "v3"],
    )
    assert calls == ["terminal", "edit_file", "terminal", "edit_file", "terminal"]


def test_nested_object_schema_is_relaxed_on_the_wire_but_not_in_the_loop(monkeypatch):
    data = {
        "type": "object",
        "properties": {
            "view_url": {"type": "string"},
            "start_cursor": {"type": "string"},
            "page_size": {"type": "integer"},
        },
        "required": ["view_url"],
    }
    tool = {
        "type": "function",
        "function": {
            "name": "mcp__notion__query",
            "parameters": {"type": "object", "properties": {"data": data}, "required": ["data"]},
        },
    }
    arguments = {"data": {"view_url": "u", "page_size": "1", "start_cursor": "c"}}
    stream = [
        _tool_call_sse("mcp__notion__query", arguments, "call_a"),
        _finish("tool_calls"),
        _done(),
    ]
    final_stream = [_sse({"content": "done"}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [stream, final_stream])
    calls = _record_tool_calls(monkeypatch, "row 2")

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "next page"}],
        [tool],
        max_tool_iterations = 2,
    )

    wire = payloads[0]["tools"][0]["function"]["parameters"]["properties"]["data"]
    assert wire == {**data, "anyOf": [{"type": "object", "additionalProperties": True}]}
    assert calls == [
        ("mcp__notion__query", {"data": {"view_url": "u", "page_size": 1, "start_cursor": "c"}})
    ]


def test_an_mcp_tool_search_leaves_a_budget_of_one_for_the_tool_it_found(monkeypatch):
    """Loading deferred MCP tools is not a tool round (#10997)."""
    from core.inference import tools as tools_mod

    ping = {
        "type": "function",
        "function": {
            "name": "mcp__srv1__ping",
            "description": "[A] Ping.",
            "parameters": {"type": "object"},
        },
    }
    monkeypatch.setattr(tools_mod, "mcp_tool_search_specs", lambda arguments: [ping])
    monkeypatch.setattr(tools_mod, "mcp_loadable_specs", lambda tools: [ping])
    monkeypatch.setattr(tools_mod, "mcp_record_loaded_tools", lambda *a: None)
    streams = [
        _structured_tool_call("mcp_tool_search", {"query": "ping"}, "call_search"),
        _structured_tool_call("mcp__srv1__ping", {}, "call_ping"),
        [_sse({"content": "Pinged."}), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "ok")

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "ping it"}],
        [tools_mod._mcp_tool_search_spec([])],
        max_tool_iterations = 1,
    )

    assert [name for name, _ in calls] == ["mcp_tool_search", "mcp__srv1__ping"]
    assert "mcp__srv1__ping" in [t["function"]["name"] for t in payloads[1].get("tools", [])]


def test_budget_exhausted_final_pass_keeps_the_cached_tool_catalogue(monkeypatch):
    # The template renders tools at the top of the system prompt. Dropping them on the final
    # pass changed the prompt from its first tokens, so llama-server re-processed the whole
    # conversation on that pass and again on the next turn.
    first = [_tool_call_sse("web_search", {"query": "flappy bird"}, "call_1"), _done()]
    final_stream = [_sse({"content": "Summary."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first, final_stream])
    _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [_web_search_tool()],
        max_tool_iterations = 1,
    )

    assert len(payloads) == 2
    assert payloads[1]["tools"] == payloads[0]["tools"]
    assert payloads[1]["tool_choice"] == "none"
    assert any(e.get("type") == "content" and e.get("text") == "Summary." for e in events)


def test_replayed_failed_tool_result_carries_the_nudge_the_model_saw(monkeypatch):
    from core.inference.tool_call_parser import TOOL_ERROR_NUDGE

    failed = "Exit code 1\nerror: aborting due to previous error"
    history = [
        {"role": "user", "content": "build it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "x"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "name": "web_search", "content": failed},
        {"role": "assistant", "content": "It failed."},
        {"role": "user", "content": "show me"},
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, [[_sse({"content": "Here."}), _done()]])

    _run_tool_loop(backend, copy.deepcopy(history), [_web_search_tool()])

    tool_messages = [m for m in payloads[0]["messages"] if m.get("role") == "tool"]
    assert [m["content"] for m in tool_messages] == [failed + TOOL_ERROR_NUDGE]


def test_parallel_batch_tool_events_share_one_round_id(monkeypatch):
    # The frontend groups a round's cards into one assistant message by provenance round_id;
    # without it a batch replayed as separate assistant/tool pairs and broke the cached prefix.
    batch = [
        _tool_call_sse("web_search", {"query": "a"}, "call_a", index = 0),
        _tool_call_sse("web_search", {"query": "b"}, "call_b", index = 1),
        _done(),
    ]
    second = [_tool_call_sse("web_search", {"query": "c"}, "call_c"), _done()]
    final_stream = [_sse({"content": "Done."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [batch, second, final_stream])
    _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [_web_search_tool()],
        max_tool_iterations = 3,
    )

    rounds = {
        e["tool_call_id"]: e["provenance"]["round_id"]
        for e in events
        if e.get("type") == "tool_end"
    }
    assert set(rounds) == {"call_a", "call_b", "call_c"}
    assert rounds["call_a"] == rounds["call_b"] != rounds["call_c"]
    batch_assistant = [m for m in payloads[1]["messages"] if m.get("tool_calls")]
    assert [len(m["tool_calls"]) for m in batch_assistant] == [2]


def test_budget_notice_folded_into_a_result_is_reported_for_replay(monkeypatch):
    from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE
    from core.inference.tool_loop_controller import restore_replayed_tool_results

    first = [_tool_call_sse("web_search", {"query": "flappy bird"}, "call_1"), _done()]
    final_stream = [_sse({"content": "Summary."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first, final_stream])
    _record_tool_calls(monkeypatch, "Exit code 1\nboom")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "search"}],
        [_web_search_tool()],
        max_tool_iterations = 1,
    )

    notes = [e for e in events if e.get("type") == "tool_model_note"]
    tool_end = next(e for e in events if e.get("type") == "tool_end")
    assert notes == [
        {
            "type": "tool_model_note",
            "tool_call_id": tool_end["tool_call_id"],
            "suffix": f"\n\n{BUDGET_EXHAUSTED_NUDGE}",
        }
    ]
    # What the client sends back next turn renders exactly what the final pass sent.
    replayed = {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "web_search",
        "content": tool_end["result"],
        "extra_content": {"studio_model_suffix": notes[0]["suffix"]},
    }
    sent = [m for m in payloads[1]["messages"] if m.get("role") == "tool"][-1]
    assert restore_replayed_tool_results([replayed])[0]["content"] == sent["content"]


def test_restored_replay_is_the_branch_the_tools_are_given(monkeypatch):
    # Restored results are new dicts and the live branch is accumulated by identity, so seeding it
    # from the unrestored messages handed a tool the same replayed result twice.
    history = [
        {"role": "user", "content": "build it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "x"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_old",
            "name": "web_search",
            "content": "Error: offline",
        },
        {"role": "assistant", "content": "It failed."},
        {"role": "user", "content": "try again"},
    ]
    backend, _payloads = _backend_and_payloads(
        monkeypatch,
        [
            [_tool_call_sse("web_search", {"query": "y"}, "call_new"), _done()],
            [_sse({"content": "Done."}), _done()],
        ],
    )
    branches: list[list[dict]] = []

    def execute_tool(
        name,
        arguments,
        conversation_branch = None,
        **_kwargs,
    ):
        branches.append(list(conversation_branch or ()))
        return "sunny"

    monkeypatch.setattr("core.inference.tools.execute_tool", execute_tool)

    _run_tool_loop(backend, copy.deepcopy(history), [_web_search_tool()])

    assert branches
    replayed = [m for m in branches[0] if m.get("tool_call_id") == "call_old"]
    assert len(replayed) == 1


def test_budget_notice_folded_into_a_denied_result_is_reported_for_replay(monkeypatch):
    # A denial is the round's last tool message too, so the notice can land on it: the client has
    # to be told, or it replays the bare denial and the next turn misses the cache there.
    from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE

    # One call runs and spends the budget; the batch's last call is declined.
    blocks = (
        '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'
        '<tool_call>{"name":"python","arguments":{"code":"print(2)"}}</tool_call>'
    )
    streams = [[_sse({"content": blocks}), _done()], [_sse({"content": "Done."}), _done()]]
    backend, _payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "OK")
    decisions = iter(["allow", "deny"])
    monkeypatch.setattr("core.inference.llama_cpp.new_approval_id", lambda: "approval-1")
    monkeypatch.setattr("core.inference.llama_cpp.begin_tool_decision", lambda *_a, **_k: object())
    monkeypatch.setattr(
        "core.inference.llama_cpp.wait_tool_decision", lambda *_a, **_k: next(decisions)
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run python"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
            confirm_tool_calls = True,
            permission_mode = "ask",
            session_id = "sess",
        )
    )

    assert calls == [("python", {"code": "print(1)"})]
    tool_end = [e for e in events if e.get("type") == "tool_end"][-1]
    notes = [e for e in events if e.get("type") == "tool_model_note"]
    assert notes == [
        {
            "type": "tool_model_note",
            "tool_call_id": tool_end["tool_call_id"],
            "suffix": f"\n\n{BUDGET_EXHAUSTED_NUDGE}",
        }
    ]


def test_appended_notice_keeps_an_image_part_an_image():
    from core.inference.tool_loop_controller import with_appended_text as _with_appended_text

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    out = _with_appended_text([{"type": "text", "text": "Over the cap."}, image], "\n\nStop.")
    assert out == [
        {"type": "text", "text": "Over the cap."},
        image,
        {"type": "text", "text": "\n\nStop."},
    ]
    assert _with_appended_text("Over the cap.", "\n\nStop.") == "Over the cap.\n\nStop."
