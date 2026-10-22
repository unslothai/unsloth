# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fake GGUF streams match b11160 by excluding the supplied prefill from their deltas."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.chat_template_helpers import (
    append_assistant_turn,
    trailing_assistant_resume_kind,
)
from core.inference.llama_cpp import LlamaCppBackend, _finalize_reasoning_only_cumulative

_THOUGHT = "The user wants three primes. Small ones are"
_QUESTION = {"role": "user", "content": "Name three primes."}
_CUT_MID_THOUGHT = {"role": "assistant", "content": "", "reasoning_content": _THOUGHT}


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


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


def _make_backend(monkeypatch, streams: list[list[str]], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48858
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = True
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
        yield SimpleNamespace(status_code = 200, chunks = streams.pop(0))

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


def _texts(events) -> list[str]:
    return [
        event if isinstance(event, str) else event["text"]
        for event in events
        if isinstance(event, str) or event.get("type") == "content"
    ]


@pytest.mark.parametrize(
    "last, kind",
    [
        ({"role": "assistant", "content": "2, 3 and"}, "content"),
        ({"role": "assistant", "content": "2, 3", "reasoning_content": "Easy."}, "content"),
        (_CUT_MID_THOUGHT, "reasoning_content"),
        ({"role": "assistant", "content": [], "reasoning_content": _THOUGHT}, "reasoning_content"),
        ({"role": "assistant", "content": "", "reasoning_content": "  \n"}, None),
        ({"role": "assistant", "content": ""}, None),
        (
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": _THOUGHT,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "x"}}],
            },
            None,
        ),
        (_QUESTION, None),
    ],
)
def test_the_resume_kind_matches_what_llama_server_resolves(last, kind):
    assert trailing_assistant_resume_kind([_QUESTION, last]) == kind


def test_the_route_resumes_a_thought_only_where_asked():
    from routes.inference import _continue_final_message

    payload = SimpleNamespace(
        continue_final_message = True,
        messages = [
            SimpleNamespace(role = "user", content = "Name three primes."),
            SimpleNamespace(
                role = "assistant", content = "", reasoning_content = _THOUGHT, tool_calls = None
            ),
        ],
    )
    # Other backends cannot resume inside a thought, so for them it stays a new turn.
    assert _continue_final_message(payload) is False
    assert _continue_final_message(payload, thought = True) is True
    payload.messages[-1].reasoning_content = "   "
    assert _continue_final_message(payload, thought = True) is False
    payload.messages[-1].content = "2, 3"
    assert _continue_final_message(payload) is True


def test_a_thought_only_turn_goes_out_as_a_reasoning_continuation(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"reasoning_content": " 2, 3 and 5."}), _sse({"content": "2, 3 and 5."}), _done()]],
        payloads,
    )

    events = list(
        backend.generate_chat_completion(
            messages = [_QUESTION, dict(_CUT_MID_THOUGHT)],
            continue_final_message = True,
        )
    )

    sent = payloads[0]
    assert sent["continue_final_message"] is True
    assert sent["add_generation_prompt"] is False
    assert sent["messages"][-1] == _CUT_MID_THOUGHT
    assert _texts(events)[-1] == "<think> 2, 3 and 5.</think>2, 3 and 5."


def test_without_the_flag_the_thought_is_history(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [[_sse({"content": "2, 3, 5."}), _done()]], payloads)
    list(backend.generate_chat_completion(messages = [_QUESTION, dict(_CUT_MID_THOUGHT)]))
    assert "continue_final_message" not in payloads[0]


def test_a_resumed_thought_promoted_as_the_reply_is_the_whole_thought(monkeypatch):
    """Promotion includes the original thought and its streamed continuation."""
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [[_sse({"reasoning_content": " 2, 3 and 5."}), _finish("stop"), _done()]],
        payloads,
    )

    events = list(
        backend.generate_chat_completion(
            messages = [_QUESTION, dict(_CUT_MID_THOUGHT)],
            continue_final_message = True,
        )
    )

    assert _texts(events)[-1] == f"<think> 2, 3 and 5.</think>{_THOUGHT} 2, 3 and 5."


def test_promotion_is_unchanged_without_a_resumed_thought():
    assert _finalize_reasoning_only_cumulative("<think>abc", "abc", "stop", True) == (
        "<think>abc</think>abc"
    )
    assert _finalize_reasoning_only_cumulative("<think>c", "c", "length", True, "ab") == (
        "<think>c</think>"
    )


def test_the_tool_loop_resumes_the_thought_and_keeps_it_whole(monkeypatch):
    """Tool calls replay the full resumed thought in a single assistant turn."""
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _sse({"reasoning_content": " best looked up."}),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": "small primes"}),
                                },
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "2, 3 and 5."}), _done()],
        ],
        payloads,
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", lambda *_a, **_k: "2, 3, 5, 7, 11")

    list(
        backend.generate_chat_completion_with_tools(
            messages = [_QUESTION, dict(_CUT_MID_THOUGHT)],
            tools = [_WEB_SEARCH_TOOL],
            continue_final_message = True,
            max_tool_iterations = 2,
            auto_heal_tool_calls = False,
        )
    )

    first = payloads[0]
    assert first["continue_final_message"] is True
    assert first["add_generation_prompt"] is False
    assert first["messages"][-1]["reasoning_content"] == _THOUGHT

    second = payloads[1]
    assert "continue_final_message" not in second, "after a tool result the turn is new"
    calls = [m for m in second["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(calls) == 1
    assert calls[0]["reasoning_content"] == f"{_THOUGHT} best looked up."
    assert [m["role"] for m in second["messages"]].count("assistant") == 1


def _replayed_turn(payload: dict) -> dict:
    """The resumed assistant turn, replayed ahead of the user turn the loop appended."""
    assert payload["messages"][-1]["role"] == "user"
    turn = payload["messages"][-2]
    assert turn["role"] == "assistant"
    return turn


@pytest.mark.parametrize("tail", [" worth checking.", ""])
def test_a_no_op_tool_call_after_a_resumed_thought_replays_the_whole_thought(monkeypatch, tail):
    """The nudge hides reasoning_content, so the thought travels as content, all of it."""
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                *([_sse({"reasoning_content": tail})] if tail else []),
                _sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "type": "function",
                                "function": {"name": "python", "arguments": '{"code": "1"}'},
                            }
                        ]
                    }
                ),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "2, 3 and 5."}), _finish("stop"), _done()],
        ],
        payloads,
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [_QUESTION, dict(_CUT_MID_THOUGHT)],
            tools = [_WEB_SEARCH_TOOL],
            continue_final_message = True,
            max_tool_iterations = 3,
            auto_heal_tool_calls = False,
        )
    )

    turn = _replayed_turn(payloads[1])
    assert turn["content"] == f"{_THOUGHT}{tail}"
    assert "reasoning_content" not in turn


def test_a_resumed_thought_cut_by_length_notes_the_whole_thought(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        [
            [_sse({"reasoning_content": " 2, then 3, then"}), _finish("length"), _done()],
            [_sse({"content": "2, 3 and 5."}), _finish("stop"), _done()],
        ],
        payloads,
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [_QUESTION, dict(_CUT_MID_THOUGHT)],
            tools = [_WEB_SEARCH_TOOL],
            continue_final_message = True,
            max_tool_iterations = 3,
            auto_heal_tool_calls = False,
        )
    )

    turn = _replayed_turn(payloads[1])
    assert f"{_THOUGHT} 2, then 3, then" in turn["content"]
    assert "reasoning_content" not in turn


def test_a_merged_thought_joins_the_old_reasoning():
    conversation = [_QUESTION, dict(_CUT_MID_THOUGHT)]
    append_assistant_turn(
        conversation,
        {"role": "assistant", "content": "2, 3, 5.", "reasoning_content": " 2, 3 and 5."},
        continue_final_message = True,
    )
    assert conversation[-1] == {
        "role": "assistant",
        "content": "2, 3, 5.",
        "reasoning_content": f"{_THOUGHT} 2, 3 and 5.",
    }


def test_a_resumed_answer_keeps_its_own_reasoning():
    """Resuming the answer text never grows the thought before it."""
    conversation = [
        _QUESTION,
        {"role": "assistant", "content": "2, 3", "reasoning_content": "Easy."},
    ]
    append_assistant_turn(
        conversation, {"role": "assistant", "content": " and 5."}, continue_final_message = True
    )
    assert conversation[-1] == {
        "role": "assistant",
        "content": "2, 3 and 5.",
        "reasoning_content": "Easy.",
    }


@pytest.mark.parametrize(
    "build, resumes",
    [(9199, False), (9200, True), (11160, True), (None, True)],
)
def test_thought_resumption_needs_llama_cpp_b9200(monkeypatch, build, resumes):
    monkeypatch.setattr(
        LlamaCppBackend, "probe_build_number", classmethod(lambda cls, b = None: build)
    )
    assert LlamaCppBackend.resumes_thoughts() is resumes


class _LiveProcess:
    """Stands in for the llama-server process: alive, and nothing to kill."""

    pid = None
    returncode = None

    def poll(self):
        return None


def _route_backend(monkeypatch, streams: list[list[str]], payloads: list[dict], *, tools: bool):
    """A constructed backend behind the real route, so every attribute the route reads exists."""
    backend = LlamaCppBackend()
    backend._process = _LiveProcess()
    backend._healthy = True
    backend._port = 48859
    backend._effective_context_length = 4096
    backend._supports_reasoning = True
    backend._reasoning_style = "enable_thinking"
    backend._supports_tools = tools
    backend._model_identifier = "unsloth/Qwen3.5-4B-GGUF"

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
        yield SimpleNamespace(status_code = 200, chunks = streams.pop(0))

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(
        backend,
        "_iter_text_cancellable",
        lambda response, _e, first_token_deadline = None: iter(response.chunks),
    )
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 64)
    return backend


def _route_client(monkeypatch, backend):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.inference as inference_route
    from auth.authentication import get_current_subject
    from utils.api_errors import install_api_error_handlers

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _stream_deltas(resp) -> list[dict]:
    deltas = []
    for line in resp.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            choice = (json.loads(line[6:]).get("choices") or [{}])[0]
            if choice.get("delta"):
                deltas.append(choice["delta"])
    return deltas


@pytest.mark.parametrize("tools", [False, True], ids = ["plain", "tool-loop"])
def test_the_chat_route_resumes_the_thought_the_ui_sends(monkeypatch, tools):
    """The request the chat UI sends for Resume on a turn stopped mid-thought, end to end."""
    payloads: list[dict] = []
    backend = _route_backend(
        monkeypatch,
        [
            [
                _sse({"reasoning_content": " 2, 3 and 5."}),
                _sse({"content": "2, 3 and 5."}),
                _finish("stop"),
                _done(),
            ]
        ],
        payloads,
        tools = tools,
    )
    body = {
        "messages": [_QUESTION, _CUT_MID_THOUGHT],
        "continue_final_message": True,
        "stream": True,
    }
    if tools:
        body.update(enable_tools = True, enabled_tools = ["web_search"])
    resp = _route_client(monkeypatch, backend).post(
        "/v1/chat/completions", json = body, headers = {"X-Unsloth-Events": "1"}
    )

    assert resp.status_code == 200, resp.text
    (sent,) = payloads
    assert ("tools" in sent) is tools
    assert sent["continue_final_message"] is True
    assert sent["add_generation_prompt"] is False
    assert sent["messages"][-1] == {**_CUT_MID_THOUGHT}
    deltas = _stream_deltas(resp)
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == " 2, 3 and 5."
    assert "".join(d.get("content") or "" for d in deltas) == "2, 3 and 5."


def test_an_older_llama_server_refuses_to_resume_a_thought(monkeypatch):
    """Older builds close the thought instead of continuing inside it."""
    payloads: list[dict] = []
    backend = _route_backend(monkeypatch, [], payloads, tools = False)
    backend._resumes_thoughts = False
    resp = _route_client(monkeypatch, backend).post(
        "/v1/chat/completions",
        json = {
            "messages": [_QUESTION, _CUT_MID_THOUGHT],
            "continue_final_message": True,
            "stream": False,
        },
    )
    assert resp.status_code == 400, resp.text
    assert "mid-thought" in resp.text
    assert payloads == []


def test_an_older_llama_server_still_resumes_an_answer(monkeypatch):
    payloads: list[dict] = []
    backend = _route_backend(
        monkeypatch,
        [[_sse({"content": " and 5."}), _finish("stop"), _done()]],
        payloads,
        tools = False,
    )
    backend._resumes_thoughts = False
    resp = _route_client(monkeypatch, backend).post(
        "/v1/chat/completions",
        json = {
            "messages": [_QUESTION, {"role": "assistant", "content": "2, 3"}],
            "continue_final_message": True,
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert payloads[0]["continue_final_message"] is True


def test_the_tool_passthrough_resumes_the_thought(monkeypatch):
    """Client tools skip Studio's loop, and the forwarded body must still continue the thought."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_openai_passthrough_body

    backend = _route_backend(monkeypatch, [], [], tools = True)
    payload = ChatCompletionRequest(
        messages = [_QUESTION, _CUT_MID_THOUGHT],
        tools = [_WEB_SEARCH_TOOL],
        continue_final_message = True,
    )
    body = _build_openai_passthrough_body(payload, llama_backend = backend)
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False
    assert body["messages"][-1]["reasoning_content"] == _THOUGHT


def test_an_older_llama_server_refuses_a_thought_in_the_tool_passthrough(monkeypatch):
    payloads: list[dict] = []
    backend = _route_backend(monkeypatch, [], payloads, tools = True)
    backend._resumes_thoughts = False
    resp = _route_client(monkeypatch, backend).post(
        "/v1/chat/completions",
        json = {
            "messages": [_QUESTION, _CUT_MID_THOUGHT],
            "tools": [_WEB_SEARCH_TOOL],
            "continue_final_message": True,
            "stream": False,
        },
    )
    assert resp.status_code == 400, resp.text
    assert "mid-thought" in resp.text
    assert payloads == []
