# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server-side tools actually EXECUTE when policy leaves them on (the `--secure`
contract). A fake llama-server stream emits native tool calls and the real
``execute_tool`` runs them: python counts 1..100, terminal returns a UTC
datetime, and web_search is exercised through real ``_web_search`` with only the
``ddgs`` network boundary mocked. No model, GPU, or live network. The policy
tie-in proves the post-fix secure path (policy ``None`` + per-request
``enable_tools``) is what keeps these executions reachable.
"""

from __future__ import annotations

import contextlib
import copy
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend
from state.tool_policy import get_tool_policy, reset_tool_policy, set_tool_policy


# ── Fake llama-server stream (mirrors test_llama_cpp_tool_loop.py) ──


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _tool_call_stream(tool_name: str, arguments: dict, call_id: str) -> list[str]:
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


def _final_stream(text: str = "Done.") -> list[str]:
    return [_sse({"content": text}), _done()]


def _make_backend(monkeypatch, streams: list[list[str]]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48851
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


def _tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _run_one_tool(monkeypatch, tool_name: str, arguments: dict) -> str:
    """Drive the agentic loop with one tool call and return its real result."""
    backend = _make_backend(
        monkeypatch,
        [_tool_call_stream(tool_name, arguments, f"call_{tool_name}"), _final_stream()],
    )
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": f"use the {tool_name} tool"}],
            tools = [_tool_schema(tool_name)],
            max_tool_iterations = 1,
        )
    )
    tool_ends = [
        e
        for e in events
        if e.get("type") == "tool_end" and e.get("tool_name") == tool_name
    ]
    assert (
        tool_ends
    ), f"loop never executed {tool_name}; events={[e.get('type') for e in events]}"
    return tool_ends[0]["result"]


@pytest.fixture(autouse = True)
def _reset_policy():
    reset_tool_policy()
    yield
    reset_tool_policy()


# ── Real tool execution under the loop ──


def test_python_tool_counts_to_100(monkeypatch):
    # "Use the python tool to count from 1 to 100."
    expected = " ".join(str(i) for i in range(1, 101))
    result = _run_one_tool(
        monkeypatch,
        "python",
        {"code": "print(' '.join(str(i) for i in range(1, 101)))"},
    )
    assert expected in result, result  # real subprocess produced the full sequence


def test_bash_tool_returns_current_datetime(monkeypatch):
    # "Use the bash tool to provide today's datetime." Bound the parsed UTC time
    # to the call window rather than a hard-coded date (survives midnight/TZ).
    before = datetime.now(timezone.utc) - timedelta(seconds = 5)
    result = _run_one_tool(
        monkeypatch, "terminal", {"command": "date -u +%Y-%m-%dT%H:%M:%SZ"}
    )
    after = datetime.now(timezone.utc) + timedelta(seconds = 5)

    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", result)
    assert match, f"no UTC datetime in terminal result: {result!r}"
    parsed = datetime.strptime(match.group(), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo = timezone.utc
    )
    assert before <= parsed <= after, f"{parsed} not in [{before}, {after}]"


def test_web_search_tool_runs_with_mocked_fetch(monkeypatch):
    # "Web search for the weather for San Francisco's weather." Mock only the
    # ddgs network boundary; real _web_search formats the canned hit.
    class _FakeDDGS:
        def __init__(self, *a, **k):
            pass

        def text(
            self,
            query,
            max_results = 5,
        ):
            return [
                {
                    "title": "San Francisco Weather",
                    "href": "https://example.test/sf",
                    "body": "San Francisco: sunny, 68F.",
                }
            ]

    monkeypatch.setattr("ddgs.DDGS", _FakeDDGS)
    result = _run_one_tool(
        monkeypatch, "web_search", {"query": "weather in San Francisco"}
    )
    assert "San Francisco: sunny, 68F." in result, result
    assert "https://example.test/sf" in result


# ── Policy tie-in: the post-fix `--secure` path keeps tools reachable ──


class _Payload:
    def __init__(self, enable_tools):
        self.enable_tools = enable_tools


def test_effective_enable_tools_honors_secure_policy():
    from routes.inference import _effective_enable_tools

    # Post-fix --secure leaves policy None, so the request's flag is honored.
    set_tool_policy(None)
    assert _effective_enable_tools(_Payload(True)) is True
    assert _effective_enable_tools(_Payload(False)) is False
    assert get_tool_policy() is None

    # --enable-tools forces on; the old --secure (forced off) suppresses tools.
    set_tool_policy(True)
    assert _effective_enable_tools(_Payload(False)) is True
    set_tool_policy(False)
    assert _effective_enable_tools(_Payload(True)) is False
