# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server drops a reply its tool-call parser rejects; Studio re-asks with tool_choice "none" and parses it."""

from __future__ import annotations

import json

import pytest

from core.inference.stream_errors import LlamaStreamError

from .test_llama_cpp_tool_loop import (
    _done,
    _finish,
    _make_backend,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
    _web_search_tool,
)

_PEG = "The model produced output that does not match the expected peg-native format"
_RAW_CALL = '{"type": "function", "function": "web_search", "parameters": {"query": "mars rover"}}'


def _progress() -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]})
        + "\n"
    )


def _error(message: str) -> str:
    return (
        "data: "
        + json.dumps({"error": {"code": 500, "message": message, "type": "server_error"}})
        + "\n"
    )


def _content_stream(text: str) -> list[str]:
    return (
        [_progress()]
        + [_sse({"content": text[i : i + 3]}) for i in range(0, len(text), 3)]
        + [_finish("stop"), _done()]
    )


@pytest.mark.parametrize(
    "message",
    [
        _PEG,
        'Failed to parse input at pos 0: {"type": "function"',
        "Failed to parse tool call arguments as JSON: [json.exception.parse_error.101]",
    ],
)
def test_rejected_reply_is_asked_again_without_the_server_parser(monkeypatch, message):
    payloads: list[dict] = []
    streams = [
        [_progress(), _error(message)],
        _content_stream(_RAW_CALL),
        _content_stream("Found it."),
    ]
    backend = _make_backend(monkeypatch, streams, payloads)
    calls = _record_tool_calls(monkeypatch, "RESULT_OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search for the mars rover."}],
        [_web_search_tool()],
        max_tool_iterations = 2,
    )

    assert calls == [("web_search", {"query": "mars rover"})]
    assert [p.get("tool_choice") for p in payloads[:2]] == ["auto", "none"]
    assert payloads[1]["messages"] == payloads[0]["messages"]
    assert payloads[1]["tools"] == payloads[0]["tools"]
    shown = [e["text"] for e in events if e.get("type") == "content"]
    assert shown and shown[-1] == "Found it."
    assert not any('"function"' in text for text in shown)


def test_complete_calls_before_a_rejected_later_call_still_run(monkeypatch):
    """Llama-3.2-1B ``;``-joins two calls: the first streams whole, then the second is rejected."""
    payloads: list[dict] = []
    streams = [
        [_progress(), _tool_call_sse("web_search", {"query": "a"}, "c1"), _error(_PEG)],
        _content_stream("Done."),
    ]
    backend = _make_backend(monkeypatch, streams, payloads)
    calls = _record_tool_calls(monkeypatch, "RESULT_OK")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search."}],
        [_web_search_tool()],
        max_tool_iterations = 2,
    )

    assert calls == [("web_search", {"query": "a"})]
    assert [p.get("tool_choice") for p in payloads] == ["auto", "auto"]
    assert [e["text"] for e in events if e.get("type") == "content"][-1] == "Done."


def test_rejection_after_an_incomplete_streamed_call_still_fails(monkeypatch):
    """A second request cannot be joined to output the caller already has, so nothing is retried."""
    payloads: list[dict] = []
    partial = {"index": 0, "id": "c1", "type": "function"}
    partial["function"] = {"name": "web_search", "arguments": '{"query": "a'}
    streams = [[_progress(), _sse({"tool_calls": [partial]}), _error(_PEG)]]
    backend = _make_backend(monkeypatch, streams, payloads)
    calls = _record_tool_calls(monkeypatch, "RESULT_OK")

    with pytest.raises(LlamaStreamError):
        _run_tool_loop(
            backend,
            [{"role": "user", "content": "Search."}],
            [_web_search_tool()],
        )
    assert len(payloads) == 1
    assert calls == []


def test_other_stream_errors_are_not_retried(monkeypatch):
    payloads: list[dict] = []
    streams = [[_progress(), _error("the request exceeds the available context size")]]
    backend = _make_backend(monkeypatch, streams, payloads)
    _record_tool_calls(monkeypatch, "RESULT_OK")

    with pytest.raises(LlamaStreamError):
        _run_tool_loop(
            backend,
            [{"role": "user", "content": "Search."}],
            [_web_search_tool()],
        )
    assert len(payloads) == 1


def test_well_formed_structured_call_is_untouched(monkeypatch):
    payloads: list[dict] = []
    streams = [
        [
            _progress(),
            _tool_call_sse("web_search", {"query": "a"}, "c1"),
            _finish("tool_calls"),
            _done(),
        ],
        _content_stream("Done."),
    ]
    backend = _make_backend(monkeypatch, streams, payloads)
    calls = _record_tool_calls(monkeypatch, "RESULT_OK")

    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search."}],
        [_web_search_tool()],
        max_tool_iterations = 2,
    )

    assert calls == [("web_search", {"query": "a"})]
    assert [p.get("tool_choice") for p in payloads] == ["auto", "auto"]
