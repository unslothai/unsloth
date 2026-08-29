# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request correlation for row-owning chat-completions streams."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routes.inference import (  # noqa: E402
    _MONITOR_ID_RESPONSE_HEADER,
    _SameTaskStreamingResponse,
    _monitor_response_headers,
)


@pytest.mark.parametrize("monitor_id", [None, "", "   ", 123])
def test_monitor_header_is_absent_without_an_owned_nonempty_row(monitor_id):
    base = {"Cache-Control": "no-cache"}

    headers = _monitor_response_headers(monitor_id, base)

    assert headers == base
    assert _MONITOR_ID_RESPONSE_HEADER not in headers


@pytest.mark.asyncio
async def test_monitor_header_preserves_stream_body_bytes_and_event_order():
    chunks = (
        b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n',
        b"data: [DONE]\n\n",
    )

    async def body():
        for chunk in chunks:
            yield chunk

    response = _SameTaskStreamingResponse(
        body(),
        media_type = "text/event-stream",
        headers = _monitor_response_headers(
            "  monitor-opaque-id  ",
            {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        ),
    )

    assert response.headers[_MONITOR_ID_RESPONSE_HEADER] == "monitor-opaque-id"
    assert response.headers["cache-control"] == "no-cache"
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"".join(chunks)


def _named_function(tree: ast.AST, name: str) -> ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def _callee_name(expr: ast.expr) -> str | None:
    if not isinstance(expr, ast.Call):
        return None
    if isinstance(expr.func, ast.Name):
        return expr.func.id
    return None


def _stream_responses(function: ast.AsyncFunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"StreamingResponse", "_SameTaskStreamingResponse"}
    ]


def _assert_monitor_wrapped(call: ast.Call) -> None:
    headers = next((kw.value for kw in call.keywords if kw.arg == "headers"), None)
    assert isinstance(headers, ast.Call)
    assert isinstance(headers.func, ast.Name)
    assert headers.func.id == "_monitor_response_headers"
    assert headers.args and isinstance(headers.args[0], ast.Name)
    assert headers.args[0].id == "monitor_id"


def test_every_row_owning_chat_stream_constructor_uses_exact_monitor_header():
    tree = ast.parse((_BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8"))

    proxy_calls = _stream_responses(_named_function(tree, "_proxy_to_external_provider"))
    proxy_by_body = {_callee_name(call.args[0]): call for call in proxy_calls}
    _assert_monitor_wrapped(proxy_by_body["_tracked_stream"])
    # Codex returns before opening a monitor row, so correlating it would be false.
    codex_headers = next(
        kw.value for kw in proxy_by_body["_codex_stream"].keywords if kw.arg == "headers"
    )
    assert isinstance(codex_headers, ast.Dict)

    production_calls = _stream_responses(_named_function(tree, "produce_openai_chat_completions"))
    production_by_body = {_callee_name(call.args[0]): call for call in production_calls}
    for body_name in (
        "audio_input_stream",
        "admitted_gguf_tool_stream",
        "admitted_gguf_stream_chunks",
        "sf_tool_stream",
        "stream_chunks",
    ):
        _assert_monitor_wrapped(production_by_body[body_name])

    queued_calls = _stream_responses(_named_function(tree, "_openai_passthrough_stream"))
    queued = next(call for call in queued_calls if _callee_name(call.args[0]) == "_queued_stream")
    _assert_monitor_wrapped(queued)

    admitted_calls = _stream_responses(_named_function(tree, "_openai_passthrough_stream_admitted"))
    admitted_by_body = {_callee_name(call.args[0]): call for call in admitted_calls}
    # Covers the pre-body cancelled response and the admitted passthrough body.
    _assert_monitor_wrapped(admitted_by_body["iter"])
    _assert_monitor_wrapped(admitted_by_body["_stream"])
