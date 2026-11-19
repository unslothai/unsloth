# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A top-level llama-server ``grammar`` / ``json_schema`` on /v1/chat/completions must reach
llama-server on GGUF, and be refused where it cannot be applied, never silently dropped."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.inference import ChatCompletionRequest
from routes import inference as inference_route

_GRAMMAR = 'root ::= "yes" | "no"'
_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
_TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}


class _PlainBackend:
    supports_tools = False
    supports_tool_passthrough = False


def _payload(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model = "default", messages = [{"role": "user", "content": "hi"}], **fields
    )


def _refusal(payload) -> dict:
    with pytest.raises(HTTPException) as caught:
        inference_route._reject_unhonorable_raw_grammar(payload)
    assert caught.value.status_code == 400
    return caught.value.detail["error"]


@pytest.mark.parametrize("field, value", [("grammar", _GRAMMAR), ("json_schema", _SCHEMA)])
def test_passthrough_body_forwards_the_constraint(field, value):
    payload = _payload(**{field: value})

    assert inference_route._takes_tool_passthrough(payload, _PlainBackend()) is True
    inference_route._reject_unhonorable_raw_grammar(payload)
    body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)

    assert body[field] == value


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field, value", [("grammar", _GRAMMAR), ("json_schema", _SCHEMA)])
def test_the_constraint_reaches_the_upstream_post(monkeypatch, field, value, stream):
    import asyncio
    import json
    import threading

    import httpx

    from .test_openai_passthrough_respawn import _FRESH, _Backend, _Lease, _Request, _Tracker

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if stream:
            return httpx.Response(
                200,
                content = b'data: {"choices": [{"delta": {"content": "yes"}}]}\n\ndata: [DONE]\n\n',
                headers = {"content-type": "text/event-stream"},
            )
        message = {"role": "assistant", "content": "yes"}
        return httpx.Response(
            200, json = {"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        inference_route.httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(
        inference_route, "_cancelable_nonstreaming_client", lambda: real_client(transport = transport)
    )
    backend = _Backend()
    backend.base_url = _FRESH
    payload = _payload(stream = stream, **{field: value})

    async def run():
        if not stream:
            return await inference_route._openai_passthrough_non_streaming_upstream(
                backend, payload, "m", request = _Request(), cancel_event = threading.Event()
            )
        response = await inference_route._openai_passthrough_stream_admitted(
            _Request(),
            threading.Event(),
            backend,
            payload,
            "m",
            "chatcmpl-local",
            admission_lease = _Lease(),
            tracker = _Tracker(),
        )
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(run())

    assert sent and all(body[field] == value for body in sent)


def test_no_constraint_keeps_the_plain_path():
    for payload in (_payload(), _payload(grammar = ""), _payload(json_schema = None)):
        assert inference_route._takes_tool_passthrough(payload, _PlainBackend()) is False
        body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)
        assert "grammar" not in body and "json_schema" not in body


def test_tool_choice_none_lets_the_grammar_through():
    payload = _payload(grammar = _GRAMMAR, tools = [_TOOL], tool_choice = "none")

    inference_route._reject_unhonorable_raw_grammar(payload)
    body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)

    assert body["grammar"] == _GRAMMAR
    # llama-server refuses a grammar beside a callable tool catalog.
    assert "tools" not in body


def test_a_schema_under_schemaless_json_object_is_forwarded():
    payload = _payload(json_schema = _SCHEMA, response_format = {"type": "json_object"})

    inference_route._reject_unhonorable_raw_grammar(payload)
    body = inference_route._build_openai_passthrough_body(payload, backend_ctx = 4096)

    assert body["json_schema"] == _SCHEMA
    assert body["response_format"] == {"type": "json_object"}


def test_saved_memory_does_not_claim_a_constrained_request(monkeypatch):
    import core.chat_memory as chat_memory

    monkeypatch.setattr(chat_memory, "thread_context", lambda thread_id: {"thread": thread_id})

    assert inference_route._memory_tool_requested(_payload(thread_id = "t1")) is True
    for field, value in (("grammar", _GRAMMAR), ("json_schema", _SCHEMA)):
        payload = _payload(thread_id = "t1", **{field: value})
        assert inference_route._memory_tool_requested(payload) is False
        assert inference_route._explicit_studio_tool_loop_requested(payload) is False


@pytest.mark.parametrize(
    "fields, param",
    [
        ({"grammar": _GRAMMAR, "json_schema": _SCHEMA}, "grammar"),
        ({"grammar": _GRAMMAR, "response_format": {"type": "json_object"}}, "grammar"),
        (
            {"json_schema": _SCHEMA, "response_format": {"type": "json_object", "schema": {}}},
            "json_schema",
        ),
        (
            {
                "json_schema": _SCHEMA,
                "response_format": {"type": "json_schema", "json_schema": {"schema": _SCHEMA}},
            },
            "json_schema",
        ),
        ({"json_schema": _SCHEMA, "tools": [_TOOL]}, "json_schema"),
        ({"grammar": _GRAMMAR, "tools": [_TOOL], "tool_choice": "auto"}, "grammar"),
        ({"grammar": ["root ::= x"]}, "grammar"),
        ({"json_schema": "object"}, "json_schema"),
    ],
)
def test_unhonorable_combinations_are_refused(fields, param):
    assert _refusal(_payload(**fields))["param"] == param


@pytest.mark.parametrize("field, value", [("grammar", _GRAMMAR), ("json_schema", _SCHEMA)])
def test_external_provider_refuses_the_constraint(monkeypatch, field, value):
    import asyncio
    from types import SimpleNamespace

    import auth.authentication as authentication

    async def _no_proxy(*args, **kwargs):
        raise AssertionError("the provider proxy would drop the constraint")

    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: False)
    monkeypatch.setattr(inference_route, "_proxy_to_external_provider", _no_proxy)
    payload = _payload(provider_type = "openai", **{field: value})

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            inference_route.openai_chat_completions(payload, SimpleNamespace(scope = {}), "tester")
        )

    assert caught.value.status_code == 400
    assert caught.value.detail["error"]["param"] == field


@pytest.mark.parametrize("field, value", [("grammar", _GRAMMAR), ("json_schema", _SCHEMA)])
def test_non_gguf_backend_refuses_the_constraint(monkeypatch, field, value):
    from .test_sf_client_tools_passthrough import _ScriptedBackend, _call, _fixed

    backend = _ScriptedBackend(_fixed("unconstrained"))
    with pytest.raises(HTTPException) as caught:
        _call(_payload(**{field: value}), monkeypatch, backend)

    assert caught.value.status_code == 400
    assert caught.value.detail["error"]["param"] == field
    assert backend.calls == []


def test_npu_refuses_the_constraint():
    with pytest.raises(HTTPException) as caught:
        inference_route._refuse_npu_chat_request(
            _payload(grammar = _GRAMMAR), supports_vision = False, supports_tools = False
        )

    assert caught.value.detail["error"]["param"] == "grammar"
