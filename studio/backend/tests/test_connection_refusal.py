# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

from auth import authentication
from core.inference import external_provider
from models.inference import ChatCompletionRequest
from routes import inference


def _wire(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


def _refusal_events():
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "refusal": None}}]},
        {"choices": [{"index": 0, "delta": {"refusal": "The provider declined "}}]},
        {"choices": [{"index": 0, "delta": {"refusal": ""}}]},
        {"choices": [{"index": 0, "delta": {"refusal": "this response."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}},
    ]


@pytest.fixture
def connection_app(monkeypatch):
    monkeypatch.setattr(
        inference.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "custom",
            "base_url": "https://api.openai.com/v1",
            "api_type": "chat_completions",
            "display_name": "Test connection",
            "is_enabled": True,
            "reasoning_config": None,
        },
    )
    monkeypatch.setattr(
        inference, "resolve_provider_api_key_or_400", lambda *args, **kwargs: "test-provider-key"
    )
    monkeypatch.setattr(inference.auth_storage, "is_internal_api_key", lambda raw: False)
    monkeypatch.setattr(
        inference.auth_storage,
        "validate_api_key_account",
        lambda raw, touch = True: (
            {"role": "owner", "key_name": "Coding agents", "key_purpose": "unsloth_start"},
            "test-subject",
        ),
    )
    monkeypatch.setattr(
        authentication, "request_admitted_without_credential", lambda request: False
    )
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def endpoint(payload: ChatCompletionRequest, request: Request):
        request.state.skip_api_monitor = True
        return await inference.openai_chat_completions(payload, request, "test-subject")

    return app


def _post(monkeypatch, app, events, stream):
    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, text = _wire(events), headers = {"content-type": "text/event-stream"}
        )

    body = {
        "model": "external::c1::test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }
    if stream is not None:
        body["stream"] = stream

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as provider:
            monkeypatch.setattr(external_provider, "_http_client", provider)
            async with httpx.AsyncClient(
                transport = httpx.ASGITransport(app = app), base_url = "http://test"
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers = {"Authorization": "Bearer sk-unsloth-test"},
                    json = body,
                )

    return asyncio.run(run())


@pytest.mark.parametrize("stream", [None, False], ids = ["omitted", "false"])
def test_nonstream_connection_preserves_refusal(monkeypatch, connection_app, stream):
    response = _post(monkeypatch, connection_app, _refusal_events(), stream)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "refusal": "The provider declined this response.",
            },
            "finish_reason": "stop",
        }
    ]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}


def test_streaming_connection_preserves_refusal_deltas(monkeypatch, connection_app):
    response = _post(monkeypatch, connection_app, _refusal_events(), True)

    assert response.status_code == 200, response.text
    chunks = [
        json.loads(line[5:].strip())
        for line in response.text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]
    assert (
        "".join(
            choice.get("delta", {}).get("refusal") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        == "The provider declined this response."
    )


def _collect(events):
    async def chunks():
        yield _wire(events)

    response = asyncio.run(
        inference._chat_completion_from_connection_stream(StreamingResponse(chunks()), "test-model")
    )
    return json.loads(response.body)


def test_collector_keeps_refusal_separate_from_text_and_reasoning():
    body = _collect(
        [
            {
                "choices": [
                    {"delta": {"content": "Partial answer. ", "reasoning_content": "Thought."}}
                ]
            },
            {"choices": [{"delta": {"refusal": "I cannot "}}]},
            {"choices": [{"delta": {"refusal": "continue."}, "finish_reason": "content_filter"}]},
        ]
    )

    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Partial answer. ",
                "reasoning_content": "Thought.",
                "refusal": "I cannot continue.",
            },
            "finish_reason": "content_filter",
        }
    ]


@pytest.mark.parametrize("delta", [{}, {"refusal": None}, {"refusal": ""}])
def test_absent_or_empty_refusal_retains_existing_message_shape(delta):
    body = _collect(
        [
            {"choices": [{"delta": delta}]},
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]},
        ]
    )

    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello"}
