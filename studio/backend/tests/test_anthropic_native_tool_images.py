# SPDX-License-Identifier: AGPL-3.0-only

import base64
import copy
import json
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from core.inference.anthropic_compat import (
    TOOL_RESULT_IMAGE_OMITTED,
    anthropic_messages_to_openai,
    fold_tool_results_into_user,
)
from models.inference import AnthropicMessagesRequest
from routes import inference as inf
from studio.backend.tests.test_anthropic_messages import _mock_backend


def image_block():
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format = "WEBP")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/webp",
            "data": base64.b64encode(buffer.getvalue()).decode(),
        },
    }


def payload(parts):
    return {
        "model": "vision-test",
        "max_tokens": 16,
        "tools": [{"name": "capture", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "Inspect these captures."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": ident, "name": "capture", "input": {}}
                    for ident in ("toolu_first", "toolu_second")
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_first", "content": parts},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_second",
                        "content": "second result",
                    },
                    {"type": "text", "text": "Compare them."},
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    "order", [("image",), ("text", "image"), ("image", "text"), ("text", "image", "text")]
)
def test_native_images_keep_order_and_tool_identity(order):
    parts = [
        image_block() if kind == "image" else {"type": "text", "text": str(i)}
        for i, kind in enumerate(order)
    ]
    request = AnthropicMessagesRequest(**payload(parts))
    converted = anthropic_messages_to_openai([m.model_dump() for m in request.messages])
    assert [m.get("tool_call_id") for m in converted if m["role"] == "tool"] == [
        "toolu_first",
        "toolu_second",
    ]
    assert [p["type"] for p in converted[2]["content"]] == [
        "image_url" if k == "image" else k for k in order
    ]
    assert converted[3]["content"] == "second result"
    assert converted[-1]["content"] == "Compare them."
    assert inf._anthropic_request_has_image(request)
    assert inf._anthropic_local_image_payloads(request) == [
        p["source"]["data"] for p in parts if p["type"] == "image"
    ]
    folded = fold_tool_results_into_user(converted)
    wrapper = json.loads(folded[2]["content"][0]["text"])["tool_response"]
    assert wrapper["tool_call_id"] == "toolu_first"
    assert wrapper["content"] == [p for p in converted[2]["content"] if p["type"] == "text"]
    assert [p["type"] for p in folded[2]["content"][1:]] == ["image_url"] * order.count("image")


@pytest.mark.parametrize(
    "source", ["invalid", {"type": "url", "url": 5}, {"type": "base64", "data": 5}]
)
def test_malformed_tool_result_image_source_is_skipped(source):
    request = AnthropicMessagesRequest(**payload([{"type": "image", "source": source}]))
    converted = anthropic_messages_to_openai([m.model_dump() for m in request.messages])
    assert converted[2]["content"] == ""
    assert inf._anthropic_local_image_payloads(request) == []


@pytest.mark.parametrize("vision", [True, False])
def test_native_image_http_generation_and_count(monkeypatch, vision):
    seen = {}

    async def switch(*args, **kwargs):
        seen.setdefault("preflight", []).append(kwargs)

    def count(messages, *args, **kwargs):
        seen["count"] = copy.deepcopy(messages)
        return 42

    _mock_backend(
        monkeypatch,
        is_vision = vision,
        supports_tool_passthrough = True,
        base_url = "http://llama.test",
        effective_parallel_slots = 4,
        count_chat_tokens = count,
    )
    monkeypatch.setattr(inf, "_maybe_auto_switch_model", switch)
    real_client = httpx.AsyncClient

    def capture(request):
        seen["wire"] = json.loads(request.content)
        return httpx.Response(
            200,
            json = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "The image is red."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr(
        inf.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport = httpx.MockTransport(capture)),
    )
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    body = payload([{"type": "text", "text": "capture"}, image_block()])
    with TestClient(app) as client:
        response = client.post("/v1/messages", json = body)
        counted = client.post("/v1/messages/count_tokens", json = body)
    assert response.status_code == 200, response.text
    assert counted.status_code == 200, counted.text
    image_tokens = inf._OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS if vision else 0
    assert counted.json()["input_tokens"] == 42 + image_tokens
    assert response.json()["content"][0]["text"] == "The image is red."
    assert [p["require_vision"] for p in seen["preflight"]] == [False, False]
    assert len(seen["preflight"][0]["image_preflight"]["b64s"]) == 1
    sent = seen["wire"]["messages"][2]["content"]
    counted_content = seen["count"][2]["content"]
    if vision:
        assert [p["type"] for p in counted_content] == ["text", "image_url"]
        url = sent[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (2, 2)
    else:
        assert sent == counted_content == f"capture {TOOL_RESULT_IMAGE_OMITTED}"


def test_text_only_tool_image_keeps_the_server_tool_permission_gate(monkeypatch):
    switched = []

    async def switch(*args, **kwargs):
        switched.append(kwargs)

    backend = _mock_backend(monkeypatch, is_vision = False)
    monkeypatch.setattr(inf, "_maybe_auto_switch_model", switch)
    body = payload([{"type": "text", "text": "capture"}, image_block()])
    del body["tools"]
    body.update(enable_tools = True, permission_mode = "ask")
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    with TestClient(app) as client:
        response = client.post("/v1/messages", json = body)
    assert response.status_code == 400, response.text
    assert "permission_mode" in response.text
    assert switched == [] and backend.calls == []
