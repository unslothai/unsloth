# SPDX-License-Identifier: AGPL-3.0-only

import base64
import json
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from core.inference.anthropic_compat import TOOL_RESULT_IMAGE_OMITTED
from routes import inference as inf
from studio.backend.tests.test_anthropic_messages import _mock_backend


def image_url():
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def payload(stream, *, user_image = False):
    user_content = [{"type": "input_text", "text": "What does the screenshot show?"}]
    if user_image:
        user_content.append({"type": "input_image", "image_url": image_url()})
    return {
        "model": "test",
        "stream": stream,
        "tools": [
            {
                "type": "function",
                "name": "view_image",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
        "input": [
            {"role": "user", "content": user_content},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "view_image",
                "arguments": json.dumps({"path": "shot.png"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "shot.png"},
                    {"type": "input_image", "image_url": image_url()},
                ],
            },
        ],
    }


def upstream_reply(stream):
    if stream:
        chunks = [
            {"choices": [{"delta": {"content": "A red square."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text = body, headers = {"content-type": "text/event-stream"})
    return httpx.Response(
        200,
        json = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "A red square."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 5},
        },
    )


def post(monkeypatch, body, *, vision):
    seen = {"preflight": []}

    async def switch(*args, **kwargs):
        seen["preflight"].append(kwargs)

    def capture(request):
        seen["wire"] = json.loads(request.content)
        return upstream_reply(body["stream"])

    _mock_backend(
        monkeypatch,
        is_vision = vision,
        supports_tool_passthrough = True,
        supports_reasoning = False,
        reasoning_always_on = False,
        _request_reasoning_kwargs = lambda *_a, **_k: None,
        base_url = "http://llama.test",
        effective_parallel_slots = 4,
    )
    monkeypatch.setattr(inf, "_maybe_auto_switch_model", switch)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf.httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(transport = httpx.MockTransport(capture)),
    )
    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[inf.get_current_subject] = lambda: "test"
    with TestClient(app) as client:
        response = client.post("/v1/responses", json = body)
    return response, seen


def tool_content(wire):
    [tool] = [m for m in wire["messages"] if m["role"] == "tool"]
    return tool["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_text_only_model_answers_after_a_tool_returned_an_image(monkeypatch, stream):
    response, seen = post(monkeypatch, payload(stream), vision = False)

    assert response.status_code == 200, response.text
    assert "A red square." in response.text
    assert tool_content(seen["wire"]) == [
        {"type": "text", "text": "shot.png"},
        {"type": "text", "text": TOOL_RESULT_IMAGE_OMITTED},
    ]
    if stream:
        assert [p["require_vision"] for p in seen["preflight"]] == [False]


@pytest.mark.parametrize("stream", [False, True])
def test_vision_model_still_receives_the_tool_image(monkeypatch, stream):
    response, seen = post(monkeypatch, payload(stream), vision = True)

    assert response.status_code == 200, response.text
    content = tool_content(seen["wire"])
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("stream", [False, True])
def test_text_only_model_still_refuses_a_user_image(monkeypatch, stream):
    response, seen = post(monkeypatch, payload(stream, user_image = True), vision = False)

    assert response.status_code == 400, response.text
    assert "does not support vision" in response.text
    assert "wire" not in seen
    if stream:
        assert [p["require_vision"] for p in seen["preflight"]] == [True]
