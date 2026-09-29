# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import base64
import contextlib
import inspect
import json
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from types import SimpleNamespace

from core.inference import local_model_resolver as resolver
from core.inference.anthropic_compat import TOOL_RESULT_IMAGE_OMITTED
from routes import inference as inf
from studio.backend.tests.test_anthropic_messages import _mock_backend
from studio.backend.tests.test_openai_auto_switch import _FakeBackend, _wired


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
                "call_id": "call_0",
                "name": "view_image",
                "arguments": json.dumps({"path": "."}),
            },
            {"type": "function_call_output", "call_id": "call_0", "output": "shot.png"},
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


def chat_payload(**extra):
    return {
        "model": "test",
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "view_image", "parameters": {}}}],
        "messages": [
            {"role": "user", "content": "What does the screenshot show?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "view_image", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "shot.png"},
                    {"type": "image_url", "image_url": {"url": image_url()}},
                ],
            },
        ],
        **extra,
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


def post(
    monkeypatch,
    body,
    *,
    vision,
    path = "/v1/responses",
    gguf = True,
):
    seen = {"preflight": []}

    async def switch(*args, **kwargs):
        seen["preflight"].append(kwargs)

    def capture(request):
        seen["wire"] = json.loads(request.content)
        return upstream_reply(body["stream"])

    seen["backend"] = _mock_backend(
        monkeypatch,
        is_vision = vision,
        supports_tool_passthrough = True,
        supports_reasoning = False,
        reasoning_always_on = False,
        _request_reasoning_kwargs = lambda *_a, **_k: None,
        base_url = "http://llama.test",
        effective_parallel_slots = 4,
        is_loaded = gguf,
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
        response = client.post(path, json = body)
    return response, seen


def tool_contents(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


@pytest.mark.parametrize("stream", [False, True])
def test_text_only_model_answers_after_a_tool_returned_an_image(monkeypatch, stream):
    response, seen = post(monkeypatch, payload(stream), vision = False)

    assert response.status_code == 200, response.text
    assert "A red square." in response.text
    assert tool_contents(seen["wire"]["messages"]) == [
        "shot.png",
        f"shot.png\n{TOOL_RESULT_IMAGE_OMITTED}",
    ]
    if stream:
        assert [p["require_vision"] for p in seen["preflight"]] == [False]


@pytest.mark.parametrize("stream", [False, True])
def test_vision_model_still_receives_the_tool_image(monkeypatch, stream):
    response, seen = post(monkeypatch, payload(stream), vision = True)

    assert response.status_code == 200, response.text
    content = tool_contents(seen["wire"]["messages"])[1]
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


@pytest.mark.parametrize("user_image", [False, True])
def test_non_streaming_preflight_marks_tool_only_images(monkeypatch, user_image):
    monkeypatch.setattr(inf, "_should_validate_before_switch", lambda: True)
    response, seen = post(monkeypatch, payload(False, user_image = user_image), vision = True)

    assert response.status_code == 200, response.text
    [preflight] = seen["preflight"]
    assert preflight["require_vision"] is True
    assert preflight["tool_images_only"] is (not user_image)


@pytest.mark.parametrize("loop", ["enable_tools", "mcp_enabled"])
def test_text_only_model_runs_the_tool_loop_after_a_tool_returned_an_image(monkeypatch, loop):
    response, seen = post(
        monkeypatch, chat_payload(**{loop: True}), vision = False, path = "/v1/chat/completions"
    )

    assert response.status_code == 200, response.text
    [(_, call)] = seen["backend"].calls
    assert tool_contents(call["messages"]) == [f"shot.png\n{TOOL_RESULT_IMAGE_OMITTED}"]


def test_text_only_non_gguf_model_still_refuses_a_tool_image(monkeypatch):
    monkeypatch.setattr(
        inf,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = "m", models = {"m": {"is_vision": False}}),
    )
    monkeypatch.setattr(
        inf, "_detect_safetensors_features", lambda *_a, **_k: {"supports_tools": True}
    )
    response, _seen = post(
        monkeypatch, chat_payload(), vision = False, path = "/v1/chat/completions", gguf = False
    )

    assert response.status_code == 400, response.text
    assert "text-only" in response.text


def test_a_legacy_image_beside_tool_images_still_needs_vision(monkeypatch):
    monkeypatch.setattr(inf, "_should_validate_before_switch", lambda: True)
    body = chat_payload(image_base64 = image_url().partition(",")[2])
    response, seen = post(monkeypatch, body, vision = True, path = "/v1/chat/completions")

    assert response.status_code == 200, response.text
    [preflight] = seen["preflight"]
    assert preflight["tool_images_only"] is False


def record_probes(monkeypatch):
    signature = inspect.signature(inf._target_accepts_request_input)
    probes = []
    monkeypatch.setattr(
        inf,
        "_target_accepts_request_input",
        lambda *a: probes.append(signature.bind(*a).arguments) or False,
    )
    return probes


@pytest.mark.parametrize("target_is_gguf", [True, False])
def test_tool_only_images_need_vision_only_from_a_non_gguf_target(monkeypatch, target_is_gguf):
    _backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/local/B", "Q8_0", "org/B"))
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_k: target_is_gguf)
    probes = record_probes(monkeypatch)
    switch = inf._maybe_auto_switch_model(
        "org/B", object(), "t", require_vision = True, tool_images_only = True
    )

    if target_is_gguf:
        asyncio.run(switch)
        assert len(rec.calls) == 1
    else:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(switch)
        assert exc.value.status_code == 400
        assert rec.calls == []
        [probe] = probes
        assert probe["needs_vision"] is True
        assert probe["need_image"] is True


@pytest.mark.parametrize("medium", ["require_audio_input", "require_video"])
def test_tool_only_images_leave_a_gguf_target_needing_its_projector(monkeypatch, medium):
    _backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/local/B", "Q8_0", "org/B"))
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_k: True)
    probes = record_probes(monkeypatch)
    switch = inf._maybe_auto_switch_model(
        "org/B",
        object(),
        "t",
        require_vision = True,
        require_image = True,
        tool_images_only = True,
        **{medium: True},
    )

    with pytest.raises(HTTPException):
        asyncio.run(switch)
    [probe] = probes
    assert probe["needs_vision"] is True
    assert probe["need_image"] is False
    assert rec.calls == []


@pytest.mark.parametrize("tool_images_only", [True, False])
def test_auto_download_needs_an_mmproj_only_for_non_tool_images(monkeypatch, tool_images_only):
    _wired(monkeypatch, _FakeBackend("org/A-GGUF"), None)
    asked = []

    async def download(*_a, require_vision, **_k):
        asked.append(require_vision)

    monkeypatch.setattr(inf, "_maybe_auto_download_model", download)
    switch = inf._maybe_auto_switch_model(
        "org/B-GGUF", object(), "t", require_vision = True, tool_images_only = tool_images_only
    )
    with contextlib.suppress(HTTPException):
        asyncio.run(switch)

    assert asked == [not tool_images_only]
