# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_sf_client_tools_passthrough as route_test
from core.inference.engine_adapters import ADAPTERS, tool_parser_for_template
from core.inference.managed_engine import validate_model
from routes import inference as api


@pytest.mark.parametrize("engine,expected", [("vllm", "hermes"), ("sglang", "qwen")])
def test_tool_parser_uses_template_not_model_family(tmp_path, engine, expected):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unrelated_finetune"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"chat_template": '{% if tools %}<tool_call>{"name": "x", "arguments": {}}{% endif %}'}
        )
    )
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    options = validate_model(config, engine = engine)
    assert options["tool_parser"] == expected
    args = ADAPTERS[engine].command("python", "model", 40000, "key", 4096, 0.5, options = options)
    assert args[args.index("--tool-call-parser") + 1] == expected
    assert ("--enable-auto-tool-choice" in args) == (engine == "vllm")
    (tmp_path / "chat_template.jinja").write_text("{{ tools }}<tool_call><function=example>")
    assert validate_model(config, engine = engine)["tool_parser"] == "qwen3_coder"
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    assert validate_model(config, engine = engine)["tool_parser"] is None


def test_named_tool_template_and_unknown_syntax(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "default", "template": "{{ messages }}"},
                    {"name": "tool_use", "template": "{{ tools }}[TOOL_CALLS]"},
                ]
            }
        )
    )
    assert (
        validate_model(SimpleNamespace(is_local = True, path = str(tmp_path)))["tool_parser"]
        == "mistral"
    )
    assert tool_parser_for_template("{{ tools }}CUSTOM", "vllm") is None


@pytest.fixture
def native(monkeypatch):
    from routes import managed_engine_chat

    # The module the route really uses: another test module re-imports httpx at collection.
    httpx = managed_engine_chat.httpx
    backend = route_test._ScriptedBackend(route_test._fixed("plain"))
    backend.models["sf-model"].update(engine = "vllm", supports_tools = True)
    backend._managed_engine = SimpleNamespace(
        model = "sf-model",
        context = 4096,
        base_url = "http://native.test",
        headers = {"Authorization": "Bearer private"},
    )
    route_test._install(monkeypatch, backend)
    requests = []
    call = {
        "id": "native-call",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"cats"}'},
    }

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["Authorization"] == "Bearer private"
        name = body.get("tools", [{}])[0].get("function", {}).get("name", "lookup")
        tc = {
            **call,
            "function": {
                "name": name,
                "arguments": '{"code":"print(3973)"}' if name == "python" else '{"q":"cats"}',
            },
        }
        finished = (
            any(m.get("role") == "tool" for m in body["messages"])
            or body.get("tool_choice") == "none"
        )
        message = {"role": "assistant", "content": "3973" if finished else None}
        if not finished:
            message["tool_calls"] = [tc]
        finish = "stop" if finished else "tool_calls"
        usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        if not body.get("stream"):
            return httpx.Response(
                200,
                json = {
                    "id": "native-id",
                    "object": "chat.completion",
                    "model": "sf-model",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                },
            )
        chunks = [
            {
                "id": "native-id",
                "model": "sf-model",
                "choices": [
                    {"index": 0, "delta": {k: v for k, v in message.items() if k != "tool_calls"}}
                ],
            }
        ]
        if not finished:
            chunks += [
                {
                    "id": "native-id",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        **tc,
                                        "index": 0,
                                        "function": {
                                            "name": name,
                                            "arguments": tc["function"]["arguments"][:6],
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "id": "native-id",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": tc["function"]["arguments"][6:]},
                                    }
                                ]
                            },
                        }
                    ],
                },
            ]
        chunks += [
            {"id": "native-id", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {"id": "native-id", "choices": [], "usage": usage},
        ]
        return httpx.Response(
            200,
            text = "".join("data: " + json.dumps(e) + "\n\n" for e in chunks) + "data: [DONE]\n\n",
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport = httpx.MockTransport(handle), **kwargs),
    )
    return backend, requests


def run(payload):
    async def request():
        response = await api.openai_chat_completions(payload, route_test._Request(), "u")
        if payload.stream:
            return [
                json.loads(line[5:])
                async for line in response.body_iterator
                if line.startswith("data:") and line[5:].strip() != "[DONE]"
            ]
        return json.loads(response.body)

    return asyncio.run(request())


@pytest.mark.parametrize("stream", [False, True])
def test_client_tools_and_history_reach_native_engine(native, stream):
    backend, requests = native
    result = run(
        route_test._request(tools = [route_test.LOOKUP_TOOL], stream = stream, enable_tools = False)
    )
    assert requests[0]["tools"] == [route_test.LOOKUP_TOOL]
    assert "enable_tools" not in requests[0]
    if stream:
        calls = [
            tc
            for e in result
            for choice in e.get("choices", [])
            for tc in choice.get("delta", {}).get("tool_calls", [])
        ]
        assert calls[0]["id"] == "native-call"
        assert "".join(tc["function"].get("arguments", "") for tc in calls) == '{"q":"cats"}'
    else:
        assert result["id"] == "native-id"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "native-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"cats"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "native-call", "content": "3973"},
    ]
    result = run(
        route_test._request(
            messages = history, tools = [route_test.LOOKUP_TOOL], tool_choice = "none", enable_tools = False
        )
    )
    assert result["choices"][0]["message"]["content"] == "3973"
    assert requests[-1]["messages"][-1]["tool_call_id"] == "native-call"
    assert requests[-1]["tool_choice"] == "none"
    assert api.active_generations.count() == 0


def test_studio_loop_executes_and_replays_native_tool_calls(native, monkeypatch):
    from core.inference import studio_tool_loop

    executed = []

    def execute(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "3973"

    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    result = run(
        route_test._request(
            enable_tools = True,
            enabled_tools = ["python"],
            permission_mode = "off",
            max_tool_calls_per_message = 1,
        )
    )
    assert executed == [("python", {"code": "print(3973)"})]
    assert result["choices"][0]["message"]["content"] == "3973"
    assert result["usage"]["prompt_tokens"] == 20
    assert "tool_calls" not in result["choices"][0]["message"]
    assert any(
        m.get("role") == "tool" and m.get("content") == "3973" for m in native[1][-1]["messages"]
    )
    assert api.active_generations.count() == 0


@pytest.mark.parametrize(
    "choice", ["auto", "required", "none", {"type": "function", "function": {"name": "lookup"}}]
)
def test_client_tool_choice_is_forwarded(native, choice):
    run(route_test._request(tools = [route_test.LOOKUP_TOOL], tool_choice = choice, enable_tools = False))
    assert native[1][0]["tool_choice"] == choice


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "tools": [route_test.LOOKUP_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            },
            "not enabled",
        ),
        (
            {"enable_tools": True, "enabled_tools": ["python"], "permission_mode": "confirm"},
            "confirmation",
        ),
    ],
)
def test_rejection_does_not_start_native_request(native, kwargs, match):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        run(route_test._request(**kwargs))
    assert match in str(error.value.detail).lower()
    assert not native[1]
    assert api.active_generations.count() == 0


def test_unknown_template_rejects_before_native_request(native):
    from fastapi import HTTPException

    native[0].models["sf-model"]["supports_tools"] = False
    with pytest.raises(HTTPException) as error:
        run(route_test._request(tools = [route_test.LOOKUP_TOOL]))
    assert "chat template" in str(error.value.detail)
    assert not native[1]


def test_unstarted_stream_releases_tracking(native):
    async def request():
        response = await api.openai_chat_completions(
            route_test._request(tools = [route_test.LOOKUP_TOOL], stream = True),
            route_test._Request(),
            "u",
        )
        assert api.active_generations.count() == 1
        await response._unstarted_cleanup()
        assert api.active_generations.count() == 0

    asyncio.run(request())
    assert not native[1]


def test_stream_close_releases_tracking(native):
    async def request():
        response = await api.openai_chat_completions(
            route_test._request(tools = [route_test.LOOKUP_TOOL], stream = True),
            route_test._Request(),
            "u",
        )
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        assert api.active_generations.count() == 0

    asyncio.run(request())


def test_native_rejection_keeps_client_error_status(native):
    from core.inference.engine_transport import EngineHTTPError

    error = EngineHTTPError(400, '{"error":{"message":"Maximum context length exceeded"}}')
    chunk = api._openai_stream_error_chunk(error)
    assert chunk["error"]["code"] == "context_length_exceeded"
    assert chunk["error"]["type"] == "invalid_request_error"


def test_plain_chat_native_rejection_returns_400(native):
    from core.inference.engine_transport import EngineHTTPError
    from fastapi import HTTPException

    def reject(messages, tools):
        raise EngineHTTPError(400, '{"error":{"message":"Maximum context length exceeded"}}')

    native[0]._responder = reject
    with pytest.raises(HTTPException) as error:
        run(route_test._request(enable_tools = False))
    assert error.value.status_code == 400
    assert error.value.detail["error"]["code"] == "context_length_exceeded"
    assert api.active_generations.count() == 0


def test_orchestrator_preserves_native_http_error():
    from core.inference.engine_transport import EngineHTTPError
    from core.inference.orchestrator import InferenceOrchestrator

    def reject(**kwargs):
        raise EngineHTTPError(400, "Maximum context length exceeded")

    backend = InferenceOrchestrator.__new__(InferenceOrchestrator)
    backend._managed_engine = SimpleNamespace(generate = reject)
    with pytest.raises(EngineHTTPError):
        list(backend._generate_inner(messages = []))


@pytest.mark.parametrize("started", [False, True])
def test_monitor_failure_cannot_leak_active_request(native, monkeypatch, started):
    def fail(*args, **kwargs):
        raise RuntimeError("monitor unavailable")

    async def request():
        response = await api.openai_chat_completions(
            route_test._request(tools = [route_test.LOOKUP_TOOL], stream = True),
            route_test._Request(),
            "u",
        )
        if started:
            await anext(response.body_iterator)
        monkeypatch.setattr(api.api_monitor, "finish", fail)
        with pytest.raises(RuntimeError, match = "monitor unavailable"):
            if started:
                await response.body_iterator.aclose()
            else:
                await response._unstarted_cleanup()
        assert api.active_generations.count() == 0

    asyncio.run(request())


@pytest.mark.parametrize("vision", [False, True])
def test_managed_tool_history_promotes_images_for_vision_only(native, vision):
    import base64
    import io
    from PIL import Image
    from core.inference.mcp_images import SENTINEL

    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format = "PNG")
    envelope = (
        "Screenshot\n"
        + SENTINEL
        + json.dumps(
            [{"data": base64.b64encode(image.getvalue()).decode(), "mimeType": "image/png"}]
        )
    )
    native[0].models["sf-model"]["is_vision"] = vision
    payload = route_test._request(enable_tools = False, tools = [route_test.LOOKUP_TOOL])
    payload.messages = [
        route_test.ChatMessage(role = "user", content = "Look"),
        route_test.ChatMessage(
            role = "assistant",
            content = "",
            tool_calls = [
                {
                    "id": "prior",
                    "type": "function",
                    "function": {"name": "mcp__test__image", "arguments": "{}"},
                }
            ],
        ),
        route_test.ChatMessage(
            role = "tool", tool_call_id = "prior", name = "mcp__test__image", content = envelope
        ),
        route_test.ChatMessage(role = "user", content = "Describe it"),
    ]
    run(payload)
    sent = native[1][0]["messages"]
    assert SENTINEL not in json.dumps(sent)
    parts = [
        part
        for message in sent
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    assert any(part.get("type") == "image_url" for part in parts) is vision


def test_managed_route_never_forwards_a_bare_image_path(native):
    native[0].models["sf-model"]["is_vision"] = True
    payload = route_test._request(enable_tools = False, tools = [route_test.LOOKUP_TOOL])
    path = "/" * 64 + "etc/hostname"
    payload.messages = [
        route_test.ChatMessage(
            role = "user",
            content = [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": path}},
            ],
        )
    ]
    run(payload)
    urls = [
        part["image_url"]["url"]
        for body in native[1]
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert urls and all(url.startswith("data:") for url in urls)
