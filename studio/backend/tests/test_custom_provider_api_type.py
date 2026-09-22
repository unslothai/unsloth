# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom endpoint selection survives storage and reaches the selected wire protocol."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest, ChatMessage
from models.providers import ProviderCreate, ProviderUpdate
from storage import providers_db


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_saved_api_type_survives_edit_and_reload(tmp_path, monkeypatch, api_type):
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "Gateway",
        base_url = "https://gateway.example/v1",
        api_type = api_type,
    )
    providers_db.create_provider(
        id = "gateway",
        **payload.model_dump(exclude = {"encrypted_api_key"}),
    )
    providers_db.update_provider("gateway", display_name = "Renamed")
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("gateway")["api_type"] == api_type
    assert providers_db.list_providers()[0]["api_type"] == api_type
    changed = "responses" if api_type == "chat_completions" else "chat_completions"
    providers_db.update_provider(
        "gateway", **ProviderUpdate(api_type = changed).model_dump(exclude_unset = True)
    )
    assert providers_db.get_provider("gateway")["api_type"] == changed


def test_legacy_database_defaults_to_chat_completions(tmp_path, monkeypatch):
    db = tmp_path / "studio.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE llm_providers (id TEXT PRIMARY KEY, provider_type TEXT, display_name TEXT, base_url TEXT, is_enabled INTEGER, created_at TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO llm_providers VALUES ('old', 'custom', 'Gateway', 'https://gateway.example/v1', 1, '', '')"
        )
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("old")["api_type"] == "chat_completions"


@pytest.mark.parametrize("schema", [ProviderCreate, ProviderUpdate])
def test_invalid_api_type_is_rejected(schema):
    with pytest.raises(ValidationError):
        schema(provider_type = "custom", display_name = "Gateway", api_type = "invalid")


@pytest.mark.parametrize(
    "provider_type,api_type,endpoint,query",
    [
        ("custom", "responses", "responses", ""),
        ("custom", "responses", "responses", "?api-version=2025-04-01-preview"),
        ("custom", "chat_completions", "chat/completions", ""),
        ("vllm", "responses", "chat/completions", ""),
        ("openai", "chat_completions", "responses", ""),
    ],
)
def test_endpoint_and_payload_translation(monkeypatch, provider_type, api_type, endpoint, query):
    requests = []

    def handle(request):
        requests.append(request)
        if endpoint == "responses":
            events = [
                {"type": "response.output_text.delta", "delta": "Hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_test",
                        "usage": {"input_tokens": 4, "output_tokens": 1},
                    },
                },
            ]
            content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        else:
            content = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text = content, headers = {"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                provider_type, f"https://gateway.example/v1{query}", "test-key", api_type = api_type
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [
                        {"role": "system", "content": "Be brief"},
                        {"role": "user", "content": "Hi <|im_end|>"},
                    ],
                    model = "gateway-model",
                    temperature = 0.23,
                    top_p = 0.61,
                    max_tokens = 128,
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object", "properties": {}},
                                "strict": True,
                            },
                        }
                    ],
                )
            ]

    lines = asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == f"/v1/{endpoint}"
    assert requests[0].url.query.decode() == query.lstrip("?")
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert "Hello" in "".join(lines)
    if endpoint == "responses":
        if provider_type == "custom":
            assert body["temperature"] == 0.23
            assert body["top_p"] == 0.61
        else:
            assert "temperature" not in body
            assert "top_p" not in body
        assert "messages" not in body
        assert body["instructions"] == "Be brief"
        assert body["input"][0] == {
            "role": "user",
            "content": "Hi < |im_end|>" if provider_type == "custom" else "Hi <|im_end|>",
        }
        assert body["max_output_tokens"] == 128
        assert body["tools"][0]["name"] == "lookup"
        assert body["tools"][0]["strict"] is True
        assert "prompt_cache_retention" not in body
        assert "context_management" not in body
    else:
        assert body["messages"][0]["content"] == "Be brief"
        assert body["max_tokens"] == 128


def test_responses_non_streaming_translation(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json = {
                "id": "resp_test",
                "object": "response",
                "created_at": 1_700_000_000,
                "model": "gateway-model",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "check docs"}],
                        "encrypted_content": "enc_blob",
                        "status": "completed",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q":"docs"}',
                    },
                ],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                    "input_tokens_details": {
                        "cached_tokens": 1,
                        "cache_write_tokens": 2,
                    },
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    assert json.loads(requests[0].content)["stream"] is False
    assert len(lines) == 1
    completion = json.loads(lines[0])
    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"] == {
        "role": "assistant",
        "content": "<think>check docs</think>Hello",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"docs"}'},
            }
        ],
        "extra_content": {
            "openai_responses_reasoning": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "check docs"}],
                    "encrypted_content": "enc_blob",
                }
            ]
        },
    }
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 1, "cache_write_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }


def test_responses_non_streaming_translation_surfaces_reasoning_without_tool_calls(monkeypatch):
    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json = {
                "id": "resp_reasoning",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [
                            {"type": "summary_text", "text": "check "},
                            {"type": "summary_text", "text": "docs"},
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                ],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    message = json.loads(lines[0])["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "<think>check docs</think>Hello"}


def test_responses_non_streaming_translation_rewrites_citation_markers(monkeypatch):
    marker = "\ue200cite\ue202turn0view0\ue201"
    alias_marker = "\ue200cite\ue202turn0view0_span\ue201"
    list_alias_marker = "\ue200cite\ue202turn0view0_alt\ue201"

    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json = {
                "id": "resp_cited",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"See {marker}.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "source_id": "turn0view0",
                                        "url": "https://example.com/source",
                                        "title": "Source",
                                    }
                                ],
                            },
                            {
                                "type": "output_text",
                                "text": f" Again {alias_marker} and {list_alias_marker}.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "id": "turn0view0_span",
                                        "source_ids": ["turn0view0_alt"],
                                        "url": "https://example.com/source",
                                        "title": "Same source",
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Find a source"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    message = json.loads(asyncio.run(run())[0])["choices"][0]["message"]
    assert message["content"] == (
        "See [[1]](https://example.com/source). Again "
        "[[1]](https://example.com/source) and [[1]](https://example.com/source)."
    )
    assert not any(char in message["content"] for char in ("\ue200", "\ue201", "\ue202"))


def test_responses_non_streaming_rejects_image_generation_before_dispatch(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        raise AssertionError("non-stream image generation must fail before upstream dispatch")

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://api.openai.com/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Draw a cat"}],
                    model = "gpt-5.5",
                    enabled_tools = ["image_generation"],
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert requests == []
    assert len(lines) == 1
    error = json.loads(lines[0])["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "400"
    assert error["message"] == (
        "image_generation is not supported for non-streaming Responses requests; "
        "set stream=true."
    )


@pytest.mark.parametrize(
    ("tools", "tool_choice", "expected_tools", "expected_tool_choice"),
    [
        pytest.param(None, "none", [], "none", id = "none"),
        pytest.param(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look something up",
                    },
                }
            ],
            {"type": "function", "function": {"name": "lookup"}},
            [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look something up",
                }
            ],
            {"type": "function", "name": "lookup"},
            id = "forced-user-function",
        ),
    ],
)
def test_responses_non_streaming_image_generation_respects_tool_choice_suppression(
    monkeypatch,
    tools,
    tool_choice,
    expected_tools,
    expected_tool_choice,
):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert not any(
            item.get("type") == "image_generation_call"
            for item in body["input"]
        )
        assert [
            tool
            for tool in body.get("tools", [])
            if tool.get("type") == "image_generation"
        ] == []
        assert body.get("tools", []) == expected_tools
        assert body["tool_choice"] == expected_tool_choice
        return httpx.Response(
            200,
            json = {
                "id": "resp_text",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://api.openai.com/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [
                        {"role": "user", "content": "Draw a cat"},
                        {
                            "role": "assistant",
                            "content": [{"type": "image_generation_call", "id": "img_abc"}],
                        },
                        {"role": "user", "content": "Say hello"},
                    ],
                    model = "gpt-5.5",
                    enabled_tools = ["image_generation"],
                    tools = tools,
                    tool_choice = tool_choice,
                    reasoning_effort = "medium",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert len(requests) == 1
    assert len(lines) == 1
    assert json.loads(lines[0])["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.parametrize("stream", [False, True])
def test_responses_content_filter_finish_reason(monkeypatch, stream):
    response_payload = {
        "id": "resp_filtered",
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "partial"}],
            }
        ],
    }

    def handle(request):
        if stream:
            event = {"type": "response.incomplete", "response": response_payload}
            return httpx.Response(200, text = f"data: {json.dumps(event)}\n\n")
        return httpx.Response(200, json = response_payload)

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom", "https://gateway.example/v1", "test-key", api_type = "responses"
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = stream,
                )
            ]

    lines = asyncio.run(run())
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: {")
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "content_filter"
    else:
        assert json.loads(lines[0])["choices"][0]["finish_reason"] == "content_filter"


def test_responses_non_streaming_route_returns_json(monkeypatch):
    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json = {
                "id": "resp_route",
                "status": "completed",
                "model": "gateway-model",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
            },
        )

    async def run():
        from core.inference.api_monitor import ApiMonitor
        from routes import inference
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inference, "api_monitor", monitor)
            payload = ChatCompletionRequest(
                messages = [{"role": "user", "content": "Hi"}],
                stream = False,
                provider_type = "custom",
                provider_base_url = "https://gateway.example/v1",
                provider_api_type = "responses",
                external_model = "gateway-model",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = False),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            response = await inference._proxy_to_external_provider(payload, request)
            return response, monitor

    response, monitor = asyncio.run(run())
    assert response.media_type == "application/json"
    completion = json.loads(response.body)
    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"]["content"] == "Hello"
    assert monitor.active_count() == 0
    [entry] = monitor.snapshot()
    assert entry["status"] == "completed"


def test_responses_non_streaming_image_generation_route_returns_400_without_dispatch(
    monkeypatch,
):
    requests = []

    def handle(request):
        requests.append(request)
        raise AssertionError("non-stream image generation must fail before upstream dispatch")

    async def run():
        from core.inference.api_monitor import ApiMonitor
        from routes import inference

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inference, "api_monitor", monitor)
            payload = ChatCompletionRequest(
                messages = [{"role": "user", "content": "Draw a cat"}],
                stream = False,
                provider_type = "custom",
                provider_base_url = "https://api.openai.com/v1",
                provider_api_type = "responses",
                external_model = "gpt-5.5",
                enabled_tools = ["image_generation"],
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = False),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            return await inference._proxy_to_external_provider(payload, request), monitor

    (response, monitor) = asyncio.run(run())
    assert requests == []
    assert response.status_code == 400
    error = json.loads(response.body)["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "400"
    [entry] = monitor.snapshot()
    assert entry["status"] == "error"


def test_responses_non_streaming_transport_error_preserves_detail(monkeypatch):
    def handle(request):
        raise httpx.ConnectError("connection refused", request = request)

    async def run():
        from core.inference.api_monitor import ApiMonitor
        from routes import inference

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inference, "api_monitor", monitor)
            payload = ChatCompletionRequest(
                messages = [{"role": "user", "content": "Hi"}],
                stream = False,
                provider_type = "custom",
                provider_base_url = "https://gateway.example/v1",
                provider_api_type = "responses",
                external_model = "gateway-model",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = False),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            return await inference._proxy_to_external_provider(payload, request)

    response = asyncio.run(run())
    assert response.status_code == 502
    assert "connection refused" in json.loads(response.body)["error"]["message"]


@pytest.mark.parametrize("upstream_status", [400, 401, 429, 503])
def test_responses_non_streaming_route_preserves_upstream_error_status(
    monkeypatch, upstream_status
):
    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            upstream_status,
            json = {"error": {"message": "upstream rejected request"}},
        )

    async def run():
        from core.inference.api_monitor import ApiMonitor
        from routes import inference

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            monkeypatch.setattr(inference, "api_monitor", ApiMonitor(max_entries = 3))
            payload = ChatCompletionRequest(
                messages = [{"role": "user", "content": "Hi"}],
                stream = False,
                provider_type = "custom",
                provider_base_url = "https://gateway.example/v1",
                provider_api_type = "responses",
                external_model = "gateway-model",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = False),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            return await inference._proxy_to_external_provider(payload, request)

    response = asyncio.run(run())
    assert response.status_code == upstream_status
    error = json.loads(response.body)["error"]
    assert error["message"]
    assert error["code"] == str(upstream_status)


@pytest.mark.parametrize(
    ("retry_after", "expected_header"),
    [
        pytest.param("30", "30", id = "delay-seconds"),
        pytest.param(
            "Wed, 21 Oct 2099 07:28:00 GMT",
            "Wed, 21 Oct 2099 07:28:00 GMT",
            id = "http-date",
        ),
        pytest.param("Wed, 21 Oct 2099 07:28:00 +0000", None, id = "non-gmt-date"),
        pytest.param("30\r\nX-Evil: injected", None, id = "header-injection"),
        pytest.param("\r\n30", None, id = "leading-crlf"),
        pytest.param("30\r\n", None, id = "trailing-crlf"),
        pytest.param("\t30", None, id = "leading-tab"),
        pytest.param("30\t", None, id = "trailing-tab"),
        pytest.param("\x0030", None, id = "leading-nul"),
        pytest.param("30\x7f", None, id = "trailing-del"),
        pytest.param("soon", None, id = "invalid"),
    ],
)
def test_responses_non_streaming_route_forwards_only_safe_retry_after(
    monkeypatch,
    retry_after,
    expected_header,
):
    async def fake_stream_chat_completion(self, **kwargs):
        yield json.dumps(
            {
                "error": {
                    "message": "rate limited",
                    "type": "provider_error",
                    "code": "429",
                    "provider": "custom",
                    "retry_after": retry_after,
                }
            }
        )

    async def run():
        from core.inference.api_monitor import ApiMonitor
        from routes import inference

        monkeypatch.setattr(
            ExternalProviderClient,
            "stream_chat_completion",
            fake_stream_chat_completion,
        )
        monkeypatch.setattr(inference, "api_monitor", ApiMonitor(max_entries = 3))
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "Hi"}],
            stream = False,
            provider_type = "custom",
            provider_base_url = "https://gateway.example/v1",
            provider_api_type = "responses",
            external_model = "gateway-model",
        )

        async def disconnected():
            return False

        request = SimpleNamespace(
            headers = {},
            state = SimpleNamespace(skip_api_monitor = False),
            url = SimpleNamespace(path = "/v1/chat/completions"),
            method = "POST",
            is_disconnected = disconnected,
        )
        return await inference._proxy_to_external_provider(payload, request)

    response = asyncio.run(run())
    assert response.status_code == 429
    assert json.loads(response.body)["error"]["retry_after"] == retry_after
    assert response.headers.get("Retry-After") == expected_header


def test_custom_responses_follow_up_preserves_reasoning_metadata():
    from routes.inference import _build_external_messages

    extra_content = {
        "openai_responses_reasoning": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "check docs"}],
                "encrypted_content": "enc_blob",
            }
        ]
    }
    messages = [
        ChatMessage(
            role = "assistant",
            content = None,
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
            extra_content = extra_content,
        ),
        ChatMessage(role = "tool", tool_call_id = "call_1", content = "result"),
    ]

    built = _build_external_messages(
        messages,
        supports_vision = False,
        provider_type = "custom",
        api_type = "responses",
    )
    assert built[0]["extra_content"] == extra_content


def test_custom_responses_image_edit_parts_reach_the_provider_body(monkeypatch):
    captured = {}

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content = (
                b'data: {"type":"response.completed",'
                b'"response":{"output":[],"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
            ),
            headers = {"content-type": "text/event-stream"},
        )

    async def run():
        from routes import inference

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            payload = ChatCompletionRequest(
                messages = [
                    {"role": "user", "content": "Draw a cat"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "reasoning",
                                "id": "rs_abc",
                                "summary": [{"type": "summary_text", "text": "edit it"}],
                                "status": "completed",
                            },
                            {"type": "image_generation_call", "id": "img_abc"},
                        ],
                    },
                    {"role": "user", "content": "Make it blue"},
                ],
                stream = True,
                provider_type = "custom",
                provider_base_url = "https://api.openai.com/v1",
                provider_api_type = "responses",
                external_model = "gpt-5.5",
                enabled_tools = ["image_generation"],
                reasoning_effort = "medium",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = True),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            response = await inference._proxy_to_external_provider(payload, request)
            return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(run())
    assert chunks
    replay = [
        item
        for item in captured["body"]["input"]
        if item.get("type") in ("reasoning", "image_generation_call")
    ]
    assert replay == [
        {
            "type": "reasoning",
            "id": "rs_abc",
            "summary": [{"type": "summary_text", "text": "edit it"}],
        },
        {"type": "image_generation_call", "id": "img_abc"},
    ]
    assert {"type": "image_generation", "action": "edit"} in captured["body"]["tools"]


@pytest.mark.parametrize("stream", [False, True])
def test_custom_responses_image_edit_missing_reasoning_keeps_error_format(
    monkeypatch, stream
):
    requests = []

    def handle(request):
        requests.append(request)
        raise AssertionError("missing reasoning must fail before upstream dispatch")

    async def run():
        from routes import inference

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            payload = ChatCompletionRequest(
                messages = [
                    {"role": "user", "content": "Draw a cat"},
                    {
                        "role": "assistant",
                        "content": [{"type": "image_generation_call", "id": "img_abc"}],
                    },
                    {"role": "user", "content": "Make it blue"},
                ],
                stream = stream,
                provider_type = "custom",
                provider_base_url = "https://api.openai.com/v1",
                provider_api_type = "responses",
                external_model = "gpt-5.5",
                enabled_tools = ["image_generation"],
                reasoning_effort = "medium",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = True),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            response = await inference._proxy_to_external_provider(payload, request)
            if stream:
                return response, "".join([chunk async for chunk in response.body_iterator])
            return response, response.body.decode()

    response, body = asyncio.run(run())
    assert requests == []
    if stream:
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"
        assert body.startswith("data: {")
        error = json.loads(body.split("\n", 1)[0].removeprefix("data: "))["error"]
    else:
        assert response.status_code == 400
        assert response.media_type == "application/json"
        error = json.loads(body)["error"]
    assert error["code"] == "400"
    assert "missing paired reasoning state" in error["message"]


@pytest.mark.parametrize("supports_vision", [False, True])
def test_custom_chat_completions_still_drops_responses_native_parts(supports_vision):
    from routes.inference import _build_external_messages

    built = _build_external_messages(
        [
            ChatMessage(
                role = "assistant",
                content = [
                    {
                        "type": "reasoning",
                        "id": "rs_abc",
                        "summary": [],
                    },
                    {"type": "image_generation_call", "id": "img_abc"},
                ],
            ),
            ChatMessage(role = "user", content = "Next"),
        ],
        supports_vision = supports_vision,
        provider_type = "custom",
        api_type = "chat_completions",
    )
    assert built == [{"role": "user", "content": "Next"}]


def test_responses_non_streaming_failure_is_not_reported_as_completion(monkeypatch):
    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json = {
                "id": "resp_failed",
                "status": "failed",
                "error": {"message": "provider unavailable"},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert len(lines) == 1
    assert json.loads(lines[0])["error"]["message"] == "provider unavailable"


@pytest.fixture()
def provider_api(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth import storage as auth_storage
    from auth.authentication import (
        authenticated_via_api_key,
        get_current_credential,
        get_current_subject,
    )
    from routes import providers
    from storage import credential_secrets

    db = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: db)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    providers_db.reset_schema_state_for_tests()
    credential_secrets._schema_ready.clear()
    app = FastAPI()
    app.include_router(providers.router, prefix = "/providers")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[get_current_credential] = lambda: ("unsloth", None)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    with TestClient(app) as client:
        yield client


def test_api_create_update_and_saved_target_binding(provider_api):
    from routes.providers import _bind_saved_provider_target
    from models.providers import ProviderTestRequest

    created = provider_api.post(
        "/providers/",
        json = {
            "provider_type": "custom",
            "display_name": "Gateway",
            "base_url": "https://gateway.example/v1",
            "api_type": "responses",
        },
    )
    assert created.status_code == 201
    saved = created.json()
    assert saved["api_type"] == "responses"
    assert provider_api.get("/providers/").json()[0]["api_type"] == "responses"
    renamed = provider_api.put(f"/providers/{saved['id']}", json = {"display_name": "Renamed"})
    assert renamed.json()["api_type"] == "responses"
    bound = _bind_saved_provider_target(
        ProviderTestRequest(
            provider_id = saved["id"],
            provider_type = "openai",
            base_url = "https://other.example/v1",
            api_type = "chat_completions",
        )
    )
    assert bound.provider_type == "custom"
    assert bound.base_url == "https://gateway.example/v1"
    assert bound.api_type == "responses"
    changed = provider_api.put(f"/providers/{saved['id']}", json = {"api_type": "chat_completions"})
    assert changed.status_code == 200
    assert changed.json()["api_type"] == "chat_completions"


@pytest.mark.parametrize("models_status", [200, 404])
@pytest.mark.parametrize("status", [200, 500])
def test_responses_only_connectivity_probe(provider_api, monkeypatch, models_status, status):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(
                models_status,
                json = {"data": [{"id": "responses-only"}]} if models_status == 200 else None,
            )
        assert request.url.path == "/v1/responses"
        assert "input" in json.loads(request.content)
        if status == 500:
            return httpx.Response(500, json = {"error": {"message": "provider unavailable"}})
        return httpx.Response(
            200,
            text = 'data: {"type":"response.output_text.delta","delta":"Hello"}\n\ndata: {"type":"response.completed","response":{"id":"resp_probe"}}\n\n',
        )

    async def run():
        from routes.providers import test_provider
        from models.providers import ProviderTestRequest
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            return await test_provider(
                ProviderTestRequest(
                    provider_type = "custom",
                    base_url = "https://gateway.example/v1",
                    api_type = "responses",
                    model_id = "responses-only",
                ),
                _current_subject = "unsloth",
                via_api_key = False,
            )

    result = asyncio.run(run())
    assert result.success == (status == 200)
    assert paths == ["/v1/models", "/v1/responses"]


def test_responses_connectivity_uses_catalog_model_when_model_is_omitted(monkeypatch):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json = {"data": [{"id": "responses-only"}]})
        assert request.url.path == "/v1/responses"
        assert json.loads(request.content)["model"] == "responses-only"
        return httpx.Response(
            200,
            text = 'data: {"type":"response.output_text.delta","delta":"Hello"}\n\ndata: {"type":"response.completed","response":{"id":"resp_probe"}}\n\n',
        )

    async def run():
        from models.providers import ProviderTestRequest
        from routes.providers import test_provider

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            return await test_provider(
                ProviderTestRequest(
                    provider_type = "custom",
                    base_url = "https://gateway.example/v1",
                    api_type = "responses",
                ),
                _current_subject = "unsloth",
                via_api_key = False,
            )

    result = asyncio.run(run())
    assert result.success is True
    assert paths == ["/v1/models", "/v1/responses"]


def test_responses_connectivity_tries_later_catalog_model(monkeypatch):
    probed_models = []

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json = {"data": [{"id": "chat-only"}, {"id": "responses-capable"}]},
            )
        assert request.url.path == "/v1/responses"
        model = json.loads(request.content)["model"]
        probed_models.append(model)
        if model == "chat-only":
            return httpx.Response(
                200,
                text = 'data: {"type":"response.output_text.delta","delta":"partial"}\n\ndata: {"type":"response.failed","response":{"error":{"message":"unsupported model"}}}\n\n',
            )
        return httpx.Response(
            200,
            text = 'data: {"type":"response.output_text.delta","delta":"Hello"}\n\ndata: {"type":"response.completed","response":{"id":"resp_probe"}}\n\n',
        )

    async def run():
        from models.providers import ProviderTestRequest
        from routes.providers import test_provider

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            return await test_provider(
                ProviderTestRequest(
                    provider_type = "custom",
                    base_url = "https://gateway.example/v1",
                    api_type = "responses",
                ),
                _current_subject = "unsloth",
                via_api_key = False,
            )

    result = asyncio.run(run())
    assert result.success is True
    assert probed_models == ["chat-only", "responses-capable"]


def test_responses_connectivity_caps_catalog_probes(monkeypatch):
    probed_models = []

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json = {"data": [{"id": f"model-{i}"} for i in range(20)]},
            )
        probed_models.append(json.loads(request.content)["model"])
        return httpx.Response(400, json = {"error": {"message": "unsupported model"}})

    async def run():
        from models.providers import ProviderTestRequest
        from routes.providers import _MAX_RESPONSES_CONNECTIVITY_MODELS, test_provider

        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            result = await test_provider(
                ProviderTestRequest(
                    provider_type = "custom",
                    base_url = "https://gateway.example/v1",
                    api_type = "responses",
                ),
                _current_subject = "unsloth",
                via_api_key = False,
            )
            return result, _MAX_RESPONSES_CONNECTIVITY_MODELS

    result, probe_limit = asyncio.run(run())
    assert result.success is False
    assert probed_models == [f"model-{i}" for i in range(probe_limit)]


def test_responses_connectivity_bounds_each_stalled_stream(monkeypatch):
    from routes import providers as provider_routes

    class StalledResponsesClient:
        def __init__(self):
            self.probed_models = []
            self.closed_models = []

        async def list_models(self):
            return [{"id": "stalled-a"}, {"id": "stalled-b"}]

        async def stream_chat_completion(self, *, model, **_kwargs):
            self.probed_models.append(model)
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                self.closed_models.append(model)

    async def run():
        client = StalledResponsesClient()
        monkeypatch.setattr(provider_routes, "_PROVIDER_CONNECTIVITY_TIMEOUT_SECONDS", 0.01)
        result = await asyncio.wait_for(
            provider_routes._test_custom_provider_connectivity(
                client, "", api_type = "responses"
            ),
            timeout = 0.5,
        )
        return client, result

    client, result = asyncio.run(run())
    assert result.success is False
    assert result.message == "Connection failed: Responses endpoint timed out after 0.01 seconds."
    assert client.probed_models == ["stalled-a", "stalled-b"]
    assert client.closed_models == ["stalled-a", "stalled-b"]
