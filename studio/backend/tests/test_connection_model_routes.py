# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest

USER_KEY = "sk-unsloth-deadbeefdeadbeef"
SCRIPT_KEY = "sk-unsloth-0123456789abcdef"
CONNECTION_MODEL = "external::c1::models%2Fgemini-2.5-flash"
TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}


class FakeExternalClient:
    calls: list = []
    tool_extra: dict = {}

    def __init__(self, **kwargs):
        FakeExternalClient.calls.append({"ctor": kwargs})

    def stream_chat_completion(self, **kwargs):
        FakeExternalClient.calls[-1]["stream"] = kwargs

        async def gen():
            for chunk in (
                {"choices": [{"index": 0, "delta": {"content": "Reading it."}}]},
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "Read", "arguments": '{"file_'},
                                        **FakeExternalClient.tool_extra,
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": 'path": "a.txt"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
            ):
                yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"

        return gen()

    async def served_context_window(self, model):
        return None

    async def close(self):
        return None


class NoLocalBackend:
    def __getattr__(self, name):
        raise AssertionError(f"a Connection request touched the local GGUF ({name})")


@pytest.fixture()
def v1(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import auth.authentication as authentication
    import routes.inference as inf
    from auth.authentication import get_current_subject
    from utils.api_errors import install_api_error_handlers

    FakeExternalClient.calls = []
    FakeExternalClient.tool_extra = {}
    inf._CONNECTION_TOOL_EXTRAS.clear()
    allowed = []

    def resolve(
        provider_id,
        encrypted,
        *,
        allow_saved_key = True,
    ):
        allowed.append(allow_saved_key)
        return "saved-key" if allow_saved_key else ""

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "custom",
            "base_url": "http://10.0.0.5:8080/v1",
            "api_type": "chat_completions",
            "display_name": "Box",
            "is_enabled": True,
            "reasoning_config": None,
        },
    )
    monkeypatch.setattr(inf, "resolve_provider_api_key_or_400", resolve)
    monkeypatch.setattr(inf, "ExternalProviderClient", FakeExternalClient)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: NoLocalBackend())
    monkeypatch.setattr(inf, "current_date_prompt_line", lambda **_kwargs: "")
    monkeypatch.setattr(inf.auth_storage, "is_internal_api_key", lambda raw: False)
    monkeypatch.setattr(
        inf.auth_storage,
        "validate_api_key_account",
        lambda raw, touch = True: {
            USER_KEY: ({"role": "owner", "key_name": "Coding agents (unsloth start)"}, "s"),
            SCRIPT_KEY: ({"role": "owner", "key_name": "my script"}, "s"),
        }.get(raw),
    )
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: False)

    app = FastAPI()
    app.include_router(inf.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {USER_KEY}"
    client.allowed = allowed
    return client


def _messages_body(**overrides):
    body = {
        "model": CONNECTION_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "read a.txt"}],
        "tools": [TOOL],
    }
    body.update(overrides)
    return body


def _sse(text):
    return [
        json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")
    ]


def test_messages_streams_a_connection_model_with_tool_use(v1):
    resp = v1.post("/v1/messages", json = _messages_body(stream = True))

    assert resp.status_code == 200, resp.text
    events = _sse(resp.text)
    assert [e["type"] for e in events][0] == "message_start"
    assert events[-1]["type"] == "message_stop"
    starts = [e["content_block"] for e in events if e["type"] == "content_block_start"]
    assert [b["type"] for b in starts] == ["text", "tool_use"]
    assert starts[1]["name"] == "Read"
    args = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"file_path": "a.txt"}
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["output_tokens"] == 7

    [call] = FakeExternalClient.calls
    assert call["ctor"]["base_url"] == "http://10.0.0.5:8080/v1"
    assert call["ctor"]["api_key"] == "saved-key"
    assert call["stream"]["model"] == "models/gemini-2.5-flash"
    assert call["stream"]["tools"][0]["function"]["name"] == "Read"
    assert v1.allowed == [True]


def test_messages_non_streaming_assembles_the_connection_reply(v1):
    resp = v1.post("/v1/messages", json = _messages_body())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == CONNECTION_MODEL
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0] == {"type": "text", "text": "Reading it."}
    assert body["content"][1]["name"] == "Read"
    assert body["content"][1]["input"] == {"file_path": "a.txt"}


def test_messages_replays_tool_results_to_the_connection(v1):
    history = [
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello"}],
        },
    ]
    resp = v1.post("/v1/messages", json = _messages_body(messages = history, stream = True))

    assert resp.status_code == 200, resp.text
    sent = FakeExternalClient.calls[0]["stream"]["messages"]
    assert sent[-2]["tool_calls"][0]["id"] == "toolu_1"
    assert sent[-1] == {"role": "tool", "content": "hello", "tool_call_id": "toolu_1"}


def test_messages_surfaces_a_connection_failure_as_an_anthropic_error(v1, monkeypatch):
    class Failing(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield 'data: {"error": {"message": "quota exceeded", "code": "429"}}'

            return gen()

    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", Failing)
    events = _sse(v1.post("/v1/messages", json = _messages_body(stream = True)).text)
    assert events[-1]["type"] == "error"
    assert events[-1]["error"] == {"type": "rate_limit_error", "message": "quota exceeded"}

    resp = v1.post("/v1/messages", json = _messages_body())
    assert resp.status_code == 429
    assert resp.json()["error"]["message"] == "quota exceeded"


def test_chat_completions_forwards_a_connection_model(v1):
    resp = v1.post(
        "/v1/chat/completions",
        json = {
            "model": CONNECTION_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200, resp.text
    assert FakeExternalClient.calls[0]["stream"]["model"] == "models/gemini-2.5-flash"
    assert v1.allowed == [True]


def test_responses_streams_a_connection_model_for_codex(v1):
    tool = {
        "type": "function",
        "name": "Read",
        "description": "Read a file",
        "parameters": TOOL["input_schema"],
    }
    resp = v1.post(
        "/v1/responses",
        json = {"model": CONNECTION_MODEL, "stream": True, "input": "read a.txt", "tools": [tool]},
    )

    assert resp.status_code == 200, resp.text
    events = _sse(resp.text)
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == (
        "Reading it."
    )
    [call] = [
        e["item"]
        for e in events
        if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
    ]
    assert call["name"] == "Read"
    assert json.loads(call["arguments"]) == {"file_path": "a.txt"}
    assert events[-1]["response"]["usage"]["output_tokens"] == 7
    [upstream] = FakeExternalClient.calls
    assert upstream["stream"]["model"] == "models/gemini-2.5-flash"
    assert upstream["stream"]["tools"][0]["function"]["name"] == "Read"
    assert v1.allowed == [True]


def test_responses_answers_a_connection_model_without_streaming(v1):
    resp = v1.post("/v1/responses", json = {"model": CONNECTION_MODEL, "input": "read a.txt"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Reading it."
    assert body["output"][1]["type"] == "function_call"
    assert json.loads(body["output"][1]["arguments"]) == {"file_path": "a.txt"}
    assert body["usage"]["input_tokens"] == 11


def test_responses_reports_the_connection_failure(v1, monkeypatch):
    class Failing(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield 'data: {"error": {"message": "API key expired.", "code": "401"}}'
                yield "data: [DONE]"

            return gen()

    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", Failing)
    events = _sse(
        v1.post(
            "/v1/responses", json = {"model": CONNECTION_MODEL, "input": "ping", "stream": True}
        ).text
    )
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["message"] == "API key expired."

    resp = v1.post("/v1/responses", json = {"model": CONNECTION_MODEL, "input": "ping"})
    assert resp.status_code == 502
    assert "API key expired." in resp.text


def test_responses_tells_codex_a_connection_is_rate_limited(v1, monkeypatch):
    class Limited(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield 'data: {"error": {"message": "Quota exceeded.", "code": "429"}}'

            return gen()

    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", Limited)
    body = {"model": CONNECTION_MODEL, "input": "ping", "stream": True}
    events = _sse(v1.post("/v1/responses", json = body).text)
    assert events[-1]["response"]["error"] == {
        "code": "rate_limit_exceeded",
        "message": "Quota exceeded.",
    }

    resp = v1.post("/v1/responses", json = {"model": CONNECTION_MODEL, "input": "ping"})
    assert resp.status_code == 429


def test_count_tokens_cannot_serve_a_connection_model(v1):
    resp = v1.post("/v1/messages/count_tokens", json = _messages_body())

    assert resp.status_code == 400
    assert "Connection model" in resp.text
    assert FakeExternalClient.calls == []


def test_a_keyless_caller_cannot_reach_a_connection(v1, monkeypatch):
    import auth.authentication as authentication

    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda r: True)
    resp = v1.post("/v1/messages", json = _messages_body(stream = True))

    assert resp.status_code == 403
    assert FakeExternalClient.calls == []


def test_a_recipe_key_still_cannot_spend_the_saved_credential(v1, monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr(inf.auth_storage, "is_internal_api_key", lambda raw: True)
    monkeypatch.setattr(inf.auth_storage, "internal_api_key_name", lambda raw: "data-recipe")
    v1.post("/v1/messages", json = _messages_body(stream = True))

    assert v1.allowed == [False]
    assert FakeExternalClient.calls[0]["ctor"]["api_key"] == ""


def test_an_unknown_key_cannot_spend_the_saved_credential(v1):
    v1.headers["Authorization"] = "Bearer sk-unsloth-notarealkey0000"
    v1.post("/v1/messages", json = _messages_body(stream = True))

    assert v1.allowed == [False]


def test_another_user_key_reaches_the_connection_without_its_saved_key(v1):
    v1.headers["Authorization"] = f"Bearer {SCRIPT_KEY}"
    v1.post("/v1/messages", json = _messages_body(stream = True))

    assert v1.allowed == [False]
    assert FakeExternalClient.calls[0]["ctor"]["api_key"] == ""


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/messages"])
def test_a_connection_stream_ends_at_done_even_if_the_provider_keeps_it_open(v1, monkeypatch, path):
    import routes.inference as inf

    class KeepsOpen(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield 'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}'
                yield 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'
                yield "data: [DONE]"
                await asyncio.sleep(3600)

            return gen()

    monkeypatch.setattr(inf, "ExternalProviderClient", KeepsOpen)
    body = (
        {"model": CONNECTION_MODEL, "stream": True, "input": "hi"}
        if path == "/v1/responses"
        else _messages_body(stream = True)
    )
    events = _sse(v1.post(path, json = body).text)

    assert events[-1]["type"] in ("response.completed", "message_stop")


def test_a_connection_stream_without_a_space_after_data_still_reaches_the_agent(v1, monkeypatch):
    class Compact(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield 'data:{"choices": [{"index": 0, "delta": {"content": "pong"}}]}'
                yield 'data:{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'
                yield "data:[DONE]"

            return gen()

    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", Compact)
    message = v1.post("/v1/messages", json = _messages_body()).json()
    assert message["content"] == [{"type": "text", "text": "pong"}]

    response = v1.post("/v1/responses", json = {"model": CONNECTION_MODEL, "input": "ping"}).json()
    assert response["output"][0]["content"][0]["text"] == "pong"


SIGNED = {"extra_content": {"google": {"thought_signature": "sig-abc"}}}


@pytest.fixture()
def gemini(v1, monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "api_type": "chat_completions",
            "display_name": "Gemini",
            "is_enabled": True,
            "reasoning_config": None,
        },
    )
    FakeExternalClient.tool_extra = SIGNED
    return v1


def _replayed_call(messages, tool_id):
    return next(
        call
        for message in messages
        for call in (message.get("tool_calls") or [])
        if call.get("id") == tool_id
    )


def test_messages_replays_a_tool_calls_thought_signature(gemini):
    # Gemini 3 rejects a replayed functionCall without its thoughtSignature; Claude Code drops it.
    v1 = gemini
    first = v1.post("/v1/messages", json = _messages_body()).json()
    tool_use = first["content"][1]
    assert tool_use["type"] == "tool_use"

    history = [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": [tool_use]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "hi"}],
        },
    ]
    resp = v1.post("/v1/messages", json = _messages_body(messages = history))

    assert resp.status_code == 200, resp.text
    call = _replayed_call(FakeExternalClient.calls[-1]["stream"]["messages"], tool_use["id"])
    assert call["extra_content"] == SIGNED["extra_content"]


def test_responses_replays_a_tool_calls_thought_signature(gemini):
    v1 = gemini
    first = v1.post("/v1/responses", json = {"model": CONNECTION_MODEL, "input": "read a.txt"})
    assert first.status_code == 200, first.text
    call_item = next(i for i in first.json()["output"] if i["type"] == "function_call")

    resp = v1.post(
        "/v1/responses",
        json = {
            "model": CONNECTION_MODEL,
            "input": [
                {"role": "user", "content": "read a.txt"},
                call_item,
                {"type": "function_call_output", "call_id": call_item["call_id"], "output": "hi"},
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    call = _replayed_call(FakeExternalClient.calls[-1]["stream"]["messages"], call_item["call_id"])
    assert call["extra_content"] == SIGNED["extra_content"]


def test_a_signature_is_only_replayed_for_its_own_account(v1):
    import routes.inference as inf

    inf._remember_connection_tool_extra("someone-else", "call_1", SIGNED["extra_content"])
    messages = [{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function"}]}]

    inf._restore_connection_tool_extras("owner", messages)

    assert "extra_content" not in messages[0]["tool_calls"][0]


def test_chat_completions_without_stream_returns_one_completion_object(v1):
    FakeExternalClient.tool_extra = SIGNED
    resp = v1.post(
        "/v1/chat/completions",
        json = {"model": CONNECTION_MODEL, "messages": [{"role": "user", "content": "read a.txt"}]},
    )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["object"] == "chat.completion"
    message = body["choices"][0]["message"]
    assert message["content"] == "Reading it."
    assert message["tool_calls"][0]["function"] == {
        "name": "Read",
        "arguments": '{"file_path": "a.txt"}',
    }
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["completion_tokens"] == 7
    assert FakeExternalClient.calls[-1]["stream"]["stream"] is True


def test_messages_forwards_an_output_format_to_the_connection(v1):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    resp = v1.post(
        "/v1/messages",
        json = _messages_body(output_config = {"format": {"type": "json_schema", "schema": schema}}),
    )

    assert resp.status_code == 200, resp.text
    assert FakeExternalClient.calls[-1]["stream"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }
