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


@pytest.mark.parametrize("path", ["/v1/responses", "/v1/messages", "/v1/chat/completions"])
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
    if path == "/v1/chat/completions":
        # stream omitted: the collected answer must not wait for the provider to hang up.
        resp = v1.post(
            path, json = {"model": CONNECTION_MODEL, "messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp.json()["choices"][0]["message"]["content"] == "hi"
        return
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

    inf._remember_connection_extra("someone-else", "call", "call_1", SIGNED["extra_content"])
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


def _two_tool_calls(**kwargs):
    async def gen():
        for index, path in enumerate(("a.txt", "b.txt")):
            call = {
                "index": index,
                "id": f"call_{index}",
                "function": {"name": "Read", "arguments": json.dumps({"file_path": path})},
            }
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'tool_calls': [call]}}]})}"
        yield 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
        yield "data: [DONE]"

    return gen()


@pytest.mark.parametrize("stream", [True, False])
def test_messages_honour_disable_parallel_tool_use(v1, monkeypatch, stream):
    import routes.inference as inf

    class TwoCalls(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            return _two_tool_calls(**kwargs)

    monkeypatch.setattr(inf, "ExternalProviderClient", TwoCalls)
    body = _messages_body(
        stream = stream, tool_choice = {"type": "auto", "disable_parallel_tool_use": True}
    )
    resp = v1.post("/v1/messages", json = body)

    assert resp.status_code == 200, resp.text
    if stream:
        events = _sse(resp.text)
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert [e["content_block"]["type"] for e in starts] == ["tool_use"]
        indices = {e["index"] for e in events if "index" in e}
        assert indices == {0}
    else:
        blocks = resp.json()["content"]
        assert [b["type"] for b in blocks] == ["tool_use"]
        assert blocks[0]["input"] == {"file_path": "a.txt"}


def test_messages_keep_parallel_tool_calls_by_default(v1, monkeypatch):
    import routes.inference as inf

    class TwoCalls(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            return _two_tool_calls(**kwargs)

    monkeypatch.setattr(inf, "ExternalProviderClient", TwoCalls)
    blocks = v1.post("/v1/messages", json = _messages_body()).json()["content"]

    assert [b["input"] for b in blocks] == [{"file_path": "a.txt"}, {"file_path": "b.txt"}]


def test_messages_replay_an_anthropic_connections_signed_thinking(v1, monkeypatch):
    # Anthropic refuses a tool continuation whose assistant turn lost its signed thinking block.
    import routes.inference as inf

    native = [
        {"type": "thinking", "thinking": "plan", "signature": "sig-xyz"},
        {"type": "tool_use", "id": "toolu_abc", "name": "Read", "input": {"file_path": "a.txt"}},
    ]

    class Signed(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            FakeExternalClient.calls[-1]["stream"] = kwargs

            async def gen():
                call = {
                    "index": 0,
                    "id": "toolu_abc",
                    "function": {"name": "Read", "arguments": '{"file_path": "a.txt"}'},
                }
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'tool_calls': [call]}}]})}"
                extra = {"extra_content": {"anthropic": {"content": native}}}
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': extra}]})}"
                yield 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
                yield "data: [DONE]"

            return gen()

    monkeypatch.setattr(inf, "ExternalProviderClient", Signed)
    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "api_type": "chat_completions",
            "display_name": "Claude",
            "is_enabled": True,
            "reasoning_config": None,
        },
    )
    first = v1.post("/v1/messages", json = _messages_body()).json()
    tool_use = next(b for b in first["content"] if b["type"] == "tool_use")
    assert tool_use["id"] == "toolu_abc"

    history = [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": [tool_use]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "hi"}],
        },
    ]
    v1.post("/v1/messages", json = _messages_body(messages = history))

    assistant = next(
        m for m in FakeExternalClient.calls[-1]["stream"]["messages"] if m["role"] == "assistant"
    )
    assert assistant["extra_content"] == {"anthropic": {"content": native}}


def test_messages_forward_the_agents_sampling_controls(v1):
    v1.post(
        "/v1/messages",
        json = _messages_body(min_p = 0.05, repetition_penalty = 1.1, presence_penalty = 0.5),
    )

    sent = FakeExternalClient.calls[-1]["stream"]
    assert (sent["min_p"], sent["repetition_penalty"], sent["presence_penalty"]) == (
        0.05,
        1.1,
        0.5,
    )


@pytest.mark.parametrize(
    "code, err_type", [("401", "authentication_error"), ("404", "not_found_error")]
)
def test_messages_keep_the_connections_error_class(v1, monkeypatch, code, err_type):
    import routes.inference as inf

    class Failing(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                yield f'data: {{"error": {{"message": "nope", "code": "{code}"}}}}'
                yield "data: [DONE]"

            return gen()

    monkeypatch.setattr(inf, "ExternalProviderClient", Failing)
    events = _sse(v1.post("/v1/messages", json = _messages_body(stream = True)).text)
    assert events[-1]["error"]["type"] == err_type

    resp = v1.post("/v1/messages", json = _messages_body())
    assert resp.status_code == int(code)
    assert resp.json()["error"]["type"] == err_type


@pytest.mark.parametrize("stream", [True, False])
def test_messages_complete_interleaved_parallel_tool_calls(v1, monkeypatch, stream):
    import routes.inference as inf

    class Interleaved(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            async def gen():
                fragments = [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "Read", "arguments": '{"file_'},
                    },
                    {
                        "index": 1,
                        "id": "call_b",
                        "function": {"name": "Read", "arguments": '{"file_'},
                    },
                    {"index": 0, "function": {"arguments": 'path": "a.txt"}'}},
                    {"index": 1, "function": {"arguments": 'path": "b.txt"}'}},
                ]
                for part in fragments:
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'tool_calls': [part]}}]})}"
                yield 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}'
                yield "data: [DONE]"

            return gen()

    monkeypatch.setattr(inf, "ExternalProviderClient", Interleaved)
    resp = v1.post("/v1/messages", json = _messages_body(stream = stream))

    assert resp.status_code == 200, resp.text
    if stream:
        events = _sse(resp.text)
        args: dict[int, str] = {}
        for event in events:
            if (
                event["type"] == "content_block_delta"
                and event["delta"]["type"] == "input_json_delta"
            ):
                # A delta after its block stopped would reach the client as a finished call with {}.
                assert event["index"] not in {
                    e["index"]
                    for e in events[: events.index(event)]
                    if e["type"] == "content_block_stop"
                }
                args[event["index"]] = args.get(event["index"], "") + event["delta"]["partial_json"]
        inputs = [json.loads(text) for _, text in sorted(args.items())]
    else:
        inputs = [b["input"] for b in resp.json()["content"] if b["type"] == "tool_use"]
    assert inputs == [{"file_path": "a.txt"}, {"file_path": "b.txt"}]


def test_messages_keep_earlier_thinking_for_a_deepseek_connection(v1, monkeypatch):
    # DeepSeek answers 400 when tools ride on a history whose reasoning_content was dropped.
    import routes.inference as inf

    monkeypatch.setattr(
        inf.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "api_type": "chat_completions",
            "display_name": "DeepSeek",
            "is_enabled": True,
            "reasoning_config": None,
        },
    )
    history = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "earlier thought", "signature": ""},
                {"type": "text", "text": "done"},
            ],
        },
        {"role": "user", "content": "second"},
    ]
    v1.post("/v1/messages", json = _messages_body(messages = history))

    assistant = next(
        m for m in FakeExternalClient.calls[-1]["stream"]["messages"] if m["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == "earlier thought"


def _content_stream(*pieces):
    class Content(FakeExternalClient):
        def stream_chat_completion(self, **kwargs):
            FakeExternalClient.calls[-1]["stream"] = kwargs

            async def gen():
                for piece in pieces:
                    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': piece}}]})}"
                yield 'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}'
                yield "data: [DONE]"

            return gen()

    return Content


@pytest.mark.parametrize("stream", [True, False])
def test_messages_turn_native_thinking_markup_into_a_thinking_block(v1, monkeypatch, stream):
    # The native Anthropic translator streams thinking as <think> markup, the Studio chat's form.
    import routes.inference as inf

    monkeypatch.setattr(
        inf, "ExternalProviderClient", _content_stream("<think>plan", "ning</think>ans", "wer")
    )
    body = _messages_body(stream = stream, thinking = {"type": "enabled", "budget_tokens": 1024})
    resp = v1.post("/v1/messages", json = body)

    assert resp.status_code == 200, resp.text
    if stream:
        events = _sse(resp.text)
        kinds = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
        text = "".join(
            e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta"
        )
        thinking = "".join(
            e["delta"].get("thinking", "") for e in events if e["type"] == "content_block_delta"
        )
        assert kinds == ["thinking", "text"]
        assert (thinking, text) == ("planning", "answer")
    else:
        blocks = resp.json()["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text"]
        assert blocks[0]["thinking"] == "planning"
        assert blocks[1]["text"] == "answer"


@pytest.mark.parametrize("stream", [True, False])
def test_messages_stop_at_a_requested_stop_sequence(v1, monkeypatch, stream):
    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", _content_stream("before E", "ND after"))
    body = _messages_body(stream = stream, stop_sequences = ["END"])
    resp = v1.post("/v1/messages", json = body)

    assert resp.status_code == 200, resp.text
    if stream:
        events = _sse(resp.text)
        text = "".join(
            e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta"
        )
        delta = next(e for e in events if e["type"] == "message_delta")["delta"]
    else:
        message = resp.json()
        text = "".join(b.get("text", "") for b in message["content"])
        delta = message
    assert text == "before "
    assert (delta["stop_reason"], delta["stop_sequence"]) == ("stop_sequence", "END")


def test_messages_without_a_stop_hit_keep_all_text(v1, monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr(inf, "ExternalProviderClient", _content_stream("before E", "N after"))
    message = v1.post("/v1/messages", json = _messages_body(stop_sequences = ["END"])).json()

    assert "".join(b.get("text", "") for b in message["content"]) == "before EN after"
    assert message["stop_reason"] == "end_turn"
