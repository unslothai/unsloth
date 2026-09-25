# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.inference as inf_mod
from core.inference.api_monitor import ApiMonitor
from core.inference.llama_admission import reset_llama_admission_queues
from models.inference import AnthropicMessagesRequest
from routes.inference import anthropic_messages
from state.tool_policy import reset_tool_policy

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
_ANSWER = '{"name": "Ada"}'
_CLIENT_TOOL = {"name": "lookup", "description": "Look up", "input_schema": {"type": "object"}}


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    reset_tool_policy()
    reset_llama_admission_queues()
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
    monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
    monkeypatch.setattr(inf_mod, "current_date_prompt_line", lambda **_kwargs: "")
    yield
    reset_llama_admission_queues()
    reset_tool_policy()


class _Request:
    def __init__(self):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"

    async def is_disconnected(self):
        return False


def _install(
    monkeypatch,
    *,
    supports_tool_passthrough = False,
    answer = _ANSWER,
    **overrides,
):
    calls = []
    upstream = []

    def _gen_plain(**kwargs):
        calls.append(("plain", kwargs))
        yield "not json"

    def _gen_tools(**kwargs):
        calls.append(("tools", kwargs))
        yield {"type": "content", "text": "not json"}

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        supports_tool_passthrough = supports_tool_passthrough,
        model_identifier = "test-model",
        context_length = 2048,
        count_chat_tokens = lambda *a, **k: 2,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
        effective_parallel_slots = 1,
        base_url = "http://llama.structured.test",
    )
    for key, value in overrides.items():
        setattr(backend, key, value)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        upstream.append(body)
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": answer}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            content = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(
                200, content = content.encode(), headers = {"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **k: real_client(transport = transport, timeout = k.get("timeout", 60)),
    )
    return calls, upstream


def _payload(**fields) -> AnthropicMessagesRequest:
    base = {"max_tokens": 64, "messages": [{"role": "user", "content": "Name a scientist."}]}
    base.update(fields)
    return AnthropicMessagesRequest(**base)


def _run(payload):
    async def _go():
        response = await anthropic_messages(payload, request = _Request(), current_subject = "t")
        if payload.stream:
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return response.status_code, "".join(chunks)
        return response.status_code, response.body.decode()

    return asyncio.run(_go())


_EXPECTED = {"type": "json_schema", "json_schema": {"name": "response", "schema": _SCHEMA}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "fields",
    [
        {"output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}}},
        {"output_format": {"type": "json_schema", "schema": _SCHEMA}},
    ],
    ids = ["output_config", "legacy_output_format"],
)
def test_format_reaches_llama_server_as_response_format(monkeypatch, fields, stream):
    calls, upstream = _install(monkeypatch)

    status, body = _run(_payload(stream = stream, **fields))

    assert status == 200
    assert calls == []
    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert "tools" not in sent
    if stream:
        assert json.dumps(_ANSWER)[1:-1] in body
    else:
        assert json.loads(json.loads(body)["content"][0]["text"]) == {"name": "Ada"}


@pytest.mark.parametrize("stream", [False, True])
def test_format_rides_along_with_uncallable_client_tools(monkeypatch, stream):
    _calls, upstream = _install(monkeypatch, supports_tool_passthrough = True)

    _run(
        _payload(
            stream = stream,
            tools = [_CLIENT_TOOL],
            tool_choice = {"type": "none"},
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert "tools" not in sent
    assert "tool_choice" not in sent


def test_format_keeps_client_tools_needed_by_replayed_history(monkeypatch):
    _calls, upstream = _install(monkeypatch, supports_tool_passthrough = True)

    _run(
        _payload(
            messages = [
                {"role": "user", "content": "Look up a scientist."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"name": "Ada"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Ada"}
                    ],
                },
            ],
            tools = [_CLIENT_TOOL],
            tool_choice = {"type": "none"},
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert [tool["function"]["name"] for tool in sent["tools"]] == ["lookup"]
    assert sent["tool_choice"] == "none"


@pytest.mark.parametrize(
    "fmt",
    [{"type": "bogus"}, {"type": "json_schema"}, "json_schema"],
    ids = ["unknown-type", "missing-schema", "not-an-object"],
)
def test_unsupported_format_is_ignored_as_before(monkeypatch, fmt):
    calls, upstream = _install(monkeypatch)

    status, _body = _run(_payload(output_config = {"format": fmt}))

    assert status == 200
    assert [path for path, _kwargs in calls] == ["plain"]
    assert upstream == []


@pytest.mark.parametrize(
    "fields",
    [
        {"enable_tools": True, "permission_mode": "off"},
        {"tools": [_CLIENT_TOOL]},
        {"tools": [_CLIENT_TOOL], "tool_choice": {"type": "auto"}},
    ],
    ids = ["server-tools", "client-tools", "client-tools-auto"],
)
def test_format_with_callable_tools_keeps_tools_callable(monkeypatch, fields):
    calls, upstream = _install(monkeypatch, supports_tool_passthrough = True)
    payload = _payload(
        output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}}, **fields
    )

    status, _body = _run(payload)

    assert status == 200
    assert calls or upstream
    assert all("response_format" not in body for body in upstream)
    assert all(kwargs.get("response_format") is None for _path, kwargs in calls)


def test_request_without_format_stays_on_the_plain_path(monkeypatch):
    calls, upstream = _install(monkeypatch)

    status, _body = _run(_payload(output_config = {"effort": "high"}))

    assert status == 200
    assert [path for path, _kwargs in calls] == ["plain"]
    assert upstream == []


_PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.mark.parametrize(
    "overrides, content",
    [
        ({"supports_tools": False}, "Name a scientist."),
        (
            {"is_vision": True},
            [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": _PNG_1PX},
                },
                {"type": "text", "text": "Name a scientist."},
            ],
        ),
    ],
    ids = ["toolless-backend", "top-level-image"],
)
def test_format_kept_when_requested_server_tools_cannot_run(monkeypatch, overrides, content):
    calls, upstream = _install(monkeypatch, **overrides)

    status, _body = _run(
        _payload(
            enable_tools = True,
            permission_mode = "off",
            messages = [{"role": "user", "content": content}],
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    assert status == 200
    assert calls == []
    [sent] = upstream
    assert sent["response_format"] == _EXPECTED


@pytest.mark.parametrize("enabled_tools", [[], ["unavailable_tool"]])
def test_format_kept_when_server_tool_selection_is_empty(monkeypatch, enabled_tools):
    calls, upstream = _install(monkeypatch)

    status, _body = _run(
        _payload(
            enable_tools = True,
            enabled_tools = enabled_tools,
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    assert status == 200
    assert calls == []
    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert "tools" not in sent


def test_format_kept_when_server_tools_are_disabled_by_tool_choice(monkeypatch):
    calls, upstream = _install(monkeypatch)

    status, _body = _run(
        _payload(
            enable_tools = True,
            permission_mode = "off",
            tool_choice = {"type": "none"},
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    assert status == 200
    assert calls == []
    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert "tools" not in sent


def test_format_with_tool_choice_none_does_not_require_server_tool_permission(monkeypatch):
    calls, upstream = _install(monkeypatch)

    status, _body = _run(
        _payload(
            enable_tools = True,
            tool_choice = {"type": "none"},
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    assert status == 200
    assert calls == []
    [sent] = upstream
    assert sent["response_format"] == _EXPECTED
    assert "tools" not in sent


def test_schema_only_server_tool_request_does_not_add_date_for_api_key(monkeypatch):
    _calls, upstream = _install(monkeypatch)
    monkeypatch.setattr(
        inf_mod, "current_date_prompt_line", lambda **_kwargs: "Current date: 2026-09-25"
    )
    monkeypatch.setattr(inf_mod, "_request_has_api_key", lambda _request: True)
    monkeypatch.setattr(inf_mod, "_request_is_internal_workflow", lambda _request: False)

    status, _body = _run(
        _payload(
            enable_tools = True,
            tool_choice = {"type": "none"},
            output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
    )

    assert status == 200
    [sent] = upstream
    assert sent["messages"] == [{"role": "user", "content": "Name a scientist."}]


@pytest.mark.parametrize("with_format", [False, True])
def test_count_tokens_matches_schema_routing_under_tool_choice_none(monkeypatch, with_format):
    counted = []

    def _count(messages, _template, tools, **_kwargs):
        counted.append((messages, tools))
        return 2

    _install(monkeypatch, count_chat_tokens = _count)
    fields = {"enable_tools": True, "permission_mode": "off", "tool_choice": {"type": "none"}}
    if with_format:
        fields["output_config"] = {"format": {"type": "json_schema", "schema": _SCHEMA}}

    response = asyncio.run(
        inf_mod.anthropic_count_tokens(_payload(**fields), request = _Request(), current_subject = "t")
    )

    assert response.status_code == 200
    [(_messages, tools)] = counted
    assert bool(tools) is not with_format


def test_count_tokens_omits_server_tool_date_for_schema_only_api_key(monkeypatch):
    counted = []

    def _count(messages, _template, tools, **_kwargs):
        counted.append((messages, tools))
        return 2

    _install(monkeypatch, count_chat_tokens = _count)
    monkeypatch.setattr(
        inf_mod, "current_date_prompt_line", lambda **_kwargs: "Current date: 2026-09-25"
    )
    monkeypatch.setattr(inf_mod, "_request_has_api_key", lambda _request: True)
    monkeypatch.setattr(inf_mod, "_request_is_internal_workflow", lambda _request: False)

    response = asyncio.run(
        inf_mod.anthropic_count_tokens(
            _payload(
                enable_tools = True,
                tool_choice = {"type": "none"},
                output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
            ),
            request = _Request(),
            current_subject = "t",
        )
    )

    assert response.status_code == 200
    [(messages, tools)] = counted
    assert tools is None
    assert messages == [{"role": "user", "content": "Name a scientist."}]


def test_count_tokens_omits_server_tool_date_when_selection_is_empty(monkeypatch):
    counted = []

    def _count(messages, _template, tools, **_kwargs):
        counted.append((messages, tools))
        return 2

    _install(monkeypatch, count_chat_tokens = _count)
    monkeypatch.setattr(
        inf_mod, "current_date_prompt_line", lambda **_kwargs: "Current date: 2026-09-25"
    )
    monkeypatch.setattr(inf_mod, "_request_has_api_key", lambda _request: True)
    monkeypatch.setattr(inf_mod, "_request_is_internal_workflow", lambda _request: False)

    response = asyncio.run(
        inf_mod.anthropic_count_tokens(
            _payload(
                enable_tools = True,
                enabled_tools = [],
                output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}},
            ),
            request = _Request(),
            current_subject = "t",
        )
    )

    assert response.status_code == 200
    [(messages, tools)] = counted
    assert tools is None
    assert messages == [{"role": "user", "content": "Name a scientist."}]


@pytest.mark.parametrize("replayed_history", [False, True])
def test_count_tokens_matches_withdrawn_client_tools(monkeypatch, replayed_history):
    counted = []

    def _count(messages, _template, tools, **_kwargs):
        counted.append((messages, tools))
        return 2

    _install(monkeypatch, supports_tool_passthrough = True, count_chat_tokens = _count)
    fields = {
        "tools": [_CLIENT_TOOL],
        "tool_choice": {"type": "none"},
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
    }
    if replayed_history:
        fields["messages"] = [
            {"role": "user", "content": "Look up a scientist."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"name": "Ada"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Ada"}],
            },
        ]

    response = asyncio.run(
        inf_mod.anthropic_count_tokens(_payload(**fields), request = _Request(), current_subject = "t")
    )

    assert response.status_code == 200
    [(_messages, tools)] = counted
    assert bool(tools) is replayed_history


@pytest.mark.parametrize("stream", [False, True])
def test_tool_markup_inside_json_is_returned_verbatim(monkeypatch, stream):
    answer = json.dumps({"a": "<function=f>", "b": 2, "c": "</function>"})
    _install(monkeypatch, answer = answer)

    status, body = _run(
        _payload(
            stream = stream, output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}}
        )
    )

    assert status == 200
    if stream:
        assert json.dumps(answer)[1:-1] in body
    else:
        assert json.loads(body)["content"][0]["text"] == answer
