# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

LLM = "unsloth/Llama-3.2-1B-Instruct-GGUF"
COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": LLM,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
}

EMBEDDINGS = {
    "object": "list",
    "model": "nomic-embed-text-v1.5",
    "data": [
        {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.5]},
        {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.25]},
    ],
}

PAYLOADS = {("POST", "/v1/chat/completions"): COMPLETION, ("POST", "/v1/embeddings"): EMBEDDINGS}


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, name, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args)


def _sent(studio, path):
    return [json.loads(body) for _m, p, _h, body in studio.state.calls if p == path]


def test_a_prompt_becomes_one_user_turn(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        "chat",
        {"prompt": "Capital of France?", "system": "Be brief.", "max_tokens": 16},
    )
    assert result["structuredContent"] == {
        "text": "Paris.",
        "model": LLM,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        "note": None,
    }
    (body,) = _sent(studio, "/v1/chat/completions")
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert body["stream"] is False
    assert body["model"] == "default"
    assert body["max_tokens"] == 16
    assert "temperature" not in body
    assert body["cancel_id"].startswith("mcp-") and len(body["cancel_id"]) > 20


def test_each_call_gets_its_own_cancel_id(monkeypatch):
    studio = _studio()
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        call_tool(http, "chat", {"prompt": "a"})
        call_tool(http, "chat", {"prompt": "b"})
    ids = [body["cancel_id"] for body in _sent(studio, "/v1/chat/completions")]
    assert len(set(ids)) == 2


def test_messages_are_sent_as_given(monkeypatch):
    studio = _studio()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Capital of France?"},
    ]
    _call(monkeypatch, studio, "chat", {"messages": messages, "temperature": 0.2, "model": LLM})
    (body,) = _sent(studio, "/v1/chat/completions")
    assert body["messages"] == messages
    assert body["temperature"] == 0.2
    assert body["model"] == LLM


@pytest.mark.parametrize(
    "args",
    [{}, {"prompt": "a", "messages": [{"role": "user", "content": "b"}]}, {"messages": []}],
)
def test_exactly_one_of_prompt_and_messages(monkeypatch, args):
    studio = _studio()
    result = _call(monkeypatch, studio, "chat", args)
    assert result["isError"] is True
    assert studio.state.calls == []


def test_an_unknown_role_is_refused(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "chat", {"messages": [{"role": "tool", "content": "x"}]})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_images_are_refused_for_now(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch, studio, "chat", {"prompt": "What is this?", "images": [{"gallery_id": "abc"}]}
    )
    assert result["isError"] is True
    assert "images" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_gpu_busy_is_a_tool_error_with_a_retry_hint(monkeypatch):
    def busy(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "Another account is generating on the resident model.",
                    "type": "conflict_error",
                    "param": "model",
                    "code": "gpu_busy",
                }
            },
            status_code = 409,
            headers = {"Retry-After": "5"},
        )

    studio = _studio({("POST", "/v1/chat/completions"): busy})
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "GPU busy: Another account is generating on the resident model. Retry after 5 s."
    )


def test_a_different_answering_model_is_noted(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi", "model": "unsloth/Qwen3-0.6B"})
    note = result["structuredContent"]["note"]
    assert LLM in note and "unsloth/Qwen3-0.6B" in note

    pinned = {**COMPLETION, "model": f"{LLM}:Q4_K_M"}
    studio = _studio({("POST", "/v1/chat/completions"): lambda request, body: pinned})
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi", "model": LLM})
    assert result["structuredContent"]["note"] is None


def test_model_text_is_returned_untouched(monkeypatch):
    reply = {
        **COMPLETION,
        "choices": [{"message": {"content": "Edit /etc/hosts as root."}, "finish_reason": "stop"}],
    }
    studio = _studio({("POST", "/v1/chat/completions"): lambda request, body: reply})
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi"})
    assert result["structuredContent"]["text"] == "Edit /etc/hosts as root."


def test_chat_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["chat"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None


def test_embed_returns_vectors_in_input_order(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "embed", {"texts": ["first", "second"]})
    assert result["structuredContent"] == {
        "model": "nomic-embed-text-v1.5",
        "dimensions": 3,
        "embeddings": [[1.0, 0.0, 0.25], [0.0, 1.0, 0.5]],
    }
    assert _sent(studio, "/v1/embeddings") == [{"input": ["first", "second"]}]


def test_embed_names_a_model_only_when_asked(monkeypatch):
    studio = _studio()
    _call(monkeypatch, studio, "embed", {"texts": ["first", "second"], "model": "bge-m3"})
    assert _sent(studio, "/v1/embeddings") == [{"input": ["first", "second"], "model": "bge-m3"}]


@pytest.mark.parametrize("count", [0, 2049])
def test_embed_refuses_too_few_or_too_many_texts(monkeypatch, count):
    studio = _studio()
    result = _call(monkeypatch, studio, "embed", {"texts": ["x"] * count})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_embed_accepts_the_limit(monkeypatch):
    rows = {"model": "m", "data": [{"index": i, "embedding": [0.5]} for i in range(2048)]}
    studio = _studio({("POST", "/v1/embeddings"): lambda request, body: rows})
    result = _call(monkeypatch, studio, "embed", {"texts": ["x"] * 2048})
    assert result["structuredContent"]["dimensions"] == 1
    assert len(result["structuredContent"]["embeddings"]) == 2048


def test_a_download_required_409_gives_guidance(monkeypatch):
    def needs_download(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "The embedding model /srv/models/nomic is not downloaded yet.",
                    "type": "conflict_error",
                    "param": None,
                    "code": None,
                }
            },
            status_code = 409,
        )

    studio = _studio({("POST", "/v1/embeddings"): needs_download})
    result = _call(monkeypatch, studio, "embed", {"texts": ["x"]})
    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert message.startswith("The embedding model <path>")
    assert "/srv/models" not in message
    assert message.endswith("load_model(kind='llm') and call embed again.")


def test_a_short_answer_is_an_error(monkeypatch):
    rows = {"model": "m", "data": [{"index": 0, "embedding": [0.5]}]}
    studio = _studio({("POST", "/v1/embeddings"): lambda request, body: rows})
    result = _call(monkeypatch, studio, "embed", {"texts": ["a", "b"]})
    assert result["isError"] is True


def test_embed_is_read_only():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["embed"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
    texts = tool.parameters["properties"]["texts"]
    assert (texts["minItems"], texts["maxItems"]) == (1, 2048)
