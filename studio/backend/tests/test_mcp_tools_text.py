# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import json

import pytest
from fastapi.responses import JSONResponse, Response
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

DECISION = {
    "model": "laya-multilingual",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.9},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.8, "bug": 0.2},
        },
        "tone": {
            "type": "score",
            "score": 2.0,
            "confidence": 0.7,
            "legend": {"0": "calm", "1": {"text": "angry"}},
            "probabilities": {"1": 0.3, "2": 0.7},
        },
    },
    "usage": {"input_tokens": 40, "output_tokens": 0},
}

PAYLOADS = {
    ("POST", "/v1/chat/completions"): COMPLETION,
    ("POST", "/v1/embeddings"): EMBEDDINGS,
    ("POST", "/v1/systemone"): DECISION,
}
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
    "topic": {"type": "choice", "criteria": {"billing": "Money", "bug": "Defect"}},
    "tone": {"type": "score", "criteria": ["calm", "angry"]},
}


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
        "cancel_id": result["structuredContent"]["cancel_id"],
    }
    (body,) = _sent(studio, "/v1/chat/completions")
    assert result["structuredContent"]["cancel_id"] == body["cancel_id"]
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


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16


def _data_url(data, mime):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_chat_images_ride_the_last_user_turn(monkeypatch):
    studio = _studio(
        {
            ("GET", "/api/inference/images/gallery/img-1/file"): lambda request, body: Response(
                PNG, media_type = "image/png"
            )
        }
    )
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "What is in these?"},
    ]
    images = [{"data_url": _data_url(JPEG, "image/jpeg")}, {"gallery_id": "img-1"}]
    result = _call(monkeypatch, studio, "chat", {"messages": messages, "images": images})
    assert result["isError"] is False
    (body,) = _sent(studio, "/v1/chat/completions")
    assert body["messages"][0] == {"role": "user", "content": "Hi"}
    assert body["messages"][2]["content"] == [
        {"type": "text", "text": "What is in these?"},
        {"type": "image_url", "image_url": {"url": _data_url(JPEG, "image/jpeg")}},
        {"type": "image_url", "image_url": {"url": _data_url(PNG, "image/png")}},
    ]
    gallery = next(c for c in studio.state.calls if c[1].endswith("/file"))
    assert gallery[2]["authorization"] == "Bearer sk-unsloth-test"


def test_chat_images_need_a_user_turn(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        "chat",
        {
            "messages": [{"role": "assistant", "content": "x"}],
            "images": [{"data_url": _data_url(PNG, "image/png")}],
        },
    )
    assert result["isError"] is True
    assert studio.state.calls == []


def test_a_bad_data_url_is_refused(monkeypatch):
    studio = _studio()
    for url in (
        "data:image/gif;base64,R0lG",
        "data:image/png;base64,@@@",
        _data_url(b"not an image", "image/png"),
    ):
        result = _call(monkeypatch, studio, "chat", {"prompt": "x", "images": [{"data_url": url}]})
        assert result["isError"] is True
    assert studio.state.calls == []


def test_chat_with_no_model_loaded_says_to_load_one(monkeypatch):
    def unloaded(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "No model loaded. Call POST /inference/load first.",
                    "type": "invalid_request_error",
                }
            },
            status_code = 400,
        )

    studio = _studio({("POST", "/v1/chat/completions"): unloaded})
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi"})
    assert result["isError"] is True
    assert result["content"][0]["text"].endswith(
        "Load a chat model with load_model first; list_models shows the downloaded ones."
    )


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


@pytest.mark.parametrize("finish_reason,noted", [("length", True), ("stop", False)])
def test_an_empty_reply_cut_off_by_the_budget_says_so(monkeypatch, finish_reason, noted):
    reply = {
        **COMPLETION,
        "choices": [{"message": {"content": ""}, "finish_reason": finish_reason}],
    }
    studio = _studio({("POST", "/v1/chat/completions"): lambda request, body: reply})
    result = _call(monkeypatch, studio, "chat", {"prompt": "hi", "max_tokens": 300})
    note = result["structuredContent"]["note"]
    assert ("raise max_tokens" in note) if noted else note is None


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


def test_system_one_sends_the_default_model_and_maps_every_answer_type(monkeypatch):
    def decide(request, body):
        return JSONResponse(DECISION, headers = {"x-typesafe-request-id": "req-1"})

    studio = _studio({("POST", "/v1/systemone"): decide})
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "Customer: I was charged twice!", "questions": QUESTIONS},
    )
    assert result["structuredContent"] == {
        "model": "laya-multilingual",
        "request_id": "req-1",
        "answers": {
            "urgent": {
                "type": "noul",
                "noul": 0.9,
                "choice": None,
                "score": None,
                "confidence": None,
                "probabilities": None,
                "legend": None,
            },
            "topic": {
                "type": "choice",
                "noul": None,
                "choice": "billing",
                "score": None,
                "confidence": 0.8,
                "probabilities": {"billing": 0.8, "bug": 0.2},
                "legend": None,
            },
            "tone": {
                "type": "score",
                "noul": None,
                "choice": None,
                "score": 2.0,
                "confidence": 0.7,
                "probabilities": {"1": 0.3, "2": 0.7},
                "legend": {"0": "calm", "1": '{"text": "angry"}'},
            },
        },
    }
    (body,) = _sent(studio, "/v1/systemone")
    assert body == {
        "state": "Customer: I was charged twice!",
        "model": "default",
        "questions": QUESTIONS,
    }


def test_system_one_takes_json_state(monkeypatch):
    studio = _studio()
    _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": {"ticket": 7}, "questions": {"urgent": QUESTIONS["urgent"]}},
    )
    assert _sent(studio, "/v1/systemone")[0]["state"] == {"ticket": 7}


def test_the_decision_api_being_off_gives_guidance(monkeypatch):
    def off(request, body):
        return JSONResponse(
            {
                "detail": {
                    "error_type": "api_usage_error",
                    "message": "The Decision API is off. The Studio owner can turn it on in Settings > API.",
                }
            },
            status_code = 404,
        )

    studio = _studio({("POST", "/v1/systemone"): off})
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "x", "questions": {"urgent": QUESTIONS["urgent"]}},
    )
    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert message.startswith("The Decision API is off.")
    assert message.endswith("Turn on the Decision API in Settings > API.")


def test_a_loading_decision_model_says_when_to_retry(monkeypatch):
    def loading(request, body):
        return JSONResponse(
            {
                "detail": {
                    "error_type": "model_loading",
                    "message": "The decision model is loading.",
                }
            },
            status_code = 503,
            headers = {"Retry-After": "12"},
        )

    studio = _studio({("POST", "/v1/systemone"): loading})
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "x", "questions": {"urgent": QUESTIONS["urgent"]}},
    )
    assert (
        result["content"][0]["text"]
        == "The decision model is loading. (HTTP 503) Retry after 12 s."
    )


def test_a_detail_error_type_message_is_used(monkeypatch):
    def unknown(request, body):
        return JSONResponse(
            {"detail": {"error_type": "api_usage_error", "message": "Unknown model: nope"}},
            status_code = 400,
        )

    studio = _studio({("POST", "/v1/systemone"): unknown})
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "x", "questions": {"urgent": QUESTIONS["urgent"]}, "model": "nope"},
    )
    assert result["content"][0]["text"] == "Unknown model: nope (HTTP 400)"
    assert _sent(studio, "/v1/systemone")[0]["model"] == "nope"


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {f"q{i}": {"type": "noul"} for i in range(65)},
        {"q": {"type": "maybe"}},
        {"q": {"type": "noul", "extra": 1}},
    ],
)
def test_system_one_validates_questions_before_calling(monkeypatch, questions):
    studio = _studio()
    result = _call(monkeypatch, studio, "system_one", {"state": "x", "questions": questions})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_system_one_sends_up_to_four_png_or_jpeg_images(monkeypatch):
    studio = _studio()
    images = [
        {"data_url": _data_url(PNG, "image/png")},
        {"data_url": _data_url(JPEG, "image/jpeg")},
    ]
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "x", "questions": {"urgent": QUESTIONS["urgent"]}, "images": images},
    )
    assert result["isError"] is False
    assert _sent(studio, "/v1/systemone")[0]["images"] == [
        _data_url(PNG, "image/png"),
        _data_url(JPEG, "image/jpeg"),
    ]


@pytest.mark.parametrize(
    "images",
    [
        [{"data_url": _data_url(WEBP, "image/webp")}],
        [{"data_url": _data_url(PNG, "image/png")}] * 5,
    ],
)
def test_system_one_refuses_images_the_decision_api_would(monkeypatch, images):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        "system_one",
        {"state": "x", "questions": {"urgent": QUESTIONS["urgent"]}, "images": images},
    )
    assert result["isError"] is True
    assert studio.state.calls == []


def test_system_one_refuses_an_image_over_4_mib(monkeypatch, tmp_path):
    # Over the Decision API's 4 MiB per image; only a file path can carry it past the request limit.
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"\x00" * (4 * 1024 * 1024))
    studio = _studio()
    with TestClient(
        served(create_studio_mcp(), studio, monkeypatch = monkeypatch),
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
    ) as http:
        result = call_tool(
            http,
            "system_one",
            {
                "state": "x",
                "questions": {"urgent": QUESTIONS["urgent"]},
                "images": [{"path": str(big)}],
            },
        )
    assert result["isError"] is True
    assert studio.state.calls == []


def test_system_one_is_read_only():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["system_one"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
    assert tool.parameters["properties"]["model"]["default"] == "default"
    assert tool.parameters["properties"]["images"]["anyOf"][0]["maxItems"] == 4
