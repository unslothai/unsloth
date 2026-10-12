# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi.responses import JSONResponse, Response

from .mcp_harness import JPEG, PNG, PNG_URL, bodies, call_to, data_url, run_tool

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
CHAT = ("POST", "/v1/chat/completions")
URGENT = {"urgent": QUESTIONS["urgent"]}


def _chat(
    monkeypatch,
    args,
    reply = None,
):
    routes = PAYLOADS if reply is None else {**PAYLOADS, CHAT: reply}
    return run_tool(monkeypatch, routes, "chat", args)


def test_a_prompt_becomes_one_user_turn(monkeypatch):
    result, studio = _chat(
        monkeypatch, {"prompt": "Capital of France?", "system": "Be brief.", "max_tokens": 16}
    )
    out = result["structuredContent"]
    assert out == {
        "text": "Paris.",
        "model": LLM,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        "note": None,
        "cancel_id": out["cancel_id"],
    }
    run_tool(monkeypatch, studio, "chat", {"prompt": "b"})
    body, second = bodies(studio, "/v1/chat/completions")
    assert out["cancel_id"] == body["cancel_id"] != second["cancel_id"]
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert body["stream"] is False
    assert body["model"] == "default"
    assert body["max_tokens"] == 16
    assert "temperature" not in body
    assert body["cancel_id"].startswith("mcp-") and len(body["cancel_id"]) > 20


def test_a_caller_chosen_cancel_id_is_sent_so_another_call_can_stop_the_reply(monkeypatch):
    result, studio = _chat(monkeypatch, {"prompt": "a", "cancel_id": "agent-run-0001"})
    assert bodies(studio, "/v1/chat/completions")[0]["cancel_id"] == "agent-run-0001"
    assert result["structuredContent"]["cancel_id"] == "agent-run-0001"
    for bad in ("short", "has space in it", "x" * 65):
        refused, studio = _chat(monkeypatch, {"prompt": "a", "cancel_id": bad})
        assert refused["isError"] is True
        assert bodies(studio, "/v1/chat/completions") == []


def test_messages_are_sent_as_given(monkeypatch):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Capital of France?"},
    ]
    _result, studio = _chat(monkeypatch, {"messages": messages, "temperature": 0.2, "model": LLM})
    (body,) = bodies(studio, "/v1/chat/completions")
    assert body["messages"] == messages
    assert body["temperature"] == 0.2
    assert body["model"] == LLM


def test_chat_images_ride_the_last_user_turn(monkeypatch):
    gallery = {
        ("GET", "/api/inference/images/gallery/img-1/file"): Response(PNG, media_type = "image/png")
    }
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "What is in these?"},
    ]
    images = [{"data_url": data_url(JPEG, "image/jpeg")}, {"gallery_id": "img-1"}]
    result, studio = run_tool(
        monkeypatch, {**PAYLOADS, **gallery}, "chat", {"messages": messages, "images": images}
    )
    assert result["isError"] is False
    (body,) = bodies(studio, "/v1/chat/completions")
    assert body["messages"][0] == {"role": "user", "content": "Hi"}
    assert body["messages"][2]["content"] == [
        {"type": "text", "text": "What is in these?"},
        {"type": "image_url", "image_url": {"url": data_url(JPEG, "image/jpeg")}},
        {"type": "image_url", "image_url": {"url": data_url(PNG, "image/png")}},
    ]
    assert call_to(studio, "/api/inference/images/gallery/img-1/file")[2]["authorization"] == (
        "Bearer sk-unsloth-test"
    )


def test_a_different_answering_model_is_noted(monkeypatch):
    result, _studio = _chat(monkeypatch, {"prompt": "hi", "model": "unsloth/Qwen3-0.6B"})
    note = result["structuredContent"]["note"]
    assert LLM in note and "unsloth/Qwen3-0.6B" in note

    pinned = {**COMPLETION, "model": f"{LLM}:Q4_K_M"}
    result, _studio = _chat(monkeypatch, {"prompt": "hi", "model": LLM}, pinned)
    assert result["structuredContent"]["note"] is None


@pytest.mark.parametrize(
    "content,finish_reason,noted",
    [("", "length", True), ("", "stop", False), ("Edit /etc/hosts as root.", "stop", False)],
)
def test_an_empty_reply_cut_off_by_the_budget_says_so(monkeypatch, content, finish_reason, noted):
    reply = {
        **COMPLETION,
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
    }
    result, _studio = _chat(monkeypatch, {"prompt": "hi", "max_tokens": 300}, reply)
    out = result["structuredContent"]
    assert out["text"] == content
    assert ("raise max_tokens" in out["note"]) if noted else out["note"] is None


@pytest.mark.parametrize("model", [None, "bge-m3"])
def test_embed_returns_vectors_in_input_order(monkeypatch, model):
    args = {"texts": ["first", "second"], **({"model": model} if model else {})}
    result, studio = run_tool(monkeypatch, PAYLOADS, "embed", args)
    assert result["structuredContent"] == {
        "model": "nomic-embed-text-v1.5",
        "dimensions": 3,
        "embeddings": [[1.0, 0.0, 0.25], [0.0, 1.0, 0.5]],
    }
    assert bodies(studio, "/v1/embeddings") == [
        {"input": args["texts"], **({"model": model} if model else {})}
    ]


# The 2048-text limit is accepted; fewer vectors than texts is an error.
@pytest.mark.parametrize("rows,texts", [(2048, 2048), (1, 2)])
def test_embed_accepts_the_limit_and_refuses_a_short_answer(monkeypatch, rows, texts):
    answer = {"model": "m", "data": [{"index": i, "embedding": [0.5]} for i in range(rows)]}
    result, _studio = run_tool(
        monkeypatch, {("POST", "/v1/embeddings"): answer}, "embed", {"texts": ["x"] * texts}
    )
    if rows < texts:
        assert result["isError"] is True
    else:
        assert result["structuredContent"]["dimensions"] == 1
        assert len(result["structuredContent"]["embeddings"]) == texts


def _decide(
    monkeypatch,
    args,
    answer = None,
):
    routes = PAYLOADS if answer is None else {**PAYLOADS, ("POST", "/v1/systemone"): answer}
    return run_tool(monkeypatch, routes, "system_one", args)


def test_system_one_sends_the_default_model_and_maps_every_answer_type(monkeypatch):
    answer = JSONResponse(DECISION, headers = {"x-typesafe-request-id": "req-1"})
    args = {"state": "Customer: I was charged twice!", "questions": QUESTIONS}
    result, studio = _decide(monkeypatch, args, answer)
    blank = dict.fromkeys(["noul", "choice", "score", "confidence", "probabilities", "legend"])
    assert result["structuredContent"] == {
        "model": "laya-multilingual",
        "request_id": "req-1",
        "answers": {
            "urgent": {**blank, "type": "noul", "noul": 0.9},
            "topic": {**blank, **DECISION["answers"]["topic"]},
            "tone": {
                **blank,
                **DECISION["answers"]["tone"],
                "legend": {"0": "calm", "1": '{"text": "angry"}'},
            },
        },
    }
    (body,) = bodies(studio, "/v1/systemone")
    assert body == {
        "state": "Customer: I was charged twice!",
        "model": "default",
        "questions": QUESTIONS,
    }


def test_a_detail_error_type_message_is_used(monkeypatch):
    unknown = JSONResponse(
        {"detail": {"error_type": "api_usage_error", "message": "Unknown model: nope"}},
        status_code = 400,
    )
    args = {"state": "x", "questions": URGENT, "model": "nope"}
    result, studio = _decide(monkeypatch, args, unknown)
    assert result["content"][0]["text"] == "Unknown model: nope (HTTP 400)"
    assert bodies(studio, "/v1/systemone")[0]["model"] == "nope"


def test_system_one_sends_json_state_and_png_or_jpeg_images(monkeypatch):
    images = [PNG_URL, {"data_url": data_url(JPEG, "image/jpeg")}]
    args = {"state": {"ticket": 7}, "questions": URGENT, "images": images}
    result, studio = _decide(monkeypatch, args)
    assert result["isError"] is False
    (body,) = bodies(studio, "/v1/systemone")
    assert body["state"] == {"ticket": 7}
    assert body["images"] == [data_url(PNG, "image/png"), data_url(JPEG, "image/jpeg")]
