# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for GGUF non-streaming chat completion usage."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route


class _GgufBackend:
    is_loaded = True
    model_identifier = "test/model.gguf"
    _is_audio = False
    is_vision = False
    supports_tools = False
    context_length = 8192

    def __init__(self, usage, context_truncation = None):
        self.usage = usage
        self.context_truncation = context_truncation

    def generate_chat_completion(self, **kwargs):
        if self.context_truncation:
            yield {"type": "context_truncated", **self.context_truncation}
        yield "answer"
        yield {
            "type": "metadata",
            "usage": self.usage,
            "timings": {"prompt_n": 23, "predicted_n": 1283},
        }


def _request_completion(monkeypatch, usage, context_truncation = None):
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: _GgufBackend(usage, context_truncation),
    )
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    return TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
            "stream": False,
        },
    )


def test_non_streaming_gguf_completion_includes_generated_usage(monkeypatch):
    response = _request_completion(
        monkeypatch,
        {"prompt_tokens": 23, "completion_tokens": 1283, "total_tokens": 1306},
    )

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 23
    assert usage["completion_tokens"] == 1283
    assert usage["total_tokens"] == 1306
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0, "audio_tokens": 0}
    assert usage["completion_tokens_details"] == {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }


def test_non_streaming_gguf_completion_defaults_nullable_usage_to_zero(monkeypatch):
    response = _request_completion(
        monkeypatch,
        {"prompt_tokens": None, "completion_tokens": 1283, "total_tokens": None},
    )

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 1283
    assert usage["total_tokens"] == 1283
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0, "audio_tokens": 0}
    assert usage["completion_tokens_details"] == {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }


def test_non_streaming_gguf_completion_includes_context_truncation(monkeypatch):
    truncation = {
        "dropped_messages": 4,
        "prompt_tokens_before": 9000,
        "prompt_tokens_after": 7000,
        "context_length": 8192,
        "fits": True,
    }
    response = _request_completion(
        monkeypatch,
        {"prompt_tokens": 7000, "completion_tokens": 20, "total_tokens": 7020},
        truncation,
    )

    assert response.status_code == 200
    assert response.json()["context_truncated"] == truncation
