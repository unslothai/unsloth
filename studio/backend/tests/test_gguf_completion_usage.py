# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for GGUF non-streaming chat completion usage."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend


class _GgufBackend(FakeLlamaCppBackend):
    def __init__(
        self,
        usage,
        context_truncation = None,
    ):
        self.usage = usage
        self.context_truncation = context_truncation
        self.generation_index = 0

    def generate_chat_completion(self, **kwargs):
        truncations = self.context_truncation
        if isinstance(truncations, list):
            truncations = truncations[self.generation_index]
        self.generation_index += 1
        if isinstance(truncations, dict):
            truncations = [truncations]
        for truncation in truncations or []:
            yield {"type": "context_truncated", **truncation}
        yield "answer"
        yield {
            "type": "metadata",
            "usage": self.usage,
            "timings": {"prompt_n": 23, "predicted_n": 1283},
        }


def _request_completion(
    monkeypatch,
    usage,
    context_truncation = None,
    n = None,
):
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
            **({"n": n} if n is not None else {}),
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


def test_non_streaming_choices_keep_distinct_later_truncation_stages(monkeypatch):
    base = {
        "dropped_messages": 4,
        "prompt_tokens_before": 9000,
        "prompt_tokens_after": 7000,
        "context_length": 8192,
        "fits": True,
    }
    additional = {
        "dropped_messages": 2,
        "prompt_tokens_before": 7000,
        "prompt_tokens_after": 3500,
        "context_length": 4096,
        "fits": True,
    }
    cumulative = {
        "dropped_messages": 6,
        "prompt_tokens_before": 9000,
        "prompt_tokens_after": 3500,
        "context_length": 4096,
        "fits": True,
    }

    response = _request_completion(
        monkeypatch,
        {"prompt_tokens": 3500, "completion_tokens": 20, "total_tokens": 3520},
        [[base], [base, additional], [cumulative]],
        n = 3,
    )

    assert response.status_code == 200
    assert len(response.json()["choices"]) == 3
    assert response.json()["context_truncated"] == {
        "dropped_messages": 6,
        "prompt_tokens_before": 9000,
        "prompt_tokens_after": 3500,
        "context_length": 4096,
        "fits": True,
    }
