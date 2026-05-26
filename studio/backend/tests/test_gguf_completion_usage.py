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

    def generate_chat_completion(self, **kwargs):
        yield "answer"
        yield {
            "type": "metadata",
            "usage": {
                "prompt_tokens": 23,
                "completion_tokens": 1283,
                "total_tokens": 1306,
            },
            "timings": {"prompt_n": 23, "predicted_n": 1283},
        }


def test_non_streaming_gguf_completion_includes_generated_usage(monkeypatch):
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: _GgufBackend()
    )
    monkeypatch.setattr(
        inference_route, "_effective_enable_tools", lambda payload: False
    )

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 23,
        "completion_tokens": 1283,
        "total_tokens": 1306,
    }
