# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /api/inference/decision: the loaded decision GGUF answers through llama-server's /v1/systemone."""

from __future__ import annotations

import json

import pytest

import routes.inference as routes_inference


class _FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _FakeBackend:
    def __init__(self, *, loaded: bool = True, decision: bool = True):
        self.is_loaded = loaded
        self.is_decision = decision
        self.base_url = "http://127.0.0.1:9999"
        self._auth_headers = {"Authorization": "Bearer secret"}
        self.model_identifier = "AtomicChat/d1-2B-GGUF"


def _payload(**overrides):
    body = {
        "state": "2+2=4",
        "questions": {
            "is_true": {
                "type": "noul",
                "instructions": "Is this statement mathematically correct?",
            }
        },
    }
    body.update(overrides)
    return routes_inference.DecisionRequestIn(**body)


@pytest.fixture
def calls(monkeypatch):
    seen: list[dict] = []

    def _post(url, **kwargs):
        seen.append({"url": url, **kwargs})
        return _FakeResponse(
            200,
            {
                "answers": {"is_true": {"type": "noul", "noul": 0.9946}},
                "usage": {"input_tokens": 32, "output_tokens": 0},
            },
        )

    monkeypatch.setattr(routes_inference.httpx, "post", _post)
    return seen


def test_the_loaded_decision_model_answers(monkeypatch, calls):
    monkeypatch.setattr(
        routes_inference, "get_llama_cpp_backend", lambda: _FakeBackend()
    )
    response = routes_inference.asyncio.run(
        routes_inference.run_loaded_decision(_payload(), current_subject = "owner")
    )
    body = json.loads(response.body)
    assert body["answers"]["is_true"]["noul"] == pytest.approx(0.9946)
    assert body["model"] == "AtomicChat/d1-2B-GGUF"
    assert isinstance(body["latency_ms"], float)
    assert calls[0]["url"] == "http://127.0.0.1:9999/v1/systemone"
    assert calls[0]["json"]["state"] == "2+2=4"
    assert calls[0]["json"]["questions"]["is_true"]["type"] == "noul"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_a_chat_model_is_not_served_here(monkeypatch, calls):
    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: _FakeBackend(decision = False),
    )
    with pytest.raises(routes_inference.HTTPException) as caught:
        routes_inference.asyncio.run(
            routes_inference.run_loaded_decision(
                _payload(), current_subject = "owner"
            )
        )
    assert caught.value.status_code == 400
    assert not calls


def test_llama_server_error_text_reaches_the_user(monkeypatch):
    def _post(url, **kwargs):
        return _FakeResponse(
            500,
            {
                "error": {
                    "code": 500,
                    "message": "failed to parse grammar",
                    "type": "server_error",
                }
            },
        )

    monkeypatch.setattr(routes_inference.httpx, "post", _post)
    monkeypatch.setattr(
        routes_inference, "get_llama_cpp_backend", lambda: _FakeBackend()
    )
    with pytest.raises(routes_inference.HTTPException) as caught:
        routes_inference.asyncio.run(
            routes_inference.run_loaded_decision(
                _payload(), current_subject = "owner"
            )
        )
    assert caught.value.status_code == 500
    assert caught.value.detail == "failed to parse grammar"


def test_images_are_forwarded_only_when_sent(monkeypatch, calls):
    monkeypatch.setattr(
        routes_inference, "get_llama_cpp_backend", lambda: _FakeBackend()
    )
    routes_inference.asyncio.run(
        routes_inference.run_loaded_decision(_payload(), current_subject = "owner")
    )
    assert "images" not in calls[0]["json"]

    routes_inference.asyncio.run(
        routes_inference.run_loaded_decision(
            _payload(images = ["data:image/png;base64,AAAA"]),
            current_subject = "owner",
        )
    )
    assert calls[1]["json"]["images"] == ["data:image/png;base64,AAAA"]
