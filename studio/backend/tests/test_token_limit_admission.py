# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import inference
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, record_api_usage


@pytest.fixture
def admission(monkeypatch):
    produced = []

    async def produce(payload, request, current_subject, *, cancel_on_disconnect):
        produced.append(payload.model)
        return JSONResponse(
            {
                "id": "chatcmpl-test",
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
            }
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", produce)
    monkeypatch.setattr(inference, "_request_used_api_key", lambda request: False)
    app = FastAPI()
    app.include_router(inference.router, prefix = "/v1")
    app.add_api_route("/api/inference/generate/stream", inference.generate_stream, methods = ["POST"])
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    body = {"model": "unsloth/model", "messages": [{"role": "user", "content": "hi"}]}
    # 100 tokens already used this period, by an earlier session request.
    assert record_api_usage(
        ApiUsageReceipt(
            id = "earlier",
            subject = "owner",
            endpoint = "/v1/chat/completions",
            model = body["model"],
            status = "completed",
            prompt_tokens = 90,
            completion_tokens = 10,
            total_tokens = 100,
            created_at = int(time.time() * 1000),
            via_api_key = False,
        )
    )
    with TestClient(app) as client:
        yield client, body, produced


def test_token_cap_blocks_new_requests_and_preserves_disabled_behavior(admission):
    client, body, produced = admission
    studio_db.upsert_app_settings({"token_limit": {"enabled": False, "limit": 100}})
    before = client.post("/v1/chat/completions", json = body)
    assert (
        before.status_code == 200 and before.json()["choices"][0]["message"]["content"] == "answer"
    )

    studio_db.upsert_app_settings(
        {"token_limit": {"enabled": True, "limit": 100, "timezone": "UTC"}}
    )
    refused = client.post("/v1/chat/completions", json = body)
    assert refused.status_code == 429
    assert "daily token limit" in refused.json()["detail"]
    assert refused.headers["X-Unsloth-Token-Limit"] == "reached"
    assert produced == [body["model"]]
    for endpoint, extra in (
        ("completions", {"prompt": "hi"}),
        ("responses", {"input": "hi"}),
        ("messages", {"messages": body["messages"], "max_tokens": 1}),
    ):
        assert (
            client.post(f"/v1/{endpoint}", json = {"model": body["model"], **extra}).status_code
            == 429
        )

    studio_db.upsert_app_settings({"token_limit": {"enabled": False, "limit": 100}})
    after = client.post("/v1/chat/completions", json = body)
    assert after.status_code == before.status_code and after.content == before.content


@pytest.mark.parametrize("api_key", [False, True])
def test_legacy_streams_charge_usage_and_stop_at_the_allowance(admission, monkeypatch, api_key):
    from types import SimpleNamespace
    from core.inference import token_limit
    from core.inference.api_monitor import ApiMonitor
    from storage.api_usage_db import record_api_usage

    client, _, _ = admission
    monitor = ApiMonitor()
    monitor.set_terminal_callback(record_api_usage)
    monkeypatch.setattr(inference, "api_monitor", monitor)
    monkeypatch.setattr(inference, "_request_used_api_key", lambda request: api_key)
    produced = []

    def generate(*, stats_holder = None, **kwargs):
        produced.append(kwargs["messages"])
        if stats_holder is not None:
            stats_holder["stats"] = {
                "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
            }
        yield "answer"

    backend = SimpleNamespace(
        active_model_name = "unsloth/model",
        generate_chat_response = generate,
        reset_generation_state = lambda *args: None,
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    # The fixture's saved reply already used 100.
    token_limit.update_settings({"enabled": True, "limit": 110})
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = client.post("/api/inference/generate/stream", json = body)
    assert first.status_code == 200 and '"content": "answer"' in first.text
    assert token_limit.status()["used"] == 110
    refused = client.post("/api/inference/generate/stream", json = body)
    assert refused.status_code == 429
    assert len(produced) == 1
    token_limit.update_settings({"enabled": False})
    after = client.post("/api/inference/generate/stream", json = body)
    assert after.content == first.content


@pytest.mark.parametrize(
    "endpoint,extra",
    [
        ("responses", {"input": "hi"}),
        ("messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}),
    ],
)
def test_delegated_request_consumes_one_override(admission, monkeypatch, endpoint, extra):
    from core.inference import token_limit

    client, body, produced = admission
    monkeypatch.setattr(inference, "_in_process_model_resident", lambda: True)
    token_limit.update_settings({"enabled": True, "limit": 100, "timezone": "UTC"})
    token_limit.grant_override()
    response = client.post(f"/v1/{endpoint}", json = {"model": body["model"], **extra})
    assert response.status_code == 200, response.text
    assert produced == [body["model"]]
    assert client.post(f"/v1/{endpoint}", json = {"model": body["model"], **extra}).status_code == 429


def test_a_stopped_legacy_stream_settles_an_estimate(admission, monkeypatch):
    from types import SimpleNamespace
    from core.inference import token_limit
    from core.inference.api_monitor import ApiMonitor
    from storage.api_usage_db import record_api_usage

    client, _, _ = admission
    monitor = ApiMonitor()
    monitor.set_terminal_callback(record_api_usage)
    monkeypatch.setattr(inference, "api_monitor", monitor)
    monkeypatch.setattr(inference, "_request_used_api_key", lambda request: False)

    def generate(**kwargs):
        # Cumulative text, and no gen_done stats: the run was cut short.
        yield "a" * 40
        yield "a" * 80

    backend = SimpleNamespace(
        active_model_name = "unsloth/model",
        generate_chat_response = generate,
        reset_generation_state = lambda *args: None,
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    token_limit.update_settings({"enabled": True, "limit": 1000})
    before = token_limit.status()["used"]
    response = client.post(
        "/api/inference/generate/stream", json = {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    # "user: hi" is 2 estimated prompt tokens; the 80-character reply is 20.
    assert token_limit.status()["used"] - before == 22


# A saved chat (with a thread) and a stateless call are both charged by their monitor receipt.
@pytest.mark.parametrize("thread_id", [None, "thread"])
def test_a_session_request_is_charged_by_its_receipt(admission, monkeypatch, thread_id):
    from core.inference import token_limit
    from core.inference.api_monitor import ApiMonitor

    client, body, _ = admission
    monitor = ApiMonitor()
    monitor.set_terminal_callback(record_api_usage)
    monkeypatch.setattr(inference, "api_monitor", monitor)

    async def produce(payload, request, current_subject, *, cancel_on_disconnect):
        entry = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = body["model"],
            prompt = "hi",
            subject = "owner",
        )
        inference._monitor_usage(
            entry, {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        )
        monitor.finish(entry)
        return JSONResponse({"choices": []})

    monkeypatch.setattr(inference, "produce_openai_chat_completions", produce)
    token_limit.update_settings({"enabled": True, "limit": 10_000})
    before = token_limit.status()["used"]
    request = {**body, **({"thread_id": thread_id} if thread_id else {})}
    assert client.post("/v1/chat/completions", json = request).status_code == 200
    assert token_limit.status()["used"] - before == 25
