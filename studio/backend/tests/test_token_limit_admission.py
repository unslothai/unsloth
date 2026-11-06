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
    now = int(time.time() * 1000)
    studio_db.upsert_chat_thread(
        {
            "id": "thread",
            "title": "thread",
            "modelType": "base",
            "modelId": body["model"],
            "createdAt": now,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "reply",
            "threadId": "thread",
            "role": "assistant",
            "createdAt": now,
            "content": [{"type": "text", "text": "answer"}],
            "metadata": {
                "contextUsage": {"promptTokens": 90, "completionTokens": 10, "totalTokens": 100}
            },
        }
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
    assert refused.headers["X-Unsloth-Token-Limit"] == "combined"
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
    token_limit.update_settings(
        {"enabled": True, "mode": "split", "localLimit": 110, "apiLimit": 10},
        password = None,
        new_password = None,
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = client.post("/api/inference/generate/stream", json = body)
    assert first.status_code == 200 and '"content": "answer"' in first.text
    pool = token_limit.status()["pools"][1 if api_key else 0]
    assert pool["used"] == pool["limit"]
    refused = client.post("/api/inference/generate/stream", json = body)
    assert refused.status_code == 429
    assert len(produced) == 1
    token_limit.update_settings({"enabled": False}, password = None, new_password = None)
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
    token_limit.update_settings(
        {"enabled": True, "limit": 100, "timezone": "UTC"}, password = None, new_password = None
    )
    token_limit.grant_override("combined")
    response = client.post(f"/v1/{endpoint}", json = {"model": body["model"], **extra})
    assert response.status_code == 200, response.text
    assert produced == [body["model"]]
    assert client.post(f"/v1/{endpoint}", json = {"model": body["model"], **extra}).status_code == 429


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize(
    "usage_steps,new_tokens,internal_workflow,parents",
    [
        ([(90, 10, None)], 100, False, [""]),
        ([(250, 20, 160), (180, 5, 185)], 295, False, ["", "private-msg-0"]),
        ([(100, 50, None), (100, 50, None)], 300, False, ["", ""]),
        ([(90, 10, None), (90, 10, None)], 200, True, ["", ""]),
    ],
)
def test_temporary_chat_counts_without_persisting_its_content(
    admission, monkeypatch, stream, external, usage_steps, new_tokens, internal_workflow, parents
):
    from fastapi.responses import StreamingResponse
    from core.inference.api_monitor import ApiMonitor
    from storage.api_usage_db import record_api_usage
    from core.inference import token_limit

    client, body, _ = admission
    monitor = ApiMonitor()
    monitor.set_terminal_callback(record_api_usage)
    monkeypatch.setattr(inference, "api_monitor", monitor)
    monkeypatch.setattr(
        inference, "_request_is_internal_workflow", lambda request: internal_workflow
    )
    completed = 0

    def finish():
        nonlocal completed
        entry = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = body["model"],
            prompt = "private prompt",
            subject = "owner",
        )
        prompt, completion, context = usage_steps[completed % len(usage_steps)]
        completed += 1
        inference._monitor_usage(
            entry,
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "context_tokens": context,
            },
        )
        monitor.set_reply(entry, "private answer")
        monitor.finish(entry)

    async def produce(payload, request, current_subject, *, cancel_on_disconnect):
        if stream:

            async def chunks():
                finish()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type = "text/event-stream")
        finish()
        return JSONResponse({"choices": []})

    monkeypatch.setattr(inference, "produce_openai_chat_completions", produce)
    studio_db.upsert_app_settings(
        {
            "token_limit": {
                "enabled": True,
                "mode": "split",
                "method": "incremental",
                "localLimit": 100 + new_tokens,
                "apiLimit": new_tokens,
            }
        }
    )
    payload = {
        **body,
        "temporary_chat": not internal_workflow,
        "thread_id": "temporary-thread",
        "stream": stream,
    }
    if external:
        payload["provider_type"] = "openai"
    for index, parent in enumerate(parents):
        payload["temporary_message_id"] = f"private-msg-{index}"
        payload["temporary_parent_id"] = parent
        assert client.post("/v1/chat/completions", json = payload).status_code == 200
    assert token_limit.status()["pools"][1 if external else 0]["used"] == new_tokens + (
        0 if external else 100
    )
    assert client.post("/v1/chat/completions", json = payload).status_code == 429

    studio_db.upsert_app_settings({"token_limit": {"enabled": False}})
    monkeypatch.setattr(inference, "_request_is_internal_workflow", lambda request: False)
    payload["temporary_chat"] = False
    assert client.post("/v1/chat/completions", json = payload).status_code == 200
    conn = studio_db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM api_usage_events").fetchone()[0] == 0
        rows = conn.execute("SELECT * FROM token_usage_ledger").fetchall()
        assert len(rows) == 1 + len(usage_steps)
        assert sum(row["total_tokens"] for row in rows) == 100 + sum(
            p + c for p, c, _ in usage_steps
        )
        assert not any(
            "private" in str(tuple(row)) or "temporary-thread" in str(tuple(row)) for row in rows
        )
    finally:
        conn.close()
