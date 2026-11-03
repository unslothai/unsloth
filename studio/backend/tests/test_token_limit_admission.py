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
def admission(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
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
def test_temporary_chat_counts_without_persisting_its_content(
    admission, monkeypatch, stream, external
):
    from fastapi.responses import StreamingResponse
    from core.inference.api_monitor import ApiMonitor
    from storage.api_usage_db import record_api_usage

    client, body, _ = admission
    monitor = ApiMonitor()
    monitor.set_terminal_callback(record_api_usage)

    def finish():
        entry = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = body["model"],
            prompt = "private prompt",
            subject = "owner",
        )
        monitor.set_usage(entry, prompt_tokens = 90, completion_tokens = 10, total_tokens = 100)
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
                "localLimit": 200,
                "apiLimit": 100,
            }
        }
    )
    payload = {**body, "temporary_chat": True, "thread_id": "temporary-thread", "stream": stream}
    if external:
        payload["provider_type"] = "openai"
    assert client.post("/v1/chat/completions", json = payload).status_code == 200
    assert client.post("/v1/chat/completions", json = payload).status_code == 429

    studio_db.upsert_app_settings({"token_limit": {"enabled": False}})
    payload["temporary_chat"] = False
    assert client.post("/v1/chat/completions", json = payload).status_code == 200
    conn = studio_db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM api_usage_events").fetchone()[0] == 0
        rows = conn.execute("SELECT * FROM token_usage_ledger").fetchall()
        assert len(rows) == 2
        assert sum(row["total_tokens"] for row in rows) == 200
        assert not any(
            "private" in str(tuple(row)) or "temporary-thread" in str(tuple(row)) for row in rows
        )
    finally:
        conn.close()
