# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import token_limit
from core.inference.api_monitor import ApiMonitor
from routes import token_limit as token_limit_routes
from storage import chat_generation_runs_db as runs
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, record_api_usage
from utils.paths import studio_db_path


def _now_ms() -> int:
    return int(time.time() * 1000)


def _charge(
    total: int,
    *,
    api_key: bool = False,
    receipt_id: str | None = None,
    at = None,
) -> bool:
    return record_api_usage(
        ApiUsageReceipt(
            id = receipt_id or f"receipt-{time.monotonic_ns()}",
            subject = "owner",
            endpoint = "/v1/chat/completions",
            model = "unsloth/model",
            status = "completed",
            prompt_tokens = total - 1,
            completion_tokens = 1,
            total_tokens = total,
            created_at = at if at is not None else _now_ms(),
            via_api_key = api_key,
        )
    )


def _enable(**settings) -> None:
    token_limit.update_settings({"enabled": True, "timezone": "UTC", **settings})


def _used() -> int:
    return token_limit.status()["used"]


def _refused() -> bool:
    try:
        token_limit.enforce()
    except HTTPException as error:
        assert error.status_code == 429
        return True
    return False


def _reach_cap() -> None:
    _charge(100)
    _enable(limit = 100)


async def _ignore(message):
    pass


def test_session_and_api_requests_share_one_allowance_without_touching_api_stats():
    _charge(300)
    _charge(200, api_key = True)
    _enable(limit = 1000)
    assert _used() == 500
    with sqlite3.connect(studio_db_path()) as conn:
        # Session requests never appear in API usage stats.
        assert conn.execute("SELECT COUNT(*) FROM api_usage_events").fetchone()[0] == 1


def test_a_repeated_terminal_receipt_is_charged_once():
    assert _charge(100, receipt_id = "once")
    assert not _charge(100, receipt_id = "once")
    _enable(limit = 1000)
    assert _used() == 100


def test_chat_history_never_changes_usage():
    now = _now_ms()
    studio_db.upsert_chat_thread(
        {"id": "t1", "title": "t1", "modelType": "base", "modelId": "m", "createdAt": now}
    )
    studio_db.upsert_chat_message(
        {
            "id": "a1",
            "threadId": "t1",
            "role": "assistant",
            "createdAt": now,
            "content": [{"type": "text", "text": "answer"}],
            "metadata": {
                "contextUsage": {"promptTokens": 900, "completionTokens": 100, "totalTokens": 1000}
            },
        }
    )
    _charge(150)
    _enable(limit = 1000)
    # Saving, forking or deleting a chat neither charges nor refunds: only requests count.
    studio_db.fork_chat_thread("t1", "a1", "fork", now + 1, id_factory = iter(["copy"]).__next__)
    studio_db.delete_chat_threads(["t1"])
    assert _used() == 150


def test_usage_before_the_period_does_not_count():
    start, _end = token_limit.period_window("day", "UTC")
    _charge(5000, at = start - 1)
    _charge(900, api_key = True, at = start - 1)
    _charge(150, at = start + 1)
    _charge(300, api_key = True, at = start + 2)
    _enable(limit = 1000)
    assert _used() == 150 + 300


def test_periods_start_at_local_midnight_monday_and_the_first():
    now = int(datetime(2026, 11, 4, 5, 0, tzinfo = timezone.utc).timestamp() * 1000)
    tz = "Asia/Kolkata"
    zone = token_limit._zone(tz)
    for period, start, end in (
        ("day", "2026-11-04 00:00", "2026-11-05 00:00"),
        ("week", "2026-11-02 00:00", "2026-11-09 00:00"),
        ("month", "2026-11-01 00:00", "2026-12-01 00:00"),
    ):
        window = token_limit.period_window(period, tz, now)
        assert [
            datetime.fromtimestamp(edge / 1000, zone).strftime("%Y-%m-%d %H:%M") for edge in window
        ] == [start, end]
    december = int(datetime(2026, 12, 20, tzinfo = timezone.utc).timestamp() * 1000)
    _start, end = token_limit.period_window("month", "UTC", december)
    assert end == int(datetime(2027, 1, 1, tzinfo = timezone.utc).timestamp() * 1000)
    for hour in (4, 5, 6):
        repeated_hour = int(datetime(2026, 11, 1, hour, 30, tzinfo = timezone.utc).timestamp() * 1000)
        start, _end = token_limit.period_window("day", "America/Havana", repeated_hour)
        assert start == int(datetime(2026, 11, 1, 4, tzinfo = timezone.utc).timestamp() * 1000)


def test_the_override_lets_exactly_one_request_through_per_period():
    _enable(limit = 1000)
    with pytest.raises(HTTPException) as below:
        token_limit.grant_override()
    assert below.value.status_code == 409
    _charge(1000)
    assert _refused()
    assert token_limit.grant_override()["override"] == "granted"
    assert not _refused()
    assert token_limit.status()["override"] == "used"
    assert _refused()
    with pytest.raises(HTTPException) as again:
        token_limit.grant_override()
    assert again.value.status_code == 409


@pytest.mark.parametrize(
    "status, output, finish, in_band_failure, refunded",
    [
        (400, False, None, False, True),
        (503, False, None, False, True),
        (502, True, "failed", False, False),
        (200, True, "failed", False, False),
        # A 200 stream can still fail in band, or be cancelled, before generating anything.
        (200, False, None, True, True),
        (200, False, "cancelled", False, True),
        (200, True, "cancelled", False, False),
        (200, False, "completed", False, False),
        # Input-only usage (Anthropic's message_start) before a cancel generated nothing.
        (200, "prompt", "cancelled", False, True),
        # A non-streaming reply records its text in one set_reply.
        (200, "whole", "cancelled", False, False),
    ],
)
def test_the_override_is_refunded_exactly_when_nothing_was_generated(
    status, output, finish, in_band_failure, refunded
):
    from core.inference.llama_keepwarm import mark_response_failed
    from routes.inference import _enforce_token_limit

    _reach_cap()
    token_limit.grant_override()
    monitor = ApiMonitor()

    async def app(scope, receive, send):
        await _enforce_token_limit(SimpleNamespace(state = SimpleNamespace()))
        await send({"type": "http.response.start", "status": status, "headers": []})
        if finish:
            entry = monitor.start(
                endpoint = "/v1/chat/completions", method = "POST", model = "m", prompt = ""
            )
            if output == "prompt":
                monitor.set_usage(entry, prompt_tokens = 50)
            elif output == "whole":
                monitor.set_reply(entry, "partial answer")
            elif output:
                monitor.append_reply(entry, "partial answer")
            monitor.finish(entry, finish)
        if in_band_failure:
            mark_response_failed(scope)
        await send({"type": "http.response.body", "body": b""})

    asyncio.run(token_limit.OverrideRefundMiddleware(app)({"type": "http"}, None, _ignore))
    assert _refused() is not refunded


def test_a_request_that_raises_after_its_response_start_refunds_the_override():
    from routes.inference import _enforce_token_limit

    _reach_cap()
    token_limit.grant_override()

    async def app(scope, receive, send):
        await _enforce_token_limit(SimpleNamespace(state = SimpleNamespace()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ConnectionResetError

    with pytest.raises(ConnectionResetError):
        asyncio.run(token_limit.OverrideRefundMiddleware(app)({"type": "http"}, None, _ignore))
    assert not _refused()


def _durable_run(spent: int) -> None:
    studio_db.upsert_chat_thread(
        {"id": "t1", "title": "t1", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {"id": "user", "threadId": "t1", "role": "user", "content": [], "createdAt": 2}
    )
    runs.create_run(
        run_id = "run",
        owner_subject = "owner",
        thread_id = "t1",
        user_message_id = "user",
        assistant_message_id = "assistant",
        request_payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            runs.TOKEN_LIMIT_OVERRIDE_FIELD: spent,
        },
    )


@pytest.mark.parametrize(
    "outcome, refunded",
    [
        ("refused", True),
        ("failed_empty", True),
        ("generated_then_failed", False),
        ("completed", False),
    ],
)
def test_a_durable_run_refunds_its_override_only_when_it_generated_nothing(
    monkeypatch, outcome, refunded
):
    from core.inference.chat_generation_runs import ChatGenerationSupervisor
    from routes import inference

    _reach_cap()
    token_limit.grant_override()
    _durable_run(token_limit.enforce())
    monitor = ApiMonitor()
    monkeypatch.setattr(inference, "api_monitor", monitor)

    async def produce(payload, request, subject, *, cancel_on_disconnect):
        assert runs.TOKEN_LIMIT_OVERRIDE_FIELD not in (payload.model_extra or {})
        if outcome == "refused":
            raise HTTPException(status_code = 400, detail = "No model loaded")
        entry = monitor.start(endpoint = "/v1/chat/completions", method = "POST", model = "m", prompt = "")

        async def body():
            if outcome != "failed_empty":
                monitor.append_reply(entry, "partial answer")
                yield 'data: {"choices": [{"delta": {"content": "partial answer"}}]}\n\n'
            if outcome == "completed":
                monitor.finish(entry)
                yield "data: [DONE]\n\n"
            else:
                monitor.fail(entry, "down")
                yield 'data: {"error": {"message": "down"}}\n\n'

        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", produce)
    app = SimpleNamespace(state = SimpleNamespace(bind_host = "127.0.0.1"))
    asyncio.run(ChatGenerationSupervisor(app)._produce("run"))
    assert _refused() is not refunded


@pytest.mark.parametrize("finished_first", [False, True])
def test_a_durable_run_stopped_before_its_payload_loads_refunds_only_once_owned(
    monkeypatch, finished_first
):
    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    _reach_cap()
    token_limit.grant_override()
    _durable_run(token_limit.enforce())
    if finished_first:
        # Already settled by its own worker: a stray re-dispatch must not refund it.
        token = runs.get_worker_token("run")
        runs.mark_running("run", token)
        runs.finish_run("run", worker_token = token, status = "completed")
    else:
        mark_running = runs.mark_running

        def stop_after_mark(run_id, token):
            started = mark_running(run_id, token)
            runs.request_cancel(run_id)
            return started

        monkeypatch.setattr(runs, "mark_running", stop_after_mark)
    app = SimpleNamespace(state = SimpleNamespace(bind_host = "127.0.0.1"))
    asyncio.run(ChatGenerationSupervisor(app)._produce("run"))
    assert _refused() is finished_first


def test_cancelling_a_queued_run_refunds_its_override():
    _reach_cap()
    token_limit.grant_override()
    _durable_run(token_limit.enforce())
    runs.request_cancel("run")
    assert not _refused()


def test_settings_reads_stay_fresh_across_keeper_lifetimes():
    studio_db.upsert_app_settings({"token_limit": {"limit": 100}})
    assert token_limit.load_settings()["limit"] == 100
    assert studio_db.open_wal_keeper()
    try:
        assert token_limit.load_settings()["limit"] == 100
        with sqlite3.connect(studio_db_path()) as writer:
            writer.execute(
                "UPDATE app_settings SET value_json = ? WHERE key = 'token_limit'",
                (json.dumps({"limit": 200}),),
            )
        assert token_limit.load_settings()["limit"] == 200
    finally:
        studio_db.close_wal_keeper_for(studio_db_path())
    assert token_limit.load_settings()["limit"] == 200


def test_disabled_monitor_cannot_silently_leave_an_allowance_unenforced(monkeypatch):
    from core.inference import api_monitor

    monkeypatch.setattr(api_monitor, "api_monitor", api_monitor.ApiMonitor(enabled = False))
    with pytest.raises(HTTPException) as error:
        _enable()
    assert error.value.status_code == 409
    assert token_limit.load_settings()["enabled"] is False
    studio_db.upsert_app_settings({"token_limit": {"enabled": True}})
    for action in (token_limit.enforce, token_limit.grant_override):
        with pytest.raises(HTTPException) as error:
            action()
        assert error.value.status_code == 409
    token_limit.update_settings({"enabled": False})
    token_limit.enforce()


def test_routes_refuse_api_key_changes_and_report_status():
    app = FastAPI()
    app.include_router(token_limit_routes.router, prefix = "/api/token-limit")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    via_api_key = {"value": True}
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key["value"]
    client = TestClient(app)

    assert client.put("/api/token-limit", json = {"enabled": True}).status_code == 403
    assert client.post("/api/token-limit/override").status_code == 403
    via_api_key["value"] = False
    saved = client.put("/api/token-limit", json = {"enabled": True, "limit": 5, "timezone": "UTC"})
    assert saved.status_code == 200
    assert client.put("/api/token-limit", json = {"limit": 0}).status_code == 422
    status = client.get("/api/token-limit/status").json()
    assert (status["used"], status["reached"], status["warning"], status["override"]) == (
        0,
        False,
        False,
        None,
    )
    assert status["resetsAt"] > status["periodStart"]
    _charge(6)
    assert client.post("/api/token-limit/override").status_code == 200
    assert not _refused()
    assert _refused()
