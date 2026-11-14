# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import hashlib
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
from routes import token_limit as token_limit_routes
from storage import chat_generation_runs_db as runs
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, record_api_usage
from storage.token_usage_ledger import record_temporary_usage
from utils.paths import studio_db_path


def _now_ms() -> int:
    return int(time.time() * 1000)


def _thread(thread_id: str, created_at: int) -> None:
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": thread_id,
            "modelType": "base",
            "modelId": "unsloth/model",
            "createdAt": created_at,
        }
    )


def _message(
    message_id,
    thread_id,
    created_at,
    metadata,
    *,
    role = "assistant",
    parent = None,
):
    studio_db.upsert_chat_message(
        {
            "id": message_id,
            "threadId": thread_id,
            "role": role,
            "parentId": parent,
            "createdAt": created_at,
            "content": [{"type": "text", "text": "answer"}],
            "metadata": metadata,
        }
    )


def _usage(prompt: int, completion: int) -> dict:
    return {
        "contextUsage": {
            "promptTokens": prompt,
            "completionTokens": completion,
            "totalTokens": prompt + completion,
        }
    }


def _reply(
    message_id,
    thread_id,
    created_at,
    *,
    prompt,
    completion,
    parent = None,
) -> None:
    _message(message_id, thread_id, created_at, _usage(prompt, completion), parent = parent)


def _api_request(receipt_id: str, created_at: int, total: int) -> None:
    assert record_api_usage(
        ApiUsageReceipt(
            id = receipt_id,
            subject = "client",
            endpoint = "/v1/chat/completions",
            model = "unsloth/model",
            status = "completed",
            prompt_tokens = total - 1,
            completion_tokens = 1,
            total_tokens = total,
            created_at = created_at,
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


def _spend_override() -> int:
    """Reaches a 100 token cap, grants the override and spends it."""
    now = _now_ms()
    _thread("seed", now)
    _reply("seed-reply", "seed", now + 1, prompt = 90, completion = 10)
    _enable(limit = 100)
    token_limit.grant_override()
    return token_limit.enforce()


def _run(request_payload, *, start = True):
    now = _now_ms()
    _thread("t1", now)
    _message("user", "t1", now, None, role = "user")
    runs.create_run(
        run_id = "run",
        owner_subject = "owner",
        thread_id = "t1",
        user_message_id = "user",
        assistant_message_id = "assistant",
        request_payload = {"model": "local", "messages": [], **request_payload},
    )
    token = runs.get_worker_token("run")
    if start:
        runs.mark_running("run", token)
    return token


def _receipt(prompt = 90, completion = 10):
    # The chat route keys a temporary receipt by sha256(thread \0 message).
    return SimpleNamespace(
        id = "receipt",
        temporary_message_id = hashlib.sha256(b"t1\0a1").hexdigest(),
        temporary_thread = hashlib.sha256(b"t1").hexdigest(),
        created_at = _now_ms(),
        prompt_tokens = prompt,
        completion_tokens = completion,
        total_tokens = prompt + completion,
    )


def test_deleting_a_chat_does_not_give_its_tokens_back():
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 900, completion = 100)
    _enable(limit = 1000)
    studio_db.delete_chat_threads(["t1"])
    assert _used() == 1000
    assert _refused()


def test_rewriting_a_reply_does_not_shrink_its_count():
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 400, completion = 100)
    _reply("a1", "t1", now + 1, prompt = 10, completion = 5)
    _enable(limit = 10_000)
    assert _used() == 500


def test_a_reported_count_replaces_an_earlier_estimate():
    now = _now_ms()
    _thread("t1", now)
    for metadata in (
        {"resumeTiming": {"tokenCount": 400}},
        _usage(50, 300),
        {"resumeTiming": {"tokenCount": 900}},
    ):
        _message("a1", "t1", now + 1, metadata)
    _enable(limit = 10_000)
    # The estimate gives way to the report; a later estimate cannot raise it again.
    assert _used() == 350


def test_client_metadata_cannot_waive_an_existing_debit():
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 90, completion = 10)
    _enable(limit = 100)
    message = studio_db.get_chat_message("t1", "a1")
    studio_db.upsert_chat_message(
        {**message, "metadata": {**(message.get("metadata") or {}), "tokenUsageCharged": True}}
    )
    assert _used() == 100
    assert "tokenUsageCharged" not in (studio_db.get_chat_message("t1", "a1").get("metadata") or {})


@pytest.mark.parametrize("receipt_first", [True, False])
def test_saving_a_temporary_chat_charges_its_reply_once(receipt_first):
    if receipt_first:
        record_temporary_usage(_receipt())
    now = _now_ms()
    _thread("t1", now)
    _message("a1", "t1", now + 1, {"tokenUsageCharged": True, **_usage(90, 10)})
    if not receipt_first:
        record_temporary_usage(_receipt())
    _enable(limit = 10_000)
    assert _used() == 100


@pytest.mark.parametrize(
    "metadata, used",
    [
        # An estimated prompt counts only once output shows inference ran (not a refusal).
        ({"tokenUsageEstimatedPrompt": 10_000, "incomplete": {"reason": "interrupted"}}, 0),
        ({"tokenUsageEstimatedPrompt": 100, "tokenUsageEstimatedToolOutput": 2_500}, 2_600),
        ({"resumeTiming": {"tokenCount": 10}, "tokenUsageEstimatedPrompt": 10_000}, 10_010),
        # A reply saved only mid-stream is charged; a continued one minus what its source cost.
        ({"resumeTiming": {"tokenCount": 150}, "tokenUsagePriorCount": 100}, 50),
        ({"timing": {"tokenCount": 100}, "incomplete": {"reason": "interrupted"}}, 100),
    ],
)
def test_saved_reply_estimates(metadata, used):
    now = _now_ms()
    _thread("t1", now)
    _message("a1", "t1", now + 1, metadata)
    _enable(limit = 100_000)
    assert _used() == used


def test_a_rejected_continuation_does_not_charge_the_reply_it_continues():
    now = _now_ms()
    _thread("t1", now)
    _message("original", "t1", now + 1, {"timing": {"tokenCount": 100}})
    _message(
        "continued", "t1", now + 2, {"timing": {"tokenCount": 100}, "tokenUsagePriorCount": 100}
    )
    _enable(limit = 1000)
    assert _used() == 100


async def _ignore(message):
    pass


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
    ],
)
def test_the_override_is_refunded_exactly_when_nothing_was_generated(
    status, output, finish, in_band_failure, refunded
):
    from core.inference.api_monitor import ApiMonitor
    from core.inference.llama_keepwarm import mark_response_failed
    from routes.inference import _enforce_token_limit

    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 90, completion = 10)
    _enable(limit = 100)
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
            elif output:
                monitor.append_reply(entry, "partial answer")
            monitor.finish(entry, finish)
        if in_band_failure:
            mark_response_failed(scope)
        await send({"type": "http.response.body", "body": b""})

    asyncio.run(token_limit.OverrideRefundMiddleware(app)({"type": "http"}, None, _ignore))
    assert _refused() is not refunded


# Refused before inference (e.g. no model loaded), or cancelled while still queued.
@pytest.mark.parametrize("queued", [False, True])
def test_a_durable_run_that_generated_nothing_refunds_its_override(queued):
    token = _run({"token_limit_override": _spend_override()}, start = not queued)
    if queued:
        runs.request_cancel("run")
    else:
        runs.finish_run("run", worker_token = token, status = "failed", error = "No model loaded")
    assert not _refused()


def test_a_finalized_tool_call_is_charged_and_keeps_the_override_spent():
    spent = _spend_override()
    token = _run({"token_limit_override": spent})
    runs.append_events(
        "run",
        token,
        [
            ("chunk", {"type": "tool_args", "tool_call_id": "r1", "text": "h" * 100}),
            ("chunk", {"type": "tool_start", "tool_call_id": "r1", "arguments_text": "h" * 100}),
            (
                "chunk",
                {"type": "tool_start", "tool_call_id": "p1", "arguments_text": "x" * 300},
            ),
        ],
    )
    assert runs.reconcile_orphaned_runs() == 1
    # 100 streamed + 300 finalized characters, the streamed call not counted twice.
    assert _used() == 100 + 100
    assert _refused()


_PASS = {"type": "context_usage", "prompt_tokens": 900, "completion_tokens": 100}


@pytest.mark.parametrize(
    "messages, events, status, used",
    [
        # Reply, reasoning and generated tool arguments count; tool output does not.
        (
            [],
            [
                {"choices": [{"delta": {"content": "a" * 400}}]},
                {"type": "tool_args", "tool_name": "render_html", "text": "h" * 400},
                {"type": "tool_end", "tool_name": "render_html", "result": "x" * 9000},
            ],
            None,
            200,
        ),
        (
            [{"role": "user", "content": "x" * 400}],
            [
                {"choices": [{"delta": {"reasoning_content": "t" * 400}}]},
                {"choices": [{"delta": {"content": "a" * 400}}]},
            ],
            "cancelled",
            100 + 200,
        ),
        # A backend that reports no stats (diffusion LMs) completes with only its text.
        (
            [{"role": "user", "content": "x" * 400}],
            [{"choices": [{"delta": {"content": "a" * 400}}]}],
            "completed",
            100 + 100,
        ),
        (
            [],
            [{"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "x" * 400}}]}}]}],
            "cancelled",
            100,
        ),
        # A stop after a pass report adds only what streamed after it.
        (
            [],
            [
                {"choices": [{"delta": {"content": "b" * 40}}]},
                {"type": "context_usage", "prompt_tokens": 100, "completion_tokens": 10},
                {"choices": [{"delta": {"content": "a" * 4000}}]},
            ],
            "cancelled",
            110 + 1000,
        ),
        # Interrupted tool passes: the last prompt plus every pass's completion.
        (
            [],
            [{**_PASS, "prompt_tokens": 700, "completion_tokens": 50}, _PASS],
            "cancelled",
            900 + 50 + 100,
        ),
        (
            [],
            [
                _PASS,
                {"usage": {"prompt_tokens": 1100, "completion_tokens": 150, "total_tokens": 1250}},
            ],
            "completed",
            1250,
        ),
    ],
)
def test_an_interrupted_run_settles_what_it_generated(messages, events, status, used):
    token = _run({"messages": messages})
    runs.append_events("run", token, [("chunk", event) for event in events])
    if status:
        runs.finish_run("run", worker_token = token, status = status)
    else:
        assert runs.reconcile_orphaned_runs() == 1
    _enable(limit = 100_000)
    assert _used() == used


def test_a_settled_estimate_gives_way_to_reported_usage():
    token = _run({"messages": [{"role": "user", "content": "x" * 400}]})
    runs.append_events("run", token, [("chunk", {"choices": [{"delta": {"content": "a" * 400}}]})])
    runs.finish_run("run", worker_token = token, status = "cancelled")
    _enable(limit = 10_000)
    assert _used() == 200
    message = studio_db.get_chat_message("t1", "assistant")
    studio_db.upsert_chat_message(
        {**message, "metadata": {**message["metadata"], **_usage(90, 60)}}
    )
    assert _used() == 150


@pytest.mark.parametrize("via_api_key", [False, True])
def test_durable_usage_is_counted_once_before_a_browser_recovers_the_run(via_api_key):
    token = _run(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "api_monitor_via_api_key": via_api_key,
        }
    )
    usage = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}
    runs.append_events("run", token, [("chunk", {"usage": usage})])
    runs.finish_run("run", worker_token = token, status = "completed")
    if via_api_key:
        # An API-key run is charged by its receipt, never again by its saved row.
        _api_request("durable-receipt", _now_ms(), 100)
    _enable(limit = 100)
    message = studio_db.get_chat_message("t1", "assistant")
    message["metadata"] = {**(message["metadata"] or {}), **_usage(90, 10)}
    studio_db.upsert_chat_message(message)
    message["metadata"] = _usage(90, 10)
    message["content"] = [{"type": "text", "text": "edited"}]
    studio_db.upsert_chat_message(message, allow_generation_edit = True)
    assert _used() == 100
    assert _refused()


@pytest.mark.parametrize("clock_offset", [-10, 1, 10])
def test_a_fork_does_not_count_the_history_it_copies(clock_offset):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 400, completion = 100)
    forked = studio_db.fork_chat_thread(
        "t1", "a1", "fork", now + clock_offset, id_factory = iter(["copy"]).__next__
    )
    assert forked is not None
    _reply("a2", "fork", now + 20, prompt = 600, completion = 50)
    _enable(limit = 10_000)
    assert _used() == 500 + 650


@pytest.mark.parametrize("old_layout", [False, True])
def test_history_from_before_the_limiter_existed_is_backfilled(old_layout):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 70, completion = 30)
    with sqlite3.connect(studio_db_path()) as conn:
        conn.execute("DROP TABLE token_usage_ledger")
        if old_layout:
            conn.execute("""CREATE TABLE token_usage_ledger (
                message_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, created_at INTEGER NOT NULL,
                source TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL
            ) WITHOUT ROWID""")
    studio_db._schema_ready.clear()
    _enable(limit = 10_000)
    assert _used() == 100


def test_malformed_metadata_never_blocks_the_chat_write():
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 70, completion = 30)
    with sqlite3.connect(studio_db_path()) as conn:
        conn.execute("UPDATE chat_messages SET metadata_json = '{broken' WHERE id = 'a1'")
    _enable(limit = 10_000)
    assert _used() == 100


def test_usage_before_the_period_does_not_count():
    start, _end = token_limit.period_window("day", "UTC")
    _thread("t1", start - 10_000)
    _reply("yesterday", "t1", start - 5_000, prompt = 5000, completion = 0)
    _reply("today", "t1", start + 1, prompt = 100, completion = 50)
    _api_request("old", start - 1, 900)
    _api_request("new", start + 2, 300)
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
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 990, completion = 10)
    assert _refused()
    assert token_limit.grant_override()["override"] == "granted"
    assert not _refused()
    assert token_limit.status()["override"] == "used"
    assert _refused()
    with pytest.raises(HTTPException) as again:
        token_limit.grant_override()
    assert again.value.status_code == 409


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
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now, prompt = 5, completion = 1)
    assert client.post("/api/token-limit/override").status_code == 200
    assert not _refused()
    assert _refused()
