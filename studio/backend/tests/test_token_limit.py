# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import token_limit
from routes import token_limit as token_limit_routes
from storage import studio_db
from storage.api_usage_db import ApiUsageReceipt, record_api_usage
from utils.paths import studio_db_path


@pytest.fixture
def limiter_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield


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


def _reply(
    message_id: str,
    thread_id: str,
    created_at: int,
    *,
    prompt: int,
    completion: int,
    provider: str = "local",
    context: int | None = None,
    parent: str | None = None,
) -> None:
    studio_db.upsert_chat_message(
        {
            "id": message_id,
            "threadId": thread_id,
            "role": "assistant",
            "parentId": parent,
            "createdAt": created_at,
            "content": [{"type": "text", "text": "answer"}],
            "metadata": {
                "contextUsage": {
                    "promptTokens": prompt,
                    "completionTokens": completion,
                    "totalTokens": prompt + completion,
                    "contextTokens": context,
                },
                "responseDetails": {"providerType": provider},
            },
        }
    )


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
    token_limit.update_settings(
        {"enabled": True, "timezone": "UTC", **settings}, password = None, new_password = None
    )


def _pools() -> dict[str, dict]:
    return {pool["pool"]: pool for pool in token_limit.status()["pools"]}


def test_deleting_a_chat_does_not_give_its_tokens_back(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 900, completion = 100)
    _enable(limit = 1000)
    assert _pools()["combined"]["used"] == 1000

    studio_db.delete_chat_threads(["t1"])

    assert _pools()["combined"]["used"] == 1000
    with pytest.raises(HTTPException) as refused:
        token_limit.enforce(api = False)
    assert refused.value.status_code == 429


def test_rewriting_a_reply_does_not_shrink_its_count(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 400, completion = 100)
    _reply("a1", "t1", now + 1, prompt = 10, completion = 5)
    _enable(limit = 10_000)
    assert _pools()["combined"]["used"] == 500


@pytest.mark.parametrize("clock_offset", [-10, 1, 10])
def test_a_fork_does_not_count_the_history_it_copies(limiter_db, clock_offset):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 400, completion = 100)
    forked = studio_db.fork_chat_thread(
        "t1", "a1", "fork", now + clock_offset, id_factory = iter(["copy"]).__next__
    )
    assert forked is not None
    _reply("a2", "fork", now + 20, prompt = 600, completion = 50)
    _enable(limit = 10_000)
    assert _pools()["combined"]["used"] == 500 + 650
    _enable(method = "incremental")
    assert _pools()["combined"]["used"] == 500 + 100 + 50


@pytest.mark.parametrize("existing_ledger", [False, True])
def test_history_from_before_the_limiter_existed_is_backfilled(limiter_db, existing_ledger):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 70, completion = 30)
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute("DROP TABLE token_usage_ledger")
        if existing_ledger:
            conn.execute("""CREATE TABLE token_usage_ledger (
                message_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, created_at INTEGER NOT NULL,
                source TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL
            ) WITHOUT ROWID""")
        conn.commit()
    finally:
        conn.close()
    studio_db._schema_ready.clear()
    _enable(limit = 10_000)
    assert _pools()["combined"]["used"] == 100


def test_malformed_metadata_never_blocks_the_chat_write(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 70, completion = 30)
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute("UPDATE chat_messages SET metadata_json = '{broken' WHERE id = 'a1'")
        conn.commit()
    finally:
        conn.close()
    _enable(limit = 10_000)
    assert _pools()["combined"]["used"] == 100


@pytest.mark.parametrize(
    "prompt,completion,context,next_prompt,next_completion,expected",
    [
        (100, 50, None, 170, 30, 200),
        (250, 20, 160, 180, 5, 295),
    ],
)
def test_incremental_counting_skips_history_the_previous_reply_saw(
    limiter_db, prompt, completion, context, next_prompt, next_completion, expected
):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = prompt, completion = completion, context = context)
    _reply("a2", "t1", now + 2, prompt = next_prompt, completion = next_completion)
    _enable(limit = 10_000, method = "total")
    assert _pools()["combined"]["used"] == prompt + completion + next_prompt + next_completion
    _enable(limit = 10_000, method = "incremental")
    assert _pools()["combined"]["used"] == expected


def test_incremental_counting_uses_the_reply_ancestor_after_switching_branches(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    for index, parent, prompt, completion, context in (
        (1, None, 100, 20, 120),
        (2, "a1", 1000, 20, 1020),
        (3, "a1", 150, 5, 155),
    ):
        studio_db.upsert_chat_message(
            {
                "id": f"u{index}",
                "threadId": "t1",
                "role": "user",
                "parentId": parent,
                "createdAt": now + index * 2,
                "content": [{"type": "text", "text": "question"}],
            }
        )
        _reply(
            f"a{index}",
            "t1",
            now + index * 2 + 1,
            prompt = prompt,
            completion = completion,
            context = context,
            parent = f"u{index}",
        )
    _enable(limit = 10_000, method = "incremental")
    assert _pools()["combined"]["used"] == 120 + 900 + 35


def test_context_refresh_cannot_refund_a_later_reply(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 100, completion = 10, context = 20)
    _reply("a2", "t1", now + 2, prompt = 120, completion = 10)
    _enable(limit = 1000, method = "incremental")
    assert _pools()["combined"]["used"] == 220
    _reply("a1", "t1", now + 1, prompt = 100, completion = 10, context = 110)
    assert _pools()["combined"]["used"] == 220


def test_split_mode_keeps_local_chats_apart_from_api_and_providers(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("local", "t1", now + 1, prompt = 90, completion = 10)
    _thread("t2", now)
    _reply("cloud", "t2", now + 2, prompt = 190, completion = 10, provider = "openai")
    _api_request("receipt", now + 3, 300)
    _enable(mode = "split", localLimit = 1000, apiLimit = 400)
    pools = _pools()
    assert pools["local"]["used"] == 100 and not pools["local"]["reached"]
    assert pools["api"]["used"] == 500 and pools["api"]["reached"]

    token_limit.enforce(api = False)
    with pytest.raises(HTTPException) as refused:
        token_limit.enforce(api = True)
    assert refused.value.headers["X-Unsloth-Token-Limit"] == "api"

    _enable(mode = "combined", limit = 600)
    with pytest.raises(HTTPException):
        token_limit.enforce(api = False)


def test_usage_before_the_period_does_not_count(limiter_db):
    start, end = token_limit.period_window("day", "UTC")
    _thread("t1", start - 10_000)
    _reply("yesterday", "t1", start - 5_000, prompt = 5000, completion = 0)
    _reply("today", "t1", start + 1, prompt = 5100, completion = 50)
    _enable(limit = 1000, method = "incremental")
    assert _pools()["combined"]["used"] == 150


def test_periods_start_at_local_midnight_monday_and_the_first(limiter_db):
    now = int(datetime(2026, 11, 4, 5, 0, tzinfo = timezone.utc).timestamp() * 1000)
    tz = "Asia/Kolkata"
    for period, start, end in (
        ("day", "2026-11-04 00:00", "2026-11-05 00:00"),
        ("week", "2026-11-02 00:00", "2026-11-09 00:00"),
        ("month", "2026-11-01 00:00", "2026-12-01 00:00"),
    ):
        window = token_limit.period_window(period, tz, now)
        zone = token_limit._zone(tz)
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


def test_the_override_lets_exactly_one_request_through_per_period(limiter_db):
    now = _now_ms()
    _thread("t1", now)
    _reply("a1", "t1", now + 1, prompt = 990, completion = 10)
    _enable(limit = 1000)
    with pytest.raises(HTTPException):
        token_limit.enforce(api = False)

    granted = token_limit.grant_override("combined")
    assert granted["pools"][0]["override"] == "granted"
    token_limit.enforce(api = False)
    assert _pools()["combined"]["override"] == "used"
    with pytest.raises(HTTPException):
        token_limit.enforce(api = False)
    with pytest.raises(HTTPException) as again:
        token_limit.grant_override("combined")
    assert again.value.status_code == 409


def test_no_override_is_granted_below_the_cap(limiter_db):
    _enable(limit = 1000)
    with pytest.raises(HTTPException) as refused:
        token_limit.grant_override("combined")
    assert refused.value.status_code == 409


@pytest.mark.parametrize("via_api_key", [False, True])
def test_durable_usage_is_counted_before_a_browser_recovers_the_run(limiter_db, via_api_key):
    from storage import chat_generation_runs_db as runs

    now = _now_ms()
    _thread("t1", now)
    studio_db.upsert_chat_message(
        {
            "id": "user",
            "threadId": "t1",
            "role": "user",
            "createdAt": now,
            "content": [{"type": "text", "text": "hi"}],
        }
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
            "api_monitor_via_api_key": via_api_key,
        },
    )
    token = runs.get_worker_token("run")
    runs.mark_running("run", token)
    runs.append_events(
        "run",
        token,
        [
            (
                "chunk",
                {
                    "usage": {
                        "prompt_tokens": 90,
                        "completion_tokens": 10,
                        "total_tokens": 100,
                    }
                },
            )
        ],
    )
    runs.finish_run("run", worker_token = token, status = "completed")
    if via_api_key:
        _api_request("durable-receipt", now, 100)
    _enable(limit = 100, method = "incremental")
    assert _pools()["combined"]["used"] == 100
    expected = 100
    if not via_api_key:
        studio_db.fork_chat_thread(
            "t1",
            "assistant",
            "fork",
            now + 10,
            id_factory = iter(["copy-user", "copy-assistant"]).__next__,
        )
        studio_db.upsert_chat_message(
            {
                "id": "fork-user",
                "threadId": "fork",
                "role": "user",
                "parentId": "copy-assistant",
                "createdAt": now + 30,
                "content": [{"type": "text", "text": "next"}],
            }
        )
        _reply("fork-reply", "fork", now + 31, prompt = 120, completion = 10, parent = "fork-user")
        expected = 130
        assert _pools()["combined"]["used"] == expected
    message = studio_db.get_chat_message("t1", "assistant")
    message["metadata"] = {
        **(message["metadata"] or {}),
        "contextUsage": {
            "promptTokens": 90,
            "completionTokens": 10,
            "totalTokens": 100,
        },
    }
    studio_db.upsert_chat_message(message)
    assert _pools()["combined"]["used"] == expected
    with pytest.raises(HTTPException):
        token_limit.enforce(api = False)


def test_a_password_locks_every_change_until_it_is_given(limiter_db):
    token_limit.update_settings(
        {"enabled": True, "limit": 1000}, password = None, new_password = "guardian"
    )
    status = token_limit.status()
    assert status["settings"]["passwordProtected"] is True
    assert "passwordHash" not in status["settings"]

    for password in (None, "wrong"):
        with pytest.raises(token_limit.TokenLimitPasswordError):
            token_limit.update_settings({"enabled": False}, password = password, new_password = None)
    assert token_limit.load_settings()["enabled"] is True

    token_limit.update_settings({"enabled": False}, password = "guardian", new_password = "")
    settings = token_limit.load_settings()
    assert settings["enabled"] is False and settings["passwordHash"] is None


def test_a_concurrent_save_cannot_remove_a_new_password(limiter_db, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    loaded = threading.Event()
    protected = threading.Event()
    original = token_limit.load_settings

    def snapshot():
        value = original()
        if threading.current_thread().name.endswith("_0") and not loaded.is_set():
            loaded.set()
            protected.wait(timeout = 1)
        return value

    def protect():
        assert loaded.wait(timeout = 5)
        token_limit.update_settings({}, password = None, new_password = "guardian")
        protected.set()

    monkeypatch.setattr(token_limit, "load_settings", snapshot)
    with ThreadPoolExecutor(max_workers = 2, thread_name_prefix = "save") as pool:
        first = pool.submit(
            token_limit.update_settings, {"limit": 100}, password = None, new_password = None
        )
        second = pool.submit(protect)
        first.result(timeout = 5)
        second.result(timeout = 5)
    assert token_limit.status()["settings"]["passwordProtected"] is True


def test_routes_refuse_api_key_changes_and_report_status(limiter_db):
    app = FastAPI()
    app.include_router(token_limit_routes.router, prefix = "/api/token-limit")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    via_api_key = {"value": True}
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key["value"]
    client = TestClient(app)

    assert client.put("/api/token-limit", json = {"enabled": True}).status_code == 403
    via_api_key["value"] = False
    saved = client.put(
        "/api/token-limit",
        json = {"enabled": True, "limit": 5, "timezone": "UTC", "newPassword": "pw"},
    )
    assert saved.status_code == 200
    assert saved.json()["settings"]["passwordProtected"] is True
    assert client.put("/api/token-limit", json = {"enabled": False}).status_code == 403
    assert client.put("/api/token-limit", json = {"limit": 0, "password": "pw"}).status_code == 422
    status = client.get("/api/token-limit/status").json()
    assert status["pools"] == [
        {
            "pool": "combined",
            "used": 0,
            "limit": 5,
            "reached": False,
            "warning": False,
            "override": None,
        }
    ]
    assert status["resetsAt"] > status["periodStart"]
