# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Profile statistics aggregation over local chat/training history."""

import json
import time
from datetime import datetime, timedelta

import pytest

from storage import profile_stats_db, studio_db
from storage.profile_stats_db import compute_profile_stats, invalidate_profile_stats_cache


@pytest.fixture
def stats_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    invalidate_profile_stats_cache()
    yield
    invalidate_profile_stats_cache()


def _ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


def _seed_thread(conn, thread_id: str, model_id: str, turns: list[tuple[datetime, dict]]):
    conn.execute(
        "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
        "VALUES (?, ?, 'base', ?, ?, ?)",
        (thread_id, f"Thread {thread_id}", model_id, _ms(turns[0][0]), _ms(turns[-1][0])),
    )
    for index, (when, metadata) in enumerate(turns):
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"{thread_id}-u{index}",
                thread_id,
                "user",
                json.dumps([{"type": "text", "text": "hi"}]),
                None,
                _ms(when),
            ),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"{thread_id}-a{index}",
                thread_id,
                "assistant",
                json.dumps([{"type": "text", "text": "hello"}]),
                json.dumps(metadata),
                _ms(when + timedelta(seconds = 10)),
            ),
        )


def _metadata(
    prompt: int,
    completion: int,
    *,
    speed: float = 40.0,
    tools: int = 0,
) -> dict:
    return {
        "contextUsage": {
            "promptTokens": prompt,
            "completionTokens": completion,
            "totalTokens": prompt + completion,
            "cachedTokens": 5,
            "modelId": "unsloth/gpt-oss-20b",
        },
        "timing": {
            "streamStartTime": 1000,
            "firstTokenTime": 1200,
            "totalStreamTime": 2000,
            "tokenCount": completion,
            "tokensPerSecond": speed,
            "toolCallCount": tools,
        },
    }


def test_empty_history_returns_zeroed_payload(stats_db):
    stats = compute_profile_stats(days = 30)

    assert stats["totals"]["messages"] == 0
    assert stats["totals"]["totalTokens"] == 0
    assert stats["streak"] == {"current": 0, "longest": 0, "lastActiveDay": None}
    assert stats["peakDay"] is None
    assert stats["longestChat"] is None
    assert len(stats["daily"]) == 30
    assert all(day["tokens"] == 0 for day in stats["daily"])


def test_tokens_streaks_and_models_are_aggregated(stats_db):
    today = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(
            conn,
            "t1",
            "unsloth/gpt-oss-20b",
            [
                (today - timedelta(days = 2), _metadata(100, 50, speed = 30.0, tools = 2)),
                (today - timedelta(days = 1), _metadata(200, 80, speed = 55.0)),
                (today, _metadata(300, 120, speed = 120.0, tools = 1)),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 30)

    totals = stats["totals"]
    assert totals["threads"] == 1
    assert totals["messages"] == 6
    assert totals["userMessages"] == 3
    assert totals["assistantMessages"] == 3
    assert totals["promptTokens"] == 600
    assert totals["completionTokens"] == 250
    assert totals["totalTokens"] == 850
    assert totals["cachedTokens"] == 15
    assert totals["toolCalls"] == 3
    assert totals["activeDays"] == 3

    assert stats["streak"] == {
        "current": 3,
        "longest": 3,
        "lastActiveDay": today.date().isoformat(),
    }
    assert stats["peakDay"] == {"date": today.date().isoformat(), "tokens": 420}
    assert stats["models"][0]["id"] == "unsloth/gpt-oss-20b"
    assert stats["models"][0]["label"] == "gpt-oss-20b"
    assert stats["models"][0]["messages"] == 3
    assert stats["speed"]["bestTokensPerSecond"] == 120.0
    assert stats["speed"]["averageTokensPerSecond"] == pytest.approx(68.333, rel = 1e-3)

    # Each turn is a user message plus an assistant reply 10s later.
    assert stats["longestChat"]["seconds"] == 30
    assert stats["longestChat"]["messages"] == 6


def test_completion_tokens_fall_back_to_adapter_count(stats_db):
    """Local engines can omit the usage chunk; timing.tokenCount stands in."""
    now = datetime.now().replace(hour = 9, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(
            conn,
            "t2",
            "local-gguf",
            [(now, {"timing": {"tokenCount": 64, "tokensPerSecond": 12.0}})],
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    assert stats["totals"]["completionTokens"] == 64
    assert stats["totals"]["totalTokens"] == 64
    # No modelId in metadata: the thread's model is used instead.
    assert stats["models"][0]["id"] == "local-gguf"


def test_session_time_ignores_long_idle_gaps(stats_db):
    """A thread reopened days later must not count the idle time as chatting."""
    start = datetime.now().replace(hour = 10, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(
            conn,
            "t3",
            "m",
            [(start - timedelta(days = 3), _metadata(10, 10)), (start, _metadata(10, 10))],
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 30)

    # Two turns of 10s each; the 3-day gap between them is excluded.
    assert stats["longestChat"]["seconds"] == 20
    assert stats["totals"]["chatSeconds"] == 20


def test_broken_metadata_does_not_break_aggregation(stats_db):
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('t4', 'Broken', 'base', 'm', ?, ?)",
            (_ms(now), _ms(now)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, created_at) "
            "VALUES ('t4-a0', 't4', 'assistant', '[]', ?, ?)",
            ("{not json", _ms(now)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, created_at) "
            "VALUES ('t4-a1', 't4', 'assistant', '[]', ?, ?)",
            (json.dumps({"contextUsage": {"totalTokens": "lots"}}), _ms(now)),
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    assert stats["totals"]["assistantMessages"] == 2
    assert stats["totals"]["totalTokens"] == 0


def test_training_totals(stats_db):
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, ended_at, total_steps, final_step, final_loss, duration_seconds) "
            "VALUES ('r1', 'completed', 'unsloth/llama-3-8b', 'tatsu-lab/alpaca', '{}', "
            "'2026-01-01T10:00:00', '2026-01-01T11:00:00', 100, 100, 0.42, 3600)",
        )
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, final_loss, duration_seconds) "
            "VALUES ('r2', 'error', 'unsloth/qwen3-4b', 'my/dataset', '{}', "
            "'2026-01-02T10:00:00', 100, 20, 1.8, 600)",
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, loss, num_tokens) VALUES (?, ?, ?, ?)",
            [("r1", step, 1.0, 1000) for step in range(10)],
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    training = stats["training"]
    assert training["runs"] == 2
    assert training["completed"] == 1
    assert training["steps"] == 120
    assert training["tokens"] == 10_000
    assert training["seconds"] == 4200
    assert training["models"] == 2
    assert training["bestLoss"] == pytest.approx(0.42)
    assert training["recent"][0]["id"] == "r2"
    assert training["recent"][0]["modelLabel"] == "qwen3-4b"


def test_repeat_calls_are_served_from_cache_until_history_changes(stats_db):
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "t5", "m", [(now, _metadata(10, 10))])
        conn.commit()
    finally:
        conn.close()

    first = compute_profile_stats(days = 7)
    second = compute_profile_stats(days = 7)
    assert first is second

    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "t6", "m", [(now, _metadata(20, 20))])
        conn.commit()
    finally:
        conn.close()

    third = compute_profile_stats(days = 7)
    assert third is not first
    assert third["totals"]["totalTokens"] == 60


def test_daily_series_is_dense_and_clamped(stats_db):
    stats = compute_profile_stats(days = 10_000)
    assert len(stats["daily"]) == profile_stats_db.MAX_DAILY_DAYS
    dates = [day["date"] for day in stats["daily"]]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_route_does_not_block_the_event_loop(stats_db, monkeypatch):
    """A cold stats pass must not stall streaming for the rest of the app.

    The aggregation is CPU-bound and can run for a second on large histories,
    so the route offloads it to a worker thread. This drives the endpoint with a
    heartbeat coroutine alongside it and asserts the loop kept ticking.
    """
    import asyncio

    from routes import profile_stats as route_module

    def slow_compute(days = 366):
        time.sleep(0.5)
        return {"totals": {"messages": 0}}

    monkeypatch.setattr(route_module, "compute_profile_stats", slow_compute)

    async def drive() -> int:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await route_module.get_profile_stats(days = 366, current_subject = "unsloth")
        finally:
            beat.cancel()
        return ticks

    ticks = asyncio.run(drive())

    # ~50 ticks fit in 0.5s; a blocking call on the loop would yield 0.
    assert ticks > 10, f"event loop stalled during stats computation ({ticks} ticks)"
