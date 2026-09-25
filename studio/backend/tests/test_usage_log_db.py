# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
import pytest
from core.inference.api_monitor import ApiMonitorEntry
from storage.usage_log import (
    record_event,
    query_events,
    aggregate_usage,
    sum_tokens_in_window,
    enforce_retention,
    count_events,
)
from storage.studio_db import upsert_app_settings


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import storage.studio_db as studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def test_record_and_query_events(isolated_db):
    entry = ApiMonitorEntry(
        id = "test1",
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "llama3",
        prompt = "hello",
        status = "completed",
        started_at = time.time(),
        updated_at = time.time(),
        provider_type = "openai",
        subject = "alice",
    )
    entry.prompt_tokens = 10
    entry.completion_tokens = 20
    entry.total_tokens = 30

    record_event(entry)

    events = query_events()
    assert len(events) == 1
    e = events[0]
    assert e["model"] == "llama3"
    assert e["source"] == "api"
    assert e["provider"] == "openai"
    assert e["total_tokens"] == 30
    assert e["session_id"] == "alice"

    # Test filters
    assert len(query_events(model = "llama3")) == 1
    assert len(query_events(model = "other")) == 0
    assert len(query_events(source = "api")) == 1
    assert len(query_events(source = "local")) == 0

    assert count_events(model = "llama3") == 1


def test_aggregate_usage(isolated_db):
    now_ms = int(time.time() * 1000)
    day_ms = 24 * 60 * 60 * 1000

    for i in range(3):
        entry = ApiMonitorEntry(
            id = f"test{i}",
            endpoint = "/chat",
            method = "POST",
            model = "llama3",
            prompt = "hi",
            status = "completed",
            started_at = (now_ms - i * day_ms) / 1000.0,
            updated_at = (now_ms - i * day_ms) / 1000.0,
        )
        entry.prompt_tokens = 10
        entry.completion_tokens = 5
        entry.total_tokens = 15
        record_event(entry)

    agg = aggregate_usage(granularity = "day")
    assert len(agg) == 3
    for row in agg:
        assert row["total_tokens"] == 15


def test_sum_tokens(isolated_db):
    entry = ApiMonitorEntry(
        id = "test1",
        endpoint = "/chat",
        method = "POST",
        model = "llama3",
        prompt = "hi",
        status = "completed",
        started_at = time.time(),
        updated_at = time.time(),
    )
    entry.total_tokens = 100
    record_event(entry)

    since = int((time.time() - 3600) * 1000)
    assert sum_tokens_in_window(since_ts = since) == 100


def test_retention(isolated_db):
    old_entry = ApiMonitorEntry(
        id = "old1",
        endpoint = "/chat",
        method = "POST",
        model = "llama3",
        prompt = "hi",
        status = "completed",
        started_at = time.time() - (90 * 24 * 3600),
        updated_at = time.time() - (90 * 24 * 3600),
    )
    record_event(old_entry)

    new_entry = ApiMonitorEntry(
        id = "new1",
        endpoint = "/chat",
        method = "POST",
        model = "llama3",
        prompt = "hi",
        status = "completed",
        started_at = time.time(),
        updated_at = time.time(),
    )
    record_event(new_entry)

    assert count_events() == 2

    upsert_app_settings({"usage_retention_policy": {"mode": "months", "value": 2}})
    enforce_retention()

    assert count_events() == 1
    assert query_events()[0]["id"] == "new1"


def test_session_id_scoping(isolated_db):
    """Events from different users are isolated by session_id filter."""
    for user, model in [("alice", "gpt4"), ("bob", "llama3"), ("alice", "llama3")]:
        entry = ApiMonitorEntry(
            id = f"{user}-{model}",
            endpoint = "/chat",
            method = "POST",
            model = model,
            prompt = "hi",
            status = "completed",
            started_at = time.time(),
            updated_at = time.time(),
            subject = user,
        )
        entry.total_tokens = 10
        record_event(entry)

    # Unscoped: all 3 events
    assert count_events() == 3

    # Scoped: only alice's events
    assert count_events(session_id = "alice") == 2
    assert count_events(session_id = "bob") == 1
    assert count_events(session_id = "charlie") == 0

    # query_events respects session_id
    alice_events = query_events(session_id = "alice")
    assert len(alice_events) == 2
    assert all(e["session_id"] == "alice" for e in alice_events)

    # aggregate_usage respects session_id
    agg = aggregate_usage(granularity = "day", session_id = "bob")
    assert len(agg) == 1
    assert agg[0]["total_tokens"] == 10

    # sum_tokens_in_window respects session_id
    since = int((time.time() - 3600) * 1000)
    assert sum_tokens_in_window(since_ts = since, session_id = "alice") == 20
    assert sum_tokens_in_window(since_ts = since, session_id = "bob") == 10


def test_retention_mode_forever(isolated_db):
    """enforce_retention is a no-op when mode=forever."""
    entry = ApiMonitorEntry(
        id = "old_forever",
        endpoint = "/chat",
        method = "POST",
        model = "llama3",
        prompt = "hi",
        status = "completed",
        started_at = time.time() - (365 * 24 * 3600),
        updated_at = time.time() - (365 * 24 * 3600),
    )
    record_event(entry)

    upsert_app_settings({"usage_retention_policy": {"mode": "forever", "value": None}})
    enforce_retention()

    assert count_events() == 1


def test_aggregate_invalid_granularity(isolated_db):
    """aggregate_usage raises ValueError on invalid granularity."""
    with pytest.raises(ValueError, match = "Invalid granularity"):
        aggregate_usage(granularity = "hour")


def test_sum_tokens_empty(isolated_db):
    """sum_tokens_in_window returns 0 when no events match."""
    since = int((time.time() - 3600) * 1000)
    assert sum_tokens_in_window(since_ts = since) == 0
    assert sum_tokens_in_window(since_ts = since, session_id = "nobody") == 0
