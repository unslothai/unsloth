# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.inference.api_monitor import ApiMonitor, _trim


def test_api_monitor_tracks_reply_usage_and_context():
    monitor = ApiMonitor(max_entries = 3)

    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "user: hello",
        context_length = 100,
    )
    monitor.append_reply(entry_id, "hi")
    monitor.append_reply(entry_id, " there")
    monitor.set_usage(
        entry_id,
        prompt_tokens = 4,
        completion_tokens = 6,
    )
    monitor.finish(entry_id)

    [entry] = monitor.snapshot()
    assert entry["status"] == "completed"
    assert entry["reply"] == "hi there"
    assert entry["total_tokens"] == 10
    assert entry["context_usage"] == 0.1
    assert entry["duration_ms"] is not None


def test_api_monitor_keeps_bounded_recent_history():
    monitor = ApiMonitor(max_entries = 2)

    first = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "first",
    )
    monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "second",
    )
    third = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "third",
    )

    entries = monitor.snapshot()
    ids = [entry["id"] for entry in entries]
    assert ids[0] == third
    assert [entry["prompt"] for entry in entries] == ["third", "second"]
    assert first not in ids
    assert monitor.active_count() == 2


def test_api_monitor_finish_is_idempotent():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.finish(entry_id)
    first = monitor.snapshot()[0]
    monitor.finish(entry_id)
    second = monitor.snapshot()[0]
    assert first["finished_at"] == second["finished_at"]
    assert first["duration_ms"] == second["duration_ms"]


def test_api_monitor_preserves_authoritative_total_tokens():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_usage(
        entry_id,
        prompt_tokens = 10,
        completion_tokens = 20,
        total_tokens = 33,
    )
    # A later partial chunk omitting `total_tokens` must not clobber 33.
    monitor.set_usage(entry_id, prompt_tokens = 11)
    assert monitor.snapshot()[0]["total_tokens"] == 33


def test_api_monitor_duration_non_negative_under_clock_step(monkeypatch):
    import core.inference.api_monitor as m

    fake_now = [1000.0]
    monkeypatch.setattr(m.time, "time", lambda: fake_now[0])
    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/x",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    fake_now[0] = 500.0
    monitor.finish(entry_id)
    assert monitor.snapshot()[0]["duration_ms"] >= 0


def test_api_monitor_trim_guards_tiny_limit():
    assert _trim("abcdefgh", 2) == ".."
    assert _trim("abcdefgh", 0) == ""
    assert _trim("abcdefgh", 3) == "..."
    assert _trim("abcdefgh", 4) == "a..."
    assert _trim("abcdefgh", 100) == "abcdefgh"
