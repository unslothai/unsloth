# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.inference.api_monitor import ApiMonitor


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
