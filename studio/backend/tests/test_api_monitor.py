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


def test_api_monitor_summary_omits_full_prompt_and_reply():
    monitor = ApiMonitor(max_entries = 3)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "local-model",
        prompt = "p" * 500,
    )
    monitor.set_reply(entry_id, "r" * 500)

    [summary] = monitor.snapshot(include_details = False)
    assert "prompt" not in summary
    assert "reply" not in summary
    assert summary["prompt_preview"].endswith("...")
    assert summary["reply_preview"].endswith("...")
    assert summary["prompt_truncated"] is True
    assert summary["reply_truncated"] is True

    detail = monitor.get(entry_id)
    assert detail is not None
    assert detail["prompt"] == "p" * 500
    assert detail["reply"] == "r" * 500


def test_api_monitor_filters_entries_by_subject():
    monitor = ApiMonitor(max_entries = 3)
    alice = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "alice prompt",
        subject = "alice",
    )
    bob = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "bob prompt",
        subject = "bob",
    )
    monitor.finish(bob)

    alice_entries = monitor.snapshot(subject = "alice")
    assert [entry["id"] for entry in alice_entries] == [alice]
    assert monitor.get(bob, subject = "alice") is None
    assert monitor.get(bob, subject = "bob")["id"] == bob
    assert monitor.active_count(subject = "alice") == 1
    assert monitor.active_count(subject = "bob") == 0


def test_api_monitor_keeps_bounded_recent_history():
    monitor = ApiMonitor(max_entries = 2)

    first = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "first",
    )
    second = monitor.start(
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
    monitor.finish(first)
    monitor.finish(second)
    monitor.finish(third)

    entries = monitor.snapshot()
    ids = [entry["id"] for entry in entries]
    assert ids[0] == third
    assert [entry["prompt"] for entry in entries] == ["third", "second"]
    assert first not in ids
    assert monitor.active_count() == 0


def test_api_monitor_keeps_running_entries_beyond_history_limit():
    monitor = ApiMonitor(max_entries = 1)

    running = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "running",
    )
    for prompt in ("done-1", "done-2", "done-3"):
        entry_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "m",
            prompt = prompt,
        )
        monitor.finish(entry_id)

    entries = monitor.snapshot()
    ids = [entry["id"] for entry in entries]
    assert running in ids
    assert monitor.active_count() == 1

    monitor.finish(running)
    [entry] = monitor.snapshot()
    assert entry["id"] == running
    assert entry["status"] == "completed"
    assert monitor.active_count() == 0


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


def test_api_monitor_recomputes_derived_total_tokens():
    monitor = ApiMonitor(max_entries = 2)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "hi",
    )
    monitor.set_usage(entry_id, prompt_tokens = 10)
    assert monitor.snapshot()[0]["total_tokens"] == 10

    monitor.set_usage(entry_id, completion_tokens = 20)
    entry = monitor.snapshot()[0]
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 20
    assert entry["total_tokens"] == 30


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


def test_api_monitor_append_reply_caps_without_regrowing():
    import core.inference.api_monitor as m

    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "go",
    )
    monitor.append_reply(entry_id, "x" * (m._MAX_REPLY_CHARS + 500))
    capped = monitor.snapshot()[0]["reply"]
    assert len(capped) == m._MAX_REPLY_CHARS and capped.endswith("...")

    # Chunks past the cap must not change or grow the stored preview.
    monitor.append_reply(entry_id, "y" * 1000)
    assert monitor.snapshot()[0]["reply"] == capped


def test_api_monitor_append_reply_exact_cap_then_more_marks_truncated():
    import core.inference.api_monitor as m

    monitor = ApiMonitor(max_entries = 1)
    entry_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "go",
    )
    # A reply landing exactly on the cap has no "..." marker yet.
    monitor.append_reply(entry_id, "x" * m._MAX_REPLY_CHARS)
    assert not monitor.snapshot()[0]["reply"].endswith("...")
    # One more chunk must record the truncation, not silently freeze.
    monitor.append_reply(entry_id, "y")
    reply = monitor.snapshot()[0]["reply"]
    assert len(reply) == m._MAX_REPLY_CHARS and reply.endswith("...")
