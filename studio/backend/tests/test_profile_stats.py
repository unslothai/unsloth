# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Profile statistics aggregation over local chat/training history."""

import json
import time
from datetime import datetime, timedelta, timezone

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


# _seed_thread writes the assistant reply this far after the user turn, and
# fork_chat_thread copies created_at verbatim, so clones must reuse it.
REPLY_DELAY = timedelta(seconds = 10)


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
                _ms(when + REPLY_DELAY),
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
            # The adapter writes streamStartTime as an epoch stamp and
            # firstTokenTime as the elapsed ms before the first chunk.
            "streamStartTime": 1_760_000_000_000,
            "firstTokenTime": 200,
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
    # No modelId on the turn, so it is not credited to any model. The thread's
    # model_id follows the current selection and would misattribute after a
    # mid-conversation switch.
    assert stats["models"] == []


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
        # num_tokens is a running total, so the last row is the run's figure.
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, loss, num_tokens) VALUES (?, ?, ?, ?)",
            [("r1", step, 1.0, (step + 1) * 1000) for step in range(10)],
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


def test_first_token_time_is_read_as_a_duration(stats_db):
    """firstTokenTime is `Date.now() - streamStartTime`, not a wall-clock stamp.

    Treating it as a stamp and subtracting streamStartTime made the comparison
    fail for every real message, so the average was always empty.
    """
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "tft", "m", [(now, _metadata(10, 10))])
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    assert stats["speed"]["averageFirstTokenMs"] == pytest.approx(200.0)


def test_forked_threads_do_not_double_count_copied_history(stats_db):
    """Forking clones the ancestry, so the copies must not be counted again."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "src", "m", [(now - timedelta(hours = 2), _metadata(100, 50))])
        conn.commit()
    finally:
        conn.close()

    before = compute_profile_stats(days = 7)
    assert before["totals"]["totalTokens"] == 150
    assert before["totals"]["messages"] == 2

    fork_at = now
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
            "updated_at, forked_from_thread_id, forked_from_message_id) "
            "VALUES ('fork', 'fork of src', 'base', 'm', ?, ?, 'src', 'src-a0')",
            (_ms(fork_at), _ms(fork_at)),
        )
        # The clone keeps the original timestamp, exactly as fork_chat_thread does.
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('fork-a0', 'fork', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(100, 50)), _ms(now - timedelta(hours = 2) + REPLY_DELAY)),
        )
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    after = compute_profile_stats(days = 7)

    assert after["totals"]["totalTokens"] == 150
    assert after["totals"]["messages"] == 2

    # A genuinely new turn in the fork still counts.
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('fork-a1', 'fork', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(10, 5)), _ms(fork_at + timedelta(minutes = 1))),
        )
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    grown = compute_profile_stats(days = 7)
    assert grown["totals"]["totalTokens"] == 165
    assert grown["totals"]["messages"] == 3


def test_resumed_runs_do_not_double_count_steps_or_tokens(stats_db):
    """A resume continues the source's counters, so only the tail is counted."""
    conn = studio_db.get_connection()
    try:
        # 'stopped' at step 10, then claimed by the resume below. The claim sets
        # resume_blocked and leaves output_dir, which is how it is told apart
        # from a cancelled run.
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('src', 'stopped', 'm', 'd', '{}', '2026-01-01T10:00:00', 20, 10, 600, "
            "'/runs/out', 1)",
        )
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('cont', 'completed', 'm', 'd', '{}', '2026-01-02T10:00:00', 20, 15, 300, "
            "'/runs/out', 0)",
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, num_tokens) VALUES (?, ?, ?)",
            # The continuation's counter picks up where the source stopped.
            [("src", step, step * 100) for step in range(1, 11)]
            + [("cont", step, step * 100) for step in range(11, 16)],
        )
        conn.commit()
    finally:
        conn.close()

    training = compute_profile_stats(days = 7)["training"]

    # Training reached step 15, not 10 + 15.
    assert training["steps"] == 15
    assert training["tokens"] == 1500
    # Both attempts still show up as runs.
    assert training["runs"] == 2


def test_cancelled_runs_keep_the_work_they_did(stats_db):
    """Cancelling sets resume_blocked too, but nothing resumed from that run."""
    conn = studio_db.get_connection()
    try:
        # mark_run_cancel_requested clears output_dir and sets resume_blocked.
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('cancelled', 'stopped', 'm', 'd', '{}', '2026-01-01T10:00:00', 20, 8, "
            "400, NULL, 1)",
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, num_tokens) VALUES (?, ?, ?)",
            [("cancelled", step, step * 100) for step in range(1, 9)],
        )
        conn.commit()
    finally:
        conn.close()

    training = compute_profile_stats(days = 7)["training"]

    assert training["runs"] == 1
    assert training["steps"] == 8
    assert training["tokens"] == 800


def test_forks_count_as_chats_before_their_first_new_turn(stats_db):
    """A fork is a visible thread the moment it exists."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "orig", "m", [(now - timedelta(hours = 2), _metadata(100, 50))])
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
            "updated_at, forked_from_thread_id, forked_from_message_id) "
            "VALUES ('branch', 'fork', 'base', 'm', ?, ?, 'orig', 'orig-a0')",
            (_ms(now), _ms(now)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('branch-a0', 'branch', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(100, 50)), _ms(now - timedelta(hours = 2) + REPLY_DELAY)),
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    # Two conversations, but the cloned turn is not counted twice.
    assert stats["totals"]["threads"] == 2
    assert stats["totals"]["messages"] == 2
    assert stats["totals"]["totalTokens"] == 150


def test_tokens_follow_the_model_that_answered(stats_db):
    """A routed provider records the real producer in responseDetails."""
    now = datetime.now()
    metadata = _metadata(100, 50)
    metadata["contextUsage"]["modelId"] = "openrouter/auto"
    metadata["responseDetails"] = {"responseModelId": "anthropic/claude-sonnet-4"}
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "routed", "openrouter/auto", [(now, metadata)])
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    assert stats["models"][0]["id"] == "anthropic/claude-sonnet-4"
    assert stats["models"][0]["tokens"] == 150


def test_recent_run_name_prefers_the_users_rename(stats_db):
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, display_name) VALUES ('named', 'completed', 'unsloth/llama-3-8b', "
            "'tatsu-lab/alpaca', '{}', '2026-01-02T10:00:00', 'Support triage v3')",
        )
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at) VALUES ('plain', 'completed', 'unsloth/qwen3-4b', 'my/dataset', '{}', "
            "'2026-01-01T10:00:00')",
        )
        conn.commit()
    finally:
        conn.close()

    recent = {run["id"]: run for run in compute_profile_stats(days = 7)["training"]["recent"]}

    assert recent["named"]["name"] == "Support triage v3"
    assert recent["named"]["modelLabel"] == "llama-3-8b"
    # Unnamed runs fall back to the short label, not the full repo id.
    assert recent["plain"]["name"] == "qwen3-4b"


def test_historical_daylight_saving_offsets_are_respected(stats_db):
    """A fixed offset would put a winter message on the wrong day.

    2026-01-15 04:30 UTC is 23:30 on the 14th in New York, which is UTC-5 in
    January. Reusing a summer offset of UTC-4 pushes it to 00:30 on the 15th,
    so the one hour of drift crosses midnight and moves the activity grid.
    """
    winter = datetime(2026, 1, 15, 4, 30, tzinfo = timezone.utc)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('dst', 'dst', 'base', 'm', ?, ?)",
            (int(winter.timestamp() * 1000), int(winter.timestamp() * 1000)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('dst-a0', 'dst', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(10, 10)), int(winter.timestamp() * 1000)),
        )
        conn.commit()
    finally:
        conn.close()

    # A browser on summer time sends offset 240 (UTC-4) with the zone name.
    named = compute_profile_stats(days = 366, tz_offset_minutes = 240, tz_name = "America/New_York")
    invalidate_profile_stats_cache()
    offset_only = compute_profile_stats(days = 366, tz_offset_minutes = 240)

    assert {day["date"] for day in named["daily"] if day["messages"]} == {"2026-01-14"}

    # The fixed offset lands an hour late, which is what the zone name fixes.
    assert {day["date"] for day in offset_only["daily"] if day["messages"]} == {"2026-01-15"}


def test_unknown_timezone_falls_back_to_the_offset(stats_db):
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "tzbad", "m", [(now, _metadata(10, 10))])
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7, tz_offset_minutes = 0, tz_name = "Not/A/Zone")

    assert stats["totals"]["totalTokens"] == 20


def test_deleting_the_source_thread_keeps_the_forks_copies(stats_db):
    """forked_from_thread_id is not a foreign key, so it outlives the source."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "gone", "m", [(now - timedelta(hours = 2), _metadata(100, 50))])
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
            "updated_at, forked_from_thread_id, forked_from_message_id) "
            "VALUES ('kept', 'fork', 'base', 'm', ?, ?, 'gone', 'gone-a0')",
            (_ms(now), _ms(now)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('kept-a0', 'kept', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(100, 50)), _ms(now - timedelta(hours = 2) + REPLY_DELAY)),
        )
        conn.commit()
    finally:
        conn.close()

    assert compute_profile_stats(days = 7)["totals"]["totalTokens"] == 150

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_threads WHERE id = 'gone'")
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    stats = compute_profile_stats(days = 7)

    # The fork now holds the only copy, so it must still be counted once.
    assert stats["totals"]["totalTokens"] == 150
    assert stats["totals"]["messages"] == 1


def test_sibling_forks_do_not_multiply_a_deleted_source(stats_db):
    """Two forks of one thread must not both re-count the shared ancestry."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    older = now - timedelta(hours = 3)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "root", "m", [(older, _metadata(100, 50))])
        for fork_id in ("forkA", "forkB"):
            conn.execute(
                "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
                "updated_at, forked_from_thread_id, forked_from_message_id) "
                "VALUES (?, 'fork', 'base', 'm', ?, ?, 'root', 'root-a0')",
                (fork_id, _ms(now), _ms(now)),
            )
            conn.execute(
                "INSERT INTO chat_messages (id, thread_id, role, content_json, "
                "metadata_json, created_at) VALUES (?, ?, 'assistant', '[]', ?, ?)",
                (
                    f"{fork_id}-a0",
                    fork_id,
                    json.dumps(_metadata(100, 50)),
                    _ms(older + REPLY_DELAY),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    assert compute_profile_stats(days = 7)["totals"]["totalTokens"] == 150

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_threads WHERE id = 'root'")
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    stats = compute_profile_stats(days = 7)

    # Exactly one surviving copy is counted, not one per sibling fork.
    assert stats["totals"]["totalTokens"] == 150
    assert stats["totals"]["messages"] == 1


def test_deleting_an_original_message_keeps_the_forks_clone(stats_db):
    """Pruning one pre-fork message leaves the clone as the only copy."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    older = now - timedelta(hours = 3)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "orig", "m", [(older, _metadata(100, 50))])
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
            "updated_at, forked_from_thread_id, forked_from_message_id) "
            "VALUES ('branch', 'fork', 'base', 'm', ?, ?, 'orig', 'orig-a0')",
            (_ms(now), _ms(now)),
        )
        # The clone keeps the original's timestamp and role.
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('branch-a0', 'branch', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(100, 50)), _ms(older + timedelta(seconds = 10))),
        )
        conn.commit()
    finally:
        conn.close()

    # While the original is there the clone is ignored.
    assert compute_profile_stats(days = 7)["totals"]["totalTokens"] == 150

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_messages WHERE id = 'orig-a0'")
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    stats = compute_profile_stats(days = 7)

    # The thread survives but that message does not, so the clone stands in.
    assert stats["totals"]["totalTokens"] == 150
    assert stats["totals"]["assistantMessages"] == 1


def test_future_history_cannot_pad_the_longest_streak(stats_db):
    """Only the current streak was guarded; longest and lastActiveDay were not."""
    base = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(
            conn,
            "skew",
            "m",
            [(base + timedelta(days = day), _metadata(10, 10)) for day in (3, 4, 5, 6)],
        )
        conn.commit()
    finally:
        conn.close()

    streak = compute_profile_stats(days = 366)["streak"]

    assert streak == {"current": 0, "longest": 0, "lastActiveDay": None}


def test_comparison_panes_count_as_one_chat(stats_db):
    """Compare mode stores a thread per pane; the sidebar shows one chat."""
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        for pane in ("left", "right"):
            conn.execute(
                "INSERT INTO chat_threads (id, title, model_type, model_id, pair_id, "
                "created_at, updated_at) VALUES (?, 'compare', 'base', 'm', 'pair-1', ?, ?)",
                (pane, _ms(now), _ms(now)),
            )
            conn.execute(
                "INSERT INTO chat_messages (id, thread_id, role, content_json, "
                "metadata_json, created_at) VALUES (?, ?, 'assistant', '[]', ?, ?)",
                (f"{pane}-a0", pane, json.dumps(_metadata(100, 50)), _ms(now)),
            )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    assert stats["totals"]["threads"] == 1
    # Both panes still contribute their own messages and tokens.
    assert stats["totals"]["messages"] == 2
    assert stats["totals"]["totalTokens"] == 300


def test_deleting_a_continuation_restores_its_source(stats_db):
    """delete_run leaves resume_blocked set, so supersession needs a live tail."""
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, output_dir, resume_blocked) "
            "VALUES ('src', 'stopped', 'm', 'd', '{}', '2026-01-01T10:00:00', 20, 10, "
            "'/runs/out', 1)",
        )
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, output_dir, resume_blocked) "
            "VALUES ('cont', 'completed', 'm', 'd', '{}', '2026-01-02T10:00:00', 20, 15, "
            "'/runs/out', 0)",
        )
        conn.commit()
    finally:
        conn.close()

    assert compute_profile_stats(days = 7)["training"]["steps"] == 15

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM training_runs WHERE id = 'cont'")
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()

    # With the continuation gone the source is the only record of that work.
    assert compute_profile_stats(days = 7)["training"]["steps"] == 10


def test_out_of_range_timestamps_do_not_break_the_panel(stats_db):
    """created_at is client supplied and SQLite stores it unchecked."""
    now = datetime.now()
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "sane", "m", [(now, _metadata(10, 10))])
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('bad', 'bad', 'base', 'm', 1, 1)",
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('bad-a0', 'bad', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(7, 3)), 99_999_999_999_999_999),
        )
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 7)

    # Totals still include the bad row; only its day bucket is dropped, so the
    # activity grid holds just the one well-formed day.
    assert stats["totals"]["totalTokens"] == 30
    assert stats["totals"]["messages"] == 3
    assert stats["totals"]["activeDays"] == 1
    assert sum(day["messages"] for day in stats["daily"]) == 2


def test_future_dated_history_is_not_a_current_streak(stats_db):
    """A client clock that ran ahead must not report a streak that has not happened."""
    future = datetime.now() + timedelta(days = 5)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "ahead", "m", [(future, _metadata(10, 10))])
        conn.commit()
    finally:
        conn.close()

    streak = compute_profile_stats(days = 366)["streak"]

    assert streak["current"] == 0


def test_days_and_hours_use_the_callers_timezone(stats_db):
    """A remote browser must not be bucketed against the server's calendar."""
    # 01:30 UTC. In UTC that is one day; at UTC-4 it is 21:30 the day before.
    when = datetime(2026, 3, 10, 1, 30, tzinfo = timezone.utc)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('tz', 'tz', 'base', 'm', ?, ?)",
            (int(when.timestamp() * 1000), int(when.timestamp() * 1000)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('tz-a0', 'tz', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(10, 10)), int(when.timestamp() * 1000)),
        )
        conn.commit()
    finally:
        conn.close()

    at_utc = compute_profile_stats(days = 366, tz_offset_minutes = 0)
    invalidate_profile_stats_cache()
    at_minus_four = compute_profile_stats(days = 366, tz_offset_minutes = 240)

    utc_days = {day["date"] for day in at_utc["daily"] if day["messages"]}
    local_days = {day["date"] for day in at_minus_four["daily"] if day["messages"]}
    assert utc_days == {"2026-03-10"}
    assert local_days == {"2026-03-09"}


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

    def slow_compute(
        days = 366,
        tz_offset_minutes = 0,
        tz_name = "",
    ):
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
            await route_module.get_profile_stats(
                days = 366, tz_offset_minutes = 0, current_subject = "unsloth"
            )
        finally:
            beat.cancel()
        return ticks

    ticks = asyncio.run(drive())

    # ~50 ticks fit in 0.5s; a blocking call on the loop would yield 0.
    assert ticks > 10, f"event loop stalled during stats computation ({ticks} ticks)"


def test_an_oversized_token_counter_does_not_break_the_panel(stats_db):
    """json parses ints of any width; float() gives up long before that."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "sane", "m", [(now - timedelta(hours = 1), _metadata(100, 50))])
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('bad', 'Bad', 'base', 'm', ?, ?)",
            (_ms(now), _ms(now)),
        )
        huge = _metadata(100, 50)
        # Wider than float can hold, so every counter reads as unusable.
        oversized = int("9" * 309)
        for field in ("promptTokens", "completionTokens", "totalTokens"):
            huge["contextUsage"][field] = oversized
        huge["timing"]["tokenCount"] = oversized
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('bad-a0', 'bad', 'assistant', '[]', ?, ?)",
            (json.dumps(huge), _ms(now)),
        )
        conn.commit()
    finally:
        conn.close()

    # The row degrades to zero instead of raising, and the absurd counter
    # never reaches the totals; the healthy thread still reports.
    stats = compute_profile_stats(days = 7)
    assert stats["totals"]["totalTokens"] == 150
    assert stats["totals"]["messages"] == 3


def test_divergent_sibling_forks_keep_their_own_branch_messages(stats_db):
    """fork_chat_thread copies one parent_id branch, so siblings differ."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    older = now - timedelta(hours = 3)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "root", "m", [(older, _metadata(100, 50))])
        # A regeneration of the same turn, a second later.
        regenerated = older + REPLY_DELAY + timedelta(seconds = 1)
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('root-a1', 'root', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(200, 100)), _ms(regenerated)),
        )
        # One fork per branch: each carries only its own reply.
        for fork_id, stamp, meta in (
            ("forkA", older + REPLY_DELAY, _metadata(100, 50)),
            ("forkB", regenerated, _metadata(200, 100)),
        ):
            conn.execute(
                "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, "
                "updated_at, forked_from_thread_id, forked_from_message_id) "
                "VALUES (?, 'fork', 'base', 'm', ?, ?, 'root', 'root-a0')",
                (fork_id, _ms(now), _ms(now)),
            )
            conn.execute(
                "INSERT INTO chat_messages (id, thread_id, role, content_json, "
                "metadata_json, created_at) VALUES (?, ?, 'assistant', '[]', ?, ?)",
                (f"{fork_id}-a0", fork_id, json.dumps(meta), _ms(stamp)),
            )
        conn.commit()
    finally:
        conn.close()

    # Both originals counted, both clones suppressed.
    assert compute_profile_stats(days = 7)["totals"]["totalTokens"] == 450

    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_threads WHERE id = 'root'")
        conn.commit()
    finally:
        conn.close()

    invalidate_profile_stats_cache()
    stats = compute_profile_stats(days = 7)

    # Each branch survives in exactly one fork, so nothing is lost or doubled.
    assert stats["totals"]["totalTokens"] == 450
    assert stats["totals"]["messages"] == 2


def test_a_resume_that_never_logged_a_step_keeps_the_source_counters(stats_db):
    """create_run claims the source before the continuation flushes a metric."""
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('src', 'stopped', 'm', 'd', '{}', '2026-01-01T10:00:00', 20, 10, 600, "
            "'/runs/out', 1)",
        )
        # Errored before its first training step: no final_step, no metrics.
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('cont', 'error', 'm', 'd', '{}', '2026-01-02T10:00:00', 20, NULL, 5, "
            "'/runs/out', 0)",
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, num_tokens) VALUES (?, ?, ?)",
            [("src", step, step * 100) for step in range(1, 11)],
        )
        conn.commit()
    finally:
        conn.close()

    training = compute_profile_stats(days = 7)["training"]

    # The source's completed work is still the only work there is.
    assert training["steps"] == 10
    assert training["tokens"] == 1000


def test_a_future_dated_message_cannot_become_the_peak_day(stats_db):
    """The grid stops at today, so the headlines have to as well."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        _seed_thread(conn, "past", "m", [(now - timedelta(days = 1), _metadata(100, 50))])
        # A skewed client clock landed this one a week out.
        _seed_thread(conn, "ahead", "m", [(now + timedelta(days = 7), _metadata(900, 900))])
        conn.commit()
    finally:
        conn.close()

    stats = compute_profile_stats(days = 30)

    assert stats["peakDay"] is not None
    assert stats["peakDay"]["date"] == (now - timedelta(days = 1)).date().isoformat()
    # The future day is not an active day either.
    assert stats["totals"]["activeDays"] == 1


def test_cancelling_a_resumed_run_keeps_the_source_superseded(stats_db):
    """mark_run_cancel_requested nulls output_dir, so lineage carries it."""
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked) "
            "VALUES ('src', 'stopped', 'm', 'd', '{}', '2026-01-01T10:00:00', 20, 10, 600, "
            "'/runs/out', 1)",
        )
        # Resumed, then cancelled: output_dir cleared, counters still cumulative.
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, total_steps, final_step, duration_seconds, output_dir, resume_blocked, "
            "resumed_from_run_id) "
            "VALUES ('cont', 'stopped', 'm', 'd', '{}', '2026-01-02T10:00:00', 20, 15, 300, "
            "NULL, 1, 'src')",
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, num_tokens) VALUES (?, ?, ?)",
            [("src", step, step * 100) for step in range(1, 11)]
            + [("cont", step, step * 100) for step in range(11, 16)],
        )
        conn.commit()
    finally:
        conn.close()

    training = compute_profile_stats(days = 7)["training"]

    # 15, not 10 + 15: the cancelled continuation still carries the shared work.
    assert training["steps"] == 15
    assert training["tokens"] == 1500


def test_server_timings_stand_in_for_a_missing_usage_chunk(stats_db):
    """llama.cpp reports its own counters; the details sheet already uses them."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('local', 'Local', 'base', 'm', ?, ?)",
            (_ms(now), _ms(now)),
        )
        metadata = _metadata(0, 0)
        del metadata["contextUsage"]
        metadata["timing"]["tokenCount"] = 0
        metadata["serverTimings"] = {"prompt_n": 400, "predicted_n": 120, "cache_n": 90}
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('local-a0', 'local', 'assistant', '[]', ?, ?)",
            (json.dumps(metadata), _ms(now)),
        )
        conn.commit()
    finally:
        conn.close()

    totals = compute_profile_stats(days = 7)["totals"]

    assert totals["promptTokens"] == 400
    assert totals["completionTokens"] == 120
    assert totals["totalTokens"] == 520
    assert totals["cachedTokens"] == 90


def test_nested_fork_keeps_one_copy_after_the_middle_thread_is_deleted(stats_db):
    """Deleting a fork reconnects its descendants to the retained ancestor."""
    now = datetime.now().replace(hour = 12, minute = 0, second = 0, microsecond = 0)
    root_time = now - timedelta(hours = 4)
    middle_time = now - timedelta(hours = 2)
    middle_new_time = now - timedelta(hours = 1)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('root', 'root', 'base', 'm', ?, ?)",
            (_ms(root_time), _ms(root_time)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES ('root-a0', 'root', 'assistant', '[]', ?, ?)",
            (json.dumps(_metadata(100, 50)), _ms(root_time)),
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at, "
            "forked_from_thread_id, forked_from_message_id) "
            "VALUES ('middle', 'middle', 'base', 'm', ?, ?, 'root', 'root-a0')",
            (_ms(middle_time), _ms(middle_new_time)),
        )
        conn.executemany(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES (?, 'middle', 'assistant', '[]', ?, ?)",
            [
                ("middle-a0", json.dumps(_metadata(100, 50)), _ms(root_time)),
                ("middle-a1", json.dumps(_metadata(20, 10)), _ms(middle_new_time)),
            ],
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at, "
            "forked_from_thread_id, forked_from_message_id) "
            "VALUES ('leaf', 'leaf', 'base', 'm', ?, ?, 'middle', 'middle-a1')",
            (_ms(now), _ms(now)),
        )
        conn.executemany(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, metadata_json, "
            "created_at) VALUES (?, 'leaf', 'assistant', '[]', ?, ?)",
            [
                ("leaf-a0", json.dumps(_metadata(100, 50)), _ms(root_time)),
                ("leaf-a1", json.dumps(_metadata(20, 10)), _ms(middle_new_time)),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    before = compute_profile_stats(days = 7)["totals"]
    assert before["messages"] == 2
    assert before["totalTokens"] == 180

    studio_db.delete_chat_threads(["middle"])
    invalidate_profile_stats_cache()

    conn = studio_db.get_connection()
    try:
        leaf = conn.execute(
            "SELECT forked_from_thread_id FROM chat_threads WHERE id = 'leaf'"
        ).fetchone()
    finally:
        conn.close()

    assert leaf["forked_from_thread_id"] == "root"
    after = compute_profile_stats(days = 7)["totals"]
    assert after["messages"] == 2
    assert after["totalTokens"] == 180


def test_nested_resume_keeps_cumulative_totals_after_middle_run_deletion(stats_db):
    """Deleting a resumed run reconnects its continuation to the source."""
    conn = studio_db.get_connection()
    try:
        conn.executemany(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, "
            "started_at, final_step, output_dir, resume_blocked, resumed_from_run_id) "
            "VALUES (?, ?, 'm', 'd', '{}', ?, ?, ?, ?, ?)",
            [
                ("source", "stopped", "2026-01-01T10:00:00", 10, "/runs/out", 1, None),
                ("middle", "stopped", "2026-01-02T10:00:00", 15, "/runs/out", 1, "source"),
                ("tail", "completed", "2026-01-03T10:00:00", 20, "/runs/out", 0, "middle"),
            ],
        )
        conn.executemany(
            "INSERT INTO training_metrics (run_id, step, num_tokens) VALUES (?, ?, ?)",
            [("source", 10, 1000), ("middle", 15, 1500), ("tail", 20, 2000)],
        )
        conn.commit()
    finally:
        conn.close()

    before = compute_profile_stats(days = 7)["training"]
    assert before["steps"] == 20
    assert before["tokens"] == 2000

    studio_db.delete_run("middle")
    invalidate_profile_stats_cache()

    conn = studio_db.get_connection()
    try:
        tail = conn.execute(
            "SELECT resumed_from_run_id FROM training_runs WHERE id = 'tail'"
        ).fetchone()
    finally:
        conn.close()

    assert tail["resumed_from_run_id"] == "source"
    after = compute_profile_stats(days = 7)["training"]
    assert after["steps"] == 20
    assert after["tokens"] == 2000


def test_inline_media_counts_as_attachments_without_double_counting(stats_db):
    """Compare messages keep uploads in content_json instead of attachments_json."""
    now = datetime.now()
    image = {"type": "image", "image": "data:image/png;base64,AAAA"}
    audio = {"type": "audio", "audio": "data:audio/wav;base64,BBBB"}
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, model_id, created_at, updated_at) "
            "VALUES ('media', 'media', 'base', 'm', ?, ?)",
            (_ms(now), _ms(now)),
        )
        conn.execute(
            "INSERT INTO chat_messages (id, thread_id, role, content_json, attachments_json, "
            "created_at) VALUES ('media-u0', 'media', 'user', ?, ?, ?)",
            (
                json.dumps([image, audio, image]),
                json.dumps([{"id": "image-upload", "content": [image]}]),
                _ms(now),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    totals = compute_profile_stats(days = 7)["totals"]

    # The image appears in both storage representations and twice inline.
    assert totals["attachments"] == 2
