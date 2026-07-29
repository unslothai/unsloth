# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Profile usage statistics derived from studio.db.

Read-only aggregation over rows the app already writes: chat threads/messages
(with their per-message ``metadata_json``) and training runs/metrics. Nothing
is recorded specifically for stats, so the numbers are only as complete as the
local history.

Token counts live inside each message's metadata blob, so they cannot be summed
in SQL portably (JSON1 is not guaranteed on every bundled SQLite). Rows are
streamed once in (thread, time) order and every metric is folded in that single
pass, then memoised against a (count, max created_at) fingerprint so reopening
the Profile tab is free until history changes.
"""

import json
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

from loggers import get_logger

from storage.studio_db import get_connection

logger = get_logger(__name__)

# Gaps longer than this end a "sitting at the keyboard" stretch: without the cap
# a thread reopened a week later would report a week-long chat.
SESSION_GAP_SECONDS = 30 * 60
# Cap on the daily activity series handed to the UI (the heatmap draws a year).
MAX_DAILY_DAYS = 366
# Top-N lists returned to the client.
TOP_MODELS = 8
RECENT_RUNS = 5
# Serve a memoised payload for this long even if the fingerprint is unchanged,
# so a chat that is mid-stream still refreshes reasonably promptly.
CACHE_TTL_SECONDS = 20.0

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"fingerprint": None, "expires_at": 0.0, "payload": None}


def _as_float(value: Any) -> Optional[float]:
    """Coerce JSON numbers defensively; metadata is written by the client."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return (
            float(value) if value == value and value not in (float("inf"), float("-inf")) else None
        )
    return None


def _as_int(value: Any) -> int:
    number = _as_float(value)
    if number is None or number < 0:
        return 0
    return int(number)


def _iso(day: date) -> str:
    return day.isoformat()


def _streaks(days: set[date], today: date) -> dict[str, Any]:
    """Current and longest run of consecutive active days.

    The current streak survives a day that has not been used yet: a streak that
    ended yesterday is still "live" until today is over.
    """
    if not days:
        return {"current": 0, "longest": 0, "lastActiveDay": None}

    ordered = sorted(days)
    longest = 1
    running = 1
    for previous, current in zip(ordered, ordered[1:]):
        running = running + 1 if current - previous == timedelta(days = 1) else 1
        longest = max(longest, running)

    last = ordered[-1]
    current_streak = 0
    if today - last <= timedelta(days = 1):
        current_streak = 1
        cursor = last
        while cursor - timedelta(days = 1) in days:
            cursor -= timedelta(days = 1)
            current_streak += 1

    return {"current": current_streak, "longest": longest, "lastActiveDay": _iso(last)}


def _model_label(model_id: str) -> str:
    """Last path segment of a repo id, e.g. ``unsloth/gpt-oss-20b`` -> ``gpt-oss-20b``."""
    cleaned = model_id.strip().replace("\\", "/")
    tail = cleaned.rstrip("/").split("/")[-1]
    return tail or cleaned


class _MessageFold:
    """Accumulators for the single streaming pass over chat messages."""

    def __init__(self) -> None:
        self.threads: set[str] = set()
        self.messages = 0
        self.user_messages = 0
        self.assistant_messages = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0
        self.tool_calls = 0
        self.attachments = 0
        self.session_seconds = 0.0
        self.longest_chat: dict[str, Any] = {
            "threadId": None,
            "title": None,
            "seconds": 0.0,
            "messages": 0,
        }
        self.by_day: dict[date, dict[str, Any]] = {}
        self.by_hour = [0] * 24
        self.by_weekday = [0] * 7
        self.models: dict[str, dict[str, Any]] = {}
        self.speed_samples: list[float] = []
        self.best_speed = 0.0
        self.best_speed_model: Optional[str] = None
        self.response_ms: list[float] = []
        self.first_token_ms: list[float] = []

    def note_model(self, model_id: str, tokens: int) -> None:
        entry = self.models.setdefault(
            model_id, {"id": model_id, "label": _model_label(model_id), "messages": 0, "tokens": 0}
        )
        entry["messages"] += 1
        entry["tokens"] += tokens

    def note_day(self, day: date, tokens: int, thread_id: str) -> None:
        bucket = self.by_day.setdefault(day, {"tokens": 0, "messages": 0, "threads": set()})
        bucket["tokens"] += tokens
        bucket["messages"] += 1
        bucket["threads"].add(thread_id)


def _fold_messages(conn) -> _MessageFold:
    fold = _MessageFold()
    rows = conn.execute(
        """
        SELECT m.thread_id, m.role, m.metadata_json, m.attachments_json, m.created_at,
               t.title, t.model_id, t.model_type
        FROM chat_messages m
        LEFT JOIN chat_threads t ON t.id = m.thread_id
        ORDER BY m.thread_id, m.created_at
        """
    )

    current_thread: Optional[str] = None
    thread_title: Optional[str] = None
    thread_seconds = 0.0
    thread_messages = 0
    previous_created: Optional[int] = None

    def close_thread() -> None:
        if current_thread is None:
            return
        fold.session_seconds += thread_seconds
        if thread_seconds > fold.longest_chat["seconds"]:
            fold.longest_chat = {
                "threadId": current_thread,
                "title": thread_title,
                "seconds": thread_seconds,
                "messages": thread_messages,
            }

    for row in rows:
        thread_id = row["thread_id"]
        created_at = _as_int(row["created_at"])
        if thread_id != current_thread:
            close_thread()
            current_thread = thread_id
            thread_title = row["title"]
            thread_seconds = 0.0
            thread_messages = 0
            previous_created = None

        fold.threads.add(thread_id)
        fold.messages += 1
        thread_messages += 1

        if previous_created is not None:
            gap = (created_at - previous_created) / 1000
            if 0 < gap <= SESSION_GAP_SECONDS:
                thread_seconds += gap
        previous_created = created_at

        stamp = datetime.fromtimestamp(created_at / 1000) if created_at > 0 else None
        role = row["role"]
        if role == "user":
            fold.user_messages += 1

        attachments_json = row["attachments_json"]
        if attachments_json:
            try:
                parsed = json.loads(attachments_json)
                if isinstance(parsed, list):
                    fold.attachments += len(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        message_tokens = 0
        metadata: Any = None
        if role == "assistant":
            fold.assistant_messages += 1
            raw_metadata = row["metadata_json"]
            if raw_metadata:
                try:
                    metadata = json.loads(raw_metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = None

        if isinstance(metadata, dict):
            usage = metadata.get("contextUsage")
            timing = metadata.get("timing")
            usage = usage if isinstance(usage, dict) else {}
            timing = timing if isinstance(timing, dict) else {}

            prompt_tokens = _as_int(usage.get("promptTokens"))
            completion_tokens = _as_int(usage.get("completionTokens"))
            total_tokens = _as_int(usage.get("totalTokens"))
            if completion_tokens == 0:
                # Local engines occasionally omit the usage chunk; the adapter's
                # own token count is the next best estimate.
                completion_tokens = _as_int(timing.get("tokenCount"))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            fold.prompt_tokens += prompt_tokens
            fold.completion_tokens += completion_tokens
            fold.total_tokens += total_tokens
            fold.cached_tokens += _as_int(usage.get("cachedTokens"))
            fold.tool_calls += _as_int(timing.get("toolCallCount"))
            message_tokens = total_tokens

            model_id = usage.get("modelId")
            if not isinstance(model_id, str) or not model_id.strip():
                model_id = row["model_id"] if isinstance(row["model_id"], str) else ""
            if model_id.strip():
                fold.note_model(model_id.strip(), message_tokens)

            speed = _as_float(timing.get("tokensPerSecond"))
            # llama.cpp reports absurd rates on no-op turns; ignore those.
            if speed is not None and 0 < speed < 100_000:
                fold.speed_samples.append(speed)
                if speed > fold.best_speed:
                    fold.best_speed = speed
                    fold.best_speed_model = _model_label(model_id) if model_id else None

            stream_ms = _as_float(timing.get("totalStreamTime"))
            if stream_ms is not None and stream_ms > 0:
                fold.response_ms.append(stream_ms)
            start_ms = _as_float(timing.get("streamStartTime"))
            first_token = _as_float(timing.get("firstTokenTime"))
            if start_ms and first_token and first_token > start_ms:
                fold.first_token_ms.append(first_token - start_ms)

        if stamp is not None:
            fold.by_hour[stamp.hour] += 1
            fold.by_weekday[stamp.weekday()] += 1
            fold.note_day(stamp.date(), message_tokens, thread_id)

    close_thread()
    return fold


def _daily_series(fold: _MessageFold, today: date, days: int) -> list[dict[str, Any]]:
    """Dense day-by-day series so the heatmap can index straight into it."""
    start = today - timedelta(days = days - 1)
    series: list[dict[str, Any]] = []
    for offset in range(days):
        day = start + timedelta(days = offset)
        bucket = fold.by_day.get(day)
        series.append(
            {
                "date": _iso(day),
                "tokens": int(bucket["tokens"]) if bucket else 0,
                "messages": int(bucket["messages"]) if bucket else 0,
                "chats": len(bucket["threads"]) if bucket else 0,
            }
        )
    return series


def _training_stats(conn) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS runs,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
               SUM(COALESCE(final_step, 0)) AS steps,
               SUM(COALESCE(duration_seconds, 0)) AS seconds,
               COUNT(DISTINCT model_name) AS models,
               COUNT(DISTINCT dataset_name) AS datasets,
               MIN(final_loss) AS best_loss
        FROM training_runs
        """
    ).fetchone()

    tokens = conn.execute("SELECT COALESCE(SUM(num_tokens), 0) FROM training_metrics").fetchone()[0]

    recent = conn.execute(
        """
        SELECT id, COALESCE(display_name, model_name) AS name, model_name, dataset_name,
               status, final_loss, final_step, duration_seconds, started_at
        FROM training_runs
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (RECENT_RUNS,),
    ).fetchall()

    return {
        "runs": _as_int(row["runs"]),
        "completed": _as_int(row["completed"]),
        "steps": _as_int(row["steps"]),
        "tokens": _as_int(tokens),
        "seconds": _as_float(row["seconds"]) or 0.0,
        "models": _as_int(row["models"]),
        "datasets": _as_int(row["datasets"]),
        "bestLoss": _as_float(row["best_loss"]),
        "recent": [
            {
                "id": item["id"],
                "name": item["name"],
                "modelLabel": _model_label(item["model_name"] or ""),
                "datasetLabel": _model_label(item["dataset_name"] or ""),
                "status": item["status"],
                "finalLoss": _as_float(item["final_loss"]),
                "steps": _as_int(item["final_step"]),
                "seconds": _as_float(item["duration_seconds"]) or 0.0,
                "startedAt": item["started_at"],
            }
            for item in recent
        ],
    }


def _fingerprint(conn) -> tuple:
    message_row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(created_at), 0) FROM chat_messages"
    ).fetchone()
    run_row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(started_at), '') FROM training_runs"
    ).fetchone()
    return (message_row[0], message_row[1], run_row[0], run_row[1])


def compute_profile_stats(days: int = MAX_DAILY_DAYS) -> dict[str, Any]:
    """Aggregate every profile statistic in one pass, memoised per history state."""
    days = max(1, min(int(days), MAX_DAILY_DAYS))
    conn = get_connection()
    try:
        fingerprint = (_fingerprint(conn), days)
        now = time.monotonic()
        with _cache_lock:
            if (
                _cache["payload"] is not None
                and _cache["fingerprint"] == fingerprint
                and _cache["expires_at"] > now
            ):
                return _cache["payload"]

        started = time.perf_counter()
        fold = _fold_messages(conn)
        training = _training_stats(conn)

        today = date.today()
        streak = _streaks(set(fold.by_day.keys()), today)
        daily = _daily_series(fold, today, days)

        peak_day = max(fold.by_day.items(), key = lambda item: item[1]["tokens"], default = None)
        models = sorted(
            fold.models.values(), key = lambda item: (item["tokens"], item["messages"]), reverse = True
        )[:TOP_MODELS]

        speed_samples = fold.speed_samples
        payload = {
            "generatedAt": int(time.time() * 1000),
            "days": days,
            "totals": {
                "threads": len(fold.threads),
                "messages": fold.messages,
                "userMessages": fold.user_messages,
                "assistantMessages": fold.assistant_messages,
                "promptTokens": fold.prompt_tokens,
                "completionTokens": fold.completion_tokens,
                "totalTokens": fold.total_tokens,
                "cachedTokens": fold.cached_tokens,
                "toolCalls": fold.tool_calls,
                "attachments": fold.attachments,
                "activeDays": len(fold.by_day),
                "chatSeconds": round(fold.session_seconds),
            },
            "streak": streak,
            "peakDay": (
                {"date": _iso(peak_day[0]), "tokens": int(peak_day[1]["tokens"])}
                if peak_day and peak_day[1]["tokens"] > 0
                else None
            ),
            "longestChat": (
                {
                    "threadId": fold.longest_chat["threadId"],
                    "title": fold.longest_chat["title"],
                    "seconds": round(fold.longest_chat["seconds"]),
                    "messages": fold.longest_chat["messages"],
                }
                if fold.longest_chat["seconds"] > 0
                else None
            ),
            "daily": daily,
            "hourly": fold.by_hour,
            "weekday": fold.by_weekday,
            "models": models,
            "speed": {
                "averageTokensPerSecond": (
                    sum(speed_samples) / len(speed_samples) if speed_samples else None
                ),
                "bestTokensPerSecond": fold.best_speed or None,
                "bestTokensPerSecondModel": fold.best_speed_model,
                "averageResponseMs": (
                    sum(fold.response_ms) / len(fold.response_ms) if fold.response_ms else None
                ),
                "averageFirstTokenMs": (
                    sum(fold.first_token_ms) / len(fold.first_token_ms)
                    if fold.first_token_ms
                    else None
                ),
                "samples": len(speed_samples),
            },
            "training": training,
        }

        logger.debug(
            "profile stats computed in %.1f ms (%d messages)",
            (time.perf_counter() - started) * 1000,
            fold.messages,
        )

        with _cache_lock:
            _cache["fingerprint"] = fingerprint
            _cache["expires_at"] = time.monotonic() + CACHE_TTL_SECONDS
            _cache["payload"] = payload
        return payload
    finally:
        conn.close()


def invalidate_profile_stats_cache() -> None:
    """Drop the memoised payload (used by tests and after history wipes)."""
    with _cache_lock:
        _cache["fingerprint"] = None
        _cache["expires_at"] = 0.0
        _cache["payload"] = None
