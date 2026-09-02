# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Profile usage statistics derived from studio.db.

Read-only aggregation over chat threads/messages (with their per-message
``metadata_json``), content-free API usage receipts, and training runs/metrics.
The numbers are only as complete as the local history retained after each
feature was introduced.

Chat and training tables predate authenticated subjects and therefore remain
install-wide. API usage receipts are filtered to the requesting subject.

Token counts live inside each message's metadata blob, so they cannot be summed
in SQL portably (JSON1 is not guaranteed on every bundled SQLite). Rows are
streamed once in (thread, time) order and every metric is folded in that single
pass, then memoised against per-source count/timestamp fingerprints so reopening
the Profile tab is free until history changes.
"""

import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Optional

from loggers import get_logger

from storage.api_usage_db import canonical_api_subject
from storage.studio_db import count_chat_message_attachments, get_connection

logger = get_logger(__name__)

# Gaps longer than this end a sitting-at-the-keyboard stretch: without the cap a thread reopened a
# week later would report a week-long chat.
SESSION_GAP_SECONDS = 30 * 60
# Cap on the daily activity series handed to the UI (the heatmap draws a year).
MAX_DAILY_DAYS = 366
# Widest real UTC offset is 14h; anything beyond that is a bad client value.
MAX_TZ_OFFSET_MINUTES = 14 * 60
# Top-N lists returned to the client.
TOP_MODELS = 8
RECENT_RUNS = 5
# Serve a memoised payload for this long even when the fingerprint is unchanged, so a chat that is
# mid-stream still refreshes promptly.
CACHE_TTL_SECONDS = 20.0

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"fingerprint": None, "expires_at": 0.0, "payload": None}


def _as_float(value: Any) -> Optional[float]:
    """Coerce JSON numbers defensively; metadata is written by the client.

    json accepts integers of any width, and float() raises OverflowError past
    ~1e308, so one oversized counter would 500 the whole panel.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None
        return number if number == number and number not in (float("inf"), float("-inf")) else None
    return None


def _as_int(value: Any) -> int:
    number = _as_float(value)
    if number is None or number < 0:
        return 0
    return int(number)


def _iso(day: date) -> str:
    return day.isoformat()


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _resolve_zone(tz_name: str, tz_offset_minutes: int):
    """The caller's zone, preferring an IANA name over a single offset.

    A fixed offset is only correct for the half of the year the caller happens
    to be in, so a winter message read during summer lands an hour out and can
    cross midnight. An IANA name carries each date's own offset. The offset
    stays as the fallback for callers that send no name, or hosts with no tzdata.
    """
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except (ValueError, ZoneInfoNotFoundError, OSError):
            logger.debug("unknown timezone %r, falling back to fixed offset", tz_name)
    return timezone(timedelta(minutes = -tz_offset_minutes))


def _local_stamp(created_at_ms: int, zone) -> Optional[datetime]:
    """Wall-clock time in the caller's timezone, not the server's.

    created_at is a client-supplied integer that SQLite stores unchecked, so a
    value outside datetime's range is possible. Drop that row rather than let
    one bad import take down the whole panel.
    """
    if created_at_ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(created_at_ms / 1000, tz = zone).replace(tzinfo = None)
    except (ValueError, OverflowError, OSError):
        return None


def _streaks(days: set[date], today: date) -> dict[str, Any]:
    """Current and longest run of consecutive active days.

    The current streak survives a day that has not been used yet: a streak that
    ended yesterday is still "live" until today is over.

    Imported history or a skewed client clock can date rows in the future.
    Those are dropped up front so they cannot pad the longest streak or be
    reported as the last active day either.
    """
    days = {day for day in days if day <= today}
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


class _ApiUsageFold:
    """Scalar-only API receipts folded separately from chat-only metrics."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.by_day: dict[date, int] = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.requests = 0


def _fold_api_usage(conn, zone, subject: str) -> _ApiUsageFold:
    fold = _ApiUsageFold()
    rows = conn.execute(
        """
        SELECT model, prompt_tokens, completion_tokens, total_tokens, created_at
        FROM api_usage_events
        WHERE subject = ?
        ORDER BY created_at
        """,
        (subject,),
    )
    for row in rows:
        prompt_tokens = _as_int(row["prompt_tokens"])
        completion_tokens = _as_int(row["completion_tokens"])
        total_tokens = _as_int(row["total_tokens"])
        fold.prompt_tokens += prompt_tokens
        fold.completion_tokens += completion_tokens
        # Preserve the provider's authoritative total even when it differs
        fold.total_tokens += total_tokens
        fold.requests += 1

        stamp = _local_stamp(_as_int(row["created_at"]), zone)
        if stamp is not None:
            day = stamp.date()
            fold.by_day[day] = fold.by_day.get(day, 0) + total_tokens

        model_id = _clean_str(row["model"])
        if model_id:
            model = fold.models.setdefault(
                model_id,
                {"id": model_id, "label": _model_label(model_id), "messages": 0, "tokens": 0},
            )
            # One terminal API request represents one model response in the combined leaderboard.
            model["messages"] += 1
            model["tokens"] += total_tokens
    return fold


def _merge_api_activity(chat: _MessageFold, api: _ApiUsageFold) -> None:
    """Merge only combined activity surfaces, leaving chat counters intact."""
    for day, tokens in api.by_day.items():
        bucket = chat.by_day.setdefault(day, {"tokens": 0, "messages": 0, "threads": set()})
        bucket["tokens"] += tokens
    for model_id, api_model in api.models.items():
        model = chat.models.setdefault(
            model_id,
            {"id": model_id, "label": api_model["label"], "messages": 0, "tokens": 0},
        )
        model["messages"] += api_model["messages"]
        model["tokens"] += api_model["tokens"]


def _fork_keepers(conn) -> dict[tuple[str, int, str], str]:
    """For each original message, the one clone elected to stand in for it.

    A clone is normally ignored because the original is counted instead. Once
    the original is gone, whether its thread was deleted or just that row was
    pruned, the clones become the only record. Letting every sibling count them
    would multiply the usage, so exactly one may.

    Electing per message rather than per fork matters because fork_chat_thread
    copies one parent_id branch, not the whole thread: sibling forks taken from
    a retry and a regeneration hold different rows, and a per-fork winner would
    silently drop whatever only the loser carries.
    """
    rows = conn.execute(
        """
        SELECT m.thread_id, m.created_at, m.role,
               t.forked_from_thread_id AS source_id
        FROM chat_messages m
        JOIN chat_threads t ON t.id = m.thread_id
        WHERE t.forked_from_thread_id IS NOT NULL
          AND m.created_at < t.created_at
        """
    )

    best: dict[tuple[str, int, str], str] = {}
    for row in rows:
        key = (row["source_id"], _as_int(row["created_at"]), row["role"])
        thread_id = row["thread_id"]
        current = best.get(key)
        # Any stable winner works; lowest id keeps the choice reproducible.
        if current is None or thread_id < current:
            best[key] = thread_id
    return best


def _surviving_original_keys(conn) -> set[tuple[str, int, str]]:
    """Identity of every message still living in a thread that has been forked.

    Clones get fresh ids, so there is nothing to join on. Within one thread the
    timestamp and role are enough to recognise the row a clone was taken from.
    """
    rows = conn.execute(
        """
        SELECT m.thread_id, m.created_at, m.role
        FROM chat_messages m
        WHERE m.thread_id IN (
            SELECT DISTINCT forked_from_thread_id FROM chat_threads
            WHERE forked_from_thread_id IS NOT NULL
        )
        """
    )
    return {(row["thread_id"], _as_int(row["created_at"]), row["role"]) for row in rows}


def _fold_messages(conn, zone) -> _MessageFold:
    fold = _MessageFold()
    keepers = _fork_keepers(conn)
    surviving = _surviving_original_keys(conn)
    rows = conn.execute(
        """
        SELECT m.thread_id, m.role, m.content_json, m.metadata_json,
               m.attachments_json, m.created_at,
               t.title, t.model_id, t.model_type, t.pair_id,
               t.created_at AS thread_created_at, t.forked_from_thread_id
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

        # Compare mode stores one thread per pane under a shared pair_id and the sidebar shows them as a
        # single conversation, so counting them twice inflates the chat total.
        conversation_id = row["pair_id"] or thread_id

        # A fork is its own visible conversation, so it counts towards the chat
        fold.threads.add(conversation_id)

        # Forking clones the whole ancestry keeping each copy's timestamp, so skip a clone while the row
        # it came from is still countable; once that original is gone the elected fork stands in.
        source_id = row["forked_from_thread_id"]
        if source_id and created_at < _as_int(row["thread_created_at"]):
            original = (source_id, created_at, row["role"])
            if original in surviving or keepers.get(original) != thread_id:
                continue

        fold.messages += 1
        thread_messages += 1

        if previous_created is not None:
            gap = (created_at - previous_created) / 1000
            if 0 < gap <= SESSION_GAP_SECONDS:
                thread_seconds += gap
        previous_created = created_at

        stamp = _local_stamp(created_at, zone)
        role = row["role"]
        if role == "user":
            fold.user_messages += 1

        fold.attachments += count_chat_message_attachments(
            row["attachments_json"],
            row["content_json"],
        )

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

            # llama.cpp reports its own counters under serverTimings when the provider sends no usage chunk, and
            # the response-details sheet already falls back to these.
            server = metadata.get("serverTimings")
            server = server if isinstance(server, dict) else {}

            prompt_tokens = _as_int(usage.get("promptTokens")) or _as_int(server.get("prompt_n"))
            completion_tokens = _as_int(usage.get("completionTokens")) or _as_int(
                server.get("predicted_n")
            )
            total_tokens = _as_int(usage.get("totalTokens"))
            if completion_tokens == 0:
                # Local engines occasionally omit the usage chunk; the adapter's
                completion_tokens = _as_int(timing.get("tokenCount"))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            fold.prompt_tokens += prompt_tokens
            fold.completion_tokens += completion_tokens
            fold.total_tokens += total_tokens
            fold.cached_tokens += _as_int(usage.get("cachedTokens")) or _as_int(
                server.get("cache_n")
            )
            fold.tool_calls += _as_int(timing.get("toolCallCount"))
            message_tokens = total_tokens

            # responseDetails carries the model that actually answered, which differs from the requested
            # checkpoint whenever a provider routes or resolves an alias.
            details = metadata.get("responseDetails")
            details = details if isinstance(details, dict) else {}
            model_id = _clean_str(details.get("responseModelId")) or _clean_str(
                usage.get("modelId")
            )
            if model_id:
                fold.note_model(model_id, message_tokens)

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
            # firstTokenTime is already an elapsed duration, not a timestamp.
            first_token = _as_float(timing.get("firstTokenTime"))
            if first_token is not None and first_token > 0:
                fold.first_token_ms.append(first_token)

        if stamp is not None:
            fold.note_day(stamp.date(), message_tokens, conversation_id)

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


def _superseded(prefix: str = "r.") -> str:
    """SQL for "a later run resumed from this one, so its counters live there".

    ``prefix`` must qualify the outer row: the EXISTS subquery selects from the
    same table, so a bare column name would bind to the subquery instead.

    ``create_run``'s resume claim sets ``resume_blocked`` and leaves
    ``output_dir`` alone. Cancelling clears ``output_dir`` while setting the
    same flag, so the flag alone cannot tell the two apart.

    ``delete_run`` never clears the flag, so the continuation has to still be
    there. Otherwise deleting it would strand the source at zero while its row
    and metrics stay visible in history.

    The continuation also has to have reached the source's step. ``create_run``
    claims the source the moment a resume starts, but ``final_step`` is only
    written on the first metric flush, so a continuation that fails before then
    would take the source's completed work down with it.

    ``resumed_from_run_id`` records the lineage outright. Runs written before
    that column existed fall back to matching ``output_dir``, which is weaker:
    cancelling a continuation nulls its ``output_dir`` and breaks the match.
    """
    return f"""
        {prefix}resume_blocked = 1
        AND EXISTS (
            SELECT 1 FROM training_runs continuation
            WHERE (
                continuation.resumed_from_run_id = {prefix}id
                OR (
                    continuation.resumed_from_run_id IS NULL
                    AND {prefix}output_dir IS NOT NULL
                    AND continuation.output_dir = {prefix}output_dir
                    AND continuation.started_at > {prefix}started_at
                )
            )
            AND continuation.id <> {prefix}id
            AND COALESCE(continuation.final_step, 0)
                >= COALESCE({prefix}final_step, 0)
        )
    """


def _training_stats(conn) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS runs,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
               SUM(COALESCE(duration_seconds, 0)) AS seconds,
               COUNT(DISTINCT model_name) AS models,
               COUNT(DISTINCT dataset_name) AS datasets,
               MIN(final_loss) AS best_loss
        FROM training_runs
        """
    ).fetchone()

    # A resumed run continues its source's counters, so only a run superseded by a resume is dropped:
    # create_run's claim sets resume_blocked while leaving output_dir intact, whereas cancelling
    # clears it, so a cancelled run keeps the work it did do.
    steps = conn.execute(
        f"SELECT COALESCE(SUM(r.final_step), 0) FROM training_runs r WHERE NOT ({_superseded()})"
    ).fetchone()[0]

    # num_tokens is state.num_input_tokens_seen, a running total logged at each step, so summing the
    # samples multiplies the real figure; take each run's final counter, the value get_run_metrics
    # reports.
    tokens = conn.execute(
        f"""
        SELECT COALESCE(SUM(run_tokens), 0) FROM (
            SELECT MAX(m.num_tokens) AS run_tokens
            FROM training_metrics m
            JOIN training_runs r ON r.id = m.run_id
            WHERE NOT ({_superseded("r.")})
            GROUP BY m.run_id
        )
        """
    ).fetchone()[0]

    recent = conn.execute(
        """
        SELECT id, display_name, model_name, dataset_name,
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
        "steps": _as_int(steps),
        "tokens": _as_int(tokens),
        "seconds": _as_float(row["seconds"]) or 0.0,
        "models": _as_int(row["models"]),
        "datasets": _as_int(row["datasets"]),
        "bestLoss": _as_float(row["best_loss"]),
        "recent": [
            {
                "id": item["id"],
                # A renamed run keeps the name the user gave it; otherwise fall
                "name": _clean_str(item["display_name"]) or _model_label(item["model_name"] or ""),
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


def _fingerprint(conn, subject: str) -> tuple:
    message_row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(created_at), 0) FROM chat_messages"
    ).fetchone()
    run_row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(started_at), '') FROM training_runs"
    ).fetchone()
    api_row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(created_at), 0)
        FROM api_usage_events
        WHERE subject = ?
        """,
        (subject,),
    ).fetchone()
    return (
        message_row[0],
        message_row[1],
        run_row[0],
        run_row[1],
        subject,
        api_row[0],
        api_row[1],
    )


def compute_profile_stats(
    days: int = MAX_DAILY_DAYS,
    tz_offset_minutes: int = 0,
    tz_name: str = "",
    *,
    subject: str = "",
) -> dict[str, Any]:
    """Aggregate profile statistics, subject-scoping only external API usage.

    Legacy Unsloth chat and training history is install-wide because those rows
    have no authenticated owner. An empty subject intentionally sees no API
    receipts, keeping non-route callers fail-closed.
    """
    days = max(1, min(int(days), MAX_DAILY_DAYS))
    tz_offset_minutes = max(
        -MAX_TZ_OFFSET_MINUTES, min(int(tz_offset_minutes), MAX_TZ_OFFSET_MINUTES)
    )
    zone = _resolve_zone(tz_name, tz_offset_minutes)
    subject = canonical_api_subject(subject)
    conn = get_connection()
    try:
        fingerprint = (_fingerprint(conn, subject), days, tz_offset_minutes, tz_name)
        now = time.monotonic()
        with _cache_lock:
            if (
                _cache["payload"] is not None
                and _cache["fingerprint"] == fingerprint
                and _cache["expires_at"] > now
            ):
                return _cache["payload"]

        started = time.perf_counter()
        fold = _fold_messages(conn, zone)
        api_fold = _fold_api_usage(conn, zone, subject)
        _merge_api_activity(fold, api_fold)
        training = _training_stats(conn)

        # "Today" has to match the buckets above, or the newest column and the current streak drift by a day
        # whenever the caller is elsewhere.
        today = (_local_stamp(int(time.time() * 1000), zone) or datetime.now()).date()
        streak = _streaks(set(fold.by_day.keys()), today)
        daily = _daily_series(fold, today, days)

        # The grid stops at today and the streaks ignore anything later, so a skewed client clock would
        # otherwise name a peak day that is nowhere in the chart.
        past_days = {day: bucket for day, bucket in fold.by_day.items() if day <= today}
        peak_day = max(past_days.items(), key = lambda item: item[1]["tokens"], default = None)
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
                "promptTokens": fold.prompt_tokens + api_fold.prompt_tokens,
                "completionTokens": fold.completion_tokens + api_fold.completion_tokens,
                "totalTokens": fold.total_tokens + api_fold.total_tokens,
                "chatPromptTokens": fold.prompt_tokens,
                "chatCompletionTokens": fold.completion_tokens,
                "chatTokens": fold.total_tokens,
                "apiPromptTokens": api_fold.prompt_tokens,
                "apiCompletionTokens": api_fold.completion_tokens,
                "apiTokens": api_fold.total_tokens,
                "cachedTokens": fold.cached_tokens,
                "toolCalls": fold.tool_calls,
                "attachments": fold.attachments,
                "activeDays": len(past_days),
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
            "profile stats computed in %.1f ms (%d messages, %d API requests)",
            (time.perf_counter() - started) * 1000,
            fold.messages,
            api_fold.requests,
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
