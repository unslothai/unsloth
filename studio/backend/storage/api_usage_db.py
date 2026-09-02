# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, content-free receipts for authenticated external API usage."""

from __future__ import annotations

import logging
import hashlib
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from storage.studio_db import get_connection, is_sqlite_busy_error


# Kept aligned with the API monitor's defensive upper bound; the storage layer validates
# independently because callers can invoke it directly.
MAX_TOKEN_COUNT = 1 << 40
MAX_RECEIPT_ID_CHARS = 128
MAX_SUBJECT_CHARS = 256
MAX_ENDPOINT_CHARS = 512
MAX_MODEL_CHARS = 1024
MAX_STATUS_CHARS = 64

_WRITE_BUSY_TIMEOUT_SECONDS = 0.05
_WRITE_RETRIES = 20
_WORKER_BUSY_RETRY_SECONDS = 0.25
_WORKER_DRAIN_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


@dataclass(frozen = True, slots = True)
class ApiUsageReceipt:
    """Terminal scalar usage only. Prompts, replies and credentials never enter it."""

    id: str
    subject: str
    endpoint: str
    model: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: int
    kind: str = "request"
    via_api_key: bool = True


def _valid_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_TOKEN_COUNT


def _bounded_text(value: object, limit: int, *, truncate: bool) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] if truncate else None


def _canonical_text(value: object, limit: int) -> Optional[str]:
    """Bound an identity string without merging values with a shared prefix."""
    if not isinstance(value, str) or not value:
        return None
    needs_digest = len(value) > limit
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        # json accepts unpaired surrogates, but utf-8 storage and hashing do not.
        encoded = value.encode("utf-8", errors = "surrogatepass")
        value = value.encode("utf-8", errors = "backslashreplace").decode("utf-8")
        needs_digest = True
    if not needs_digest:
        return value
    digest = hashlib.blake2s(encoded, digest_size = 16).hexdigest()
    return f"{value[: limit - len(digest) - 1]}~{digest}"


def canonical_api_subject(subject: object) -> str:
    """Stable database/cache key for an authenticated subject."""
    return _canonical_text(subject, MAX_SUBJECT_CHARS) or ""


def canonical_api_model(model: object) -> str:
    """Stable bounded model key that keeps long shared prefixes distinct."""
    return _canonical_text(model or "default", MAX_MODEL_CHARS) or "default"


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    # One definition, in the module that owns the contended file.
    return is_sqlite_busy_error(exc)


def _sleep_after_busy(delay: float) -> None:
    time.sleep(delay)


def _insert_api_usage(receipt: ApiUsageReceipt) -> bool:
    receipt_id = _bounded_text(receipt.id, MAX_RECEIPT_ID_CHARS, truncate = False)
    subject = canonical_api_subject(receipt.subject)
    endpoint = _bounded_text(receipt.endpoint, MAX_ENDPOINT_CHARS, truncate = True)
    model = canonical_api_model(receipt.model)
    status = _bounded_text(receipt.status, MAX_STATUS_CHARS, truncate = True)
    if receipt_id is None or not subject or endpoint is None or not model or status is None:
        return False

    conn = get_connection(busy_timeout_seconds = _WRITE_BUSY_TIMEOUT_SECONDS)
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO api_usage_events
                (id, subject, endpoint, model, status,
                 prompt_tokens, completion_tokens, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                subject,
                endpoint,
                model,
                status,
                receipt.prompt_tokens,
                receipt.completion_tokens,
                receipt.total_tokens,
                receipt.created_at,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def record_api_usage(receipt: ApiUsageReceipt) -> bool:
    """Insert one external request receipt, returning whether a row was added.

    The monitor id is the idempotency key, so repeated completion notification
    cannot inflate profile totals. Invalid or zero-usage receipts are ignored.
    """
    if receipt.kind != "request" or receipt.via_api_key is not True:
        return False
    counts = (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens)
    if not all(_valid_token_count(value) for value in counts) or not any(counts):
        return False
    if (
        not isinstance(receipt.created_at, int)
        or isinstance(receipt.created_at, bool)
        or receipt.created_at <= 0
        or receipt.created_at > (1 << 63) - 1
    ):
        return False

    for attempt in range(_WRITE_RETRIES):
        try:
            inserted = _insert_api_usage(receipt)
            break
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt + 1 == _WRITE_RETRIES:
                raise
            # The worker is the only production writer of these receipts, so a short bounded backoff lets
            # unrelated transactions finish without holding up the streaming caller.
            _sleep_after_busy(min(0.01 * (2**attempt), _WORKER_BUSY_RETRY_SECONDS))

    if inserted:
        # Lazy import avoids making profile aggregation part of schema startup.
        from storage.profile_stats_db import invalidate_profile_stats_cache
        invalidate_profile_stats_cache()
    return inserted


_STOP = object()


class ApiUsageWriter:
    """One serialized background writer for terminal API usage receipts."""

    def __init__(self, sink: Callable[[ApiUsageReceipt], bool] = record_api_usage):
        self._sink = sink
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(
            target = self._run,
            name = "api-usage-writer",
            daemon = True,
        )
        self._state_lock = threading.Lock()
        self._stopped = False
        self._thread.start()

    def submit(self, receipt: ApiUsageReceipt) -> bool:
        """Enqueue without waiting for SQLite or running caller-controlled code."""
        with self._state_lock:
            if self._stopped:
                return False
            self._queue.put_nowait(receipt)
            return True

    def stop(self, timeout: float = _WORKER_DRAIN_TIMEOUT_SECONDS) -> bool:
        """Stop accepting receipts and wait boundedly for the queue to drain.

        Returns ``True`` once the daemon consumed the stop sentinel. On timeout,
        the daemon keeps retrying the already accepted head receipt and exits
        after it succeeds and drains the remaining queue.
        """
        with self._state_lock:
            if not self._stopped:
                self._stopped = True
                self._queue.put_nowait(_STOP)
        # Production calls this through asyncio.to_thread so even the bounded
        self._thread.join(timeout = max(0.0, timeout))
        drained = not self._thread.is_alive()
        if not drained:
            logger.warning(
                "api usage writer drain timed out after %.1f seconds; the daemon will keep "
                "retrying accepted receipts, which may be lost if the process exits before "
                "SQLite becomes writable",
                timeout,
            )
        return drained

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                busy_failures = 0
                while True:
                    try:
                        self._sink(item)  # type: ignore[arg-type]
                        break
                    except sqlite3.OperationalError as exc:
                        if not _is_busy_error(exc):
                            logger.warning("api usage receipt persistence failed", exc_info = True)
                            break
                        # record_api_usage already made its bounded fast retries: retain this accepted item at
                        # the head of
                        # the single writer until a long transaction releases SQLite, with the stop sentinel behind it.
                        busy_failures += 1
                        if busy_failures == 1 or busy_failures % 20 == 0:
                            logger.warning(
                                "api usage database remains busy; retaining receipt for retry"
                            )
                        _sleep_after_busy(_WORKER_BUSY_RETRY_SECONDS)
                    except Exception:  # noqa: BLE001 - usage accounting cannot break inference.
                        logger.warning("api usage receipt persistence failed", exc_info = True)
                        break
            finally:
                self._queue.task_done()


_writer_condition = threading.Condition()
_writer: Optional[ApiUsageWriter] = None
_writer_leases: set[str] = set()
_writer_stopping = False


def acquire_api_usage_writer() -> str:
    """Lease the process writer; overlapping app lifespans share one worker."""
    global _writer
    lease = uuid.uuid4().hex
    with _writer_condition:
        while _writer_stopping:
            _writer_condition.wait()
        if _writer is None:
            _writer = ApiUsageWriter()
        _writer_leases.add(lease)
    return lease


def enqueue_api_usage(receipt: ApiUsageReceipt) -> None:
    """Fast production monitor callback; it performs no database I/O."""
    with _writer_condition:
        if _writer is not None:
            _writer.submit(receipt)


def release_api_usage_writer(lease: str) -> None:
    """Release one lifespan and boundedly drain after the last owner exits.

    A timed-out daemon retains its accepted queue, but the global gate is always
    cleared so a successor lifespan can start a fresh writer.
    """
    global _writer, _writer_stopping
    writer = None
    with _writer_condition:
        _writer_leases.discard(lease)
        if not _writer_leases and _writer is not None and not _writer_stopping:
            writer = _writer
            _writer_stopping = True
    if writer is not None:
        try:
            writer.stop()
        finally:
            with _writer_condition:
                if _writer is writer:
                    _writer = None
                _writer_stopping = False
                _writer_condition.notify_all()
