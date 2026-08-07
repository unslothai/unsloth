# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small in-memory monitor for OpenAI-compatible API traffic."""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


_MAX_ENTRIES = 50
_MAX_PROMPT_CHARS = 12000
_MAX_REPLY_CHARS = 12000
_PREVIEW_CHARS = 360
# Far above any real context window; larger means a broken upstream payload.
_MAX_TOKEN_COUNT = 1 << 40

# Opt-in startup kill switch for Studio's in-memory API monitor.
_DISABLE_ENV = "UNSLOTH_STUDIO_DISABLE_API_MONITOR"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _api_monitor_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES


def _token_count_or_none(value: Any) -> Optional[int]:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # snapshot() divides by these; an unbounded int makes that division raise too.
    if count < 0 or count > _MAX_TOKEN_COUNT:
        return None
    return count


def _finite_float_or_none(value: Any) -> Optional[float]:
    # float() on a huge upstream int raises OverflowError, not ValueError/TypeError.
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _trim(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    # Guard against limit < 3 (slice would underflow).
    if limit <= 3:
        return "..."[:limit]
    return text[: limit - 3] + "..."


@dataclass
class ApiMonitorEntry:
    id: str
    endpoint: str
    method: str
    model: str
    prompt: str
    status: str
    started_at: float
    updated_at: float
    # Who this row is attributed to; on a shared row it does not restrict visibility.
    subject: Optional[str] = None
    # True for sk-unsloth callers only: the panel auto-opens on these, not Studio's chat.
    via_api_key: bool = False
    # Monotonic anchors so duration math survives wall-clock steps (NTP).
    started_monotonic: float = 0.0
    finished_monotonic: Optional[float] = None
    reply: str = ""
    finished_at: Optional[float] = None
    context_length: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_tokens_authoritative: bool = False
    error: Optional[str] = None
    # "request" (HTTP call) or "lifecycle" (model load/unload: event/reason, not a prompt; shared).
    kind: str = "request"
    event: Optional[str] = None
    reason: Optional[str] = None
    shared: bool = False
    # 0-100 for a running download row; None when not applicable.
    progress: Optional[float] = None
    # Stamped on the first reply text; snapshot() prefers it over engine timings.
    first_token_monotonic: Optional[float] = None
    prompt_ms: Optional[float] = None
    tok_per_sec: Optional[float] = None
    stop_reason: Optional[str] = None
    # Every finish reason seen so far. An n > 1 stream reports each choice in its own
    # chunk, so agreement can only be judged across the whole request. Not serialized.
    stop_reasons_seen: set[str] = field(default_factory = set)

    def snapshot(
        self,
        *,
        include_details: bool = True,
        attributed: bool = True,
    ) -> dict[str, Any]:
        duration_ms = None
        if self.finished_monotonic is not None:
            duration_ms = max(
                0,
                int((self.finished_monotonic - self.started_monotonic) * 1000),
            )
        elif self.finished_at is not None:
            duration_ms = max(0, int((self.finished_at - self.started_at) * 1000))
        context_usage = None
        if self.total_tokens is not None and self.context_length:
            context_usage = min(1.0, max(0.0, self.total_tokens / self.context_length))
        ttft_ms = None
        if self.first_token_monotonic is not None:
            # Preferred over the prompt_ms fallback: that is prefill only, so it misses
            # the admission-queue wait before llama-server sees the request.
            ttft_ms = max(0, int((self.first_token_monotonic - self.started_monotonic) * 1000))
        elif self.prompt_ms is not None:
            ttft_ms = max(0, int(self.prompt_ms))
        tok_per_sec = self.tok_per_sec
        if (
            tok_per_sec is None
            and self.completion_tokens
            # The clock starts at the first token, so it spans only the gaps that
            # followed it: one token has no gap to measure, hence no rate at all.
            and self.completion_tokens > 1
            and self.finished_monotonic is not None
            and self.first_token_monotonic is not None
        ):
            gen_s = self.finished_monotonic - self.first_token_monotonic
            if gen_s > 0.05:
                tok_per_sec = (self.completion_tokens - 1) / gen_s
        payload = {
            "id": self.id,
            "endpoint": self.endpoint,
            "method": self.method,
            "model": self.model,
            # A shared row reaches subjects with nothing to do with it, and the overlay
            # auto-opens on this flag, so report it only to the attributed caller.
            "via_api_key": self.via_api_key and attributed,
            "prompt_preview": _trim(self.prompt, _PREVIEW_CHARS),
            "reply_preview": _trim(self.reply, _PREVIEW_CHARS),
            "prompt_truncated": len(self.prompt) > _PREVIEW_CHARS,
            "reply_truncated": len(self.reply) > _PREVIEW_CHARS,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "duration_ms": duration_ms,
            "context_length": self.context_length,
            "context_usage": context_usage,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "kind": self.kind,
            "event": self.event,
            "reason": self.reason,
            "progress": self.progress,
            "ttft_ms": ttft_ms,
            "tok_per_sec": round(tok_per_sec, 2) if tok_per_sec is not None else None,
            "stop_reason": self.stop_reason,
        }
        if include_details:
            payload["prompt"] = self.prompt
            payload["reply"] = self.reply
        return payload


class ApiMonitor:
    def __init__(
        self,
        max_entries: int = _MAX_ENTRIES,
        *,
        enabled: bool = True,
    ):
        self._entries: deque[ApiMonitorEntry] = deque()
        # Shared rows one subject cleared: deleting would erase another caller's history.
        self._hidden_shared: dict[str, set[str]] = {}
        self._max_entries = max(0, max_entries)
        self._lock = threading.Lock()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether rows are being recorded. Read-only: the kill switch is a
        startup env var, so nothing may flip it on a live monitor."""
        return self._enabled

    def start(
        self,
        *,
        endpoint: str,
        method: str,
        model: str,
        prompt: str,
        context_length: Optional[int] = None,
        subject: Optional[str] = None,
        via_api_key: bool = False,
    ) -> str:
        if not self._enabled:
            return ""
        now = time.time()
        entry = ApiMonitorEntry(
            id = f"apireq_{uuid.uuid4().hex[:12]}",
            endpoint = endpoint,
            method = method,
            # str(): a raw JSON body can carry any type, and a non-string breaks the UI.
            model = str(model) if model else "default",
            prompt = _trim(prompt, _MAX_PROMPT_CHARS),
            status = "running",
            started_at = now,
            updated_at = now,
            subject = subject,
            via_api_key = via_api_key,
            started_monotonic = time.monotonic(),
            context_length = context_length,
        )
        with self._lock:
            self._entries.appendleft(entry)
            self._trim_terminal_locked()
        return entry.id

    def record_lifecycle(
        self,
        *,
        event: str,
        model: str,
        reason: Optional[str] = None,
        running: bool = False,
        via_api_key: bool = False,
        subject: Optional[str] = None,
    ) -> str:
        """Record a model load/unload alongside the request traffic that caused it.

        ``running=True`` opens the row for the caller to close with :meth:`finish` /
        :meth:`fail`; an unload is terminal on arrival. Rows are shared (visible to
        every subject) and share the request retention budget.

        ``subject`` names the caller whose request drove this, which is what
        ``via_api_key`` is reported to; it does not narrow who sees the row. Pass it
        whenever ``via_api_key`` is set, or the attribution reaches nobody.
        """
        if not self._enabled:
            return ""
        now = time.time()
        entry = ApiMonitorEntry(
            id = f"apievt_{uuid.uuid4().hex[:12]}",
            endpoint = f"model.{event}",
            method = "",
            model = model or "default",
            prompt = "",
            status = "running" if running else "completed",
            started_at = now,
            updated_at = now,
            started_monotonic = time.monotonic(),
            finished_at = None if running else now,
            finished_monotonic = None if running else time.monotonic(),
            kind = "lifecycle",
            event = event,
            reason = reason,
            shared = True,
            # The overlay opens on API-key traffic only, and a refused switch never
            # reaches api_monitor.start, so this row is its whole trace: without the
            # attribution the monitor stayed shut on the failures it exists to surface.
            via_api_key = via_api_key,
            # Shared rows reach every subject, so attribution needs an owner.
            subject = subject,
        )
        with self._lock:
            self._entries.appendleft(entry)
            self._trim_terminal_locked()
        return entry.id

    def relabel(self, entry_id: Optional[str], model: str) -> None:
        """Rename an open lifecycle row once the load resolves its real id: up front
        the caller only has the load path, which may be an HF snapshot dir."""
        if not entry_id or not model:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is not None:
                entry.model = model
                entry.updated_at = time.time()

    def set_progress(self, entry_id: Optional[str], progress: Optional[float]) -> None:
        """Update an open download row's percentage (clamped to 0-100)."""
        if not entry_id or progress is None:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is not None and entry.status == "running":
                entry.progress = min(100.0, max(0.0, float(progress)))
                entry.updated_at = time.time()

    def discard(self, entry_id: Optional[str]) -> None:
        """Drop a row that turned out not to be an event (an already-satisfied load)."""
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is not None:
                self._entries.remove(entry)

    def append_reply(
        self,
        entry_id: Optional[str],
        text: str,
        *,
        stamp_first_token: bool = True,
    ) -> None:
        if not entry_id or not text:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            # Only a streaming delta stamps TTFT; a full-response append is end-to-end latency.
            if stamp_first_token and entry.first_token_monotonic is None:
                entry.first_token_monotonic = time.monotonic()
            # Preview is capped: once the "..." marker is present the head is
            # frozen, so skip the per-chunk re-concat (avoids O(n^2) on long
            # generations). A reply that landed exactly on the cap has no marker
            # yet, so let one more append record the truncation before freezing.
            if len(entry.reply) >= _MAX_REPLY_CHARS:
                if not entry.reply.endswith("..."):
                    entry.reply = _trim(entry.reply + text, _MAX_REPLY_CHARS)
                entry.updated_at = time.time()
                return
            entry.reply = _trim(entry.reply + text, _MAX_REPLY_CHARS)
            entry.updated_at = time.time()

    def mark_first_token(self, entry_id: Optional[str]) -> None:
        """Stamp TTFT for deltas with no reply text, e.g. reasoning tokens."""
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.first_token_monotonic is not None:
                return
            entry.first_token_monotonic = time.monotonic()

    def set_reply(self, entry_id: Optional[str], text: str) -> None:
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            entry.reply = _trim(text, _MAX_REPLY_CHARS)
            entry.updated_at = time.time()

    def set_perf(
        self,
        entry_id: Optional[str],
        *,
        tok_per_sec: Optional[float] = None,
        prompt_ms: Optional[float] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        if not entry_id:
            return
        # Coerce before locking: arbitrary payloads, and a raise here (this runs inside
        # streaming generators) would truncate the user's response.
        tok_per_sec = _finite_float_or_none(tok_per_sec)
        prompt_ms = _finite_float_or_none(prompt_ms)
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if tok_per_sec is not None:
                entry.tok_per_sec = tok_per_sec
            if prompt_ms is not None:
                entry.prompt_ms = prompt_ms
            if stop_reason is not None:
                entry.stop_reason = str(stop_reason)
            entry.updated_at = time.time()

    def note_stop_reason(self, entry_id: Optional[str], reason: Optional[str]) -> None:
        """Record one choice's finish reason, without publishing it yet.

        The row carries a single stop reason, so it only describes the request once every
        choice has reported one. An n > 1 stream finishes its choices in separate chunks,
        so publishing here would state a request-level verdict from the first one and
        retract it when a later choice disagrees. :meth:`finish` resolves it instead.
        """
        if not entry_id or not reason:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            entry.stop_reasons_seen.add(str(reason))
            entry.updated_at = time.time()

    @staticmethod
    def _settle_stop_reason_locked(entry: ApiMonitorEntry, completed: bool) -> None:
        """Fix the row's stop reason as it becomes terminal.

        Only a completed request has one: a cancelled or failed stream stopped for that
        reason, so any natural reason recorded on the way describes what it was doing
        rather than how it ended. Clearing here catches the direct ``set_perf`` writers
        too, which several streams reach before the cancellation is stamped.
        """
        if not completed:
            entry.stop_reason = None
            return
        seen = entry.stop_reasons_seen
        if seen:
            entry.stop_reason = next(iter(seen)) if len(seen) == 1 else None

    def set_usage(
        self,
        entry_id: Optional[str],
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> None:
        if not entry_id:
            return
        # Coerce first: snapshot() does math on these arbitrary payload values.
        prompt_tokens = _token_count_or_none(prompt_tokens)
        completion_tokens = _token_count_or_none(completion_tokens)
        total_tokens = _token_count_or_none(total_tokens)
        context_length = _token_count_or_none(context_length)
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if prompt_tokens is not None:
                entry.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                entry.completion_tokens = completion_tokens
            if total_tokens is not None:
                entry.total_tokens = total_tokens
                entry.total_tokens_authoritative = True
            elif not entry.total_tokens_authoritative and (
                prompt_tokens is not None or completion_tokens is not None
            ):
                # Derive only when no authoritative total has been set;
                # a later partial chunk must not clobber a provider total.
                entry.total_tokens = (entry.prompt_tokens or 0) + (entry.completion_tokens or 0)
            if context_length is not None:
                entry.context_length = context_length
            entry.updated_at = time.time()

    def finish(
        self,
        entry_id: Optional[str],
        status: str = "completed",
    ) -> None:
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            # Idempotent: second call (e.g. [DONE] after the finally block
            # already ran) must not move finished_*.
            if entry.finished_at is not None:
                return
            self._settle_stop_reason_locked(entry, status == "completed")
            now = time.time()
            entry.status = status
            entry.updated_at = now
            entry.finished_at = now
            entry.finished_monotonic = time.monotonic()
            self._entries.remove(entry)
            self._entries.appendleft(entry)
            self._trim_terminal_locked()

    def fail_open(self, entry_id: Optional[str], error: str) -> None:
        """Fail only a still-open row: unlike :meth:`fail`, a catch-all in a
        ``finally`` cannot stamp an error onto a request that already succeeded."""
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
                return
            # Same lock as the check, so a finish() cannot land in between.
            self._fail_locked(entry, error)

    def fail(self, entry_id: Optional[str], error: str) -> None:
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if entry.finished_at is not None:
                # Already terminal; refresh error text only.
                if error:
                    entry.error = _trim(error, 1000)
                return
            self._fail_locked(entry, error)

    def _fail_locked(self, entry: ApiMonitorEntry, error: str) -> None:
        self._settle_stop_reason_locked(entry, False)
        now = time.time()
        entry.status = "error"
        entry.error = _trim(error, 1000)
        entry.updated_at = now
        entry.finished_at = now
        entry.finished_monotonic = time.monotonic()
        self._entries.remove(entry)
        self._entries.appendleft(entry)
        self._trim_terminal_locked()

    def snapshot(
        self,
        *,
        include_details: bool = True,
        subject: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return [
                entry.snapshot(
                    include_details = include_details,
                    attributed = self._attributed(entry, subject),
                )
                for entry in self._entries
                if self._visible(entry, subject)
            ]

    def get(
        self,
        entry_id: str,
        *,
        subject: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return None
            if not self._visible(entry, subject):
                return None
            return entry.snapshot(
                include_details = True,
                attributed = self._attributed(entry, subject),
            )

    def active_count(self, *, subject: Optional[str] = None) -> int:
        # Lifecycle rows show as "running" while loading but are not in-flight API requests.
        with self._lock:
            return sum(
                1
                for entry in self._entries
                if entry.status == "running"
                and entry.kind != "lifecycle"
                and (subject is None or entry.subject == subject)
            )

    def clear(self, *, subject: Optional[str] = None) -> None:
        """Drop recorded entries. ``subject`` limits the wipe to one caller's.

        Every other read on this class is subject-scoped, so an unscoped clear
        would let one user erase another's history (and zero their active count
        mid-generation). Callers that genuinely mean "everything" pass None.
        """
        with self._lock:
            if subject is None:
                self._entries.clear()
                self._hidden_shared.clear()
                return
            # A running shared row is a load in progress, not history, so it stays.
            hidden = self._hidden_shared.setdefault(subject, set())
            for entry in self._entries:
                if entry.shared and entry.status != "running":
                    hidden.add(entry.id)
            # Shared rows are hidden, never dropped, even when owned: they are another
            # caller's history too. An own running row is not history either, and dropping
            # it loses the request outright: active_count falls to zero and the finish or
            # fail that follows has no entry left to land on.
            self._entries = deque(
                entry
                for entry in self._entries
                if entry.shared or entry.subject != subject or entry.status == "running"
            )

    def _visible(self, entry: ApiMonitorEntry, subject: Optional[str]) -> bool:
        if subject is None:
            return True
        if entry.shared:
            # Every subject minus the cleared ones. Before ownership, so a clear hides own rows.
            return entry.id not in self._hidden_shared.get(subject, ())
        return entry.subject == subject

    def _attributed(self, entry: ApiMonitorEntry, subject: Optional[str]) -> bool:
        """Whether *subject* is the caller this row's API traffic belongs to.

        Only they should have the overlay pop open for it. An unscoped read (no
        subject: internal callers and tests) sees the row's own flag.
        """
        if subject is None:
            return True
        return entry.subject == subject

    def _find_locked(self, entry_id: str) -> Optional[ApiMonitorEntry]:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def _trim_terminal_locked(self) -> None:
        terminal_seen = 0
        kept: deque[ApiMonitorEntry] = deque()
        for entry in self._entries:
            if entry.status == "running":
                kept.append(entry)
                continue
            if terminal_seen < self._max_entries:
                kept.append(entry)
                terminal_seen += 1
        self._entries = kept
        # Keep hidden sets to live rows so they stay bounded by the ring buffer.
        live = {entry.id for entry in kept}
        for subject, hidden in list(self._hidden_shared.items()):
            hidden &= live
            if not hidden:
                del self._hidden_shared[subject]


api_monitor = ApiMonitor(enabled = not _api_monitor_disabled())
