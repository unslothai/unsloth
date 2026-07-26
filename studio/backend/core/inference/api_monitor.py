# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small in-memory monitor for OpenAI-compatible API traffic."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


_MAX_ENTRIES = 50
_MAX_PROMPT_CHARS = 12000
_MAX_REPLY_CHARS = 12000
_PREVIEW_CHARS = 360


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
    subject: Optional[str] = None
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
    redact_reply: bool = False

    def snapshot(self, *, include_details: bool = True) -> dict[str, Any]:
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
        payload = {
            "id": self.id,
            "endpoint": self.endpoint,
            "method": self.method,
            "model": self.model,
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
        }
        if include_details:
            payload["prompt"] = self.prompt
            payload["reply"] = self.reply
        return payload


class ApiMonitor:
    def __init__(self, max_entries: int = _MAX_ENTRIES):
        self._entries: deque[ApiMonitorEntry] = deque()
        self._max_entries = max(0, max_entries)
        self._lock = threading.Lock()

    def start(
        self,
        *,
        endpoint: str,
        method: str,
        model: str,
        prompt: str,
        context_length: Optional[int] = None,
        subject: Optional[str] = None,
        redact_reply: bool = False,
    ) -> str:
        now = time.time()
        entry = ApiMonitorEntry(
            id = f"apireq_{uuid.uuid4().hex[:12]}",
            endpoint = endpoint,
            method = method,
            model = model or "default",
            prompt = _trim(prompt, _MAX_PROMPT_CHARS),
            status = "running",
            started_at = now,
            updated_at = now,
            subject = subject,
            started_monotonic = time.monotonic(),
            context_length = context_length,
            reply = "[memory capture]" if redact_reply else "",
            redact_reply = redact_reply,
        )
        with self._lock:
            self._entries.appendleft(entry)
            self._trim_terminal_locked()
        return entry.id

    def append_reply(self, entry_id: Optional[str], text: str) -> None:
        if not entry_id or not text:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.redact_reply:
                return
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

    def set_reply(self, entry_id: Optional[str], text: str) -> None:
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.redact_reply:
                return
            entry.reply = _trim(text, _MAX_REPLY_CHARS)
            entry.updated_at = time.time()

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
            now = time.time()
            entry.status = status
            entry.updated_at = now
            entry.finished_at = now
            entry.finished_monotonic = time.monotonic()
            self._entries.remove(entry)
            self._entries.appendleft(entry)
            self._trim_terminal_locked()

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
                entry.snapshot(include_details = include_details)
                for entry in self._entries
                if subject is None or entry.subject == subject
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
            if subject is not None and entry.subject != subject:
                return None
            return entry.snapshot(include_details = True)

    def active_count(self, *, subject: Optional[str] = None) -> int:
        with self._lock:
            return sum(
                1
                for entry in self._entries
                if entry.status == "running" and (subject is None or entry.subject == subject)
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

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


api_monitor = ApiMonitor()
