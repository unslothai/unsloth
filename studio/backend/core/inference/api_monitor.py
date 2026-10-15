# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-memory API monitor with an optional content-free terminal receipt sink."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.account_context import current_account_id
from storage.api_usage_db import (
    MAX_ENDPOINT_CHARS,
    MAX_STATUS_CHARS,
    ApiUsageReceipt,
    canonical_api_model,
    canonical_api_subject,
)


logger = logging.getLogger(__name__)
TerminalCallback = Callable[[ApiUsageReceipt], None]


_MAX_ENTRIES = 50
_MAX_PROMPT_BYTES = 64 * 1024 * 1024
_MAX_REPLY_CHARS = 12000
_PREVIEW_CHARS = 360
_MAX_STREAM_TOOL_CALLS = 64
_MAX_STREAM_TOOL_FIELD_CHARS = 501
_MAX_DECODE_MS = 24 * 60 * 60 * 1000
_MAX_TOKEN_COUNT = 1 << 40

_DISABLE_ENV = "UNSLOTH_STUDIO_DISABLE_API_MONITOR"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _api_monitor_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES


def _token_count_or_none(value: Any) -> Optional[int]:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if count < 0 or count > _MAX_TOKEN_COUNT:
        return None
    return count


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _advance_updated_at(entry: "ApiMonitorEntry", now: Optional[float] = None) -> None:
    """Advance the row's freshness key even on a coarse or regressing wall clock."""
    observed = time.time() if now is None else now
    entry.updated_at = max(observed, math.nextafter(entry.updated_at, math.inf))


def _trim(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "..."[:limit]
    return text[: limit - 3] + "..."


@dataclass
class _OpenAIStreamToolCall:
    choice_index: int
    tool_index: int
    call_id: Optional[str] = None
    name: str = ""
    arguments: str = ""


def _stream_tool_fragment(fragment: Any, *, serialize_json: bool = False) -> Optional[str]:
    if fragment is None:
        return None
    if isinstance(fragment, str):
        return fragment
    return json.dumps(fragment, default = str) if serialize_json else str(fragment)


def _append_stream_tool_field(
    current: str,
    fragment: Any,
    budget: int,
    *,
    serialize_json: bool = False,
) -> str:
    if fragment is None or budget <= 0 or len(current) >= _MAX_STREAM_TOOL_FIELD_CHARS:
        return current
    fragment = _stream_tool_fragment(fragment, serialize_json = serialize_json)
    if fragment is None:
        return current
    remaining = min(_MAX_STREAM_TOOL_FIELD_CHARS - len(current), budget)
    return current + fragment[:remaining]


def _merge_stream_tool_name(current: str, fragment: Any, budget: int) -> str:
    fragment = _stream_tool_fragment(fragment)
    if fragment is None:
        return current
    if fragment.startswith(current):
        limit = min(_MAX_STREAM_TOOL_FIELD_CHARS, len(current) + max(0, budget))
        return fragment[:limit]
    return _append_stream_tool_field(current, fragment, budget)


def _stream_tool_call_is_complete(state: _OpenAIStreamToolCall) -> bool:
    if not state.name or not state.arguments:
        return False
    try:
        arguments = json.loads(state.arguments)
    except (TypeError, ValueError):
        return False
    return isinstance(arguments, (dict, list))


def _stream_tool_call_preview(state: _OpenAIStreamToolCall) -> str:
    name = state.name or "<unknown>"
    if len(name) > 500:
        name = name[:497] + "..."
    if not state.arguments:
        return f"Tool call: {name}"
    arguments = state.arguments
    if len(arguments) > 500:
        arguments = arguments[:497] + "..."
    return f"Tool call: {name}({arguments})"


def _append_stream_tool_previews(
    entry: ApiMonitorEntry, states: list[_OpenAIStreamToolCall]
) -> None:
    if not states:
        return
    text = "\n".join(_stream_tool_call_preview(state) for state in states)
    if entry.reply and not entry.reply.endswith("\n"):
        text = "\n" + text
    entry.reply = _trim(entry.reply + text, _MAX_REPLY_CHARS)


def _stream_tool_call_id(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= _MAX_STREAM_TOOL_FIELD_CHARS:
        return value
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors = "replace")).hexdigest()


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
    # Usernames are reusable, so the immutable account id fences off a predecessor's traffic.
    account_id: str = field(default_factory = current_account_id)
    via_api_key: bool = False
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
    kind: str = "request"
    event: Optional[str] = None
    reason: Optional[str] = None
    shared: bool = False
    progress: Optional[float] = None
    running_phase: Optional[str] = None
    prompt_progress_total: Optional[int] = None
    prompt_progress_processed: Optional[int] = None
    prompt_progress_cached: Optional[int] = None
    prompt_progress_time_ms: Optional[float] = None
    first_token_monotonic: Optional[float] = None
    # Tool cards count for TTFT but not the token-rate clock.
    first_decode_monotonic: Optional[float] = None
    prompt_ms: Optional[float] = None
    tok_per_sec: Optional[float] = None
    prompt_tok_per_sec: Optional[float] = None
    decode_ms: Optional[float] = None
    stop_reason: Optional[str] = None
    # An n > 1 stream reports each choice in its own chunk. Not serialized.
    stop_reasons_seen: set[str] = field(default_factory = set)
    openai_stream_tool_calls: list[_OpenAIStreamToolCall] = field(default_factory = list)
    openai_stream_last_tool_indexes: dict[int, int] = field(default_factory = dict)
    openai_stream_last_segment_was_tool: bool = False
    prompt_complete: bool = True

    def snapshot(
        self,
        *,
        include_details: bool = True,
        include_prompt: bool = True,
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
            # Preferred over prompt_ms, which misses the admission-queue wait.
            ttft_ms = max(0, int((self.first_token_monotonic - self.started_monotonic) * 1000))
        elif self.prompt_ms is not None:
            ttft_ms = max(0, int(self.prompt_ms))
        tok_per_sec = self.tok_per_sec
        if (
            tok_per_sec is None
            and self.completion_tokens
            and self.completion_tokens > 1
            and self.finished_monotonic is not None
            and self.first_decode_monotonic is not None
        ):
            gen_s = self.finished_monotonic - self.first_decode_monotonic
            if gen_s > 0.05:
                tok_per_sec = (self.completion_tokens - 1) / gen_s
        prompt_progress = None
        if self.prompt_progress_processed is not None:
            percent = None
            if self.prompt_progress_total is not None and self.prompt_progress_total > 0:
                percent = min(
                    100.0,
                    max(
                        0.0,
                        self.prompt_progress_processed / self.prompt_progress_total * 100.0,
                    ),
                )
            prompt_progress = {
                "total": self.prompt_progress_total,
                "processed": self.prompt_progress_processed,
                "cached": self.prompt_progress_cached,
                "time_ms": (
                    round(self.prompt_progress_time_ms, 2)
                    if self.prompt_progress_time_ms is not None
                    else None
                ),
                "percent": round(percent, 1) if percent is not None else None,
            }
        payload = {
            "id": self.id,
            "endpoint": self.endpoint,
            "method": self.method,
            "model": self.model,
            "via_api_key": self.via_api_key and attributed,
            "prompt_preview": _trim(self.prompt, _PREVIEW_CHARS),
            "reply_preview": _trim(self.reply, _PREVIEW_CHARS),
            "prompt_truncated": not self.prompt_complete or len(self.prompt) > _PREVIEW_CHARS,
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
            "running_phase": self.running_phase,
            "prompt_progress": prompt_progress,
            "ttft_ms": ttft_ms,
            "tok_per_sec": round(tok_per_sec, 2) if tok_per_sec is not None else None,
            "prompt_tok_per_sec": (
                round(self.prompt_tok_per_sec, 2) if self.prompt_tok_per_sec is not None else None
            ),
            # Engine decode span, not the streamed window, which reasoning tokens inflate.
            "decode_ms": int(self.decode_ms) if self.decode_ms is not None else None,
            "stop_reason": self.stop_reason,
        }
        if include_details:
            if include_prompt and self.prompt_complete:
                payload["prompt"] = self.prompt
            payload["reply"] = self.reply
        return payload


class ApiMonitor:
    def __init__(
        self,
        max_entries: int = _MAX_ENTRIES,
        *,
        enabled: bool = True,
        terminal_callback: Optional[TerminalCallback] = None,
    ):
        self._entries: deque[ApiMonitorEntry] = deque()
        self._hidden_shared: dict[tuple[str, str], set[str]] = {}
        self._max_entries = max(0, max_entries)
        self._lock = threading.Lock()
        self._callback_condition = threading.Condition(self._lock)
        self._enabled = enabled
        self._terminal_callback = terminal_callback
        self._terminal_callback_leases: dict[str, TerminalCallback] = {}
        self._terminal_callbacks_inflight: dict[Optional[str], int] = {}

    @property
    def enabled(self) -> bool:
        """Whether rows are being recorded. Read-only: the kill switch is a
        startup env var, so nothing may flip it on a live monitor."""
        return self._enabled

    def set_terminal_callback(self, callback: Optional[TerminalCallback]) -> None:
        """Replace the unleased terminal sink used by tests and embedders."""
        with self._lock:
            self._terminal_callback = callback

    def acquire_terminal_callback(self, callback: TerminalCallback) -> str:
        """Only the newest live lease is notified, so overlapping lifespans do not write duplicate
        receipts."""
        lease = uuid.uuid4().hex
        with self._lock:
            self._terminal_callback_leases[lease] = callback
        return lease

    def release_terminal_callback(self, lease: str) -> None:
        """Remove only the terminal callback registration owned by ``lease``."""
        with self._callback_condition:
            self._terminal_callback_leases.pop(lease, None)
            while self._terminal_callbacks_inflight.get(lease, 0):
                self._callback_condition.wait()

    def _terminal_callback_locked(self) -> tuple[Optional[str], Optional[TerminalCallback]]:
        if self._terminal_callback_leases:
            lease = next(reversed(self._terminal_callback_leases))
            return lease, self._terminal_callback_leases[lease]
        return None, self._terminal_callback

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
            id = f"apireq_{uuid.uuid4().hex}",
            endpoint = endpoint,
            method = method,
            model = str(model) if model else "default",
            prompt = prompt or "",
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
            remaining = _MAX_PROMPT_BYTES
            for retained in self._entries:
                if not retained.prompt_complete or len(retained.prompt) <= _PREVIEW_CHARS:
                    continue
                prompt_bytes = sys.getsizeof(retained.prompt)
                if prompt_bytes <= remaining:
                    remaining -= prompt_bytes
                else:
                    retained.prompt = _trim(retained.prompt, _PREVIEW_CHARS)
                    retained.prompt_complete = False
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
        """Pass ``subject`` whenever ``via_api_key`` is set, or the attribution reaches nobody."""
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
            via_api_key = via_api_key,
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
                _advance_updated_at(entry)

    def set_progress(self, entry_id: Optional[str], progress: Optional[float]) -> None:
        """Update an open download row's percentage (clamped to 0-100)."""
        if not entry_id or progress is None:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is not None and entry.status == "running":
                entry.progress = min(100.0, max(0.0, float(progress)))
                _advance_updated_at(entry)

    def set_prompt_progress(
        self,
        entry_id: Optional[str],
        *,
        total: Any = None,
        processed: Any = None,
        cached: Any = None,
        time_ms: Any = None,
    ) -> None:
        if not entry_id:
            return
        total = _token_count_or_none(total)
        processed = _token_count_or_none(processed)
        cached = _token_count_or_none(cached)
        time_ms = _finite_float_or_none(time_ms)
        if processed is None:
            return
        if total is not None and total > 0:
            processed = min(processed, total)
        if cached is not None:
            cached = min(cached, processed)
        if time_ms is not None and time_ms < 0:
            time_ms = None
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.status != "running" or entry.kind != "request":
                return
            if entry.running_phase != "prompt_processing":
                entry.prompt_progress_total = None
                entry.prompt_progress_processed = None
                entry.prompt_progress_cached = None
                entry.prompt_progress_time_ms = None
            entry.running_phase = "prompt_processing"
            if total is not None:
                entry.prompt_progress_total = total
            entry.prompt_progress_processed = processed
            if cached is not None:
                entry.prompt_progress_cached = cached
            if time_ms is not None:
                entry.prompt_progress_time_ms = time_ms
            _advance_updated_at(entry)

    def set_running_phase(self, entry_id: Optional[str], phase: Optional[str]) -> None:
        if not entry_id or phase not in ("prompt_processing", "token_generation"):
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.status != "running" or entry.kind != "request":
                return
            entry.running_phase = phase
            _advance_updated_at(entry)

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
        separate_from_openai_tool: bool = False,
    ) -> None:
        if not entry_id or not text:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
                return
            if separate_from_openai_tool and entry.openai_stream_last_segment_was_tool:
                if entry.reply and not entry.reply.endswith("\n"):
                    text = "\n" + text
                entry.openai_stream_last_segment_was_tool = False
            if stamp_first_token:
                entry.running_phase = "token_generation"
                now = time.monotonic()
                if entry.first_token_monotonic is None:
                    entry.first_token_monotonic = now
                if entry.first_decode_monotonic is None:
                    entry.first_decode_monotonic = now
            # Once the "..." marker is present the head is frozen; skip the O(n^2) re-concat.
            if len(entry.reply) >= _MAX_REPLY_CHARS:
                if not entry.reply.endswith("..."):
                    entry.reply = _trim(entry.reply + text, _MAX_REPLY_CHARS)
                _advance_updated_at(entry)
                return
            entry.reply = _trim(entry.reply + text, _MAX_REPLY_CHARS)
            _advance_updated_at(entry)

    def accumulate_openai_tool_call(
        self,
        entry_id: Optional[str],
        *,
        choice_index: int,
        tool_index: Optional[int],
        tool_index_explicit: bool = True,
        call_id: Optional[str],
        name_fragment: Any,
        arguments_fragment: Any,
    ) -> None:
        """Merge one streamed OpenAI tool-call delta into its monitor row."""
        if not entry_id:
            return
        call_id = _stream_tool_call_id(call_id)
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
                return
            same_choice = [
                state
                for state in entry.openai_stream_tool_calls
                if state.choice_index == choice_index
            ]
            if tool_index is None:
                tool_index = entry.openai_stream_last_tool_indexes.get(choice_index, 0)
            state = None
            same_index = [
                candidate for candidate in same_choice if candidate.tool_index == tool_index
            ]
            if tool_index_explicit:
                state = next(
                    (
                        candidate
                        for candidate in reversed(same_index)
                        if candidate.call_id == call_id
                    ),
                    None,
                )
                if state is None and call_id is not None:
                    state = next(
                        (
                            candidate
                            for candidate in reversed(same_index)
                            if candidate.call_id is None
                        ),
                        None,
                    )
            elif call_id is not None:
                state = next(
                    (
                        candidate
                        for candidate in reversed(same_choice)
                        if candidate.call_id == call_id
                    ),
                    None,
                )
            if state is None and call_id is None:
                state = same_index[-1] if same_index else None
            if state is None:
                pending_chars = sum(
                    len(candidate.name) + len(candidate.arguments)
                    for candidate in entry.openai_stream_tool_calls
                )
                if (
                    len(entry.openai_stream_tool_calls) >= _MAX_STREAM_TOOL_CALLS
                    or pending_chars >= _MAX_REPLY_CHARS
                ):
                    return
                state = _OpenAIStreamToolCall(
                    choice_index = choice_index,
                    tool_index = tool_index,
                    call_id = call_id,
                )
                entry.openai_stream_tool_calls.append(state)
            elif call_id is not None and state.call_id is None:
                state.call_id = call_id
            entry.openai_stream_last_tool_indexes[choice_index] = tool_index
            pending_chars = sum(
                len(candidate.name) + len(candidate.arguments)
                for candidate in entry.openai_stream_tool_calls
            )
            budget = _MAX_REPLY_CHARS - pending_chars
            previous_name_length = len(state.name)
            state.name = _merge_stream_tool_name(state.name, name_fragment, budget)
            budget -= len(state.name) - previous_name_length
            state.arguments = _append_stream_tool_field(
                state.arguments,
                arguments_fragment,
                budget,
                serialize_json = True,
            )
            entry.running_phase = "token_generation"
            now = time.monotonic()
            if entry.first_token_monotonic is None:
                entry.first_token_monotonic = now
            if entry.first_decode_monotonic is None:
                entry.first_decode_monotonic = now
            _advance_updated_at(entry)

    def take_openai_tool_calls(
        self,
        entry_id: Optional[str],
        *,
        choice_index: Optional[int] = None,
        completed_only: bool = False,
    ) -> tuple[list[tuple[str, str]], bool]:
        """Remove and return completed preview calls in first-seen order."""
        if not entry_id:
            return [], False
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
                return [], False
            selected = [
                state
                for state in entry.openai_stream_tool_calls
                if (choice_index is None or state.choice_index == choice_index)
                and (not completed_only or _stream_tool_call_is_complete(state))
            ]
            if not selected:
                return [], False
            separate = bool(entry.reply) and not entry.reply.endswith("\n")
            entry.openai_stream_last_segment_was_tool = True
            selected_ids = {id(state) for state in selected}
            entry.openai_stream_tool_calls = [
                state for state in entry.openai_stream_tool_calls if id(state) not in selected_ids
            ]
            remaining_indexes = {
                state.choice_index: state.tool_index for state in entry.openai_stream_tool_calls
            }
            if choice_index is None:
                entry.openai_stream_last_tool_indexes = remaining_indexes
            elif choice_index not in remaining_indexes:
                entry.openai_stream_last_tool_indexes.pop(choice_index, None)
            else:
                entry.openai_stream_last_tool_indexes[choice_index] = remaining_indexes[
                    choice_index
                ]
            _advance_updated_at(entry)
            return [(state.name, state.arguments) for state in selected], separate

    def mark_first_token(
        self,
        entry_id: Optional[str],
        *,
        decoded: bool = True,
    ) -> None:
        """``decoded=False`` is for output the model did not generate: starts TTFT, not the token-
        rate clock."""
        if not entry_id:
            return
        now = time.monotonic()
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if entry.first_token_monotonic is None:
                entry.first_token_monotonic = now
            if decoded and entry.first_decode_monotonic is None:
                entry.first_decode_monotonic = now
            entry.running_phase = "token_generation" if decoded else None

    def set_reply(self, entry_id: Optional[str], text: str) -> None:
        if not entry_id:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            entry.reply = _trim(text, _MAX_REPLY_CHARS)
            _advance_updated_at(entry)

    def set_perf(
        self,
        entry_id: Optional[str],
        *,
        tok_per_sec: Optional[float] = None,
        prompt_tok_per_sec: Optional[float] = None,
        prompt_ms: Optional[float] = None,
        decode_ms: Optional[float] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        if not entry_id:
            return
        # Coerce before locking: a raise here would truncate the streamed response.
        tok_per_sec = _finite_float_or_none(tok_per_sec)
        prompt_tok_per_sec = _finite_float_or_none(prompt_tok_per_sec)
        prompt_ms = _finite_float_or_none(prompt_ms)
        decode_ms = _finite_float_or_none(decode_ms)
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if tok_per_sec is not None and tok_per_sec >= 0:
                entry.tok_per_sec = tok_per_sec
            if prompt_tok_per_sec is not None and prompt_tok_per_sec >= 0:
                entry.prompt_tok_per_sec = prompt_tok_per_sec
            if prompt_ms is not None:
                entry.prompt_ms = prompt_ms
            if decode_ms is not None and 0 <= decode_ms <= _MAX_DECODE_MS:
                entry.decode_ms = decode_ms
            if stop_reason is not None:
                entry.stop_reason = str(stop_reason)
            _advance_updated_at(entry)

    def note_stop_reason(self, entry_id: Optional[str], reason: Optional[str]) -> None:
        """Not published here: with n > 1 a later choice can disagree, so finish() resolves it."""
        if not entry_id or not reason:
            return
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            entry.stop_reasons_seen.add(str(reason))
            _advance_updated_at(entry)

    @staticmethod
    def _settle_stop_reason_locked(entry: ApiMonitorEntry, completed: bool) -> None:
        """Only a completed request keeps its stop reason; a cancelled or failed stream clears it."""
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
        prompt_tokens = _token_count_or_none(prompt_tokens)
        completion_tokens = _token_count_or_none(completion_tokens)
        total_tokens = _token_count_or_none(total_tokens)
        context_length = _token_count_or_none(context_length)
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
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
                entry.total_tokens = (entry.prompt_tokens or 0) + (entry.completion_tokens or 0)
            if context_length is not None:
                entry.context_length = context_length
            _advance_updated_at(entry)

    def finish(
        self,
        entry_id: Optional[str],
        status: str = "completed",
    ) -> None:
        if not entry_id:
            return
        notification = None
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if entry.finished_at is not None:
                return
            _append_stream_tool_previews(entry, entry.openai_stream_tool_calls)
            entry.openai_stream_tool_calls.clear()
            entry.openai_stream_last_tool_indexes.clear()
            entry.openai_stream_last_segment_was_tool = False
            self._settle_stop_reason_locked(entry, status == "completed")
            now = time.time()
            entry.status = status
            _advance_updated_at(entry, now)
            entry.finished_at = now
            entry.finished_monotonic = time.monotonic()
            self._entries.remove(entry)
            self._entries.appendleft(entry)
            self._trim_terminal_locked()
            notification = self._terminal_notification_locked(entry)
        self._notify_terminal(notification)

    def fail_open(self, entry_id: Optional[str], error: str) -> None:
        """Fail only a still-open row: unlike :meth:`fail`, a catch-all in a
        ``finally`` cannot stamp an error onto a request that already succeeded."""
        if not entry_id:
            return
        notification = None
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None or entry.finished_at is not None:
                return
            self._fail_locked(entry, error)
            notification = self._terminal_notification_locked(entry)
        self._notify_terminal(notification)

    def fail(self, entry_id: Optional[str], error: str) -> None:
        if not entry_id:
            return
        notification = None
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return
            if entry.finished_at is not None:
                if error:
                    entry.error = _trim(error, 1000)
                    _advance_updated_at(entry)
                return
            self._fail_locked(entry, error)
            notification = self._terminal_notification_locked(entry)
        self._notify_terminal(notification)

    def _fail_locked(self, entry: ApiMonitorEntry, error: str) -> None:
        _append_stream_tool_previews(entry, entry.openai_stream_tool_calls)
        entry.openai_stream_tool_calls.clear()
        entry.openai_stream_last_tool_indexes.clear()
        entry.openai_stream_last_segment_was_tool = False
        self._settle_stop_reason_locked(entry, False)
        now = time.time()
        entry.status = "error"
        entry.error = _trim(error, 1000)
        _advance_updated_at(entry, now)
        entry.finished_at = now
        entry.finished_monotonic = time.monotonic()
        self._entries.remove(entry)
        self._entries.appendleft(entry)
        self._trim_terminal_locked()

    def _terminal_notification_locked(
        self, entry: ApiMonitorEntry
    ) -> Optional[tuple[Optional[str], TerminalCallback, ApiUsageReceipt]]:
        callback_owner, callback = self._terminal_callback_locked()
        subject = canonical_api_subject(entry.subject)
        if (
            callback is None
            or entry.kind != "request"
            or entry.via_api_key is not True
            or not subject
            or entry.finished_at is None
        ):
            return None
        self._terminal_callbacks_inflight[callback_owner] = (
            self._terminal_callbacks_inflight.get(callback_owner, 0) + 1
        )
        return (
            callback_owner,
            callback,
            ApiUsageReceipt(
                id = entry.id,
                subject = subject,
                endpoint = str(entry.endpoint)[:MAX_ENDPOINT_CHARS],
                model = canonical_api_model(entry.model),
                status = str(entry.status)[:MAX_STATUS_CHARS],
                prompt_tokens = entry.prompt_tokens or 0,
                completion_tokens = entry.completion_tokens or 0,
                total_tokens = entry.total_tokens or 0,
                created_at = int(entry.finished_at * 1000),
                kind = entry.kind,
                via_api_key = entry.via_api_key,
            ),
        )

    def _notify_terminal(
        self, notification: Optional[tuple[Optional[str], TerminalCallback, ApiUsageReceipt]]
    ) -> None:
        if notification is None:
            return
        callback_owner, callback, receipt = notification
        try:
            callback(receipt)
        except Exception:  # noqa: BLE001 - monitoring must never break inference.
            logger.warning("api_monitor.terminal_callback_failed", exc_info = True)
        finally:
            with self._callback_condition:
                remaining = self._terminal_callbacks_inflight.get(callback_owner, 0) - 1
                if remaining > 0:
                    self._terminal_callbacks_inflight[callback_owner] = remaining
                else:
                    self._terminal_callbacks_inflight.pop(callback_owner, None)
                self._callback_condition.notify_all()

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
        include_prompt: bool = True,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is None:
                return None
            if not self._visible(entry, subject):
                return None
            return entry.snapshot(
                include_details = True,
                include_prompt = include_prompt,
                attributed = self._attributed(entry, subject),
            )

    def active_count(self, *, subject: Optional[str] = None) -> int:
        with self._lock:
            return sum(
                1
                for entry in self._entries
                if entry.status == "running"
                and entry.kind != "lifecycle"
                and self._attributed(entry, subject)
            )

    def clear(self, *, subject: Optional[str] = None) -> None:
        """Scoped to ``subject``: an unscoped clear would let one user erase every other user's history."""
        with self._lock:
            if subject is None:
                self._entries.clear()
                self._hidden_shared.clear()
                return
            hidden = self._hidden_shared.setdefault((current_account_id(), subject), set())
            for entry in self._entries:
                if entry.shared and entry.status != "running":
                    hidden.add(entry.id)
            # Shared rows are hidden, never dropped; dropping an own running row loses the request.
            self._entries = deque(
                entry
                for entry in self._entries
                if entry.shared or not self._attributed(entry, subject) or entry.status == "running"
            )

    def _visible(self, entry: ApiMonitorEntry, subject: Optional[str]) -> bool:
        if subject is None:
            return True
        if entry.shared:
            if entry.id in self._hidden_shared.get((current_account_id(), subject), ()):
                return False
            return _lifecycle_row_visible_to_caller(entry, subject)
        return entry.subject == subject and entry.account_id == current_account_id()

    def _attributed(self, entry: ApiMonitorEntry, subject: Optional[str]) -> bool:
        """Unscoped reads (no subject) see the row's own flag; the overlay is for its owner only."""
        if subject is None:
            return True
        return entry.subject == subject and entry.account_id == current_account_id()

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
        live = {entry.id for entry in kept}
        for key, hidden in list(self._hidden_shared.items()):
            hidden &= live
            if not hidden:
                del self._hidden_shared[key]


api_monitor = ApiMonitor(enabled = not _api_monitor_disabled())


def _lifecycle_row_visible_to_caller(entry: "ApiMonitorEntry", subject: str) -> bool:
    """Lifecycle rows name a model path that may sit inside the loading account's workspace, so only the owner and that account see them."""
    if entry.kind != "lifecycle":
        return True
    if entry.account_id == current_account_id() and entry.subject in (None, subject):
        return True
    from utils.account_context import is_owner_context

    if is_owner_context():
        return True
    return False
