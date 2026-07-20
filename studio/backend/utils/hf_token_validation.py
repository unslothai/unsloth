# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached, rate-limited Hugging Face token validation."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


TokenValidationStatus = Literal["valid", "invalid", "rate_limited", "unavailable"]


@dataclass(frozen = True)
class TokenValidationResult:
    status: TokenValidationStatus
    retry_after_seconds: int | None = None


_WINDOW_SECONDS = 3600.0
_MAX_ATTEMPTS = 3
_CACHE_TTL_SECONDS = 3600.0
_TEMPORARY_CACHE_TTL_SECONDS = 15.0
_MAX_BUCKETS = 4096
_MAX_CACHE_ENTRIES = 4096
_INFLIGHT_WAIT_SECONDS = 30.0

_attempts: dict[str, deque[float]] = {}
_cache: dict[str, tuple[float, TokenValidationResult]] = {}
_inflight: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _prune_attempts(bucket: deque[float], now: float) -> None:
    while bucket and now - bucket[0] >= _WINDOW_SECONDS:
        bucket.popleft()


def _prune_locked(now: float) -> None:
    for key in list(_attempts):
        bucket = _attempts[key]
        _prune_attempts(bucket, now)
        if not bucket:
            del _attempts[key]
    for key, (expires_at, _result) in list(_cache.items()):
        if expires_at <= now:
            del _cache[key]


def _cached_locked(fingerprint: str, now: float) -> TokenValidationResult | None:
    cached = _cache.get(fingerprint)
    if cached is None:
        return None
    expires_at, result = cached
    if expires_at <= now:
        del _cache[fingerprint]
        return None
    return result


def _retry_after(bucket: deque[float], now: float) -> int:
    return max(1, int(_WINDOW_SECONDS - (now - bucket[0])) + 1)


def _reserve_attempt_locked(rate_key: str, now: float) -> TokenValidationResult | None:
    bucket = _attempts.get(rate_key)
    if bucket is None:
        if len(_attempts) >= _MAX_BUCKETS:
            _prune_locked(now)
        if len(_attempts) >= _MAX_BUCKETS:
            return TokenValidationResult(
                status = "rate_limited",
                retry_after_seconds = max(1, int(_WINDOW_SECONDS)),
            )
        bucket = _attempts[rate_key] = deque()
    _prune_attempts(bucket, now)
    if len(bucket) >= _MAX_ATTEMPTS:
        return TokenValidationResult(
            status = "rate_limited",
            retry_after_seconds = _retry_after(bucket, now),
        )
    bucket.append(now)
    return None


def _http_status(exc: HfHubHTTPError) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _remote_retry_after(exc: HfHubHTTPError) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After")
    try:
        return max(1, int(float(raw))) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _check_remote(token: str) -> TokenValidationResult:
    try:
        HfApi().whoami(token = token)
        return TokenValidationResult(status = "valid")
    except HfHubHTTPError as exc:
        status = _http_status(exc)
        if status == 401:
            return TokenValidationResult(status = "invalid")
        if status == 429:
            return TokenValidationResult(
                status = "rate_limited",
                retry_after_seconds = _remote_retry_after(exc),
            )
        return TokenValidationResult(status = "unavailable")
    except Exception:
        return TokenValidationResult(status = "unavailable")


def validate_hf_token(token: str, *, rate_key: str) -> TokenValidationResult:
    """Validate ``token`` without retaining it, sharing results across callers.

    Cached checks do not consume the caller's three-per-hour network budget. A
    single-flight event also prevents simultaneously mounted UI surfaces from
    sending duplicate ``whoami`` requests for the same token.
    """
    normalized = token.strip()
    if not normalized:
        return TokenValidationResult(status = "invalid")
    token_fingerprint = _fingerprint(normalized)
    owner_event: threading.Event | None = None

    try:
        while True:
            now = time.monotonic()
            with _lock:
                cached = _cached_locked(token_fingerprint, now)
                if cached is not None:
                    return cached
                waiting = _inflight.get(token_fingerprint)
                if waiting is None:
                    limited = _reserve_attempt_locked(rate_key, now)
                    if limited is not None:
                        return limited
                    owner_event = threading.Event()
                    _inflight[token_fingerprint] = owner_event
                    break
            if not waiting.wait(_INFLIGHT_WAIT_SECONDS):
                return TokenValidationResult(status = "unavailable")

        result = _check_remote(normalized)
        now = time.monotonic()
        ttl = (
            _CACHE_TTL_SECONDS
            if result.status in ("valid", "invalid")
            else max(_TEMPORARY_CACHE_TTL_SECONDS, float(result.retry_after_seconds or 0))
        )
        with _lock:
            if len(_cache) >= _MAX_CACHE_ENTRIES:
                _prune_locked(now)
            if len(_cache) < _MAX_CACHE_ENTRIES:
                _cache[token_fingerprint] = (now + ttl, result)
        return result
    finally:
        if owner_event is not None:
            with _lock:
                event = _inflight.get(token_fingerprint)
                if event is owner_event:
                    _inflight.pop(token_fingerprint, None)
                    event.set()


def reset_hf_token_validation_state() -> None:
    """Clear process state for test isolation."""
    with _lock:
        for event in _inflight.values():
            event.set()
        _inflight.clear()
        _attempts.clear()
        _cache.clear()
