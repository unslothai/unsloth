# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging handlers and middleware: LoggingMiddleware (request/response logging with timing),
filter_sensitive_data (structlog processor for sanitization), and get_logger (factory for structured loggers).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING

import structlog

# Annotations only: a runtime import makes the ASGI stack a hard dependency of every CLI command.
if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.native_path_leases import redact_native_paths

logger = structlog.get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


_ACCESS_LOG_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", 300)
_QUIET_POLL_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", 10000)
# Watchdog probes are ~19s apart, so they need a window wider than the 10s quiet one.
_WATCHDOG_POLL_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_WATCHDOG_DEDUP_MS", 60000)
_VERBOSE_ACCESS_LOG = _ACCESS_LOG_DEDUP_MS <= 0 and _QUIET_POLL_DEDUP_MS <= 0
_QUIET_POLL_PATHS = {
    "/api/health",
    "/api/auth/status",
    "/api/inference/status",
    "/api/inference/monitor",
    "/api/inference/images/status",
    "/api/inference/video/status",
    "/api/inference/audio/stt/status",
    "/api/settings/remote-access",
    "/api/train/runs",
    "/api/models/checkpoints",
    "/api/models/local",
    "/api/rag/knowledge-bases",
    "/api/models/download-progress",
    "/api/models/gguf-download-progress",
    "/api/datasets/download-progress",
    "/api/train/diffusion/status",
    "/api/chat/threads/{id}",
    "/api/chat/threads/{id}/forks",
}
# Pure-liveness paths share one bucket; /api/health and /api/inference/status excluded on purpose.
_LIVENESS_POLL_PATHS = frozenset(
    {
        "/api/auth/status",
        "/api/inference/monitor",
        "/api/inference/images/status",
        "/api/inference/video/status",
        "/api/inference/audio/stt/status",
    }
)
_WATCHDOG_POLL_PATHS = {"/api/liveness"}
_LIVENESS_DEDUP_KEY = ("GET", "\x00liveness", b"", 200)
_DEDUP_MAP_MAX = 4096
_NATIVE_PATH_LEASE_RE = re.compile(
    r"(?i)(\b(?:native_path_lease|nativePathLease)[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)
_EXCLUDED_PATHS = {
    "/api/train/status",
    "/api/train/metrics",
    "/api/train/hardware",
    "/api/system",
}
_EXCLUDED_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
)
_QUIET_SUCCESS_PATHS = {
    "/api/inference/load-progress",
    "/api/inference/images/load-progress",
    "/api/inference/video/load-progress",
    "/api/inference/images/generate-progress",
    "/api/inference/video/generate-progress",
    "/api/llama/update-status",
    "/api/export/logs",
    "/api/export/status",
    "/api/hub/download-status",
    "/api/hub/download-progress",
    "/api/hub/gguf-download-progress",
    "/api/hub/active-downloads",
    "/api/hub/transport-status",
    "/api/hub/datasets/download-status",
    "/api/hub/datasets/download-progress",
    "/api/hub/datasets/active-downloads",
    "/api/hub/datasets/transport-status",
    "/api/providers/registry",
    "/api/providers/",
    "/api/models/loras",
    "/api/settings/personalization",
}
# After its first 2xx, chat list 401s are real failures, not the bootstrap race.
_AUTH_REFRESH_PATH = "/api/auth/refresh"
_CHAT_LIST_PATHS = {
    "/api/chat/threads",
    "/api/chat/projects",
}
_CHAT_THREAD_DETAIL = "/api/chat/threads/{id}"
_CHAT_THREAD_FORKS = "/api/chat/threads/{id}/forks"
_TEMPLATED_POLL_PATHS = frozenset({_CHAT_THREAD_DETAIL, _CHAT_THREAD_FORKS})
# One id segment: deeper paths (/threads/{id}/messages/...) must NOT join the detail bucket.
_CHAT_THREAD_PATH_RE = re.compile(r"^/api/chat/threads/(?!$)[^/]+(/forks)?$")


def normalize_poll_path(path: str) -> str:
    """Collapse a per-resource id so a templated path can join a suppression class. Used for classification and
    the de-duplication bucket only; the emitted line still carries the real path. One bucket across ids is
    deliberate, as with the liveness group: four tabs polling four threads are still one question.
    """
    m = _CHAT_THREAD_PATH_RE.match(path)
    if m is None:
        return path
    return _CHAT_THREAD_FORKS if m.group(1) else _CHAT_THREAD_DETAIL


# Log viewer reads this very file; --verbose must NOT lift this suppression.
_SELF_READ_PATHS = {
    "/api/settings/debug/logs",
    "/api/settings/debug/logs/sources",
}


def _is_quiet_success(method: str, path: str, status_code: int, pre_auth: bool) -> bool:
    """GET-only. Suppress a 2xx poll line that carries no signal, plus a chat list poll's transient pre-auth 401
    (only in the bootstrap window before the first successful token refresh). Mutations, real (post-refresh)
    auth failures, and all other errors always log. --verbose disables the whole suppressor, except for the log
    viewer's own reads."""
    if method != "GET":
        return False
    if 200 <= status_code < 300 and path in _SELF_READ_PATHS:
        return True
    if _VERBOSE_ACCESS_LOG:
        return False
    if 200 <= status_code < 300:
        return path in _QUIET_SUCCESS_PATHS or path in _CHAT_LIST_PATHS
    return pre_auth and status_code == 401 and path in _CHAT_LIST_PATHS


# uvicorn re-logs exceptions already logged here as request_failed; keep only the structured copy.
_UVICORN_ASGI_EXC_MSG = "Exception in ASGI application"
# Marked on the exception itself so the match cannot go stale or hit a recycled id.
_LOGGED_EXC_ATTR = "_unsloth_request_failed_logged"


def _mark_exception_logged(exc: BaseException) -> None:
    """Flag exc as already reported by request_failed. Best effort: an exception type
    that refuses attributes just means both copies are logged, as before."""
    try:
        setattr(exc, _LOGGED_EXC_ATTR, True)
    except Exception:
        pass


class _DropDuplicateAsgiException(logging.Filter):
    """Drop uvicorn's "Exception in ASGI application" record when request_failed has already logged that same
    exception. Anything else, including a failure that never reached this middleware, passes through untouched.
    --verbose keeps both copies."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _VERBOSE_ACCESS_LOG:
            return True
        try:
            msg = record.msg if isinstance(record.msg, str) else ""
            if not msg.startswith(_UVICORN_ASGI_EXC_MSG):
                return True
            exc_info = record.exc_info
            exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
            return not getattr(exc, _LOGGED_EXC_ATTR, False)
        except Exception:
            return True


def install_uvicorn_duplicate_exception_filter() -> None:
    """Attach the duplicate-traceback filter to uvicorn's error logger. Same logger-level filter technique as
    run.py's startup-line rewrite; safe to call more than once because a second identical install only re-checks
    the same records."""
    logging.getLogger("uvicorn.error").addFilter(_DropDuplicateAsgiException())


class LoggingMiddleware:
    """ASGI request logger that avoids BaseHTTPMiddleware streaming wrappers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._last_log: dict[tuple[str, str, bytes, int], float] = {}
        self._auth_refreshed = False

    def _is_redundant_repeat(
        self, method: str, path: str, query: bytes, status_code: int, now: float
    ) -> bool:
        """True if an identical GET/2xx log fired < window ago (query string is part of the identity). Non-GET/non-2xx
        never dedup; quiet-poll paths use the longer heartbeat. Stamps only on emit, so steady polls still log."""
        if method != "GET" or not (200 <= status_code < 300):
            return False
        is_liveness = path in _LIVENESS_POLL_PATHS and not query
        norm = normalize_poll_path(path) if not query else path
        if path in _WATCHDOG_POLL_PATHS:
            # Zeroed with the quiet window, so --verbose still logs every probe.
            window_ms = _WATCHDOG_POLL_DEDUP_MS if _QUIET_POLL_DEDUP_MS > 0 else 0
        elif is_liveness or norm in _QUIET_POLL_PATHS:
            window_ms = _QUIET_POLL_DEDUP_MS
        else:
            window_ms = _ACCESS_LOG_DEDUP_MS
        if window_ms <= 0:
            return False
        key = _LIVENESS_DEDUP_KEY if is_liveness else (method, norm, query, status_code)
        last = self._last_log.get(key)
        if last is not None and (now - last) * 1000.0 < window_ms:
            return True
        self._last_log[key] = now
        if len(self._last_log) > _DEDUP_MAP_MAX:
            widest = max(_ACCESS_LOG_DEDUP_MS, _QUIET_POLL_DEDUP_MS, _WATCHDOG_POLL_DEDUP_MS)
            cutoff = now - (widest / 1000.0)
            self._last_log = {k: v for k, v in self._last_log.items() if v >= cutoff}
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        excluded = (
            path in _EXCLUDED_PATHS
            or path.startswith("/assets/")
            or path.endswith(_EXCLUDED_SUFFIXES)
        )
        start_time = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.error(
                "request_failed",
                path = path,
                method = scope["method"],
                status_code = status_code,
                error = str(exc),
                process_time_ms = round((time.perf_counter() - start_time) * 1000, 2),
                exc_info = True,
            )
            _mark_exception_logged(exc)
            raise
        else:
            end_time = time.perf_counter()
            if 200 <= status_code < 300 and path == _AUTH_REFRESH_PATH:
                self._auth_refreshed = True
            if (
                not excluded
                and not _is_quiet_success(
                    scope["method"], path, status_code, not self._auth_refreshed
                )
                and not self._is_redundant_repeat(
                    scope["method"], path, scope.get("query_string", b""), status_code, end_time
                )
            ):
                logger.info(
                    "request_completed",
                    method = scope["method"],
                    path = path,
                    status_code = status_code,
                    process_time_ms = round((end_time - start_time) * 1000, 2),
                )


def filter_sensitive_data(logger, method_name, event_dict):
    """Structlog processor to redact native path leases from logs."""

    def filter_value(value):
        if isinstance(value, str):
            try:
                value = redact_native_paths(value)
            except Exception:
                pass
            value = _NATIVE_PATH_LEASE_RE.sub(r"\1<redacted native path lease>", value)
            return value
        elif isinstance(value, dict):
            return {
                k: "<redacted native path lease>"
                if str(k).replace("_", "").lower() == "nativepathlease"
                else filter_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [filter_value(item) for item in value]
        return value

    return {
        k: "<redacted native path lease>"
        if str(k).replace("_", "").lower() == "nativepathlease"
        else filter_value(v)
        for k, v in event_dict.items()
    }


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a bound structured logger for a module (name is usually __name__)."""
    return structlog.get_logger(name)
