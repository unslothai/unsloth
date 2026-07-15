# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging handlers and middleware.

LoggingMiddleware (request/response logging with timing),
filter_sensitive_data (structlog processor for sanitization), and
get_logger (factory for structured loggers).
"""

import os
import re
import time

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.native_path_leases import redact_native_paths

logger = structlog.get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


# Collapse identical GET/2xx logs within the window (the SPA fans one invalidation
# into many list fetches). Mutations and errors always log. 0 = off.
_ACCESS_LOG_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", 300)
# Liveness/UI polls whose line means only "still polling"; collapse to a longer
# heartbeat. First hit and errors still log. 0 = off.
_QUIET_POLL_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", 10000)
_QUIET_POLL_PATHS = {
    "/api/health",
    "/api/auth/status",
    "/api/inference/status",
    "/api/inference/monitor",
    # List polls the tabs refetch on a timer and on every tab switch.
    "/api/train/runs",
    "/api/models/checkpoints",
    "/api/models/local",
    "/api/rag/knowledge-bases",
    # Legacy download polls emit no progress events (unlike /api/hub/*), so heartbeat them.
    "/api/models/download-progress",
    "/api/models/gguf-download-progress",
    "/api/datasets/download-progress",
}
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
# GET polls whose 2xx line carries no signal (their progress/phase events and the UI
# do), so drop it entirely; non-2xx still logs. Only /api/hub download polls emit
# events; the legacy /api/models and /api/datasets ones heartbeat via _QUIET_POLL_PATHS.
_QUIET_SUCCESS_PATHS = {
    "/api/inference/load-progress",
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
}
# The token-refresh route. Its first 2xx means the client has obtained a valid
# session, so from then on chat 401s are real failures and must stay visible.
_AUTH_REFRESH_PATH = "/api/auth/refresh"
# High-frequency chat list polls; their 2xx is covered by generation/tool-call/stats
# events. Exact paths only, so detail/message reads (/threads/{id}, .../messages,
# /projects/{id}) keep their logs. The pre-auth 401 race also fires on these polls.
_CHAT_LIST_PATHS = {
    "/api/chat/threads",
    "/api/chat/projects",
}


def _is_quiet_success(method: str, path: str, status_code: int, pre_auth: bool) -> bool:
    """GET-only. Suppress a 2xx poll line that carries no signal, plus a chat list
    poll's transient pre-auth 401 (only in the bootstrap window before the first
    successful token refresh). Mutations, real (post-refresh) auth failures, and
    all other errors always log."""
    if method != "GET":
        return False
    if 200 <= status_code < 300:
        return path in _QUIET_SUCCESS_PATHS or path in _CHAT_LIST_PATHS
    return pre_auth and status_code == 401 and path in _CHAT_LIST_PATHS


class LoggingMiddleware:
    """ASGI request logger that avoids BaseHTTPMiddleware streaming wrappers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # (method, path, query, status_code) -> monotonic ts of the last EMITTED log.
        self._last_log: dict[tuple[str, str, bytes, int], float] = {}
        # Flips True after the first successful /api/auth/refresh; before that, chat
        # list-poll 401s are the transient bootstrap race and are suppressed.
        self._auth_refreshed = False

    def _is_redundant_repeat(
        self, method: str, path: str, query: bytes, status_code: int, now: float
    ) -> bool:
        """True if an identical GET/2xx log fired < window ago (query string is part
        of the identity). Non-GET/non-2xx never dedup; quiet-poll paths use the longer
        heartbeat. Stamps only on emit, so steady polls still log."""
        if method != "GET" or not (200 <= status_code < 300):
            return False
        window_ms = _QUIET_POLL_DEDUP_MS if path in _QUIET_POLL_PATHS else _ACCESS_LOG_DEDUP_MS
        if window_ms <= 0:
            return False
        key = (method, path, query, status_code)
        last = self._last_log.get(key)
        if last is not None and (now - last) * 1000.0 < window_ms:
            return True
        self._last_log[key] = now
        if len(self._last_log) > _DEDUP_MAP_MAX:
            cutoff = now - (max(_ACCESS_LOG_DEDUP_MS, _QUIET_POLL_DEDUP_MS) / 1000.0)
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
