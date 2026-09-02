# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging handlers and middleware.

LoggingMiddleware (request/response logging with timing),
filter_sensitive_data (structlog processor for sanitization), and
get_logger (factory for structured loggers).
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


# Collapse identical GET/2xx logs within the window, since the SPA fans one invalidation into many
# list fetches; mutations and errors always log.
_ACCESS_LOG_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", 300)
# Liveness/UI polls whose line means only "still polling"; collapse to a longer
_QUIET_POLL_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", 10000)
# The desktop watchdog probe rounds are ~19s apart, so a 10s window that stamps only on emit would
# collapse nothing; it gets its own, wider window.
_WATCHDOG_POLL_DEDUP_MS = _env_int("UNSLOTH_STUDIO_ACCESS_LOG_WATCHDOG_DEDUP_MS", 60000)
# Both windows off is what --verbose sets, and the drop-the-2xx suppressor below has no window of
# its own, so it reads the same signal.
_VERBOSE_ACCESS_LOG = _ACCESS_LOG_DEDUP_MS <= 0 and _QUIET_POLL_DEDUP_MS <= 0
_QUIET_POLL_PATHS = {
    "/api/health",
    "/api/auth/status",
    "/api/inference/status",
    "/api/inference/monitor",
    # The loaded-models indicator polls all four runtimes every 5s for as long as the app is open, and
    # on the desktop every line is mirrored into tauri.log.
    "/api/inference/images/status",
    "/api/inference/video/status",
    "/api/inference/audio/stt/status",
    # Re-read whenever the settings dialog or the remote-access section is open, and it is a plain read
    # of a toggle: 116 lines in the 4h sample.
    "/api/settings/remote-access",
    # List polls the tabs refetch on a timer and on every tab switch.
    "/api/train/runs",
    "/api/models/checkpoints",
    "/api/models/local",
    "/api/rag/knowledge-bases",
    # Legacy download polls emit no progress events (unlike /api/hub/*), so heartbeat them.
    "/api/models/download-progress",
    "/api/models/gguf-download-progress",
    "/api/datasets/download-progress",
    # Generation is fire-and-forget: its outcome only reaches the UI via these polls.
    "/api/inference/images/generate-progress",
    "/api/inference/video/generate-progress",
    # Polled every 1.5s while the train UI is open.
    "/api/train/diffusion/status",
    # These handlers log nothing and diffusion.loaded / video.loaded are terminal, so a minutes-long
    # load would otherwise emit nothing at all.
    "/api/inference/images/load-progress",
    "/api/inference/video/load-progress",
    # Templated; matched through normalize_poll_path. See _TEMPLATED_POLL_PATHS.
    "/api/chat/threads/{id}",
    "/api/chat/threads/{id}/forks",
}
# The pure-liveness subset: the SPA fires them together in one burst, so they share a single bucket,
# the first logging with its real path and the rest of the window dropping.
# Only this group is shared: the other quiet paths each report on a different subsystem, so they
# keep their own heartbeat.
# Two paths are deliberately NOT here because their latency is worth seeing: /api/health waits on
# hardware detection, and /api/inference/status reads llama.cpp capabilities in an executor.
_LIVENESS_POLL_PATHS = frozenset(
    {
        "/api/auth/status",
        "/api/inference/monitor",
        "/api/inference/images/status",
        "/api/inference/video/status",
        "/api/inference/audio/stt/status",
    }
)
# The desktop shell's watchdog probe was in no suppressor at all: 760 lines on an idle 4h session.
# Out of _QUIET_POLL_PATHS because that window is narrower than the poll interval.
_WATCHDOG_POLL_PATHS = {"/api/liveness"}
# Bucket key for the group above; not a real (method, path, query, status), so it can never collide with one.
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
# GET polls whose 2xx line carries no signal, so drop it entirely; non-2xx still logs. Only /api/hub
# download polls emit events.
_QUIET_SUCCESS_PATHS = {
    "/api/inference/load-progress",
    "/api/llama/update-status",
    "/api/export/logs",
    "/api/export/status",
    # The Settings > Logs viewer polls are suppressed in _SELF_READ_PATHS below.
    "/api/hub/download-status",
    "/api/hub/download-progress",
    "/api/hub/gguf-download-progress",
    "/api/hub/active-downloads",
    "/api/hub/transport-status",
    "/api/hub/datasets/download-status",
    "/api/hub/datasets/download-progress",
    "/api/hub/datasets/active-downloads",
    "/api/hub/datasets/transport-status",
    # Boot-burst catalog reads: the SPA refetches these on every auth change and dialog open, and each
    # outcome is already visible in the UI.
    # syncExternalProvidersFromBackend fetches the pair in one Promise.all, so they always arrive as two
    # lines saying the same thing.
    "/api/providers/registry",
    "/api/providers/",
    # Fetched in the same Promise.all as two routes that keep their access line, so the resync is still
    # traceable without it.
    "/api/models/loras",
    # Suppression is GET-only, so the PUT that actually writes a profile change still logs.
    "/api/settings/personalization",
}
# The token-refresh route: its first 2xx means the client has a valid session, so from then on chat
# 401s are real failures and must stay visible.
_AUTH_REFRESH_PATH = "/api/auth/refresh"
# Exact paths only, so detail and message reads keep their logs; the pre-auth 401 race also fires on these polls.
_CHAT_LIST_PATHS = {
    "/api/chat/threads",
    "/api/chat/projects",
}
# The sets above match EXACTLY so detail reads keep their access line, but streaming drives these
# two on a loop (34% of the access log over a 20s session), so they get a heartbeat instead.
_CHAT_THREAD_DETAIL = "/api/chat/threads/{id}"
_CHAT_THREAD_FORKS = "/api/chat/threads/{id}/forks"
_TEMPLATED_POLL_PATHS = frozenset({_CHAT_THREAD_DETAIL, _CHAT_THREAD_FORKS})
# One id segment, no slashes: a deeper path such as /threads/{id}/messages/{mid} must NOT collapse
# into the detail bucket.
_CHAT_THREAD_PATH_RE = re.compile(r"^/api/chat/threads/(?!$)[^/]+(/forks)?$")


def normalize_poll_path(path: str) -> str:
    """Collapse a per-resource id so a templated path can join a suppression class.

    Used for classification and the de-duplication bucket only; the emitted line still
    carries the real path. One bucket across ids is deliberate, as with the liveness
    group: four tabs polling four threads are still one question.
    """
    m = _CHAT_THREAD_PATH_RE.match(path)
    if m is None:
        return path
    return _CHAT_THREAD_FORKS if m.group(1) else _CHAT_THREAD_DETAIL


# The log viewer polls these while reading the very file this middleware writes, so unsuppressed
# each poll appends a record the next reads back; _is_redundant_repeat cannot cover it.
# Separate from _QUIET_SUCCESS_PATHS because --verbose must NOT lift this one: the extra noise
# buries the failure the viewer was opened for.
_SELF_READ_PATHS = {
    "/api/settings/debug/logs",
    "/api/settings/debug/logs/sources",
}


def _is_quiet_success(method: str, path: str, status_code: int, pre_auth: bool) -> bool:
    """GET-only. Suppress a 2xx poll line that carries no signal, plus a chat list
    poll's transient pre-auth 401 (only in the bootstrap window before the first
    successful token refresh). Mutations, real (post-refresh) auth failures, and
    all other errors always log. --verbose disables the whole suppressor, except
    for the log viewer's own reads."""
    if method != "GET":
        return False
    if 200 <= status_code < 300 and path in _SELF_READ_PATHS:
        return True
    if _VERBOSE_ACCESS_LOG:
        return False
    if 200 <= status_code < 300:
        return path in _QUIET_SUCCESS_PATHS or path in _CHAT_LIST_PATHS
    return pre_auth and status_code == 401 and path in _CHAT_LIST_PATHS


# An unhandled request exception is logged twice: here as a structured request_failed event whose
# "exception" field carries the whole traceback (format_exc_info renders it, see
# loggers/config.py), and again by uvicorn on stderr, which the desktop shell mirrors into
# tauri.log at ~90 lines per failure. Keep the structured copy.
_UVICORN_ASGI_EXC_MSG = "Exception in ASGI application"
# Set on the exception instance itself rather than a side table: the object is what uvicorn hands
# us, so the match cannot go stale or collide with a recycled id, and an exception raised above
# this middleware (CORS, remote-access, the protocol layer) keeps uvicorn's traceback.
_LOGGED_EXC_ATTR = "_unsloth_request_failed_logged"


def _mark_exception_logged(exc: BaseException) -> None:
    """Flag exc as already reported by request_failed. Best effort: an exception type
    that refuses attributes just means both copies are logged, as before."""
    try:
        setattr(exc, _LOGGED_EXC_ATTR, True)
    except Exception:
        pass


class _DropDuplicateAsgiException(logging.Filter):
    """Drop uvicorn's "Exception in ASGI application" record when request_failed has
    already logged that same exception. Anything else, including a failure that never
    reached this middleware, passes through untouched. --verbose keeps both copies."""

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
    """Attach the duplicate-traceback filter to uvicorn's error logger. Same
    logger-level filter technique as run.py's startup-line rewrite; safe to call more
    than once because a second identical install only re-checks the same records."""
    logging.getLogger("uvicorn.error").addFilter(_DropDuplicateAsgiException())


class LoggingMiddleware:
    """ASGI request logger that avoids BaseHTTPMiddleware streaming wrappers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # (method, path, query, status_code) -> monotonic ts of the last EMITTED log.
        self._last_log: dict[tuple[str, str, bytes, int], float] = {}
        # Before the first successful /api/auth/refresh, chat list-poll 401s are the transient bootstrap
        # race and are suppressed.
        self._auth_refreshed = False

    def _is_redundant_repeat(
        self, method: str, path: str, query: bytes, status_code: int, now: float
    ) -> bool:
        """True if an identical GET/2xx log fired < window ago (query string is part
        of the identity). Non-GET/non-2xx never dedup; quiet-poll paths use the longer
        heartbeat. Stamps only on emit, so steady polls still log."""
        if method != "GET" or not (200 <= status_code < 300):
            return False
        # A query makes the request something other than the background poll, so it keeps its own identity
        # rather than joining the bucket.
        is_liveness = path in _LIVENESS_POLL_PATHS and not query
        # Bucketed by template, so four tabs polling four threads share one heartbeat.
        norm = normalize_poll_path(path) if not query else path
        if path in _WATCHDOG_POLL_PATHS:
            # Zeroed along with the quiet window, so --verbose still logs every probe.
            window_ms = _WATCHDOG_POLL_DEDUP_MS if _QUIET_POLL_DEDUP_MS > 0 else 0
        elif is_liveness or norm in _QUIET_POLL_PATHS:
            window_ms = _QUIET_POLL_DEDUP_MS
        else:
            window_ms = _ACCESS_LOG_DEDUP_MS
        if window_ms <= 0:
            return False
        # The liveness group shares one bucket, so a burst logs once rather than once per path; only the
        # query-less form joins it.
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
