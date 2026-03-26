# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Logging handlers and middleware for structured logging.

This module provides FastAPI middleware and structlog processors for:
- Request/response logging with timing
- Sensitive data filtering in logs
- Structured logging configuration
- Error handling with detailed context

Key Components:
- LoggingMiddleware: FastAPI middleware for request/response logging
- filter_sensitive_data: Structlog processor for data sanitization
- get_logger: Factory function for structured loggers
"""

import time

import structlog

logger = structlog.get_logger(__name__)

# Paths and suffixes excluded from request logging.
_EXCLUDED_PATHS = frozenset({
    "/api/train/status",
    "/api/train/metrics",
    "/api/train/hardware",
    "/api/system",
})
_EXCLUDED_SUFFIXES = (".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".ttf")


class LoggingMiddleware:
    """Pure ASGI middleware for request logging.

    Unlike Starlette's BaseHTTPMiddleware, this wraps the ``send``
    callable directly -- no anyio memory channel, no extra coroutine
    context switches per response body chunk.  This matters for SSE
    streaming where hundreds of small chunks are sent per second.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            path = scope.get("path", "")
            method = scope.get("method", "")
            logger.error(
                "request_failed",
                path=path,
                method=method,
                error=str(e),
                exc_info=True,
            )
            raise

        # Log after response completes (same exclusion logic as before)
        process_time = (time.time() - start_time) * 1000
        path = scope.get("path", "")
        is_excluded = (
            path in _EXCLUDED_PATHS
            or path.startswith("/assets/")
            or path.endswith(_EXCLUDED_SUFFIXES)
        )
        if not is_excluded:
            method = scope.get("method", "")
            logger.info(
                "request_completed",
                method=method,
                path=path,
                status_code=status_code,
                process_time_ms=round(process_time, 2),
            )


def filter_sensitive_data(logger, method_name, event_dict):
    """Structlog processor to filter out base64 data from logs."""

    def filter_value(value):
        if (
            isinstance(value, str)
            and len(value) > 100
            and ("," in value or "/" in value)
        ):
            # Likely base64 data, truncate it
            return value[:20] + "..."
        elif isinstance(value, dict):
            return {k: filter_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [filter_value(item) for item in value]
        return value

    return {k: filter_value(v) for k, v in event_dict.items()}


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance for a specific module.
    Args:
        name: Usually __name__ of the module
    Returns:
        A bound structured logger
    """
    return structlog.get_logger(name)
