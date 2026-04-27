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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger(__name__)


_EXCLUDED_PATHS = {
    "/api/train/status",
    "/api/train/metrics",
    "/api/train/hardware",
    "/api/system",
}
_EXCLUDED_SUFFIXES = (".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".ttf")


class LoggingMiddleware:
    """Pure ASGI request/response logger.

    Implemented as a raw ASGI middleware rather than ``BaseHTTPMiddleware``
    because the latter buffers the response body and cancels the response
    task group on client disconnect — which corrupts streaming responses
    (notably ``StaticFiles`` + ``FileResponse``) and surfaces in browsers
    as ``net::ERR_TOO_MANY_RETRIES`` on asset fetches.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if (
            path in _EXCLUDED_PATHS
            or path.startswith("/assets/")
            or path.endswith(_EXCLUDED_SUFFIXES)
        ):
            await self.app(scope, receive, send)
            return

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
                exc_info = True,
            )
            raise
        else:
            logger.info(
                "request_completed",
                method = scope["method"],
                path = path,
                status_code = status_code,
                process_time_ms = round((time.perf_counter() - start_time) * 1000, 2),
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
