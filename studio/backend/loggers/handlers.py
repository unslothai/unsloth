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
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        try:
            response = await call_next(request)

            # Log response
            process_time = (time.time() - start_time) * 1000

            EXCLUDED_PATHS = {
                "/api/train/status",
                "/api/train/metrics",
                "/api/train/hardware",
                "/api/system",
            }
            is_excluded = (
                request.url.path in EXCLUDED_PATHS
                or request.url.path.startswith("/assets/")
                or request.url.path.endswith(
                    (".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".ttf")
                )
            )

            if not is_excluded:
                logger.info(
                    "request_completed",
                    method = request.method,
                    path = request.url.path,
                    status_code = response.status_code,
                    process_time_ms = round(process_time, 2),
                )

            return response

        except Exception as e:
            logger.error(
                "request_failed",
                path = request.url.path,
                method = request.method,
                error = str(e),
                exc_info = True,
            )
            raise


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
