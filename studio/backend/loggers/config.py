# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging configuration via structlog.

Environment-specific formats (JSON for prod, console for dev), ISO timestamps,
context-var integration, log-level filtering, and logger caching.
"""

import logging
import os
import sys
from typing import Optional

import structlog

from loggers.handlers import filter_sensitive_data

_TRUTHY = {"1", "true", "yes", "on"}

# Libraries whose INFO/DEBUG chatter carries no operational signal for Studio
# (per-request "HTTP Request: ... 200 OK", HF/transformers banners, multipart
# part dumps). Raised to WARNING unless verbose, so their errors still surface.
_NOISY_LIBS = (
    "httpx", "httpcore", "huggingface_hub", "transformers", "datasets",
    "multipart", "watchfiles", "urllib3", "filelock", "fsspec", "asyncio", "PIL",
)


def logs_verbose() -> bool:
    """True when the user asked to keep everything (`--verbose` / LOG_LEVEL=DEBUG).

    The single switch every log-noise suppression checks, so verbose restores the
    full firehose and nothing is permanently hidden."""
    if (os.getenv("UNSLOTH_STUDIO_VERBOSE", "") or "").strip().lower() in _TRUTHY:
        return True
    return os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"


class LogConfig:
    """Structured logging configuration for the application."""

    @staticmethod
    def setup_logging(
        service_name: str = "unsloth-studio-backend", env: Optional[str] = None
    ) -> structlog.BoundLogger:
        """Configure structured logging for the application.
        Args:
            service_name: Name of the service for logging identification
            env: Environment (development/production), affects logging format
        """
        # Log level from environment; fall back to INFO if invalid.
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        if sys.platform == "win32":
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, "reconfigure"):
                    try:
                        stream.reconfigure(encoding = "utf-8", errors = "replace")
                    except Exception:
                        pass

        structlog.configure(
            processors = [
                # Ordered to control output field order.
                structlog.processors.TimeStamper(fmt = "iso"),  # timestamp first
                structlog.processors.add_log_level,  # level second
                structlog.contextvars.merge_contextvars,
                structlog.processors.format_exc_info,
                filter_sensitive_data,
                # Flatten the extra field into the main dict.
                lambda logger, method_name, event_dict: {
                    "timestamp": event_dict.get("timestamp"),
                    "level": event_dict.get("level"),
                    "event": event_dict.get("event"),
                    **(event_dict.get("extra", {})),  # Flatten extra into main dict
                    **{
                        k: v
                        for k, v in event_dict.items()
                        if k not in ["timestamp", "level", "event", "extra"]
                    },
                },
                (
                    structlog.processors.JSONRenderer(sort_keys = False)  # Preserve order
                    if env == "production"
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            wrapper_class = structlog.make_filtering_bound_logger(log_level),
            logger_factory = structlog.PrintLoggerFactory(file = sys.stdout),
            cache_logger_on_first_use = True,
        )

        if not logs_verbose():
            for name in _NOISY_LIBS:
                logging.getLogger(name).setLevel(logging.WARNING)
            logging.captureWarnings(True)

        return structlog.get_logger(service_name)
