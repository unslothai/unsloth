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


class _DropTorchDtypeDeprecation(logging.Filter):
    """Drop transformers' once-per-run "`torch_dtype` is deprecated" warning_once.
    It is emitted via logging (not warnings), so a warnings filter cannot catch it."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("torch_dtype" in msg and "deprecated" in msg)


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

        # Drop transformers' cosmetic "`torch_dtype` is deprecated" warning_once (see filter).
        _dtype_filter = _DropTorchDtypeDeprecation()
        for _name in (
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "transformers.pipelines.base",
        ):
            logging.getLogger(_name).addFilter(_dtype_filter)

        return structlog.get_logger(service_name)
