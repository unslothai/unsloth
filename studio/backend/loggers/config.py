# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Logging configuration for structured logging with structlog.

This module provides centralized logging configuration with environment-specific
formats and processors. Supports both development and production environments
with consistent structured logging.

Key Features:
- Environment-specific formatting (JSON for production, console for development)
- Timestamp standardization (ISO format)
- Context variable integration
- Log level filtering
- Logger caching for performance
"""

import logging
import os
import sys
from typing import Optional

import structlog


class LogConfig:
    """Structured logging configuration for the application.

    Provides static method to configure structlog with environment-specific
    formatting and processors for consistent structured logging.
    """

    @staticmethod
    def setup_logging(
        service_name: str = "unsloth-studio-backend", env: Optional[str] = None
    ) -> structlog.BoundLogger:
        """Configure structured logging for the application.
        Args:
            service_name: Name of the service for logging identification
            env: Environment (development/production), affects logging format
        """
        # Determine log level from environment
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        # Fallback to INFO if an invalid level is provided
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
                # Reorder processors to control field order
                structlog.processors.TimeStamper(fmt = "iso"),  # timestamp first
                structlog.processors.add_log_level,  # level second
                structlog.contextvars.merge_contextvars,
                # Custom processor to flatten the extra field
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

        return structlog.get_logger(service_name)
