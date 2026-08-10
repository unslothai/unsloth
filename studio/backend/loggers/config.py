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


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


# Cap on the rendered traceback in a single log record. A traceback is normally a few
# KB, but an exception whose message embeds a request body is not: a binary upload
# rejected by request validation produced one 2.2 MB line. Keep the head (where the
# raising frame is) and the tail (where the actual exception type and message are),
# and say how much was dropped. 0 disables the cap.
_MAX_EXC_CHARS = _env_int("UNSLOTH_STUDIO_MAX_EXCEPTION_CHARS", 16384)
_EXC_TAIL_CHARS = 2048


def truncate_exception(event_dict: dict) -> dict:
    """Structlog processor: bound the "exception" field to a readable size."""
    if _MAX_EXC_CHARS <= 0:
        return event_dict
    text = event_dict.get("exception")
    if not isinstance(text, str) or len(text) <= _MAX_EXC_CHARS:
        return event_dict
    head = _MAX_EXC_CHARS - _EXC_TAIL_CHARS
    dropped = len(text) - _MAX_EXC_CHARS
    event_dict["exception"] = (
        text[:head] + f"\n... [{dropped} chars of traceback omitted; "
        "raise UNSLOTH_STUDIO_MAX_EXCEPTION_CHARS to see it all] ...\n" + text[-_EXC_TAIL_CHARS:]
    )
    return event_dict


def _truncate_exception_processor(logger, method_name, event_dict):
    return truncate_exception(event_dict)


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

        # Non-ASCII on a non-UTF-8 stream raises UnicodeEncodeError (Windows,
        # LANG=C), so key off the stream, not the platform.
        for stream in (sys.stdout, sys.stderr):
            if getattr(stream, "encoding", "") and not str(stream.encoding).lower().replace(
                "-", ""
            ).startswith("utf8"):
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
                # Immediately after format_exc_info, which is what renders "exception".
                _truncate_exception_processor,
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
