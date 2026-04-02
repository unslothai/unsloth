# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Shared pytest configuration for the backend test suite.
Ensures that the backend root is on sys.path so that
`import utils.utils` (and similar flat imports) resolve correctly.
"""

import sys
from pathlib import Path

# Add backend root to sys.path (mirrors how the app itself is launched)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


def _install_structlog_stub() -> None:
    try:
        import structlog  # type: ignore  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import types

    class _DummyLogger:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return self

            return _noop

    def _get_logger(*args, **kwargs):
        return _DummyLogger()

    structlog_stub = types.SimpleNamespace(
        BoundLogger = type("BoundLogger", (), {}),
        get_logger = _get_logger,
        configure = lambda *args, **kwargs: None,
        make_filtering_bound_logger = lambda *args, **kwargs: _get_logger,
        PrintLoggerFactory = lambda *args, **kwargs: None,
        processors = types.SimpleNamespace(
            TimeStamper = lambda *args, **kwargs: None,
            add_log_level = lambda *args, **kwargs: None,
            JSONRenderer = lambda *args, **kwargs: None,
            format_exc_info = lambda *args, **kwargs: None,
        ),
        contextvars = types.SimpleNamespace(
            merge_contextvars = lambda *args, **kwargs: None,
        ),
        dev = types.SimpleNamespace(
            ConsoleRenderer = lambda *args, **kwargs: None,
        ),
        stdlib = types.SimpleNamespace(
            add_log_level = lambda *args, **kwargs: None,
            filter_by_level = lambda *args, **kwargs: None,
            PositionalArgumentsFormatter = lambda *args, **kwargs: None,
            ProcessorFormatter = types.SimpleNamespace(
                wrap_for_formatter = lambda *args, **kwargs: None,
            ),
        ),
    )
    sys.modules["structlog"] = structlog_stub


_install_structlog_stub()
