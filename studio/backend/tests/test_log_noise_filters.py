# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Log-noise suppression: scanner-request skip, uvicorn h11 drop-filter,
library quieting, and the --verbose / LOG_LEVEL=DEBUG escape hatch."""

import asyncio
import logging

import pytest

from loggers import handlers as hmod
from loggers.config import LogConfig, logs_verbose, _NOISY_LIBS
from loggers.handlers import LoggingMiddleware
from run import _install_uvicorn_startup_log_rewrite


class _LogCapture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append(("info", event, kw))

    def error(self, event, **kw):
        self.events.append(("error", event, kw))


@pytest.fixture
def logs(monkeypatch):
    capture = _LogCapture()
    monkeypatch.setattr(hmod, "logger", capture)
    return capture


@pytest.fixture(autouse=True)
def _not_verbose(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_VERBOSE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def _scope(path, method="GET"):
    return {"type": "http", "path": path, "method": method}


async def _noop_receive():
    return {"type": "http.disconnect"}


def _run(coro):
    return asyncio.run(coro)


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})


async def _send(message):
    pass


# ── scanner request skip (B3) ──────────────────────────────────────────

@pytest.mark.parametrize("scope", [
    _scope("www.baidu.com:443", method="CONNECT"),
    _scope("*", method="PRI"),
    _scope("/", method="FOOBAR"),
    _scope("http://example.com/", method="GET"),  # absolute-form
])
def test_scanner_requests_are_not_logged(logs, scope):
    _run(LoggingMiddleware(_ok_app)(scope, _noop_receive, _send))
    assert logs.events == []


def test_normal_404_is_still_logged(logs):
    _run(LoggingMiddleware(_ok_app)(_scope("/api/does-not-exist"), _noop_receive, _send))
    assert len(logs.events) == 1
    assert logs.events[0][1] == "request_completed"
    assert logs.events[0][2]["status_code"] == 404


def test_verbose_keeps_scanner_requests(logs, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_VERBOSE", "1")
    _run(LoggingMiddleware(_ok_app)(_scope("www.baidu.com:443", method="CONNECT"),
                                    _noop_receive, _send))
    assert len(logs.events) == 1
    assert logs.events[0][2]["method"] == "CONNECT"


# ── verbose helper (B1/B2) ─────────────────────────────────────────────

def test_logs_verbose_env_and_debug(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_VERBOSE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    assert logs_verbose() is False
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert logs_verbose() is True
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("UNSLOTH_STUDIO_VERBOSE", "1")
    assert logs_verbose() is True


# ── uvicorn h11 drop-filter (B2) ───────────────────────────────────────

def _uvicorn_record(msg):
    return logging.LogRecord("uvicorn.error", logging.WARNING, __file__, 0, msg, None, None)


def test_uvicorn_drop_filter_drops_invalid_http(monkeypatch):
    log = logging.getLogger("uvicorn.error")
    log.filters = []
    try:
        _install_uvicorn_startup_log_rewrite("127.0.0.1", "127.0.0.1")
        # filter() returns False to drop, else the (truthy) record.
        assert log.filter(_uvicorn_record("Invalid HTTP request received")) is False
        # real warnings/errors pass through
        assert log.filter(_uvicorn_record("Worker failed to boot"))
    finally:
        log.filters = []


def test_uvicorn_drop_filter_verbose_keeps_all(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_VERBOSE", "1")
    log = logging.getLogger("uvicorn.error")
    log.filters = []
    try:
        _install_uvicorn_startup_log_rewrite("127.0.0.1", "127.0.0.1")
        assert log.filter(_uvicorn_record("Invalid HTTP request received"))
    finally:
        log.filters = []


# ── library quieting (B1) ──────────────────────────────────────────────

def test_setup_logging_quiets_libraries(monkeypatch):
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    monkeypatch.delenv("UNSLOTH_STUDIO_VERBOSE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    LogConfig.setup_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("transformers").level == logging.WARNING


def test_setup_logging_verbose_does_not_quiet(monkeypatch):
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    monkeypatch.setenv("UNSLOTH_STUDIO_VERBOSE", "1")
    LogConfig.setup_logging()
    assert logging.getLogger("httpx").level == logging.NOTSET
