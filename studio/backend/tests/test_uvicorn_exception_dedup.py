# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unhandled request exception must be logged once, not twice.

LoggingMiddleware emits request_failed with the full traceback in its structured
"exception" field, then re-raises; uvicorn logs the very same exception again on
stderr as "Exception in ASGI application", and the desktop shell copies every
stderr line into tauri.log separately (~90 lines per failure). The filter drops
uvicorn's copy only for exceptions the middleware already reported, so a failure
raised above the middleware, or a run with --verbose, keeps both.
"""

import asyncio
import logging

import pytest

from loggers import handlers as hmod
from loggers.handlers import (
    LoggingMiddleware,
    _DropDuplicateAsgiException,
    install_uvicorn_duplicate_exception_filter,
)

_UVICORN_MSG = "Exception in ASGI application\n"


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


def _http_scope(path, method = "GET"):
    return {"type": "http", "path": path, "method": method}


async def _noop_receive():
    return {"type": "http.disconnect"}


async def _drop(message):
    pass


def _uvicorn_record(exc, msg = _UVICORN_MSG):
    """The record uvicorn builds: logger.error(msg, exc_info=exc) on uvicorn.error."""
    return logging.LogRecord(
        name = "uvicorn.error",
        level = logging.ERROR,
        pathname = __file__,
        lineno = 1,
        msg = msg,
        args = (),
        exc_info = (type(exc), exc, exc.__traceback__),
    )


def _raise_through_middleware(exc, path = "/api/rag/knowledge-bases"):
    """Run a failing app under the middleware and hand back the exception uvicorn
    would see (the same object, re-raised)."""

    async def app(scope, receive, send):
        raise exc

    with pytest.raises(type(exc)) as caught:
        asyncio.run(LoggingMiddleware(app)(_http_scope(path), _noop_receive, _drop))
    return caught.value


def test_middleware_marks_the_exception_it_logged(logs):
    raised = _raise_through_middleware(RuntimeError("RAG unavailable"))
    assert logs.events[0][1] == "request_failed"
    assert getattr(raised, hmod._LOGGED_EXC_ATTR, False) is True


def test_uvicorn_duplicate_traceback_is_dropped(logs):
    raised = _raise_through_middleware(RuntimeError("RAG unavailable"))
    assert _DropDuplicateAsgiException().filter(_uvicorn_record(raised)) is False


def test_exception_never_seen_by_the_middleware_still_logs():
    # Raised above LoggingMiddleware (CORS, remote-access, the protocol layer): it
    # carries no marker, so uvicorn's traceback is the only record of it and must stay.
    try:
        raise RuntimeError("cors blew up")
    except RuntimeError as exc:
        record = _uvicorn_record(exc)
    assert _DropDuplicateAsgiException().filter(record) is True


def test_other_uvicorn_error_records_pass_through(logs):
    # Only the ASGI-application traceback is a duplicate; every other uvicorn error
    # line is uvicorn's alone.
    raised = _raise_through_middleware(RuntimeError("boom"))
    record = _uvicorn_record(raised, msg = "ASGI callable returned without starting response.")
    assert _DropDuplicateAsgiException().filter(record) is True


def test_record_without_exc_info_passes_through():
    record = logging.LogRecord(
        name = "uvicorn.error",
        level = logging.ERROR,
        pathname = __file__,
        lineno = 1,
        msg = _UVICORN_MSG,
        args = (),
        exc_info = None,
    )
    assert _DropDuplicateAsgiException().filter(record) is True


def test_verbose_keeps_both_copies(logs, monkeypatch):
    raised = _raise_through_middleware(RuntimeError("RAG unavailable"))
    monkeypatch.setattr(hmod, "_VERBOSE_ACCESS_LOG", True)
    assert _DropDuplicateAsgiException().filter(_uvicorn_record(raised)) is True


def test_installed_filter_suppresses_the_record_on_uvicorn_error(logs):
    # End to end through the logging module: the filter is attached to the
    # uvicorn.error logger, so the handler never sees the duplicate.
    uvicorn_logger = logging.getLogger("uvicorn.error")
    before = list(uvicorn_logger.filters)
    seen = []

    class _Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = _Collect()
    uvicorn_logger.addHandler(handler)
    install_uvicorn_duplicate_exception_filter()
    try:
        raised = _raise_through_middleware(RuntimeError("RAG unavailable"))
        uvicorn_logger.error(_UVICORN_MSG, exc_info = raised)
        assert seen == []

        try:
            raise RuntimeError("not ours")
        except RuntimeError as exc:
            uvicorn_logger.error(_UVICORN_MSG, exc_info = exc)
        assert [m.strip() for m in seen] == ["Exception in ASGI application"]
    finally:
        uvicorn_logger.removeHandler(handler)
        uvicorn_logger.filters = before
