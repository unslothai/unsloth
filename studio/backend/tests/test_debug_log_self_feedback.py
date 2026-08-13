# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The log viewer reads the file the access logger writes to.

That is a feedback loop, and it is the one failure mode this feature can create
on its own: if a poll logs a line, the next poll reads that line back and logs
another, and the log the user opened the viewer to read fills with the viewer.
So this exercises the real middleware over a real file rather than asserting on
the contents of a set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.settings as settings_route
from loggers.handlers import LoggingMiddleware, _is_quiet_success

POLL_PATHS = ("/api/settings/debug/logs", "/api/settings/debug/logs/sources")


@pytest.fixture
def session_log(tmp_path, monkeypatch):
    """A studio home whose server log is also where structlog writes.

    run.py tees stdout into that file, so a record the middleware emits lands in
    the file the viewer is reading. Reproduced here by pointing the logger
    factory straight at it.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    directory = tmp_path / "logs" / "server"
    directory.mkdir(parents = True)
    path = directory / f"server-20260813-120000-pid{os.getpid()}.log"
    handle = path.open("w", encoding = "utf-8", buffering = 1)

    previous = structlog.get_config()
    structlog.configure(
        processors = [structlog.processors.JSONRenderer()],
        logger_factory = structlog.PrintLoggerFactory(file = handle),
        cache_logger_on_first_use = False,
    )
    monkeypatch.setattr("loggers.handlers.logger", structlog.get_logger("access"))
    try:
        yield path
    finally:
        structlog.configure(**previous)
        handle.close()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route._require_ui_session] = lambda: None
    app.add_middleware(LoggingMiddleware)
    return TestClient(app, raise_server_exceptions = False)


def test_polling_the_viewer_does_not_grow_the_log_it_reads(session_log, client):
    session_log.write_text("a line that was there before the viewer opened\n")
    before = session_log.stat().st_size

    cursor = None
    for _ in range(25):
        params = {"cursor": cursor} if cursor else {}
        cursor = client.get("/api/settings/debug/logs", params = params).json()["cursor"]
        client.get("/api/settings/debug/logs/sources")

    assert session_log.stat().st_size == before
    assert "request_completed" not in session_log.read_text()


def test_the_viewer_only_ever_returns_content_it_did_not_write(session_log, client):
    session_log.write_text("first\n")
    first = client.get("/api/settings/debug/logs").json()
    assert first["lines"] == ["first"]
    for _ in range(10):
        body = client.get("/api/settings/debug/logs", params = {"cursor": first["cursor"]}).json()
        # Every poll after the first has nothing to say. A line here would be
        # the viewer reading its own access record.
        assert body["lines"] == []


@pytest.mark.parametrize("path", POLL_PATHS)
def test_verbose_does_not_lift_the_suppression_for_these_two(path, monkeypatch):
    """--verbose turning the suppressor off is fine everywhere else, because
    everywhere else the extra lines only go to a file. These come back at the
    reader, and --verbose is what someone debugging turns on."""
    monkeypatch.setattr("loggers.handlers._VERBOSE_ACCESS_LOG", True)
    assert _is_quiet_success("GET", path, 200, False) is True


def test_verbose_still_lifts_it_for_an_ordinary_quiet_path(monkeypatch):
    monkeypatch.setattr("loggers.handlers._VERBOSE_ACCESS_LOG", True)
    assert _is_quiet_success("GET", "/api/hub/download-status", 200, False) is False


@pytest.mark.parametrize("path", POLL_PATHS)
def test_a_failure_on_the_viewer_endpoints_still_logs(path):
    # The whole point of the suppression is that a poll carries no signal. A
    # 404 or a 500 does.
    assert _is_quiet_success("GET", path, 404, False) is False
    assert _is_quiet_success("GET", path, 500, False) is False
    assert _is_quiet_success("POST", path, 200, False) is False


def test_the_suppression_is_an_exact_path_match(session_log, client):
    """Neither a prefix nor a suffix of these paths may be silenced by them."""
    for path in ("/api/settings", "/api/settings/debug", "/api/settings/debug/logs/x"):
        assert _is_quiet_success("GET", path, 200, False) is False

    # A neighbouring settings GET that is not on any quiet list still logs, so
    # the suppression cannot have widened to the router.
    before = session_log.stat().st_size
    assert client.get("/api/settings/upload-limit").status_code == 200
    assert "/api/settings/upload-limit" in session_log.read_text()[before:]
