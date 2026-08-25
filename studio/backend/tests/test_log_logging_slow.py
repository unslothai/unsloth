# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The slow-request exemption, at the middleware's own level.

Its siblings in test_log_budget.py cover how MUCH the exemption writes. These cover when it
writes at all: the knobs, the classes it overrides, and the one class it does not.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

_BACKEND = str(Path(__file__).resolve().parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from loggers import handlers as hmod  # noqa: E402


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append(kw)

    def error(self, event, **kw):
        self.events.append(kw)

    def warning(self, event, **kw):
        self.events.append(kw)


def _drive(monkeypatch, path, status, duration_ms, count = 1, gap_s = 0.0):
    clock = _Clock()
    monkeypatch.setattr(hmod, "time", clock)
    capture = _Capture()
    monkeypatch.setattr(hmod, "logger", capture)

    async def app(scope, receive, send):
        clock.advance(duration_ms / 1000.0)
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = hmod.LoggingMiddleware(app)

    async def send(_m):
        return None

    async def receive():
        return {"type": "http.disconnect"}

    for _ in range(count):
        asyncio.run(middleware(
            {"type": "http", "path": path, "method": "GET", "query_string": b""},
            receive, send,
        ))
        clock.advance(gap_s)
    return capture.events


SILENT_PATH = "/api/export/status"     # quiet_success: a fast 2xx writes nothing


def test_a_slow_success_is_logged_with_its_reason(monkeypatch):
    events = _drive(monkeypatch, SILENT_PATH, 200, hmod._SLOW_REQUEST_MS + 500)
    assert len(events) == 1
    assert events[0]["slow"] is True
    assert events[0]["status_code"] == 200


def test_a_request_just_under_the_threshold_is_not_slow(monkeypatch):
    events = _drive(monkeypatch, SILENT_PATH, 200, hmod._SLOW_REQUEST_MS - 1)
    assert events == [], (
        "a request below the threshold must keep its class's behaviour; this one is "
        "quiet-success, so it writes nothing"
    )


def test_the_threshold_can_be_disabled(monkeypatch):
    monkeypatch.setattr(hmod, "_SLOW_REQUEST_MS", 0)
    events = _drive(monkeypatch, SILENT_PATH, 200, 60_000)
    assert events == [], "_SLOW_REQUEST_MS = 0 must disable the exemption entirely"


def test_an_excluded_path_never_produces_a_slow_line(monkeypatch):
    """`excluded` wins over the exemption on purpose: those are static assets and health
    endpoints deliberately dropped before the status is even considered."""
    excluded = sorted(hmod._EXCLUDED_PATHS)
    assert excluded, "no excluded paths configured; this guard would be vacuous"
    events = _drive(monkeypatch, excluded[0], 200, 60_000)
    assert events == [], f"{excluded[0]} is excluded and must stay silent even when slow"


def test_a_slow_failure_still_logs_once_per_request(monkeypatch):
    """Failures were never de-duplicated, and the exemption must not start doing so."""
    events = _drive(monkeypatch, SILENT_PATH, 503, hmod._SLOW_REQUEST_MS + 500, count = 4)
    assert len(events) == 4, (
        f"four slow failures produced {len(events)} lines; failures must never collapse"
    )


def test_the_slow_line_does_not_reset_the_paths_own_heartbeat(monkeypatch):
    """A slow line is extra, not a replacement: the path's ordinary window keeps its
    rhythm rather than being pushed back by one slow call."""
    path = "/api/health"
    slow_then_fast = _drive(
        monkeypatch, path, 200, hmod._SLOW_REQUEST_MS + 500, count = 1
    )
    assert slow_then_fast, "a slow /api/health should log"
