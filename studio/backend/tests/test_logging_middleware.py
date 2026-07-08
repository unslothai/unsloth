# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.staticfiles import StaticFiles

from loggers import handlers as hmod
from loggers.handlers import LoggingMiddleware


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


def _run(coro):
    return asyncio.run(coro)


def test_success_logs_status_and_forwards_chunks(logs):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 206, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    seen = []

    async def send(message):
        seen.append(message)

    _run(LoggingMiddleware(app)(_http_scope("/api/health"), _noop_receive, send))

    assert [m["type"] for m in seen] == [
        "http.response.start",
        "http.response.body",
        "http.response.body",
    ]
    assert logs.events[0][1] == "request_completed"
    assert logs.events[0][2]["status_code"] == 206


def test_excluded_asset_success_skips_log(logs):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    for path in ("/assets/index.css", "/huggingface.svg", "/font.woff2"):
        _run(LoggingMiddleware(app)(_http_scope(path), _noop_receive, send))

    assert logs.events == []


def test_exception_logs_real_status_and_reraises(logs):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 418, "headers": []})
        raise RuntimeError("stream failed")

    async def send(message):
        pass

    with pytest.raises(RuntimeError, match = "stream failed"):
        _run(LoggingMiddleware(app)(_http_scope("/api/health"), _noop_receive, send))

    assert logs.events[0][1] == "request_failed"
    assert logs.events[0][2]["status_code"] == 418
    assert logs.events[0][2]["error"] == "stream failed"
    assert "process_time_ms" in logs.events[0][2]


def test_cancelled_error_propagates_without_error_log(logs):
    async def app(scope, receive, send):
        raise asyncio.CancelledError()

    async def send(message):
        pass

    with pytest.raises(asyncio.CancelledError):
        _run(LoggingMiddleware(app)(_http_scope("/api/health"), _noop_receive, send))

    assert logs.events == []


def test_non_http_scope_passes_through(logs):
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    async def send(message):
        pass

    _run(LoggingMiddleware(app)({"type": "websocket", "path": "/ws"}, _noop_receive, send))

    assert seen == ["websocket"]
    assert logs.events == []


def test_duplicate_get_within_window_deduped(logs, monkeypatch):
    monkeypatch.setattr(hmod, "_ACCESS_LOG_DEDUP_MS", 1000)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    mw = LoggingMiddleware(app)
    for _ in range(3):
        _run(mw(_http_scope("/api/chat/projects"), _noop_receive, send))

    # Only the first of the identical GET/200 burst is logged.
    assert len(logs.events) == 1
    assert logs.events[0][1] == "request_completed"


def test_mutations_and_errors_are_never_deduped(logs, monkeypatch):
    monkeypatch.setattr(hmod, "_ACCESS_LOG_DEDUP_MS", 1000)

    async def post_ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def get_404(scope, receive, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message):
        pass

    mw = LoggingMiddleware(post_ok)
    for _ in range(2):
        _run(mw(_http_scope("/api/chat/threads", method = "POST"), _noop_receive, send))
    mw_404 = LoggingMiddleware(get_404)
    for _ in range(2):
        _run(mw_404(_http_scope("/api/models"), _noop_receive, send))

    # 2 mutations + 2 errors all logged (dedup only touches GET/2xx).
    assert len(logs.events) == 4


def test_quiet_poll_paths_use_longer_heartbeat_window(logs, monkeypatch):
    # Burst dedup off, quiet-poll heartbeat on: only liveness paths collapse.
    monkeypatch.setattr(hmod, "_ACCESS_LOG_DEDUP_MS", 0)
    monkeypatch.setattr(hmod, "_QUIET_POLL_DEDUP_MS", 1000)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    mw = LoggingMiddleware(app)
    for _ in range(3):
        _run(mw(_http_scope("/api/inference/monitor"), _noop_receive, send))  # quiet
    for _ in range(3):
        _run(mw(_http_scope("/api/chat/projects"), _noop_receive, send))  # normal

    paths = [e[2]["path"] for e in logs.events]
    assert paths.count("/api/inference/monitor") == 1  # collapsed to one heartbeat
    assert paths.count("/api/chat/projects") == 3  # base dedup off -> all logged


def test_distinct_query_strings_are_not_deduped(logs, monkeypatch):
    monkeypatch.setattr(hmod, "_ACCESS_LOG_DEDUP_MS", 1000)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        pass

    def scope(query):
        return {
            "type": "http",
            "path": "/api/models/browse-folders",
            "method": "GET",
            "query_string": query,
        }

    mw = LoggingMiddleware(app)
    _run(mw(scope(b"path=/tmp/a"), _noop_receive, send))
    _run(mw(scope(b"path=/tmp/b"), _noop_receive, send))  # distinct query -> logs
    _run(mw(scope(b"path=/tmp/a"), _noop_receive, send))  # repeat of first -> deduped

    # Two distinct query strings log; the immediate repeat of the first does not.
    assert len(logs.events) == 2


def test_fastapi_static_asset_success_skips_log(tmp_path, logs):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.css").write_text("body { color: black; }", encoding = "utf-8")

    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    app.mount("/assets", StaticFiles(directory = assets_dir), name = "assets")
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    assert logs.events[0][1] == "request_completed"
    assert logs.events[0][2]["path"] == "/api/health"

    log_count = len(logs.events)
    response = client.get("/assets/app.css")
    assert response.status_code == 200
    assert response.text == "body { color: black; }"
    assert len(logs.events) == log_count
