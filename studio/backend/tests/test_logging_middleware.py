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
