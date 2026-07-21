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

    _run(
        LoggingMiddleware(app)(
            {"type": "websocket", "path": "/ws"}, _noop_receive, send
        )
    )

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
        _run(mw(_http_scope("/api/models/browse-folders"), _noop_receive, send))

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
        _run(
            mw(_http_scope("/api/models/browse-folders"), _noop_receive, send)
        )  # normal

    paths = [e[2]["path"] for e in logs.events]
    assert paths.count("/api/inference/monitor") == 1  # collapsed to one heartbeat
    assert (
        paths.count("/api/models/browse-folders") == 3
    )  # base dedup off -> all logged


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


def _status_app(status):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


async def _drop(message):
    pass


def _paths_logged(logs):
    return [e[2]["path"] for e in logs.events]


def test_quiet_success_get_2xx_suppressed(logs):
    # A GET/2xx poll on a quiet-success path logs nothing; the signal is in events.
    for path in ("/api/chat/threads", "/api/export/status", "/api/hub/download-status"):
        _run(
            LoggingMiddleware(_status_app(200))(_http_scope(path), _noop_receive, _drop)
        )
    assert logs.events == []


def test_chat_detail_and_message_reads_still_log(logs):
    # Only the exact list polls are suppressed; detail/message reads carry latency
    # signal and keep their access line.
    for path in (
        "/api/chat/threads/abc123",
        "/api/chat/threads/abc123/messages",
        "/api/chat/threads/abc123/messages/m1",
        "/api/chat/projects/p1",
    ):
        _run(
            LoggingMiddleware(_status_app(200))(_http_scope(path), _noop_receive, _drop)
        )
    assert _paths_logged(logs) == [
        "/api/chat/threads/abc123",
        "/api/chat/threads/abc123/messages",
        "/api/chat/threads/abc123/messages/m1",
        "/api/chat/projects/p1",
    ]


def test_quiet_success_is_get_only(logs):
    # Mutations on the same paths still log (suppression is GET-only).
    for method in ("POST", "PUT", "DELETE"):
        _run(
            LoggingMiddleware(_status_app(200))(
                _http_scope("/api/chat/threads", method = method), _noop_receive, _drop
            )
        )
    assert len(logs.events) == 3


def test_chat_pre_auth_401_suppressed_other_errors_logged(logs):
    # The transient bootstrap 401 on a chat list GET is dropped, but a 500 (or any
    # other status) still logs so real failures stay visible.
    _run(
        LoggingMiddleware(_status_app(401))(
            _http_scope("/api/chat/projects"), _noop_receive, _drop
        )
    )
    assert logs.events == []
    _run(
        LoggingMiddleware(_status_app(500))(
            _http_scope("/api/chat/projects"), _noop_receive, _drop
        )
    )
    assert _paths_logged(logs) == ["/api/chat/projects"]


def test_chat_401_logged_after_first_auth_refresh(logs):
    # A chat 401 before any successful token refresh is the bootstrap race and is
    # dropped, but once /api/auth/refresh has succeeded on this instance later chat
    # 401s are real failures and stay visible.
    responses: dict[tuple[str, str], int] = {}

    async def app(scope, receive, send):
        status = responses.get((scope["method"], scope["path"]), 200)
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = LoggingMiddleware(app)

    responses[("GET", "/api/chat/threads")] = 401
    _run(mw(_http_scope("/api/chat/threads"), _noop_receive, _drop))
    assert logs.events == []  # bootstrap race: suppressed

    # A successful refresh (POST, always logged) closes the bootstrap window.
    responses[("POST", "/api/auth/refresh")] = 200
    _run(mw(_http_scope("/api/auth/refresh", method = "POST"), _noop_receive, _drop))
    assert _paths_logged(logs) == ["/api/auth/refresh"]

    # Now the same chat 401 is a real failure and logs.
    _run(mw(_http_scope("/api/chat/threads"), _noop_receive, _drop))
    assert _paths_logged(logs) == ["/api/auth/refresh", "/api/chat/threads"]


def test_export_status_error_still_logs(logs):
    # 2xx suppressed, but an HTTP-level error on export status remains visible.
    _run(
        LoggingMiddleware(_status_app(200))(
            _http_scope("/api/export/status"), _noop_receive, _drop
        )
    )
    assert logs.events == []
    _run(
        LoggingMiddleware(_status_app(500))(
            _http_scope("/api/export/status"), _noop_receive, _drop
        )
    )
    assert _paths_logged(logs) == ["/api/export/status"]


def test_legacy_download_progress_heartbeats_not_suppressed(logs, monkeypatch):
    # Legacy /api/models download polls emit no progress events, so they heartbeat
    # (first hit logs, the burst collapses) rather than vanish entirely.
    monkeypatch.setattr(hmod, "_ACCESS_LOG_DEDUP_MS", 0)
    monkeypatch.setattr(hmod, "_QUIET_POLL_DEDUP_MS", 1000)
    mw = LoggingMiddleware(_status_app(200))
    for _ in range(3):
        _run(mw(_http_scope("/api/models/download-progress"), _noop_receive, _drop))
    assert _paths_logged(logs) == ["/api/models/download-progress"]
