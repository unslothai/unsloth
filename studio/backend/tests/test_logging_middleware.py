# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavior tests for the pure-ASGI LoggingMiddleware.

Covers exclusion semantics (paths/suffixes/prefix), failure logging on
excluded paths, status-code capture across the wrapped send, success-log
suppression, the wrapper's transparency to the inner app's send messages,
and pass-through of non-HTTP scopes.
"""

import asyncio

import pytest

from loggers import handlers as hmod
from loggers.handlers import (
    LoggingMiddleware,
    _EXCLUDED_PATHS,
    _EXCLUDED_SUFFIXES,
)


class _LogCapture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append(("info", event, kw))

    def error(self, event, **kw):
        self.events.append(("error", event, kw))

    def reset(self):
        self.events = []


@pytest.fixture
def caplog(monkeypatch):
    cap = _LogCapture()
    monkeypatch.setattr(hmod, "logger", cap)
    return cap


def _http_scope(path, method = "GET"):
    return {"type": "http", "path": path, "method": method}


async def _noop_receive():
    return {"type": "http.disconnect"}


def _run(coro):
    return asyncio.run(coro)


def _ok_messages():
    return [
        {"type": "http.response.start", "status": 200, "headers": []},
        {"type": "http.response.body", "body": b""},
    ]


def _make_inner(messages = None, exc = None):
    async def inner(scope, receive, send):
        for msg in messages or []:
            await send(msg)
        if exc is not None:
            raise exc

    return inner


# ── exclusion list contents ──────────────────────────────────────────


def test_svg_is_in_excluded_suffixes():
    assert ".svg" in _EXCLUDED_SUFFIXES


# ── success-log suppression ─────────────────────────────────────────


def test_excluded_path_success_emits_no_log(caplog):
    mw = LoggingMiddleware(_make_inner(messages=_ok_messages()))

    async def sink(_msg):
        pass

    _run(mw(_http_scope("/api/system"), _noop_receive, sink))
    assert caplog.events == []


def test_all_excluded_paths_skip_success_log(caplog):
    for path in _EXCLUDED_PATHS:
        caplog.reset()
        mw = LoggingMiddleware(_make_inner(messages=_ok_messages()))

        async def sink(_msg):
            pass

        _run(mw(_http_scope(path), _noop_receive, sink))
        assert caplog.events == [], f"excluded path {path} logged: {caplog.events}"


def test_excluded_suffix_success_skips_logging(caplog):
    for path in ("/huggingface.svg", "/favicon.png", "/foo.woff2"):
        caplog.reset()
        mw = LoggingMiddleware(_make_inner(messages=_ok_messages()))

        async def sink(_msg):
            pass

        _run(mw(_http_scope(path), _noop_receive, sink))
        assert caplog.events == [], f"path {path} unexpectedly logged: {caplog.events}"


def test_assets_prefix_success_skips_logging(caplog):
    mw = LoggingMiddleware(_make_inner(messages=_ok_messages()))

    async def sink(_msg):
        pass

    _run(mw(_http_scope("/assets/index-abc.css"), _noop_receive, sink))
    assert caplog.events == []


# ── failure logging across the exclusion boundary ────────────────────


def test_excluded_path_exception_still_emits_request_failed(caplog):
    mw = LoggingMiddleware(_make_inner(exc=RuntimeError("boom")))

    async def sink(_msg):
        pass

    with pytest.raises(RuntimeError):
        _run(mw(_http_scope("/api/system"), _noop_receive, sink))

    err = next(e for e in caplog.events if e[0] == "error")
    assert err[1] == "request_failed"
    assert err[2]["path"] == "/api/system"
    assert err[2]["error"] == "boom"
    assert "process_time_ms" in err[2]


def test_excluded_path_pre_response_exception_keeps_default_status(caplog):
    mw = LoggingMiddleware(_make_inner(exc=ValueError("early")))

    async def sink(_msg):
        pass

    with pytest.raises(ValueError):
        _run(mw(_http_scope("/api/train/status"), _noop_receive, sink))

    err = next(e for e in caplog.events if e[0] == "error")
    assert err[2]["status_code"] == 500
    assert err[2]["error"] == "early"


def test_excluded_path_streaming_then_raise_logs_real_status(caplog):
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 206, "headers": []})
        raise RuntimeError("mid-stream")

    mw = LoggingMiddleware(inner)
    captured = []

    async def real_send(msg):
        captured.append(msg)

    with pytest.raises(RuntimeError):
        _run(mw(_http_scope("/api/system"), _noop_receive, real_send))

    assert captured[0]["type"] == "http.response.start"
    assert captured[0]["status"] == 206
    err = next(e for e in caplog.events if e[0] == "error")
    assert err[2]["status_code"] == 206


def test_assets_path_streaming_then_raise_logs_real_status(caplog):
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("disconnect")

    mw = LoggingMiddleware(inner)

    async def real_send(_msg):
        pass

    with pytest.raises(RuntimeError):
        _run(mw(_http_scope("/assets/app.css"), _noop_receive, real_send))

    err = next(e for e in caplog.events if e[0] == "error")
    assert err[2]["status_code"] == 200


# ── non-excluded paths ───────────────────────────────────────────────


def test_non_excluded_success_logs_request_completed(caplog):
    inner = _make_inner(messages=[
        {"type": "http.response.start", "status": 201, "headers": []},
        {"type": "http.response.body", "body": b"ok"},
    ])
    mw = LoggingMiddleware(inner)

    async def sink(_msg):
        pass

    _run(mw(_http_scope("/api/health", method="POST"), _noop_receive, sink))

    assert len(caplog.events) == 1
    kind, ev, kw = caplog.events[0]
    assert (kind, ev) == ("info", "request_completed")
    assert kw["path"] == "/api/health"
    assert kw["method"] == "POST"
    assert kw["status_code"] == 201
    assert "process_time_ms" in kw


def test_non_excluded_exception_logs_request_failed_with_timing(caplog):
    mw = LoggingMiddleware(_make_inner(exc=ValueError("nope")))

    async def sink(_msg):
        pass

    with pytest.raises(ValueError):
        _run(mw(_http_scope("/api/health"), _noop_receive, sink))

    assert len(caplog.events) == 1
    kind, ev, kw = caplog.events[0]
    assert (kind, ev) == ("error", "request_failed")
    assert kw["error"] == "nope"
    assert "process_time_ms" in kw
    assert kw["exc_info"] is True


def test_non_excluded_streaming_then_raise_logs_real_status(caplog):
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 418, "headers": []})
        raise RuntimeError("teapot")

    mw = LoggingMiddleware(inner)

    async def real_send(_msg):
        pass

    with pytest.raises(RuntimeError):
        _run(mw(_http_scope("/api/health"), _noop_receive, real_send))

    err = next(e for e in caplog.events if e[0] == "error")
    assert err[2]["status_code"] == 418


# ── inner-app send wiring ────────────────────────────────────────────


def test_excluded_path_inner_app_receives_send_wrapper():
    received = []

    async def inner(scope, receive, send):
        received.append(send)
        for msg in _ok_messages():
            await send(msg)

    mw = LoggingMiddleware(inner)

    async def real_send(_msg):
        pass

    _run(mw(_http_scope("/huggingface.svg"), _noop_receive, real_send))
    assert received and received[0] is not real_send


def test_non_excluded_path_inner_app_receives_send_wrapper():
    received = []

    async def inner(scope, receive, send):
        received.append(send)
        for msg in _ok_messages():
            await send(msg)

    mw = LoggingMiddleware(inner)

    async def real_send(_msg):
        pass

    _run(mw(_http_scope("/api/health"), _noop_receive, real_send))
    assert received and received[0] is not real_send


def test_send_wrapper_forwards_messages_unchanged():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-test", b"1")]})
        for chunk in (b"a", b"bb", b"ccc"):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = LoggingMiddleware(inner)
    seen = []

    async def real_send(msg):
        seen.append(msg)

    _run(mw(_http_scope("/api/health"), _noop_receive, real_send))

    assert seen[0]["type"] == "http.response.start"
    assert seen[0]["status"] == 200
    assert seen[0]["headers"] == [(b"x-test", b"1")]
    body_chunks = [m["body"] for m in seen if m["type"] == "http.response.body"]
    assert body_chunks == [b"a", b"bb", b"ccc", b""]


# ── non-HTTP scopes ──────────────────────────────────────────────────


def test_websocket_scope_passes_through(caplog):
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    mw = LoggingMiddleware(inner)

    async def sink(_msg):
        pass

    _run(mw({"type": "websocket", "path": "/ws"}, _noop_receive, sink))
    assert seen == ["websocket"]
    assert caplog.events == []


def test_lifespan_scope_passes_through(caplog):
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    mw = LoggingMiddleware(inner)

    async def sink(_msg):
        pass

    _run(mw({"type": "lifespan"}, _noop_receive, sink))
    assert seen == ["lifespan"]
    assert caplog.events == []
