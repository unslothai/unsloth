# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extra coverage for utils.api_concurrency: /v1 mirrors, lifecycle edges
(http.disconnect, app exception, more_body omitted, lifespan bypass), limiter
invariants, reject burst, and ordering against Security/MaxBody."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.api_concurrency import (
    AsyncConcurrencyLimiter,
    InferenceConcurrencyMiddleware,
    is_limited_inference_request,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


async def _call_asgi(
    app,
    *,
    path: str = "/v1/chat/completions",
    method: str = "POST",
    headers: List[Tuple[bytes, bytes]] | None = None,
    body: bytes = b"",
) -> Dict[str, Any]:
    sent: List[Dict[str, Any]] = []
    received = {"sent": False}

    async def receive():
        if not received["sent"]:
            received["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8888),
        "scheme": "http",
    }
    await app(scope, receive, send)
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    return {
        "messages": sent,
        "status": start["status"] if start else None,
        "headers": dict(start["headers"]) if start else {},
        "body": b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        ),
    }


# ----------------------------------------------------------------------
# Endpoint allowlist: /v1 mirrors of inference routes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/generate/stream",
        "/v1/audio/generate",
        # Regression anchors for the existing entries.
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/responses",
        "/api/inference/generate/stream",
        "/api/inference/audio/generate",
    ],
)
def test_post_inference_path_is_gated(path):
    assert is_limited_inference_request(
        {"type": "http", "method": "POST", "path": path}
    )


@pytest.mark.parametrize(
    "path",
    [
        # Management / status: not gated.
        "/api/inference/load",
        "/api/inference/unload",
        "/api/inference/validate",
        "/api/inference/status",
        "/api/inference/cancel",
        # Embeddings: intentionally excluded (cheaper than generation).
        "/api/inference/embeddings",
        "/v1/embeddings",
        # Unrelated.
        "/api/health",
        "/v1/models",
    ],
)
def test_excluded_path_is_not_gated(path):
    assert not is_limited_inference_request(
        {"type": "http", "method": "POST", "path": path}
    )


def test_overmatch_negative():
    # /v1/responses_other must not match the /v1/responses prefix.
    assert not is_limited_inference_request(
        {"type": "http", "method": "POST", "path": "/v1/responses_other"}
    )


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
def test_non_http_scope_not_gated(scope_type):
    assert not is_limited_inference_request(
        {"type": scope_type, "method": "POST", "path": "/v1/chat/completions"}
    )


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_non_post_method_not_gated(method):
    assert not is_limited_inference_request(
        {"type": "http", "method": method, "path": "/v1/chat/completions"}
    )


# ----------------------------------------------------------------------
# Limiter primitive: reject correctness, cancel-in-wait, multiple drains
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limiter_reject_does_not_mutate_active():
    lim = AsyncConcurrencyLimiter(1)
    assert await lim.acquire(wait = True) is True
    assert lim.active == 1
    assert await lim.acquire(wait = False) is False
    assert lim.active == 1
    await lim.release()
    assert lim.active == 0


@pytest.mark.asyncio
async def test_limiter_cancel_while_waiting_does_not_leak():
    lim = AsyncConcurrencyLimiter(1)
    await lim.acquire(wait = True)
    task = asyncio.create_task(lim.acquire(wait = True))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lim.active == 1, "cancelled wait must not mutate _active"
    await lim.release()
    assert lim.active == 0
    # Limiter is still usable.
    assert await lim.acquire(wait = True) is True
    await lim.release()


@pytest.mark.asyncio
async def test_limiter_multiple_waiters_all_drain():
    lim = AsyncConcurrencyLimiter(1)
    await lim.acquire(wait = True)

    async def wait_then_done(i, results):
        await lim.acquire(wait = True)
        results.append(i)
        await asyncio.sleep(0.01)
        await lim.release()

    results: List[int] = []
    waiters = [asyncio.create_task(wait_then_done(i, results)) for i in range(5)]
    await asyncio.sleep(0.05)
    assert results == []
    await lim.release()
    await asyncio.gather(*waiters)
    assert sorted(results) == [0, 1, 2, 3, 4]
    assert lim.active == 0


# ----------------------------------------------------------------------
# Middleware lifecycle gaps not covered by the base test file
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_more_body_omitted_treated_as_final_chunk():
    """ASGI spec: missing `more_body` defaults to False, so the slot must
    release after the body event even when the app omits the key."""

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        # NOTE: no `more_body` key.
        await send({"type": "http.response.body", "body": b"done"})

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)
    r = await _call_asgi(mw)
    assert r["status"] == 200
    assert r["body"] == b"done"
    assert mw._limiter.active == 0
    # Subsequent request can acquire.
    r2 = await _call_asgi(mw)
    assert r2["status"] == 200


@pytest.mark.asyncio
async def test_app_exception_releases_slot():
    async def inner(scope, receive, send):
        raise RuntimeError("boom")

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)
    with pytest.raises(RuntimeError):
        await _call_asgi(mw)
    assert mw._limiter.active == 0
    # Limiter recovers; next request can run.
    with pytest.raises(RuntimeError):
        await _call_asgi(mw)


@pytest.mark.asyncio
async def test_streaming_exception_releases_slot():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x", "more_body": True})
        raise RuntimeError("stream-fail")

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)
    with pytest.raises(RuntimeError):
        await _call_asgi(mw)
    assert mw._limiter.active == 0


@pytest.mark.asyncio
async def test_client_disconnect_during_stream_releases_slot():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        msg = await receive()
        if msg.get("type") == "http.disconnect":
            return
        await send({"type": "http.response.body", "body": b"b", "more_body": False})

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)

    sent: List[Dict[str, Any]] = []

    async def send_to_capture(msg):
        sent.append(msg)

    async def receive_disconnect():
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }
    await mw(scope, receive_disconnect, send_to_capture)
    assert mw._limiter.active == 0


@pytest.mark.asyncio
async def test_release_idempotent_on_duplicate_final_chunks():
    """Defensive: if an app accidentally sends two final body events the
    underlying _active counter must not go negative."""

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": False})
        await send({"type": "http.response.body", "body": b"b", "more_body": False})

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)
    r = await _call_asgi(mw)
    assert r["body"] == b"ab"
    assert mw._limiter._active == 0


@pytest.mark.asyncio
async def test_lifespan_scope_bypasses_limiter():
    forwarded = {"startup": False, "shutdown": False}

    async def inner(scope, receive, send):
        assert scope["type"] == "lifespan"
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                forwarded["startup"] = True
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                forwarded["shutdown"] = True
                await send({"type": "lifespan.shutdown.complete"})
                return

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1)
    incoming: asyncio.Queue = asyncio.Queue()
    outgoing: asyncio.Queue = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(msg):
        await outgoing.put(msg)

    task = asyncio.create_task(mw({"type": "lifespan"}, receive, send))
    await incoming.put({"type": "lifespan.startup"})
    assert (await outgoing.get())["type"] == "lifespan.startup.complete"
    await incoming.put({"type": "lifespan.shutdown"})
    assert (await outgoing.get())["type"] == "lifespan.shutdown.complete"
    await task
    assert forwarded == {"startup": True, "shutdown": True}
    assert mw._limiter.active == 0


# ----------------------------------------------------------------------
# Reject-burst invariant
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reject_burst_invariant():
    """With max_concurrency=1 and policy=reject, exactly one of N concurrent
    requests reaches the inner app; the rest receive 429."""
    held = asyncio.Event()
    inner_calls = {"count": 0}

    async def inner(scope, receive, send):
        inner_calls["count"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await held.wait()
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1, queue_policy = "reject")
    holder = asyncio.create_task(_call_asgi(mw))
    await asyncio.sleep(0.05)
    rejected = await asyncio.gather(*[_call_asgi(mw) for _ in range(9)])
    assert all(r["status"] == 429 for r in rejected)
    held.set()
    r = await holder
    assert r["status"] == 200
    assert inner_calls["count"] == 1, "rejected requests must not invoke the inner app"


@pytest.mark.asyncio
async def test_reject_429_response_shape():
    held = asyncio.Event()

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await held.wait()
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 1, queue_policy = "reject")
    holder = asyncio.create_task(_call_asgi(mw))
    await asyncio.sleep(0.05)
    r = await _call_asgi(mw)
    assert r["status"] == 429
    headers = r["headers"]
    assert headers[b"content-type"] == b"application/json"
    assert headers[b"content-length"] == str(len(r["body"])).encode("ascii")
    assert headers[b"retry-after"] == b"1"
    payload = json.loads(r["body"])
    assert payload["error"]["code"] == "max_concurrency_exceeded"
    assert payload["error"]["type"] == "rate_limit_exceeded"
    held.set()
    await holder


# ----------------------------------------------------------------------
# Wait policy parallelism
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_policy_allows_up_to_max():
    """With max_concurrency=3, never more than 3 inflight under contention."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def inner(scope, receive, send):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.03)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})
        async with lock:
            in_flight -= 1

    mw = InferenceConcurrencyMiddleware(inner, max_concurrency = 3, queue_policy = "wait")
    results = await asyncio.gather(*[_call_asgi(mw) for _ in range(8)])
    assert all(r["status"] == 200 for r in results)
    assert peak == 3, f"expected peak=3 under max=3, got {peak}"


# ----------------------------------------------------------------------
# Middleware ordering against MaxBody + SecurityHeaders (uses main.py)
# ----------------------------------------------------------------------


@pytest.fixture(scope = "module")
def main_module():
    import main as _main  # noqa: F401

    return _main


def _build_chat_app(
    main_module, *, max_concurrency: int, queue_policy: str, max_body: int
):
    """Mirror the production middleware registration order from main.py with a
    minimal /v1/chat/completions and /api/inference/load route."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    # NOTE: must match main.py's add_middleware order to be a faithful test.
    app.add_middleware(main_module.SecurityHeadersMiddleware)
    app.add_middleware(
        main_module.InferenceConcurrencyMiddleware,
        max_concurrency = max_concurrency,
        queue_policy = queue_policy,
    )
    app.add_middleware(
        main_module.MaxBodyMiddleware,
        max_bytes = max_body,
        protected_prefixes = ("/v1/chat/completions", "/api/inference"),
    )

    slow_event = asyncio.Event()
    app.state.slow_event = slow_event

    @app.post("/v1/chat/completions")
    async def chat(payload: dict):
        return JSONResponse({"ok": True, "n": len(payload.get("text", ""))})

    @app.post("/api/inference/load")
    async def load(payload: dict):
        return JSONResponse({"loaded": True})

    return app


@pytest.mark.asyncio
async def test_429_carries_security_headers(main_module):
    """429 must carry CSP and X-Content-Type-Options (Security wraps the gate)."""
    holding = asyncio.Event()

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    app.add_middleware(
        main_module.InferenceConcurrencyMiddleware,
        max_concurrency = 1,
        queue_policy = "reject",
    )
    app.add_middleware(main_module.SecurityHeadersMiddleware)

    @app.post("/v1/chat/completions")
    async def chat():
        await holding.wait()
        return JSONResponse({"ok": True})

    async with httpx.AsyncClient(
        transport = httpx.ASGITransport(app = app), base_url = "http://t"
    ) as client:
        holder = asyncio.create_task(client.post("/v1/chat/completions", json = {}))
        await asyncio.sleep(0.05)
        r = await client.post("/v1/chat/completions", json = {})
        holding.set()
        await holder

    assert r.status_code == 429
    header_keys = {k.lower() for k in r.headers.keys()}
    assert "content-security-policy" in header_keys
    assert r.headers.get("x-content-type-options") == "nosniff"


@pytest.mark.asyncio
async def test_413_does_not_hold_concurrency_slot(main_module):
    """Oversized POST must 413 before the gate acquires a slot, so a slow
    uploader cannot block other inference. Driven via a slow chunked receive."""
    inner_calls = {"count": 0}

    async def inner(scope, receive, send):
        inner_calls["count"] += 1
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            if not msg.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    # Production order (after the PR's fix): gate INSIDE MaxBody.
    gate = main_module.InferenceConcurrencyMiddleware(
        inner, max_concurrency = 1, queue_policy = "reject"
    )
    app = main_module.MaxBodyMiddleware(
        gate, max_bytes = 16, protected_prefixes = ("/v1/chat/completions",)
    )

    body_done = asyncio.Event()
    peak_active = {"value": 0}

    async def watcher():
        while not body_done.is_set():
            peak_active["value"] = max(peak_active["value"], gate._limiter.active)
            await asyncio.sleep(0.001)

    chunks = [b"x" * 4] * 8  # 32 bytes, > 16 byte cap
    idx = {"value": 0}

    async def slow_receive():
        if idx["value"] < len(chunks):
            await asyncio.sleep(0.005)
            body = chunks[idx["value"]]
            idx["value"] += 1
            return {
                "type": "http.request",
                "body": body,
                "more_body": idx["value"] < len(chunks),
            }
        return {"type": "http.disconnect"}

    sent: List[Dict[str, Any]] = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }
    w = asyncio.create_task(watcher())
    try:
        await app(scope, slow_receive, send)
    finally:
        body_done.set()
        await w

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 413
    assert inner_calls["count"] == 0, "413 must short-circuit before the inner app"
    assert (
        peak_active["value"] == 0
    ), f"413 path must never acquire a concurrency slot; peak={peak_active['value']}"


@pytest.mark.asyncio
async def test_excluded_load_path_not_gated_by_full_stack(main_module):
    """Excluded /api/inference/load must respond while the gate holds a slot."""
    holding = asyncio.Event()

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()
    app.add_middleware(main_module.SecurityHeadersMiddleware)
    app.add_middleware(
        main_module.InferenceConcurrencyMiddleware,
        max_concurrency = 1,
        queue_policy = "wait",
    )

    @app.post("/v1/chat/completions")
    async def chat():
        await holding.wait()
        return JSONResponse({"ok": True})

    @app.post("/api/inference/load")
    async def load():
        return JSONResponse({"loaded": True})

    async with httpx.AsyncClient(
        transport = httpx.ASGITransport(app = app), base_url = "http://t"
    ) as client:
        holder = asyncio.create_task(client.post("/v1/chat/completions", json = {}))
        await asyncio.sleep(0.05)
        r = await client.post("/api/inference/load", json = {})
        assert r.status_code == 200
        assert r.json() == {"loaded": True}
        holding.set()
        await holder
