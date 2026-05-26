# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest

from utils.api_concurrency import (
    InferenceConcurrencyMiddleware,
    is_limited_inference_request,
    parse_api_max_concurrency,
    parse_api_queue_policy,
)


async def _call_app(app, path: str = "/v1/chat/completions"):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {"type": "http", "method": "POST", "path": path, "headers": []},
        receive,
        send,
    )
    return messages


def _status(messages):
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def _body(messages):
    chunks = [m.get("body", b"") for m in messages if m["type"] == "http.response.body"]
    return b"".join(chunks)


def _headers(messages):
    start = next(m for m in messages if m["type"] == "http.response.start")
    return dict(start["headers"])


def test_parse_api_max_concurrency_defaults_to_one_for_invalid_values(monkeypatch):
    monkeypatch.delenv("UNSLOTH_API_MAX_CONCURRENCY", raising = False)

    assert parse_api_max_concurrency(None) == 1
    assert parse_api_max_concurrency("bad") == 1
    assert parse_api_max_concurrency("0") == 1
    assert parse_api_max_concurrency("-3") == 1
    assert parse_api_max_concurrency("4") == 4


def test_parse_api_queue_policy_defaults_to_wait_for_invalid_values(monkeypatch):
    monkeypatch.delenv("UNSLOTH_API_QUEUE_POLICY", raising = False)

    assert parse_api_queue_policy(None) == "wait"
    assert parse_api_queue_policy("reject") == "reject"
    assert parse_api_queue_policy("WAIT") == "wait"
    assert parse_api_queue_policy("drop") == "wait"


def test_only_generation_posts_are_limited():
    assert is_limited_inference_request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
    )
    assert is_limited_inference_request(
        {"type": "http", "method": "POST", "path": "/api/inference/responses"}
    )
    assert not is_limited_inference_request(
        {"type": "http", "method": "POST", "path": "/api/inference/load"}
    )
    assert not is_limited_inference_request(
        {"type": "http", "method": "GET", "path": "/v1/models"}
    )


@pytest.mark.asyncio
async def test_reject_policy_returns_429_when_limit_is_reached():
    first_request_can_finish = asyncio.Event()
    first_request_started = asyncio.Event()

    async def slow_app(scope, receive, send):
        first_request_started.set()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await first_request_can_finish.wait()
        await send({"type": "http.response.body", "body": b"ok"})

    app = InferenceConcurrencyMiddleware(
        slow_app,
        max_concurrency = 1,
        queue_policy = "reject",
    )

    first = asyncio.create_task(_call_app(app))
    await asyncio.wait_for(first_request_started.wait(), timeout = 1)

    second_messages = await _call_app(app)
    assert _status(second_messages) == 429
    body = _body(second_messages)
    headers = _headers(second_messages)
    assert headers[b"content-length"] == str(len(body)).encode("ascii")
    payload = json.loads(body)
    assert payload["error"]["code"] == "max_concurrency_exceeded"

    first_request_can_finish.set()
    assert _status(await first) == 200


@pytest.mark.asyncio
async def test_wait_policy_holds_streaming_slot_until_final_body_chunk():
    release_first = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def streaming_app(scope, receive, send):
        if not first_started.is_set():
            first_started.set()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"a", "more_body": True})
            await release_first.wait()
            await send({"type": "http.response.body", "body": b"b", "more_body": False})
        else:
            second_started.set()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"second"})

    app = InferenceConcurrencyMiddleware(streaming_app, max_concurrency = 1)

    first = asyncio.create_task(_call_app(app))
    await asyncio.wait_for(first_started.wait(), timeout = 1)

    second = asyncio.create_task(_call_app(app))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    release_first.set()
    assert _body(await first) == b"ab"
    assert _body(await asyncio.wait_for(second, timeout = 1)) == b"second"
