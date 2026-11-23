# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The message read routes dump their JSON in the threadpool, byte for byte what FastAPI sent."""

import asyncio
import base64
import inspect
import json
import os
import statistics
import sys
import time

import fastapi.routing
import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from auth.authentication import get_current_subject
from routes import chat_history
from storage import studio_db

IMAGE = "data:image/png;base64," + base64.b64encode(bytes(range(256)) * 400).decode()
ODD_TEXT = (
    'quote " backslash \\ slash / tab \t nl \n nul \x00 del \x7f \u2028 \u2029 \ufeff \ud7ff'
    " \u00e9 \u6f22\u5b57 \U0001f600 \u0085"
)
FLOATS = [0.1, 1e16, 1e-7, -0.0, 1.5e300, 123456789.123456789, 2**53 + 1, -(2**63), 2**63 - 1]


def _thread(thread_id: str) -> None:
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "Chat",
            "modelType": "base",
            "modelId": "m",
            "pairId": None,
            "archived": False,
            "createdAt": 1_700_000_000_000,
        }
    )


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    for thread_id in ("text", "images", "odd", "empty"):
        _thread(thread_id)
    studio_db.sync_chat_messages(
        "text",
        [
            {
                "id": "t1",
                "threadId": "text",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "createdAt": 1,
            },
            {
                "id": "t2",
                "threadId": "text",
                "parentId": "t1",
                "role": "assistant",
                "content": "plain string content",
                "metadata": {"custom": {"timings": {"predicted_n": 3, "predicted_ms": 12.5}}},
                "createdAt": 2,
            },
        ],
    )
    studio_db.sync_chat_messages(
        "images",
        [
            {
                "id": f"i{i}",
                "threadId": "images",
                "parentId": f"i{i - 1}" if i else None,
                "role": "user",
                "content": [{"type": "text", "text": "look"}, {"type": "image", "image": IMAGE}],
                "attachments": [
                    {
                        "id": "a",
                        "type": "image",
                        "name": "x.png",
                        "content": [{"type": "image", "image": IMAGE}],
                    },
                    {
                        "id": "d",
                        "type": "document",
                        "name": "n.txt",
                        "content": [{"type": "text", "text": "<pasted_text name=n>\nB"}],
                    },
                ],
                "createdAt": 10 + i,
            }
            for i in range(3)
        ],
    )
    studio_db.sync_chat_messages(
        "odd",
        [
            {
                "id": "o1",
                "threadId": "odd",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ODD_TEXT},
                    {
                        "type": "tool-call",
                        "toolName": "python",
                        "args": {"x": FLOATS},
                        "result": {"ok": True, "n": None},
                    },
                ],
                "attachments": [],
                "metadata": {
                    "floats": FLOATS,
                    ODD_TEXT: [ODD_TEXT, {"nested": {"deep": [[[1, 2.5]]]}}],
                },
                "createdAt": 2**63 - 1,
            }
        ],
    )


def _old_get_thread_messages(thread_id: str):
    """The handler before this change: FastAPI validates and serializes the returned model."""
    if chat_history.get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return chat_history.ChatMessageListResponse(
        messages = [chat_history.ChatMessage(**m) for m in chat_history.list_chat_messages(thread_id)]
    )


def _old_batch_thread_messages(payload: chat_history.ChatMessagesBatchRequest):
    by_thread = {tid: [] for tid in payload.threadIds}
    search_view = payload.view == "search"
    for m in chat_history.list_chat_messages_for_threads(
        payload.threadIds, include_metadata = not search_view
    ):
        if m["threadId"] in by_thread:
            by_thread[m["threadId"]].append(
                chat_history.ChatMessage(
                    **(chat_history._search_view_message(m) if search_view else m)
                )
            )
    return chat_history.ChatMessagesBatchResponse(messagesByThreadId = by_thread)


def _old_app(subject = True) -> FastAPI:
    app = FastAPI()

    @app.get(
        "/api/chat/threads/{thread_id}/messages",
        response_model = chat_history.ChatMessageListResponse,
    )
    def old_get(thread_id: str, current_subject: str = Depends(get_current_subject)):
        return _old_get_thread_messages(thread_id)

    @app.post("/api/chat/messages:batch", response_model = chat_history.ChatMessagesBatchResponse)
    def old_batch(
        payload: chat_history.ChatMessagesBatchRequest,
        current_subject: str = Depends(get_current_subject),
    ):
        return _old_batch_thread_messages(payload)

    if subject:
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def _new_app(subject = True) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    if subject:
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


# The Python 3.10+ pin (0.141.1) writes a response_model with dump_json; the Python 3.9 pin (0.128.8) uses json.dumps.
FASTAPI_DUMPS_JSON = "dump_json" in inspect.signature(fastapi.routing.serialize_response).parameters

REQUESTS = [
    ("GET", "/api/chat/threads/text/messages", None),
    ("GET", "/api/chat/threads/images/messages", None),
    ("GET", "/api/chat/threads/odd/messages", None),
    ("GET", "/api/chat/threads/empty/messages", None),
    ("GET", "/api/chat/threads/missing/messages", None),
    (
        "POST",
        "/api/chat/messages:batch",
        {"threadIds": ["text", "images", "odd", "empty", "missing"]},
    ),
    (
        "POST",
        "/api/chat/messages:batch",
        {"threadIds": ["images", "odd", "text"], "view": "search"},
    ),
    ("POST", "/api/chat/messages:batch", {"threadIds": []}),
    ("POST", "/api/chat/messages:batch", {"threadIds": ["odd", "odd", ODD_TEXT, "", "missing"]}),
    ("POST", "/api/chat/messages:batch", {"threadIds": "not-a-list"}),
    ("POST", "/api/chat/messages:batch", {"threadIds": ["text"], "view": "bogus"}),
]


def _exchange(client, method, path, body):
    r = client.request(method, path, json = body)
    headers = {k: v for k, v in r.headers.items() if k.lower() not in ("date",)}
    return r.status_code, headers, r.content


@pytest.mark.parametrize("subject", [True, False], ids = ["signed-in", "anonymous"])
def test_responses_are_byte_identical_to_fastapi_serialization(history, subject):
    with TestClient(_old_app(subject)) as old, TestClient(_new_app(subject)) as new:
        for method, path, body in REQUESTS:
            before = _exchange(old, method, path, body)
            after = _exchange(new, method, path, body)
            if FASTAPI_DUMPS_JSON:
                assert after == before, (method, path, body)
            else:
                # json.dumps spells some floats differently (1e+16 against 1e16), so only the parsed body and length-free headers match.
                assert after[0] == before[0], (method, path, body)
                assert {**after[1], "content-length": ""} == {**before[1], "content-length": ""}
                assert json.loads(after[2]) == json.loads(before[2]), (method, path, body)
            if subject and method == "GET":
                assert before[0] == (404 if "missing" in path else 200)
                assert before[1]["content-type"] == "application/json"
            if not subject:
                assert before[0] in (401, 403)


def test_image_thread_round_trips(history):
    with TestClient(_new_app()) as client:
        messages = client.get("/api/chat/threads/images/messages").json()["messages"]
        odd = client.get("/api/chat/threads/odd/messages").json()["messages"][0]
    assert [m["content"][1]["image"] for m in messages] == [IMAGE] * 3
    assert odd["content"][0]["text"] == ODD_TEXT
    assert odd["metadata"]["floats"] == FLOATS


def test_fastapi_never_serializes_on_the_event_loop(history, monkeypatch):
    """FastAPI's serialize_response runs on the loop; these routes hand it nothing to do."""
    calls = []
    real = fastapi.routing.serialize_response

    async def counting(**kwargs):
        calls.append(kwargs.get("field"))
        return await real(**kwargs)

    monkeypatch.setattr(fastapi.routing, "serialize_response", counting)
    with TestClient(_new_app()) as client:
        for method, path, body in REQUESTS:
            client.request(method, path, json = body)
        assert calls == []
    with TestClient(_old_app()) as client:
        client.get("/api/chat/threads/text/messages")
    # The counter itself is live.
    assert len(calls) == 1


def test_openapi_response_schemas_unchanged():
    old = _old_app().openapi()
    new = _new_app().openapi()
    for path, method in (
        ("/api/chat/threads/{thread_id}/messages", "get"),
        ("/api/chat/messages:batch", "post"),
    ):
        o, n = old["paths"][path][method], new["paths"][path][method]
        assert n["responses"] == o["responses"]
        assert n.get("requestBody") == o.get("requestBody")
        assert n.get("parameters") == o.get("parameters")
    for name in ("ChatMessage", "ChatMessageListResponse", "ChatMessagesBatchResponse"):
        assert new["components"]["schemas"][name] == old["components"]["schemas"][name]


def _max_loop_lag_ms(
    app,
    method,
    path,
    body,
    repeats = 5,
) -> float:
    async def run():
        lags = []
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(
            transport = transport, base_url = "http://t", timeout = 120
        ) as client:
            for _ in range(repeats + 1):
                worst = 0.0
                stop = asyncio.Event()

                async def ticker():
                    nonlocal worst
                    last = time.perf_counter()
                    while not stop.is_set():
                        await asyncio.sleep(0.001)
                        now = time.perf_counter()
                        worst = max(worst, now - last - 0.001)
                        last = now

                task = asyncio.create_task(ticker())
                await asyncio.sleep(0.02)
                worst = 0.0
                r = await client.request(method, path, json = body)
                stop.set()
                await task
                assert r.status_code == 200
                lags.append(worst * 1000)
        return statistics.median(lags[1:])

    return asyncio.run(run())


def test_large_image_thread_does_not_stall_the_event_loop(history):
    """About 96 MB of data URLs: the old path held the loop for one whole dump, the new one per message."""
    big = "data:image/jpeg;base64," + base64.b64encode(os.urandom(750_000)).decode()
    _thread("big")
    studio_db.sync_chat_messages(
        "big",
        [
            {
                "id": f"b{i}",
                "threadId": "big",
                "role": "user",
                "content": [{"type": "image", "image": big}],
                "attachments": [
                    {"id": "a", "type": "image", "content": [{"type": "image", "image": big}]}
                ],
                "createdAt": i,
            }
            for i in range(48)
        ],
    )
    for method, path, body in (
        ("GET", "/api/chat/threads/big/messages", None),
        ("POST", "/api/chat/messages:batch", {"threadIds": ["big"]}),
    ):
        old_lag = _max_loop_lag_ms(_old_app(), method, path, body)
        new_lag = _max_loop_lag_ms(_new_app(), method, path, body)
        assert new_lag * 2 < old_lag, (path, old_lag, new_lag)
