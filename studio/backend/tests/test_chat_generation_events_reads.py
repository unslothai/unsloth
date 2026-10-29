# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import random
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from routes import chat_generation_runs as run_routes
from storage import chat_generation_runs_db as runs_db
from storage import studio_db


@pytest.fixture
def running_run():
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )
    history = [{"role": "user", "content": "x" * 2000} for _ in range(20)]
    runs_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {"model": "local", "messages": history, "stream": True},
    )
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token) is True
    return token


async def _stream(after = 0):
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = AsyncMock(return_value = False)),
        after = after,
        last_event_id = None,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return raw


def test_chunk_only_batches_do_not_parse_the_request(running_run, monkeypatch):
    for i in range(5):
        runs_db.append_events("run-1", running_run, [("chunk", {"delta": f"t{i}"})])
    runs_db.finish_run("run-1", worker_token = running_run, status = "completed", finish_reason = "stop")
    original_list = runs_db.list_events
    monkeypatch.setattr(
        runs_db, "list_events", lambda run_id, after = 0, limit = 1000: original_list(run_id, after, 1)
    )
    full_reads = []
    original_get = runs_db.get_run
    monkeypatch.setattr(
        runs_db, "get_run", lambda *a, **k: full_reads.append(a) or original_get(*a, **k)
    )

    raw = asyncio.run(_stream())

    frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
    assert [frame["type"] for frame in frames] == ["run.created", "run.started"] + ["chunk"] * 5 + [
        "run.completed"
    ]
    for frame in frames:
        if frame["type"] == "chunk":
            assert "run" not in frame
        else:
            assert len(frame["run"]["requestPayload"]["messages"]) == 20
    # Ownership + opening checks, plus one per run-event batch (3).
    assert len(full_reads) == 5


def test_caught_up_wait_reads_events_twice(running_run, monkeypatch):
    calls = []
    original_list = runs_db.list_events
    monkeypatch.setattr(
        runs_db, "list_events", lambda *a, **k: calls.append(a) or original_list(*a, **k)
    )
    cursor = int(runs_db.get_run("run-1")["lastEventSeq"])
    timer = threading.Timer(
        0.3, lambda: runs_db.append_events("run-1", running_run, [("chunk", {"delta": "a"})])
    )
    timer.start()
    try:
        events = runs_db.wait_for_events("run-1", cursor, 10)
    finally:
        timer.cancel()
    assert [event["payload"] for event in events] == [{"delta": "a"}]
    assert len(calls) == 2


def test_commit_between_read_and_wait_is_not_missed(running_run, monkeypatch):
    cursor = int(runs_db.get_run("run-1")["lastEventSeq"])
    original_list = runs_db.list_events
    state = {"first": True}

    def racing_list(*args, **kwargs):
        events = original_list(*args, **kwargs)
        if state["first"]:
            state["first"] = False
            # Lands after the unlocked read saw nothing and before the waiter takes the lock.
            runs_db.append_events("run-1", running_run, [("chunk", {"delta": "late"})])
        return events

    monkeypatch.setattr(runs_db, "list_events", racing_list)
    started = time.monotonic()
    events = runs_db.wait_for_events("run-1", cursor, 10)
    assert [event["payload"] for event in events] == [{"delta": "late"}]
    assert time.monotonic() - started < 5


def test_concurrent_followers_never_miss_a_commit(running_run, monkeypatch):
    # Widens the window between each unlocked read and the lock, so commits keep landing inside it.
    original_list = runs_db.list_events
    rng_lock = threading.Lock()
    rng = random.Random(0)

    def slow_list(*args, **kwargs):
        events = original_list(*args, **kwargs)
        with rng_lock:
            delay = rng.random() * 0.002
        time.sleep(delay)
        return events

    monkeypatch.setattr(runs_db, "list_events", slow_list)
    total = 150
    start = int(runs_db.get_run("run-1")["lastEventSeq"])
    stalls = []
    seen = {}

    def follower(name):
        cursor = start
        received = []
        while len(received) < total:
            began = time.monotonic()
            events = runs_db.wait_for_events("run-1", cursor, 30)
            if time.monotonic() - began > 5:
                stalls.append(name)
                return
            for event in events:
                cursor = int(event["seq"])
                received.append(event["payload"]["delta"])
        seen[name] = received

    threads = [threading.Thread(target = follower, args = (i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for i in range(total):
        runs_db.append_events("run-1", running_run, [("chunk", {"delta": str(i)})])
        time.sleep(rng.random() * 0.003)
    for thread in threads:
        thread.join(60)
    assert not stalls
    assert all(not thread.is_alive() for thread in threads)
    assert seen == {i: [str(n) for n in range(total)] for i in range(6)}
