# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The events route splices stored payload_json into frames; these pin it to the old parse-and-encode bytes."""

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

PAYLOADS = [
    {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
    {"choices": [{"delta": {"reasoning_content": "naïve 数据 \U0001f600"}}]},
    {"text": 'quote " backslash \\ slash / tab \t nl \n cr \r nul \x00 bell \x07 del \x7f'},
    {"text": "line sep   para sep   nel \x85 bom ﻿"},
    {"floats": [0.1, -0.0, 1e300, 1e-300, 3.141592653589793, float("inf"), float("-inf")]},
    {"ints": [0, -1, 2**63, -(2**80), 10**200], "flags": [True, False, None]},
    {"nested": {"a": [{"b": [[], {}, ""]}], "z": {"y": {"x": "deep"}}}},
    {},
    {
        "type": "tool_end",
        "tool_name": "web_fetch",
        "tool_call_id": "call-1",
        "result": "page éè " * 20000,
        "provenance": "studio",
    },
    {"_toolEvent": {"type": "tool_start", "arguments": {"url": "https://example.com/?q=a&b=<c>"}}},
]


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
    runs_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "héllo \U0001f600  "}],
            "stream": True,
        },
    )
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token) is True
    return token


async def _stream(after = 0, last_event_id = None):
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = AsyncMock(return_value = False)),
        after = after,
        last_event_id = last_event_id,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return raw


def _frames(raw):
    """Event frames only; keep-alive comments depend on wake timing, not on the events."""
    return [block + "\n\n" for block in raw.split("\n\n") if block and not block.startswith(":")]


def _old_frame(event, run = None):
    """The frame as the route encoded it before: the parsed payload re-encoded in one json.dumps."""
    data = {
        "seq": event["seq"],
        "type": event["type"],
        "payload": event["payload"],
        "createdAt": event["createdAt"],
    }
    if event["type"] != "chunk":
        data["run"] = run
    encoded = json.dumps(data, ensure_ascii = False, separators = (",", ":"))
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {encoded}\n\n"


def _assert_old_bytes(
    frames,
    after = 0,
    settled_run = None,
):
    events = runs_db.list_events("run-1", after, 100_000)
    assert len(frames) == len(events)
    for frame, event in zip(frames, events):
        run = None
        if event["type"] != "chunk":
            # A live frame carries whichever snapshot that batch read; byte-check the encoding of that.
            run = json.loads(frame.split("\ndata: ", 1)[1])["run"]
            if settled_run is not None:
                assert run == settled_run
        assert frame == _old_frame(event, run)


def _fill(token):
    for payload in PAYLOADS:
        runs_db.append_events("run-1", token, [("chunk", payload)])
    runs_db.append_events("run-1", token, [("chunk", payload) for payload in PAYLOADS])
    runs_db.finish_run(
        "run-1",
        worker_token = token,
        status = "completed",
        finish_reason = "stop",
        pending_events = [("chunk", {"choices": [{"delta": {}, "finish_reason": "stop"}]})],
    )


def test_replay_frames_match_parse_and_reencode_bytes(running_run):
    _fill(running_run)
    settled = runs_db.get_run("run-1")
    frames = _frames(asyncio.run(_stream()))
    _assert_old_bytes(frames, settled_run = settled)
    assert [int(frame[4 : frame.index("\n")]) for frame in frames] == list(
        range(1, settled["lastEventSeq"] + 1)
    )
    assert frames[-1].startswith(f"id: {settled['lastEventSeq']}\nevent: run.completed\n")


@pytest.mark.parametrize("cursor", [1, 2, 7, 14])
def test_replay_from_a_cursor_resumes_with_the_same_bytes(running_run, cursor):
    _fill(running_run)
    full = _frames(asyncio.run(_stream()))
    assert _frames(asyncio.run(_stream(after = cursor))) == full[cursor:]
    assert _frames(asyncio.run(_stream(last_event_id = str(cursor)))) == full[cursor:]


def test_replay_of_a_fully_read_settled_run_sends_nothing(running_run):
    _fill(running_run)
    last = runs_db.get_run("run-1")["lastEventSeq"]
    assert asyncio.run(_stream(after = last)) == ""


def test_live_followers_get_identical_old_bytes_and_the_terminal_frame(running_run):
    async def scenario():
        followers = [asyncio.create_task(_stream()) for _ in range(3)]
        await asyncio.sleep(0.05)

        def produce():
            for i, payload in enumerate(PAYLOADS * 3):
                runs_db.append_events("run-1", running_run, [("chunk", payload)])
                if i % 4 == 0:
                    time.sleep(0.005)
            runs_db.finish_run(
                "run-1", worker_token = running_run, status = "completed", finish_reason = "stop"
            )

        await asyncio.to_thread(produce)
        return await asyncio.wait_for(asyncio.gather(*followers), 30)

    outputs = [_frames(raw) for raw in asyncio.run(scenario())]
    last = runs_db.get_run("run-1")["lastEventSeq"]
    for frames in outputs:
        _assert_old_bytes(frames)
        assert frames[-1].startswith(f"id: {last}\nevent: run.completed\n")
        assert json.loads(frames[-1].split("\ndata: ", 1)[1])["run"]["status"] == "completed"
    chunks = [[frame for frame in frames if "\nevent: chunk\n" in frame] for frames in outputs]
    assert chunks[0] == chunks[1] == chunks[2]


def test_cancelled_run_terminal_frames_keep_old_bytes(running_run):
    runs_db.append_events("run-1", running_run, [("chunk", PAYLOADS[1])])
    runs_db.request_cancel("run-1", "alice")
    runs_db.finish_run(
        "run-1",
        worker_token = running_run,
        status = "cancelled",
        finish_reason = "cancelled",
        pending_events = [("chunk", PAYLOADS[2])],
    )
    frames = _frames(asyncio.run(_stream()))
    _assert_old_bytes(frames, settled_run = runs_db.get_run("run-1"))
    assert [frame.split("\n")[1] for frame in frames][-3:] == [
        "event: run.cancelling",
        "event: chunk",
        "event: run.cancelled",
    ]


def test_an_empty_stored_payload_still_reads_as_an_empty_object(running_run):
    runs_db.append_events("run-1", running_run, [("chunk", {"x": 1})])
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_generation_events SET payload_json='' WHERE run_id='run-1'")
        conn.commit()
    finally:
        conn.close()
    runs_db.finish_run("run-1", worker_token = running_run, status = "completed", finish_reason = "stop")
    frames = _frames(asyncio.run(_stream()))
    _assert_old_bytes(frames, settled_run = runs_db.get_run("run-1"))
    assert all(json.loads(frame.split("\ndata: ", 1)[1])["payload"] == {} for frame in frames[:-1])


def _random_value(rng, depth = 0):
    kind = rng.randrange(8 if depth < 4 else 5)
    if kind == 0:
        return rng.choice([None, True, False])
    if kind == 1:
        return rng.randint(-(2**70), 2**70)
    if kind == 2:
        return rng.choice([rng.uniform(-1e6, 1e6), rng.random() * 10 ** rng.randint(-300, 300)])
    if kind in (3, 4):
        alphabet = 'ab "\\/\n\r\t\x00\x1f\x7fé  ﻿\U0001f600数'
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(12)))
    if kind == 5:
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {f'k{i}é"\n': _random_value(rng, depth + 1) for i in range(rng.randrange(4))}


def test_stored_payloads_are_a_fixed_point_of_parse_and_reencode(running_run):
    rng = random.Random(1234)
    payloads = [{"v": _random_value(rng)} for _ in range(400)]
    runs_db.append_events("run-1", running_run, [("chunk", payload) for payload in payloads])
    parsed = runs_db.list_events("run-1", 2, 1000)
    raw = runs_db.list_events("run-1", 2, 1000, raw = True)
    assert [event["seq"] for event in raw] == [event["seq"] for event in parsed]
    assert [event["createdAt"] for event in raw] == [event["createdAt"] for event in parsed]
    for parsed_event, raw_event in zip(parsed, raw):
        assert raw_event["payloadJson"] == run_routes._encode_json(parsed_event["payload"])
        assert "payload" not in raw_event


def test_streaming_chunks_does_not_parse_stored_payloads(running_run, monkeypatch):
    for start in range(0, 300, 50):
        runs_db.append_events(
            "run-1", running_run, [("chunk", {"delta": f"t{i}"}) for i in range(start, start + 50)]
        )
    runs_db.finish_run("run-1", worker_token = running_run, status = "completed", finish_reason = "stop")
    parsed = []
    original = runs_db._loads
    monkeypatch.setattr(
        runs_db, "_loads", lambda value, fallback: parsed.append(value) or original(value, fallback)
    )

    frames = _frames(asyncio.run(_stream()))

    assert len(frames) == 303
    # Only request_json, for the opening check and the run-event batch snapshot.
    payload_parses = [value for value in parsed if '"delta"' in str(value)]
    assert payload_parses == []
    assert len(parsed) <= 3


def test_wait_for_events_raw_waits_like_the_parsed_read(running_run):
    cursor = int(runs_db.get_run("run-1")["lastEventSeq"])
    timer = threading.Timer(
        0.2, lambda: runs_db.append_events("run-1", running_run, [("chunk", {"d": "é"})])
    )
    timer.start()
    try:
        events = runs_db.wait_for_events("run-1", cursor, 10, True)
    finally:
        timer.cancel()
    assert [event["payloadJson"] for event in events] == ['{"d":"é"}']
