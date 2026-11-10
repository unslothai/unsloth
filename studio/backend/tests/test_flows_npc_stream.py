# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from core.flows import engine as engine_mod
import pytest

from core.flows.specs import SPECS_BY_TYPE
from utils.account_context import OWNER, bind_account, reset_account

from ._flows_helpers import chat_entry_spec, doc, make_engine, node, wait_until

STATE = {"history": [], "player_message": "hi"}


@pytest.fixture(autouse = True)
def _reply_spec(monkeypatch):
    monkeypatch.setitem(SPECS_BY_TYPE, "test.reply", chat_entry_spec("test.reply"))


def _reply_flow():
    return doc([node("r", "test.reply")], [], id = "village")


def _streaming_reply(pieces, streamed = True):
    async def reply(ctx):
        for piece in pieces:
            ctx.text_delta("reply", piece)
            if streamed:
                # The invoke flusher batches deltas; wait until this piece went out as its own event.
                await wait_until(lambda: not ctx._rs._pending_text, what = "the delta flush")
        return {"reply": {"text": "".join(pieces)}}

    return reply


def _sse(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_streamed_invoke_sends_text_then_the_result(flows_client):
    engine_mod.ENGINE.handlers["test.reply"] = _streaming_reply(["Hel", "lo ", "there"])
    flows_client.post("/api/flows", json = {"doc": _reply_flow()})
    body = {"entry": "r", "inputs": {"state": STATE}, "stream": True, "rev": 3}
    res = flows_client.post("/api/flows/village/invoke", json = body)
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _sse(res.text)
    deltas = [e for e in events if e["type"] == "delta"]
    assert len(deltas) >= 2 and "".join(e["text"] for e in deltas) == "Hello there"
    assert {e["port"] for e in deltas} == {"reply"}
    assert events[-1]["type"] == "result" and events[-1]["rev"] == 3
    assert events[-1]["outputs"]["reply"]["text"] == "Hello there"


def test_plain_invoke_still_answers_in_one_json(flows_client):
    engine_mod.ENGINE.handlers["test.reply"] = _streaming_reply(["Hi"], streamed = False)
    flows_client.post("/api/flows", json = {"doc": _reply_flow()})
    res = flows_client.post(
        "/api/flows/village/invoke", json = {"entry": "r", "inputs": {"state": STATE}}
    )
    assert res.status_code == 200
    assert res.json()["outputs"]["reply"]["text"] == "Hi"


def test_a_failed_streamed_answer_ends_with_an_error_event(flows_client):
    from core.flows.registry import FlowNodeError

    async def broken(ctx):
        ctx.text_delta("reply", "Hm")
        raise FlowNodeError("The model is busy", status = 409)

    engine_mod.ENGINE.handlers["test.reply"] = broken
    flows_client.post("/api/flows", json = {"doc": _reply_flow()})
    res = flows_client.post(
        "/api/flows/village/invoke", json = {"entry": "r", "inputs": {"state": STATE}, "stream": True}
    )
    last = _sse(res.text)[-1]
    assert last["type"] == "error" and "busy" in last["detail"]


def test_warm_loads_the_entry_model_once_for_concurrent_callers(monkeypatch):
    eng = make_engine()
    loads = []
    release = asyncio.Event()

    async def fake_ensure(self, model_kind, choice):
        loads.append((model_kind, choice))
        await release.wait()

    monkeypatch.setattr(engine_mod.NodeContextImpl, "ensure_model", fake_ensure)
    model = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}
    d = doc([node("r", "test.reply", model = model)], [], id = "village")

    async def go():
        token = bind_account(OWNER)
        try:
            callers = asyncio.gather(
                *(eng.warm(app = None, flow_id = "village", doc = d, entry = "r") for _ in range(3))
            )
            # Every caller has joined while the first load is still held open.
            await wait_until(lambda: loads, what = "the first load")
            release.set()
            first = await asyncio.wait_for(callers, 10)
            again = await eng.warm(app = None, flow_id = "village", doc = d, entry = "r")
        finally:
            reset_account(token)
        return first, again

    first, again = asyncio.run(go())
    assert first == [True, True, True] and again is True
    assert loads == [("chat", model), ("chat", model)]


def test_warm_refuses_a_node_that_is_not_an_entry_point(flows_client):
    flows_client.post(
        "/api/flows", json = {"doc": doc([node("a", "input.text", value = "x")], [], id = "plain")}
    )
    res = flows_client.post("/api/flows/plain/invoke", json = {"entry": "a", "warm": True})
    assert res.status_code == 422 and "entry point" in res.json()["detail"]


def test_in_process_streams_arrive_as_the_route_writes_them():
    from types import SimpleNamespace

    from core.flows import caller as flow_caller

    finished, cancelled = [], []
    acks: list[asyncio.Event] = []

    async def app(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        try:
            for i in range(3):
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'data: {{"i": {i}}}\n\n'.encode(),
                        "more_body": True,
                    }
                )
                # The next chunk is written only once the reader has seen this one, so a transport that
                # held the body back until the end would never get past the first chunk.
                await acks[i].wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        finished.append(True)

    caller = SimpleNamespace(studio_app = app)

    monkey_headers = flow_caller._headers
    flow_caller._headers = lambda caller, content_type, extra = None: {"content-type": content_type}
    flow_caller.checked_path, saved = (lambda url, method = None: url.path), flow_caller.checked_path

    async def read_all():
        acks[:] = [asyncio.Event() for _ in range(3)]
        seen = []
        async for event in flow_caller.stream_sse(caller, "/v1/chat/completions", {}):
            seen.append(event["i"])
            acks[event["i"]].set()
        return seen

    async def leave_early():
        acks[:] = [asyncio.Event() for _ in range(3)]
        stream = flow_caller.stream_sse(caller, "/v1/chat/completions", {})
        async for _ in stream:
            break
        await stream.aclose()
        await wait_until(lambda: cancelled, what = "the route to be cancelled")

    try:
        assert asyncio.run(asyncio.wait_for(read_all(), 10)) == [0, 1, 2]
        assert finished == [True] and cancelled == []
        finished.clear()
        asyncio.run(asyncio.wait_for(leave_early(), 10))
        assert finished == [] and cancelled == [True]
    finally:
        flow_caller._headers = monkey_headers
        flow_caller.checked_path = saved


def test_a_left_player_cancels_a_non_streamed_invoke(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from routes import flows as flows_routes

    started = asyncio.Event()
    cancelled = []

    class Engine:
        async def invoke(self, **_):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    class Request:
        app = None
        headers: dict = {}

        async def is_disconnected(self):
            return started.is_set()

    monkeypatch.setattr(flows_routes, "get_engine", lambda: Engine())
    monkeypatch.setattr(flows_routes, "_flow_or_404", lambda fid: {"doc": {}})
    monkeypatch.setattr(flows_routes, "_hf_token", lambda request: None)
    body = flows_routes.InvokeBody(entry = "r", inputs = {})

    async def go():
        with pytest.raises(HTTPException) as err:
            await asyncio.wait_for(flows_routes.invoke(Request(), "village", body), 10)
        await wait_until(lambda: cancelled, what = "the engine call to be cancelled")
        return err.value.status_code

    assert asyncio.run(go()) == 499
    assert cancelled == [True]
