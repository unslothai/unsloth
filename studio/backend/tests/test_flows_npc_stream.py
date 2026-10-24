# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Player entry points answer fast: the model loads ahead of the first call, and an NPC's line streams as written."""

import asyncio
import json

from core.flows import engine as engine_mod
from core.flows.nodes._common import JsonFieldStream
from utils.account_context import OWNER, bind_account, reset_account

from ._flows_helpers import doc, node
from .test_flows_routes import client  # noqa: F401 - the routes fixture

STATE = {"world": {"npcs": []}, "npc_id": "tom", "history": [], "player_message": "hi"}


def _feed_all(raw: str, step: int) -> str:
    stream = JsonFieldStream("text")
    return "".join(stream.feed(raw[:end]) for end in range(step, len(raw) + step, step))


def test_text_field_streams_decoded_whatever_the_chunking():
    raw = json.dumps(
        {"text": 'Hi "friend", a \\ path\nnext line é \U0001f41d done', "give_item": "bell"}
    )
    want = json.loads(raw)["text"]
    for step in (1, 2, 3, 5, 7, len(raw)):
        assert _feed_all(raw, step) == want, step


def test_text_field_ignores_what_comes_before_and_after():
    stream = JsonFieldStream("text")
    assert stream.feed('{"te') == ""
    assert stream.feed('{"text": "ab') == "ab"
    assert stream.feed('{"text": "abc", "take_item": "x"}') == "c"
    assert stream.feed('{"text": "abc", "take_item": "xyz"}') == ""


def _npc_flow():
    return doc([node("r", "world.npc_reply")], [], id = "village")


def _streaming_reply(pieces):
    async def reply(ctx):
        for piece in pieces:
            ctx.text_delta("reply", piece)
            await asyncio.sleep(0.08)
        return {"reply": {"text": "".join(pieces)}}

    return reply


def _sse(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_streamed_invoke_sends_text_then_the_result(client):  # noqa: F811
    engine_mod.ENGINE.handlers["world.npc_reply"] = _streaming_reply(["Hel", "lo ", "there"])
    client.post("/api/flows", json = {"doc": _npc_flow()})
    body = {"entry": "r", "inputs": {"state": STATE}, "stream": True, "rev": 3}
    res = client.post("/api/flows/village/invoke", json = body)
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _sse(res.text)
    deltas = [e for e in events if e["type"] == "delta"]
    assert len(deltas) >= 2 and "".join(e["text"] for e in deltas) == "Hello there"
    assert {e["port"] for e in deltas} == {"reply"}
    assert events[-1]["type"] == "result" and events[-1]["rev"] == 3
    assert events[-1]["outputs"]["reply"]["text"] == "Hello there"


def test_plain_invoke_still_answers_in_one_json(client):  # noqa: F811
    engine_mod.ENGINE.handlers["world.npc_reply"] = _streaming_reply(["Hi"])
    client.post("/api/flows", json = {"doc": _npc_flow()})
    res = client.post("/api/flows/village/invoke", json = {"entry": "r", "inputs": {"state": STATE}})
    assert res.status_code == 200
    assert res.json()["outputs"]["reply"]["text"] == "Hi"


def test_a_failed_streamed_answer_ends_with_an_error_event(client):  # noqa: F811
    from core.flows.registry import FlowNodeError

    async def broken(ctx):
        ctx.text_delta("reply", "Hm")
        raise FlowNodeError("The model is busy", status = 409)

    engine_mod.ENGINE.handlers["world.npc_reply"] = broken
    client.post("/api/flows", json = {"doc": _npc_flow()})
    res = client.post(
        "/api/flows/village/invoke", json = {"entry": "r", "inputs": {"state": STATE}, "stream": True}
    )
    last = _sse(res.text)[-1]
    assert last["type"] == "error" and "busy" in last["detail"]


def test_warm_loads_the_entry_model_once_for_concurrent_callers(monkeypatch):
    eng = engine_mod.FlowEngine()
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    loads = []

    async def fake_ensure(self, model_kind, choice):
        loads.append((model_kind, choice))
        await asyncio.sleep(0.05)

    monkeypatch.setattr(engine_mod.NodeContextImpl, "ensure_model", fake_ensure)
    model = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}
    d = doc([node("r", "world.npc_reply", model = model)], [], id = "village")

    async def go():
        token = bind_account(OWNER)
        try:
            first = await asyncio.gather(
                *(eng.warm(app = None, flow_id = "village", doc = d, entry = "r") for _ in range(3))
            )
            again = await eng.warm(app = None, flow_id = "village", doc = d, entry = "r")
        finally:
            reset_account(token)
        return first, again

    first, again = asyncio.run(go())
    assert first == [True, True, True] and again is True
    # Three players arriving together share one load; a later call runs ensure_model again, which is a no-op
    # when the model is resident.
    assert loads == [("chat", model), ("chat", model)]


def test_warm_refuses_a_node_that_is_not_an_entry_point(client):  # noqa: F811
    client.post(
        "/api/flows", json = {"doc": doc([node("a", "input.text", value = "x")], [], id = "plain")}
    )
    res = client.post("/api/flows/plain/invoke", json = {"entry": "a", "warm": True})
    assert res.status_code == 422 and "entry point" in res.json()["detail"]


def test_in_process_streams_arrive_as_the_route_writes_them():
    import time

    from core.flows import caller as flow_caller

    finished = []

    async def app(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for i in range(3):
            await send(
                {
                    "type": "http.response.body",
                    "body": f'data: {{"i": {i}}}\n\n'.encode(),
                    "more_body": True,
                }
            )
            await asyncio.sleep(0.2)
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        finished.append(True)

    from types import SimpleNamespace

    caller = SimpleNamespace(studio_app = app)

    monkey_headers = flow_caller._headers
    flow_caller._headers = lambda caller, content_type, extra = None: {"content-type": content_type}
    flow_caller.checked_path, saved = (lambda url, method = None: url.path), flow_caller.checked_path

    async def go():
        t0 = time.monotonic()
        seen = []
        async for event in flow_caller.stream_sse(caller, "/v1/chat/completions", {}):
            seen.append((event["i"], time.monotonic() - t0))
        return seen

    async def leave_early():
        async for _ in flow_caller.stream_sse(caller, "/v1/chat/completions", {}):
            break
        await asyncio.sleep(0.3)

    try:
        seen = asyncio.run(go())
        # httpx's own ASGITransport would hand all three over together, after 0.6 s.
        assert [i for i, _ in seen] == [0, 1, 2] and seen[0][1] < 0.15 and seen[2][1] > 0.35
        finished.clear()
        asyncio.run(leave_early())
        assert finished == []  # the route was stopped, not run to the end
    finally:
        flow_caller._headers = monkey_headers
        flow_caller.checked_path = saved


def test_a_left_player_cancels_a_non_streamed_invoke(monkeypatch):
    """The background sim's calls are not streamed; when the player aborts one, the invocation stops too."""
    import pytest
    from fastapi import HTTPException

    from routes import flows as flows_routes

    started = asyncio.Event()
    cancelled = []

    class Engine:
        async def invoke(self, **_):
            started.set()
            try:
                await asyncio.sleep(60)
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
        await asyncio.sleep(0)
        return err.value.status_code

    assert asyncio.run(go()) == 499
    assert cancelled == [True]
