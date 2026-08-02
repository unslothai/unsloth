# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end handshake test for the tool-confirmation gate, no model.

The real Unsloth stream wrappers in ``routes/inference.py`` drive the
synchronous agentic generator with ``await asyncio.to_thread(next, gen,
...)`` so the blocking ``threading.Event`` wait runs off the event loop.
This test rebuilds that exact pattern around the real
``state.tool_approvals`` functions, served by a real uvicorn process on
loopback (the same server Unsloth uses), and proves the load-bearing
property:

* ``tool_start`` reaches the client before the gate blocks, and
* the separate ``/tool-confirm`` POST is served *while* the stream
  connection is blocked, after which the stream resumes with the executed
  (allow) or rejected (deny) result -- i.e. no deadlock.

Each scenario runs under a socket-level timeout, so a regression that
reintroduces a deadlock fails fast instead of hanging the suite.
"""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from state import tool_approvals
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    begin_tool_decision,
    new_approval_id,
    resolve_tool_decision,
    wait_tool_decision,
)

_EXECUTED_RESULT = "tool executed: 2"


@pytest.fixture(autouse = True)
def _clear_pending():
    with tool_approvals._lock:
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()


def _build_app() -> FastAPI:
    """Minimal app mirroring the real stream/confirm wiring."""
    app = FastAPI()

    def agentic_gen(session_id, cancel_event):
        # Same shape as the real loops: register the approval slot, announce
        # the call (echoing approval_id), gate on the decision, then either
        # execute or feed back the rejection.
        approval_id = new_approval_id()
        slot = begin_tool_decision(session_id, approval_id)
        yield {
            "type": "tool_start",
            "tool_name": "python",
            "approval_id": approval_id,
            "awaiting_confirmation": True,
        }
        denied = wait_tool_decision(slot, approval_id, cancel_event = cancel_event) == "deny"
        result = TOOL_REJECTED_MESSAGE if denied else _EXECUTED_RESULT
        yield {"type": "tool_end", "tool_name": "python", "result": result}

    @app.post("/stream")
    async def stream(req: Request):
        body = await req.json()
        session_id = body.get("session_id")
        cancel_event = threading.Event()
        sentinel = object()

        async def wrapper():
            gen = agentic_gen(session_id, cancel_event)
            while True:
                event = await asyncio.to_thread(next, gen, sentinel)
                if event is sentinel:
                    break
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(wrapper(), media_type = "text/event-stream")

    @app.post("/tool-confirm")
    async def tool_confirm(req: Request):
        body = await req.json()
        resolved = resolve_tool_decision(
            body.get("approval_id"),
            body.get("decision"),
            session_id = body.get("session_id"),
        )
        return {"resolved": resolved}

    return app


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server:
    """Run a uvicorn server in a background thread for the test's lifetime."""

    def __init__(self, app):
        self.port = _free_port()
        config = uvicorn.Config(app, host = "127.0.0.1", port = self.port, log_level = "warning")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target = self.server.run, daemon = True)

    def __enter__(self):
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self.server.started:
                return self
            time.sleep(0.02)
        raise AssertionError("uvicorn did not start in time")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self._thread.join(timeout = 10.0)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


async def _gate_is_blocking(approval_id) -> None:
    """Wait until the stream thread is parked on this approval's slot.

    The slot is registered before ``tool_start`` is yielded, so it exists
    by the time the client receives the event -- exactly as in reality,
    where the confirm POST only arrives after the card renders.
    """
    for _ in range(400):
        with tool_approvals._lock:
            slot = tool_approvals._pending.get(approval_id)
            if slot is not None and not slot["event"].is_set():
                return
        await asyncio.sleep(0.005)
    raise AssertionError("gate never started waiting")


async def _drive(base_url, session_id, decision):
    events = []
    resolved = None
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(base_url = base_url, timeout = timeout) as client:
        async with client.stream("POST", "/stream", json = {"session_id": session_id}) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                events.append(event)
                if event["type"] == "tool_start":
                    # The stream is now blocked on the gate; the confirm
                    # POST (echoing approval_id) must still be served over a
                    # second connection.
                    approval_id = event["approval_id"]
                    await _gate_is_blocking(approval_id)
                    r = await client.post(
                        "/tool-confirm",
                        json = {
                            "session_id": session_id,
                            "approval_id": approval_id,
                            "decision": decision,
                        },
                    )
                    resolved = r.json()["resolved"]
    return events, resolved


def _run(session_id, decision):
    with _Server(_build_app()) as srv:
        return asyncio.run(
            asyncio.wait_for(_drive(srv.base_url, session_id, decision), timeout = 15.0)
        )


def _types(events):
    return [e["type"] for e in events]


def test_allow_resumes_stream_with_executed_result():
    events, resolved = _run("sess-allow", "allow")
    assert resolved is True
    assert _types(events) == ["tool_start", "tool_end"]
    assert events[-1]["result"] == _EXECUTED_RESULT


def test_deny_resumes_stream_with_rejection_result():
    events, resolved = _run("sess-deny", "deny")
    assert resolved is True
    assert _types(events) == ["tool_start", "tool_end"]
    assert events[-1]["result"] == TOOL_REJECTED_MESSAGE


def test_tool_start_precedes_the_block_and_carries_approval_id():
    # The first streamed event is always tool_start, proving the buttons
    # can render before the backend pauses for the decision -- and it
    # carries the approval_id / awaiting_confirmation the UI needs.
    events, _ = _run("sess-order", "allow")
    assert events[0]["type"] == "tool_start"
    assert events[0]["awaiting_confirmation"] is True
    assert events[0]["approval_id"]
