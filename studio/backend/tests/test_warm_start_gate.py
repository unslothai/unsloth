# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The torch warm's GIL-heavy stages start only after the first HTTP response has gone out.

import torch and the CUDA probe hold the GIL for whole C calls, so a liveness or desktop-login probe that lands
during one waits it out and the desktop's 2 s port probe can miss. The warm thread still starts in the
lifespan (torch_warm_in_progress reads as before); only its stages wait for the gate."""

from __future__ import annotations

import ast
import threading
import time
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.torch_warmup as warmup  # noqa: E402


def _gate_middleware(app, event):
    from utils.first_response_gate import FirstResponseGateMiddleware
    return FirstResponseGateMiddleware(app, event = event)


@pytest.fixture
def recorded(monkeypatch):
    """A fresh warm latch with one stage that records when it ran."""
    monkeypatch.setattr(warmup, "_thread", None, raising = False)
    monkeypatch.setattr(warmup, "_thread_epoch", None, raising = False)
    monkeypatch.setattr(
        warmup, "_status", {"started": False, "finished": False, "stages": {}}, raising = False
    )
    runs: list[float] = []
    monkeypatch.setattr(warmup, "_STAGES", (("probe", lambda: runs.append(time.monotonic())),))
    yield runs
    warmup.join_background_warm(30)


def test_a_gated_warm_is_in_progress_but_runs_no_stage_until_the_gate_opens(recorded, monkeypatch):
    monkeypatch.setattr(warmup, "START_GATE_MAX_WAIT_S", 60.0)
    gate = threading.Event()
    assert warmup.start_background_warm(gate = gate) is True

    # The marker the desktop watchdog keys its startup grace on is up from the start, as before.
    status = warmup.warm_status()
    assert status["started"] and status["alive"] and not status["finished"]
    time.sleep(0.3)
    assert recorded == [], "a warm stage ran before the first response was sent"

    opened = time.monotonic()
    gate.set()
    assert warmup.join_background_warm(30) is True
    assert len(recorded) == 1 and recorded[0] >= opened
    assert warmup.warm_status()["finished"] is True


def test_the_warm_starts_anyway_when_no_request_arrives(recorded, monkeypatch, caplog):
    monkeypatch.setattr(warmup, "START_GATE_MAX_WAIT_S", 0.05)
    assert warmup.start_background_warm(gate = threading.Event()) is True
    assert warmup.join_background_warm(30) is True
    assert len(recorded) == 1, "a launch nobody connects to never warmed"


def test_an_ungated_warm_starts_at_once(recorded, monkeypatch):
    """Control: callers that pass no gate (and the code before the gate existed) warm immediately."""
    monkeypatch.setattr(warmup, "START_GATE_MAX_WAIT_S", 60.0, raising = False)
    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True
    assert len(recorded) == 1


def test_a_shutdown_while_parked_runs_no_stage(recorded, monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(warmup, "START_GATE_MAX_WAIT_S", 60.0)
    gate = threading.Event()
    assert warmup.start_background_warm(gate = gate) is True
    hw.invalidate_detection()
    gate.set()
    assert warmup.join_background_warm(30) is True
    assert recorded == [], "a warm parked at the gate ran after its lifespan ended"


def _http_scope(path: str = "/api/liveness") -> dict:
    return {"type": "http", "method": "GET", "path": path, "headers": []}


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_the_gate_opens_only_once_the_last_body_chunk_is_sent():
    event = threading.Event()
    seen: list[tuple[str, bool]] = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await send({"type": "http.response.body", "body": b"b", "more_body": False})

    async def send(message):
        seen.append((message["type"], event.is_set()))

    _run(_gate_middleware(app, event)(_http_scope(), _receive, send))
    assert seen == [
        ("http.response.start", False),
        ("http.response.body", False),
        ("http.response.body", False),
    ]
    assert event.is_set()


def test_a_failed_first_request_still_opens_the_gate():
    event = threading.Event()

    async def app(scope, receive, send):
        raise RuntimeError("boom")

    async def send(message):
        pass

    with pytest.raises(RuntimeError):
        _run(_gate_middleware(app, event)(_http_scope(), _receive, send))
    assert event.is_set()


def test_lifespan_and_websocket_scopes_do_not_open_the_gate():
    event = threading.Event()
    calls: list[str] = []

    async def app(scope, receive, send):
        calls.append(scope["type"])

    async def send(message):
        pass

    middleware = _gate_middleware(app, event)
    _run(middleware({"type": "lifespan"}, _receive, send))
    _run(middleware({"type": "websocket", "path": "/ws", "headers": []}, _receive, send))
    assert calls == ["lifespan", "websocket"]
    assert not event.is_set()


def test_a_real_request_reaches_the_client_before_the_warm_stage_runs(recorded, monkeypatch):
    """End to end over Starlette: the stage's start time is after the client already has the reply."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from utils.first_response_gate import FirstResponseGateMiddleware

    monkeypatch.setattr(warmup, "START_GATE_MAX_WAIT_S", 60.0)
    gate = threading.Event()
    app = FastAPI()

    @app.get("/api/liveness")
    async def liveness():
        return {"status": "alive"}

    app.add_middleware(FirstResponseGateMiddleware, event = gate)
    assert warmup.start_background_warm(gate = gate) is True
    with TestClient(app) as client:
        time.sleep(0.2)
        assert recorded == []
        assert client.get("/api/liveness").json() == {"status": "alive"}
        answered = time.monotonic()
        assert warmup.join_background_warm(30) is True
    assert len(recorded) == 1
    # The stage starts once the body is sent; the client may read it a hair later, so allow scheduling slack.
    assert recorded[0] >= answered - 0.5


def _lifespan_node() -> ast.AsyncFunctionDef:
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            return node
    raise AssertionError("lifespan not found in main.py")


def test_the_lifespan_gates_the_warm_on_the_first_response():
    calls = [
        sub
        for sub in ast.walk(_lifespan_node())
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "start_background_warm"
    ]
    assert len(calls) == 1
    gates = [kw.value for kw in calls[0].keywords if kw.arg == "gate"]
    assert (
        gates and isinstance(gates[0], ast.Name) and gates[0].id == "_first_response_sent"
    ), "the lifespan starts the warm ungated, so its first GIL-holding import races the first probe"


def test_shutdown_opens_the_gate_after_retiring_the_epoch():
    source = ast.unparse(_lifespan_node())
    head, _, shutdown = source.partition("yield")
    assert "_first_response_sent.clear()" in head
    assert "_first_response_sent.set()" in shutdown
    assert shutdown.index("_invalidate_detection()") < shutdown.index(
        "_first_response_sent.set()"
    ), "opening the gate before the epoch is retired lets a parked warm run after shutdown"


def test_the_app_registers_the_gate_middleware_on_the_same_event():
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_middleware"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "FirstResponseGateMiddleware"
        ):
            event = [kw.value for kw in node.keywords if kw.arg == "event"]
            found = (
                bool(event)
                and isinstance(event[0], ast.Name)
                and event[0].id == "_first_response_sent"
            )
    assert found
