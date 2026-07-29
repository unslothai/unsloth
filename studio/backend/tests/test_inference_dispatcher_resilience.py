# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference dispatcher resilience.

The dispatcher thread is the sole consumer of the response queue; if a malformed
response killed it, every in-flight generation would hang forever. A bad response
must be logged and skipped, not fatal. Fakes only.
"""

from __future__ import annotations

import ast
import queue
import sys
import threading
import time
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.orchestrator import InferenceOrchestrator  # noqa: E402


class _ScriptedQueue:
    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout = None):
        if self._items:
            return self._items.pop(0)
        raise queue.Empty


def _dispatcher():
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._dispatcher_stop = threading.Event()
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._request_cancel_events = {}
    return o


def test_dispatcher_survives_malformed_response_and_routes_next():
    o = _dispatcher()
    rid = "req-1"
    mbox = queue.Queue()
    o._mailboxes = {rid: mbox}
    # A non-dict response (resp.get -> AttributeError) must not kill the loop;
    # the following valid response must still reach its mailbox.
    o._resp_queue = _ScriptedQueue([12345, {"request_id": rid, "type": "token", "text": "hi"}])

    t = threading.Thread(target = o._dispatcher_loop, daemon = True)
    t.start()
    try:
        got = mbox.get(timeout = 5)
        assert got["text"] == "hi", "valid response must route despite the prior bad one"
        assert t.is_alive(), "dispatcher must survive a malformed response"
    finally:
        o._dispatcher_stop.set()
        t.join(timeout = 5)
    assert not t.is_alive()


def test_dispatcher_survives_mailbox_put_error():
    o = _dispatcher()
    rid = "req-2"

    class _BadMailbox:
        def put(self, _resp):
            raise RuntimeError("mailbox is broken")

    good = queue.Queue()
    o._mailboxes = {rid: _BadMailbox(), "req-3": good}
    o._resp_queue = _ScriptedQueue(
        [
            {"request_id": rid, "type": "token", "text": "boom"},
            {"request_id": "req-3", "type": "token", "text": "ok"},
        ]
    )

    t = threading.Thread(target = o._dispatcher_loop, daemon = True)
    t.start()
    try:
        got = good.get(timeout = 5)
        assert got["text"] == "ok"
        assert t.is_alive()
    finally:
        o._dispatcher_stop.set()
        t.join(timeout = 5)
    assert not t.is_alive()


def test_route_llama_streaming_async_clients_disable_proxy_env():
    """Local llama-server streaming proxies must ignore ambient HTTP_PROXY."""
    source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "AsyncClient"
            and isinstance(func.value, ast.Name)
            and func.value.id == "httpx"
        ):
            continue
        calls.append(node)

    assert len(calls) == 5
    for call in calls:
        assert any(
            kw.arg == "trust_env" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in call.keywords
        ), f"httpx.AsyncClient at line {call.lineno} must set trust_env=False"


def _direct_reader_host():
    """Orchestrator with only what _direct_reader and the ownership helpers touch."""
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._mailbox_lock = threading.Lock()
    o._mailboxes = {}
    o._direct_mailboxes = {}
    o._request_cancel_events = {}
    o._active_cancel_lock = threading.Lock()
    o._active_cancel_events = []
    o._executing_cancel_events = []
    o._dispatcher_thread = None
    return o


def test_rerouting_a_foreign_response_moves_worker_ownership():
    # A _gen_lock reader already blocked on resp_queue can beat the compare dispatcher to
    # that request's first response. The compare consumer passes mark_started=False, so if
    # this path does not promote it nothing does: the direct request stays recorded as the
    # executor, so the compare chat's Stop is ignored and a late reset from the direct one
    # cancels the compare generation instead.
    o = _direct_reader_host()
    mine, theirs = threading.Event(), threading.Event()
    o._request_cancel_events = {"mine": mine, "theirs": theirs}
    o._claim_worker(mine)
    o._mark_worker_started(mine)
    o._claim_worker(theirs)
    compare_mailbox = queue.Queue()
    o._mailboxes["theirs"] = compare_mailbox

    read_one, _drain, release = _direct_reader_calls(o, "mine")
    o._scripted = [{"request_id": "theirs", "type": "token", "text": "hi"}]

    assert read_one(timeout = 0.1) is None, "a foreign response is routed, not returned"
    assert compare_mailbox.get_nowait()["text"] == "hi"
    assert o._owns_worker(theirs), "the compare request is the one the worker answered"
    assert not o._owns_worker(mine), "so a late reset from the direct request must not fire"
    release()


def test_rerouting_a_foreign_gen_done_retires_that_request():
    # The other half of the dispatcher's move: once its last response is routed, the
    # request no longer owns the worker, or a Stop for it would end whatever starts next.
    o = _direct_reader_host()
    mine, theirs = threading.Event(), threading.Event()
    o._request_cancel_events = {"mine": mine, "theirs": theirs}
    o._claim_worker(theirs)
    o._mark_worker_started(theirs)
    o._claim_worker(mine)
    o._mailboxes["theirs"] = queue.Queue()

    read_one, _drain, release = _direct_reader_calls(o, "mine")
    o._scripted = [{"request_id": "theirs", "type": "gen_done"}]

    assert read_one(timeout = 0.1) is None
    assert not o._owns_worker(theirs), "retired once its last response was routed"
    assert o._owns_worker(mine), "the next claim takes over"
    release()


def _direct_reader_calls(o, request_id):
    """_direct_reader wired to a scripted _read_resp (o._scripted, popped in order)."""
    o._read_resp = lambda timeout = 1.0: o._scripted.pop(0) if o._scripted else None
    return o._direct_reader(request_id)
