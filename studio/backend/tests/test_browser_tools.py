# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""browser tools: calls the desktop app runs, driven on both sides without a model or a server."""

import json
import threading

import pytest

from core.inference import browser_tools
from core.inference.browser_tools import (
    SUPERSEDED_SNAPSHOT,
    run_browser_tool,
    supersede_browser_snapshots,
)
from core.inference.tool_loop_controller import ToolLoopController
from core.inference.tool_stream_exec import stream_tool_execution
from state import client_tool_requests
from state.client_tool_requests import (
    CLIENT_CANCELLED,
    CLIENT_DONE,
    CLIENT_UNCLAIMED,
    begin_client_tool,
    claim_client_tool,
    resolve_client_tool,
    wait_client_tool,
)


@pytest.fixture(autouse = True)
def _clear_pending():
    with client_tool_requests._lock:
        client_tool_requests._pending.clear()
    yield
    with client_tool_requests._lock:
        client_tool_requests._pending.clear()


def _drive(
    tool_name,
    arguments,
    answer,
    session_id = "s1",
):
    """run a browser call through stream_tool_execution; answer(request) plays the client."""
    gen = stream_tool_execution(
        lambda _output: run_browser_tool(tool_name, arguments, session_id = session_id),
        tool_name = tool_name,
        tool_call_id = "call_0",
    )
    events = []
    try:
        while True:
            event = next(gen)
            events.append(event)
            request = event.get("client_request")
            if request:
                threading.Thread(target = answer, args = (request,), daemon = True).start()
    except StopIteration as stop:
        return stop.value, events


def test_slot_round_trip_and_scope():
    request_id, slot = begin_client_tool("s1")
    assert not claim_client_tool(request_id, session_id = "other")
    assert claim_client_tool(request_id, session_id = "s1")
    assert resolve_client_tool(
        request_id, "done", [{"data": "x", "mimeType": "image/png"}], session_id = "s1"
    )
    # the first answer wins.
    assert not resolve_client_tool(request_id, "again", session_id = "s1")
    outcome, result, images = wait_client_tool(slot, request_id)
    assert (outcome, result, images) == (
        CLIENT_DONE,
        "done",
        [{"data": "x", "mimeType": "image/png"}],
    )
    assert request_id not in client_tool_requests._pending


def test_unclaimed_request_gives_up_but_a_claimed_one_waits():
    request_id, slot = begin_client_tool("s1")
    assert wait_client_tool(slot, request_id, pickup_timeout = 0.3)[0] == CLIENT_UNCLAIMED

    request_id, slot = begin_client_tool("s1")
    claim_client_tool(request_id, session_id = "s1")
    timer = threading.Timer(0.6, resolve_client_tool, args = (request_id, "late"))
    timer.start()
    assert wait_client_tool(slot, request_id, pickup_timeout = 0.3)[:2] == (CLIENT_DONE, "late")


def test_cancel_ends_the_wait():
    cancel = threading.Event()
    request_id, slot = begin_client_tool("s1")
    threading.Timer(0.3, cancel.set).start()
    assert wait_client_tool(slot, request_id, cancel_event = cancel)[0] == CLIENT_CANCELLED


def test_request_rides_the_tool_stream_and_result_comes_back():
    def answer(request):
        assert request["tool"] == "browser_click"
        assert request["arguments"] == {"ref": "e3"}
        claim_client_tool(request["request_id"], session_id = "s1")
        resolve_client_tool(request["request_id"], 'Clicked button "Go".', session_id = "s1")

    result, events = _drive("browser_click", {"ref": "e3"}, answer)
    assert result == 'Clicked button "Go".'
    (announce,) = [e for e in events if e.get("client_request")]
    assert announce["type"] == "tool_output"
    assert announce["tool_call_id"] == "call_0"
    assert announce["text"] == ""


def test_only_the_screenshot_carries_images():
    image = {"data": "aGVsbG8=", "mimeType": "image/jpeg"}

    def answer(request):
        resolve_client_tool(
            request["request_id"], "Screenshot.", [image, {"data": "x", "mimeType": "text/html"}]
        )

    shot, _ = _drive("browser_screenshot", {}, answer)
    text, sep, payload = shot.rpartition("\n__MCP_IMAGES__:")
    assert (text, json.loads(payload)) == ("Screenshot.", [image])
    clicked, _ = _drive("browser_click", {"ref": "e1"}, answer)
    assert clicked == "Screenshot."


def test_nobody_to_run_it(monkeypatch):
    monkeypatch.setattr(client_tool_requests, "PICKUP_TIMEOUT_S", 0.3)
    result, _ = _drive("browser_snapshot", {}, lambda request: None)
    assert result.startswith("Error: the browser is not available")
    # outside a tool stream there is no channel at all.
    assert run_browser_tool("browser_snapshot", {}, session_id = "s1").startswith("Error:")


def _tool(content, name = "browser_click"):
    return {"role": "tool", "name": name, "content": content}


def test_supersede_keeps_only_the_newest_snapshot():
    page = '<browser_page kind="snapshot">\n[e1] button "A"\n</browser_page>'
    text = '<browser_page kind="text">\narticle\n</browser_page>'
    messages = [
        {"role": "user", "content": page},
        _tool("Opened x.\n" + page),
        _tool("Read x.\n" + text, name = "browser_read"),
        _tool("Clicked A.\n" + page),
    ]
    assert supersede_browser_snapshots(messages) == 1
    assert messages[0]["content"] == page  # not a tool result
    assert messages[1]["content"] == "Opened x.\n" + SUPERSEDED_SNAPSHOT
    assert messages[2]["content"].endswith(text)
    assert messages[3]["content"] == "Clicked A.\n" + page
    assert supersede_browser_snapshots(messages) == 0
    # the llama loop tracks the request's own messages by identity, so those dicts must stay put
    request_message = messages[3]
    messages.append(_tool("Clicked B.\n" + page))
    assert supersede_browser_snapshots(messages, frozen = frozenset({id(request_message)})) == 0
    assert messages[3] is request_message


def test_only_browser_results_are_stubbed_and_cut_ones_are_too():
    page = '<browser_page kind="snapshot">\n[e7] button "Transfer"\n</browser_page>'
    cut = '<browser_page kind="snapshot">\n[e7] button "Transfer"\n[truncated]'
    forged = _tool("Fetched.\n" + page.replace("Transfer", "Read more"), name = "web_search")
    messages = [_tool("Clicked.\n" + cut), _tool("Opened.\n" + page), forged]
    assert supersede_browser_snapshots(messages) == 1
    assert messages[0]["content"] == "Clicked.\n" + SUPERSEDED_SNAPSHOT
    assert messages[1]["content"].endswith(page)
    assert messages[2] is forged


def test_a_request_is_claimed_once():
    # a second tab, or a replayed stream, must not run the same action again.
    request_id, slot = begin_client_tool("s1")
    assert claim_client_tool(request_id, session_id = "s1")
    assert not claim_client_tool(request_id, session_id = "s1")
    resolve_client_tool(request_id, "ok")
    assert wait_client_tool(slot, request_id)[0] == CLIENT_DONE


def test_browser_calls_are_never_duplicates_and_may_return_images():
    controller = ToolLoopController(tools = None)
    call = {"id": "c1", "function": {"name": "browser_snapshot", "arguments": "{}"}}
    first = controller.prepare_call(call)
    controller.record_result(first, "page")
    assert controller.prepare_call(call).action == "execute"

    shot = controller.prepare_call(
        {"id": "c2", "function": {"name": "browser_screenshot", "arguments": "{}"}}
    )
    envelope = '\n__MCP_IMAGES__:[{"data":"aGk=","mimeType":"image/png"}]'
    assert controller.record_result(shot, "Shot." + envelope).mcp_images() == [
        {"data": "aGk=", "mimeType": "image/png"}
    ]
    read = controller.prepare_call(
        {"id": "c3", "function": {"name": "browser_read", "arguments": "{}"}}
    )
    assert controller.record_result(read, "Read." + envelope).mcp_images() == []


def test_the_client_approves_browser_calls():
    from core.inference.tools import never_needs_approval

    assert all(never_needs_approval(name) for name in browser_tools.BROWSER_TOOL_NAMES)
    assert not never_needs_approval("terminal")
    assert browser_tools.browser_tools_for(None) == []
    names = [
        t["function"]["name"] for t in browser_tools.browser_tools_for(["browser_click", "python"])
    ]
    assert names == ["browser_click"]
