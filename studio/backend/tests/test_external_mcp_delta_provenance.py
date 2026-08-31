# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MCP display name has to reach the card the provider's own delta paints.

This loop relays that delta as it came, and it carries no provenance, so the
client labelled the card with the internal server id until the real tool_start
landed -- which is only after the whole turn finished streaming. A stamp riding
on the same chunk relabels it immediately.

It rides on the chunk rather than arriving as a second card event because the
client accumulates arguments by appending to what the card already holds: an
event carrying its own (empty) arguments would reset that text mid-stream and
every later fragment would extend a corrupted string.
"""

from __future__ import annotations

import json
import threading

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference import tool_loop_controller as controller_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)

_DONE = "data: [DONE]"

MCP_NAME = "mcp__a3f9c1d2e4b6f807__create_issue"
DISPLAY = "GitHub"


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _delta(
    fragment: str = "",
    arguments: str = "",
    call_id: str = "c1",
) -> str:
    function: dict = {}
    if fragment:
        function["name"] = fragment
    if arguments:
        function["arguments"] = arguments
    return "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": call_id, "type": "function", "function": function}
                        ]
                    },
                }
            ]
        }
    )


def _finish(reason: str = "tool_calls") -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class FakeTransport:
    heals_text_tool_calls = False

    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]
        self.requests: list[dict] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append({"messages": [dict(m) for m in messages]})
        assert len(self.requests) <= 10, "loop never terminated"
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.fixture
def named(monkeypatch):
    """Resolve only the one MCP name, so an incomplete one cannot look valid."""

    def _parts(tool_name: str):
        return (DISPLAY, "create_issue") if tool_name == MCP_NAME else None

    # Two references to patch: this loop imported its own, and
    # provisional_tool_provenance reads the controller's module global.
    monkeypatch.setattr(loop_mod, "mcp_display_parts", _parts)
    monkeypatch.setattr(controller_mod, "mcp_display_parts", _parts)
    monkeypatch.setattr(loop_mod, "execute_tool", lambda name, arguments, **kw: "ok")
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)


def _run(transport, tools):
    import asyncio
    async def _collect():
        out: list[str] = []
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
                model = "m",
            ),
            policy = ToolLoopPolicy(
                tools = tools,
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(_collect())


def _stamps(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        if not line.startswith("data: ") or line == _DONE:
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("_mcp_provenance"):
            out.append(payload["_mcp_provenance"])
    return out


def _index_of_first_tool_start(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if '"type": "tool_start"' in line or '"type":"tool_start"' in line:
            return i
    return -1


def test_the_display_name_rides_on_the_delta_that_completes_the_name(named):
    lines = _run(
        FakeTransport([[_delta(MCP_NAME), _delta(arguments = '{"a":1}'), _finish()], [_DONE]]),
        [_tool(MCP_NAME)],
    )
    stamps = _stamps(lines)
    assert stamps, "the streamed delta carried no display name"
    assert stamps[0] == {"c1": stamps[0]["c1"]}
    assert stamps[0]["c1"]["mcp_server"] == DISPLAY


def test_the_stamp_beats_the_real_tool_start(named):
    """The whole point: the card is relabelled while the turn is still streaming."""
    lines = _run(
        FakeTransport([[_delta(MCP_NAME), _delta(arguments = "{}"), _finish()], [_DONE]]),
        [_tool(MCP_NAME)],
    )
    first_stamp = next(i for i, line in enumerate(lines) if '"_mcp_provenance"' in line)
    start = _index_of_first_tool_start(lines)
    assert start != -1, "the turn never ran the tool"
    assert first_stamp < start


def test_it_is_keyed_by_the_streamed_call_id_and_sent_once(named):
    lines = _run(
        FakeTransport(
            [
                [
                    _delta(MCP_NAME),
                    _delta(arguments = '{"a":'),
                    _delta(arguments = "1}"),
                    _finish(),
                ],
                [_DONE],
            ]
        ),
        [_tool(MCP_NAME)],
    )
    stamps = _stamps(lines)
    assert len(stamps) == 1, "the stamp repeated on later argument fragments"
    assert list(stamps[0]) == ["c1"]


def test_a_name_split_across_fragments_waits_until_it_is_whole(named):
    """``mcp__srv__cre`` is well formed too, so only the declared name may stamp."""
    lines = _run(
        FakeTransport(
            [
                [
                    _delta("mcp__a3f9c1d2e4b6f807__cre"),
                    _delta("ate_issue"),
                    _delta(arguments = "{}"),
                    _finish(),
                ],
                [_DONE],
            ]
        ),
        [_tool(MCP_NAME)],
    )
    stamped_at = [i for i, line in enumerate(lines) if '"_mcp_provenance"' in line]
    assert len(stamped_at) == 1
    # Not on the first fragment: that one named a tool that does not exist.
    assert '"cre"' not in lines[stamped_at[0]]
    assert _stamps(lines)[0]["c1"]["mcp_server"] == DISPLAY


def test_a_plain_tool_is_left_alone(named):
    lines = _run(
        FakeTransport([[_delta("web_search"), _delta(arguments = "{}"), _finish()], [_DONE]]),
        [_tool("web_search")],
    )
    assert _stamps(lines) == []


def test_an_undeclared_mcp_name_never_stamps(named):
    """A name the request did not offer cannot be trusted to name a real server."""
    lines = _run(
        FakeTransport([[_delta(MCP_NAME), _delta(arguments = "{}"), _finish()], [_DONE]]),
        [_tool("web_search")],
    )
    assert _stamps(lines) == []


def test_a_reused_call_id_is_named_again_on_the_next_turn(named):
    """Providers restart ids every turn, and the client drops its id mapping at
    tool_end, so the second ``c1`` is a different card that also needs naming."""
    lines = _run(
        FakeTransport(
            [
                [_delta(MCP_NAME), _delta(arguments = "{}"), _finish()],
                [_delta(MCP_NAME), _delta(arguments = "{}"), _finish()],
                [_DONE],
            ]
        ),
        [_tool(MCP_NAME)],
    )
    stamps = _stamps(lines)
    assert len(stamps) == 2, "the second turn's card kept the internal id"
    assert all(s["c1"]["mcp_server"] == DISPLAY for s in stamps)


def test_a_provider_cannot_forge_the_stamp(named):
    """The key is Studio's own: a provider that sends one must not be believed."""
    forged = "data: " + json.dumps(
        {
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
            "_mcp_provenance": {"c1": {"mcp_server": "Totally Legit"}},
        }
    )
    lines = _run(FakeTransport([[forged, _finish("stop")], [_DONE]]), [_tool(MCP_NAME)])
    assert all("Totally Legit" not in line for line in lines)
