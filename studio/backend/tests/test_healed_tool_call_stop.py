# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A healed text-form tool call must withhold its turn's finish_reason, like a structured one.

A provider that writes ``<tool_call>...</tool_call>`` as ordinary content instead of
``delta.tool_calls`` -- the small self-hosted GGUF models ``heals_text_tool_calls`` exists
for -- has that markup healed into a real call and executed by the loop. But the call never
reaches the wire as a ``tool_calls`` key, so ``ServerToolCallStripper`` cannot see it, and
the turn's ``finish_reason: "stop"`` used to reach a headerless caller intact. A client that
ends on the first finish reason then returns the sentence before the tool ran and never
reads the answer, which is the exact failure the control-frame gate exists to prevent.

These drive the real loop and the real stripper in the route's relay order, because the bug
lives in the interleaving: the loop must flag the call while the stripper is still upstream
of the chunk that closes the turn.
"""

import asyncio
import json
import threading

import pytest

import core.inference.studio_tool_loop as loop_mod
from core.inference.sse_control_frames import ServerToolCallStripper, is_ui_control_sse_line
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)


_DONE = "data: [DONE]"

WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _sse(delta = None, finish = None):
    choice = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]}, ensure_ascii = False)


_ANSWER_TURN = [_sse({"content": "The answer is 42."}), _sse(finish = "stop"), _DONE]

# The call arrives as content, closed tag: the healer promotes it during feed().
_TEXT_FORM_TURN = [
    _sse({"content": "Let me look that up. "}),
    _sse({"content": '<tool_call>{"name": "web_search", '}),
    _sse({"content": '"arguments": {"query": "42"}}</tool_call>'}),
    _sse(finish = "stop"),
    _DONE,
]

# Same call with the closing tag missing: promotion happens only in finalize(), after the
# provider already sent its finish_reason. Arming at promotion time would be too late here,
# which is why the loop holds the chunk back instead.
_UNTERMINATED_TURN = [
    _sse({"content": "Let me look that up. "}),
    _sse({"content": '<tool_call>{"name": "web_search", '}),
    _sse({"content": '"arguments": {"query": "42"}}'}),
    _sse(finish = "stop"),
    _DONE,
]

# The control: the same call in the structured shape the stripper already handled.
_STRUCTURED_TURN = [
    _sse({"content": "Let me look that up. "}),
    _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "function": {"name": "web_search", "arguments": '{"query":"42"}'},
                }
            ]
        }
    ),
    _sse(finish = "stop"),
    _DONE,
]


class _HealingTransport:
    """An OAI-compatible transport, i.e. one that heals text-form calls."""

    heals_text_tool_calls = True

    def __init__(self, turns):
        self._turns = [list(turn) for turn in turns]

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        lines = self._turns.pop(0) if self._turns else [_DONE]

        async def _generate():
            for line in lines:
                yield line

        return _generate()


@pytest.fixture
def loop_env(monkeypatch):
    """Run tools inline and keep RAG and the risk check out of the way."""
    executed: list[str] = []
    monkeypatch.setattr(
        loop_mod,
        "execute_tool",
        lambda name, arguments, **kwargs: (executed.append(name), f"RESULT<{name}>")[1],
    )
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    return executed


def _relay(turns, *, ui_events):
    """The route's relay for one request: drop control frames, then strip, lazily.

    Laziness matters. Draining the loop and stripping afterwards would let the loop's flag
    arrive before any line was stripped, which passes even when the fix is absent.
    """

    async def _run():
        stripper = ServerToolCallStripper()
        generator = stream_with_studio_tools(
            _HealingTransport(turns),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = "s1",
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [WEB_SEARCH],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
                on_withheld_tool_call = None if ui_events else stripper.arm,
                on_provider_turn_end = None if ui_events else stripper.end_turn,
            ),
            cancel_event = threading.Event(),
        )
        seen: list[str] = []
        async for line in generator:
            if not ui_events:
                if is_ui_control_sse_line(line):
                    continue
                line = stripper.strip(line)
                if line is None:
                    continue
            seen.append(line)
        if not ui_events:
            owed = stripper.owed_terminal_chunk()
            if owed:
                seen.append(owed)
        return seen

    return asyncio.run(asyncio.wait_for(_run(), timeout = 30.0))


def _finish_reasons(lines):
    reasons = []
    for line in lines:
        if not line.startswith("data: ") or line == _DONE:
            continue
        try:
            payload = json.loads(line[len("data: ") :])
        except ValueError:
            continue
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                reasons.append(choice["finish_reason"])
    return reasons


def _text(lines):
    out = []
    for line in lines:
        if not line.startswith("data: ") or line == _DONE:
            continue
        try:
            payload = json.loads(line[len("data: ") :])
        except ValueError:
            continue
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict):
                content = (choice.get("delta") or {}).get("content")
                if isinstance(content, str):
                    out.append(content)
    return "".join(out)


@pytest.mark.parametrize(
    "turns, label",
    [
        (_TEXT_FORM_TURN, "closed tag, promoted in feed()"),
        (_UNTERMINATED_TURN, "unterminated, promoted in finalize()"),
    ],
)
def test_a_healed_call_does_not_leak_its_turns_stop(loop_env, turns, label):
    lines = _relay([turns, _ANSWER_TURN], ui_events = False)

    assert loop_env == ["web_search"], f"the tool must actually run ({label})"
    # Exactly one finish_reason, and it arrives last: a client ending on the first one still
    # reads the post-tool answer.
    assert _finish_reasons(lines) == ["stop"], label
    assert "The answer is 42." in _text(lines), label
    assert _finish_reasons(lines[:-1]) == [], f"the terminal must be last ({label})"


def test_a_structured_call_behaves_the_same_way(loop_env):
    """The shape that already worked, as the reference the healed ones must match."""
    lines = _relay([_STRUCTURED_TURN, _ANSWER_TURN], ui_events = False)

    assert loop_env == ["web_search"]
    assert _finish_reasons(lines) == ["stop"]
    assert "The answer is 42." in _text(lines)


def test_the_opt_in_stream_still_sees_both_turns_reasons(loop_env):
    """Nothing is withheld from the Studio UI: it reads the cards and needs the real turns."""
    lines = _relay([_TEXT_FORM_TURN, _ANSWER_TURN], ui_events = True)

    assert loop_env == ["web_search"]
    assert _finish_reasons(lines) == ["stop", "stop"]
    assert any(is_ui_control_sse_line(line) for line in lines), "tool cards must survive"


def test_a_turn_with_no_call_keeps_its_only_finish_reason(loop_env):
    """The healer is live on every turn here, so the ordinary path must be untouched."""
    plain = [_sse({"content": "Just an answer."}), _sse(finish = "stop"), _DONE]
    lines = _relay([plain], ui_events = False)

    assert loop_env == []
    assert _finish_reasons(lines) == ["stop"]
    assert _text(lines) == "Just an answer."


def test_a_healed_call_owes_a_terminal_even_with_no_finish_chunk(loop_env):
    """A provider may close a turn on [DONE] alone, sending no finish_reason at all.

    Nothing is held back in that case, but the call is still promoted and run, so the debt
    has to be armed anyway. Otherwise a loop that then ends without a genuine terminal --
    here the second pass says nothing -- closes the stream on [DONE] carrying no
    finish_reason, which openai-node rejects with "missing finish_reason for choice 0".
    """
    no_finish = [
        _sse({"content": "Let me look that up. "}),
        _sse({"content": '<tool_call>{"name": "web_search", '}),
        _sse({"content": '"arguments": {"query": "42"}}</tool_call>'}),
        _DONE,
    ]
    lines = _relay([no_finish, [_DONE]], ui_events = False)

    assert loop_env == ["web_search"], "the tool must actually run"
    assert _finish_reasons(lines) == ["stop"], "a terminal must be minted"


def test_holding_the_turn_end_does_not_reorder_the_text(loop_env):
    """Only the finish_reason waits for healing; the content on that chunk goes out in place.

    The healer withholds any trailing run that could still become a tool marker, so a final
    delta ending in "<to" releases its prose and buffers the rest. Parking that whole chunk
    let the residue flushed by finalize() overtake the prose and reverse it on the wire --
    "Comparing: <tothe value " -- even though the conversation replay stayed correct. No
    tool call is involved: any healing turn whose last delta ends in "<" hits this.
    """
    trailing_marker = [
        _sse({"content": "Comparing: "}),
        _sse({"content": "the value <to"}, finish = "stop"),
        _DONE,
    ]
    lines = _relay([trailing_marker], ui_events = False)

    assert loop_env == [], "no tool call in this stream"
    assert _text(lines) == "Comparing: the value <to"
    assert _finish_reasons(lines) == ["stop"]
    # The reason is last, so a client reading in order sees the whole turn before it ends.
    assert _finish_reasons(lines[:-1]) == []


def test_the_opt_in_stream_keeps_that_order_too(loop_env):
    """The reordering happened inside the loop, upstream of the stripper, so it hit both."""
    trailing_marker = [
        _sse({"content": "Comparing: "}),
        _sse({"content": "the value <to"}, finish = "stop"),
        _DONE,
    ]
    lines = _relay([trailing_marker], ui_events = True)

    assert _text(lines) == "Comparing: the value <to"


def test_the_next_turns_legacy_call_keeps_its_own_reason(loop_env):
    """A withheld call must not reach past the turn it belonged to.

    The wire is not always enough to close a turn: a provider can end one on [DONE] alone,
    and the loop eats that sentinel rather than relaying it, so the withheld-call flag stayed
    raised into the next turn. A legacy delta.function_call there is the caller's own to
    dispatch, and it dispatches on the finish_reason, so stripping that reason as though it
    closed the previous call means the call never runs.
    """
    server_call_then_done = [
        _sse({"content": "looking. "}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "function": {"name": "web_search", "arguments": '{"query":"42"}'},
                    }
                ]
            }
        ),
        _DONE,
    ]
    legacy_offer = [
        _sse({"content": "now yours: "}),
        _sse({"function_call": {"name": "caller_tool", "arguments": '{"x":1}'}}),
        _sse(finish = "function_call"),
        _DONE,
    ]
    lines = _relay([server_call_then_done, legacy_offer], ui_events = False)

    assert loop_env == ["web_search"], "only the server call runs; the legacy one is the caller's"
    assert "function_call" in _text(lines) or any(
        '"function_call"' in line for line in lines
    ), "the legacy call itself must reach the caller"
    # Its own reason, not a minted stop: the caller dispatches on this.
    assert _finish_reasons(lines) == ["function_call"]


def test_a_truncated_turn_keeps_its_reason(loop_env):
    """ "length" cut the call off half-written, so the loop refuses to run it.

    Nothing follows, so that reason is genuinely the end of the response and withholding it
    would leave the caller with none at all.
    """
    truncated = list(_TEXT_FORM_TURN)
    truncated[-2] = _sse(finish = "length")
    lines = _relay([truncated, _ANSWER_TURN], ui_events = False)

    assert loop_env == [], "a truncated call must not run"
    assert _finish_reasons(lines) == ["length"]
