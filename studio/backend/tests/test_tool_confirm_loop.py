# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration tests for the confirmation gate inside the real tool loop.

These drive ``run_safetensors_tool_loop`` (no model -- hand-crafted fake
generators) with ``confirm_tool_calls=True`` and resolve each pending
decision inline. The slot is registered before ``tool_start`` is yielded,
so resolving right after receiving that event always lands before the
loop blocks. Covers: allow executes once, deny skips execution and feeds
back the rejection, disabled/duplicate calls are not prompted, and a
denied call does not pollute duplicate detection.
"""

import pytest

from core.inference.safetensors_agentic import run_safetensors_tool_loop
from state import tool_approvals
from state.tool_approvals import TOOL_REJECTED_MESSAGE, resolve_tool_decision

_SESSION = "loop-session"


@pytest.fixture(autouse = True)
def _clear_pending():
    with tool_approvals._lock:
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()


class _FakeExecuteTool:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        return f"RESULT[{name}]"


def _tool_call(name, args_json):
    return f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'


def _multi_turn(turns):
    """A single_turn generator that yields one full snapshot per turn."""
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            yield next(turn_iter)
        except StopIteration:
            return

    return _gen


_DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "python"}},
    {"type": "function", "function": {"name": "web_search"}},
]


def _drive(
    turns,
    decisions,
    *,
    tools = None,
):
    """Run the loop, resolving each gated tool_start with the next decision.

    The advertised ``tools`` list drives the loop's enabled-tool filter
    (pass a list omitting a tool to make a call to it "disabled").
    Returns (events, execute_calls).
    """
    decision_iter = iter(decisions)
    exec_fn = _FakeExecuteTool()
    gen = run_safetensors_tool_loop(
        single_turn = _multi_turn(turns),
        messages = [{"role": "user", "content": "hi"}],
        tools = _DEFAULT_TOOLS if tools is None else tools,
        execute_tool = exec_fn,
        session_id = _SESSION,
        confirm_tool_calls = True,
        # These exercise the confirm-gate mechanics (allow/deny/reissue/dedup),
        # which require every call to prompt; "ask" gates all calls. (Unset now
        # defaults to "auto", which would only gate high-risk calls.)
        permission_mode = "ask",
    )
    events = []
    for ev in gen:
        events.append(ev)
        if ev["type"] == "tool_start" and ev.get("awaiting_confirmation"):
            # Slot is already registered (begin ran before this yield), so
            # the decision lands before the loop enters its blocking wait.
            resolve_tool_decision(ev["approval_id"], next(decision_iter), session_id = _SESSION)
    return events, exec_fn.calls


def _tool_starts(events):
    return [e for e in events if e["type"] == "tool_start"]


def _tool_ends(events):
    return [e for e in events if e["type"] == "tool_end"]


def test_allow_executes_the_tool_once():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final answer"],
        ["allow"],
    )
    starts = _tool_starts(events)
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    assert starts[0]["approval_id"]
    assert calls == [("python", {"code": "print(1)"})]
    assert _tool_ends(events)[0]["result"] == "RESULT[python]"


def test_deny_skips_execution_and_feeds_rejection():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final answer"],
        ["deny"],
    )
    assert calls == []  # tool never ran
    assert _tool_ends(events)[0]["result"] == TOOL_REJECTED_MESSAGE


def test_disabled_tool_is_not_prompted():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final answer"],
        [],
        tools = [{"type": "function", "function": {"name": "web_search"}}],
    )
    assert _tool_starts(events) == []
    assert _tool_ends(events) == []
    assert calls == []


def test_duplicate_call_is_not_prompted():
    same = _tool_call("python", '{"code": "print(1)"}')
    events, calls = _drive([same, same, "final answer"], ["allow"])
    starts = _tool_starts(events)
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    assert calls == [("python", {"code": "print(1)"})]
    assert len(_tool_ends(events)) == 1


def test_denied_call_can_be_reissued_and_approved():
    # Deny, then the model re-issues the identical call -> approving it must
    # execute, not get suppressed as a duplicate (denied calls are not added
    # to the duplicate-detection history).
    same = _tool_call("python", '{"code": "print(1)"}')
    events, calls = _drive([same, same, "final answer"], ["deny", "allow"])
    starts = _tool_starts(events)
    assert len(starts) == 2
    assert starts[0]["awaiting_confirmation"] is True
    assert starts[1]["awaiting_confirmation"] is True  # not treated as dup
    assert calls == [("python", {"code": "print(1)"})]  # ran once, on approve
    ends = _tool_ends(events)
    assert ends[0]["result"] == TOOL_REJECTED_MESSAGE
    assert ends[1]["result"] == "RESULT[python]"
