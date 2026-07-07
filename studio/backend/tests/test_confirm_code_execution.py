# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``confirm_code_execution``: a narrower confirmation gate that
pauses only before local code-execution tools (python/terminal) while other
tools (web_search, render_html, ...) still run without a prompt.

These drive the real ``run_safetensors_tool_loop`` with hand-crafted fake
generators (no model), mirroring ``test_tool_confirm_loop.py``, and cover the
scoping predicate the route layer uses to require streaming.
"""

import pytest

from core.inference.safetensors_agentic import run_safetensors_tool_loop
from state import tool_approvals
from state.tool_approvals import resolve_tool_decision

_SESSION = "code-exec-session"

_TOOLS = [
    {"type": "function", "function": {"name": "python"}},
    {"type": "function", "function": {"name": "terminal"}},
    {"type": "function", "function": {"name": "web_search"}},
]


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
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        return f"RESULT[{name}]"


def _tool_call(name, args_json):
    return f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'


def _multi_turn(turns):
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            yield next(turn_iter)
        except StopIteration:
            return

    return _gen


def _drive(turns, decisions, **loop_kwargs):
    """Run the loop, resolving each gated tool_start with the next decision.

    Non-gated calls (awaiting_confirmation False) execute without consuming a
    decision. Returns (events, execute_calls).
    """
    decision_iter = iter(decisions)
    exec_fn = _FakeExecuteTool()
    gen = run_safetensors_tool_loop(
        single_turn = _multi_turn(turns),
        messages = [{"role": "user", "content": "hi"}],
        tools = _TOOLS,
        execute_tool = exec_fn,
        session_id = _SESSION,
        **loop_kwargs,
    )
    events = []
    for ev in gen:
        events.append(ev)
        if ev["type"] == "tool_start" and ev.get("awaiting_confirmation"):
            resolve_tool_decision(ev["approval_id"], next(decision_iter), session_id = _SESSION)
    return events, exec_fn.calls


def _starts(events):
    return [e for e in events if e["type"] == "tool_start"]


def _ends(events):
    return [e for e in events if e["type"] == "tool_end"]


# ── confirm_code_execution gates only python/terminal ────────────────────────


def test_python_call_is_gated_and_executes_on_allow():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "done"],
        ["allow"],
        confirm_code_execution = True,
    )
    starts = _starts(events)
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    assert starts[0]["approval_id"]
    assert calls == [("python", {"code": "print(1)"})]
    assert _ends(events)[0]["result"] == "RESULT[python]"


def test_terminal_call_is_gated_and_skipped_on_deny():
    events, calls = _drive(
        [_tool_call("terminal", '{"command": "ls"}'), "done"],
        ["deny"],
        confirm_code_execution = True,
    )
    starts = _starts(events)
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is True
    # Denied: the tool never runs.
    assert calls == []


def test_web_search_is_not_gated_by_confirm_code_execution():
    # No decision is supplied: a gated call would block waiting for one.
    events, calls = _drive(
        [_tool_call("web_search", '{"query": "cats"}'), "done"],
        [],
        confirm_code_execution = True,
    )
    starts = _starts(events)
    assert len(starts) == 1
    assert starts[0]["awaiting_confirmation"] is False
    assert not starts[0]["approval_id"]
    assert calls == [("web_search", {"query": "cats"})]


def test_bypass_permissions_overrides_confirm_code_execution():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "done"],
        [],
        confirm_code_execution = True,
        bypass_permissions = True,
    )
    starts = _starts(events)
    assert starts[0]["awaiting_confirmation"] is False
    assert calls == [("python", {"code": "print(1)"})]


def test_default_off_does_not_gate_code_execution():
    events, calls = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "done"],
        [],
        # Neither flag set: unchanged legacy behavior, python runs immediately.
    )
    starts = _starts(events)
    assert starts[0]["awaiting_confirmation"] is False
    assert calls == [("python", {"code": "print(1)"})]


def test_confirm_tool_calls_still_gates_every_tool():
    # confirm_tool_calls is the broad gate; web_search is prompted under it even
    # though confirm_code_execution would not touch it.
    events, calls = _drive(
        [_tool_call("web_search", '{"query": "cats"}'), "done"],
        ["allow"],
        confirm_tool_calls = True,
    )
    starts = _starts(events)
    assert starts[0]["awaiting_confirmation"] is True
    assert calls == [("web_search", {"query": "cats"})]


# ── scoping predicates used by the route streaming requirement ───────────────


def _spec(name):
    return {"type": "function", "function": {"name": name}}


def test_enables_code_execution_tool_predicate():
    from routes.inference import _enables_code_execution_tool

    assert _enables_code_execution_tool([_spec("python")]) is True
    assert _enables_code_execution_tool([_spec("terminal")]) is True
    assert _enables_code_execution_tool([_spec("web_search"), _spec("python")]) is True
    # A non-code tool list must not trip the streaming requirement.
    assert _enables_code_execution_tool([_spec("web_search"), _spec("render_html")]) is False
    assert _enables_code_execution_tool([]) is False
    assert _enables_code_execution_tool(None) is False


def test_payload_may_enable_code_execution_predicate(monkeypatch):
    import routes.inference as inf

    monkeypatch.setattr("state.tool_policy.get_tool_policy", lambda: None)

    class _P:
        def __init__(
            self,
            enable_tools = None,
            enabled_tools = None,
        ):
            self.enable_tools = enable_tools
            self.enabled_tools = enabled_tools

    # Built-ins off -> never code execution, even if enabled_tools lists python.
    assert inf._payload_may_enable_code_execution(_P(enable_tools = None)) is False
    assert (
        inf._payload_may_enable_code_execution(_P(enable_tools = False, enabled_tools = ["python"]))
        is False
    )
    # Built-ins on, explicit filter without a code tool -> not code execution (the fix).
    assert (
        inf._payload_may_enable_code_execution(_P(enable_tools = True, enabled_tools = ["web_search"]))
        is False
    )
    # Built-ins on, code tool in the filter -> code execution.
    assert (
        inf._payload_may_enable_code_execution(_P(enable_tools = True, enabled_tools = ["terminal"]))
        is True
    )
    # Built-ins on, no filter -> all built-ins including python/terminal.
    assert inf._payload_may_enable_code_execution(_P(enable_tools = True)) is True
