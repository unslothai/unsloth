# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What each tool loop does when a local tool raises.

``_session_in_flight`` no longer swallows exceptions, so a failing ``python`` /
``terminal`` / ``edit_file`` call now propagates instead of coming back as
``"Unknown tool: <name>"``. ``studio_tool_loop`` and ``safetensors_agentic``
already turned that into a model-visible result; ``llama_cpp`` did not, so a bad
argument killed the whole GGUF answer.

Each loop is asserted against its own contract, not forced into one shape.
Reuses the fake llama-server and fake-transport harnesses next door, so no
model, subprocess, GPU or network is involved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Backend root plus the tests dir: the two harnesses this borrows sit alongside
# and import as top-level modules.
_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from test_llama_cpp_tool_loop import (  # noqa: E402
    _done,
    _make_backend,
    _sse,
    _structured_tool_call,
)

# The reported call: `code` is a number, so `_python_exec` fails on `.strip()`
# before anything runs.
BAD_PYTHON = {"code": 42}
REAL_ERROR = "'int' object has no attribute 'strip'"


def _gguf_events(
    monkeypatch,
    arguments,
    tool_name = "python",
):
    first = _structured_tool_call(tool_name, arguments, "call_bad_arg")
    second = [_sse({"content": "I will fix the argument."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first, second], payloads)
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run some python"}],
            tools = [{"type": "function", "function": {"name": tool_name}}],
            max_tool_iterations = 2,
        )
    ), payloads


@pytest.mark.parametrize(
    "tool_name, arguments",
    [("python", BAD_PYTHON), ("terminal", {"command": 42})],
)
def test_a_raising_local_tool_does_not_kill_the_gguf_answer(monkeypatch, tool_name, arguments):
    """The GGUF loop must report the tool's failure and keep answering.

    Unguarded, the per-iteration handler re-raises, the route reports a generic
    internal error, and the user loses a reply the model could have corrected.
    """
    events, _payloads = _gguf_events(monkeypatch, arguments, tool_name)

    ends = [e for e in events if e.get("type") == "tool_end"]
    assert len(ends) == 1, f"expected exactly one tool_end, got {ends}"
    assert REAL_ERROR in ends[0]["result"], ends[0]["result"]
    assert ends[0]["result"].startswith("Error: tool raised an exception:")
    assert "Unknown tool" not in ends[0]["result"]

    content = "".join(e.get("text", "") for e in events if e.get("type") == "content")
    assert (
        "I will fix the argument." in content
    ), "the loop stopped instead of letting the model recover"


def test_the_gguf_loop_still_reports_a_genuinely_unknown_tool(monkeypatch):
    """The unknown-tool contract itself is untouched."""
    events, _payloads = _gguf_events(
        monkeypatch,
        {"x": 1},
        tool_name = "no_such_tool_at_all",
    )
    ends = [e for e in events if e.get("type") == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["result"] == "Unknown tool: no_such_tool_at_all"


def test_the_failing_tool_result_reaches_the_model(monkeypatch):
    """The error must be in the next request, or the model cannot correct it."""
    _events, payloads = _gguf_events(monkeypatch, BAD_PYTHON)
    assert len(payloads) >= 2, "no second turn: the loop did not continue"
    tool_messages = [m for m in payloads[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages, payloads[1]["messages"]
    assert REAL_ERROR in json.dumps(tool_messages)


# ── The external-provider loop ────────────────────────────────────


def test_the_studio_loop_reports_the_real_error_and_continues(monkeypatch):
    """stream_with_studio_tools already had the handler; prove it end to end."""
    import test_studio_tool_loop as studio_h
    from core.inference import studio_tool_loop as loop_mod

    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)

    transport = studio_h.FakeTransport(
        [
            [
                studio_h._sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_bad_arg",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": json.dumps(BAD_PYTHON),
                                },
                            }
                        ]
                    }
                ),
                studio_h._DONE,
            ],
            [studio_h._sse({"content": "I will fix the argument."}), studio_h._DONE],
        ]
    )
    lines = studio_h._run(transport, tools = [studio_h.PY], permission_mode = "off")

    ends = studio_h._events(lines, "tool_end")
    assert len(ends) == 1, ends
    assert ends[0]["result"].startswith("Error: tool raised an exception:")
    assert REAL_ERROR in ends[0]["result"]
    assert "Unknown tool" not in ends[0]["result"]
    assert "I will fix the argument." in studio_h._visible_text(lines)


def test_the_studio_loop_still_reports_a_genuinely_unknown_tool(monkeypatch):
    import test_studio_tool_loop as studio_h
    from core.inference import studio_tool_loop as loop_mod

    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)

    unknown = studio_h._tool("no_such_tool_at_all")
    transport = studio_h.FakeTransport(
        [
            [
                studio_h._sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_unknown",
                                "type": "function",
                                "function": {
                                    "name": "no_such_tool_at_all",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                ),
                studio_h._DONE,
            ],
            [studio_h._sse({"content": "ok"}), studio_h._DONE],
        ]
    )
    lines = studio_h._run(transport, tools = [unknown], permission_mode = "off")
    ends = studio_h._events(lines, "tool_end")
    assert len(ends) == 1
    assert ends[0]["result"] == "Unknown tool: no_such_tool_at_all"


# ── The duplicate-call ledger ─────────────────────────────────────


def test_a_repeated_failing_call_stays_bounded(monkeypatch):
    """The result's classification changes, so check the loop still ends.

    ``"Unknown tool: python"`` matches no ``TOOL_ERROR_PREFIXES``, so a failed
    call was filed as a *success* and an identical retry refused. ``"Error:
    ..."`` is a failure, so the retry is allowed -- right, since the model can
    now see what went wrong, but only while the loop stays bounded.
    """
    import test_studio_tool_loop as studio_h
    from core.inference import studio_tool_loop as loop_mod

    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)

    executions: list[dict] = []
    real_execute = loop_mod.execute_tool

    def _counting_execute(name, arguments, **kwargs):
        executions.append({"name": name, "arguments": arguments})
        return real_execute(name, arguments, **kwargs)

    monkeypatch.setattr(loop_mod, "execute_tool", _counting_execute)

    bad_turn = [
        studio_h._sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_same",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps(BAD_PYTHON),
                        },
                    }
                ]
            }
        ),
        studio_h._DONE,
    ]
    max_calls = 4
    # More identical turns scripted than the budget allows.
    transport = studio_h.FakeTransport([list(bad_turn) for _ in range(max_calls + 6)])
    lines = studio_h._run(
        transport,
        tools = [studio_h.PY],
        permission_mode = "off",
        max_calls = max_calls,
    )

    assert len(executions) <= max_calls, (
        f"the loop ran the same failing call {len(executions)} times "
        f"with a budget of {max_calls}"
    )
    ends = studio_h._events(lines, "tool_end")
    assert ends, "no tool_end at all"
    # Every execution reports the real error; the trailing card is the
    # controller's budget notice, which is how the loop says it stopped.
    executed_ends = [e for e in ends if REAL_ERROR in e["result"]]
    assert len(executed_ends) == len(executions), [e["result"] for e in ends]
    assert "limit was reached" in ends[-1]["result"], ends[-1]["result"]
    assert not any("Unknown tool" in e["result"] for e in ends)
    assert transport.turns, "the loop consumed every scripted turn instead of stopping"


# ── research_runs stays outside the blast radius ──────────────────


def test_research_only_calls_tools_that_are_not_session_guarded():
    """research_runs has no enclosing tool handler, so it must stay clear.

    Its safety is a property of which tools it names. Grow that list with a
    guarded tool and this fails until research_runs handles them too.
    """
    import ast
    import inspect

    from core import research_runs

    guarded = {"python", "terminal", "edit_file"}
    tree = ast.parse(inspect.getsource(research_runs))
    named: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # execute_tool(name, ...) directly, or via asyncio.to_thread(execute_tool, name, ...)
        if getattr(func, "id", None) == "execute_tool" and node.args:
            first = node.args[0]
        elif (
            getattr(func, "attr", None) == "to_thread"
            and len(node.args) >= 2
            and getattr(node.args[0], "id", None) == "execute_tool"
        ):
            first = node.args[1]
        else:
            continue
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            named.add(first.value)
        else:
            named.add(f"<dynamic:{ast.dump(first)[:60]}>")

    assert named, "no execute_tool call sites found; the scan is broken"
    assert not (named & guarded), (
        f"research_runs now calls a session-guarded tool: {sorted(named & guarded)}. "
        "It has no enclosing tool-exception handler, so give it one first."
    )
    assert all(
        not n.startswith("<dynamic:") for n in named
    ), f"a research_runs tool name is computed, so this scan cannot vouch for it: {named}"
