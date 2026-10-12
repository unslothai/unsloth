# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Closing the local GGUF tool loop mid-round tells every overlapped call to stop.

The calls of a parallel round are driven on threads of their own, so a close while the first
is being read left the later ones running with nothing to say the request had ended: a tool
that only stops on the cancel event went on to its side effect after the response was gone.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402
from test_studio_tool_loop import WEB  # noqa: E402


def _call(index, call_id, query):
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": query})},
    }


def test_a_close_while_draining_the_first_call_stops_the_second(monkeypatch):
    first_running = threading.Event()
    outcome: dict = {}
    finished = threading.Event()

    def _execute(name, arguments, *, cancel_event = None, output_callback = None, **_kwargs):
        query = (arguments or {}).get("query")
        if query == "alpha":
            first_running.set()
            # Speaks while it works, so the loop is reading this call when it is closed.
            for _ in range(200):
                if cancel_event is not None and cancel_event.is_set():
                    return "alpha-stopped"
                if output_callback is not None:
                    output_callback("tick ")
                time.sleep(0.02)
            return "alpha-done"
        # The sibling only stops on the request's cancel event.
        stopped = cancel_event is not None and cancel_event.wait(3.0)
        outcome["beta"] = "stopped" if stopped else "side effect after the request ended"
        finished.set()
        return "beta"

    backend = _make_backend(
        monkeypatch,
        [
            [_gguf_sse({"tool_calls": [_call(0, "call_a", "alpha"), _call(1, "call_b", "beta")]}), _gguf_done()],
            [_gguf_sse({"content": "Final answer."}), _gguf_done()],
        ],
        [],
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
    gen = backend.generate_chat_completion_with_tools(
        messages = [{"role": "user", "content": "go"}],
        tools = [WEB],
        max_tool_iterations = 2,
        cancel_event = threading.Event(),
    )
    for event in gen:
        if first_running.is_set() and isinstance(event, dict) and event.get("type") == "tool_output":
            break
    else:
        raise AssertionError("the round never reached its first call's output")
    gen.close()
    assert finished.wait(5.0), "the sibling never returned"
    assert outcome["beta"] == "stopped", outcome["beta"]


def test_an_ordinary_error_in_the_drain_is_still_reported(monkeypatch):
    # Setting the request's cancel event there would make the handler above end the turn
    # silently instead of reporting the failure.
    import pytest

    from core.inference.tool_loop_controller import ToolLoopController

    def _execute(name, arguments, **_kwargs):
        return "result"

    def _broken(self, *args, **kwargs):
        raise RuntimeError("ledger broke")

    backend = _make_backend(
        monkeypatch,
        [
            [_gguf_sse({"tool_calls": [_call(0, "call_a", "alpha"), _call(1, "call_b", "beta")]}), _gguf_done()],
            [_gguf_sse({"content": "Final answer."}), _gguf_done()],
        ],
        [],
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", _execute)
    monkeypatch.setattr(ToolLoopController, "record_result", _broken)
    cancel = threading.Event()
    with pytest.raises(Exception):
        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "go"}],
                tools = [WEB],
                max_tool_iterations = 2,
                cancel_event = cancel,
            )
        )
    assert not cancel.is_set()
