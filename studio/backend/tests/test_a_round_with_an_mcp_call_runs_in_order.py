# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MCP tool can change what a sibling call reads or writes, and an HTTP MCP session does not
serialise its calls. Two edits of one document overlapped each read the old copy, so one update
was lost while both reported success. A round holding an MCP call runs in the model's order in
both tool loops; built-in reads alone still overlap."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import studio_tool_loop as loop_mod  # noqa: E402
from core.inference.mcp_client import MCP_TOOL_PREFIX  # noqa: E402
from test_studio_tool_loop import FakeTransport, _DONE, _run, _sse, _tool  # noqa: E402
from test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402

_EDIT = f"{MCP_TOOL_PREFIX}docs__append_line"
_EDIT_TOOL = _tool(_EDIT)


@pytest.fixture(autouse = True)
def parallel_opted_in(monkeypatch):
    """Overlap is opt-in where preemption is not armed; these cases ask for an overlapped round."""
    monkeypatch.setenv("UNSLOTH_PARALLEL_TOOL_CALLS", "1")


def _call(
    index,
    query,
    name = _EDIT,
):
    return {
        "index": index,
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"query": query})},
    }


class _Document:
    """A remote document edited by read, modify, write: overlapped edits both read the old copy."""

    def __init__(self):
        self.lines: list[str] = []
        self._barrier = threading.Barrier(2, timeout = 1)

    def execute(self, name, arguments, **_kwargs):
        if name != _EDIT:
            return "unrelated"
        snapshot = list(self.lines)
        try:
            # Holds the first edit until a second arrives, as a slow server would.
            self._barrier.wait()
        except threading.BrokenBarrierError:
            pass
        self.lines = snapshot + [(arguments or {}).get("query")]
        return "edited"


def test_a_round_with_an_mcp_call_runs_in_order():
    in_order = loop_mod.round_needs_order
    edit_a, edit_b = _call(0, "a"), _call(1, "b")
    search = _call(2, "q", name = "web_search")
    assert in_order([edit_a, edit_b])
    assert in_order([search, edit_a]), "an MCP call can change what a sibling reads"
    assert not in_order([edit_a]), "a lone call has nothing to order against"
    other_search = _call(3, "r", name = "web_search")
    assert not in_order([search, other_search]), "built-in reads still overlap"


def test_the_provider_loop_keeps_both_mcp_edits(monkeypatch):
    document = _Document()
    monkeypatch.setattr(loop_mod, "execute_tool", document.execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    transport = FakeTransport(
        [
            [
                _sse({"tool_calls": [_call(0, "a"), _call(1, "b")]}),
                _sse(finish = "tool_calls"),
                _DONE,
            ],
            [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
        ]
    )
    _run(transport, tools = [_EDIT_TOOL])
    assert document.lines == [
        "a",
        "b",
    ], f"an MCP edit that reported success was lost: {document.lines}"


def test_the_gguf_loop_keeps_both_mcp_edits(monkeypatch):
    document = _Document()
    backend = _make_backend(
        monkeypatch,
        [
            [_gguf_sse({"tool_calls": [_call(0, "a"), _call(1, "b")]}), _gguf_done()],
            [_gguf_sse({"content": "Done."}), _gguf_done()],
        ],
        [],
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", document.execute)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "append a and b"}],
            tools = [_EDIT_TOOL],
            max_tool_iterations = 2,
        )
    )
    assert document.lines == [
        "a",
        "b",
    ], f"an MCP edit that reported success was lost: {document.lines}"
