# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``on_conversation_grew`` took ``callback(conversation)`` before the tool catalogue was added.
A caller still passing a one-argument hook must keep being called at every recost site."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _sse,
    _structured_tool_call,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _run(monkeypatch, hook, max_tool_iterations):
    first_stream = _structured_tool_call("web_search", {"query": "kernel"}, "call_search")
    second_stream = [_sse({"content": "6.10."}), _done()]
    backend, payloads = _backend_and_payloads(monkeypatch, [first_stream, second_stream])
    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "Linux 6.10."
    )
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Search for the kernel version."}],
            tools = _TOOLS,
            max_tool_iterations = max_tool_iterations,
            permission_mode = "off",
            on_conversation_grew = hook,
        )
    )
    return payloads


def test_a_one_argument_hook_is_called_on_every_tool_round(monkeypatch):
    two: list = []
    _run(monkeypatch, lambda conversation, _tools: two.append(len(conversation)), 4)
    one: list = []
    _run(monkeypatch, lambda conversation: one.append(len(conversation)), 4)
    assert two, "the two-argument hook this test compares against never ran"
    assert one == two, f"a one-argument hook saw {one}, a two-argument one {two}"


def test_a_one_argument_hook_is_called_on_the_synthesized_final_pass(monkeypatch):
    seen: list = []
    payloads = _run(monkeypatch, lambda conversation: seen.append(copy.deepcopy(conversation)), 1)
    assert len(payloads) == 2, "expected one tool round and one synthesized final pass"
    assert seen, "the one-argument hook never ran"
    assert len(seen[-1]) == len(payloads[-1]["messages"])

