# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parallel round is decided on the calls as the controller will run them.

Two calls that differ only in what auto-healing repairs heal into one call, so overlapped
both would run before either result could mark the other a repeat. And every call of an
overlapped round is handed the branch as it stood before the round, not whatever its
siblings had settled into by the time its worker thread read it.
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

import pytest  # noqa: E402

from core.inference import studio_tool_loop as loop_mod  # noqa: E402
from core.inference.tool_loop_controller import ToolLoopController  # noqa: E402
from test_studio_tool_loop import WEB, FakeTransport, _DONE, _run, _sse  # noqa: E402
from test_llama_cpp_tool_loop import _done as _gguf_done  # noqa: E402
from test_llama_cpp_tool_loop import _make_backend  # noqa: E402
from test_llama_cpp_tool_loop import _sse as _gguf_sse  # noqa: E402

_BARE = {"name": "web_search", "arguments": "alpha"}
_OBJECT = {"name": "web_search", "arguments": json.dumps({"query": "alpha"})}


def _call(index, call_id, function):
    return {"index": index, "id": call_id, "type": "function", "function": dict(function)}


class TestTheRoundKeyIsTheHealedOne:
    def test_a_bare_and_an_object_call_share_a_key(self):
        controller = ToolLoopController(tools = [WEB])
        keys = loop_mod.round_call_keys(
            controller, [{"function": _BARE}, {"function": _OBJECT}]
        )
        assert keys[0] == keys[1]
        # The raw key the gate used to read tells them apart, which is the gap.
        assert loop_mod.round_call_key("web_search", "alpha") != loop_mod.round_call_key(
            "web_search", _OBJECT["arguments"]
        )

    def test_a_controller_without_a_key_falls_back_to_the_raw_one(self):
        keys = loop_mod.round_call_keys(object(), [{"function": _OBJECT}])
        assert keys == [loop_mod.round_call_key("web_search", _OBJECT["arguments"])]


class TestTheProviderLoopRunsAHealedPairOnce:
    def test_the_side_effect_happens_once(self, monkeypatch):
        ran: list = []

        def _execute(name, arguments, **_kwargs):
            ran.append((arguments or {}).get("query"))
            time.sleep(0.05)
            return "search-result"

        monkeypatch.setattr(loop_mod, "execute_tool", _execute)
        monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
        transport = FakeTransport(
            [
                [
                    _sse({"tool_calls": [_call(0, "call_a", _BARE), _call(1, "call_b", _OBJECT)]}),
                    _sse(finish = "tool_calls"),
                    _DONE,
                ],
                [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
            ]
        )
        _run(transport)
        assert ran == ["alpha"], f"a call that heals into its sibling ran again: {ran}"


def _gguf_events(monkeypatch, tool_calls, execute):
    backend = _make_backend(
        monkeypatch,
        [
            [_gguf_sse({"tool_calls": tool_calls}), _gguf_done()],
            [_gguf_sse({"content": "Final answer."}), _gguf_done()],
        ],
        [],
    )
    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "go"}],
            tools = [WEB],
            max_tool_iterations = 2,
        )
    )


class TestTheLocalGgufLoopRunsAHealedPairOnce:
    def test_the_side_effect_happens_once(self, monkeypatch):
        ran: list = []

        def _execute(name, arguments, **_kwargs):
            ran.append((arguments or {}).get("query"))
            time.sleep(0.05)
            return "search-result"

        _gguf_events(
            monkeypatch, [_call(0, "call_a", _BARE), _call(1, "call_b", _OBJECT)], _execute
        )
        assert ran == ["alpha"], f"a call that heals into its sibling ran again: {ran}"


class TestEveryCallOfARoundSeesTheSameBranch:
    def test_each_call_holds_its_own_pre_round_branch(self, monkeypatch):
        handed: dict = {}

        def _execute(name, arguments, *, conversation_branch = None, **_kwargs):
            query = (arguments or {}).get("query")
            handed[query] = (conversation_branch, len(conversation_branch or []))
            return f"RESULT<{query}>"

        _gguf_events(
            monkeypatch,
            [
                _call(0, "call_a", {"name": "web_search", "arguments": json.dumps({"query": "alpha"})}),
                _call(1, "call_b", {"name": "web_search", "arguments": json.dumps({"query": "beta"})}),
            ],
            _execute,
        )
        assert set(handed) == {"alpha", "beta"}
        (first, first_len), (second, second_len) = handed["alpha"], handed["beta"]
        assert first is not second, "both workers were handed the one shared live list"
        # The live branch keeps growing as the round settles; a call's view must not.
        assert len(first) == first_len and len(second) == second_len, (
            "the branch a call was handed changed under it after the round settled"
        )
        assert [id(m) for m in first] == [id(m) for m in second]
