# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`search_conversation` under a safetensors model.

The tool is advertised by thread, not by backend: a chat compacted under a GGUF model
keeps its archive when the user switches models, and `_select_request_tools` is shared,
so the safetensors loop offers the tool too. Both guards the GGUF loop applies to it
have to apply here as well.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference.safetensors_agentic import run_safetensors_tool_loop  # noqa: E402
from core.inference.tool_call_parser import RAG_MAX_SEARCHES_PER_TURN  # noqa: E402

MESSAGES = [
    {"role": "user", "content": "what was the code from earlier"},
    {"role": "assistant", "content": "let me look"},
    {"role": "user", "content": "please"},
]


def _searching_model(calls_wanted):
    """A model that issues one distinct conversation search per iteration."""
    state = {"n": 0}

    def single_turn(messages):
        state["n"] += 1
        if state["n"] > calls_wanted:
            yield "done"
            return
        call = (
            '<tool_call>{"name":"search_conversation","arguments":'
            '{"query":"the code %d"}}</tool_call>' % state["n"]
        )
        buffer = ""
        for char in call:
            buffer += char
            yield buffer

    return single_turn


def _run(calls_wanted, execute_tool):
    return list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(calls_wanted),
            messages = list(MESSAGES),
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            execute_tool = execute_tool,
            cancel_event = threading.Event(),
            max_tool_iterations = calls_wanted + 1,
            thread_id = "t-sf",
        )
    )


def test_the_loop_hands_its_search_the_active_branch():
    """Without it the search falls back to the whole stored DAG, siblings included."""
    seen = []

    def execute_tool(name, arguments, **kwargs):
        seen.append(kwargs.get("conversation_branch"))
        return "an earlier turn"

    _run(1, execute_tool)

    assert seen == [MESSAGES]


def test_conversation_searches_share_the_per_turn_cap():
    """Paraphrased re-searches slip past the exact-args duplicate guard.

    Each one appends archived passages into the current tool exchange, which the rolling
    window protects and so cannot evict: uncapped, the turn can only end in a
    context-length error.
    """
    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append(arguments.get("query"))
        return "an earlier turn"

    _run(RAG_MAX_SEARCHES_PER_TURN + 2, execute_tool)

    assert len(executed) == RAG_MAX_SEARCHES_PER_TURN
