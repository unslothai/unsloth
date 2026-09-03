# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`search_conversation` under a safetensors model.

The tool is advertised by thread, not by backend: a GGUF-compacted chat keeps its
archive across a model switch and `_select_request_tools` is shared, so this loop offers
the tool too and needs both guards the GGUF loop applies.
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

    Each appends passages into the protected current tool exchange, so uncapped the turn
    can only end in a context-length error.
    """
    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append(arguments.get("query"))
        return "an earlier turn"

    _run(RAG_MAX_SEARCHES_PER_TURN + 2, execute_tool)

    assert len(executed) == RAG_MAX_SEARCHES_PER_TURN


def test_the_budget_charges_token_dense_text_at_its_real_rate():
    """Four characters per token is about right for English and half the truth for CJK.

    The result side already prices non-ASCII at a token per character; the spend side did
    not, so a CJK chat reported roughly twice the room it had. This path runs no rolling
    fit, so nothing downstream recovers: the tool exchange it sized lands in the next
    prompt and takes it past the window.
    """
    from core.inference.context_window import (
        estimate_messages_tokens,
        estimate_messages_tokens_dense,
        prompt_budget,
    )

    dense_messages = [
        {"role": "user", "content": "設定を確認してください。" * 40},
        {"role": "assistant", "content": "承知しました。" * 40},
        {"role": "user", "content": "先ほどのコードは何でしたか"},
    ]
    seen = {}
    tools = [{"type": "function", "function": {"name": "search_conversation"}}]

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = list(dense_messages),
            tools = tools,
            execute_tool = lambda name, arguments, **kwargs: (
                seen.update(kwargs) or "an earlier turn"
            ),
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
        )
    )

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    # The flat estimate is the one that overstates the room. Charged densely, the budget
    # has to be at most what is left after the real cost of what is already there.
    assert budget <= prompt_budget(4096, 512) - estimate_messages_tokens_dense(dense_messages)
    assert estimate_messages_tokens_dense(dense_messages) > estimate_messages_tokens(dense_messages)


def test_the_dense_estimate_matches_the_flat_one_on_plain_ascii():
    """It corrects a known undercount, it does not make every English chat pessimistic."""
    from core.inference.context_window import (
        estimate_messages_tokens,
        estimate_messages_tokens_dense,
    )
    assert estimate_messages_tokens_dense(MESSAGES) == estimate_messages_tokens(MESSAGES)


def test_the_loop_budgets_its_search_against_this_models_context():
    """Without it the clamp in the tool is skipped and top_k 8 lands unbudgeted.

    Roughly 4K tokens appended into the current tool exchange, which the rolling window
    protects and cannot evict.
    """
    from core.inference.context_window import estimate_messages_tokens, prompt_budget

    seen = {}
    tools = [{"type": "function", "function": {"name": "search_conversation"}}]

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = list(MESSAGES),
            tools = tools,
            execute_tool = execute_tool,
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
        )
    )

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    # Everything already in the prompt is charged: the messages and the catalogue.
    assert budget <= prompt_budget(4096, 512) - estimate_messages_tokens(tools)
    assert budget < prompt_budget(4096, 512) - estimate_messages_tokens(MESSAGES)


def test_the_loop_omits_the_budget_when_the_context_is_unknown():
    """An absent budget must not become a budget of zero, which would refuse every search."""
    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    _run(1, execute_tool)

    assert "conversation_budget_tokens" not in seen
