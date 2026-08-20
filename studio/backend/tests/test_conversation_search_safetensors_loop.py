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


def _model_that_thinks_out_loud(preamble, stats_holder, reported):
    """A model that writes prose before its one search, reporting usage as a turn does."""
    state = {"n": 0}

    def single_turn(messages):
        stats_holder["stats"] = None
        state["n"] += 1
        if state["n"] > 1:
            yield "done"
            return
        call = (
            preamble
            + '<tool_call>{"name":"search_conversation","arguments":{"query":"the code"}}'
            + "</tool_call>"
        )
        buffer = ""
        for char in call:
            buffer += char
            yield buffer
        stats_holder["stats"] = {"usage": {"prompt_tokens": reported}}

    return single_turn


def _budget_after_a_preamble(preamble, reported, stats_holder):
    seen = {}
    list(
        run_safetensors_tool_loop(
            single_turn = _model_that_thinks_out_loud(preamble, stats_holder, reported),
            messages = [
                {"role": "user", "content": "recall the deployment steps. " * 420},
                {"role": "user", "content": "what was the code from earlier"},
            ],
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            execute_tool = lambda name, arguments, **kwargs: (
                seen.update(kwargs) or "an earlier turn"
            ),
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 8192,
            max_tokens = 512,
            generation_stats_holder = stats_holder,
        )
    )
    return seen["conversation_budget_tokens"]


def test_the_budget_charges_this_turns_own_output_in_tokens_not_bytes():
    """A 4600-character English preamble costs about 1150 tokens, not 4600.

    Charged by its byte length it takes the whole prompt budget of an 8K window, so the
    search is refused with roughly 3.4K tokens still free, which is worse than the
    estimate it replaces. The fixture is deliberately large: on a three-line chat the
    two numbers differ by too little to fail an assertion.
    """
    from core.inference.context_window import retrieval_budget

    preamble = "the answer is somewhere in the earlier turns. " * 100
    budget = _budget_after_a_preamble(preamble, 3_067, {})

    # Charged as its 4600 bytes this is nothing. Charged as tokens it is a real search.
    assert budget > 3_000
    # And it IS charged: the room is smaller than it was before the turn wrote anything.
    assert budget < retrieval_budget(8192, 512, 3_067, reply_returns = True)


def test_the_budget_spends_the_count_the_turn_reported():
    """The reported prompt count is a tokenizer's, so it outranks the character estimate.

    Six thousand tokens of a window this size leaves room for a small retrieval and no
    more; the estimate alone would price the same thread at a third of that and hand out
    room the next prompt does not have.
    """
    budget = _budget_after_a_preamble("thinking. ", 6_000, {})

    assert 0 < budget < 1_800


def test_the_budget_estimates_when_no_usage_was_reported():
    """A cancelled or unreported turn falls back to the estimate, never to zero.

    Zero reaches the tool as "there is no room left in this context to search earlier
    conversation", the same failure `_recall_top_k` carries a note about: it switches
    recall off on exactly the tight windows that need it.
    """
    from core.inference.context_window import (
        estimate_messages_tokens_dense,
        retrieval_budget,
    )

    seen = {}
    tools = [{"type": "function", "function": {"name": "search_conversation"}}]

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = list(MESSAGES),
            tools = tools,
            execute_tool = lambda name, arguments, **kwargs: (
                seen.update(kwargs) or "an earlier turn"
            ),
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
            generation_stats_holder = {"stats": None},
        )
    )

    budget = seen["conversation_budget_tokens"]
    assert budget > 0
    assert budget <= retrieval_budget(
        4096, 512, estimate_messages_tokens_dense(tools), reply_returns = True
    )


def test_the_orchestrator_gives_the_loop_a_holder_of_its_own(monkeypatch):
    """One holder, one contract: the request's summed stats are not the loop's input.

    `stats_holder` accumulates every turn for the reply's usage report, so reading it
    here would mix turns. The loop gets the per-turn holder instead.
    """
    import core.inference.safetensors_agentic as agentic
    from core.inference.orchestrator import InferenceOrchestrator

    seen = {}

    def capture_loop(**kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(agentic, "run_safetensors_tool_loop", capture_loop)
    backend = InferenceOrchestrator.__new__(InferenceOrchestrator)
    backend.active_model_name = "sf-model"
    backend.models = {"sf-model": {"context_length": 4096}}
    stats_holder = {"stats": {"usage": {"prompt_tokens": 999}}}

    list(
        backend.generate_chat_completion_with_tools(
            messages = list(MESSAGES),
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            stats_holder = stats_holder,
        )
    )

    assert isinstance(seen["generation_stats_holder"], dict)
    assert seen["generation_stats_holder"] is not stats_holder
    assert seen["generation_stats_holder"] == {}


def test_the_loop_omits_the_budget_when_the_context_is_unknown():
    """An absent budget must not become a budget of zero, which would refuse every search."""
    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    _run(1, execute_tool)

    assert "conversation_budget_tokens" not in seen
