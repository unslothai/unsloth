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


def test_the_budget_uses_the_generated_prompts_exact_token_count():
    from core.inference.context_window import prompt_budget

    dense_messages = [
        {"role": "user", "content": "設定を確認してください。" * 40},
        {"role": "assistant", "content": "承知しました。" * 40},
        {"role": "user", "content": "先ほどのコードは何でしたか"},
    ]
    seen = {}
    tools = [{"type": "function", "function": {"name": "search_conversation"}}]
    generation_stats_holder = {
        "stats": {"usage": {"prompt_tokens": 3_000, "completion_tokens": 20}}
    }

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = list(dense_messages),
            tools = tools,
            execute_tool = lambda name, arguments, **kwargs: seen.update(kwargs) or "an earlier turn",
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
            generation_stats_holder = generation_stats_holder,
        )
    )

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    assert budget < prompt_budget(4096, 512) - 3_000


def test_the_budget_fails_closed_without_generation_usage():
    seen = {}

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = list(MESSAGES),
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            execute_tool = lambda name, arguments, **kwargs: seen.update(kwargs) or "an earlier turn",
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
        )
    )

    assert seen["conversation_budget_tokens"] == 0


def test_the_orchestrator_passes_generation_usage_to_the_tool_loop(monkeypatch):
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
    stats_holder = {}

    list(
        backend.generate_chat_completion_with_tools(
            messages = list(MESSAGES),
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            stats_holder = stats_holder,
        )
    )

    assert seen["generation_stats_holder"] is stats_holder


def test_the_budget_charges_an_in_place_continuation():
    from core.inference.context_window import prompt_budget

    seen = {}
    generation_stats_holder = {
        "stats": {"usage": {"prompt_tokens": 100, "completion_tokens": 3_000}}
    }

    list(
        run_safetensors_tool_loop(
            single_turn = _searching_model(1),
            messages = [
                {"role": "user", "content": "Find the earlier turn"},
                {"role": "assistant", "content": "I will"},
            ],
            tools = [{"type": "function", "function": {"name": "search_conversation"}}],
            execute_tool = lambda name, arguments, **kwargs: seen.update(kwargs) or "found",
            cancel_event = threading.Event(),
            max_tool_iterations = 2,
            thread_id = "t-sf",
            context_length = 4096,
            max_tokens = 512,
            continue_final_message = True,
            generation_stats_holder = generation_stats_holder,
        )
    )

    assert seen["conversation_budget_tokens"] <= prompt_budget(4096, 512) - 3_164


def test_the_loop_budgets_its_search_against_this_models_context():
    """Without it the clamp in the tool is skipped and top_k 8 lands unbudgeted.

    Roughly 4K tokens appended into the current tool exchange, which the rolling window
    protects and cannot evict.
    """
    from core.inference.context_window import prompt_budget

    seen = {}
    tools = [{"type": "function", "function": {"name": "search_conversation"}}]
    generation_stats_holder = {
        "stats": {"usage": {"prompt_tokens": 1_000, "completion_tokens": 20}}
    }

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
            generation_stats_holder = generation_stats_holder,
        )
    )

    budget = seen.get("conversation_budget_tokens")
    assert budget is not None
    assert budget < prompt_budget(4096, 512) - 1_000


def test_the_loop_omits_the_budget_when_the_context_is_unknown():
    """An absent budget must not become a budget of zero, which would refuse every search."""
    seen = {}

    def execute_tool(name, arguments, **kwargs):
        seen.update(kwargs)
        return "an earlier turn"

    _run(1, execute_tool)

    assert "conversation_budget_tokens" not in seen
