# SPDX-License-Identifier: AGPL-3.0-only


def _word_count(messages, *_args, **_kwargs):
    def text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
        return ""

    return 2 * len(messages) + sum(
        len(text(message.get("content")).split()) for message in messages
    )


def test_orchestrator_compacts_an_mlx_prompt_before_generation():
    from core.inference.orchestrator import InferenceOrchestrator

    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {
        "is_mlx": True,
        "context_length": 34,
    }
    backend.count_chat_tokens = lambda *a, **kw: (_word_count(*a, **kw), "mlx-test")
    messages = [
        {"role": "user", "content": "What happened in the older part of this conversation?"},
        {
            "role": "assistant",
            "content": "This older answer contains enough words to make the complete prompt exceed its budget.",
        },
        {"role": "user", "content": "Answer the newest question now."},
    ]

    result = backend.compact_chat_context(
        messages,
        system_prompt = "You are concise.",
        context_overflow = "truncate_oldest",
        max_tokens = 8,
    )

    assert result["system_prompt"] == ""
    assert result["truncation"]["fits"] is True
    assert result["truncation"]["dropped_messages"] >= 1
    assert (
        result["truncation"]["prompt_tokens_before"] > result["truncation"]["prompt_tokens_after"]
    )
    assert result["messages"][-1]["content"] == "Answer the newest question now."
    assert result["events"][-1]["type"] == "context_truncated"


def test_mlx_media_prompt_stays_untouched():
    from core.inference.orchestrator import InferenceOrchestrator

    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-vlm-test"
    backend.models["mlx-vlm-test"] = {"is_mlx": True, "context_length": 64}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                {"type": "text", "text": "describe"},
            ],
        }
    ]

    result = backend.compact_chat_context(
        messages,
        system_prompt = "system",
        context_overflow = "truncate_oldest",
        max_tokens = 8,
    )

    assert result["messages"] == messages
    assert result["system_prompt"] == "system"
    assert result["truncation"] is None


def test_safetensors_loop_applies_the_fitter_before_generation():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    seen = []

    def fitter(conversation, active_tools):
        assert active_tools[0]["function"]["name"] == "search"
        return {
            "messages": [
                {"role": "system", "content": "compacted"},
                conversation[-1],
            ],
            "events": [
                {
                    "type": "context_truncated",
                    "fits": True,
                    "dropped_messages": 2,
                }
            ],
        }

    def single_turn(conversation, *, active_tools = None):
        seen.append((conversation, active_tools))
        yield "final answer"

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "hello"}],
            tools = [{"type": "function", "function": {"name": "search"}}],
            execute_tool = lambda *_args, **_kwargs: "unused",
            max_tool_iterations = 1,
            nudge_tool_calls = False,
            context_fitter = fitter,
        )
    )

    assert events[0]["type"] == "context_truncated"
    assert seen[0][0][0] == {"role": "system", "content": "compacted"}
    assert any(event.get("type") == "content" for event in events)


def test_safetensors_loop_refits_after_each_tool_result_in_the_same_response():
    """A long-running agent must not treat compaction as a one-shot turn event."""
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    fits = []
    turns = iter(
        [
            '<tool_call>{"name":"search","arguments":{"query":"next"}}</tool_call>',
            "finished after the tool",
        ]
    )

    def fitter(conversation, active_tools):
        fits.append([dict(message) for message in conversation])
        return {"messages": conversation, "events": []}

    def single_turn(_conversation, *, active_tools = None):
        yield next(turns)

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "keep working"}],
            tools = [{"type": "function", "function": {"name": "search"}}],
            execute_tool = lambda *_args, **_kwargs: "a large tool result",
            max_tool_iterations = 2,
            nudge_tool_calls = False,
            context_fitter = fitter,
        )
    )

    assert len(fits) == 2, "the fitter must run again within the same response"
    assert fits[0][-1] == {"role": "user", "content": "keep working"}
    assert fits[1][-1]["role"] == "tool"
    assert fits[1][-1]["content"] == "a large tool result"
    assert any(event.get("type") == "content" for event in events)


def test_preflight_uses_upstream_count_contract_and_request_policy(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator
    from core.inference import llama_cpp

    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {"is_mlx": True, "context_length": 128}
    seen = {}
    tools = [{"type": "function", "function": {"name": "search"}}]

    def count(messages, system_prompt, **kwargs):
        seen.update(kwargs)
        assert system_prompt == ""
        return 12, "mlx-test"

    def fit(messages, **kwargs):
        assert kwargs["count_tokens"](messages) == 12
        assert kwargs["context_policy"] == "rolling"
        assert kwargs["headroom_ratio"] == 0.1
        return messages, None

    backend.count_chat_tokens = count
    monkeypatch.setattr(llama_cpp, "_fit_with_instruction_pins", fit)
    result = backend.compact_chat_context(
        [{"role": "user", "content": "hello"}],
        tools = tools,
        context_overflow = "truncate_oldest",
        context_policy = "rolling",
        compaction_headroom_ratio = 0.1,
        enable_thinking = True,
        reasoning_effort = "high",
        preserve_thinking = True,
    )
    assert result["system_prompt"] == ""
    assert seen == dict(
        tools = tools, enable_thinking = True, reasoning_effort = "high", preserve_thinking = True
    )


def test_preflight_failure_and_continuation_preserve_original_request():
    from core.inference.orchestrator import InferenceOrchestrator

    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {"is_mlx": True, "context_length": 64}
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("worker busy")

    backend.count_chat_tokens = unavailable
    messages = [{"role": "user", "content": "hello"}]
    result = backend.compact_chat_context(
        messages, system_prompt = "system", context_overflow = "truncate_oldest"
    )
    assert calls
    assert result["messages"] == messages
    assert result["system_prompt"] == "system"
    assert result["events"] == []
    calls.clear()
    messages.append({"role": "assistant", "content": "unfinished"})
    result = backend.compact_chat_context(
        messages, context_overflow = "truncate_oldest", continue_final_message = True
    )
    assert not calls
    assert result["messages"] == messages
