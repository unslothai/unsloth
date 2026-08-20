# SPDX-License-Identifier: AGPL-3.0-only

from types import SimpleNamespace


def test_mlx_token_count_uses_the_generation_prompt_and_special_token_rule():
    from core.inference.mlx_inference import MLXInferenceBackend

    class Tokenizer:
        bos_token = "<s>"

        def __init__(self):
            self.calls = []

        def encode(self, prompt, *, add_special_tokens):
            self.calls.append((prompt, add_special_tokens))
            return [1, 2, 3, 4]

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = Tokenizer()
    backend._is_vlm = False
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {}
    rendered = []

    def render(messages, **kwargs):
        rendered.append((messages, kwargs))
        return SimpleNamespace(prompt = "<s>exact rendered prompt")

    backend._render_text_chat_prompt = render
    count = backend.count_chat_tokens(
        [{"role": "user", "content": "hello"}],
        "system",
        [{"type": "function", "function": {"name": "search"}}],
        enable_thinking = True,
    )

    assert count == 4
    assert rendered[0][0][0] == {"role": "system", "content": "system"}
    assert rendered[0][1]["enable_thinking"] is True
    assert backend._tokenizer.calls == [("<s>exact rendered prompt", False)]


def test_worker_returns_request_scoped_mlx_token_count():
    from core.inference.worker import _handle_count_chat_tokens

    class Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    class Backend:
        def count_chat_tokens(self, messages, system_prompt, tools, **kwargs):
            assert messages[-1]["content"] == "hello"
            assert system_prompt == "system"
            assert tools[0]["function"]["name"] == "search"
            assert kwargs["reasoning_effort"] == "high"
            return 17

    queue = Queue()
    _handle_count_chat_tokens(
        Backend(),
        {
            "request_id": "count-1",
            "messages": [{"role": "user", "content": "hello"}],
            "system_prompt": "system",
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "reasoning_effort": "high",
        },
        queue,
    )

    assert queue.items[0]["type"] == "token_count"
    assert queue.items[0]["request_id"] == "count-1"
    assert queue.items[0]["count"] == 17


def test_orchestrator_requests_the_count_from_the_mlx_worker(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    backend = InferenceOrchestrator()
    backend.active_model_name = "mlx-test"
    backend.models["mlx-test"] = {"is_mlx": True}
    commands = []
    released = []

    monkeypatch.setattr(backend, "_ensure_subprocess_alive", lambda: True)
    monkeypatch.setattr(backend, "_wait_dispatcher_idle", lambda: True)
    monkeypatch.setattr(backend, "_send_cmd", commands.append)
    monkeypatch.setattr(
        backend,
        "_direct_reader",
        lambda _request_id: (
            lambda timeout = 1.0: {"type": "token_count", "count": 23},
            lambda timeout = 5.0: True,
            lambda: released.append(True),
        ),
    )

    count = backend.count_chat_tokens(
        [{"role": "user", "content": "hello"}],
        tools = [{"type": "function", "function": {"name": "search"}}],
        preserve_thinking = True,
    )

    assert count == 23
    assert commands[0]["type"] == "count_chat_tokens"
    assert commands[0]["preserve_thinking"] is True
    assert released == [True]


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
    backend.count_chat_tokens = _word_count
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
