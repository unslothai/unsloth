# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic backend-wiring test for the safetensors / MLX tool-calling path.

The parser and the cumulative-text state machine are already covered exhaustively by
``test_safetensors_tool_loop.py`` with fake generators. What that suite does not touch is the
*backend's own tool-injection seam*: both ``InferenceBackend`` (transformers) and
``MLXInferenceBackend`` render the prompt through the shared
``apply_chat_template_for_generation(..., tools=...)`` helper and stream cumulative text into the
shared ``run_safetensors_tool_loop`` (see ``core/inference/inference.py`` and
``core/inference/mlx_inference.py`` -- both call the same helper and the same loop, so a single CPU
test of that seam covers the macOS MLX path too).

This test drives that exact seam with deterministic fakes -- a fake tokenizer that records the
``tools`` it is handed, a canned tool-call generation, and a stub executor -- and asserts the full
agentic chain end to end:

    tools injected into the template -> loop parses the call -> tool dispatched once ->
    tool result fed back -> generation re-entered -> final answer streamed.

It is the deterministic, download-free stand-in for the real-model MLX / GGUF browser tool-calling
end-to-end: it imports no torch / unsloth / mlx, so it runs in the portable Backend CI alongside the
tool-call parser tests. Follow-up to the parser test PRs (#5620 / #5704).
"""

from core.inference.chat_template_helpers import apply_chat_template_for_generation
from core.inference.safetensors_agentic import run_safetensors_tool_loop

TOOL_NAME = "get_weather"
TOOL_ARGS = {"city": "Paris"}
FAKE_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
# Full parser matrix lives in test_safetensors_tool_loop.py.
TOOL_CALL_TEXT = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
)
FINAL_ANSWER = "The weather in Paris is sunny and 22C."
TOOL_RESULT = "Paris: sunny, 22C"


class RecordingTokenizer:
    """Fake tokenizer that records the ``tools`` handed to ``apply_chat_template``.

    Modelled on ``TestChatTemplateHelper._Tok`` in ``test_safetensors_tool_loop.py``: it accepts the
    real helper's kwargs and returns a canned prompt, so the test can assert the backend seam actually
    forwarded the tool schema -- a silent drop on a chat-template fallback would leave ``tools_seen``
    holding ``None``.
    """

    def __init__(self):
        self.tools_seen: list = []
        self.call_count = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kwargs,
    ):
        self.call_count += 1
        self.tools_seen.append(kwargs.get("tools"))
        return "PROMPT"


class StubExecutor:
    """Stand-in for ``core.inference.tools.execute_tool``: records calls, returns a fixed result.

    A fake tool name plus this stub means no real python / terminal / web / RAG side effect can run.
    """

    def __init__(self, result: str):
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        return self.result


def _collect(generator, max_events = 200):
    events = []
    for ev in generator:
        events.append(ev)
        if len(events) >= max_events:
            break
    return events


def _tool_names(tools):
    return [(t.get("function") or {}).get("name") for t in (tools or [])]


def test_backend_seam_injects_tools_and_drives_full_tool_loop():
    """The shared backend seam forwards tools into the chat template, and the loop parses the call,
    dispatches it once, feeds the result back, and re-enters generation for the final answer."""
    tok = RecordingTokenizer()
    executor = StubExecutor(TOOL_RESULT)
    turns = iter([TOOL_CALL_TEXT, FINAL_ANSWER])
    active_tools_seen: list = []
    conversations_seen: list = []

    def single_turn(conversation, *, active_tools = None):
        # Mirror the real _single_turn: render via the shared helper, then yield cumulative snapshots.
        active_tools_seen.append(active_tools)
        conversations_seen.append([dict(m) for m in conversation])
        apply_chat_template_for_generation(tok, conversation, tools = active_tools)
        text = next(turns)
        mid = len(text) // 2
        acc = ""
        for chunk in (text[:mid], text[mid:]):
            acc += chunk
            yield acc

    events = _collect(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "What is the weather in Paris?"}],
            tools = [FAKE_TOOL],
            execute_tool = executor,
            max_tool_iterations = 3,
        )
    )

    # 1. Helper forwarded the tool schema to the tokenizer (seam does not drop tools).
    assert tok.tools_seen, "tokenizer.apply_chat_template was never called"
    assert tok.tools_seen[0], "tool schema was dropped before reaching the tokenizer"
    assert TOOL_NAME in _tool_names(tok.tools_seen[0])

    # 2. Loop offered the tool to the first generation turn.
    assert active_tools_seen and active_tools_seen[0] is not None
    assert TOOL_NAME in _tool_names(active_tools_seen[0])

    # 3 / 4 / 5. Exactly one tool_start, one dispatch with parsed args, one tool_end with the result.
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_starts) == 1 and tool_starts[0]["tool_name"] == TOOL_NAME
    assert executor.calls == [(TOOL_NAME, TOOL_ARGS)], executor.calls
    assert len(tool_ends) == 1 and tool_ends[0]["result"] == TOOL_RESULT

    # 6. Final answer streams after the tool result: loop appended it and re-entered generation.
    contents = [e for e in events if e["type"] == "content"]
    assert contents and FINAL_ANSWER in contents[-1]["text"]
    last_tool_end_idx = max(i for i, e in enumerate(events) if e["type"] == "tool_end")
    last_content_idx = max(i for i, e in enumerate(events) if e["type"] == "content")
    assert (
        last_content_idx > last_tool_end_idx
    ), "final answer must stream after the tool result"

    # 6b. Tool result fed back into the conversation before the final turn (6 alone misses this:
    #     the fake generation ignores the conversation).
    assert (
        len(conversations_seen) >= 2
    ), "loop did not re-enter generation after the tool call"
    final_turn_convo = conversations_seen[1]
    assert any(
        TOOL_RESULT in str(m.get("content", "")) for m in final_turn_convo
    ), "tool result was not fed back into the conversation before the final generation turn"

    # 7. Guard: raw tool-call markup never leaked to the client as content.
    for e in contents:
        assert "<tool_call>" not in e["text"]
        assert TOOL_NAME not in e["text"]
