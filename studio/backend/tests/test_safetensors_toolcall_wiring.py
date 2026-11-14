# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drives the tool-injection seam with fakes, so the whole agentic chain runs without torch or mlx."""

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
TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
FINAL_ANSWER = "The weather in Paris is sunny and 22C."
TOOL_RESULT = "Paris: sunny, 22C"


class RecordingTokenizer:
    """Records the tools handed to apply_chat_template; a silent drop would leave tools_seen as None."""

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
    """Stub executor records calls and returns a fixed result, so no real tool side effect can run."""

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

    assert tok.tools_seen, "tokenizer.apply_chat_template was never called"
    assert tok.tools_seen[0], "tool schema was dropped before reaching the tokenizer"
    assert TOOL_NAME in _tool_names(tok.tools_seen[0])

    assert active_tools_seen and active_tools_seen[0] is not None
    assert TOOL_NAME in _tool_names(active_tools_seen[0])

    tool_starts = [e for e in events if e["type"] == "tool_start"]
    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_starts) == 1 and tool_starts[0]["tool_name"] == TOOL_NAME
    assert executor.calls == [(TOOL_NAME, TOOL_ARGS)], executor.calls
    assert len(tool_ends) == 1 and tool_ends[0]["result"] == TOOL_RESULT

    contents = [e for e in events if e["type"] == "content"]
    assert contents and FINAL_ANSWER in contents[-1]["text"]
    last_tool_end_idx = max(i for i, e in enumerate(events) if e["type"] == "tool_end")
    last_content_idx = max(i for i, e in enumerate(events) if e["type"] == "content")
    assert last_content_idx > last_tool_end_idx, "final answer must stream after the tool result"

    assert len(conversations_seen) >= 2, "loop did not re-enter generation after the tool call"
    final_turn_convo = conversations_seen[1]
    assert any(
        TOOL_RESULT in str(m.get("content", "")) for m in final_turn_convo
    ), "tool result was not fed back into the conversation before the final generation turn"

    for e in contents:
        assert "<tool_call>" not in e["text"]
        assert TOOL_NAME not in e["text"]
