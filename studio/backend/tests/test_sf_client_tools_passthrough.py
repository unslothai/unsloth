# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Client-tools passthrough healing for the safetensors/MLX backend.

Parity for #6801: when a NON-GGUF model is loaded and the request declares its
own ``tools`` with server-side tools OFF, text-form tool calls are promoted back
into structured ``tool_calls`` (declared tools only) via the shared healer. MLX
rides the same orchestrator path, so a single scripted backend covers both.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import openai_chat_completions
from core.inference.api_monitor import ApiMonitor


# #12382: with the date setting on and no system prompt, the chat's first message opens with
# this note. It is the only rewrite of the user's text these assertions allow.
_DATE_NOTE = re.compile(r"\A\[Current date: \d{4}-\d{2}-\d{2}\]\n\n")


def _without_date_note(text):
    return _DATE_NOTE.sub("", text, count = 1) if isinstance(text, str) else text


LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_CALL_XML = '<tool_call>{"name": "lookup", "arguments": {"q": "cats"}}</tool_call>'
_SEARCH_XML = '<tool_call>{"name": "search", "arguments": {"query": "dogs"}}</tool_call>'


class _Request:
    state = SimpleNamespace()
    url = SimpleNamespace(path = "/v1/chat/completions")
    method = "POST"
    scope: dict = {}
    headers = {"X-Unsloth-Events": "1"}

    async def is_disconnected(self):
        return False


class _ScriptedBackend:
    """Non-GGUF backend: ``generate_chat_response`` replays scripted
    CUMULATIVE snapshots. ``responder(messages, tools)`` returns the snapshot
    list for one generation, so nudge tests can vary output across turns."""

    active_model_name = "sf-model"

    def __init__(
        self,
        responder,
        *,
        stats = None,
    ):
        self.models = {
            "sf-model": {
                "chat_template_info": {"template": "<tool_call> chatml"},
                "context_length": 2048,
            }
        }
        self._responder = responder
        self._stats = stats
        self.calls: list = []
        self.batch_calls: list = []
        self.reset_count = 0

    def generate_chat_response(
        self,
        *,
        messages,
        tools = None,
        stats_holder = None,
        **kwargs,
    ):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        snapshots = self._responder(messages, tools)
        _stats = self._stats
        if isinstance(_stats, list):
            _stats = _stats[min(len(self.calls), len(_stats)) - 1]
        if stats_holder is not None and _stats is not None:
            stats_holder["stats"] = _stats
        for snap in snapshots:
            yield snap

    def generate_chat_batch(
        self,
        rows,
        *,
        stats_holder = None,
        **kwargs,
    ):
        """Every choice's first reply in one command, as the real bridge does."""
        self.batch_calls.append({"rows": rows, "shared": kwargs})
        reported: list = []
        for index, row in enumerate(rows):
            holder: dict = {}
            for snapshot in self.generate_chat_response(stats_holder = holder, **{**kwargs, **row}):
                yield index, snapshot
            reported.append(holder.get("stats"))
            yield index, None
        if stats_holder is not None:
            stats_holder["stats"] = reported

    def reset_generation_state(self, caller_cancel_event = None):
        self.reset_count += 1

    def resize_image(self, image):
        return image


def _fixed(*snapshots):
    """Responder that always replays the given cumulative snapshots."""
    return lambda messages, tools: list(snapshots)


def _llama_stub():
    return SimpleNamespace(
        is_loaded = False,
        supports_tools = False,
        is_vision = False,
        context_length = None,
    )


def _install(
    monkeypatch,
    backend,
    *,
    supports_tools = True,
    supports_reasoning = False,
):
    import routes.inference as inf
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    monitor = ApiMonitor(max_entries = 8)
    monkeypatch.setattr(inf, "api_monitor", monitor)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _llama_stub())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inf,
        "_detect_safetensors_features",
        lambda *a, **k: {
            "supports_tools": supports_tools,
            "supports_reasoning": supports_reasoning,
        },
    )
    return monitor


def _request(**kwargs):
    base = dict(model = "default", messages = [ChatMessage(role = "user", content = "hi")])
    base.update(kwargs)
    return ChatCompletionRequest(**base)


def _call(payload, monkeypatch, backend, **install_kwargs):
    _install(monkeypatch, backend, **install_kwargs)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    return asyncio.run(_run())


def _json_body(response):
    return json.loads(response.body if hasattr(response, "body") else response.content)


@pytest.mark.parametrize("supports_tools", [False, True])
def test_mcp_replay_preserves_multiple_caller_attachments(monkeypatch, supports_tools):
    import base64
    import io

    from PIL import Image

    def encoded(color):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, format = "PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def user(color, text):
        return ChatMessage(
            role = "user",
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + encoded(color)},
                },
                {"type": "text", "text": text},
            ],
        )

    backend = _ScriptedBackend(_fixed("done"))
    backend.models["sf-model"].update(
        is_vision = True,
        chat_template_info = {
            "template": _TOKENIZER_TEMPLATE_WITH_TOOLS,
            "processor_template": _TOKENIZER_TEMPLATE_WITH_TOOLS,
            "renders_image": True,
            "accepts_multiple_images": True,
        },
    )
    payload = _request(
        messages = [
            user("red", "Remember this image."),
            ChatMessage(
                role = "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "mcp__test__image", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role = "tool",
                tool_call_id = "call_1",
                name = "mcp__test__image",
                content = "[1 image returned]\n__MCP_IMAGES__:"
                + json.dumps([{"mimeType": "image/png", "data": encoded("green")}]),
            ),
            ChatMessage(role = "assistant", content = "I have the tool image."),
            user("blue", "Compare all three images."),
        ],
        enable_tools = False,
        stream = False,
    )

    _call(payload, monkeypatch, backend, supports_tools = supports_tools)

    [call] = backend.calls
    colors = []
    for image in call["images"]:
        if isinstance(image, str):
            image = Image.open(io.BytesIO(base64.b64decode(image)))
        colors.append(image.getpixel((0, 0)))
    assert colors == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
    image_turns = [
        message
        for message in call["messages"]
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image" for part in message["content"])
    ]
    assert len(image_turns) == 3


def _collect_sse(response):
    async def _run():
        return [c async for c in response.body_iterator]

    return asyncio.run(_run())


def _sse_objects(chunks):
    out = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        for line in str(chunk).splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                if data != "[DONE]":
                    out.append(json.loads(data))
    return out


def test_non_reasoning_backend_keeps_literal_think_tags(monkeypatch):
    backend = _ScriptedBackend(_fixed("show <think>example</think> tags"))
    response = _call(_request(stream = False), monkeypatch, backend, supports_tools = False)

    message = _json_body(response)["choices"][0]["message"]
    assert message["content"] == "show <think>example</think> tags"
    assert message["reasoning_content"] is None


def test_xml_healed_to_tool_calls_non_streaming(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "cats"}
    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]


def test_undeclared_call_stays_text(monkeypatch):
    xml = '<tool_call>{"name": "other", "arguments": {}}</tool_call>'
    backend = _ScriptedBackend(_fixed(xml))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") is None
    assert choice["message"]["content"] == xml


def test_opt_out_relays_verbatim(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False, auto_heal_tool_calls = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") is None
    assert choice["message"]["content"] == _CALL_XML


def test_env_kill_switch_relays_verbatim(monkeypatch):
    import core.inference.passthrough_healing as ph

    monkeypatch.setattr(ph, "_HEALING_DISABLED", True)
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") is None
    assert choice["message"]["content"] == _CALL_XML


def test_no_tools_request_untouched(monkeypatch):
    backend = _ScriptedBackend(_fixed("just a plain answer"))
    payload = _request(stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "just a plain answer"
    assert choice["message"].get("tool_calls") is None


def test_participant_names_reach_the_local_backend(monkeypatch):
    backend = _ScriptedBackend(_fixed("ok"))
    payload = _request(
        stream = False,
        messages = [
            ChatMessage(role = "user", name = "alice", content = "hi"),
            ChatMessage(role = "assistant", name = "researcher", content = "hello"),
            ChatMessage(role = "user", name = "bob", content = "again"),
        ],
    )
    _call(payload, monkeypatch, backend)
    assert [(m["role"], m.get("name")) for m in backend.calls[0]["messages"]] == [
        ("user", "alice"),
        ("assistant", "researcher"),
        ("user", "bob"),
    ]


def test_a_named_system_message_does_not_restructure_the_request(monkeypatch):
    """Moving it into the history is what changes how much of a thread the vision path renders."""
    sent = []
    for name in (None, "supervisor"):
        backend = _ScriptedBackend(_fixed("ok"))
        _call(
            _request(
                stream = False,
                messages = [
                    ChatMessage(role = "system", name = name, content = "be brief"),
                    ChatMessage(role = "user", content = "hi"),
                ],
            ),
            monkeypatch,
            backend,
        )
        sent.append(backend.calls[0])
    assert sent[0]["messages"] == sent[1]["messages"] == [{"role": "user", "content": "hi"}]
    assert sent[0]["system_prompt"] == sent[1]["system_prompt"]
    assert sent[1]["system_prompt"].endswith("be brief")


def test_prose_around_call_retained(monkeypatch):
    text = "Let me look:\n" + _CALL_XML + "\ndone"
    backend = _ScriptedBackend(_fixed(text))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Let me look:\n\ndone"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_empty_output_is_valid_stop(monkeypatch):
    backend = _ScriptedBackend(_fixed(""))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] in ("", None)
    assert choice["message"].get("tool_calls") is None


def test_tool_role_follow_up_turn_preserves_history(monkeypatch):
    backend = _ScriptedBackend(_fixed("The weather is sunny."))
    payload = _request(
        tools = [LOOKUP_TOOL],
        stream = False,
        messages = [
            ChatMessage(role = "user", content = "weather?"),
            ChatMessage(
                role = "assistant",
                content = None,
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "weather"}'},
                    }
                ],
            ),
            ChatMessage(role = "tool", tool_call_id = "call_0", content = "sunny"),
        ],
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["message"]["content"] == "The weather is sunny."
    sent = backend.calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert "tool" in roles
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant.get("tool_calls")


def test_dict_arguments_history_does_not_crash(monkeypatch):
    backend = _ScriptedBackend(_fixed("ok"))
    payload = _request(
        tools = [LOOKUP_TOOL],
        stream = False,
        messages = [
            ChatMessage(role = "user", content = "hi"),
            ChatMessage(
                role = "assistant",
                content = None,
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "x"}},
                    }
                ],
            ),
            ChatMessage(role = "tool", tool_call_id = "call_0", content = "y"),
        ],
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["message"]["content"] == "ok"


def test_forced_tool_choice_narrows_promotion(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(
        tools = [LOOKUP_TOOL, SEARCH_TOOL],
        stream = False,
        tool_choice = {"type": "function", "function": {"name": "search"}},
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("tool_calls") is None


def test_parallel_cap_non_streaming(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML + _SEARCH_XML))
    payload = _request(tools = [LOOKUP_TOOL, SEARCH_TOOL], stream = False, parallel_tool_calls = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    calls = body["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup"


def test_usage_recorded_when_stats_present(monkeypatch):
    stats = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    backend = _ScriptedBackend(_fixed(_CALL_XML), stats = stats)
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    asyncio.run(_run())
    [entry] = monitor.snapshot()
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 3


class _ToolLoopBackend(_ScriptedBackend):
    """Server-side tool loop: the event stream the GGUF loop also emits."""

    def generate_chat_completion_with_tools(
        self,
        *,
        messages,
        tools = None,
        stats_holder = None,
        **kwargs,
    ):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if stats_holder is not None and self._stats is not None:
            stats_holder["stats"] = self._stats
        yield {"type": "content", "text": "done"}


@pytest.mark.parametrize(
    "kind, stream, expected",
    [
        ("plain", True, "stop"),
        ("plain", False, "stop"),
        ("healed", True, "tool_calls"),
        ("healed", False, "tool_calls"),
        ("tool_loop", True, "stop"),
        ("tool_loop", False, "stop"),
    ],
)
def test_stop_reason_recorded_without_backend_stats(monkeypatch, kind, stream, expected):
    # last_generation_stats is MLX-only, so a transformers generation leaves
    # stats_holder empty; the stop reason must not ride along with it.
    if kind == "tool_loop":
        backend = _ToolLoopBackend(_fixed("done"))
        payload = _request(stream = stream, enable_tools = True)
    elif kind == "healed":
        backend = _ScriptedBackend(_fixed(_CALL_XML))
        payload = _request(stream = stream, tools = [LOOKUP_TOOL])
    else:
        backend = _ScriptedBackend(_fixed("plain answer"))
        payload = _request(stream = stream)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    response = asyncio.run(_run())
    if stream:
        _collect_sse(response)
    [entry] = monitor.snapshot()
    assert entry["stop_reason"] == expected


def test_nudge_default_off_single_generation(monkeypatch):
    truncated = '<tool_call>{"name": "lookup"'
    backend = _ScriptedBackend(_fixed(truncated))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    _call(payload, monkeypatch, backend)
    assert len(backend.calls) == 1


def test_nudge_opt_in_retry_recovers(monkeypatch):
    truncated = '<tool_call>{"name": "lookup"'

    def responder(messages, tools):
        nudged = any(
            "native tool-call format" in (m.get("content") or "")
            for m in messages
            if m.get("role") == "user"
        )
        return [_CALL_XML] if nudged else [truncated]

    backend = _ScriptedBackend(responder)
    payload = _request(tools = [LOOKUP_TOOL], stream = False, nudge_tool_calls = True)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert len(backend.calls) == 2
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_nudge_double_failure_relays_original(monkeypatch):
    truncated = '<tool_call>{"name": "lookup"'
    backend = _ScriptedBackend(_fixed(truncated))
    payload = _request(tools = [LOOKUP_TOOL], stream = False, nudge_tool_calls = True)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert len(backend.calls) == 2
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == truncated


def test_streaming_heals_split_call_into_one_delta(monkeypatch):
    pieces = ["<tool", '<tool_call>{"name": "loo', '<tool_call>{"name": "lookup", "argum']
    cumulative = pieces + [_CALL_XML]
    backend = _ScriptedBackend(_fixed(*cumulative))
    payload = _request(tools = [LOOKUP_TOOL], stream = True)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    tool_deltas = [
        tc
        for o in objs
        for tc in (o.get("choices", [{}])[0].get("delta", {}) or {}).get("tool_calls", []) or []
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["function"]["name"] == "lookup"
    finishes = [
        o["choices"][0]["finish_reason"]
        for o in objs
        if o["choices"] and o["choices"][0].get("finish_reason")
    ]
    assert finishes == ["tool_calls"]


def test_what_this_backend_can_serve_reaches_it_rather_than_being_refused(monkeypatch):
    """An empty stop sequence is dropped rather than forwarded: it would match at
    position 0 and end every turn before its first token. ``{"type": "text"}``
    constrains nothing, so refusing it for want of a grammar engine would turn a
    request this backend serves into a 400."""
    backend = _ScriptedBackend(_fixed("hi"), stats = {"usage": {"prompt_tokens": 7}})
    payload = _request(stop = ["END", ""], response_format = {"type": "text"})
    body = _json_body(_call(payload, monkeypatch, backend, supports_tools = False))
    assert backend.calls[0]["stop"] == ["END"]
    assert body["choices"][0]["message"]["content"] == "hi"


def _totals(body):
    return {k: body["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


@pytest.mark.parametrize("tool_loop", [False, True])
def test_a_non_streaming_reply_reports_the_tokens_it_spent(monkeypatch, tool_loop):
    """The response model defaults usage to a zero-filled object, so omitting it
    reports zeros a client cannot tell from a real count. Both non-streaming
    shapes answer from the same stats the monitor reads."""
    spent = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    build = _ToolLoopBackend if tool_loop else _ScriptedBackend
    backend = build(_fixed("hello"), stats = {"usage": spent})
    payload = _request(stream = False, enable_tools = True) if tool_loop else _request(stream = False)
    body = _json_body(_call(payload, monkeypatch, backend, supports_tools = tool_loop))
    assert _totals(body) == spent
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def _monitor_entry(payload, monkeypatch, backend, **install_kwargs):
    """The one monitor row a request leaves behind, and what it raised, if it did."""
    from fastapi import HTTPException

    monitor = _install(monkeypatch, backend, **install_kwargs)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    error = None
    try:
        asyncio.run(_run())
    except HTTPException as exc:
        error = exc
    [entry] = monitor.snapshot()
    return entry, error


def test_one_monitor_row_describes_the_whole_turn(monkeypatch):
    """The last choice cannot speak for the turn: the row shows every reply, and a
    choice that ended without publishing stats is not billed the previous one's."""
    turns = iter(["first", "second"])
    backend = _ScriptedBackend(
        lambda messages, tools: [next(turns)],
        stats = [{"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, None],
    )
    entry, _ = _monitor_entry(_request(n = 2), monkeypatch, backend, supports_tools = False)
    assert "first" in entry["reply"] and "second" in entry["reply"]
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 3
    assert entry.get("stop_reason") is None


def test_streaming_cancel_does_not_finalize_tool_call(monkeypatch):
    # A cancelled stream must not promote buffered unclosed tool markup at finalize.
    import routes.inference as inf

    cancel_id = "cancel-me-6870"
    held = '<tool_call>{"name": "lookup", "arguments": {"q": "cats"}}'

    class _CancelMidStream(_ScriptedBackend):
        def __init__(self):
            super().__init__(_fixed(held))

        def generate_chat_response(
            self,
            *,
            messages,
            tools = None,
            stats_holder = None,
            **kwargs,
        ):
            self.calls.append({"messages": messages, "tools": tools, **kwargs})
            yield held
            inf._cancel_by_cancel_id_or_stash(cancel_id)

    backend = _CancelMidStream()
    payload = _request(tools = [LOOKUP_TOOL], stream = True, cancel_id = cancel_id)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    tool_deltas = [
        tc
        for o in objs
        for tc in (o.get("choices", [{}])[0].get("delta", {}) or {}).get("tool_calls", []) or []
    ]
    assert tool_deltas == []
    finishes = [
        o["choices"][0]["finish_reason"]
        for o in objs
        if o["choices"] and o["choices"][0].get("finish_reason")
    ]
    assert "tool_calls" not in finishes


def test_streaming_no_tools_verbatim(monkeypatch):
    backend = _ScriptedBackend(_fixed("hello ", "hello world"))
    payload = _request(stream = True)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    text = "".join(
        (o["choices"][0]["delta"].get("content") or "")
        for o in objs
        if o["choices"] and "delta" in o["choices"][0]
    )
    assert text == "hello world"
    finishes = [
        o["choices"][0]["finish_reason"]
        for o in objs
        if o["choices"] and o["choices"][0].get("finish_reason")
    ]
    assert finishes == ["stop"]


def test_streaming_gen_stream_error_is_not_model_text(monkeypatch):
    from core.inference.orchestrator import GenStreamError

    class _ErrorAfterPartial(_ScriptedBackend):
        def __init__(self):
            super().__init__(_fixed())

        def generate_chat_response(self, **_kwargs):
            yield "<think>partial"
            yield GenStreamError("Error: /tmp/secret traceback")

    backend = _ErrorAfterPartial()
    payload = _request(stream = True)
    response = _call(payload, monkeypatch, backend, supports_tools = False)
    chunks = _collect_sse(response)
    objs = _sse_objects(chunks)

    deltas = [o.get("choices", [{}])[0].get("delta", {}) for o in objs if o.get("choices")]
    assert any("partial" in json.dumps(delta) for delta in deltas)
    assert not any("/tmp/secret" in json.dumps(delta) for delta in deltas)
    errors = [o["error"]["message"] for o in objs if "error" in o]
    assert errors == ["An internal error occurred."]
    assert any(
        "data: [DONE]" in (chunk.decode() if isinstance(chunk, bytes) else chunk)
        for chunk in chunks
    )


def test_server_tool_streaming_invalid_event_is_error(monkeypatch):
    class _InvalidEventBackend(_ScriptedBackend):
        def __init__(self):
            super().__init__(_fixed())

        def generate_chat_completion_with_tools(self, **_kwargs):
            yield {"type": "content", "text": "partial"}
            yield "not-an-event"

    backend = _InvalidEventBackend()
    payload = _request(tools = [LOOKUP_TOOL], enable_tools = True, stream = True)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))

    errors = [o["error"]["message"] for o in objs if "error" in o]
    assert errors == ["An internal error occurred."]


def test_server_tool_heartbeat_is_not_sent_as_a_stall_keepalive(monkeypatch):
    import routes.inference as inf

    class _SilentToolBackend(_ScriptedBackend):
        def __init__(self):
            super().__init__(_fixed())

        def generate_chat_completion_with_tools(self, **_kwargs):
            yield {"type": "heartbeat"}
            yield {"type": "content", "text": "done"}

    payload = _request(tools = [LOOKUP_TOOL], enable_tools = True, stream = True)
    chunks = _collect_sse(_call(payload, monkeypatch, _SilentToolBackend()))
    assert inf._OPENAI_TOOL_HEARTBEAT_SSE in chunks
    assert inf._OPENAI_PASSTHROUGH_SSE_KEEPALIVE not in chunks


def test_streaming_repeated_snapshot_no_duplicate_call(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML, _CALL_XML, _CALL_XML[:5], _CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = True)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    tool_deltas = [
        tc
        for o in objs
        for tc in (o.get("choices", [{}])[0].get("delta", {}) or {}).get("tool_calls", []) or []
    ]
    assert len(tool_deltas) == 1


def test_streaming_parallel_cap(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML + _SEARCH_XML))
    payload = _request(tools = [LOOKUP_TOOL, SEARCH_TOOL], stream = True, parallel_tool_calls = False)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    tool_deltas = [
        tc
        for o in objs
        for tc in (o.get("choices", [{}])[0].get("delta", {}) or {}).get("tool_calls", []) or []
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["function"]["name"] == "lookup"


def test_streaming_generator_error_closes_cleanly(monkeypatch):
    def responder(messages, tools):
        raise RuntimeError("boom /secret/path")

    backend = _ScriptedBackend(responder)
    payload = _request(tools = [LOOKUP_TOOL], stream = True)
    response = _call(payload, monkeypatch, backend)
    chunks = _collect_sse(response)
    joined = "".join(c.decode() if isinstance(c, bytes) else c for c in chunks)
    assert "An internal error occurred" in joined
    assert "secret/path" not in joined  # CWE-209: no path leak
    assert backend.reset_count >= 1


def test_streaming_disconnect_resets_once(monkeypatch):
    class _DisconnectRequest(_Request):
        async def is_disconnected(self):
            return True

    backend = _ScriptedBackend(_fixed("a", "ab", "abc"))
    payload = _request(tools = [LOOKUP_TOOL], stream = True)
    _install(monkeypatch, backend)

    async def _run():
        resp = await openai_chat_completions(
            payload, request = _DisconnectRequest(), current_subject = "u"
        )
        return [c async for c in resp.body_iterator]

    asyncio.run(_run())
    assert backend.reset_count == 1


def test_mlx_uses_same_path(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_does_not_advertise_tools(monkeypatch):
    backend = _ScriptedBackend(_fixed("plain answer"))
    payload = _request(tools = [LOOKUP_TOOL], tool_choice = "none", stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["message"]["content"] == "plain answer"
    assert backend.calls[0]["tools"] is None


_CLIENT_TOOL_HISTORY = [
    {"role": "user", "content": "look up cats"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "cats"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "cats are mammals"},
]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tools = [LOOKUP_TOOL], tool_choice = "required"),
        dict(tools = [LOOKUP_TOOL]),
        dict(tools = [LOOKUP_TOOL], messages = _CLIENT_TOOL_HISTORY),
        dict(tools = [LOOKUP_TOOL], enable_tools = True),
    ],
)
@pytest.mark.parametrize("gptoss", [False, True])
def test_client_tools_the_model_cannot_take_are_refused(monkeypatch, kwargs, gptoss):
    backend = _ScriptedBackend(_fixed("prose instead of a call"))
    if gptoss:
        backend._is_gpt_oss_model = lambda: True
    payload = _request(stream = False, **kwargs)
    entry, error = _monitor_entry(payload, monkeypatch, backend, supports_tools = gptoss)

    assert error is not None and error.status_code == 400
    assert error.detail["error"]["code"] == "unsupported_parameter"
    assert error.detail["error"]["param"] == "tools"
    assert ("gpt-oss" in error.detail["error"]["message"]) is gptoss
    assert backend.calls == []
    assert entry["status"] == "error"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tools = [LOOKUP_TOOL], tool_choice = "none"),
        dict(tools = [LOOKUP_TOOL], tool_choice = "none", messages = _CLIENT_TOOL_HISTORY),
        dict(messages = _CLIENT_TOOL_HISTORY),
        dict(enable_tools = True),
    ],
)
def test_requests_without_an_active_client_catalog_still_answer(monkeypatch, kwargs):
    backend = _ScriptedBackend(_fixed("plain answer"))
    payload = _request(stream = False, **kwargs)
    body = _json_body(_call(payload, monkeypatch, backend, supports_tools = False))
    assert body["choices"][0]["message"]["content"] == "plain answer"
    assert backend.calls[0]["tools"] is None


def test_developer_message_folded_into_system_prompt(monkeypatch):
    backend = _ScriptedBackend(_fixed("ok"))
    payload = _request(
        messages = [
            ChatMessage(role = "developer", content = "always be terse"),
            ChatMessage(role = "user", content = "hi"),
        ],
        tools = [LOOKUP_TOOL],
        stream = False,
    )
    _call(payload, monkeypatch, backend)
    sent = backend.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "always be terse" in sent[0]["content"]
    assert all(m.get("role") != "developer" for m in sent)


def test_failed_nudge_retry_keeps_original_response(monkeypatch):
    state = {"n": 0}

    def responder(messages, tools):
        state["n"] += 1
        if state["n"] == 1:
            return ['<tool_call>{"name":"lookup"']
        raise RuntimeError("retry blew up")

    backend = _ScriptedBackend(responder)
    payload = _request(tools = [LOOKUP_TOOL], nudge_tool_calls = True, stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert state["n"] == 2
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == '<tool_call>{"name":"lookup"'


def test_a_discarded_nudge_retry_still_bills_the_tokens_it_spent(monkeypatch):
    # Double-failure nudge: report the delivered attempt's prompt, but sum both completions.
    first_stats = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    retry_stats = {"usage": {"prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 198}}

    class _PerCallStatsBackend(_ScriptedBackend):
        def __init__(self):
            super().__init__(lambda m, t: ['<tool_call>{"name":"lookup"'])
            self._stats_seq = [first_stats, retry_stats]

        def generate_chat_response(
            self,
            *,
            messages,
            tools = None,
            stats_holder = None,
            **kwargs,
        ):
            self.calls.append({"messages": messages, "tools": tools, **kwargs})
            stats = self._stats_seq[min(len(self.calls) - 1, len(self._stats_seq) - 1)]
            if stats_holder is not None:
                stats_holder["stats"] = stats
            for snap in self._responder(messages, tools):
                yield snap

    backend = _PerCallStatsBackend()
    payload = _request(tools = [LOOKUP_TOOL], nudge_tool_calls = True, stream = False)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    asyncio.run(_run())
    assert len(backend.calls) == 2
    [entry] = monitor.snapshot()
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 3 + 99


def test_a_nudge_retry_that_never_reported_is_not_billed_twice(monkeypatch):
    # The retry raises before publishing, so folding stats_holder into itself would double-count.
    first_stats = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}

    class _RetryRaisesBackend(_ScriptedBackend):
        def __init__(self):
            super().__init__(lambda m, t: ['<tool_call>{"name":"lookup"'])

        def generate_chat_response(
            self,
            *,
            messages,
            tools = None,
            stats_holder = None,
            **kwargs,
        ):
            self.calls.append({"messages": messages, "tools": tools, **kwargs})
            if len(self.calls) > 1:
                raise RuntimeError("retry blew up before reporting anything")
            if stats_holder is not None:
                stats_holder["stats"] = first_stats
            for snap in self._responder(messages, tools):
                yield snap

    backend = _RetryRaisesBackend()
    payload = _request(tools = [LOOKUP_TOOL], nudge_tool_calls = True, stream = False)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    body = _json_body(asyncio.run(_run()))
    assert len(backend.calls) == 2
    assert body["usage"]["prompt_tokens"] == 7
    assert body["usage"]["completion_tokens"] == 3
    [entry] = monitor.snapshot()
    assert entry["completion_tokens"] == 3


def test_cached_prompt_tokens_reach_the_usage_details(monkeypatch):
    # MLX folds its reused prefix into prompt_tokens, so a caller reading the
    # OpenAI field must see the same count rather than a flat zero.
    stats = {
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 4,
            "total_tokens": 1204,
            "prompt_tokens_details": {"cached_tokens": 1100},
        }
    }
    backend = _ScriptedBackend(_fixed("hi"), stats = stats)
    body = _json_body(_call(_request(stream = False), monkeypatch, backend))
    details = body["usage"]["prompt_tokens_details"]
    assert details["cached_tokens"] == 1100
    assert details["cached_tokens"] <= body["usage"]["prompt_tokens"]


def test_cached_tokens_never_exceed_the_prompt_they_describe(monkeypatch):
    # Two choices can report different prompt counts (the nudge rebuilds a longer
    # prompt), so the count and its details must come from the same choice.
    rich = {
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 1100},
        }
    }
    lean = {"usage": {"prompt_tokens": 1000, "completion_tokens": 4}}
    backend = _ScriptedBackend(_fixed("hi"), stats = [rich, lean])
    body = _json_body(_call(_request(stream = False, n = 2), monkeypatch, backend))
    usage = body["usage"]
    assert usage["prompt_tokens"] == 1000
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert usage["prompt_tokens_details"]["cached_tokens"] <= usage["prompt_tokens"]


def test_a_successful_nudge_retry_bills_both_attempts(monkeypatch):
    # The healed retry is delivered, but the first attempt's tokens still count.
    first_stats = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    retry_stats = {"usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}}

    class _HealsOnRetryBackend(_ScriptedBackend):
        def __init__(self):
            super().__init__(
                lambda m, t: [_CALL_XML if len(self.calls) > 1 else '<tool_call>{"name":"lookup"']
            )
            self._stats_seq = [first_stats, retry_stats]

        def generate_chat_response(
            self,
            *,
            messages,
            tools = None,
            stats_holder = None,
            **kwargs,
        ):
            self.calls.append({"messages": messages, "tools": tools, **kwargs})
            if stats_holder is not None:
                stats_holder["stats"] = self._stats_seq[
                    min(len(self.calls) - 1, len(self._stats_seq) - 1)
                ]
            for snap in self._responder(messages, tools):
                yield snap

    backend = _HealsOnRetryBackend()
    payload = _request(tools = [LOOKUP_TOOL], nudge_tool_calls = True, stream = False)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    body = _json_body(asyncio.run(_run()))
    assert len(backend.calls) == 2
    assert body["choices"][0]["message"]["tool_calls"]
    assert body["usage"]["prompt_tokens"] == 20
    assert body["usage"]["completion_tokens"] == 3 + 5
    assert body["usage"]["total_tokens"] == 28


def test_monitor_records_healed_call_not_raw_xml(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    asyncio.run(_run())
    snap = monitor.snapshot(include_details = True)
    replies = json.dumps(snap)
    assert "<tool_call>" not in replies
    assert "lookup" in replies


def test_streaming_monitor_records_healed_call_not_raw_xml(monkeypatch):
    backend = _ScriptedBackend(
        _fixed("Sure. ", 'Sure. <tool_call>{"name": "loo', "Sure. " + _CALL_XML)
    )
    payload = _request(tools = [LOOKUP_TOOL], stream = True)
    monitor = _install(monkeypatch, backend)

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    response = asyncio.run(_run())
    _collect_sse(response)
    replies = json.dumps(monitor.snapshot(include_details = True))
    assert "<tool_call>" not in replies
    assert "Sure. " in replies
    assert "[tool_calls] lookup(" in replies


def test_forced_tool_choice_narrows_templated_tools(monkeypatch):
    backend = _ScriptedBackend(_fixed(_SEARCH_XML))
    payload = _request(
        tools = [LOOKUP_TOOL, SEARCH_TOOL],
        stream = False,
        tool_choice = {"type": "function", "function": {"name": "search"}},
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    templated = backend.calls[0]["tools"]
    assert [t["function"]["name"] for t in templated] == ["search"]
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "search"


def test_multimodal_content_parts_flattened_for_local_template(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,"},
                    },
                ],
            )
        ],
        tools = [LOOKUP_TOOL],
        stream = False,
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    templated = backend.calls[0]["messages"]
    assert all(isinstance(m.get("content"), str) for m in templated)
    assert any(_without_date_note(m["content"]) == "what is this?" for m in templated)
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_string_arguments_history_deserialized_for_template(monkeypatch):
    backend = _ScriptedBackend(_fixed("done"))
    payload = _request(
        tools = [LOOKUP_TOOL],
        stream = False,
        messages = [
            ChatMessage(role = "user", content = "weather?"),
            ChatMessage(
                role = "assistant",
                content = None,
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "weather"}'},
                    }
                ],
            ),
            ChatMessage(role = "tool", tool_call_id = "call_0", content = "sunny"),
        ],
    )
    _json_body(_call(payload, monkeypatch, backend))
    assistant = next(m for m in backend.calls[0]["messages"] if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"q": "weather"}


def test_unparseable_arguments_string_left_untouched(monkeypatch):
    backend = _ScriptedBackend(_fixed("ok"))
    payload = _request(
        tools = [LOOKUP_TOOL],
        stream = False,
        messages = [
            ChatMessage(role = "user", content = "hi"),
            ChatMessage(
                role = "assistant",
                content = None,
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "not json {"},
                    }
                ],
            ),
            ChatMessage(role = "tool", tool_call_id = "call_0", content = "y"),
        ],
    )
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["message"]["content"] == "ok"
    assistant = next(m for m in backend.calls[0]["messages"] if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == "not json {"


def test_mcp_enabled_without_server_tools_uses_passthrough(monkeypatch):
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False, mcp_enabled = True)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]


def _fold(*turns):
    """Run the tool loop's fold over turns in order, as the orchestrator does."""
    from core.inference.orchestrator import _summed_tool_loop_stats

    total = None
    for turn in turns:
        total = _summed_tool_loop_stats(total, turn)
    return total


def _turn(
    prompt,
    completion,
    *,
    timings = True,
    **extra,
):
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        **extra,
    }
    stats = {"usage": usage}
    if timings:
        stats["timings"] = {"predicted_ms": completion * 10.0, "predicted_n": completion}
    return stats


def test_every_tool_loop_turn_is_billed_not_just_the_last():
    """The turns that produced the tool call spent tokens too, so the reply sums
    them; only the prompt is the last turn's, since it already carries the
    earlier results."""
    folded = _fold(_turn(100, 20), _turn(160, 30), _turn(220, 5))

    assert folded["usage"] == {"prompt_tokens": 220, "completion_tokens": 55, "total_tokens": 275}
    assert folded["timings"]["predicted_n"] == 55
    assert folded["timings"]["predicted_ms"] == pytest.approx(550.0)
    assert folded["timings"]["predicted_per_token_ms"] == pytest.approx(10.0)


def test_a_turn_that_ends_before_reporting_does_not_erase_the_loop():
    """A cancelled or errored final turn has no counts of its own. Seeding the
    fold from it would drop everything the loop already spent."""
    assert _fold(_turn(100, 20), None, _turn(160, 30))["usage"]["completion_tokens"] == 50
    assert _fold(_turn(100, 20), _turn(160, 30), None)["usage"]["completion_tokens"] == 50

    partial = _fold(_turn(100, 20), _turn(160, 30, timings = False))
    assert partial["timings"]["predicted_n"] == 20
    errored = _fold(_turn(100, 20), {"timings": {"predicted_ms": 1.0, "predicted_n": 1}})
    assert errored["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def test_completion_details_are_summed_with_the_completion_they_describe():
    """Carrying the last turn's details would report them against every turn's
    tokens."""
    folded = _fold(
        _turn(100, 20, completion_tokens_details = {"reasoning_tokens": 7}),
        _turn(160, 30, completion_tokens_details = {"reasoning_tokens": 3}),
    )
    assert folded["usage"]["completion_tokens"] == 50
    assert folded["usage"]["completion_tokens_details"] == {"reasoning_tokens": 10}


_PNG_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _vision_backend(*snapshots):
    backend = _ScriptedBackend(_fixed(*snapshots))
    backend.models["sf-model"]["is_vision"] = True
    return backend


def _image_message(text = "run the tests"):
    return ChatMessage(
        role = "user",
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_1x1}"}},
        ],
    )


def test_image_turn_keeps_the_client_tool_catalog(monkeypatch):
    backend = _vision_backend(_CALL_XML)
    payload = _request(messages = [_image_message()], tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))

    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]
    assert backend.calls[0]["image"] is not None
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_legacy_image_field_keeps_the_client_tool_catalog(monkeypatch):
    backend = _vision_backend(_CALL_XML)
    payload = _request(
        messages = [ChatMessage(role = "user", content = "run the tests")],
        image_base64 = _PNG_1x1,
        tools = [LOOKUP_TOOL],
        stream = False,
    )
    _call(payload, monkeypatch, backend)

    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]
    assert backend.calls[0]["image"] is not None


def test_a_turn_asking_for_several_replies_sends_them_as_one_batch(monkeypatch):
    """One command carries every choice, each with its own seed."""
    from routes.inference import _choice_seed

    backend = _ScriptedBackend(_fixed("hi"))
    body = _json_body(_call(_request(stream = False, n = 3, seed = 11), monkeypatch, backend))

    assert len(body["choices"]) == 3
    assert len(backend.batch_calls) == 1, "the choices did not go out together"
    call = backend.batch_calls[0]
    effective = [row.get("seed", call["shared"].get("seed")) for row in call["rows"]]
    assert effective == [11, _choice_seed(11, 1), _choice_seed(11, 2)], effective


class _StoppedAfterFirstRowBackend(_ScriptedBackend):
    """A backend that cannot batch: rows run apart and a Stop skips the rest."""

    def generate_chat_batch(
        self,
        rows,
        *,
        stats_holder = None,
        cancel_event = None,
        **kwargs,
    ):
        self.batch_calls.append({"rows": rows, "shared": kwargs})
        yield 0, "partial"
        cancel_event.set()
        yield 0, None
        for row in range(1, len(rows)):
            yield row, None
        if stats_holder is not None:
            stats_holder["stats"] = [{"completion_tokens": 1}] + [None] * (len(rows) - 1)


def test_a_stop_during_the_first_choice_returns_no_empty_choices(monkeypatch):
    backend = _StoppedAfterFirstRowBackend(_fixed("unused"))
    body = _json_body(_call(_request(stream = False, n = 3), monkeypatch, backend))

    assert len(backend.batch_calls) == 1
    assert [c["message"]["content"] for c in body["choices"]] == ["partial"], body["choices"]


_RF_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
_RF_FORMAT = {"type": "json_schema", "json_schema": {"name": "c", "schema": _RF_SCHEMA}}
_MARKUP_DOC = '{"city":"<think>Paris</think>"}'


@pytest.mark.parametrize("chunked", [False, True], ids = ["whole", "streamed"])
@pytest.mark.parametrize(
    "prefix, prefilled, reasoning",
    [
        ("", False, ""),
        ("<think>weighing</think>", False, "weighing"),
        ("<think>weighing</think>", True, "weighing"),
        ("", True, ""),
    ],
)
def test_markers_inside_a_constrained_document_stay_in_it(chunked, prefix, prefilled, reasoning):
    from routes.inference import _ResponsesReasoningExtractor

    extractor = _ResponsesReasoningExtractor(
        parse_think_markers = True, reasoning_prefilled = prefilled, single_block = True
    )
    text = prefix + _MARKUP_DOC
    got_reasoning = got_content = ""
    for piece in text if chunked else [text]:
        delta_reasoning, delta_content = extractor.feed(piece)
        got_reasoning += delta_reasoning
        got_content += delta_content
    delta_reasoning, delta_content = extractor.finish()

    assert got_content + delta_content == _MARKUP_DOC
    assert got_reasoning + delta_reasoning == reasoning
    assert json.loads(_MARKUP_DOC)["city"] == "<think>Paris</think>"


@pytest.mark.parametrize("reasoning", [False, True], ids = ["plain", "reasoning"])
def test_the_contract_reaches_the_backend_and_its_reply_comes_back_whole(monkeypatch, reasoning):
    pytest.importorskip("llguidance.mlx")
    backend = _ScriptedBackend(_fixed(_MARKUP_DOC))
    backend.models["sf-model"]["is_mlx"] = True
    caps = {"supports_tools": False, "supports_reasoning": reasoning}
    reply = _call(_request(response_format = _RF_FORMAT), monkeypatch, backend, **caps)
    body = _json_body(reply)
    assert backend.calls[0]["response_format"] == _RF_FORMAT
    assert backend.calls[0]["reasoning_is_extracted"] is reasoning
    message = body["choices"][0]["message"]
    assert message["content"] == _MARKUP_DOC
    assert not message.get("reasoning_content")


class _FittedToolLoopBackend(_ToolLoopBackend):
    def __init__(self, responder, **kwargs):
        super().__init__(responder, **kwargs)
        self.fits: list = []

    def compact_chat_context(
        self,
        messages,
        *,
        system_prompt = "",
        **kwargs,
    ):
        self.fits.append(kwargs)
        return {"messages": messages, "system_prompt": system_prompt}


def _serve(
    monkeypatch,
    *,
    frames = True,
    **fields,
):
    """Send to an installed backend; ``frames`` off drops the control-frame header."""
    if fields.get("response_format"):
        pytest.importorskip("llguidance.mlx")
    if not frames:
        monkeypatch.setattr(_Request, "headers", {})
    payload = _request(**fields)
    response = asyncio.run(
        openai_chat_completions(payload, request = _Request(), current_subject = "u")
    )
    return _collect_sse(response) if payload.stream else response


@pytest.mark.parametrize(
    "is_mlx, checkpointed, extra, offered",
    [
        (True, True, {}, ["search_conversation"]),
        (False, True, {}, None),
        (True, False, {}, None),
        (True, True, {"context_policy": "rolling"}, None),
        (True, True, {"context_overflow": None}, None),
        (True, True, {"tool_choice": "none"}, None),
        (True, True, {"max_tool_calls_per_message": 0}, None),
        (True, True, {"n": 2}, None),
        (True, True, {"permission_mode": "ask"}, None),
        (True, True, {"stream": True, "frames": False, "permission_mode": "ask"}, None),
        (True, True, {"response_format": _RF_FORMAT}, None),
        (True, True, {"tools": [LOOKUP_TOOL]}, ["lookup"]),
    ],
)
def test_a_compacted_mlx_thread_keeps_archive_search_with_tools_off(
    monkeypatch, is_mlx, checkpointed, extra, offered
):
    import routes.inference as inf

    monkeypatch.setattr(inf, "_thread_has_conversation_archive", lambda thread_id: True)
    monkeypatch.setattr(
        inf, "_thread_has_checkpoint", lambda thread_id, messages = None: checkpointed
    )
    backend = _FittedToolLoopBackend(_fixed("done"))
    backend.models["sf-model"]["is_mlx"] = is_mlx
    fields = {
        "stream": False,
        "enable_tools": False,
        "thread_id": "saved",
        "context_overflow": "truncate_oldest",
        "context_policy": "checkpoint",
        **extra,
    }
    probed = []

    def _features(*_args, **kwargs):
        probed.append(bool(kwargs.get("tools")))
        return {"supports_tools": True, "supports_reasoning": False}

    _install(monkeypatch, backend)
    monkeypatch.setattr(inf, "_detect_safetensors_features", _features)
    _serve(monkeypatch, **fields)
    tools = backend.calls[0]["tools"]
    assert (tools and [tool["function"]["name"] for tool in tools]) == offered
    # The template branch that is classified is the one the request renders.
    assert probed[0] is bool(offered)


@pytest.mark.parametrize("asked", [{"enable_tools": True}, {"mcp_enabled": True}])
def test_tool_choice_none_keeps_a_compacted_mlx_thread_tool_free(monkeypatch, asked):
    import routes.inference as inf

    monkeypatch.setattr(inf, "_thread_has_conversation_archive", lambda thread_id: True)
    monkeypatch.setattr(inf, "_thread_has_checkpoint", lambda thread_id, messages = None: True)
    backend = _FittedToolLoopBackend(_fixed("done"))
    backend.models["sf-model"]["is_mlx"] = True
    _install(monkeypatch, backend)
    # A GGUF model that is still loading already reports its tool support.
    loading = SimpleNamespace(**{**vars(_llama_stub()), "supports_tools": True})
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: loading)
    fields = {"thread_id": "saved", "context_overflow": "truncate_oldest", "tool_choice": "none"}
    _serve(monkeypatch, stream = False, context_policy = "checkpoint", **fields, **asked)

    assert backend.calls[0]["tools"] is None
    assert [fit["recall_reachable"] for fit in backend.fits] == [False]


_TURNS = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
_WITH_TOOLS = "{% if tools %}<tool_call>{{ tools }}</tool_call>{% endif %}" + _TURNS
# Named templates: only the branch a turn carrying tools renders decides, either way round.
_TOOL_BRANCH = {"default": _TURNS, "tool_use": _WITH_TOOLS}
_DEFAULT_BRANCH = {"default": _WITH_TOOLS, "tool_use": _TURNS}


@pytest.mark.parametrize(
    "extra, supports_tools, gpt_oss, reachable",
    [
        ({}, True, False, True),
        ({}, False, False, False),
        ({}, _TOOL_BRANCH, False, True),
        ({}, _DEFAULT_BRANCH, False, False),
        ({}, True, True, False),
        ({"tool_choice": "none"}, True, False, False),
        ({"stream": True, "permission_mode": "ask"}, True, False, True),
        ({"stream": True, "frames": False}, True, False, True),
        ({"stream": True, "frames": False, "permission_mode": "ask"}, True, False, False),
        ({"response_format": _RF_FORMAT}, True, False, False),
        ({"tools": [LOOKUP_TOOL]}, True, False, False),
    ],
)
def test_a_plain_mlx_fit_may_reset_only_where_the_loop_can_reopen(
    monkeypatch, extra, supports_tools, gpt_oss, reachable
):
    backend = _FittedToolLoopBackend(_fixed("done"))
    backend.models["sf-model"]["is_mlx"] = True
    backend._is_gpt_oss_model = lambda: gpt_oss
    import routes.inference as inf

    classify = inf._detect_safetensors_features
    _install(monkeypatch, backend, supports_tools = supports_tools is True)
    if isinstance(supports_tools, dict):
        # The real classifier, so the branch it selects is the one under test.
        backend.models["sf-model"]["chat_template_info"] = {"template": supports_tools}
        monkeypatch.setattr(inf, "_detect_safetensors_features", classify)
    base = {"stream": False, "enable_tools": False, "context_overflow": "truncate_oldest"}
    _serve(monkeypatch, **{**base, **extra})
    assert [fit["recall_reachable"] for fit in backend.fits] == [reachable]


@pytest.mark.parametrize(
    "tokenizer_body, processor_body, reachable",
    [(_WITH_TOOLS, _TURNS, False), (_TURNS, _WITH_TOOLS, True)],
    ids = ["processor_drops_tools", "processor_renders_tools"],
)
def test_a_vision_model_is_asked_about_the_body_its_text_turn_renders(
    monkeypatch, tokenizer_body, processor_body, reachable
):
    import routes.inference as inf

    backend = _FittedToolLoopBackend(_fixed("done"))
    backend.models["sf-model"]["is_mlx"] = True
    classify = inf._detect_safetensors_features
    _install(monkeypatch, backend)
    backend.models["sf-model"]["chat_template_info"] = {
        "template": tokenizer_body,
        "processor_template": processor_body,
    }
    monkeypatch.setattr(inf, "_detect_safetensors_features", classify)
    _serve(monkeypatch, stream = False, enable_tools = False, context_overflow = "truncate_oldest")
    assert [fit["recall_reachable"] for fit in backend.fits] == [reachable]


class _TrimmingBackend(_FittedToolLoopBackend):
    """Fits by keeping only the newest message."""

    def compact_chat_context(self, messages, **kwargs):
        self.fits.append(kwargs)
        event = {"type": "context_truncated", "dropped_messages": len(messages) - 1, "fits": True}
        return {"messages": messages[-1:], "system_prompt": "fitted", "events": [event]}


def _texts(messages):
    return [message["content"] for message in messages]


def _overflowing(**kwargs):
    turns = [("user", "old"), ("assistant", "ok"), ("user", "new")]
    return _request(
        messages = [ChatMessage(role = role, content = text) for role, text in turns],
        stream = False,
        context_overflow = "truncate_oldest",
        **kwargs,
    )


@pytest.mark.parametrize("slots", [4, 1], ids = ["batched", "one_by_one"])
def test_every_choice_of_an_mlx_request_starts_from_one_fit(monkeypatch, slots):
    from routes.inference import _choice_seed

    backend = _TrimmingBackend(_fixed("done"))
    backend.models["sf-model"]["is_mlx"] = True
    backend.effective_parallel_slots = slots
    body = _json_body(_call(_overflowing(n = 3, seed = 11), monkeypatch, backend))

    assert len(backend.fits) == 1
    assert len(backend.batch_calls) == (slots > 1)
    sent = [(_texts(c["messages"]), c["system_prompt"], c["seed"]) for c in backend.calls]
    seeds = (11, _choice_seed(11, 1), _choice_seed(11, 2))
    assert sent == [(["new"], "fitted", seed) for seed in seeds]
    assert body["context_truncated"]["dropped_messages"] == 2


def test_an_mlx_prompt_carrying_a_picture_is_sent_unfitted(monkeypatch):
    backend = _TrimmingBackend(_fixed("done"))
    backend.models["sf-model"].update(is_mlx = True, is_vision = True)
    _call(_overflowing(image_base64 = _PNG_1x1), monkeypatch, backend)

    assert backend.fits == [] and len(backend.calls[0]["messages"]) == 3


def test_a_nudge_retry_extends_the_fitted_prompt_and_is_not_refitted(monkeypatch):
    backend = _TrimmingBackend(_fixed('<tool_call>{"name": "lookup"'))
    backend.models["sf-model"]["is_mlx"] = True
    payload = _overflowing(tools = [LOOKUP_TOOL], nudge_tool_calls = True)
    _call(payload, monkeypatch, backend)

    first, retry = (_texts(call["messages"]) for call in backend.calls)
    assert len(backend.fits) == 1 and first == ["new"]
    assert retry[:1] == first and len(retry) > 1


class _VisionToolLoopBackend(_ToolLoopBackend):
    def __init__(self, responder, **kwargs):
        super().__init__(responder, **kwargs)
        self.models["sf-model"]["is_vision"] = True

    @staticmethod
    def resize_image(image):
        return image


def test_an_attached_image_still_reaches_the_tool_loop(monkeypatch):
    backend = _VisionToolLoopBackend(_fixed("done"))
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "what is in this picture"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
                    },
                ],
            )
        ],
        enable_tools = True,
        stream = False,
    )

    _call(payload, monkeypatch, backend)

    [call] = backend.calls
    assert call["tools"], "tools must survive an attached image on a vision model"
    assert call["images"], "the attachment has to ride into the loop"
    markers = [
        part
        for message in call["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    ]
    assert len(markers) == len(call["images"])


_TOKENIZER_TEMPLATE_WITH_TOOLS = (
    "{% if tools %}<|im_start|>system\n"
    "{% for t in tools %}{{ t.function.name }}\n{% endfor %}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
)
_PROCESSOR_TEMPLATE_WITHOUT_TOOLS = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
)


def test_the_image_tool_loop_is_gated_on_the_body_that_renders(monkeypatch):
    """The loop now runs on a vision model with an attachment, and generation renders
    through the PROCESSOR template. Classifying its tool support from the tokenizer body
    started the loop on a model whose prompt carries no schemas at all."""
    import routes.inference as inf

    backend = _VisionToolLoopBackend(_fixed("done"))
    backend.models["sf-model"]["chat_template_info"] = {
        "template": _TOKENIZER_TEMPLATE_WITH_TOOLS,
        "processor_template": _PROCESSOR_TEMPLATE_WITHOUT_TOOLS,
        "renders_image": True,
    }
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "what is in this picture"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
                    },
                ],
            )
        ],
        enable_tools = True,
        stream = False,
    )

    _install(monkeypatch, backend)
    monkeypatch.setattr(
        inf,
        "_detect_safetensors_features",
        lambda _backend, template, **k: {
            "supports_tools": template == _TOKENIZER_TEMPLATE_WITH_TOOLS
        },
    )

    async def _run():
        return await openai_chat_completions(payload, request = _Request(), current_subject = "u")

    asyncio.run(_run())

    assert backend.calls, "generation never ran"
    assert not any(
        call.get("tools") for call in backend.calls
    ), "the tool loop was driven from a template the image render never selects"


def test_a_client_catalog_keeps_an_image_out_of_the_server_loop(monkeypatch):
    """Letting the loop take a picture must not take the request with it: #10092 routes
    image-plus-tools to the passthrough so the CALLER's schemas are the ones rendered.
    Claiming it here answered the client with Unsloth's built-ins instead."""
    backend = _VisionToolLoopBackend(_fixed("a plain answer"))
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "what is in this picture"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
                    },
                ],
            )
        ],
        tools = [LOOKUP_TOOL],
        enable_tools = True,
        stream = False,
    )

    _call(payload, monkeypatch, backend)

    assert backend.calls, "generation never ran"
    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]
    assert "images" not in backend.calls[0] or not backend.calls[0].get("images")


def test_a_replayed_picture_sits_beside_the_result_that_produced_it(monkeypatch):
    """The passthrough flatten costs the markers their positions. Promoting after it
    puts each batch's turn straight after its result, so "the tool call above" names
    the right call -- one detached block of every payload, placed wherever the
    attachment's turn happened to be, did not."""
    import base64
    import io
    import json

    from PIL import Image

    from core.inference import mcp_images

    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), (10, 120, 200)).save(buffer, format = "PNG")
    envelope = json.dumps(
        [{"data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/png"}]
    )

    backend = _ScriptedBackend(_fixed("a plain answer"))
    backend.models["sf-model"]["is_vision"] = True
    backend.models["sf-model"]["chat_template_info"] = {
        "template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "renders_image": True,
    }
    payload = _request(
        messages = [
            ChatMessage(role = "user", content = "take a shot"),
            ChatMessage(
                role = "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "mcp__fs__shot", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role = "tool",
                tool_call_id = "call_0",
                content = "[1 image returned]\n" + mcp_images.SENTINEL + envelope,
            ),
            ChatMessage(role = "assistant", content = "a blue square"),
            ChatMessage(role = "user", content = "and what about now"),
        ],
        tools = [LOOKUP_TOOL],
        stream = False,
    )

    _call(payload, monkeypatch, backend)

    [call] = backend.calls
    sent = call["messages"]
    assert call["images"] and len(call["images"]) == 1
    [marker_at] = [
        index
        for index, message in enumerate(sent)
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image" for part in message["content"])
    ]
    assert sent[marker_at - 1]["role"] == "tool", [m["role"] for m in sent]
    lead = next(part["text"] for part in sent[marker_at]["content"] if part.get("type") == "text")
    assert lead.startswith(mcp_images.IMAGE_TURN_TEXT), lead
    roles = [m["role"] for m in sent]
    assert all(a != "user" or b != "user" for a, b in zip(roles, roles[1:])), roles


def test_a_replay_only_image_turn_also_keeps_the_client_catalog(monkeypatch):
    """The gate read `image is None`, so a resumed chat whose only pictures are
    replayed MCP ones looked image-free and the loop took the request -- swapping the
    caller's schemas for Unsloth's on exactly the path this change adds."""
    import base64
    import io
    import json

    from PIL import Image

    from core.inference import mcp_images

    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), (10, 120, 200)).save(buffer, format = "PNG")
    envelope = json.dumps(
        [{"data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/png"}]
    )

    backend = _VisionToolLoopBackend(_fixed("a plain answer"))
    payload = _request(
        messages = [
            ChatMessage(role = "user", content = "take a shot"),
            ChatMessage(
                role = "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "mcp__fs__shot", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role = "tool",
                tool_call_id = "call_0",
                content = "[1 image returned]\n" + mcp_images.SENTINEL + envelope,
            ),
            ChatMessage(role = "user", content = "what colour was it"),
        ],
        tools = [LOOKUP_TOOL],
        enable_tools = True,
        stream = False,
    )

    _call(payload, monkeypatch, backend)

    assert backend.calls, "generation never ran"
    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]


def test_the_plain_route_leaves_the_attachment_marker_to_the_backend(monkeypatch):
    """The backends snapshot the conversation's existing markers as history's before
    topping up, so a marker the ROUTE pre-added is counted as a replayed picture's.
    With the attachment on an earlier turn than a tool's picture the two pixels then
    bind to each other's turns, and the model reads the screenshot as the diagram."""
    import base64
    import io
    import json

    from PIL import Image

    from core.inference import mcp_images

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(buffer, format = "PNG")
    envelope = json.dumps(
        [{"data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/png"}]
    )

    backend = _ScriptedBackend(_fixed("a plain answer"))
    backend.models["sf-model"]["is_vision"] = True
    backend.models["sf-model"]["chat_template_info"] = {
        "template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "renders_image": True,
    }
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "here is my diagram"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
                    },
                ],
            ),
            ChatMessage(role = "assistant", content = "noted"),
            ChatMessage(
                role = "assistant",
                content = "",
                tool_calls = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "mcp__s__shot", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role = "tool",
                tool_call_id = "c1",
                content = "[1 image returned]\n" + mcp_images.SENTINEL + envelope,
            ),
            ChatMessage(role = "user", content = "which one is bluer?"),
        ],
        stream = False,
    )

    # supports_tools=False makes this the plain path; a tools template would route to the passthrough.
    _call(payload, monkeypatch, backend, supports_tools = False)

    [call] = backend.calls
    sent, replayed = call["messages"], call["images"] or []
    assert call["image"] is not None, "the attachment reached the backend"
    assert len(replayed) == 1, "so did the replayed picture"

    prior = mcp_images.image_marker_parts(sent)
    topped = mcp_images.top_up_image_markers(sent, len(replayed) + 1, ordinal = call["image_ordinal"])
    ordered = mcp_images.pixels_in_marker_order(topped, prior, ["MCP"], "ATTACHMENT")

    assert len(mcp_images.image_marker_parts(topped)) == 2, topped
    assert ordered == [
        "ATTACHMENT",
        "MCP",
    ], "the route pre-marked, so the pixels bound to each other's markers"


def test_video_turn_with_tools_enabled_keeps_the_client_tool_catalog(monkeypatch):
    """A clip rules out the server loop like an image; the passthrough keeps catalog and clip."""
    backend = _vision_backend(_CALL_XML)
    backend.models["sf-model"]["has_video_input"] = True
    clip = "AAAAGGZ0eXBtcDQy"
    payload = _request(
        messages = [ChatMessage(role = "user", content = "run the tests")],
        video_base64 = clip,
        tools = [LOOKUP_TOOL],
        enable_tools = True,
        stream = False,
    )
    body = _json_body(_call(payload, monkeypatch, backend))

    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]
    assert backend.calls[0]["video"] == clip
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_a_nudge_retry_keeps_the_video_on_the_question_turn(monkeypatch):
    """Without the clip's turn marked first, the retry's correction turn would take the clip."""
    truncated = '<tool_call>{"name": "lookup"'

    def responder(messages, tools):
        nudged = any(
            "native tool-call format" in (m.get("content") or "")
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        return [_CALL_XML] if nudged else [truncated]

    backend = _vision_backend(_CALL_XML)
    backend._responder = responder
    backend.models["sf-model"]["has_video_input"] = True
    clip = "AAAAGGZ0eXBtcDQy"
    payload = _request(
        messages = [ChatMessage(role = "user", content = "run the tests")],
        video_base64 = clip,
        tools = [LOOKUP_TOOL],
        stream = False,
        nudge_tool_calls = True,
    )
    _call(payload, monkeypatch, backend)

    assert len(backend.calls) == 2, "the nudge retry did not run"
    retry = backend.calls[1]["messages"]
    assert backend.calls[1]["video"] == clip
    question = next(m for m in retry if m["role"] == "user")
    assert question["content"][0] == {"type": "video"}
    assert retry[-1]["role"] == "user" and isinstance(retry[-1]["content"], str)


def test_an_input_audio_part_beside_a_clip_is_refused_too(monkeypatch):
    """The part is lifted onto audio_base64 before the clip gate, so one rule covers both spellings."""
    from fastapi import HTTPException

    import routes.inference as inf

    backend = _vision_backend("a plain answer")
    backend.models["sf-model"]["has_video_input"] = True
    payload = _request(
        video_base64 = "AAAAGGZ0eXBtcDQy",
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "AAAA", "format": "wav"},
                    },
                    {"type": "text", "text": "what do you hear and see?"},
                ],
            }
        ],
        stream = False,
    )

    with pytest.raises(HTTPException) as exc:
        _call(payload, monkeypatch, backend)

    assert exc.value.status_code == 400
    assert exc.value.detail == inf._AUDIO_VIDEO_INPUT_DETAIL
    assert backend.calls == []


def test_audio_beside_a_clip_is_refused_before_any_dispatch(monkeypatch):
    """A model without audio input never enters the audio path, so the conflict is settled first."""
    from fastapi import HTTPException

    import routes.inference as inf

    backend = _vision_backend("a plain answer")
    backend.models["sf-model"]["has_video_input"] = True
    payload = _request(video_base64 = "AAAAGGZ0eXBtcDQy", audio_base64 = "AAAA", stream = False)

    with pytest.raises(HTTPException) as exc:
        _call(payload, monkeypatch, backend)

    assert exc.value.status_code == 400
    assert exc.value.detail == inf._AUDIO_VIDEO_INPUT_DETAIL
    assert backend.calls == []
