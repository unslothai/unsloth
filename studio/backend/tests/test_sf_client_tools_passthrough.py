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
from types import SimpleNamespace

from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import openai_chat_completions
from core.inference.api_monitor import ApiMonitor


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
        if stats_holder is not None and self._stats is not None:
            stats_holder["stats"] = self._stats
        for snap in snapshots:
            yield snap

    def reset_generation_state(self):
        self.reset_count += 1


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
        lambda *a, **k: {"supports_tools": supports_tools},
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


# ── Non-streaming ─────────────────────────────────────────────────


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
    # The client tools reached the generator (template injection).
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
    # No tools and no tool messages -> plain path, normal ChatCompletion.
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "just a plain answer"
    assert choice["message"].get("tool_calls") is None


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
    # The tool history reached the generator intact (role=tool + assistant.tool_calls).
    sent = backend.calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert "tool" in roles
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant.get("tool_calls")


def test_dict_arguments_history_does_not_crash(monkeypatch):
    # Non-spec client: assistant tool_calls[].function.arguments as a dict.
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
    # tool_choice forces `search`; a `lookup` text call must NOT promote.
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


# ── Nudge ─────────────────────────────────────────────────────────


def test_nudge_default_off_single_generation(monkeypatch):
    # Signal present but unparseable; without opt-in, no retry.
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
    assert len(backend.calls) == 2  # exactly one retry
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == truncated


# ── Streaming ─────────────────────────────────────────────────────


def test_streaming_heals_split_call_into_one_delta(monkeypatch):
    # Cumulative snapshots that build the call across many increments.
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


def test_streaming_cancel_does_not_finalize_tool_call(monkeypatch):
    # A stream cancelled via the registry ("Stop") must NOT promote the
    # buffered-but-unclosed tool markup at finalize, else it executes a tool
    # the user just cancelled. Guarded on cancel_event at the finalize step.
    import routes.inference as inf

    cancel_id = "cancel-me-6870"
    # Balanced JSON but no closing </tool_call> -> healer HOLDS it until finalize.
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
            yield held  # healer holds the unclosed call
            inf._cancel_by_cancel_id_or_stash(cancel_id)  # user hits Stop before EOF

    backend = _CancelMidStream()
    payload = _request(tools = [LOOKUP_TOOL], stream = True, cancel_id = cancel_id)
    response = _call(payload, monkeypatch, backend)
    objs = _sse_objects(_collect_sse(response))
    tool_deltas = [
        tc
        for o in objs
        for tc in (o.get("choices", [{}])[0].get("delta", {}) or {}).get("tool_calls", []) or []
    ]
    assert tool_deltas == []  # no tool promoted after cancel
    finishes = [
        o["choices"][0]["finish_reason"]
        for o in objs
        if o["choices"] and o["choices"][0].get("finish_reason")
    ]
    assert "tool_calls" not in finishes  # ends with finish_reason=stop, not tool_calls


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


def test_streaming_repeated_snapshot_no_duplicate_call(monkeypatch):
    # Repeated then shrunk cumulative snapshots must not double-heal.
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
    # MLX and safetensors share get_inference_backend(); one scripted backend covers both.
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_does_not_advertise_tools(monkeypatch):
    # tool_choice="none": no tools rendered into the template; history templating still applies.
    backend = _ScriptedBackend(_fixed("plain answer"))
    payload = _request(tools = [LOOKUP_TOOL], tool_choice = "none", stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert body["choices"][0]["message"]["content"] == "plain answer"
    assert backend.calls[0]["tools"] is None


def test_developer_message_folded_into_system_prompt(monkeypatch):
    # The "developer" role folds into one leading system message (local templates reject it).
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
    # A raising retry must not 500; the first response is returned.
    state = {"n": 0}

    def responder(messages, tools):
        state["n"] += 1
        if state["n"] == 1:
            return ['<tool_call>{"name":"lookup"']  # unhealable signal
        raise RuntimeError("retry blew up")

    backend = _ScriptedBackend(responder)
    payload = _request(tools = [LOOKUP_TOOL], nudge_tool_calls = True, stream = False)
    body = _json_body(_call(payload, monkeypatch, backend))
    assert state["n"] == 2
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == '<tool_call>{"name":"lookup"'


def test_discarded_nudge_retry_reports_first_attempt_usage(monkeypatch):
    # Double-failure nudge: the first response is delivered, but the retry's
    # generate() overwrites stats_holder. The monitor must record the FIRST
    # attempt's usage, not the discarded retry's.
    first_stats = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    retry_stats = {"usage": {"prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 198}}

    class _PerCallStatsBackend(_ScriptedBackend):
        def __init__(self):
            # Unhealable truncated markup on both attempts -> retry is discarded.
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
    assert len(backend.calls) == 2  # first attempt + one discarded retry
    [entry] = monitor.snapshot()
    # The delivered response is the first attempt, so its usage must be reported.
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 3


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
    # Monitor mirrors what the client received, never the healed-away raw markup.
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
    # A forced function is the only schema rendered into the template.
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
    # Remote image URLs leave image=None, so content arrives as a part LIST:
    # text parts are kept, the image part dropped.
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
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
    assert any(m["content"] == "what is this?" for m in templated)
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_string_arguments_history_deserialized_for_template(monkeypatch):
    # JSON-string tool_calls arguments become dicts in the templated copy;
    # the HTTP response stays OpenAI-shaped.
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
    # mcp_enabled=true with an empty registry must not silently drop the
    # declared tools; the gate keys on the server-side path claiming the request.
    backend = _ScriptedBackend(_fixed(_CALL_XML))
    payload = _request(tools = [LOOKUP_TOOL], stream = False, mcp_enabled = True)
    body = _json_body(_call(payload, monkeypatch, backend))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert backend.calls[0]["tools"] == [LOOKUP_TOOL]
