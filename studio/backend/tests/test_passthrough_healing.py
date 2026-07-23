# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for core/inference/passthrough_healing.py: promoting text-form
tool calls back into structured calls on the client-tool passthrough. The
route-level wiring (OpenAI / Anthropic / Responses endpoints) is covered in
their own endpoint test files; this file exercises the shared state machine
and helpers directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.passthrough_healing import (  # noqa: E402
    StreamToolCallHealer,
    heal_gate,
    heal_openai_message,
    nudge_messages,
    nudge_should_retry,
    response_has_promotable_calls,
)

TOOLS = [
    {"type": "function", "function": {"name": "Bash", "parameters": {}}},
    {"type": "function", "function": {"name": "Read", "parameters": {}}},
]

BASH_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
XML_BASH = '<tool_call>{"name":"Bash","arguments":{"cmd":"ls"}}</tool_call>'
XML_UNDECLARED = '<tool_call>{"name":"Nuke","arguments":{}}</tool_call>'


def _events_text(events):
    return "".join(text for kind, text in events if kind == "text")


def _events_calls(events):
    return [call for kind, call in events if kind == "tool_call"]


class TestHealGate:
    def test_returns_declared_names(self):
        assert heal_gate(None, TOOLS) == {"Bash", "Read"}
        assert heal_gate(True, TOOLS) == {"Bash", "Read"}

    def test_opt_out_and_no_tools(self):
        assert heal_gate(False, TOOLS) is None
        assert heal_gate(None, []) is None
        assert heal_gate(None, None) is None

    def test_malformed_tool_entries_ignored(self):
        assert heal_gate(None, ["nonsense", {"function": "x"}, {}]) is None

    def test_tool_choice_none_disables(self):
        assert heal_gate(None, TOOLS, "none") is None

    def test_tool_choice_forced_function_narrows_allowlist(self):
        forced = {"type": "function", "function": {"name": "Bash"}}
        assert heal_gate(None, TOOLS, forced) == {"Bash"}

    def test_tool_choice_forced_undeclared_function_disables(self):
        forced = {"type": "function", "function": {"name": "Nuke"}}
        assert heal_gate(None, TOOLS, forced) is None

    def test_tool_choice_auto_and_required_keep_full_set(self):
        assert heal_gate(None, TOOLS, "auto") == {"Bash", "Read"}
        assert heal_gate(None, TOOLS, "required") == {"Bash", "Read"}

    def test_tool_choice_unrecognized_dict_keeps_full_set(self):
        assert heal_gate(None, TOOLS, {"type": "function"}) == {"Bash", "Read"}


class TestHealOpenaiMessage:
    def test_promotes_xml_and_strips_content(self):
        msg = {"role": "assistant", "content": XML_BASH}
        assert heal_openai_message(msg, {"Bash"}) is True
        assert msg["content"] is None
        (call,) = msg["tool_calls"]
        assert call["function"]["name"] == "Bash"
        assert json.loads(call["function"]["arguments"]) == {"cmd": "ls"}

    def test_keeps_surrounding_prose(self):
        msg = {"role": "assistant", "content": f"Let me check.\n{XML_BASH}"}
        assert heal_openai_message(msg, {"Bash"}) is True
        assert msg["content"] == "Let me check."

    def test_undeclared_name_not_promoted(self):
        msg = {"role": "assistant", "content": XML_UNDECLARED}
        assert heal_openai_message(msg, {"Bash"}) is False
        assert msg["content"] == XML_UNDECLARED
        assert "tool_calls" not in msg

    def test_structured_calls_untouched(self):
        msg = {"role": "assistant", "content": XML_BASH, "tool_calls": [{"id": "x"}]}
        assert heal_openai_message(msg, {"Bash"}) is False
        assert msg["content"] == XML_BASH

    def test_prose_only_untouched(self):
        msg = {"role": "assistant", "content": "just an answer"}
        assert heal_openai_message(msg, {"Bash"}) is False

    def test_bare_string_arguments_use_schema_key(self):
        msg = {
            "role": "assistant",
            "content": '<tool_call>{"name":"Bash","arguments":"echo hi"}</tool_call>',
        }
        assert heal_openai_message(msg, {"Bash"}, [BASH_COMMAND_TOOL]) is True
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert args == {"command": "echo hi"}

    def test_bare_string_arguments_decline_ambiguous_schema(self):
        msg = {
            "role": "assistant",
            "content": '<tool_call>{"name":"Bash","arguments":"echo hi"}</tool_call>',
        }
        assert heal_openai_message(msg, {"Bash"}, TOOLS) is False
        assert "tool_calls" not in msg

    def test_mixed_declared_and_undeclared_promotes_declared_keeps_undeclared_text(self):
        # Span-exact removal: only the promoted Bash markup is dropped; the
        # undeclared Nuke call's text stays in the content byte-intact.
        content = f"pre {XML_BASH} mid {XML_UNDECLARED} post"
        msg = {"role": "assistant", "content": content}
        assert heal_openai_message(msg, {"Bash"}) is True
        (call,) = msg["tool_calls"]
        assert call["function"]["name"] == "Bash"
        assert XML_UNDECLARED in msg["content"]
        assert "pre" in msg["content"] and "post" in msg["content"]
        assert XML_BASH not in msg["content"]

    def test_multiple_declared_calls_all_promoted(self):
        content = f"{XML_BASH} and {XML_BASH}"
        msg = {"role": "assistant", "content": content}
        assert heal_openai_message(msg, {"Bash"}) is True
        assert len(msg["tool_calls"]) == 2

    def test_mixed_formats_promote_in_document_order(self):
        func_read = "<function=Read><parameter=path>a.txt</parameter></function>"
        content = f"{func_read} then {XML_BASH}"
        msg = {"role": "assistant", "content": content}
        assert heal_openai_message(msg, {"Bash", "Read"}) is True
        assert [call["function"]["name"] for call in msg["tool_calls"]] == ["Read", "Bash"]
        assert msg["content"] == "then"

    def test_unparseable_closed_block_not_deleted(self):
        # A closed <tool_call> block whose body never parses is model output,
        # not a promotable call; it must survive promotion of its neighbor.
        garbage = "<tool_call>not json at all</tool_call>"
        content = f"{XML_BASH} {garbage}"
        msg = {"role": "assistant", "content": content}
        assert heal_openai_message(msg, {"Bash"}) is True
        assert garbage in msg["content"]


class TestStreamHealer:
    def test_plain_text_passes_through(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed("hello ") + healer.feed("world") + healer.finalize()
        assert _events_text(events) == "hello world"
        assert not _events_calls(events)

    def test_complete_call_in_one_chunk(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed(f"On it. {XML_BASH}") + healer.finalize()
        assert _events_text(events) == "On it. "
        (call,) = _events_calls(events)
        assert call["function"]["name"] == "Bash"
        assert healer.healed

    def test_signal_split_across_chunks(self):
        healer = StreamToolCallHealer({"Bash"})
        events = []
        for piece in ["<tool", '_call>{"name":"Bash",', '"arguments":{}}</tool_call>']:
            events += healer.feed(piece)
        events += healer.finalize()
        assert _events_text(events) == ""
        assert len(_events_calls(events)) == 1

    def test_closed_malformed_tool_block_flushes_immediately(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed("<tool_call>not json</tool_call> after")
        assert _events_text(events) == "<tool_call>not json</tool_call> after"
        assert not _events_calls(events)

    def test_mixed_formats_stream_in_document_order(self):
        healer = StreamToolCallHealer({"Bash", "Read"})
        func_read = "<function=Read><parameter=path>a.txt</parameter></function>"
        events = healer.feed(f"{func_read} then {XML_BASH}") + healer.finalize()
        assert [call["function"]["name"] for call in _events_calls(events)] == ["Read", "Bash"]
        assert _events_text(events).strip() == "then"

    def test_false_alarm_html_flushes(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed("use the <div> tag") + healer.finalize()
        assert _events_text(events) == "use the <div> tag"
        assert not _events_calls(events)

    def test_partial_signal_tail_held_then_flushed_at_end(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed("trailing <tool")
        assert _events_text(events) == "trailing "  # tail held back
        events += healer.finalize()
        assert _events_text(events) == "trailing <tool"

    def test_mixed_calls_promote_declared_flush_undeclared_in_order(self):
        # Declared + undeclared in the same buffer: the declared call is
        # promoted, the undeclared markup flushes as text, and event order
        # follows document order (call first here, since it came first).
        healer = StreamToolCallHealer({"Bash"})
        text = f"{XML_BASH} then {XML_UNDECLARED} post"
        events = healer.feed(text) + healer.finalize()
        assert [k for k, _ in events if k == "tool_call"] == ["tool_call"]
        assert events[0][0] == "tool_call"
        joined = _events_text(events)
        assert XML_UNDECLARED in joined
        assert "then" in joined and "post" in joined

    def test_text_between_two_healed_calls_keeps_document_order(self):
        # call A, " middle ", call B in ONE buffer must stream as
        # call A -> text -> call B, never both calls then the text.
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed(f"{XML_BASH} middle {XML_BASH}") + healer.finalize()
        kinds = [k for k, _ in events]
        assert kinds == ["tool_call", "text", "tool_call"]
        assert events[1][1] == " middle "

    def test_undeclared_then_declared_keeps_document_order(self):
        # The undeclared block precedes the declared call; its raw text must
        # be emitted BEFORE the promoted call event, never after.
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed(f"{XML_UNDECLARED} then {XML_BASH}") + healer.finalize()
        kinds = [k for k, _ in events]
        assert kinds.index("tool_call") == len(kinds) - 1
        (call,) = _events_calls(events)
        assert call["function"]["name"] == "Bash"
        assert XML_UNDECLARED in _events_text(events)

    def test_declared_promoted_then_late_undeclared_flushes_raw(self):
        # Streaming causality: the declared call completed and was already
        # emitted before the undeclared one arrived. The undeclared markup
        # must still reach the client as raw text (no data loss).
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed(f"{XML_BASH} then ")
        assert len(_events_calls(events)) == 1
        events += healer.feed(XML_UNDECLARED) + healer.finalize()
        assert XML_UNDECLARED in _events_text(events)
        assert len(_events_calls(events)) == 1

    def test_undeclared_tool_flushes_raw(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed(XML_UNDECLARED) + healer.finalize()
        assert _events_text(events) == XML_UNDECLARED
        assert not _events_calls(events)

    def test_two_calls_and_text_between(self):
        healer = StreamToolCallHealer({"Bash", "Read"})
        xml_read = '<tool_call>{"name":"Read","arguments":{"path":"f"}}</tool_call>'
        events = healer.feed(f"{XML_BASH} then {xml_read}") + healer.finalize()
        calls = _events_calls(events)
        assert [c["function"]["name"] for c in calls] == ["Bash", "Read"]
        assert [c["id"] for c in calls] == ["call_0", "call_1"]
        assert _events_text(events).strip() == "then"

    def test_mistral_array_multiple_calls_all_promoted_in_stream(self):
        # A canonical Mistral [TOOL_CALLS] array carries several calls under a
        # SINGLE signal. Draining only the first call would leave the residue
        # starting at ",{...}]" (no signal), so later calls in the same array
        # must be promoted in the same pass, not flushed as raw text.
        healer = StreamToolCallHealer({"get_weather", "get_time"})
        array = (
            '[TOOL_CALLS][{"name":"get_weather","arguments":{"city":"Paris"}},'
            '{"name":"get_time","arguments":{"tz":"UTC"}}]'
        )
        events = healer.feed(array) + healer.finalize()
        calls = _events_calls(events)
        assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
        assert [c["id"] for c in calls] == ["call_0", "call_1"]
        assert _events_text(events) == ""

    def test_mistral_array_multiple_calls_promoted_char_by_char(self):
        healer = StreamToolCallHealer({"get_weather", "get_time"})
        array = (
            '[TOOL_CALLS][{"name":"get_weather","arguments":{"city":"Paris"}},'
            '{"name":"get_time","arguments":{"tz":"UTC"}}]'
        )
        events = []
        for ch in array:
            events += healer.feed(ch)
        events += healer.finalize()
        calls = _events_calls(events)
        assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
        assert _events_text(events) == ""

    def test_mistral_array_undeclared_middle_kept_as_text_others_promoted(self):
        # A mid-array element for a tool that is not declared must survive as
        # text while the declared neighbours on either side still promote in
        # document order.
        healer = StreamToolCallHealer({"a", "c"})
        array = (
            '[TOOL_CALLS][{"name":"a","arguments":{}},'
            '{"name":"b","arguments":{}},{"name":"c","arguments":{}}]'
        )
        events = healer.feed(array) + healer.finalize()
        assert [c["function"]["name"] for c in _events_calls(events)] == ["a", "c"]
        assert '"b"' in _events_text(events)

    def test_mistral_array_then_trailing_prose(self):
        healer = StreamToolCallHealer({"a", "b"})
        array = '[TOOL_CALLS][{"name":"a","arguments":{}},{"name":"b","arguments":{}}]'
        events = healer.feed(f"{array} all done") + healer.finalize()
        assert [c["function"]["name"] for c in _events_calls(events)] == ["a", "b"]
        assert "all done" in _events_text(events)

    def test_incomplete_call_healed_at_finalize(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed('<tool_call>{"name":"Bash","arguments":{"cmd":"ls"}}')
        assert events == []  # held
        events = healer.finalize()
        (call,) = _events_calls(events)
        assert call["function"]["name"] == "Bash"

    def test_teaching_text_flushes_at_finalize(self):
        healer = StreamToolCallHealer({"Bash"})
        events = healer.feed("<tool_call> is the marker syntax") + healer.finalize()
        assert _events_text(events) == "<tool_call> is the marker syntax"
        assert not _events_calls(events)

    def test_hold_bound_flushes(self):
        healer = StreamToolCallHealer({"Bash"})
        blob = "<tool_call>" + "x" * (64 * 1024 + 10)
        events = healer.feed(blob) + healer.finalize()
        assert _events_text(events) == blob
        assert not _events_calls(events)

    def test_dormant_after_structured_delta(self):
        healer = StreamToolCallHealer({"Bash"})
        held = healer.feed("prefix <tool")
        flush = healer.structured_tool_call_seen()
        after = healer.feed(XML_BASH) + healer.finalize()
        assert _events_text(held + flush + after) == f"prefix <tool{XML_BASH}"
        assert not _events_calls(after)


class TestNudgeHelpers:
    def _resp(
        self,
        content,
        tool_calls = None,
    ):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"choices": [{"message": msg, "finish_reason": "stop"}]}

    def test_retry_on_unparseable_signal(self):
        # Signal present but the JSON never parses and no declared name matches.
        data = self._resp("<tool_call>call Bash somehow???")
        assert nudge_should_retry(data, {"Read"}) is True

    def test_no_retry_on_clean_prose(self):
        assert nudge_should_retry(self._resp("all done"), {"Bash"}) is False

    def test_no_retry_when_heal_would_succeed(self):
        assert nudge_should_retry(self._resp(XML_BASH), {"Bash"}) is False

    def test_no_retry_with_structured_calls(self):
        data = self._resp("", tool_calls = [{"id": "x"}])
        assert nudge_should_retry(data, {"Bash"}) is False

    def test_no_retry_when_healing_disabled(self):
        assert nudge_should_retry(self._resp("<tool_call>???"), None) is False

    def test_nudge_messages_shape(self):
        data = self._resp("<tool_call>garbage")
        suffix = nudge_messages(data, {"Bash", "Read"})
        assert [m["role"] for m in suffix] == ["assistant", "user"]
        assert suffix[0]["content"] == "<tool_call>garbage"
        assert "`Bash` or `Read`" in suffix[1]["content"]

    def test_retry_with_undeclared_structured_call_is_not_an_improvement(self):
        # The retry replaces the original only when it carries a USABLE call:
        # a structured call naming an undeclared tool must not count.
        undeclared = [
            {"id": "x", "type": "function", "function": {"name": "Nuke", "arguments": "{}"}}
        ]
        declared = [
            {"id": "y", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}
        ]
        assert response_has_promotable_calls(self._resp("", undeclared), {"Bash"}) is False
        assert response_has_promotable_calls(self._resp("", declared), {"Bash"}) is True

    def test_retry_with_mixed_structured_calls_is_not_an_improvement(self):
        # ALL structured calls must be declared: the caller forwards the whole
        # list (and a parallel cap could keep only the FIRST), so a mixed retry
        # could still hand the client an undeclared tool.
        mixed = [
            {"id": "x", "type": "function", "function": {"name": "Nuke", "arguments": "{}"}},
            {"id": "y", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
        ]
        assert response_has_promotable_calls(self._resp("", mixed), {"Bash"}) is False
        assert (
            response_has_promotable_calls(self._resp("", list(reversed(mixed))), {"Bash"}) is False
        )

    @pytest.mark.parametrize(
        "data",
        [
            None,
            "not a dict",
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": None}]},  # llama-server error bodies do this
            {"choices": [{"message": "not a dict"}]},
            {"choices": [{"message": {"content": None}}]},
            {"error": {"message": "boom"}},
        ],
    )
    def test_malformed_response_shapes_never_raise(self, data):
        # A malformed upstream body must degrade to "nothing to heal/nudge",
        # never crash the request with an AttributeError.
        assert nudge_should_retry(data, {"Bash"}) is False
        assert response_has_promotable_calls(data, {"Bash"}) is False
        suffix = nudge_messages(data, {"Bash"})
        assert suffix[0] == {"role": "assistant", "content": ""}


# ── Route-level wiring (OpenAI passthrough) ─────────────────────────────
# Mirrors the fake-llama-server patterns in test_openai_tool_passthrough.py.

import asyncio  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import httpx  # noqa: E402

from core.inference.api_monitor import ApiMonitor  # noqa: E402
from models.inference import ChatCompletionRequest, ChatMessage  # noqa: E402
from routes.inference import (  # noqa: E402
    _openai_passthrough_non_streaming,
    _openai_passthrough_stream,
)

LOOKUP_TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}
LOOKUP_XML = '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'


def _payload(**kwargs):
    defaults = dict(
        model = "default",
        messages = [ChatMessage(role = "user", content = "hi")],
        tools = [LOOKUP_TOOL],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _llama_backend():
    return SimpleNamespace(
        base_url = "http://llama.test",
        context_length = 4096,
        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
    )


def _upstream_message(
    content,
    tool_calls = None,
    finish_reason = "stop",
):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-up",
        "object": "chat.completion",
        "created": 1,
        "model": "gguf",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


class ScriptedClient:
    """Fake nonstreaming_client() returning scripted JSON bodies, counting POSTs."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.posts = []

    async def post(
        self,
        _url,
        json = None,
        timeout = None,
        headers = None,
    ):
        self.posts.append(json)
        return httpx.Response(200, json = self.bodies[min(len(self.posts) - 1, len(self.bodies) - 1)])


async def _drive_non_streaming(monkeypatch, payload, bodies):
    import routes.inference as inf_mod

    client = ScriptedClient(bodies)
    monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: client)
    response = await _openai_passthrough_non_streaming(
        _llama_backend(), payload, "gguf", monitor_id = None
    )
    return client, json.loads(response.body)


async def _drive_stream(monkeypatch, payload, lines):
    import routes.inference as inf_mod

    class Request:
        async def is_disconnected(self):
            return False

    async def fake_send(*_args, **_kwargs):
        return httpx.Response(200, content = b"")

    async def fake_items(*_args, **_kwargs):
        for line in lines:
            yield line

    monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
    monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
    response = await _openai_passthrough_stream(
        Request(),
        threading.Event(),
        _llama_backend(),
        payload,
        "gguf",
        "chatcmpl-test",
        monitor_id = None,
    )
    return [chunk async for chunk in response.body_iterator]


def _stream_payloads(chunks):
    out = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                out.append(json.loads(line[6:]))
    return out


class TestOpenaiNonStreamingRoute:
    def test_heals_xml_to_tool_calls(self, monkeypatch):
        async def _run():
            client, data = await _drive_non_streaming(
                monkeypatch, _payload(), [_upstream_message(LOOKUP_XML)]
            )
            choice = data["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            (call,) = choice["message"]["tool_calls"]
            assert call["function"]["name"] == "lookup"
            assert json.loads(call["function"]["arguments"]) == {"q": "x"}
            assert choice["message"]["content"] is None
            assert data["usage"]["total_tokens"] == 3  # usage preserved
            assert len(client.posts) == 1  # healing never re-requests

        asyncio.run(_run())

    def test_bare_string_uses_client_schema_key(self, monkeypatch):
        async def _run():
            content = '<tool_call>{"name":"Bash","arguments":"echo hi"}</tool_call>'
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(tools = [BASH_COMMAND_TOOL]),
                [_upstream_message(content)],
            )
            (call,) = data["choices"][0]["message"]["tool_calls"]
            assert json.loads(call["function"]["arguments"]) == {"command": "echo hi"}

        asyncio.run(_run())

    def test_opt_out_relays_verbatim(self, monkeypatch):
        async def _run():
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(auto_heal_tool_calls = False),
                [_upstream_message(LOOKUP_XML)],
            )
            choice = data["choices"][0]
            assert choice["message"]["content"] == LOOKUP_XML
            assert "tool_calls" not in choice["message"]
            assert choice["finish_reason"] == "stop"

        asyncio.run(_run())

    def test_no_tools_untouched(self, monkeypatch):
        async def _run():
            _, data = await _drive_non_streaming(
                monkeypatch, _payload(tools = None), [_upstream_message(LOOKUP_XML)]
            )
            assert data["choices"][0]["message"]["content"] == LOOKUP_XML

        asyncio.run(_run())

    def test_undeclared_tool_not_promoted(self, monkeypatch):
        async def _run():
            xml = '<tool_call>{"name":"rogue","arguments":{}}</tool_call>'
            _, data = await _drive_non_streaming(monkeypatch, _payload(), [_upstream_message(xml)])
            assert data["choices"][0]["message"]["content"] == xml
            assert "tool_calls" not in data["choices"][0]["message"]

        asyncio.run(_run())

    def test_structured_calls_untouched(self, monkeypatch):
        async def _run():
            native = [
                {
                    "id": "call_up",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(),
                [_upstream_message("", tool_calls = native, finish_reason = "tool_calls")],
            )
            assert data["choices"][0]["message"]["tool_calls"] == native

        asyncio.run(_run())

    def test_length_finish_reason_preserved(self, monkeypatch):
        async def _run():
            # Truncated generation: the healed call stays attached but the
            # client must still see the truncation, so length is never
            # upgraded to tool_calls.
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(),
                [_upstream_message(LOOKUP_XML, finish_reason = "length")],
            )
            choice = data["choices"][0]
            assert choice["finish_reason"] == "length"
            (call,) = choice["message"]["tool_calls"]
            assert call["function"]["name"] == "lookup"

        asyncio.run(_run())

    def test_tool_choice_none_relays_verbatim(self, monkeypatch):
        async def _run():
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(tool_choice = "none"),
                [_upstream_message(LOOKUP_XML)],
            )
            message = data["choices"][0]["message"]
            assert message["content"] == LOOKUP_XML
            assert "tool_calls" not in message

        asyncio.run(_run())

    def test_tool_choice_forcing_other_function_not_promoted(self, monkeypatch):
        async def _run():
            _, data = await _drive_non_streaming(
                monkeypatch,
                _payload(tool_choice = {"type": "function", "function": {"name": "other"}}),
                [_upstream_message(LOOKUP_XML)],
            )
            message = data["choices"][0]["message"]
            assert message["content"] == LOOKUP_XML
            assert "tool_calls" not in message

        asyncio.run(_run())

    def test_mixed_declared_and_undeclared_promotes_and_keeps_text(self, monkeypatch):
        async def _run():
            rogue = '<tool_call>{"name":"rogue","arguments":{}}</tool_call>'
            mixed = f"{LOOKUP_XML} also {rogue}"
            _, data = await _drive_non_streaming(
                monkeypatch, _payload(), [_upstream_message(mixed)]
            )
            choice = data["choices"][0]
            (call,) = choice["message"]["tool_calls"]
            assert call["function"]["name"] == "lookup"
            assert rogue in choice["message"]["content"]
            assert choice["finish_reason"] == "tool_calls"

        asyncio.run(_run())

    def test_healed_then_native_stream_indexes_disjoint(self, monkeypatch):
        async def _run():
            # A healed text-form call goes out first (index 0); a native
            # structured delta follows. Clients merge deltas by index, so the
            # native call must be shifted off index 0 or the two would merge.
            native_line = (
                'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_native","type":"function","function":'
                '{"name":"lookup","arguments":"{}"}}]}}]}'
            )
            lines = [
                'data: {"id":"c1","choices":[{"index":0,"delta":{"content":'
                + json.dumps(LOOKUP_XML)
                + "}}]}",
                native_line,
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(stream = True), lines)
            indexes = {}
            for payload_data in _stream_payloads(chunks):
                for ch in payload_data.get("choices", []):
                    for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                        indexes.setdefault(tc["index"], tc.get("id"))
            assert indexes.get(0, "").startswith("call_") and indexes[0] != "call_native"
            assert indexes.get(1) == "call_native"

        asyncio.run(_run())

    def test_role_delta_precedes_healed_stream_content(self, monkeypatch):
        async def _run():
            lines = [
                'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":'
                + json.dumps(LOOKUP_XML)
                + "}}]}",
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(stream = True), lines)
            payloads = _stream_payloads(chunks)
            first_delta = payloads[0]["choices"][0]["delta"]
            assert first_delta == {"role": "assistant"}
            assert "tool_calls" in payloads[1]["choices"][0]["delta"]

        asyncio.run(_run())

    def test_same_chunk_role_content_finish_delays_finish_until_after_healed_tool(
        self, monkeypatch
    ):
        async def _run():
            lines = [
                'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":'
                + json.dumps(LOOKUP_XML)
                + '},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(stream = True), lines)
            payloads = _stream_payloads(chunks)
            assert payloads[0]["choices"][0]["finish_reason"] is None
            assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
            assert "tool_calls" in payloads[1]["choices"][0]["delta"]
            assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"

        asyncio.run(_run())


GARBAGE_SIGNAL = "<tool_call>call lookup somehow???"


class TestNudgeRetryOpenai:
    def test_retry_recovers_call(self, monkeypatch):
        async def _run():
            client, data = await _drive_non_streaming(
                monkeypatch,
                _payload(nudge_tool_calls = True),
                [_upstream_message(GARBAGE_SIGNAL), _upstream_message(LOOKUP_XML)],
            )
            assert len(client.posts) == 2  # exactly one retry
            # Prefix byte-identical, nudge suffix appended (KV-cache reuse guard).
            original, retry = client.posts
            assert retry["messages"][: len(original["messages"])] == original["messages"]
            suffix = retry["messages"][len(original["messages"]) :]
            assert [m["role"] for m in suffix] == ["assistant", "user"]
            assert suffix[0]["content"] == GARBAGE_SIGNAL
            # The healed retry response is returned.
            (call,) = data["choices"][0]["message"]["tool_calls"]
            assert call["function"]["name"] == "lookup"
            assert data["choices"][0]["finish_reason"] == "tool_calls"

        asyncio.run(_run())

    def test_retry_still_garbage_returns_original(self, monkeypatch):
        async def _run():
            client, data = await _drive_non_streaming(
                monkeypatch,
                _payload(nudge_tool_calls = True),
                [_upstream_message(GARBAGE_SIGNAL), _upstream_message(GARBAGE_SIGNAL + "2")],
            )
            assert len(client.posts) == 2
            assert data["choices"][0]["message"]["content"] == GARBAGE_SIGNAL
            assert "tool_calls" not in data["choices"][0]["message"]

        asyncio.run(_run())

    def test_default_off_single_post(self, monkeypatch):
        async def _run():
            client, _ = await _drive_non_streaming(
                monkeypatch, _payload(), [_upstream_message(GARBAGE_SIGNAL)]
            )
            assert len(client.posts) == 1

        asyncio.run(_run())

    def test_no_retry_on_clean_prose(self, monkeypatch):
        async def _run():
            client, _ = await _drive_non_streaming(
                monkeypatch,
                _payload(nudge_tool_calls = True),
                [_upstream_message("all done")],
            )
            assert len(client.posts) == 1

        asyncio.run(_run())

    def test_no_retry_when_heal_succeeds(self, monkeypatch):
        async def _run():
            client, data = await _drive_non_streaming(
                monkeypatch,
                _payload(nudge_tool_calls = True),
                [_upstream_message(LOOKUP_XML)],
            )
            assert len(client.posts) == 1
            assert data["choices"][0]["message"]["tool_calls"]

        asyncio.run(_run())

    def test_heal_opt_out_disables_nudge_too(self, monkeypatch):
        async def _run():
            client, _ = await _drive_non_streaming(
                monkeypatch,
                _payload(auto_heal_tool_calls = False, nudge_tool_calls = True),
                [_upstream_message(GARBAGE_SIGNAL)],
            )
            assert len(client.posts) == 1

        asyncio.run(_run())


class TestNudgeRetryAnthropic:
    async def _drive(
        self,
        monkeypatch,
        bodies,
        nudge = None,
    ):
        import routes.inference as inf_mod
        from routes.inference import _anthropic_passthrough_non_streaming

        client = ScriptedClient(bodies)
        monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: client)
        response = await _anthropic_passthrough_non_streaming(
            _llama_backend(),
            [{"role": "user", "content": "hi"}],
            [LOOKUP_TOOL],
            0.7,
            0.95,
            None,
            256,
            "msg_test",
            "gguf",
            nudge_tool_calls = nudge,
        )
        return client, json.loads(response.body)

    def test_retry_recovers_tool_use(self, monkeypatch):
        async def _run():
            client, data = await self._drive(
                monkeypatch,
                [_upstream_message(GARBAGE_SIGNAL), _upstream_message(LOOKUP_XML)],
                nudge = True,
            )
            assert len(client.posts) == 2
            (block,) = [b for b in data["content"] if b["type"] == "tool_use"]
            assert block["name"] == "lookup"
            assert data["stop_reason"] == "tool_use"

        asyncio.run(_run())

    def test_healed_tool_use_precedes_trailing_text(self, monkeypatch):
        async def _run():
            _, data = await self._drive(monkeypatch, [_upstream_message(f"{LOOKUP_XML} done")])
            assert [block["type"] for block in data["content"]] == ["tool_use", "text"]
            assert data["content"][1]["text"] == "done"

        asyncio.run(_run())

    def test_default_off(self, monkeypatch):
        async def _run():
            client, _ = await self._drive(monkeypatch, [_upstream_message(GARBAGE_SIGNAL)])
            assert len(client.posts) == 1

        asyncio.run(_run())


class TestAnthropicPassthroughHealingText:
    """Non-streaming Anthropic passthrough must relay unpromoted (undeclared)
    text-form calls as text, matching the OpenAI passthrough contract. Once
    heal_openai_message promotes the declared call it span-trims only that
    markup and deliberately leaves the undeclared bytes in the content; the
    legacy blanket _TOOL_XML_RE strip must not delete them.
    """

    async def _drive(self, monkeypatch, upstream):
        import routes.inference as inf_mod
        from routes.inference import _anthropic_passthrough_non_streaming

        client = ScriptedClient([upstream])
        monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: client)
        response = await _anthropic_passthrough_non_streaming(
            _llama_backend(),
            [{"role": "user", "content": "hi"}],
            [LOOKUP_TOOL],
            0.7,
            0.95,
            None,
            256,
            "msg_test",
            "gguf",
        )
        return json.loads(response.body)

    def test_mixed_declared_and_undeclared_relays_undeclared_as_text(self, monkeypatch):
        async def _run():
            content = f"Running now. {LOOKUP_XML} then {XML_UNDECLARED} done."
            data = await self._drive(monkeypatch, _upstream_message(content))
            # Declared lookup call is promoted into a structured tool_use block.
            (tool_use,) = [b for b in data["content"] if b["type"] == "tool_use"]
            assert tool_use["name"] == "lookup"
            text = " ".join(b["text"] for b in data["content"] if b["type"] == "text")
            assert XML_UNDECLARED in text
            assert "Running now." in text and "done." in text
            assert LOOKUP_XML not in text

        asyncio.run(_run())


class TestAnthropicEmitterHealing:
    def _events(
        self,
        emitter,
        chunks,
        finish = True,
    ):
        lines = []
        for chunk in chunks:
            lines += emitter.feed_chunk(chunk)
        if finish:
            lines += emitter.finish()
        return [json.loads(ln.split("data: ", 1)[1]) for ln in lines if "data: " in ln]

    def _emitter(
        self,
        allowed = ("lookup",),
        **kwargs,
    ):
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter()
        emitter.enable_healing(set(allowed), **kwargs)
        return emitter

    def _chunk(
        self,
        content = None,
        tool_calls = None,
        finish_reason = None,
    ):
        delta = {}
        if content is not None:
            delta["content"] = content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        return {"choices": [{"delta": delta, "finish_reason": finish_reason}]}

    def test_xml_becomes_tool_use_block_and_stop_reason(self):
        events = self._events(
            self._emitter(),
            [
                self._chunk(content = LOOKUP_XML),
                self._chunk(finish_reason = "stop"),
            ],
        )
        starts = [e for e in events if e.get("type") == "content_block_start"]
        (tool_start,) = [e for e in starts if e["content_block"]["type"] == "tool_use"]
        assert tool_start["content_block"]["name"] == "lookup"
        assert tool_start["content_block"]["id"].startswith("toolu_")
        (args,) = [
            e["delta"]["partial_json"]
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
        ]
        assert json.loads(args) == {"q": "x"}
        (message_delta,) = [e for e in events if e.get("type") == "message_delta"]
        assert message_delta["delta"]["stop_reason"] == "tool_use"

    def test_mid_block_signal_closes_text_block_first(self):
        events = self._events(
            self._emitter(),
            [
                self._chunk(content = f"Let me check {LOOKUP_XML}"),
                self._chunk(finish_reason = "stop"),
            ],
        )
        kinds = [
            (e["type"], (e.get("content_block") or e.get("delta") or {}).get("type"))
            for e in events
            if e["type"].startswith("content_block")
        ]
        # text opens, streams the safe prefix, closes; then the tool_use block.
        assert kinds[0] == ("content_block_start", "text")
        assert kinds[1] == ("content_block_delta", "text_delta")
        assert kinds[2] == ("content_block_stop", None)
        assert kinds[3] == ("content_block_start", "tool_use")
        texts = [
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert "".join(texts) == "Let me check "

    def test_false_alarm_streams_as_text(self):
        events = self._events(
            self._emitter(),
            [self._chunk(content = "use the <div> tag"), self._chunk(finish_reason = "stop")],
        )
        texts = [
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert "".join(texts) == "use the <div> tag"
        (message_delta,) = [e for e in events if e.get("type") == "message_delta"]
        assert message_delta["delta"]["stop_reason"] == "end_turn"

    def test_signal_split_across_chunks(self):
        events = self._events(
            self._emitter(),
            [
                self._chunk(content = "<tool"),
                self._chunk(content = '_call>{"name":"lookup","arguments":{}}'),
                self._chunk(finish_reason = "stop"),
            ],
        )
        starts = [e for e in events if e.get("type") == "content_block_start"]
        assert [e["content_block"]["type"] for e in starts] == ["tool_use"]
        texts = [
            e
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert texts == []

    def test_max_tokens_wins_over_healed_stop_reason(self):
        events = self._events(
            self._emitter(),
            [self._chunk(content = LOOKUP_XML), self._chunk(finish_reason = "length")],
        )
        (message_delta,) = [e for e in events if e.get("type") == "message_delta"]
        assert message_delta["delta"]["stop_reason"] == "max_tokens"

    def test_structured_deltas_disable_healing_and_flush(self):
        structured = [
            {
                "index": 0,
                "id": "call_up",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
        events = self._events(
            self._emitter(),
            [
                self._chunk(content = "held <tool"),
                self._chunk(tool_calls = structured),
                self._chunk(finish_reason = "tool_calls"),
            ],
        )
        texts = [
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert "".join(texts) == "held <tool"  # nothing swallowed
        starts = [e for e in events if e.get("type") == "content_block_start"]
        assert [e["content_block"]["type"] for e in starts] == ["text", "tool_use"]

    def test_disable_parallel_caps_healed_calls(self):
        two = LOOKUP_XML + '<tool_call>{"name":"lookup","arguments":{"q":"y"}}</tool_call>'
        events = self._events(
            self._emitter(disable_parallel_tool_use = True),
            [self._chunk(content = two), self._chunk(finish_reason = "stop")],
        )
        starts = [
            e
            for e in events
            if e.get("type") == "content_block_start" and e["content_block"]["type"] == "tool_use"
        ]
        assert len(starts) == 1

    def test_disable_parallel_drops_native_after_healed(self):
        # A healed call consumed the single allowed slot; a later native
        # structured call (index 0, so it survives the caller's chunk-level
        # cap) must not open a second tool_use block.
        structured = [
            {
                "index": 0,
                "id": "call_up",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
        events = self._events(
            self._emitter(disable_parallel_tool_use = True),
            [
                self._chunk(content = LOOKUP_XML),
                self._chunk(tool_calls = structured),
                self._chunk(finish_reason = "tool_calls"),
            ],
        )
        starts = [
            e
            for e in events
            if e.get("type") == "content_block_start" and e["content_block"]["type"] == "tool_use"
        ]
        assert len(starts) == 1

    def test_no_healing_means_verbatim_text(self):
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter()  # enable_healing never called
        events = self._events(
            emitter,
            [self._chunk(content = LOOKUP_XML), self._chunk(finish_reason = "stop")],
        )
        texts = [
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert "".join(texts) == LOOKUP_XML


class TestAnthropicNonStreamingRoute:
    async def _drive(
        self,
        monkeypatch,
        bodies,
        auto_heal = None,
        tools = None,
        tool_choice = "auto",
    ):
        import routes.inference as inf_mod
        from routes.inference import _anthropic_passthrough_non_streaming

        client = ScriptedClient(bodies)
        monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: client)
        response = await _anthropic_passthrough_non_streaming(
            _llama_backend(),
            [{"role": "user", "content": "hi"}],
            tools if tools is not None else [LOOKUP_TOOL],
            0.7,
            0.95,
            None,
            256,
            "msg_test",
            "gguf",
            tool_choice = tool_choice,
            auto_heal_tool_calls = auto_heal,
        )
        return client, json.loads(response.body)

    def test_promotes_xml_to_tool_use(self, monkeypatch):
        async def _run():
            _, data = await self._drive(monkeypatch, [_upstream_message(LOOKUP_XML)])
            (block,) = [b for b in data["content"] if b["type"] == "tool_use"]
            assert block["name"] == "lookup"
            assert block["input"] == {"q": "x"}
            assert data["stop_reason"] == "tool_use"
            assert not any(b["type"] == "text" for b in data["content"])

        asyncio.run(_run())

    def test_opt_out_keeps_legacy_strip(self, monkeypatch):
        async def _run():
            _, data = await self._drive(
                monkeypatch, [_upstream_message(f"plan {LOOKUP_XML}")], auto_heal = False
            )
            assert data["stop_reason"] == "end_turn"
            (block,) = data["content"]
            assert block["type"] == "text"
            assert block["text"] == "plan"  # XML stripped, nothing promoted

        asyncio.run(_run())

    def test_undeclared_tool_not_promoted(self, monkeypatch):
        async def _run():
            xml = '<tool_call>{"name":"rogue","arguments":{}}</tool_call>'
            _, data = await self._drive(monkeypatch, [_upstream_message(xml)])
            assert data["stop_reason"] == "end_turn"
            assert not any(b["type"] == "tool_use" for b in data["content"])
            # Healing preserves what it does not promote: the undeclared call
            # reaches the client as text instead of being silently stripped.
            (text_block,) = [b for b in data["content"] if b["type"] == "text"]
            assert text_block["text"] == xml

        asyncio.run(_run())

    def test_mixed_undeclared_text_preserved_after_heal(self, monkeypatch):
        async def _run():
            # Declared call promoted to tool_use; the undeclared call's markup
            # stays in the text block (the legacy strip must not run after a
            # span-exact heal), matching the OpenAI passthrough.
            rogue = '<tool_call>{"name":"rogue","arguments":{}}</tool_call>'
            _, data = await self._drive(monkeypatch, [_upstream_message(f"{LOOKUP_XML} {rogue}")])
            (tool_block,) = [b for b in data["content"] if b["type"] == "tool_use"]
            assert tool_block["name"] == "lookup"
            (text_block,) = [b for b in data["content"] if b["type"] == "text"]
            assert rogue in text_block["text"]
            assert data["stop_reason"] == "tool_use"

        asyncio.run(_run())

    def test_length_beats_tool_use(self, monkeypatch):
        async def _run():
            _, data = await self._drive(
                monkeypatch, [_upstream_message(LOOKUP_XML, finish_reason = "length")]
            )
            assert data["stop_reason"] == "max_tokens"
            assert any(b["type"] == "tool_use" for b in data["content"])

        asyncio.run(_run())

    def test_tool_choice_none_keeps_legacy_strip(self, monkeypatch):
        async def _run():
            # Anthropic {"type": "none"} arrives here converted to "none":
            # the request forbade tool calls, so nothing is promoted and the
            # legacy XML strip applies as before healing existed.
            _, data = await self._drive(
                monkeypatch,
                [_upstream_message(f"plan {LOOKUP_XML}")],
                tool_choice = "none",
            )
            assert data["stop_reason"] == "end_turn"
            (block,) = data["content"]
            assert block["type"] == "text"
            assert block["text"] == "plan"

        asyncio.run(_run())


class TestOpenaiStreamingRoute:
    def test_heals_streamed_xml(self, monkeypatch):
        async def _run():
            pieces = ["<tool_call>", '{"name":"lookup",', '"arguments":{"q":"x"}}', "</tool_call>"]
            lines = [
                'data: {"id":"c1","model":"gguf","created":1,"choices":[{"index":0,"delta":{"content":%s}}]}'
                % json.dumps(p)
                for p in pieces
            ]
            lines += [
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(), lines)
            payloads = _stream_payloads(chunks)
            tool_deltas = [
                tc
                for p in payloads
                for c in p.get("choices", [])
                for tc in (c.get("delta") or {}).get("tool_calls") or []
            ]
            (call,) = tool_deltas
            assert call["function"]["name"] == "lookup"
            assert json.loads(call["function"]["arguments"]) == {"q": "x"}
            finishes = [
                c["finish_reason"]
                for p in payloads
                for c in p.get("choices", [])
                if c.get("finish_reason")
            ]
            assert finishes == ["tool_calls"]
            # None of the XML leaked as visible content.
            text = "".join(
                (c.get("delta") or {}).get("content") or ""
                for p in payloads
                for c in p.get("choices", [])
            )
            assert "<tool_call>" not in text
            assert chunks[-1] == "data: [DONE]\n\n"

        asyncio.run(_run())

    def test_parallel_cap_drops_native_after_healed(self, monkeypatch):
        async def _run():
            # parallel_tool_calls=false: a healed call consumed the single
            # allowed slot, and the upstream SSE cap keeps native index 0, so
            # the route must drop the later native call itself.
            xml = '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'
            native = (
                'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_up","type":"function","function":'
                '{"name":"lookup","arguments":"{}"}}]}}]}'
            )
            lines = [
                'data: {"id":"c1","model":"gguf","created":1,"choices":'
                '[{"index":0,"delta":{"content":%s}}]}' % json.dumps(xml),
                native,
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(parallel_tool_calls = False), lines)
            payloads = _stream_payloads(chunks)
            tool_deltas = [
                tc
                for p in payloads
                for c in p.get("choices", [])
                for tc in (c.get("delta") or {}).get("tool_calls") or []
            ]
            (call,) = tool_deltas
            assert call["id"] == "call_0"  # the healed call; native was dropped

        asyncio.run(_run())

    def test_false_alarm_text_flushes(self, monkeypatch):
        async def _run():
            lines = [
                'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"use the <div> tag"}}]}',
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(), lines)
            payloads = _stream_payloads(chunks)
            text = "".join(
                (c.get("delta") or {}).get("content") or ""
                for p in payloads
                for c in p.get("choices", [])
            )
            assert text == "use the <div> tag"
            finishes = [
                c["finish_reason"]
                for p in payloads
                for c in p.get("choices", [])
                if c.get("finish_reason")
            ]
            assert finishes == ["stop"]

        asyncio.run(_run())

    def test_incomplete_xml_healed_at_done(self, monkeypatch):
        async def _run():
            # No close tag and no finish chunk: healed at the [DONE] boundary,
            # synthetic finish must say tool_calls.
            lines = [
                'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"<tool_call>{\\"name\\":\\"lookup\\",\\"arguments\\":{}}"}}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(), lines)
            payloads = _stream_payloads(chunks)
            tool_deltas = [
                tc
                for p in payloads
                for c in p.get("choices", [])
                for tc in (c.get("delta") or {}).get("tool_calls") or []
            ]
            assert len(tool_deltas) == 1
            finishes = [
                c["finish_reason"]
                for p in payloads
                for c in p.get("choices", [])
                if c.get("finish_reason")
            ]
            assert finishes == ["tool_calls"]

        asyncio.run(_run())

    def test_structured_upstream_calls_relay_verbatim(self, monkeypatch):
        async def _run():
            line = (
                'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_up","type":"function","function":'
                '{"name":"lookup","arguments":"{}"}}]}}]}'
            )
            lines = [
                line,
                'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
            chunks = await _drive_stream(monkeypatch, _payload(), lines)
            assert chunks[0] == line + "\n\n"  # byte-for-byte relay

        asyncio.run(_run())


class TestHealerSignalAlignment:
    """The passthrough healer buffers only formats its parser can promote.
    The loops' bare [ARGS] rehearsal signal is gated on active tool names
    there; ungated in the healer it would stall legitimate prose until
    finalization without ever producing a promotable call."""

    def test_heal_signals_are_promotable_formats_only(self):
        from core.inference.passthrough_healing import _HEAL_SIGNALS
        assert set(_HEAL_SIGNALS) == {
            "<tool_call>",
            "<|tool_call>",
            "<function=",
            "[TOOL_CALLS]",
            "<|content_invoke_tool_json|>",
        }

    def test_prose_with_bare_args_marker_streams_through(self):
        healer = StreamToolCallHealer({"Bash"})
        chunks = [
            "Use the pattern foo",
            "[ARGS] in templates when calling tools, ",
            "and remember to close it.",
        ]
        streamed = ""
        for chunk in chunks:
            streamed += _events_text(healer.feed(chunk))
        # Incremental relay: nothing withheld for finalize.
        assert streamed == "".join(chunks)
        final = healer.finalize()
        assert not _events_calls(final)
        assert not healer.healed

    def test_bracket_tool_calls_still_promote_in_stream(self):
        healer = StreamToolCallHealer({"web_search"})
        events = healer.feed('[TOOL_CALLS]web_search{"query": "unsloth docs"}') + healer.finalize()
        (call,) = _events_calls(events)
        assert call["function"]["name"] == "web_search"
        assert healer.healed
