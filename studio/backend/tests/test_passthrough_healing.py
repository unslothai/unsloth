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

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.passthrough_healing import (  # noqa: E402
    StreamToolCallHealer,
    heal_gate,
    heal_openai_message,
    nudge_messages,
    nudge_should_retry,
)

TOOLS = [
    {"type": "function", "function": {"name": "Bash", "parameters": {}}},
    {"type": "function", "function": {"name": "Read", "parameters": {}}},
]
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

    def test_bad_json_arguments_coerced_under_canonical_key(self):
        msg = {
            "role": "assistant",
            "content": '<tool_call>{"name":"Bash","arguments":"echo hi"}</tool_call>',
        }
        assert heal_openai_message(msg, {"Bash"}) is True
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert args == {"query": "echo hi"}


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
    def _resp(self, content, tool_calls = None):
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
