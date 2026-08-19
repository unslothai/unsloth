# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for unslothai/unsloth#9039: malformed tool-call arguments must not
be replayed to custom OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.tool_loop_controller import (  # noqa: E402
    coerce_messages_tool_calls_for_wire,
    coerce_tool_call_replay_arguments,
)
from routes.inference import _build_external_messages  # noqa: E402
from models.inference import ChatMessage  # noqa: E402


class TestCoerceToolCallReplayArguments:
    def test_keeps_parsable_streamed_text(self):
        assert coerce_tool_call_replay_arguments('{"query":"first"}', {"query": "first"}) == (
            '{"query":"first"}'
        )

    def test_falls_back_to_structured_args_for_concatenated_fragments(self):
        raw = '{"query":"first"}{"query":"second"}'
        assert coerce_tool_call_replay_arguments(
            raw,
            {"_raw": raw},
        ) == json.dumps({"_raw": raw}, separators = (",", ":"))

    def test_stringifies_dict_arguments(self):
        assert coerce_tool_call_replay_arguments({"query": "x"}) == '{"query":"x"}'

    def test_empty_structured_args_becomes_empty_object(self):
        assert coerce_tool_call_replay_arguments("", {"query": "first"}) == '{"query":"first"}'
        assert coerce_tool_call_replay_arguments(None, None) == "{}"


class TestCoerceMessagesToolCallsForWire:
    def test_repairs_malformed_assistant_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"first"}{"query":"second"}',
                        },
                        "arguments": {"_raw": '{"query":"first"}{"query":"second"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "content": "ok",
            },
        ]
        out = coerce_messages_tool_calls_for_wire(messages)
        wire = out[1]["tool_calls"][0]["function"]["arguments"]
        json.loads(wire)
        assert "arguments" not in out[1]["tool_calls"][0]

    def test_stringifies_dict_function_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "x"}},
                    }
                ],
            }
        ]
        out = coerce_messages_tool_calls_for_wire(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"q":"x"}'


class TestBuildExternalMessagesWireShape:
    def test_custom_endpoint_messages_have_string_arguments(self):
        built = _build_external_messages(
            [
                ChatMessage(
                    role = "assistant",
                    content = None,
                    tool_calls = [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"first"}{"query":"second"}',
                            },
                        }
                    ],
                ),
                ChatMessage(role = "tool", tool_call_id = "call_0", content = "ok"),
            ],
            supports_vision = False,
            provider_type = "custom",
            base_url = "http://127.0.0.1:8000/v1",
        )
        args = built[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        json.loads(args)
