# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import LoadRequest


def _base_load_request(**overrides):
    data = {
        "model_path": "unsloth/test-model-GGUF",
        "hf_token": None,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "is_lora": False,
        "gguf_variant": "Q4_K_M",
    }
    data.update(overrides)
    return LoadRequest.model_validate(data)


def test_blank_chat_template_override_normalizes_to_none():
    req = _base_load_request(chat_template_override = "   \n\t")

    assert req.chat_template_override is None


def test_nonblank_chat_template_override_is_preserved_verbatim():
    template = "  {{ messages }}  "
    req = _base_load_request(chat_template_override = template)

    assert req.chat_template_override == template


# ---------- ChatCompletionRequest tool_call_id walkback ----------

from models.inference import ChatCompletionRequest


def _req(messages, **overrides):
    payload = {"model": "x", "messages": messages, **overrides}
    return ChatCompletionRequest.model_validate(payload)


def test_tool_message_inherits_id_from_prior_assistant_tool_call():
    req = _req(
        [
            {"role": "user", "content": "what is 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_real123",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},  # no tool_call_id
        ]
    )
    assert req.messages[-1].tool_call_id == "call_real123"


def test_tool_message_with_explicit_id_unchanged():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_user_supplied", "content": "ok"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_user_supplied"


def test_walkback_prefers_function_name_match():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "call_y",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_y"


def test_walkback_takes_first_unconsumed_when_no_name():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "content": "first result"},
            {"role": "tool", "content": "second result"},
        ]
    )
    assert req.messages[-2].tool_call_id == "call_a"
    assert req.messages[-1].tool_call_id == "call_b"


def test_walkback_falls_back_to_synth_when_no_assistant_turn():
    req = _req(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "orphan"},
        ]
    )
    tcid = req.messages[-1].tool_call_id
    assert tcid is not None and tcid.startswith("call_") and len(tcid) > 5


def test_walkback_does_not_cross_user_turn():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "old_call",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old_call", "content": "4"},
            {"role": "user", "content": "next turn"},
            {"role": "tool", "content": "no parent in this turn"},
        ]
    )
    last = req.messages[-1].tool_call_id
    # The walkback must NOT pick old_call because a user turn intervenes;
    # falls back to synth.
    assert last is not None
    assert last != "old_call"
    assert last.startswith("call_")


def test_walkback_skips_explicitly_consumed_tool_call_id():
    """Sibling tool result with an explicit id must reserve its assistant
    slot so a follow-up missing-id result picks the OTHER tool call."""
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "4"},
            {"role": "tool", "content": "second result"},
        ]
    )
    assert [m.tool_call_id for m in req.messages if m.role == "tool"] == [
        "call_a",
        "call_b",
    ]


def test_walkback_handles_malformed_function_string():
    """A tool_call with ``function`` as a string (provider quirk) must not
    raise; resolution falls back to fallback id selection."""
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": "calc"},
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_a"
