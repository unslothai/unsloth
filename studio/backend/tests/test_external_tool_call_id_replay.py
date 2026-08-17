# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replayed tool-call ids must fit provider limits (#8913).

The frontend stores tool-call ids as "<provider id>:<uuid4>" (66 chars for
OpenAI), and providers that validate ids reject the whole request on the next
turn, permanently breaking the chat.
"""

import re

from models.inference import ChatMessage
from routes.inference import _build_external_messages

MINTED = "call_AbCdEfGhIjKlMnOpQrStUvWx:071e73c8-5d38-4d4c-821a-62fe32c7a54a"
ORIGINAL = "call_AbCdEfGhIjKlMnOpQrStUvWx"


def _history(tool_call_id):
    return [
        ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            }
        ),
        ChatMessage.model_validate(
            {"role": "tool", "tool_call_id": tool_call_id, "content": "ok"}
        ),
    ]


def _replayed_ids(tool_call_id, provider_type = "openai"):
    out = _build_external_messages(
        _history(tool_call_id), supports_vision = True, provider_type = provider_type
    )
    assistant = next(m for m in out if m.get("tool_calls"))
    tool = next(m for m in out if m["role"] == "tool")
    return assistant["tool_calls"][0]["id"], tool["tool_call_id"]


def test_minted_frontend_id_is_restored_to_provider_id():
    call_id, output_id = _replayed_ids(MINTED)
    assert call_id == ORIGINAL
    assert output_id == ORIGINAL


def test_oversized_foreign_id_is_shortened_symmetrically():
    long_id = "x" * 80
    call_id, output_id = _replayed_ids(long_id)
    assert call_id == output_id
    assert len(call_id) == 64
    assert call_id.startswith("x" * 31)


def test_short_ids_pass_through_unchanged():
    call_id, output_id = _replayed_ids("call_xyz")
    assert call_id == "call_xyz"
    assert output_id == "call_xyz"


def test_replay_applies_on_generic_chat_completions_providers():
    call_id, output_id = _replayed_ids(MINTED, provider_type = "deepseek")
    assert call_id == ORIGINAL
    assert output_id == ORIGINAL


def test_mistral_maps_foreign_ids_to_nine_alnum_chars():
    for foreign in (MINTED, "tool_call_0", "toolu_01A09q90qw90lq917835lq9", "x" * 80):
        call_id, output_id = _replayed_ids(foreign, provider_type = "mistral")
        assert call_id == output_id
        assert re.fullmatch(r"[a-zA-Z0-9]{9}", call_id)


def test_colliding_bases_keep_the_full_stored_ids():
    a = "call_0:071e73c8-5d38-4d4c-821a-62fe32c7a54a"
    b = "call_0:11111111-2222-4333-8444-555555555555"
    out = _build_external_messages(
        _history(a) + _history(b), supports_vision = True, provider_type = "openai"
    )
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    output_ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert call_ids == output_ids == [a, b]


def test_mistral_colliding_bases_stay_distinct():
    a = "call_0:071e73c8-5d38-4d4c-821a-62fe32c7a54a"
    b = "call_0:11111111-2222-4333-8444-555555555555"
    out = _build_external_messages(
        _history(a) + _history(b), supports_vision = True, provider_type = "mistral"
    )
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    output_ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert call_ids == output_ids
    assert call_ids[0] != call_ids[1]
    assert all(re.fullmatch(r"[a-zA-Z0-9]{9}", cid) for cid in call_ids)


def test_mistral_native_ids_pass_through_unchanged():
    call_id, output_id = _replayed_ids(
        "AbCdEfGhI:071e73c8-5d38-4d4c-821a-62fe32c7a54a", provider_type = "mistral"
    )
    assert call_id == "AbCdEfGhI"
    assert output_id == "AbCdEfGhI"
