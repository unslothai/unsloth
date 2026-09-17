# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replayed tool-call ids must fit provider limits (#8913).

The frontend stores them as "<provider id>:<uuid4>" (66 chars for OpenAI), and a
provider that validates ids rejects the whole request, permanently breaking the chat.
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
        ChatMessage.model_validate({"role": "tool", "tool_call_id": tool_call_id, "content": "ok"}),
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


# Anthropic states its charset in the 400 it raises: "tool_use.id: String should match
# pattern '^[a-zA-Z0-9_-]+$'". A colon is not in it, and the two stored shapes carrying
# one (the duplicate-base fallback, and "<sandbox>:<thread>:<approval>" confirmation ids)
# are both under 64 chars, so the length branch never touched them.
ANTHROPIC_ID = re.compile(r"[a-zA-Z0-9_-]+")


def test_anthropic_rejects_nothing_it_would_have_rejected():
    call_id, output_id = _replayed_ids("sandboxsess:threadid:approvalid", provider_type = "anthropic")
    assert call_id == output_id
    assert ANTHROPIC_ID.fullmatch(call_id), call_id


def test_anthropic_colliding_bases_stay_legal_and_distinct():
    a = "call_0:071e73c8-5d38-4d4c-821a-62fe32c7a54a"
    b = "call_0:11111111-2222-4333-8444-555555555555"
    out = _build_external_messages(
        _history(a) + _history(b), supports_vision = True, provider_type = "anthropic"
    )
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    output_ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert call_ids == output_ids
    assert call_ids[0] != call_ids[1]
    assert all(ANTHROPIC_ID.fullmatch(cid) for cid in call_ids), call_ids


def test_anthropic_legal_ids_pass_through_unchanged():
    # Only ids Anthropic would already have refused may change, so a chat that works
    # today keeps byte-identical ids.
    for legal in ("toolu_01A1B2C3D4E5F6G7H8I9J0K1", "call_abc123", "a-b_c"):
        call_id, output_id = _replayed_ids(legal, provider_type = "anthropic")
        assert call_id == output_id == legal


def test_anthropic_sanitizing_alone_would_collide():
    # "pre:fix" and "pre_fix" both sanitize to "pre_fix", a silent mispairing, so the
    # sha256 tail over the unsanitized value is what keeps the map injective.
    out = _build_external_messages(
        _history("pre:fix") + _history("pre_fix"),
        supports_vision = True,
        provider_type = "anthropic",
    )
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    assert len(set(call_ids)) == 2, call_ids


def test_replay_is_idempotent_for_every_provider():
    # A normalized id replayed again on turn three must not drift, or the call and its
    # result stop matching.
    for provider in ("openai", "anthropic", "mistral", "gemini", "deepseek", None):
        once, _ = _replayed_ids(MINTED, provider_type = provider)
        twice, paired = _replayed_ids(once, provider_type = provider)
        assert twice == once == paired, (provider, once, twice)
