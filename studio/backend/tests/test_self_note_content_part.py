# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The `self_note` content part: how the model's note round-trips through the client.

The part is how the note stays stateless -- it rides on the assistant message and the
client re-sends it, so nothing is stored server-side.
"""

from __future__ import annotations

from models.inference import ChatMessage


def test_self_note_part_validates_on_an_assistant_message():
    message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "self_note", "content": "Approach A fails: re-entrant lock."},
                {"type": "text", "text": "Here is the answer."},
            ],
        }
    )
    parts = message.content
    assert parts[0].type == "self_note"
    assert parts[0].content == "Approach A fails: re-entrant lock."


def test_self_note_part_survives_a_round_trip_through_dump():
    payload = {
        "role": "assistant",
        "content": [{"type": "self_note", "content": "note body"}],
    }
    message = ChatMessage.model_validate(payload)
    dumped = message.model_dump(exclude_none = True)
    again = ChatMessage.model_validate(dumped)
    assert again.content[0].content == "note body"


def test_an_unknown_part_type_still_degrades_rather_than_raising():
    # Back-compat: the union falls back to UnknownContentPart, so a client that
    # has never heard of a part type does not 400.
    message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": [{"type": "not_a_real_part", "content": "x"}],
        }
    )
    assert message.content[0].type == "not_a_real_part"
