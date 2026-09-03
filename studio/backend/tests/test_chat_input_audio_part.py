# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI's documented `input_audio` content part must reach the audio path.

`ContentPart` was a closed tagged union, so that shape 400'd with `union_tag_invalid` before any
model ran -- though llama-server takes the part and `_inject_audio_part` builds one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import ChatCompletionRequest, InputAudioContentPart, UnknownContentPart
from routes.inference import _normalise_chat_content_parts


AUDIO_B64 = "UklGRiQAAABXQVZF"


def _request(*messages, **fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(model = "local", messages = list(messages), **fields)


def _audio_message(data = AUDIO_B64, role = "user", text = "what is said here?"):
    return {
        "role": role,
        "content": [
            {"type": "text", "text": text},
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
        ],
    }


def test_input_audio_part_validates():
    payload = _request(_audio_message())
    assert isinstance(payload.messages[0].content[1], InputAudioContentPart)
    assert payload.messages[0].content[1].input_audio.data == AUDIO_B64


def test_input_audio_part_is_lifted_onto_the_audio_field():
    payload = _request(_audio_message())

    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 == AUDIO_B64
    assert [p.type for p in payload.messages[0].content] == ["text"]


def test_an_explicit_audio_base64_wins():
    payload = _request(_audio_message(data = "b2xkZXI="), audio_base64 = AUDIO_B64)

    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 == AUDIO_B64


def test_the_newest_user_part_wins():
    payload = _request(_audio_message(data = "Zmlyc3Q="), _audio_message(data = "c2Vjb25k"))

    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 == "c2Vjb25k"


def test_an_assistant_audio_part_is_not_treated_as_an_attachment():
    payload = _request(_audio_message(role = "assistant"))

    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 is None


def test_an_unmodelled_part_type_names_itself_in_a_typed_400():
    payload = _request(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarise this"},
                {"type": "file", "file": {"file_id": "file_abc"}},
            ],
        }
    )
    assert isinstance(payload.messages[0].content[1], UnknownContentPart)

    with pytest.raises(HTTPException) as exc:
        _normalise_chat_content_parts(payload)
    assert exc.value.status_code == 400
    assert "'file'" in str(exc.value.detail)


def test_a_part_with_no_type_is_still_a_validation_error():
    with pytest.raises(ValidationError):
        _request({"role": "user", "content": [{"text": "hi"}]})
