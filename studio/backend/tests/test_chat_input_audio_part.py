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


def _audio_message(
    data = AUDIO_B64,
    role = "user",
    text = "what is said here?",
):
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


def _route_client(prefix = ""):
    """The real inference router, with only the auth dependency stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    import routes.inference as inference_route

    app = FastAPI()
    app.include_router(inference_route.router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app, raise_server_exceptions = False)


def _count_tokens_client():
    return _route_client()


def test_the_count_route_refuses_an_audio_part_the_way_it_refuses_the_field():
    """/chat/count_tokens already refuses audio, and a part is audio.

    It guards images at the part level but audio only through ``audio_base64``, which was safe
    only while an ``input_audio`` part could not validate at all.
    """
    with _count_tokens_client() as client:
        response = client.post(
            "/chat/count_tokens",
            json = {"model": "default", "messages": [_audio_message()]},
        )

    assert response.status_code == 503
    assert "audio" in response.json()["detail"]


def test_the_count_route_refuses_an_unmodelled_part_like_the_completion_does():
    with _count_tokens_client() as client:
        response = client.post(
            "/chat/count_tokens",
            json = {
                "model": "default",
                "messages": [
                    {"role": "user", "content": [{"type": "file", "file": {"file_id": "file_abc"}}]}
                ],
            },
        )

    assert response.status_code == 400
    assert "'file'" in response.json()["detail"]["error"]["message"]


def test_a_string_content_message_passes_through_the_lift_untouched():
    """Only list content carries parts; a plain-string turn must not be rewritten."""
    payload = _request({"role": "system", "content": "be terse"}, _audio_message())

    _normalise_chat_content_parts(payload)

    assert payload.messages[0].content == "be terse"
    assert payload.audio_base64 == AUDIO_B64


def test_the_completion_route_takes_the_documented_audio_part():
    """The defect itself: the part used to be refused at body validation, before any model ran.

    What happens after validation depends on which models the host has, so this pins the only
    part that is about the union: the request is no longer rejected as an unknown tag.
    """
    with _route_client("/v1") as client:
        response = client.post(
            "/v1/chat/completions",
            json = {"model": "local", "messages": [_audio_message()]},
        )

    assert response.status_code != 422
    assert "union_tag_invalid" not in response.text


def test_the_completion_route_refuses_an_unmodelled_part():
    """Raised at the normalisation call site, so it lands before any model resolution."""
    with _route_client("/v1") as client:
        response = client.post(
            "/v1/chat/completions",
            json = {
                "model": "local",
                "messages": [
                    {"role": "user", "content": [{"type": "file", "file": {"file_id": "file_abc"}}]}
                ],
            },
        )

    assert response.status_code == 400
    assert "'file'" in response.json()["detail"]["error"]["message"]
