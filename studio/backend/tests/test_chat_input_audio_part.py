# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI's documented `input_audio` content part must reach the audio path.

`ContentPart` was a closed tagged union, so that shape 400'd with `union_tag_invalid` before any
model ran -- though llama-server takes the part and `_inject_audio_part` builds one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import ChatCompletionRequest, InputAudioContentPart, UnknownContentPart
import routes.inference as inference_route
from routes.inference import (
    _normalise_chat_content_parts,
    _reject_unsupported_content_parts,
)


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


def test_two_recordings_are_refused_rather_than_reduced_to_one():
    """``audio_base64`` holds one recording, so a second was silently discarded.

    A request to compare two clips would have been answered from the last one alone.
    """
    payload = _request(_audio_message(data = "Zmlyc3Q="), _audio_message(data = "c2Vjb25k"))

    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_content_parts(payload)
    assert exc.value.status_code == 400
    assert "one audio recording" in str(exc.value.detail)


def test_an_audio_part_on_a_non_user_role_is_refused():
    """Only a user turn carries a recording into the model, and the lift strips every role.

    Dropping it in silence let a later question about an assistant-history clip be answered from
    text alone, where this shape used to fail validation outright.
    """
    payload = _request(_audio_message(role = "assistant"))

    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_content_parts(payload)
    assert exc.value.status_code == 400
    assert "'assistant'" in str(exc.value.detail)


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
        _reject_unsupported_content_parts(payload)
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

    _reject_unsupported_content_parts(payload)
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


# ── The four review findings on the first two commits ──────────────────────────


def test_a_non_string_part_type_is_a_validation_error_not_a_500():
    """A list or dict ``type`` is unhashable against the known-tag set.

    Testing membership on it raised TypeError out of the discriminator, which escaped request
    validation as a 500 where the closed union had answered 422.
    """
    with _route_client("/v1") as client:
        response = client.post(
            "/v1/chat/completions",
            json = {
                "model": "local",
                "messages": [{"role": "user", "content": [{"type": [{"a": 1}], "x": 1}]}],
            },
        )

    assert response.status_code == 422


def test_the_external_path_refuses_audio_rather_than_dropping_it():
    """_build_external_messages has no input_audio case, so the part would be stripped.

    The provider would then answer the text alone -- a plausible reply about a recording it
    never received. Refused the way video is refused on the same branch.
    """
    with _route_client("/v1") as client:
        response = client.post(
            "/v1/chat/completions",
            json = {"model": "gpt-4o", "provider_type": "openai", "messages": [_audio_message()]},
        )

    assert response.status_code == 400
    assert "Audio input is only supported" in str(response.json()["detail"])


def test_the_external_path_refuses_an_unmodelled_part_rather_than_dropping_it():
    with _route_client("/v1") as client:
        response = client.post(
            "/v1/chat/completions",
            json = {
                "model": "gpt-4o",
                "provider_type": "openai",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "summarise this"},
                            {"type": "file", "file": {"file_id": "file_abc"}},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert "'file'" in response.json()["detail"]["error"]["message"]


def test_a_recording_carried_on_an_earlier_turn_is_refused():
    """``audio_base64`` cannot express which turn a recording came from.

    _inject_audio_part appends it to the last user message, so lifting an earlier turn's audio
    replays it against a later question -- the model is asked about something the caller did not
    ask. Refuse instead, until the field can carry a recording with its turn.
    """
    payload = _request(
        _audio_message(text = "transcribe this"),
        {"role": "assistant", "content": "It says hello."},
        {"role": "user", "content": [{"type": "text", "text": "who is in the background?"}]},
    )

    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_content_parts(payload)
    assert exc.value.status_code == 400
    assert "latest user message" in str(exc.value.detail)


def test_a_recording_on_the_latest_user_turn_is_still_lifted():
    """The shape the SDK documents, and the one the refusal above must not catch."""
    payload = _request(
        {"role": "user", "content": [{"type": "text", "text": "transcribe this"}]},
        {"role": "assistant", "content": "Sure."},
        _audio_message(text = "what about this one?"),
    )

    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 == AUDIO_B64
    assert [p.type for p in payload.messages[2].content] == ["text"]


def test_an_empty_audio_payload_is_refused_rather_than_dropped():
    """``{"data": ""}`` is falsy, so it was never lifted but the part was still removed.

    "transcribe this" then ran as a text-only prompt and answered about a recording nobody sent.
    """
    with pytest.raises(ValidationError):
        _request(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
                ],
            }
        )


def test_the_tts_route_refuses_an_unmodelled_part():
    """/audio/generate shares this request model but reads only text parts.

    Before the catch-all it 422'd on an unknown tag; without this it would voice the text alone.
    """
    with _route_client() as client:
        response = client.post(
            "/audio/generate",
            json = {
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "read this out"},
                            {"type": "file", "file": {"file_id": "file_abc"}},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert "'file'" in response.json()["detail"]["error"]["message"]


def test_the_tts_route_refuses_an_audio_part():
    """/audio/generate voices the message text; a recording has nowhere to go there.

    _extract_content_parts keeps only text, so the part was discarded and speech was returned
    for an incomplete request that used to fail validation outright.
    """
    with _route_client() as client:
        response = client.post(
            "/audio/generate",
            json = {"model": "default", "messages": [_audio_message(text = "read this out")]},
        )

    assert response.status_code == 400
    assert "Audio input is not supported here" in response.json()["detail"]["error"]["message"]


def _durable_run(content):
    from routes.chat_generation_runs import CreateChatGenerationRun
    return CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {"model": "default", "messages": [{"role": "user", "content": content}]},
    )


def test_a_durable_run_refuses_a_nested_recording():
    """The durable sanitizer refuses media because the payload persists verbatim.

    It read only the top-level fields, so a recording carried in a content part would have lived
    in ``request_json`` for the life of the thread.
    """
    from routes.chat_generation_runs import _sanitize_request

    with pytest.raises(HTTPException) as exc:
        _sanitize_request(
            _durable_run(
                [
                    {"type": "text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": AUDIO_B64, "format": "wav"}},
                ]
            )
        )
    assert exc.value.status_code == 400
    assert "Media chat runs" in str(exc.value.detail)


def test_a_durable_run_refuses_an_unmodelled_part_immediately():
    """Otherwise the create endpoint returns 202 and the supervisor fails it out of band."""
    from routes.chat_generation_runs import _sanitize_request

    with pytest.raises(HTTPException) as exc:
        _sanitize_request(
            _durable_run(
                [
                    {"type": "text", "text": "summarise this"},
                    {"type": "file", "file": {"file_id": "file_abc"}},
                ]
            )
        )
    assert exc.value.status_code == 400
    assert "'file'" in str(exc.value.detail)


def test_a_plain_durable_run_is_still_queued():
    from routes.chat_generation_runs import _sanitize_request

    sanitized = _sanitize_request(_durable_run([{"type": "text", "text": "hello"}]))

    assert sanitized["stream"] is True
    assert sanitized["thread_id"] == "thread-1"


def test_the_tts_route_refuses_a_recording_that_was_already_lifted():
    """/chat/completions routes a loaded TTS model into generate_audio after normalisation.

    By then the part is gone and only ``audio_base64`` is set, so a parts-only guard would let
    the route speak the text and drop the recording.
    """
    payload = _request(_audio_message(text = "read this out"))
    _normalise_chat_content_parts(payload)
    assert payload.audio_base64 == AUDIO_B64

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.generate_audio(payload, None))
    assert exc.value.status_code == 400
    assert "not supported here" in str(exc.value.detail)


def test_the_preview_route_refuses_before_it_loads_a_checkpoint():
    """_serve_chat holds the preview lock and loads the checkpoint before delegating.

    The delegate refuses the part, but by then an invalid request has evicted the resident model.
    """
    import routes.preview as preview_route

    payload = _request(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "file", "file": {"file_id": "file_abc"}},
            ],
        }
    )
    loads: list[int] = []
    with mock.patch.object(preview_route, "_resolve_or_4xx", lambda run, cp: Path("/tmp")):
        with mock.patch.object(
            preview_route, "load_model_for_preview", lambda *a, **k: loads.append(1)
        ):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(preview_route._serve_chat("run-1", None, payload, None))

    assert exc.value.status_code == 400
    assert "'file'" in str(exc.value.detail)
    assert loads == []


def test_the_preview_route_refuses_misplaced_audio_before_it_loads():
    """The placement checks used to live behind routing, so preview reached them after the load.

    A request that was always going to 400 would have taken the preview lock and swapped the
    resident checkpoint on its way there.
    """
    import routes.preview as preview_route

    payload = _request(
        _audio_message(text = "transcribe this"),
        {"role": "assistant", "content": "It says hello."},
        {"role": "user", "content": [{"type": "text", "text": "who is in the background?"}]},
    )
    loads: list[int] = []
    with mock.patch.object(preview_route, "_resolve_or_4xx", lambda run, cp: Path("/tmp")):
        with mock.patch.object(
            preview_route, "load_model_for_preview", lambda *a, **k: loads.append(1)
        ):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(preview_route._serve_chat("run-1", None, payload, None))

    assert exc.value.status_code == 400
    assert "latest user message" in str(exc.value.detail)
    assert loads == []


def test_the_text_only_checkpoint_refusal_precedes_the_branch_that_consumes_audio():
    """A source-order guard, not an end-to-end one: reaching that branch needs the ML stack.

    The transformers path consumes audio only when the checkpoint declares audio input, and the
    capability check that would otherwise catch a text-only one runs only when an automatic load
    could fix it. With auto-switch off the branch is skipped and the turn is answered from its
    text alone, so the refusal has to sit in front of it. This pins that ordering; whether the
    refusal fires for a real checkpoint is covered by the GGUF/transformers suites, not here.
    """
    source = Path(inference_route.__file__).read_text(encoding = "utf-8")
    branch = source.index('if payload.audio_base64 and not model_info.get("has_audio_input"):')
    consume = source.index('if payload.audio_base64 and model_info.get("has_audio_input"):')

    # the refusal has to precede the consuming branch, or it never runs
    assert branch < consume
    assert "cannot read audio input" in source[branch:consume]
