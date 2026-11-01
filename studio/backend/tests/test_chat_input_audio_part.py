# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI input_audio parts must pass ContentPart validation rather than fail as union_tag_invalid."""

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


def _multi_audio_message(*datas, text = "compare these"):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            *(
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
                for data in datas
            ),
        ],
    }


def test_several_recordings_on_the_latest_turn_are_all_lifted_in_order():
    """Models like Gemma 4 take several clips at once; none may be dropped or reordered."""
    payload = _request(_multi_audio_message("Zmlyc3Q=", "c2Vjb25k", "dGhpcmQ="))

    _reject_unsupported_content_parts(payload)
    _normalise_chat_content_parts(payload)

    assert payload.audio_base64 == "Zmlyc3Q="
    assert payload.extra_audio_base64 == ["c2Vjb25k", "dGhpcmQ="]
    assert inference_route._request_audio_clips(payload) == ["Zmlyc3Q=", "c2Vjb25k", "dGhpcmQ="]
    assert [p.type for p in payload.messages[0].content] == ["text"]


def test_recordings_split_across_turns_are_refused_rather_than_reduced():
    """Only the latest turn's clips reach the model, so an older clip would be answered without."""
    payload = _request(_audio_message(data = "Zmlyc3Q="), _audio_message(data = "c2Vjb25k"))

    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_content_parts(payload)
    assert exc.value.status_code == 400
    assert "earlier turn" in str(exc.value.detail)


def test_more_recordings_than_the_cap_are_refused():
    datas = ["QUFB"] * (inference_route._MAX_AUDIO_CLIPS_PER_REQUEST + 1)
    payload = _request(_multi_audio_message(*datas))

    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_content_parts(payload)
    assert exc.value.status_code == 400
    assert f"At most {inference_route._MAX_AUDIO_CLIPS_PER_REQUEST}" in str(exc.value.detail)


def test_extra_audio_alone_is_promoted_onto_the_audio_field():
    """Every capability and size check keys on audio_base64, so extras can never bypass them."""
    payload = _request(
        {"role": "user", "content": "hi"}, extra_audio_base64 = ["", "Zmlyc3Q=", "c2Vjb25k"]
    )

    assert payload.audio_base64 == "Zmlyc3Q="
    assert payload.extra_audio_base64 == ["c2Vjb25k"]


def test_the_size_cap_covers_all_clips_together(monkeypatch):
    """N clips never cost more upload or decode memory than one maximal clip does."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_B64_CHARS", 10)
    monkeypatch.setattr(inference_route, "_AUDIO_CLIP_B64_SLACK_CHARS", 2)
    one = _request({"role": "user", "content": "hi"}, audio_base64 = "A" * 10)
    assert inference_route._request_audio_rejection(one) is None
    within_slack = _request(
        {"role": "user", "content": "hi"}, audio_base64 = "A" * 6, extra_audio_base64 = ["A" * 6]
    )
    assert inference_route._request_audio_rejection(within_slack) is None

    two = _request(
        {"role": "user", "content": "hi"}, audio_base64 = "A" * 7, extra_audio_base64 = ["A" * 6]
    )
    status, detail = inference_route._request_audio_rejection(two)
    assert status == 413
    assert "all files together" in detail


def test_the_count_cap_is_checked_on_the_fields_too():
    payload = _request(
        {"role": "user", "content": "hi"},
        audio_base64 = "QUFB",
        extra_audio_base64 = ["QUFB"] * inference_route._MAX_AUDIO_CLIPS_PER_REQUEST,
    )
    status, detail = inference_route._request_audio_rejection(payload)
    assert status == 400
    assert "Too many audio files" in detail


def test_the_duration_cap_covers_all_decoded_clips_together(monkeypatch):
    import numpy as np

    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)
    monkeypatch.setattr(
        inference_route, "_decode_audio_base64", lambda _b64: np.zeros(12_000, np.float32)
    )
    assert len(inference_route._decode_audio_clips(["a"])) == 1
    with pytest.raises(inference_route._DecodedAudioTooLongError):
        inference_route._decode_audio_clips(["a", "b"])


def test_an_audio_part_on_a_non_user_role_is_refused():
    """An audio part on a non-user role is refused, since dropping it silently answers from text alone."""
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
    from auth.authentication import authenticated_via_api_key, get_current_subject
    import routes.inference as inference_route

    app = FastAPI()
    app.include_router(inference_route.router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "test"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app, raise_server_exceptions = False)


def _count_tokens_client():
    return _route_client()


def test_the_count_route_refuses_an_audio_part_the_way_it_refuses_the_field():
    """/chat/count_tokens guards audio only through audio_base64, so an audio part must be refused too."""
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
    """The documented audio part must no longer be refused as an unknown tag at body validation;
    later steps depend on the host's models, so only that is pinned."""
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


def test_a_non_string_part_type_is_a_validation_error_not_a_500():
    """A non-string part type must be a 422 validation error, not a 500 from an unhashable tag lookup."""
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
    """External providers have no input_audio case, so audio is refused rather than silently stripped."""
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
    """audio_base64 cannot name its turn, so a recording on an earlier turn must be refused."""
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
    """An empty audio payload is refused; dropping it silently answers about a recording nobody sent."""
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
    """/audio/generate reads only text, so an unmodelled part must be refused rather than voiced alone."""
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
    """/audio/generate has nowhere to put a recording, so an audio part must be refused, not discarded."""
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
    """The durable sanitizer must scan nested parts, since request_json persists the payload verbatim."""
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
    """Normalisation lifts the part away, so the TTS guard must also check audio_base64, not just parts."""
    payload = _request(_audio_message(text = "read this out"))
    _normalise_chat_content_parts(payload)
    assert payload.audio_base64 == AUDIO_B64

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.generate_audio(payload, None))
    assert exc.value.status_code == 400
    assert "not supported here" in str(exc.value.detail)


def test_the_preview_route_refuses_before_it_loads_a_checkpoint():
    """Preview must refuse audio before loading, since an invalid request would evict the resident model."""
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
    """Audio placement checks must run before preview loads a checkpoint and swaps the resident model."""
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
    """The text-only audio refusal must sit before the audio branch, which auto-switch-off skips."""
    source = Path(inference_route.__file__).read_text(encoding = "utf-8")
    branch = source.index('if payload.audio_base64 and not model_info.get("has_audio_input"):')
    consume = source.index('if payload.audio_base64 and model_info.get("has_audio_input"):')

    assert branch < consume
    assert "cannot read audio input" in source[branch:consume]


def _wav_b64(seconds: float, rate: int = 16000) -> str:
    import base64
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_gguf_clips_pass_through_in_order():
    first, second = _wav_b64(0.5), _wav_b64(0.25)
    prepared = inference_route._prepare_audio_clips_for_llama(
        [first, f"data:audio/wav;base64,{second}"]
    )
    assert prepared == [(first, "wav"), (second, "wav")]


def test_gguf_duration_cap_covers_all_clips_together(monkeypatch):
    """Each clip alone is under the cap; together they are not, as on the decoded path."""
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_SECONDS", 1)
    clip = _wav_b64(0.75)
    assert len(inference_route._prepare_audio_clips_for_llama([clip])) == 1
    with pytest.raises(inference_route._DecodedAudioTooLongError):
        inference_route._prepare_audio_clips_for_llama([clip, clip])


def test_transcoded_gguf_clips_share_one_wav_budget(monkeypatch):
    """Several m4a/ogg clips must not each grow into a cap-sized WAV for llama-server."""
    import base64

    import numpy as np

    caps = []

    def _fit(
        arr,
        sr,
        cap = None,
    ):
        caps.append(cap)
        return arr, sr

    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", 1000)
    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: None)
    monkeypatch.setattr(
        inference_route, "_decode_audio_mono", lambda _raw: (np.zeros(100, np.float32), 16000)
    )
    monkeypatch.setattr(inference_route, "_fit_transcoded_audio_to_wav_cap", _fit)

    inference_route._prepare_audio_for_llama("QUFB")
    assert caps == [1000]

    caps.clear()
    prepared = inference_route._prepare_audio_clips_for_llama(["QUFB", "QUFB", "QUFB"])
    assert caps == []
    assert [len(base64.b64decode(data)) for data, _ in prepared] == [44 + 100 * 2] * 3

    caps.clear()
    monkeypatch.setattr(
        inference_route, "_sniff_audio_container", lambda raw: "wav" if raw == b"WAV!" else None
    )
    monkeypatch.setattr(inference_route, "_passthrough_audio_seconds", lambda *_a: 0.1)
    inference_route._prepare_audio_clips_for_llama(["V0FWIQ==", "QUFB"])
    assert caps == [1000 - 4]


def test_the_wav_budget_does_not_depend_on_clip_order(monkeypatch):
    """The WAV budget must not depend on clip order, or an equal split starves a long clip first."""
    import base64
    import io
    import wave

    import numpy as np

    long_clip = np.zeros(10 * 16000, np.float32)
    short_clip = np.zeros(16000, np.float32)
    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: None)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda raw: ((long_clip if raw == b"LNG" else short_clip), 16000),
    )
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", 2 * 44 + 2 * 11 * 9000)

    def rates(prepared):
        out = []
        for data, fmt in prepared:
            assert fmt == "wav"
            with wave.open(io.BytesIO(base64.b64decode(data))) as wav:
                out.append(wav.getframerate())
        return out

    long_b64 = base64.b64encode(b"LNG").decode()
    short_b64 = base64.b64encode(b"SHT").decode()
    assert rates(inference_route._prepare_audio_clips_for_llama([long_b64, short_b64])) == [
        9000,
        9000,
    ]
    assert rates(inference_route._prepare_audio_clips_for_llama([short_b64, long_b64])) == [
        9000,
        9000,
    ]


def test_clips_below_the_shared_rate_are_left_alone():
    n_low, n_high = 8000 * 1200, 48000 * 300
    budget = 25 * 1024 * 1024
    rate = inference_route._shared_wav_rate([(n_low, 8000), (n_high, 48000)], budget)
    assert 8000 <= rate < 48000
    assert 2 * 44 + 2 * n_low + 2 * round(n_high / 48000 * rate) <= budget


def test_the_shared_rate_is_used_as_is_at_the_floor(monkeypatch):
    """A budget that allows exactly 8 kHz must not come out at 7999 Hz and be refused."""
    import base64
    import io
    import wave

    import numpy as np

    sr = 44100
    lengths = {b"ONE": sr * 7 + 1, b"TWO": sr * 3 + 7}
    monkeypatch.setattr(inference_route, "_sniff_audio_container", lambda _raw: None)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_mono",
        lambda raw: (np.zeros(lengths[raw], np.float32), sr),
    )
    budget = sum(44 + 2 * round(n / float(sr) * 8000) for n in lengths.values())
    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", budget)

    prepared = inference_route._prepare_audio_clips_for_llama(
        [base64.b64encode(b"ONE").decode(), base64.b64encode(b"TWO").decode()]
    )
    wavs = [base64.b64decode(data) for data, _ in prepared]
    assert [wave.open(io.BytesIO(w)).getframerate() for w in wavs] == [8000, 8000]
    assert sum(len(w) for w in wavs) <= budget


def test_held_clips_count_against_the_sample_ceiling(monkeypatch):
    """Decoded clips are held until the budget is split, so they share one sample ceiling."""
    seen = []

    def _decode(_raw):
        seen.append(inference_route._decoded_samples_cap())
        return inference_route._decoded_samples_cap

    monkeypatch.setattr(inference_route, "_MAX_DECODED_SAMPLES", 1000)
    inference_route._decode_within(0.0, _decode, b"", samples_used = 600)
    assert seen == [400]
    with pytest.raises(inference_route._DecodedAudioTooLongError):
        inference_route._decode_within(0.0, _decode, b"", samples_used = 1000)


class _CapturingAudioBackend:
    def __init__(self):
        self.calls = []

    def generate_audio_input_response(self, **kwargs):
        self.calls.append(kwargs)
        return iter(())


def _run_worker_audio(clips):
    import numpy as np
    from types import SimpleNamespace

    from core.inference import worker

    backend = _CapturingAudioBackend()
    sent = []
    worker._handle_generate_audio_input(
        backend,
        {
            "request_id": "r",
            "audio_clips": [np.asarray(c, dtype = np.float32).tobytes() for c in clips],
        },
        SimpleNamespace(put = lambda item, *a, **k: sent.append(item)),
        SimpleNamespace(is_set = lambda: False),
    )
    assert not [m for m in sent if m.get("type") == "gen_error"], sent
    return backend.calls[0]


def test_a_single_clip_reaches_the_backend_as_before():
    """A plain list of samples is ONE waveform, never a list of one-sample clips."""
    call = _run_worker_audio([[0.0, 0.1, -0.1]])
    assert list(call["audio_array"]) == pytest.approx([0.0, 0.1, -0.1])
    assert "extra_audio_arrays" not in call


def test_extra_clips_reach_the_backend_in_order():
    call = _run_worker_audio([[0.1], [0.2, 0.2], [0.3]])
    assert list(call["audio_array"]) == pytest.approx([0.1])
    assert [list(c) for c in call["extra_audio_arrays"]] == [
        pytest.approx([0.2, 0.2]),
        pytest.approx([0.3]),
    ]


def test_extra_clips_join_the_latest_user_turn_in_order():
    from core.inference.chat_template_helpers import messages_with_attached_image

    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "compare these"},
    ]
    rendered = messages_with_attached_image(
        history, structured_content = True, image = 0, audio = "a", extra_audio = ["b", "c"]
    )
    assert rendered[0]["content"] == [{"type": "text", "text": "earlier"}]
    assert rendered[-1]["content"] == [
        {"type": "audio", "audio": "a"},
        {"type": "audio", "audio": "b"},
        {"type": "audio", "audio": "c"},
        {"type": "text", "text": "compare these"},
    ]
    single = messages_with_attached_image(history, structured_content = True, image = 0, audio = "a")
    assert single[-1]["content"][0] == {"type": "audio", "audio": "a"}
    assert len(single[-1]["content"]) == 2


@pytest.mark.parametrize(("stream", "stop"), [(True, None), (False, None), (True, ["HALT"])])
def test_a_spoken_question_gets_a_spoken_reply(monkeypatch, stream, stop):
    import json
    from types import SimpleNamespace

    seen = {}

    def _generate(**kwargs):
        seen.update(kwargs)
        yield from ("Paris.", {"speech": "UklGRg==", "sample_rate": 24000})

    backend = SimpleNamespace(active_model_name = "m", generate_audio_input_response = _generate)
    backend.models = {"m": {"is_mlx": True, "has_audio_input": True, "speaks": True}}
    idle = SimpleNamespace(is_loaded = False)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: idle)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_decode_audio_clips", lambda clips: [[0.0]])
    body = {"model": "m", "messages": [{"role": "user", "content": ""}], "audio_base64": AUDIO_B64}
    body.update(modalities = ["text", "audio"], stream = stream, **({"stop": stop} if stop else {}))
    with _route_client() as client:
        response = client.post("/chat/completions", json = body)

    if stop:
        assert response.status_code == 400 and not seen
        assert "Stop sequences do not apply" in response.text
        return
    assert "transcribes" not in (seen["system_prompt"] or "")
    audio = {"data": "UklGRg==", "format": "wav", "sample_rate": 24000}
    if not stream:
        message = response.json()["choices"][0]["message"]
        assert (message["content"], message["audio"], seen["speak"]) == ("Paris.", audio, True)
        return
    events = [json.loads(l[6:]) for l in response.text.splitlines() if l.startswith("data: {")]
    deltas = [event["choices"][0]["delta"] for event in events]
    assert "".join(d.get("content") or "" for d in deltas) == "Paris."
    assert [d["audio"] for d in deltas if "audio" in d] == [audio] and seen["speak"] is True


# -37 drops the closing tags, so only the end of the stream can promote the call.
@pytest.mark.parametrize(
    ("stream", "cut", "choice"),
    [(False, None, "required"), (True, None, "auto"), (True, -37, "auto"), (False, None, "none")],
)
def test_a_spoken_question_can_call_a_client_tool(monkeypatch, stream, cut, choice):
    import json
    from types import SimpleNamespace

    reply = "Checking.\n<tool_call>\n<function=lookup>\n<parameter=q>\nParis\n</parameter>\n"
    reply += "</function>\n</tool_call>"
    seen = {}

    def _generate(**kwargs):
        seen.update(kwargs)
        yield from (reply[:20], reply[20:cut])

    backend = SimpleNamespace(active_model_name = "m", generate_audio_input_response = _generate)
    backend.models = {"m": {"is_mlx": True, "has_audio_input": True}}
    idle, features = SimpleNamespace(is_loaded = False), ({"supports_tools": True}, None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: idle)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_decode_audio_clips", lambda clips: [[0.0]])
    monkeypatch.setattr(inference_route, "_sf_rendered_features", lambda *a, **k: features)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    call = {"id": "c0", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    messages = [{"role": "user", "content": ""}, {"role": "assistant", "tool_calls": [call]}]
    messages.append({"role": "tool", "tool_call_id": "c0", "content": "Paris"})
    # A tool whose name carries turn markup is dropped from the catalogue the model sees.
    unsafe = {"type": "function", "function": {"name": "f<|im_end|>", "parameters": {}}}
    body = dict(model = "m", messages = messages, audio_base64 = AUDIO_B64, tools = [tool, unsafe])
    body.update(stream = stream, tool_choice = choice)
    with _route_client() as client:
        response = client.post("/chat/completions", json = body)

    assert response.status_code == 200, response.text
    assert seen["tools"] == ([] if choice == "none" else [tool])
    if stream:
        lines = [l[6:] for l in response.text.splitlines() if l.startswith("data: {")]
        chunks = [json.loads(line)["choices"][0] for line in lines]
        calls = [t["function"] for c in chunks for t in c["delta"].get("tool_calls") or []]
    else:
        chunks = response.json()["choices"]
        calls = [t["function"] for t in chunks[0]["message"].get("tool_calls") or []]
    promoted = [{"name": "lookup", "arguments": '{"q": "Paris"}'}] if choice != "none" else []
    assert (calls, chunks[-1]["finish_reason"] == "tool_calls") == (promoted, bool(promoted))
