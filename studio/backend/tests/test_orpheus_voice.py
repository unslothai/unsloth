# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Orpheus built-in voices (#5829): the "{voice}: {text}" prefix the -ft checkpoints were trained on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference.audio_errors import AudioRuntimeError
from core.inference.chat_template_helpers import (
    ORPHEUS_VOICES,
    tts_prompt_with_voice,
    tts_voice_options,
)


def test_only_orpheus_offers_voices():
    (option,) = tts_voice_options("snac")
    assert option["name"] == "voice" and option["type"] == "enum"
    assert option["values"] == list(ORPHEUS_VOICES)
    # No default: an unset voice must keep today's unprefixed prompt.
    assert option["default"] is None
    for audio_type in ("csm", "bicodec", "dac", "whisper", None):
        assert tts_voice_options(audio_type) is None


@pytest.mark.parametrize("options", [None, {}, {"voice": None}, {"voice": ""}])
def test_no_voice_leaves_the_prompt_unchanged(options):
    assert tts_prompt_with_voice("Hello there.", "snac", options) == "Hello there."


def test_a_voice_prefixes_the_prompt():
    assert (
        tts_prompt_with_voice("Hello <laugh>.", "snac", {"voice": "leo"}) == "leo: Hello <laugh>."
    )


def test_other_codecs_ignore_a_voice():
    assert tts_prompt_with_voice("Hello.", "bicodec", {"voice": "tara"}) == "Hello."


def test_an_unknown_voice_is_a_client_error():
    with pytest.raises(AudioRuntimeError) as err:
        tts_prompt_with_voice("Hello.", "snac", {"voice": "alloy"})
    assert err.value.status == 400


def test_transformers_orpheus_prompt_carries_the_voice(monkeypatch):
    pytest.importorskip("peft")
    from core.inference.inference import InferenceBackend

    backend = InferenceBackend.__new__(InferenceBackend)
    import threading

    backend._generation_lock = threading.Lock()
    backend.active_model_name = "unsloth/orpheus-3b-0.1-ft"
    backend.models = {
        backend.active_model_name: {"audio_type": "snac", "model": object(), "tokenizer": None}
    }
    seen = []

    def fake_snac(_model, _tokenizer, text, *_args, **_kwargs):
        seen.append(text)
        return b"wav", 24000

    monkeypatch.setattr(backend, "_generate_snac", fake_snac, raising = False)
    monkeypatch.setattr(backend, "_cancel_stopping_criteria", lambda _event: None, raising = False)
    backend.generate_audio_response("Hi.", audio_options = {"voice": "tara"})
    backend.generate_audio_response("Hi.")
    assert seen == ["tara: Hi.", "Hi."]


class _Captured(Exception):
    pass


def test_gguf_orpheus_prompt_carries_the_voice(monkeypatch):
    from core.inference import llama_cpp

    payloads = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, _url, json):
            payloads.append(json)
            raise _Captured

    monkeypatch.setattr(llama_cpp.httpx, "Client", FakeClient)
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._hf_variant = backend._gguf_path = None  # read by the low-quant name trim
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "base_url", "http://127.0.0.1:1", raising = False)
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_auth_headers", {}, raising = False)
    # The voice slot re-ensures its codec before each clip; a resident one skips that load.
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_codec_mgr", SimpleNamespace(has_codec = lambda _t: True)
    )
    for options in ({"voice": "zoe"}, None):
        with pytest.raises(_Captured):
            backend.generate_audio_response("Hi.", "snac", audio_options = options)
    assert [p["prompt"] for p in payloads] == [
        "<custom_token_3>zoe: Hi.<|eot_id|><custom_token_4>",
        "<custom_token_3>Hi.<|eot_id|><custom_token_4>",
    ]


def test_gguf_status_lists_voices_only_for_orpheus():
    from core.inference.llama_cpp import LlamaCppBackend
    from routes.inference import _llama_runtime_fields

    def status(**attrs):
        backend = LlamaCppBackend()
        for name, value in attrs.items():
            setattr(backend, name, value)
        return _llama_runtime_fields(backend)["audio_options"]

    assert status(_is_audio = True, _audio_type = "snac") == tts_voice_options("snac")
    assert status(_is_audio = True, _audio_type = "bicodec") is None
    assert status(_is_audio = False, _audio_type = None) is None


def test_speech_api_voice_maps_to_a_listed_speaker():
    from routes.inference import _apply_builtin_voice
    options = tts_voice_options("snac")
    for voice, sent, expected in (
        ("tara", None, {"voice": "tara"}),
        ("alloy", None, None),
        ("tara", {"voice": "leo"}, {"voice": "leo"}),
    ):
        payload = SimpleNamespace(audio_options = sent)
        _apply_builtin_voice(payload, voice, options)
        assert payload.audio_options == expected
