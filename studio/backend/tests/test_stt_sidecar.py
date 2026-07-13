# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference.stt_sidecar import (
    SttLanguageError,
    WhisperSttSidecar,
    normalize_whisper_language,
)


class _FakeWhisperModel:
    def __init__(self, supported_languages = None) -> None:
        self.options = None
        self.supported_languages = supported_languages or ["en", "fr", "zh"]

    def transcribe(self, _audio, **options):
        self.options = options
        return iter([SimpleNamespace(start = 0, text = " hello")]), SimpleNamespace(
            language = "en", duration = 0.5
        )


def test_fast_transcription_uses_low_latency_decode_options(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    result = sidecar.transcribe(b"encoded audio", language = "en", fast = True)

    assert result["text"] == "hello"
    assert model.options == {
        "language": "en",
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "best_of": 1,
        "temperature": 0.0,
        "without_timestamps": True,
    }


def test_accurate_transcription_keeps_beam_search_default(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    sidecar.transcribe(b"encoded audio")

    assert model.options == {
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (None, None),
        ("auto", None),
        ("en-US", "en"),
        ("en-GB", "en"),
        ("zh-CN", "zh"),
        ("ja-JP", "ja"),
        ("ko-KR", "ko"),
        ("es-ES", "es"),
        ("fr-FR", "fr"),
        ("de-DE", "de"),
        ("it-IT", "it"),
        ("pt_BR", "pt"),
        ("ru-RU", "ru"),
        ("hi-IN", "hi"),
        ("ar-SA", "ar"),
        ("iw-IL", "he"),
        ("nb-NO", "no"),
    ],
)
def test_normalize_whisper_language_accepts_bcp47(language, expected):
    assert normalize_whisper_language(language) == expected


def test_transcription_normalizes_region_qualified_language(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    sidecar.transcribe(b"encoded audio", language = "fr-FR")

    assert model.options["language"] == "fr"


def test_english_only_model_rejects_other_languages(monkeypatch):
    sidecar = WhisperSttSidecar()
    # The distilled model retains a multilingual tokenizer, so enforce its
    # actual English-only training limitation independently of tokenizer metadata.
    model = _FakeWhisperModel(supported_languages = ["en", "fr", "zh"])
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    with pytest.raises(SttLanguageError, match = "only supports English"):
        sidecar.transcribe(
            b"encoded audio",
            model = "distil-large-v3",
            language = "fr-FR",
        )


def test_unknown_language_is_not_reported_as_bad_audio(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    with pytest.raises(SttLanguageError, match = "is not supported"):
        sidecar.transcribe(b"encoded audio", language = "xx-YY")
