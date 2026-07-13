# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import core.inference.stt_sidecar as stt_sidecar_module
from core.inference.stt_sidecar import (
    SttAudioTooLongError,
    SttLanguageError,
    SttModelNotDownloadedError,
    WhisperSttSidecar,
    normalize_whisper_language,
)

_REAL_DECODE_AUDIO_BOUNDED = stt_sidecar_module._decode_audio_bounded


@pytest.fixture(autouse = True)
def stub_audio_decoder(monkeypatch):
    """Unit tests below exercise orchestration, not PyAV container parsing."""
    monkeypatch.setattr(
        stt_sidecar_module,
        "_decode_audio_bounded",
        lambda _audio: np.zeros(8000, dtype = np.float32),
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
        "vad_filter": False,
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


def test_english_only_language_is_rejected_before_decode_or_model_load(monkeypatch):
    sidecar = WhisperSttSidecar()

    def should_not_run(*_args, **_kwargs):
        pytest.fail("invalid language must be rejected before expensive work")

    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_run)
    monkeypatch.setattr(sidecar, "load", should_not_run)

    with pytest.raises(SttLanguageError, match = "only supports English"):
        sidecar.transcribe(
            b"encoded audio",
            model = "distil-large-v3",
            language = "fr-FR",
        )


def test_unknown_language_is_rejected_before_decode_or_model_load(monkeypatch):
    sidecar = WhisperSttSidecar()

    def should_not_run(*_args, **_kwargs):
        pytest.fail("unknown language must be rejected before expensive work")

    monkeypatch.setattr(stt_sidecar_module, "_known_whisper_languages", lambda: frozenset({"en"}))
    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_run)
    monkeypatch.setattr(sidecar, "load", should_not_run)

    with pytest.raises(SttLanguageError, match = "is not supported"):
        sidecar.transcribe(b"encoded audio", language = "xx-YY")


def test_unknown_language_is_not_reported_as_bad_audio(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()
    monkeypatch.setattr(sidecar, "load", lambda _model: model)

    with pytest.raises(SttLanguageError, match = "is not supported"):
        sidecar.transcribe(b"encoded audio", language = "xx-YY")


def test_transcription_result_keeps_requested_model_id_during_switch(monkeypatch):
    sidecar = WhisperSttSidecar()
    model = _FakeWhisperModel()

    def load(_model):
        # Simulate another request changing the mutable resident-model state
        # after this request acquired its own model reference.
        sidecar._model_id = "large-v3"
        return model

    monkeypatch.setattr(sidecar, "load", load)

    result = sidecar.transcribe(b"encoded audio", model = "base")

    assert result["model"] == "base"


def test_lazy_inference_failure_is_not_returned_as_partial_success(monkeypatch):
    sidecar = WhisperSttSidecar()

    class LazyFailureModel(_FakeWhisperModel):
        def transcribe(self, _audio, **options):
            self.options = options

            def segments():
                yield SimpleNamespace(start = 0, text = " partial")
                raise RuntimeError("inference failed")

            return segments(), SimpleNamespace(language = "en", duration = 0.5)

    monkeypatch.setattr(sidecar, "load", lambda _model: LazyFailureModel())

    with pytest.raises(RuntimeError, match = "inference failed"):
        sidecar.transcribe(b"encoded audio")


def _wav_bytes(sample_count: int, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.zeros(sample_count, dtype = np.int16).tobytes())
    return output.getvalue()


def test_bounded_decoder_returns_16khz_float_pcm():
    pytest.importorskip("av")

    decoded = _REAL_DECODE_AUDIO_BOUNDED(_wav_bytes(1600))

    assert decoded.dtype == np.float32
    assert decoded.shape == (1600,)


def test_bounded_decoder_rejects_audio_as_soon_as_sample_cap_is_crossed(monkeypatch):
    pytest.importorskip("av")
    monkeypatch.setattr(stt_sidecar_module, "_MAX_AUDIO_SECONDS", 1)

    with pytest.raises(SttAudioTooLongError, match = "Audio must"):
        _REAL_DECODE_AUDIO_BOUNDED(_wav_bytes(16001))


def test_unload_releases_model_and_device():
    sidecar = WhisperSttSidecar()
    sidecar._model = object()
    sidecar._model_id = "base"
    sidecar._device = "cuda"

    sidecar.unload()

    assert sidecar.loaded_model is None
    assert sidecar.device is None


def test_load_uses_model_hub_cache_without_implicit_download(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel = FakeWhisperModel),
    )
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "int8"))

    sidecar = WhisperSttSidecar()
    sidecar.load("base")

    assert calls == [
        (
            "Systran/faster-whisper-base",
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        )
    ]


def test_load_reports_model_hub_cache_miss(monkeypatch):
    class LocalEntryNotFoundError(RuntimeError):
        pass

    class MissingWhisperModel:
        def __init__(self, *_args, **_kwargs):
            raise LocalEntryNotFoundError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel = MissingWhisperModel),
    )
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "int8"))

    with pytest.raises(SttModelNotDownloadedError, match = "Settings → Voice"):
        WhisperSttSidecar().load("large-v3")
