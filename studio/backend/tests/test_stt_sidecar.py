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
    DEFAULT_STT_MODEL,
    STT_MODELS,
    SttAudioTooLongError,
    SttLanguageError,
    SttModelNotDownloadedError,
    WhisperSttSidecar,
    normalize_whisper_language,
    resolve_model_id,
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


class _CaptureInference:
    """Stand-in for the model inference step; records how it was called."""

    def __init__(
        self,
        text = "hello",
        mutate = None,
    ) -> None:
        self.text = text
        self.mutate = mutate
        self.generate_kwargs = None

    def __call__(self, model_id, decoded, generate_kwargs):
        self.generate_kwargs = generate_kwargs
        if self.mutate is not None:
            self.mutate()
        return self.text


def test_only_unsloth_models_are_offered():
    # Guard against ever pointing STT at third-party weights.
    assert set(STT_MODELS) == {"small", "large-v3-turbo", "large-v3"}
    assert all(repo.startswith("unsloth/") for repo in STT_MODELS.values())
    assert DEFAULT_STT_MODEL in STT_MODELS


def test_unknown_model_id_falls_back_to_default():
    assert resolve_model_id("tiny") == DEFAULT_STT_MODEL
    assert resolve_model_id(None) == DEFAULT_STT_MODEL
    assert resolve_model_id("large-v3") == "large-v3"


def test_fast_transcription_uses_greedy_decoding(monkeypatch):
    sidecar = WhisperSttSidecar()
    infer = _CaptureInference()
    monkeypatch.setattr(sidecar, "_transcribe_decoded", infer)

    result = sidecar.transcribe(b"encoded audio", language = "en", fast = True)

    assert result["text"] == "hello"
    assert result["duration"] == 0.5
    assert result["model"] == DEFAULT_STT_MODEL
    assert infer.generate_kwargs == {
        "task": "transcribe",
        "condition_on_prev_tokens": False,
        "num_beams": 1,
        "language": "en",
    }


def test_accurate_transcription_keeps_beam_search_default(monkeypatch):
    sidecar = WhisperSttSidecar()
    infer = _CaptureInference()
    monkeypatch.setattr(sidecar, "_transcribe_decoded", infer)

    sidecar.transcribe(b"encoded audio")

    assert infer.generate_kwargs == {
        "task": "transcribe",
        "condition_on_prev_tokens": False,
        "num_beams": 5,
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
    infer = _CaptureInference()
    monkeypatch.setattr(sidecar, "_transcribe_decoded", infer)

    sidecar.transcribe(b"encoded audio", language = "fr-FR")

    assert infer.generate_kwargs["language"] == "fr"


def test_unknown_language_is_rejected_before_decode_or_model_load(monkeypatch):
    sidecar = WhisperSttSidecar()

    def should_not_run(*_args, **_kwargs):
        pytest.fail("unknown language must be rejected before expensive work")

    monkeypatch.setattr(stt_sidecar_module, "_known_whisper_languages", lambda: frozenset({"en"}))
    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_run)
    monkeypatch.setattr(sidecar, "_transcribe_decoded", should_not_run)

    with pytest.raises(SttLanguageError, match = "is not supported"):
        sidecar.transcribe(b"encoded audio", language = "xx-YY")


def test_unknown_language_is_not_reported_as_bad_audio(monkeypatch):
    sidecar = WhisperSttSidecar()
    infer = _CaptureInference()
    monkeypatch.setattr(sidecar, "_transcribe_decoded", infer)
    monkeypatch.setattr(stt_sidecar_module, "_known_whisper_languages", lambda: frozenset({"en"}))

    with pytest.raises(SttLanguageError, match = "is not supported"):
        sidecar.transcribe(b"encoded audio", language = "xx-YY")


def test_transcription_result_keeps_requested_model_id_during_switch(monkeypatch):
    sidecar = WhisperSttSidecar()

    # Simulate another request changing the mutable resident-model state after
    # this request pinned its own model id.
    infer = _CaptureInference(mutate = lambda: setattr(sidecar, "_model_id", "large-v3"))
    monkeypatch.setattr(sidecar, "_transcribe_decoded", infer)

    result = sidecar.transcribe(b"encoded audio", model = "small")

    assert result["model"] == "small"


def test_inference_failure_propagates(monkeypatch):
    sidecar = WhisperSttSidecar()

    def boom(*_args, **_kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(sidecar, "_transcribe_decoded", boom)

    with pytest.raises(RuntimeError, match = "inference failed"):
        sidecar.transcribe(b"encoded audio")


class _FakeModel:
    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


class _FakeTimer:
    def __init__(
        self,
        interval,
        function,
        args = (),
        kwargs = None,
    ):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self.cancelled = False
        self.daemon = False
        self.started = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function(*self.args, **self.kwargs)


def _install_fake_torch(monkeypatch):
    fake_torch = SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        device = lambda value: value,
        cuda = SimpleNamespace(is_available = lambda: False),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_load_uses_model_hub_cache_without_implicit_download(monkeypatch):
    calls = []
    _install_fake_torch(monkeypatch)

    class FakeWhisperForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            calls.append(("model", repo, kwargs))
            return _FakeModel()

    class FakeWhisperProcessor:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            calls.append(("processor", repo, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperForConditionalGeneration = FakeWhisperForConditionalGeneration,
            WhisperProcessor = FakeWhisperProcessor,
        ),
    )
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    WhisperSttSidecar(keep_alive_seconds = 0).load("small")

    assert {(kind, repo) for kind, repo, _ in calls} == {
        ("processor", "unsloth/whisper-small"),
        ("model", "unsloth/whisper-small"),
    }
    # Never fetch weights implicitly; the Model Hub owns downloads.
    assert all(kwargs.get("local_files_only") is True for _, _, kwargs in calls)


def test_load_reports_model_hub_cache_miss(monkeypatch):
    _install_fake_torch(monkeypatch)

    class LocalEntryNotFoundError(RuntimeError):
        pass

    class MissingWhisperProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise LocalEntryNotFoundError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperForConditionalGeneration = object,
            WhisperProcessor = MissingWhisperProcessor,
        ),
    )
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        WhisperSttSidecar(keep_alive_seconds = 0).load("large-v3")


def test_loaded_model_stays_warm_until_idle_timer_fires(monkeypatch):
    timers = []
    _install_fake_torch(monkeypatch)

    def make_timer(*args, **kwargs):
        timer = _FakeTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    sidecar = WhisperSttSidecar(keep_alive_seconds = 300)
    monkeypatch.setattr(stt_sidecar_module.threading, "Timer", make_timer)
    monkeypatch.setattr(sidecar, "_build_model", lambda *_args: (object(), object()))
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar.load("small")

    assert sidecar.loaded_model == "small"
    assert timers[-1].interval == 300
    assert timers[-1].started

    timers[-1].fire()

    assert sidecar.loaded_model is None


def test_reusing_loaded_model_refreshes_idle_timer(monkeypatch):
    timers = []
    _install_fake_torch(monkeypatch)

    def make_timer(*args, **kwargs):
        timer = _FakeTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    sidecar = WhisperSttSidecar(keep_alive_seconds = 300)
    monkeypatch.setattr(stt_sidecar_module.threading, "Timer", make_timer)
    monkeypatch.setattr(sidecar, "_build_model", lambda *_args: (object(), object()))
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar.load("small")
    first = timers[-1]
    sidecar.load("small")

    assert first.cancelled
    assert timers[-1] is not first


def test_new_stt_load_uses_cpu_while_training(monkeypatch):
    fake_torch = SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        cuda = SimpleNamespace(is_available = lambda: True),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(stt_sidecar_module, "_training_active", lambda: True)

    assert stt_sidecar_module._pick_device() == ("cpu", "float32")


def test_new_stt_load_prefers_cuda_when_training_is_idle(monkeypatch):
    fake_torch = SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        cuda = SimpleNamespace(is_available = lambda: True),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(stt_sidecar_module, "_training_active", lambda: False)

    assert stt_sidecar_module._pick_device() == ("cuda", "float16")


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
    sidecar._engine = object()
    sidecar._model_id = "small"
    sidecar._device = "cpu"

    sidecar.unload()

    assert sidecar.loaded_model is None
    assert sidecar.device is None
