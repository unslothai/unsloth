# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import core.inference.stt_sidecar as stt_sidecar_module
from core.inference.stt_sidecar import (
    DEFAULT_STT_MODEL,
    STT_MODELS,
    SttAudioDecodeError,
    SttAudioTooLongError,
    SttLanguageError,
    SttLoadCancelledError,
    SttModelCompatibilityError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttUnavailableError,
    WhisperSttSidecar,
    normalize_whisper_language,
    resolve_model_id,
    resolve_model_repo,
    validate_remote_model,
)

_REAL_DECODE_AUDIO_BOUNDED = stt_sidecar_module._decode_audio_bounded
_REAL_ENSURE_STT_AVAILABLE = stt_sidecar_module.ensure_stt_available


@pytest.fixture(autouse = True)
def stub_audio_decoder(monkeypatch):
    """Unit tests below exercise orchestration, not PyAV container parsing."""
    monkeypatch.setattr(
        stt_sidecar_module,
        "_decode_audio_bounded",
        lambda _audio: np.zeros(8000, dtype = np.float32),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: "/cached/model",
    )
    # transcribe() gates on the runtime up front; treat it as present so these
    # orchestration tests run without PyTorch/Transformers/PyAV installed.
    # The runtime-specific tests restore the real check.
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)


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


def test_five_curated_whisper_models_are_offered():
    assert STT_MODELS == {
        "tiny": "unsloth/whisper-tiny",
        "base": "unsloth/whisper-base",
        "small": "unsloth/whisper-small",
        "large-v3-turbo": "unsloth/whisper-large-v3-turbo",
        "large-v3": "unsloth/whisper-large-v3",
    }
    assert all(repo.startswith(("unsloth/", "unslothai/")) for repo in STT_MODELS.values())
    assert DEFAULT_STT_MODEL in STT_MODELS


def test_av_is_required_for_stt_availability(monkeypatch):
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", _REAL_ENSURE_STT_AVAILABLE)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "av", None)

    assert stt_sidecar_module.is_available() is False


def test_transformers_is_required_for_stt_availability(monkeypatch):
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", _REAL_ENSURE_STT_AVAILABLE)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert stt_sidecar_module.is_available() is False


@pytest.mark.parametrize("missing", ["transformers", "av"])
def test_load_rejects_an_incomplete_stt_runtime(monkeypatch, missing):
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", _REAL_ENSURE_STT_AVAILABLE)
    for module in ("torch", "transformers", "av"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda _model: pytest.fail("runtime must be checked before the model cache"),
    )

    with pytest.raises(SttUnavailableError, match = "needs PyTorch, Transformers, and PyAV"):
        sidecar.load("small")


def test_model_id_accepts_defaults_and_custom_hub_repositories():
    assert resolve_model_id("tiny") == "tiny"
    assert resolve_model_id(None) == DEFAULT_STT_MODEL
    assert resolve_model_id("large-v3") == "large-v3"
    assert resolve_model_id("openai/whisper-medium") == "openai/whisper-medium"
    assert resolve_model_repo("tiny") == "unsloth/whisper-tiny"
    assert resolve_model_repo("openai/whisper-medium") == "openai/whisper-medium"


@pytest.mark.parametrize("model", ["tiny-ish", "owner/model/extra", "../model", "owner/"])
def test_invalid_custom_model_id_is_rejected(model):
    with pytest.raises(SttModelIdError, match = "owner/model"):
        resolve_model_id(model)


def test_remote_custom_model_validation_requires_whisper_config(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", token))

        def model_info(self, repo, **kwargs):
            calls.append(("model_info", repo, kwargs))
            return SimpleNamespace(
                config = {
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                }
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    result = validate_remote_model("owner/custom-whisper", "hf_private")

    assert result == {"model": "owner/custom-whisper", "repo": "owner/custom-whisper"}
    assert calls == [
        ("token", "hf_private"),
        (
            "model_info",
            "owner/custom-whisper",
            {"expand": ["config"], "timeout": 10},
        ),
    ]


def test_remote_custom_model_validation_rejects_non_whisper(monkeypatch):
    class FakeApi:
        def __init__(self, token):
            assert token is False

        def model_info(self, _repo, **_kwargs):
            return SimpleNamespace(
                config = {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                }
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    with pytest.raises(SttModelCompatibilityError, match = "not a compatible"):
        validate_remote_model("owner/chat-model")


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


def test_english_only_model_rejects_non_english_before_decode(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}')
    (tmp_path / "generation_config.json").write_text('{"is_multilingual": false}')
    sidecar = WhisperSttSidecar()

    def should_not_decode(_audio):
        pytest.fail("English-only language mismatch must be rejected before decode")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_decode)

    with pytest.raises(SttLanguageError, match = "English-only"):
        sidecar.transcribe(
            b"encoded audio",
            model = "owner/whisper-small.en",
            language = "fr-FR",
        )


def test_english_only_model_omits_forbidden_generation_controls(monkeypatch):
    calls = []

    class FakeTensor:
        def to(self, *_args):
            return self

    class FakeProcessor:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_features = FakeTensor())

        def batch_decode(self, *_args, **_kwargs):
            return ["hello"]

    class FakeModel:
        dtype = None
        device = "cpu"
        generation_config = SimpleNamespace(is_multilingual = False)

        def generate(self, _features, **kwargs):
            calls.append(kwargs)
            return [[1]]

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad = NoGrad))
    sidecar = WhisperSttSidecar()
    monkeypatch.setattr(sidecar, "load", lambda _model: (FakeModel(), FakeProcessor()))

    text = sidecar._transcribe_decoded(
        "owner/whisper-small.en",
        np.zeros(160, dtype = np.float32),
        {
            "task": "transcribe",
            "language": "en",
            "condition_on_prev_tokens": False,
            "num_beams": 1,
        },
    )

    assert text == "hello"
    assert calls == [{"condition_on_prev_tokens": False, "num_beams": 1}]


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
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
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


@pytest.mark.parametrize(
    ("model_id", "repo_id"),
    [
        ("small", "unsloth/whisper-small"),
        ("tiny", "unsloth/whisper-tiny"),
        ("openai/whisper-medium", "openai/whisper-medium"),
    ],
)
def test_model_cache_preflight_is_local_only(monkeypatch, tmp_path, model_id, repo_id):
    calls = []
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}')
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(tmp_path),
    )

    WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(model_id)

    assert calls == [{"repo_id": repo_id, "local_files_only": True}]


def test_model_cache_preflight_reports_missing_snapshot(monkeypatch):
    class LocalEntryNotFoundError(RuntimeError):
        pass

    def missing(**_kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded("large-v3")


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


def test_unavailable_runtime_is_rejected_before_audio_decode(monkeypatch):
    sidecar = WhisperSttSidecar()

    def unavailable() -> None:
        raise SttUnavailableError("needs PyTorch, Transformers, and PyAV")

    def should_not_decode(_audio):
        pytest.fail("runtime must be checked before audio decode")

    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", unavailable)
    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_decode)

    with pytest.raises(SttUnavailableError, match = "needs PyTorch"):
        sidecar.transcribe(b"encoded audio", model = "small")


def test_missing_model_is_rejected_before_audio_decode(monkeypatch):
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)

    def missing(_model_id):
        raise SttModelNotDownloadedError("not downloaded")

    def should_not_decode(_audio):
        pytest.fail("missing models must be rejected before audio decode")

    monkeypatch.setattr(sidecar, "_ensure_model_downloaded", missing, raising = False)
    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", should_not_decode)

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        sidecar.transcribe(b"encoded audio", model = "large-v3")


def test_missing_model_switch_keeps_resident_model(monkeypatch):
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    resident = object()
    sidecar._engine = resident
    sidecar._model_id = "small"
    sidecar._device = "cpu"

    def missing(_model_id):
        raise SttModelNotDownloadedError("not downloaded")

    monkeypatch.setattr(sidecar, "_ensure_model_downloaded", missing, raising = False)
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(
        sidecar,
        "_build_model",
        lambda *_args: pytest.fail("cache miss must be detected before model replacement"),
    )
    _install_fake_torch(monkeypatch)

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        sidecar.load("large-v3")

    assert sidecar._engine is resident
    assert sidecar.loaded_model == "small"


def test_incompatible_custom_model_switch_keeps_resident_model(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}'
    )
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    resident = (object(), object())
    sidecar._engine = resident
    sidecar._model_id = "small"
    sidecar._device = "cpu"
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )

    with pytest.raises(SttModelCompatibilityError, match = "not a compatible"):
        sidecar.load("owner/chat-model")

    assert sidecar._engine is resident
    assert sidecar.loaded_model == "small"


def test_loaded_model_stays_warm_until_idle_timer_fires(monkeypatch):
    timers = []
    _install_fake_torch(monkeypatch)

    def make_timer(*args, **kwargs):
        timer = _FakeTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    sidecar = WhisperSttSidecar(keep_alive_seconds = 300)
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)
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
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar_module.threading, "Timer", make_timer)
    monkeypatch.setattr(sidecar, "_build_model", lambda *_args: (object(), object()))
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar.load("small")
    first = timers[-1]
    sidecar.load("small")

    assert first.cancelled
    assert timers[-1] is not first

    first.fire()

    assert sidecar.loaded_model == "small"


def test_unload_waits_for_inflight_transcription(monkeypatch):
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    started = threading.Event()
    release = threading.Event()

    def transcribe(*_args):
        started.set()
        assert release.wait(timeout = 2)
        return "hello"

    monkeypatch.setattr(sidecar, "_transcribe_decoded", transcribe)
    transcribe_thread = threading.Thread(target = lambda: sidecar.transcribe(b"audio"))
    transcribe_thread.start()
    assert started.wait(timeout = 2)

    unload_thread = threading.Thread(target = sidecar.unload)
    unload_thread.start()
    time.sleep(0.02)
    assert unload_thread.is_alive()

    release.set()
    transcribe_thread.join(timeout = 2)
    unload_thread.join(timeout = 2)

    assert not transcribe_thread.is_alive()
    assert not unload_thread.is_alive()


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


def test_new_stt_load_prefers_mps_when_cuda_is_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        cuda = SimpleNamespace(is_available = lambda: False),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(stt_sidecar_module, "_training_active", lambda: False)

    assert stt_sidecar_module._pick_device() == ("mps", "float32")


def test_new_stt_load_uses_cpu_without_accelerators(monkeypatch):
    fake_torch = SimpleNamespace(
        float16 = "float16",
        float32 = "float32",
        cuda = SimpleNamespace(is_available = lambda: False),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(stt_sidecar_module, "_training_active", lambda: False)

    assert stt_sidecar_module._pick_device() == ("cpu", "float32")


def test_accelerator_load_failure_retries_on_cpu(monkeypatch):
    fake_torch = _install_fake_torch(monkeypatch)
    calls = []
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)

    def build(_repo, device, dtype, _cancel_event):
        calls.append((device, dtype))
        if device == "cuda":
            raise RuntimeError("accelerator allocation failed")
        return object(), object()

    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cuda", "float16"))
    monkeypatch.setattr(sidecar, "_build_model", build)

    sidecar.load("small")

    assert calls == [("cuda", "float16"), ("cpu", fake_torch.float32)]
    assert sidecar.device == "cpu"


def test_pending_load_can_be_cancelled_without_waiting_for_model_lock(monkeypatch):
    _install_fake_torch(monkeypatch)
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    build_started = threading.Event()
    release_build = threading.Event()
    errors = []

    def build(_repo, _device, _dtype, _cancel_event):
        build_started.set()
        assert release_build.wait(timeout = 2)
        return object(), object()

    def run_load():
        try:
            sidecar.load("small")
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(stt_sidecar_module, "ensure_stt_available", lambda: None)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    monkeypatch.setattr(sidecar, "_build_model", build)

    load_thread = threading.Thread(target = run_load)
    load_thread.start()
    assert build_started.wait(timeout = 2)

    result = []
    cancel_thread = threading.Thread(target = lambda: result.append(sidecar.cancel_pending_load()))
    cancel_thread.start()
    cancel_thread.join(timeout = 2)

    assert not cancel_thread.is_alive()
    assert result == [True]
    assert load_thread.is_alive()

    release_build.set()
    load_thread.join(timeout = 2)

    assert not load_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SttLoadCancelledError)
    assert sidecar.loaded_model is None
    assert sidecar.is_loading() is False


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


def test_bounded_decoder_resamples_stereo_48khz_to_mono_16khz():
    pytest.importorskip("av")
    output = io.BytesIO()
    frames = np.zeros((4800, 2), dtype = np.int16)
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(frames.tobytes())

    decoded = _REAL_DECODE_AUDIO_BOUNDED(output.getvalue())

    assert decoded.dtype == np.float32
    assert 1590 <= len(decoded) <= 1610


@pytest.mark.parametrize("audio", [b"", b"not audio", b"RIFF\x00\x00"])
def test_bounded_decoder_rejects_malformed_audio(audio):
    pytest.importorskip("av")

    with pytest.raises(SttAudioDecodeError, match = "Could not decode"):
        _REAL_DECODE_AUDIO_BOUNDED(audio)


def test_bounded_decoder_rejects_container_without_audio_stream(monkeypatch):
    class FakeFFmpegError(Exception):
        pass

    class FakeResampler:
        def __init__(self, **_kwargs):
            pass

    class FakeFifo:
        samples = 0

    class FakeContainer:
        streams = SimpleNamespace(audio = [])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_av = SimpleNamespace(
        audio = SimpleNamespace(
            resampler = SimpleNamespace(AudioResampler = FakeResampler),
            fifo = SimpleNamespace(AudioFifo = FakeFifo),
        ),
        open = lambda *_args, **_kwargs: FakeContainer(),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setitem(
        sys.modules,
        "av.error",
        SimpleNamespace(
            FFmpegError = FakeFFmpegError,
            InvalidDataError = FakeFFmpegError,
        ),
    )

    with pytest.raises(SttAudioDecodeError, match = "Could not decode"):
        _REAL_DECODE_AUDIO_BOUNDED(b"video-only")


def test_unload_releases_model_and_device():
    sidecar = WhisperSttSidecar()
    sidecar._engine = object()
    sidecar._model_id = "small"
    sidecar._device = "cpu"

    sidecar.unload()

    assert sidecar.loaded_model is None
    assert sidecar.device is None


# ---------------------------------------------------------------------------
# Snapshot download tracking
# ---------------------------------------------------------------------------


def test_download_status_is_idle_before_any_download():
    state = stt_sidecar_module._SnapshotDownloadState()

    status = state.status()

    assert status == {
        "downloading": False,
        "model": None,
        "error": None,
        "bytes_total": None,
        "bytes_done": None,
    }


def test_download_rejects_a_second_model_while_one_is_in_flight(monkeypatch):
    state = stt_sidecar_module._SnapshotDownloadState()
    release = threading.Event()
    monkeypatch.setattr(
        state, "_run", lambda repo, token: release.wait(timeout = 5)
    )

    state.start("small")
    try:
        # Re-requesting the in-flight model is a no-op, not an error.
        state.start("small")
        with pytest.raises(SttModelIdError, match = "still"):
            state.start("tiny")
        assert state.status()["downloading"] is True
        assert state.status()["model"] == "small"
    finally:
        release.set()


def test_download_failure_is_reported_in_status(monkeypatch):
    state = stt_sidecar_module._SnapshotDownloadState()
    # Mask huggingface_hub so the import inside _run fails fast.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    state.start("small")
    state._thread.join(timeout = 5)

    status = state.status()
    assert status["downloading"] is False
    assert "Download failed" in (status["error"] or "")


def test_is_model_downloaded_is_false_for_a_cache_miss(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert stt_sidecar_module.is_model_downloaded("small") is False
