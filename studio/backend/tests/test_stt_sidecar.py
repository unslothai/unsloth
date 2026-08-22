# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import gc
import io
import json
import sys
import threading
import time
import wave
import weakref
from pathlib import Path
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
    SttModelBusyError,
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
_REAL_SNAPSHOT_IS_COMPLETE = stt_sidecar_module._snapshot_is_complete
_REAL_FIND_COMPLETE_CACHED_SNAPSHOT = stt_sidecar_module._find_complete_cached_snapshot


@pytest.fixture(autouse = True)
def stub_audio_decoder(monkeypatch):
    """Unit tests below exercise orchestration, not PyAV container parsing."""
    monkeypatch.setattr(
        stt_sidecar_module,
        "_decode_audio_bounded",
        # Second arg is the cancel event the real decoder polls in its frame loop.
        lambda _audio, _cancel_event = None: np.zeros(8000, dtype = np.float32),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: "/cached/model",
    )
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        lambda _model: Path("/cached/model"),
    )
    # The stubbed snapshot path holds no files; snapshot-integrity tests
    # restore the real check.
    monkeypatch.setattr(stt_sidecar_module, "_snapshot_is_complete", lambda _snapshot: True)
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
        return self.text, None


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
                sha = "a" * 40,
                config = {
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                },
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    result = validate_remote_model("owner/custom-whisper", "hf_private")

    assert result == {
        "model": "owner/custom-whisper",
        "repo": "owner/custom-whisper",
        "revision": "a" * 40,
    }
    assert calls == [
        ("token", "hf_private"),
        (
            "model_info",
            "owner/custom-whisper",
            {"expand": ["config", "sha"], "timeout": 10},
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


def test_remote_custom_model_validation_requires_an_immutable_sha(monkeypatch):
    class FakeApi:
        def __init__(self, token):
            pass

        def model_info(self, _repo, **_kwargs):
            return SimpleNamespace(sha = None, config = {"model_type": "whisper"})

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    with pytest.raises(SttModelCompatibilityError, match = "immutable revision"):
        validate_remote_model("owner/custom-whisper")


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
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        lambda _model: tmp_path,
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

    class FakeWorker:
        generation_config = SimpleNamespace(is_multilingual = False)

        def transcribe_window(
            self,
            _pcm,
            generate_kwargs,
            _cancel_event = None,
        ):
            calls.append(generate_kwargs)
            return "hello"

    sidecar = WhisperSttSidecar()
    monkeypatch.setattr(sidecar, "load", lambda _model: FakeWorker())

    text, segments = sidecar._transcribe_decoded(
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
    rate = stt_sidecar_module._TARGET_SAMPLE_RATE
    assert segments == [{"start": 0.0, "end": 160 / rate, "text": "hello"}]
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


def _install_fake_worker(monkeypatch, start = None):
    """Replace the spawn child with a handle that records how it was started."""
    started = []

    class FakeWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.closed = False
            self.alive = True

        def start(
            self,
            snapshot_path,
            device,
            dtype_name,
            cancel_event = None,
        ):
            started.append((snapshot_path, device, dtype_name))
            if start is not None:
                start(snapshot_path, device, dtype_name, cancel_event)
            self.device = device

        def is_alive(self):
            return self.alive

        def close(self):
            self.closed = True
            self.alive = False

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker", FakeWorker, raising = True
    )
    return started


def test_load_hands_the_cached_snapshot_to_the_worker_process(monkeypatch):
    # The model never enters this process: an accelerator context taken here is
    # never given back, which is the whole reason the engine is out of process.
    _install_fake_torch(monkeypatch)
    started = _install_fake_worker(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")

    # str(Path(...)), so the separator is the platform's.
    assert started == [(str(Path("/cached/model")), "cpu", "float32")]
    assert sidecar.loaded_model == "small"


def test_load_sends_the_dtype_by_name_so_torch_stays_out_of_the_command(monkeypatch):
    fake_torch = _install_fake_torch(monkeypatch)
    fake_torch.float16 = "torch.float16"
    started = _install_fake_worker(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cuda", fake_torch.float16))

    WhisperSttSidecar(keep_alive_seconds = 0).load("small")

    assert started == [(str(Path("/cached/model")), "cuda", "float16")]


def test_a_worker_that_died_is_not_resident_and_is_replaced_on_the_next_load(monkeypatch):
    _install_fake_torch(monkeypatch)
    started = _install_fake_worker(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    worker = sidecar.load("small")
    worker.alive = False

    assert sidecar.loaded_model is None

    replacement = sidecar.load("small")

    assert replacement is not worker
    assert len(started) == 2
    assert sidecar.loaded_model == "small"


def test_unload_stops_the_worker_process(monkeypatch):
    # empty_cache cannot return the context; ending the process is what does.
    _install_fake_torch(monkeypatch)
    _install_fake_worker(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    worker = sidecar.load("small")
    sidecar.unload()

    assert worker.closed is True
    assert sidecar.loaded_model is None


def test_a_worker_rejected_by_a_late_cancel_is_stopped_rather_than_leaked(monkeypatch):
    # cancel_pending_load() does not wait for the model lock, so it can land after
    # start() came back with a live child. Nothing installed that child, and dropping
    # the handle does not end the process holding the context training waits for.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    workers = []

    class LateCancelWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.closed = False
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            # The load succeeded; the cancel arrives a moment later.
            self.device = device
            assert sidecar.cancel_pending_load() is True

        def is_alive(self):
            return not self.closed

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker", LateCancelWorker, raising = True
    )

    with pytest.raises(SttLoadCancelledError):
        sidecar.load("small")

    assert len(workers) == 1
    assert workers[0].closed is True
    assert sidecar.loaded_model is None
    assert sidecar.is_loading() is False


def test_a_late_cancelled_worker_that_outlived_the_kill_stays_resident(monkeypatch):
    # The same late cancel, against a child that answers False: it survived terminate
    # and kill and still holds its accelerator memory, so reporting nothing resident
    # would let training be admitted against memory that is not free.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    workers = []

    class UnkillableLateCancelWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.closes = 0
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            self.device = device
            assert sidecar.cancel_pending_load() is True

        def is_alive(self):
            return True

        def close(self):
            self.closes += 1
            return False

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker",
        UnkillableLateCancelWorker,
        raising = True,
    )

    with pytest.raises(SttLoadCancelledError):
        sidecar.load("small")

    assert len(workers) == 1
    # Once. A second terminate-and-kill round on the way out would only spend
    # another wait on a child that just proved it does not answer them.
    assert workers[0].closes == 1
    assert sidecar.loaded_model == "small"
    assert sidecar.device == "cpu"
    assert sidecar.is_loading() is False


def _install_unkillable_worker(monkeypatch):
    """A worker whose child outlived terminate and kill, so close() answers False."""
    workers = []

    class UnkillableWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.closes = 0
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            self.device = device

        def is_alive(self):
            return True

        def close(self):
            self.closes += 1
            return False

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker", UnkillableWorker, raising = True
    )
    return workers


def test_a_worker_that_outlived_the_kill_stays_resident_rather_than_reported_unloaded(monkeypatch):
    # close() answers False for a child wedged in a driver call that outlives SIGKILL. It
    # still holds its memory, so reporting the model unloaded would let training be
    # admitted against memory that is not free.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    workers = _install_unkillable_worker(monkeypatch)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")
    sidecar.unload()

    assert workers[0].closes == 1
    assert sidecar.loaded_model == "small"
    assert sidecar.device == "cpu"


def test_a_new_load_is_refused_while_the_previous_worker_still_holds_its_memory(monkeypatch):
    # Starting a second child over one that never exited doubles the memory the
    # first was unloaded to release, so refuse and let the caller retry.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    workers = _install_unkillable_worker(monkeypatch)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")

    with pytest.raises(SttModelBusyError, match = "did not exit"):
        sidecar.load("base")

    assert len(workers) == 1
    assert sidecar.loaded_model == "small"
    assert sidecar.is_loading() is False


def test_a_surviving_worker_is_not_handed_to_the_next_dictation(monkeypatch):
    # It is held for its memory, not for its answers: it already had its shutdown, a
    # terminate and a kill, so handing it to a transcription costs the caller the whole
    # command timeout under the model lock. Refuse, and let the retry kill it again.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    workers = _install_unkillable_worker(monkeypatch)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")
    sidecar.unload()

    with pytest.raises(SttModelBusyError, match = "did not exit"):
        sidecar.load("small")

    assert len(workers) == 1
    assert sidecar.loaded_model == "small"


def test_a_worker_wedged_by_a_cancelled_transcription_is_not_handed_to_the_next_dictation(
    monkeypatch,
):
    # A cancel that outruns the grace closes the worker from inside the handle, so
    # close() answers False to nobody and the sidecar never learns the child outlived
    # both signals. Handing it on spends the whole command timeout under the model lock.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    workers = []

    class WedgedByCancelWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.survived_kill = False
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            self.device = device

        def is_alive(self):
            return True

        def transcribe_window(
            self,
            _pcm,
            _generate_kwargs,
            _cancel_event = None,
        ):
            # What _await does once the cancel grace expires: close() terminates and
            # kills, the child answers neither, and the cancel is raised over its False.
            self.survived_kill = True
            raise stt_sidecar_module.SttTranscriptionCancelledError("Transcription cancelled.")

        def close(self):
            return False

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker",
        WedgedByCancelWorker,
        raising = True,
    )

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    cancel_event = threading.Event()
    with pytest.raises(stt_sidecar_module.SttTranscriptionCancelledError):
        sidecar.transcribe(b"audio", "small", cancel_event = cancel_event)

    with pytest.raises(SttModelBusyError, match = "did not exit"):
        sidecar.load("small")

    assert len(workers) == 1
    # Still accounted for: it holds its memory until the kill finally takes.
    assert sidecar.loaded_model == "small"


def _install_worker_that_survives_its_own_start(monkeypatch, error):
    """A worker whose start() fails over a child that outlived its own kill.

    start() ends its child on every failure, so a handle still reporting a live
    process is one holding memory nothing else knows about.
    """
    workers = []

    class SurvivingStartWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            # start() never got as far as naming the device it loaded on.
            self.device = None
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            _device,
            _dtype_name,
            _cancel_event = None,
        ):
            raise error

        def is_alive(self):
            return True

        def close(self):
            return False

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker",
        SurvivingStartWorker,
        raising = True,
    )
    return workers


def test_a_child_that_outlived_the_kill_inside_start_stays_accounted(monkeypatch):
    # The cancel lands while the child is inside from_pretrained, which never reads it,
    # so the load kills the child on the way out and the child outlives it. Nothing
    # installed that handle and it is the only one on the process: dropping it reports
    # nothing resident while the context is taken, admitting training against it.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    workers = _install_worker_that_survives_its_own_start(
        monkeypatch, SttLoadCancelledError("STT model loading was cancelled so training can start.")
    )

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    with pytest.raises(SttLoadCancelledError):
        sidecar.load("small")

    assert len(workers) == 1
    assert sidecar.loaded_model == "small"
    # The child never named a device, so the one the attempt was made on stands in.
    assert sidecar.device == "cpu"
    assert sidecar.is_loading() is False


def test_a_load_whose_child_outlived_the_kill_is_not_retried_onto_a_second_child(monkeypatch):
    # The accelerator attempt failed leaving a live child, so the CPU retry would
    # put a second child beside it and, installing that one, forget the first.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cuda", "float16"))
    workers = _install_worker_that_survives_its_own_start(
        monkeypatch, RuntimeError("the dictation worker stopped responding")
    )

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    with pytest.raises(SttModelBusyError, match = "did not exit"):
        sidecar.load("small")

    assert len(workers) == 1
    assert sidecar.loaded_model == "small"
    assert sidecar.device == "cuda"
    assert sidecar.is_loading() is False


def test_an_idle_unload_retries_a_worker_that_outlived_the_kill(monkeypatch):
    # Keeping the survivor resident must not strand it: the idle timer is rearmed
    # so the release is tried again once the child finally goes.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    monkeypatch.setattr(stt_sidecar_module.threading, "Timer", _FakeTimer)
    workers = _install_unkillable_worker(monkeypatch)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 60)
    sidecar.load("small")
    sidecar._idle_timer.fire()

    assert workers[0].closes == 1
    assert sidecar.loaded_model == "small"
    assert sidecar._idle_timer.started is True


def _install_worker(monkeypatch, *, is_alive, close):
    """Install a worker handle whose liveness and close behaviour a test chooses."""
    workers = []

    class _ScriptedWorker:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(is_multilingual = None)
            self.device = None
            self.closes = 0
            self.alive = True
            workers.append(self)

        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            self.device = device

        def is_alive(self):
            return is_alive(self)

        def close(self):
            self.closes += 1
            return close(self)

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker", _ScriptedWorker, raising = True
    )
    return workers


def _closed(worker):
    worker.alive = False
    return True


def test_a_worker_whose_liveness_cannot_be_read_stays_resident(monkeypatch):
    # An unanswerable probe is not evidence that the context was released, so reporting
    # nothing resident would let training be admitted against memory still held.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    def unreadable(_worker):
        raise OSError("the process handle cannot be inspected")

    workers = _install_worker(monkeypatch, is_alive = unreadable, close = _closed)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")

    assert stt_sidecar_module._engine_is_alive(workers[0]) is True
    assert sidecar.loaded_model == "small"
    assert workers[0].closes == 0


def test_a_close_that_raises_is_a_failed_release(monkeypatch):
    # close() raising out of join/terminate/kill confirms nothing dead, so discarding
    # the only handle would advertise memory the child is still holding as free.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    def raising(_worker):
        raise OSError("the worker could not be joined")

    workers = _install_worker(monkeypatch, is_alive = lambda worker: True, close = raising)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")
    sidecar.unload()

    assert workers[0].closes == 1
    assert sidecar.loaded_model == "small"
    assert sidecar.device == "cpu"


def test_a_close_that_raises_over_a_child_already_gone_still_releases(monkeypatch):
    # The other direction: bookkeeping that raised over a confirmed dead child
    # holds no memory, and keeping it would refuse every later load.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    def raising_after_death(worker):
        worker.alive = False
        raise OSError("the worker pid could not be forgotten")

    workers = _install_worker(
        monkeypatch, is_alive = lambda worker: worker.alive, close = raising_after_death
    )

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")
    sidecar.unload()

    assert workers[0].closes == 1
    assert sidecar.loaded_model is None
    assert sidecar.device is None


def test_a_worker_being_reaped_stays_visible_to_training_admission(monkeypatch):
    # loaded_model and summarize_resident_stt() read the fields without the model
    # lock, so clearing them before the reap would report nothing resident for the
    # whole close, and training would start into the child's accelerator memory.
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))
    release = threading.Event()

    def slow_close(worker):
        release.wait(timeout = 10)
        worker.alive = False
        return True

    _install_worker(monkeypatch, is_alive = lambda worker: worker.alive, close = slow_close)

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")

    seen = {}
    unloading = threading.Thread(target = sidecar.unload, daemon = True)
    unloading.start()
    try:
        time.sleep(0.2)  # inside the close
        seen["model"] = sidecar.loaded_model
        seen["device"] = sidecar.device
    finally:
        release.set()
        unloading.join(timeout = 10)

    assert seen["model"] == "small", "the reaping worker read as gone; training would miss its VRAM"
    assert seen["device"] == "cpu"
    # Only once the child is confirmed dead does it read as unloaded.
    assert sidecar.loaded_model is None
    assert sidecar.device is None


def test_a_host_that_cannot_spawn_keeps_dictation_in_process_on_cpu(monkeypatch):
    # Moving the engine out of process may not take dictation away from a host that
    # cannot create a child. The accelerator attempt goes through the usual CPU retry
    # first, so nobody is downgraded before that has run.
    import core.inference.stt_transformers_worker as worker_module

    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cuda", "float16"))
    spawned = []
    loaded = []

    class RefusedWorker:
        def start(
            self,
            _snapshot_path,
            device,
            _dtype_name,
            _cancel_event = None,
        ):
            spawned.append(device)
            raise worker_module.SttWorkerSpawnError("spawn is not permitted here")

    monkeypatch.setattr(
        "core.inference.stt_transformers_worker.WhisperWorker", RefusedWorker, raising = True
    )
    monkeypatch.setattr(
        worker_module,
        "load_whisper",
        lambda path, device, dtype, _cancel = None: (
            loaded.append((path, device, dtype)) or (_FakeModel(), object())
        ),
    )
    monkeypatch.setattr(worker_module, "transcribe_window", lambda *_args, **_kwargs: "hello")

    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    engine = sidecar.load("small")

    assert spawned == ["cuda", "cpu"]
    # In process only after both spawns failed, and only ever on the CPU.
    assert loaded == [(str(Path("/cached/model")), "cpu", "float32")]
    assert sidecar.device == "cpu"
    assert sidecar.loaded_model == "small"
    assert engine.transcribe_window(b"", {}) == "hello"


def test_model_cache_preflight_uses_shared_offline_resolver(monkeypatch):
    seen = []
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        lambda model: seen.append(model) or Path("/cached/model"),
    )

    WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded("small")

    assert seen == ["small"]


def test_model_cache_preflight_reports_missing_snapshot(monkeypatch):
    monkeypatch.setattr(stt_sidecar_module, "_find_complete_cached_snapshot", lambda _model: None)

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded("large-v3")


def test_load_reports_model_hub_cache_miss(monkeypatch):
    _install_fake_torch(monkeypatch)

    def missing(*_args, **_kwargs):
        # What the worker reports for a cache miss it hit inside from_pretrained.
        raise SttModelNotDownloadedError("The dictation model is not downloaded.")

    _install_fake_worker(monkeypatch, start = missing)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("cpu", "float32"))

    with pytest.raises(SttModelNotDownloadedError, match = "large-v3"):
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
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        lambda _model: tmp_path,
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
        return "hello", None

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
    # The bound that carries the contract is the CANCEL thread's: it must come back while the
    # build still holds the model lock, so two seconds against an indefinite wait is the whole
    # question. The other waits below are liveness only -- they say "this must happen", and the
    # assertion after each is what fails if it does not. Two seconds there was a budget on
    # runner speed instead: after the build is released the load thread still has to be
    # scheduled and run the cancel path's two full gc.collect()s, which a loaded two-core CI
    # runner did not finish inside it.
    settle = 30.0
    _install_fake_torch(monkeypatch)
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    build_started = threading.Event()
    release_build = threading.Event()
    errors = []

    def build(_repo, _device, _dtype, _cancel_event):
        build_started.set()
        assert release_build.wait(timeout = settle)
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
    assert build_started.wait(timeout = settle)

    result = []
    cancel_thread = threading.Thread(target = lambda: result.append(sidecar.cancel_pending_load()))
    cancel_thread.start()
    cancel_thread.join(timeout = 2)  # the contract: back while the build still holds the lock

    assert not cancel_thread.is_alive()
    assert result == [True]
    assert load_thread.is_alive()

    release_build.set()
    load_thread.join(timeout = settle)

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


def _write_complete_snapshot(snapshot: Path, *, model_type: str = "whisper") -> None:
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}))
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")


def _sibling(name: str, size: int, key: str):
    return SimpleNamespace(rfilename = name, size = size, blob_id = key, lfs = None)


def test_sha_snapshot_without_main_ref_survives_restart_and_cache_relocation(monkeypatch, tmp_path):
    repo = "openai/whisper-tiny.en"
    revision = "c" * 40
    studio_home = tmp_path / "studio"
    first_cache = tmp_path / "first-hub"
    second_cache = tmp_path / "second-hub"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(stt_sidecar_module, "_snapshot_is_complete", _REAL_SNAPSHOT_IS_COMPLETE)
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        _REAL_FIND_COMPLETE_CACHED_SNAPSHOT,
    )

    first = first_cache / "models--openai--whisper-tiny.en" / "snapshots" / revision
    _write_complete_snapshot(first)
    monkeypatch.setenv("HF_HUB_CACHE", str(first_cache))
    stt_sidecar_module._write_revision_record(repo, revision)
    assert stt_sidecar_module._find_complete_cached_snapshot(repo) == first.resolve()

    second = second_cache / "models--openai--whisper-tiny.en" / "snapshots" / revision
    _write_complete_snapshot(second)
    monkeypatch.setenv("HF_HUB_CACHE", str(second_cache))
    assert stt_sidecar_module._find_complete_cached_snapshot(repo) == second.resolve()


def test_corrupt_or_escaping_revision_record_is_ignored(monkeypatch, tmp_path):
    repo = "openai/whisper-tiny.en"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        _REAL_FIND_COMPLETE_CACHED_SNAPSHOT,
    )
    record = stt_sidecar_module._revision_record_path(repo)
    record.parent.mkdir(parents = True)
    record.write_text(json.dumps({"version": 1, "repo": repo, "revision": "../../outside"}))
    assert stt_sidecar_module._find_complete_cached_snapshot(repo) is None

    outside = tmp_path / "outside"
    _write_complete_snapshot(outside)
    snapshots = tmp_path / "hub" / "models--openai--whisper-tiny.en" / "snapshots"
    snapshots.mkdir(parents = True)
    (snapshots / ("d" * 40)).symlink_to(outside, target_is_directory = True)
    assert stt_sidecar_module._find_complete_cached_snapshot(repo) is None


def test_adapter_only_snapshot_is_not_complete(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}')
    (tmp_path / "preprocessor_config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    assert _REAL_SNAPSHOT_IS_COMPLETE(tmp_path) is False


def test_snapshot_selection_prefers_safetensors_and_excludes_unrelated_files():
    info = SimpleNamespace(
        siblings = [
            _sibling("config.json", 10, "config"),
            _sibling("preprocessor_config.json", 20, "preprocessor"),
            _sibling("tokenizer.json", 30, "tokenizer"),
            _sibling("model.safetensors", 100, "safe"),
            _sibling("pytorch_model.bin", 110, "torch"),
            _sibling("README.md", 1000, "readme"),
        ]
    )

    selected = stt_sidecar_module._select_snapshot_files(
        info, lambda _name: pytest.fail("unsharded selection must not load an index")
    )

    assert {item.path for item in selected} == {
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    assert sum(item.size for item in selected) == 160


def test_snapshot_selection_includes_every_indexed_shard():
    info = SimpleNamespace(
        siblings = [
            _sibling("config.json", 10, "config"),
            _sibling("model.safetensors.index.json", 5, "index"),
            _sibling("model-00001-of-00002.safetensors", 50, "shard1"),
            _sibling("model-00002-of-00002.safetensors", 60, "shard2"),
            _sibling("pytorch_model.bin", 120, "torch"),
        ]
    )

    selected = stt_sidecar_module._select_snapshot_files(
        info,
        lambda name: {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
    )

    assert {item.path for item in selected} == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_snapshot_selection_rejects_pickle_only_weights():
    # A custom repo shipping only pytorch_model.bin (pickle) must fail closed:
    # selecting it would download a checkpoint that runs code at load time.
    info = SimpleNamespace(
        siblings = [
            _sibling("config.json", 10, "config"),
            _sibling("preprocessor_config.json", 20, "preprocessor"),
            _sibling("tokenizer.json", 30, "tokenizer"),
            _sibling("pytorch_model.bin", 110, "torch"),
        ]
    )

    with pytest.raises(SttModelCompatibilityError, match = "safetensors"):
        stt_sidecar_module._select_snapshot_files(
            info, lambda _name: pytest.fail("pickle weights must not be selected")
        )


def test_snapshot_selection_rejects_safe_index_pointing_at_pickle_shards():
    # A safetensors index can name .bin shards; Transformers dispatches shard
    # loading by extension, so those shards would still pickle-load. The index
    # is attacker-controlled, so a non-safetensors shard must fail closed.
    info = SimpleNamespace(
        siblings = [
            _sibling("config.json", 10, "config"),
            _sibling("model.safetensors.index.json", 5, "index"),
            _sibling("pytorch_model-00001-of-00001.bin", 90, "shard"),
        ]
    )

    with pytest.raises(SttModelCompatibilityError, match = "non-safetensors shards"):
        stt_sidecar_module._select_snapshot_files(
            info,
            lambda _name: {"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}},
        )


def test_progress_counts_only_selected_blobs_and_caps_incomplete_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    blobs = tmp_path / "hub" / "models--owner--whisper" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "one").write_bytes(b"x" * 10)
    (blobs / "two.incomplete").write_bytes(b"x" * 30)
    (blobs / "unrelated").write_bytes(b"x" * 1000)
    state = stt_sidecar_module._SnapshotDownloadState()
    state._repo = "owner/whisper"
    state._selected_files = (
        stt_sidecar_module._SelectedHubFile("config.json", 10, "one"),
        stt_sidecar_module._SelectedHubFile("model.safetensors", 20, "two"),
    )
    state._total_bytes = 30
    state._complete = True

    status = state.status()

    assert status["bytes_total"] == 30
    assert status["bytes_done"] == 30


class _FakeDownloadProcess:
    """Stands in for the worker subprocess: immediate success."""

    returncode = 0

    def poll(self):
        return 0

    def communicate(self):
        return (None, b"")


def _worker_args(args) -> dict:
    """Parse the download worker's argv into the fields tests assert on."""
    parsed: dict = {"filenames": []}
    remaining = iter(args)
    for flag in remaining:
        value = next(remaining)
        if flag == "--repo-id":
            parsed["repo_id"] = value
        elif flag == "--revision":
            parsed["revision"] = value
        elif flag == "--filename":
            parsed["filenames"].append(value)
    return parsed


def test_download_metadata_and_snapshot_use_the_same_revision(monkeypatch, tmp_path):
    revision = "e" * 40
    calls = []
    siblings = [
        _sibling("config.json", 10, "config"),
        _sibling("preprocessor_config.json", 20, "preprocessor"),
        _sibling("tokenizer.json", 30, "tokenizer"),
        _sibling("model.safetensors", 100, "safe"),
        _sibling("pytorch_model.bin", 110, "torch"),
    ]

    class FakeApi:
        def __init__(self, token):
            pass

        def model_info(self, repo, **kwargs):
            calls.append(("info", repo, kwargs))
            return SimpleNamespace(sha = revision, siblings = siblings)

    def fake_spawn_download(
        args,
        hf_token = None,
        *,
        hub_cache = None,
    ):
        calls.append(("snapshot", _worker_args(args)))
        return _FakeDownloadProcess()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    # The transfer runs in a worker process, so assert on its argv.
    monkeypatch.setattr("core.inference.stt_download_worker.spawn_download", fake_spawn_download)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: pytest.fail("unsharded selection must not load an index"),
    )
    monkeypatch.setattr(stt_sidecar_module, "_snapshot_is_complete", lambda _path: True)
    monkeypatch.setattr(stt_sidecar_module, "_write_revision_record", lambda *_args: None)
    state = stt_sidecar_module._SnapshotDownloadState()

    state._run("owner/whisper", None, revision)

    assert calls[0] == (
        "info",
        "owner/whisper",
        {"revision": revision, "files_metadata": True, "timeout": 30},
    )
    assert calls[1][0] == "snapshot"
    assert calls[1][1]["revision"] == revision
    assert "model.safetensors" in calls[1][1]["filenames"]
    assert "pytorch_model.bin" not in calls[1][1]["filenames"]


def test_download_status_is_idle_before_any_download():
    state = stt_sidecar_module._SnapshotDownloadState()

    status = state.status()

    assert status == {
        "downloading": False,
        "model": None,
        "error": None,
        "cancelled": False,
        # Only set alongside cancelled: "model" goes None once the download thread stops, so
        # this is what tells a settled cancellation from an unrelated one.
        "cancelled_model": None,
        "bytes_total": None,
        "bytes_done": None,
    }


def test_download_rejects_a_second_model_while_one_is_in_flight(monkeypatch):
    state = stt_sidecar_module._SnapshotDownloadState()
    release = threading.Event()
    monkeypatch.setattr(
        state,
        "_run",
        lambda repo, token, revision: release.wait(timeout = 5),
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
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        _REAL_FIND_COMPLETE_CACHED_SNAPSHOT,
    )
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent/stt-test-cache")

    assert stt_sidecar_module.is_model_downloaded("small") is False


def test_sharded_snapshot_with_missing_shard_is_not_downloaded(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        _REAL_FIND_COMPLETE_CACHED_SNAPSHOT,
    )
    monkeypatch.setattr(stt_sidecar_module, "_snapshot_is_complete", _REAL_SNAPSHOT_IS_COMPLETE)
    snap = tmp_path / "hub" / "models--unsloth--whisper-small" / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True)
    (snap / "config.json").write_bytes(b"{}")
    (snap / "preprocessor_config.json").write_bytes(b"{}")
    (snap / "tokenizer.json").write_bytes(b"{}")
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"w" * 8)

    assert stt_sidecar_module.is_model_downloaded("small") is False

    # Completing the second shard flips the verdict.
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"w" * 8)
    assert stt_sidecar_module.is_model_downloaded("small") is True


@pytest.mark.parametrize("model_id", ["small", "openai/whisper-medium"])
def test_preflight_rejects_partial_snapshot(monkeypatch, tmp_path, model_id):
    # A resolvable snapshot with metadata but no weights must fail preflight,
    # not survive until load() after the audio has already been decoded.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(
        stt_sidecar_module,
        "_find_complete_cached_snapshot",
        _REAL_FIND_COMPLETE_CACHED_SNAPSHOT,
    )
    monkeypatch.setattr(stt_sidecar_module, "_snapshot_is_complete", _REAL_SNAPSHOT_IS_COMPLETE)
    repo = STT_MODELS.get(model_id, model_id)
    snapshot = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text('{"model_type": "whisper"}')

    with pytest.raises(SttModelNotDownloadedError, match = "not downloaded"):
        WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(model_id)

    # Completing the snapshot clears the preflight.
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"w" * 8)
    WhisperSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(model_id)


def test_cpu_retry_releases_failed_accelerator_load(monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(stt_sidecar_module, "_pick_device", lambda: ("mps", "float16"))

    class Marker:
        pass

    seen = {}

    def fake_build(self, repo, device, dtype, cancel_event):
        if device != "cpu":
            # The frame local stands in for a partly loaded accelerator model
            # kept alive only through the raised traceback.
            marker = Marker()
            seen["ref"] = weakref.ref(marker)
            raise RuntimeError("accelerator load failed")
        gc.collect()
        seen["alive_during_retry"] = seen["ref"]() is not None
        return (_FakeModel(), object())

    monkeypatch.setattr(WhisperSttSidecar, "_build_model", fake_build)
    sidecar = WhisperSttSidecar(keep_alive_seconds = 0)
    sidecar.load("small")

    # The failed attempt must be collectable before the CPU model loads, or
    # its accelerator memory stays stranded for the whole retry.
    assert seen["alive_during_retry"] is False
    assert sidecar.device == "cpu"


def test_decoding_stops_as_soon_as_the_request_is_cancelled(monkeypatch):
    """Checking only after the decode returned let an abandoned upload run to EOF or the
    30-minute cap, and several could do that at once."""
    import threading

    monkeypatch.setattr(stt_sidecar_module, "_decode_audio_bounded", _REAL_DECODE_AUDIO_BOUNDED)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(16000 * 5, dtype = np.int16)).tobytes())

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(stt_sidecar_module.SttTranscriptionCancelledError):
        stt_sidecar_module._decode_audio_bounded(buf.getvalue(), cancelled)
