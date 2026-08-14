# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic contracts for native audio adapters. No Hub or model load occurs."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from core.inference.native_audio import (
    HIGGS_TTS3_CODEC_REPO,
    MOSS_LOCAL_CODEC_REPO,
    MOSS_NANO_CODEC_REPO,
    NativeAudioBackend,
    is_native_audio_model,
    native_audio_type_from_local_path,
    native_audio_security_targets,
)
from utils.models.model_config import _detect_audio_from_config


@pytest.mark.parametrize(
    "repo",
    (
        "bosonai/higgs-tts-2-3b-base",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "multimodalart/higgs-audio-v3-tts-4b-transformers",
        "MiniMaxAI/MiniMax-Music3",
    ),
)
def test_curated_repo_uses_native_audio_worker(repo):
    assert is_native_audio_model(repo)


def test_local_minimax_modular_index_is_detected_without_config(tmp_path):
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxMusic3ModularPipeline",
                "_blocks_class_name": "MiniMaxMusic3Blocks",
            }
        ),
        encoding = "utf-8",
    )

    assert native_audio_type_from_local_path(str(tmp_path)) == "minimax_music3"
    assert is_native_audio_model(str(tmp_path))
    assert _detect_audio_from_config(str(tmp_path)) == "minimax_music3"

    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "UnrelatedPipeline"}), encoding = "utf-8"
    )
    assert native_audio_type_from_local_path(str(tmp_path)) is None


@pytest.mark.parametrize(
    ("repo", "companion"),
    (
        ("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", MOSS_LOCAL_CODEC_REPO),
        ("OpenMOSS-Team/MOSS-TTS-Nano-100M", MOSS_NANO_CODEC_REPO),
        ("multimodalart/higgs-audio-v3-tts-4b-transformers", HIGGS_TTS3_CODEC_REPO),
    ),
)
def test_native_audio_security_targets_include_companion_repositories(repo, companion):
    assert native_audio_security_targets(repo) == [repo, companion]


def _backend(audio_type: str, **entry):
    backend = NativeAudioBackend.__new__(NativeAudioBackend)
    backend.device = "cpu"
    backend.active_model_name = "test/model"
    backend.models = {
        "test/model": {
            "audio_type": audio_type,
            "sample_rate": entry.pop("sample_rate", 24000),
            **entry,
        }
    }
    return backend


def test_native_audio_dtype_tracks_accelerator_bf16_support(monkeypatch):
    backend = _backend("higgs_tts2")
    backend.device = "cuda"

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    assert backend._dtype() is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert backend._dtype() is torch.bfloat16

    monkeypatch.setattr(torch.version, "hip", "6.0", raising = False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert backend._dtype() is torch.float16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert backend._dtype() is torch.bfloat16


def test_native_audio_context_uses_text_config_and_requested_cap():
    config = SimpleNamespace(
        max_position_embeddings = 8192,
        text_config = SimpleNamespace(max_position_embeddings = 4096),
    )
    entry = {"model": SimpleNamespace(config = config)}

    assert NativeAudioBackend._context_length(entry, 2048, "higgs_tts2") == 2048
    assert NativeAudioBackend._context_length(entry, 8192, "higgs_tts2") == 4096
    assert NativeAudioBackend._context_length(entry, 8192, "minimax_music3") == 0

    moss_entry = {
        "model": SimpleNamespace(
            config = SimpleNamespace(language_config = SimpleNamespace(max_position_embeddings = 32768))
        )
    }
    assert NativeAudioBackend._context_length(moss_entry, 0, "moss_tts_local") == 32768


def test_higgs_tts2_follows_chat_template_and_decode_contract():
    seen = {}

    class Batch(dict):
        def to(self, device):
            seen["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            seen["conversation"] = conversation
            seen["template_kwargs"] = kwargs
            return Batch(input_ids = torch.tensor([[1]]))

        def batch_decode(self, outputs):
            seen["outputs"] = outputs
            return [torch.zeros(240)]

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            seen["generate"] = kwargs
            return torch.tensor([[1, 2]])

    backend = _backend("higgs_tts2", model = Model(), processor = Processor())
    wav, sample_rate = backend.generate_audio_response(
        "Hello",
        instructions = "Close-mic studio speech.",
        top_k = -1,
        max_new_tokens = 321,
    )

    assert wav[:4] == b"RIFF" and sample_rate == 24000
    assert seen["conversation"][1]["role"] == "scene"
    assert seen["conversation"][1]["content"][0]["text"] == "Close-mic studio speech."
    assert seen["template_kwargs"]["sampling_rate"] == 24000
    assert seen["generate"]["max_new_tokens"] == 321
    assert seen["generate"]["top_k"] == 0


def test_remote_code_audio_requires_explicit_consent_before_loading():
    backend = NativeAudioBackend.__new__(NativeAudioBackend)
    backend.device = "cpu"
    backend.models = {}
    backend.active_model_name = None
    backend.loading_models = set()
    config = SimpleNamespace(
        identifier = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        path = None,
        audio_type = "moss_tts_local",
    )

    with pytest.raises(RuntimeError, match = "trust_remote_code=True"):
        backend.load_model(config, trust_remote_code = False)

    seen = {}
    backend._load_moss_local = lambda entry, source, token, trust: seen.update(
        source = source, trust = trust
    )
    assert backend.load_model(config, trust_remote_code = True)
    assert seen == {"source": config.identifier, "trust": True}


def test_native_audio_rejects_multi_gpu_before_loader_runs():
    backend = NativeAudioBackend.__new__(NativeAudioBackend)
    backend.device = "cuda"
    backend.models = {}
    backend.active_model_name = None
    backend.loading_models = set()
    backend._load_higgs_tts2 = lambda *_args: pytest.fail("loader must not run")
    config = SimpleNamespace(
        identifier = "bosonai/higgs-tts-2-3b-base",
        path = None,
        audio_type = "higgs_tts2",
    )

    with pytest.raises(RuntimeError, match = "single selected GPU"):
        backend.load_model(config, gpu_ids = [0, 1])


def test_minimax_component_load_receives_hub_token(monkeypatch):
    seen = {}

    class Pipeline:
        sampling_rate = 44100

        def load_components(self, **kwargs):
            seen["components"] = kwargs

        def to(self, device):
            seen["device"] = device

    class ModularPipeline:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["source"] = source
            seen["index"] = kwargs
            return Pipeline()

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(ModularPipeline = ModularPipeline))
    backend = _backend("minimax_music3")
    entry = {}
    backend._load_minimax_music3(entry, "MiniMaxAI/MiniMax-Music3", "secret")

    assert seen["index"]["token"] == "secret"
    assert seen["components"]["token"] == "secret"
    assert seen["device"] == "cpu"


def test_moss_local_uses_generation_messages_and_stereo_decode():
    seen = {}
    stereo = torch.zeros((2, 480))

    class Processor:
        def build_user_message(self, **kwargs):
            seen["message"] = kwargs
            return {"text": kwargs["text"]}

        def __call__(self, conversations, mode):
            seen["conversations"] = conversations
            seen["mode"] = mode
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, _outputs):
            return [SimpleNamespace(audio_codes_list = [stereo])]

    class Model:
        def generate(self, **kwargs):
            seen["generate"] = kwargs
            return torch.tensor([[1, 2]])

    backend = _backend(
        "moss_tts_local",
        model = Model(),
        processor = Processor(),
        sample_rate = 48000,
    )
    wav, sample_rate = backend.generate_audio_response("Bonjour", max_new_tokens = 400)

    assert wav[:4] == b"RIFF" and sample_rate == 48000
    assert seen["message"] == {"text": "Bonjour"}
    assert seen["mode"] == "generation"
    assert seen["generate"]["audio_top_k"] == 50
    assert seen["generate"]["do_sample"] is True

    backend.generate_audio_response("Bonjour", temperature = 0)
    assert seen["generate"]["do_sample"] is False
    assert seen["generate"]["audio_temperature"] == 0


def test_moss_nano_passes_companion_codec_and_returns_written_wav():
    seen = {}

    class Model:
        def inference(self, **kwargs):
            seen.update(kwargs)
            kwargs["output_audio_path"].write_bytes(b"RIFFnano")
            return {"sample_rate": 48000}

    codec = object()
    tokenizer = object()
    backend = _backend(
        "moss_tts_nano",
        model = Model(),
        processor = tokenizer,
        audio_codec = codec,
        sample_rate = 48000,
    )
    wav, sample_rate = backend.generate_audio_response("Portable speech", max_new_tokens = 375)

    assert wav == b"RIFFnano" and sample_rate == 48000
    assert seen["mode"] == "continuation"
    assert seen["text_tokenizer"] is tokenizer
    assert seen["audio_tokenizer"] is codec
    assert seen["max_new_frames"] == 375

    backend.generate_audio_response("Portable speech", temperature = 0, max_new_tokens = 375)
    assert seen["do_sample"] is False
    assert seen["text_temperature"] == 0
    assert seen["audio_temperature"] == 0


def test_higgs_tts3_uses_generate_speech_contract():
    seen = {}

    class Model:
        def generate_speech(self, text, tokenizer, **kwargs):
            seen.update(text = text, tokenizer = tokenizer, **kwargs)
            return torch.zeros(240)

    tokenizer = object()
    backend = _backend("higgs_tts3", model = Model(), processor = tokenizer)
    wav, sample_rate = backend.generate_audio_response(
        "Hello from v3", temperature = 0, max_new_tokens = 777
    )

    assert wav[:4] == b"RIFF" and sample_rate == 24000
    assert seen["text"] == "Hello from v3"
    assert seen["tokenizer"] is tokenizer
    assert seen["max_new_tokens"] == 777
    assert seen["temperature"] == 0

    backend.generate_audio_response("Hello from v3", temperature = 0.7, max_new_tokens = 777)
    assert seen["temperature"] == 0.7


def test_minimax_music3_passes_lyrics_description_duration_and_seed():
    seen = {}

    class Pipeline:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return [torch.zeros((2, 441))]

    backend = _backend("minimax_music3", pipeline = Pipeline(), sample_rate = 44100)
    wav, sample_rate = backend.generate_audio_response(
        "[verse]\nMorning light",
        instructions = "Acoustic pop, 96 BPM.",
        max_new_tokens = 1500,
        seed = 7,
    )

    assert wav[:4] == b"RIFF" and sample_rate == 44100
    assert seen["lyrics"] == "[verse]\nMorning light"
    assert seen["prompt"] == "Acoustic pop, 96 BPM."
    assert seen["audio_duration"] == 60.0
    assert seen["output"] == "audios"
    assert seen["generator"].initial_seed() == 7


def test_minimax_music3_omits_generator_without_a_seed():
    seen = {}

    class Pipeline:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return [torch.zeros((2, 441))]

    backend = _backend("minimax_music3", pipeline = Pipeline(), sample_rate = 44100)
    backend.generate_audio_response(
        "[verse]\nMorning light",
        instructions = "Acoustic pop, 96 BPM.",
        max_new_tokens = 750,
    )

    assert seen["audio_duration"] == 30.0
    assert "generator" not in seen


def test_minimax_music3_cancellation_checks_autoregressive_forwards():
    cancelled = threading.Event()
    removed = []

    class Handle:
        def remove(self):
            removed.append(True)

    class LanguageModelCore:
        hook = None

        def register_forward_pre_hook(self, hook):
            self.hook = hook
            return Handle()

    class LanguageModel:
        model = LanguageModelCore()

    class Pipeline:
        language_model = LanguageModel()

        def __call__(self, **_kwargs):
            self.language_model.model.hook(self.language_model.model, ())
            cancelled.set()
            self.language_model.model.hook(self.language_model.model, ())
            pytest.fail("cancelled autoregressive generation must stop before audio")

    backend = _backend("minimax_music3", pipeline = Pipeline(), sample_rate = 44100)
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate_audio_response(
            "lyrics",
            instructions = "description",
            cancel_event = cancelled,
        )
    assert removed == [True]
    assert backend.active_model_name == "test/model"


def test_minimax_music3_cancellation_checks_denoising_forwards():
    cancelled = threading.Event()
    removed = []

    class Handle:
        def remove(self):
            removed.append(True)

    class Transformer:
        hook = None

        def register_forward_pre_hook(self, hook):
            self.hook = hook
            return Handle()

    class Pipeline:
        transformer = Transformer()

        def __call__(self, **_kwargs):
            self.transformer.hook(self.transformer, ())
            cancelled.set()
            self.transformer.hook(self.transformer, ())
            pytest.fail("cancelled denoising must stop before audio")

    backend = _backend("minimax_music3", pipeline = Pipeline(), sample_rate = 44100)
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate_audio_response(
            "lyrics",
            instructions = "description",
            cancel_event = cancelled,
        )
    assert removed == [True]


def test_minimax_music3_requires_a_separate_music_description():
    backend = _backend("minimax_music3", pipeline = object(), sample_rate = 44100)
    with pytest.raises(RuntimeError, match = "music description"):
        backend.generate_audio_response("lyrics only")
