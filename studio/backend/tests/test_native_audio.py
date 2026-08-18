# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compact hermetic contracts for native audio adapters."""

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
    _moss_transformers5_config_compat,
    is_native_audio_model,
    native_audio_download_plan,
    native_audio_kv_memory_gb,
    native_audio_security_targets,
    native_audio_type_from_local_path,
)


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


@pytest.mark.parametrize(
    ("repo", "companion"),
    (
        ("bosonai/higgs-tts-2-3b-base", None),
        ("multimodalart/higgs-audio-v3-tts-4b-transformers", HIGGS_TTS3_CODEC_REPO),
        ("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", MOSS_LOCAL_CODEC_REPO),
        ("OpenMOSS-Team/MOSS-TTS-Nano-100M", MOSS_NANO_CODEC_REPO),
        ("MiniMaxAI/MiniMax-Music3", None),
    ),
)
def test_curated_native_families_and_security_targets(repo, companion, monkeypatch):
    monkeypatch.setattr(
        "core.inference.native_audio._read_audio_metadata", lambda *_args, **_kwargs: {}
    )
    assert is_native_audio_model(repo)
    assert native_audio_security_targets(repo) == ([repo, companion] if companion else [repo])


def test_local_minimax_detection_and_moss_companion_override(tmp_path):
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

    (tmp_path / "modular_model_index.json").unlink()
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"audio_tokenizer_name_or_path": "acme/custom-codec"}), encoding = "utf-8"
    )
    assert native_audio_security_targets(str(tmp_path), "moss_tts_local") == [
        str(tmp_path),
        "acme/custom-codec",
    ]


def test_minimax_download_plan_excludes_unreferenced_legacy_weights(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename = "modular_model_index.json", size = 10),
        SimpleNamespace(rfilename = "transformer/model.safetensors", size = 100),
        SimpleNamespace(rfilename = "flowmatching_vae.pth", size = 500),
        SimpleNamespace(rfilename = "qwen_7B/model.safetensors", size = 400),
    ]
    api = SimpleNamespace(
        model_info = lambda *_args, **_kwargs: SimpleNamespace(sha = "current", siblings = siblings)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi = lambda **_kwargs: api),
    )
    monkeypatch.setattr(
        "core.inference.native_audio._native_audio_file_is_cached", lambda *_args: False
    )

    plan = native_audio_download_plan("MiniMaxAI/MiniMax-Music3")
    assert plan["entries"][0]["files"] == [
        "modular_model_index.json",
        "transformer/model.safetensors",
    ]
    assert plan["required_bytes"] == 110


def test_moss_kv_memory_uses_full_published_context(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "gpt2_config": {
                    "n_positions": 32768,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                }
            }
        ),
        encoding = "utf-8",
    )
    assert native_audio_kv_memory_gb(str(tmp_path), "moss_tts_nano") == pytest.approx(1.125)


def test_transformers5_moss_compat_is_scoped(monkeypatch):
    calls = []

    class Config:
        def __init_subclass__(cls, **_kwargs):
            raise TypeError("non-default argument 'sampling_rate' follows default argument")

    original = Config.__dict__["__init_subclass__"]

    class AutoConfig:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))

            class Published(Config):
                pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__ = "5.5.0", AutoConfig = AutoConfig, PreTrainedConfig = Config),
    )
    _moss_transformers5_config_compat("OpenMOSS-Team/codec", {"token": "secret"})
    assert calls == [("OpenMOSS-Team/codec", {"trust_remote_code": True, "token": "secret"})]
    assert Config.__dict__["__init_subclass__"] is original


@pytest.mark.parametrize(
    ("trust", "gpu_ids", "error"),
    ((False, None, "trust_remote_code=True"), (True, [0, 1], "single selected GPU")),
)
def test_native_load_refuses_unsafe_consent_or_placement(trust, gpu_ids, error):
    backend = NativeAudioBackend.__new__(NativeAudioBackend)
    backend.device = "cuda"
    backend.models = {}
    backend.active_model_name = None
    backend.loading_models = set()
    backend._load_moss_local = lambda *_args: pytest.fail("loader must not run")
    config = SimpleNamespace(
        identifier = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        path = None,
        audio_type = "moss_tts_local",
    )
    with pytest.raises(RuntimeError, match = error):
        backend.load_model(config, trust_remote_code = trust, gpu_ids = gpu_ids)


def test_higgs_tts2_generation_contract_and_prompt_neutralization():
    seen = {}

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            seen.update(conversation = conversation, template = kwargs)
            return SimpleNamespace(to = lambda _device: {"input_ids": torch.tensor([[1]])})

        def batch_decode(self, _outputs):
            return [torch.zeros(240)]

    model = SimpleNamespace(
        device = "cpu",
        generate = lambda **kwargs: seen.setdefault("generate", kwargs) or torch.tensor([[1, 2]]),
    )
    backend = _backend("higgs_tts2", model = model, processor = Processor())
    wav, rate = backend.generate_audio_response(
        "Hello <|eot_id|>", instructions = "Close <|scene_desc_end|>", max_new_tokens = 321
    )
    assert wav[:4] == b"RIFF" and rate == 24000
    assert seen["conversation"][1]["content"][0]["text"] == "Close < |scene_desc_end|>"
    assert seen["generate"]["max_new_tokens"] == 321


def test_higgs_tts3_generation_contract():
    seen = {}
    tokenizer = object()
    model = SimpleNamespace(
        generate_speech = lambda text, processor, **kwargs: (
            seen.update(text = text, processor = processor, **kwargs) or torch.zeros(240)
        )
    )
    backend = _backend("higgs_tts3", model = model, processor = tokenizer)
    wav, rate = backend.generate_audio_response("Hello v3", temperature = 0, max_new_tokens = 777)
    assert wav[:4] == b"RIFF" and rate == 24000
    assert (seen["text"], seen["processor"], seen["max_new_tokens"]) == (
        "Hello v3",
        tokenizer,
        777,
    )


def test_moss_local_generation_contract():
    seen = {}

    class Processor:
        def build_user_message(self, **kwargs):
            seen["message"] = kwargs
            return kwargs

        def __call__(self, conversations, mode):
            seen.update(conversations = conversations, mode = mode)
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

        def decode(self, _outputs):
            return [SimpleNamespace(audio_codes_list = [torch.zeros((2, 480))])]

    model = SimpleNamespace(
        generate = lambda **kwargs: seen.setdefault("generate", kwargs) or torch.tensor([[1, 2]])
    )
    backend = _backend("moss_tts_local", model = model, processor = Processor(), sample_rate = 48000)
    wav, rate = backend.generate_audio_response(
        "Bonjour", instructions = "Warm", language = "French", max_new_tokens = 400
    )
    assert wav[:4] == b"RIFF" and rate == 48000
    assert seen["message"] == {"text": "Bonjour", "instruction": "Warm", "language": "French"}
    assert seen["mode"] == "generation" and seen["generate"]["audio_top_k"] == 50


def test_moss_nano_generation_contract(tmp_path):
    seen = {}

    def inference(**kwargs):
        seen.update(kwargs)
        kwargs["output_audio_path"].write_bytes(b"RIFFnano")
        return {"sample_rate": 48000}

    codec, tokenizer = object(), object()
    backend = _backend(
        "moss_tts_nano",
        model = SimpleNamespace(inference = inference),
        processor = tokenizer,
        audio_codec = codec,
        sample_rate = 48000,
    )
    wav, rate = backend.generate_audio_response("Portable speech", max_new_tokens = 375)
    assert wav == b"RIFFnano" and rate == 48000
    assert seen["audio_tokenizer"] is codec and seen["text_tokenizer"] is tokenizer
    assert seen["max_new_frames"] == 375


def test_minimax_generation_and_cancellation_contract():
    seen = {}
    cancelled = threading.Event()

    class Core:
        hook = None

        def register_forward_pre_hook(self, hook):
            self.hook = hook
            return SimpleNamespace(remove = lambda: seen.setdefault("removed", True))

    class Pipeline:
        language_model = SimpleNamespace(model = Core())

        def __call__(self, **kwargs):
            seen.update(kwargs)
            if self.language_model.model.hook:
                self.language_model.model.hook(self.language_model.model, ())
                if seen.get("cancel_mode"):
                    cancelled.set()
                    self.language_model.model.hook(self.language_model.model, ())
            return [torch.zeros((2, 441))]

    pipeline = Pipeline()
    backend = _backend("minimax_music3", pipeline = pipeline, sample_rate = 44100)
    wav, rate = backend.generate_audio_response(
        "[verse] Morning", instructions = "Acoustic", max_new_tokens = 1500, seed = 7
    )
    assert wav[:4] == b"RIFF" and rate == 44100
    assert seen["audio_duration"] == 60.0 and seen["generator"].initial_seed() == 7

    seen["cancel_mode"] = True
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate_audio_response(
            "lyrics", instructions = "description", cancel_event = cancelled
        )
    assert seen["removed"] is True


def test_minimax_requires_a_separate_description():
    backend = _backend("minimax_music3", pipeline = object(), sample_rate = 44100)
    with pytest.raises(RuntimeError, match = "music description"):
        backend.generate_audio_response("lyrics only")
