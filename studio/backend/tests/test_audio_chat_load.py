# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3-Audio-2B (audio in, no image processor) loads through AutoModelForMultimodalLM, not the text load."""

import types

import pytest

import core.inference.inference as inf


class _Lfm3AudioModel:  # what the text load builds: no generate()
    device = "cpu"


class _Lfm3AudioForConditionalGeneration(_Lfm3AudioModel):
    def generate(self, **kwargs):
        return None


class _Processor:
    feature_extractor = object()
    tokenizer = object()


def _config(**overrides):
    fields = dict(
        identifier = "LiquidAI/LFM3-Audio-2B",
        path = "LiquidAI/LFM3-Audio-2B",
        is_vision = False,
        is_lora = False,
        is_audio = False,
        audio_type = "audio_vlm",
        has_audio_input = True,
        is_local = False,
        base_model = None,
        gguf_variant = None,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.fixture
def backend(monkeypatch):
    import unsloth
    import utils.models.diffusion_lm as diffusion_lm

    calls = {}

    def fast_model(**kwargs):
        calls["fast_model"] = kwargs
        return _Lfm3AudioForConditionalGeneration(), _Processor()

    def fast_language_model(**kwargs):
        calls["fast_language_model"] = kwargs
        return _Lfm3AudioModel(), object()

    noop = lambda *a, **k: None
    monkeypatch.setattr(
        unsloth, "FastModel", types.SimpleNamespace(from_pretrained = fast_model, for_inference = noop)
    )
    monkeypatch.setattr(
        inf,
        "FastLanguageModel",
        types.SimpleNamespace(from_pretrained = fast_language_model, for_inference = noop),
    )
    monkeypatch.setattr(diffusion_lm, "diffusion_lm_profile_for_model", lambda *a, **k: None)
    monkeypatch.setattr(diffusion_lm, "loaded_diffusion_profile", lambda *a, **k: None)
    monkeypatch.setattr(inf, "get_device_map", lambda *a, **k: "sequential")
    monkeypatch.setattr(inf, "get_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(inf, "log_gpu_memory", noop)
    monkeypatch.setattr(inf, "_exact_model_name_for_load", lambda *a, **k: None)
    monkeypatch.setattr(inf, "_offload_kwargs_for_fit", lambda *a, **k: {})
    monkeypatch.setattr(inf, "raise_if_offloaded", noop)
    monkeypatch.setattr(inf, "runtime_context_length", lambda *a, **k: 4096)

    b = inf.InferenceBackend.__new__(inf.InferenceBackend)
    b.models, b.loading_models, b.active_model_name = {}, set(), None
    b._resolve_chat_eos = noop
    b._load_chat_template_info = noop
    return b, calls


def test_audio_input_chat_loads_through_the_multimodal_class(backend):
    import transformers

    b, calls = backend
    assert inf.InferenceBackend.load_model(b, _config(), trust_remote_code = True) is True
    assert calls["fast_model"]["auto_model"] is transformers.AutoModelForMultimodalLM
    assert "fast_language_model" not in calls
    loaded = b.models["LiquidAI/LFM3-Audio-2B"]
    assert isinstance(loaded["processor"], _Processor)
    assert loaded["has_audio_input"] is True


def test_a_plain_text_model_still_takes_the_text_load(backend):
    b, calls = backend
    config = _config(audio_type = None, has_audio_input = False)
    # The bare backbone the text load builds still meets the refusal.
    with pytest.raises(Exception, match = "cannot generate text"):
        inf.InferenceBackend.load_model(b, config)
    assert "fast_model" not in calls and "fast_language_model" in calls


def test_audio_chat_auto_model_needs_audio_vlm_without_vision():
    import transformers

    assert inf._audio_chat_auto_model(_config()) is transformers.AutoModelForMultimodalLM
    assert inf._audio_chat_auto_model(_config(is_vision = True)) is None  # Omni, Gemma 3n
    assert inf._audio_chat_auto_model(_config(audio_type = "whisper")) is None
    assert inf._audio_chat_auto_model(_config(has_audio_input = False, audio_type = None)) is None
