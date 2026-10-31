# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3-Audio-2B (audio in, no image processor) loads through AutoModelForMultimodalLM, not the text load."""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Stubs a dep the backend pytest job lacks, so collection succeeds; real installs are left alone."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - any import failure means "not usable here", so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))

import core.inference.inference as inf  # noqa: E402

# Drop the stubs once inf is bound: leftover stubs make _shared_policy return None.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


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
    import utils.models.diffusion_lm as diffusion_lm

    # load_model imports FastModel from unsloth at call time; a per-test module keeps it hermetic.
    unsloth = types.ModuleType("unsloth")
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    calls = {}

    def fast_model(**kwargs):
        calls["fast_model"] = kwargs
        return _Lfm3AudioForConditionalGeneration(), _Processor()

    def fast_language_model(**kwargs):
        calls["fast_language_model"] = kwargs
        return _Lfm3AudioModel(), object()

    noop = lambda *a, **k: None
    unsloth.FastModel = types.SimpleNamespace(from_pretrained = fast_model, for_inference = noop)
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
