# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Raw config.json decides, not the sidecar probe, which can't run LFM3-Audio's repo code.

import pytest

import utils.models.model_config as mc

LFM3_AUDIO = {
    "model_type": "lfm3_audio",
    "architectures": ["Lfm3AudioForConditionalGeneration"],
    "auto_map": {
        "AutoModelForMultimodalLM": "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration"
    },
}
AUDIO_FLAMINGO_3 = {"model_type": "audioflamingo3", "audio_config": {}, "text_config": {}}
GEMMA_3N = {"model_type": "gemma3n", "vision_config": {}, "audio_config": {}}
LFM3_OMNI = {"model_type": "lfm3_omni", "vision_config": {}}


@pytest.fixture
def config(monkeypatch):
    def use(raw):
        monkeypatch.setattr(mc, "_read_raw_config_json", lambda *a, **k: raw)

    def probe(*a, **k):
        raise AssertionError("sidecar probe must not run")

    monkeypatch.setattr(mc, "_is_vision_model_subprocess", probe)
    return use


def test_lfm3_audio_takes_the_multimodal_class(config):
    config(LFM3_AUDIO)
    assert mc.audio_vlm_multimodal_lm_checked("LiquidAI/LFM3-Audio-2B") == (True, True)


def test_a_native_multimodal_mapping_takes_it(config):
    names = pytest.importorskip("transformers.models.auto.modeling_auto")
    if "qwen2_audio" not in getattr(names, "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {}):
        pytest.skip("this transformers has no AutoModelForMultimodalLM mapping for qwen2_audio")
    config({"model_type": "qwen2_audio", "audio_config": {}})
    assert mc.audio_vlm_multimodal_lm_checked("m") == (True, True)


def test_audio_without_a_multimodal_class_keeps_the_default(config):
    config(AUDIO_FLAMINGO_3)
    assert mc.audio_vlm_multimodal_lm_checked("nvidia/audio-flamingo-3-hf") == (False, True)


@pytest.mark.parametrize("raw", [GEMMA_3N, LFM3_OMNI])
def test_declared_vision_keeps_the_default(config, raw):
    config(raw)
    assert mc.audio_vlm_multimodal_lm_checked("m") == (False, True)


def test_unreadable_config_is_unknown(config):
    config(None)
    assert mc.audio_vlm_multimodal_lm_checked("m") == (False, False)
