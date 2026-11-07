# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# An audio VLM's vision tower is read from config.json, not the sidecar probe, which can't run
# LFM3-Audio's repo code and answered "unknown" on every load.

import pytest

import utils.models.model_config as mc

LFM3_AUDIO = {"model_type": "lfm3_audio", "architectures": ["Lfm3AudioForConditionalGeneration"]}
GEMMA_3N = {"model_type": "gemma3n", "vision_config": {}, "audio_config": {}}
LFM3_OMNI = {"model_type": "lfm3_omni", "vision_config": {}}


@pytest.fixture
def config(monkeypatch):
    def use(raw):
        monkeypatch.setattr(mc, "_read_raw_config_json", lambda *a, **k: raw)

    # The latest-tier sidecar probe that failed on the VM: any call means the config wasn't trusted.
    def probe(*a, **k):
        raise AssertionError("sidecar probe must not run")

    monkeypatch.setattr(mc, "_is_vision_model_subprocess", probe)
    return use


def test_lfm3_audio_is_a_definite_no(config):
    config(LFM3_AUDIO)
    assert mc.audio_vlm_vision_checked("LiquidAI/LFM3-Audio-2B") == (False, True)


@pytest.mark.parametrize("raw", [GEMMA_3N, LFM3_OMNI])
def test_declared_vision_is_a_definite_yes(config, raw):
    config(raw)
    assert mc.audio_vlm_vision_checked("m") == (True, True)


def test_unreadable_config_is_unknown(config):
    config(None)  # network failure or missing file
    assert mc.audio_vlm_vision_checked("m") == (False, False)
