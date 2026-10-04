# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Voxtral / Qwen2-Audio are not mapped under AutoModelForImageTextToText on transformers 5."""

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models.loader import _resolve_omni_auto_model, resolve_model_class  # noqa: E402
from unsloth.models.vision import _multimodal_auto_classes  # noqa: E402

try:
    from transformers import AutoModelForImageTextToText as IMAGE_TEXT_CLASS
except ImportError:  # transformers 4.x
    from transformers import AutoModelForVision2Seq as IMAGE_TEXT_CLASS


def _config(name):
    cls = getattr(transformers, name, None)
    if cls is None:
        pytest.skip(f"this transformers has no {name}")
    if getattr(transformers, "AutoModelForMultimodalLM", None) is None:
        pytest.skip("this transformers has no AutoModelForMultimodalLM")
    return cls()


@pytest.mark.parametrize("name", ["VoxtralConfig", "VoxtralRealtimeConfig", "Qwen2AudioConfig"])
def test_speech_to_text_models_resolve_to_a_class_that_maps(name):
    config = _config(name)
    assert resolve_model_class(IMAGE_TEXT_CLASS, config) is None
    resolved = _resolve_omni_auto_model(config)
    assert resolved is not None
    assert resolve_model_class(resolved, config) is not None
    assert resolved in _multimodal_auto_classes()


def test_qwen3_omni_keeps_text_to_waveform():
    config = _config("Qwen3OmniMoeConfig")
    assert _resolve_omni_auto_model(config) is transformers.AutoModelForTextToWaveform
