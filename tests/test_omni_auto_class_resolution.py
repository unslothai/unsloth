# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Auto-class resolution for omni checkpoints, and what it must NOT move.

`Qwen/Qwen3-Omni-30B-A3B-Instruct` names `Qwen3OmniMoeForConditionalGeneration`
and so reads as a VLM, but transformers registers `qwen3_omni_moe` only under
`AutoModelForTextToWaveform`. Every other auto class raises "Unrecognized
configuration class" on it, which is a hard load failure before the weights are
touched.

The resolver is consulted ONLY when the already-chosen class has no mapping, so
these tests spend as much effort on the models that must keep taking exactly
the branch they take today as on the one that changes.
"""

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models.loader import (  # noqa: E402
    _resolve_omni_auto_model,
    resolve_model_class,
)
from unsloth.models.vision import (  # noqa: E402
    _embeddings_or_none,
    _multimodal_auto_classes,
)

try:
    from transformers import AutoModelForImageTextToText as IMAGE_TEXT_CLASS
except ImportError:  # transformers 4.x
    from transformers import AutoModelForVision2Seq as IMAGE_TEXT_CLASS


def _omni_config():
    config_class = getattr(transformers, "Qwen3OmniMoeConfig", None)
    if config_class is None:
        pytest.skip("this transformers has no Qwen3-Omni")
    return config_class()


def test_the_defect_is_real_on_this_transformers():
    """Without this, the test below would prove nothing on a build that maps it."""
    config = _omni_config()
    assert resolve_model_class(IMAGE_TEXT_CLASS, config) is None


def test_omni_resolves_to_the_class_its_family_registered():
    config = _omni_config()
    resolved = _resolve_omni_auto_model(config, ["Qwen3OmniMoeForConditionalGeneration"])
    assert resolved is not None
    # and it really maps, rather than merely being a different name to fail on
    assert resolve_model_class(resolved, config) is not None


@pytest.mark.parametrize("config_name", ["Gemma3Config", "Qwen2VLConfig", "LlavaConfig"])
def test_ordinary_vlms_are_never_rerouted(config_name):
    """The resolver is behind `resolve_model_class(...) is None`, so a model
    that resolves today must never reach it."""
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"this transformers has no {config_name}")
    assert resolve_model_class(IMAGE_TEXT_CLASS, config_class()) is not None


def test_unmappable_config_falls_back_to_the_named_architecture():
    """Nothing matching must return the concrete class, not None-by-accident."""
    resolved = _resolve_omni_auto_model(
        transformers.LlamaConfig(),
        ["LlamaForCausalLM"],
    )
    assert resolved is not None


def test_no_architecture_and_no_mapping_returns_None():
    """None means "keep your own choice", so the existing error surfaces."""
    assert _resolve_omni_auto_model(object(), []) is None


def test_speech_seq2seq_is_not_treated_as_a_vision_model():
    """Whisper must stay out of this set.

    `is_vlm` also arms the image-processor repair path in `vision.py`, and a
    WhisperProcessor legitimately has no `image_processor`. Including
    `AutoModelForSpeechSeq2Seq` here made loading Whisper call
    `_construct_vlm_processor_fallback("openai/whisper-tiny", "whisper")`,
    an image processor build for an audio model. Whisper already reaches
    AutoProcessor through `is_whisper`, so it has no business here.
    """
    speech = getattr(transformers, "AutoModelForSpeechSeq2Seq", None)
    if speech is None:
        pytest.skip("this transformers has no AutoModelForSpeechSeq2Seq")
    assert speech not in _multimodal_auto_classes()


def test_image_text_class_is_still_in_the_processor_set():
    assert IMAGE_TEXT_CLASS in _multimodal_auto_classes()


class _CannotAnswer:
    def get_input_embeddings(self):
        raise NotImplementedError("composite model, no single embedding")


class _WrongSignature:
    def get_input_embeddings(self, input_ids):  # remote code does this
        raise AssertionError("must not be reached")


class _Normal:
    def get_input_embeddings(self):
        return "embeddings"


@pytest.mark.parametrize(
    "model, expected",
    [
        (_CannotAnswer(), None),  # transformers 5 base impl raises
        (_WrongSignature(), None),  # TypeError, not AttributeError
        (object(), None),  # method absent entirely
        (_Normal(), "embeddings"),
    ],
)
def test_embeddings_or_none(model, expected):
    """`hasattr` does not answer the question; only calling does."""
    assert _embeddings_or_none(model, "get_input_embeddings") == expected
