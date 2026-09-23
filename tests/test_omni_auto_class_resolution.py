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
    resolved = _resolve_omni_auto_model(config)
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


def test_nothing_matching_returns_None_rather_than_a_concrete_class():
    """Returning the concrete class the checkpoint names is WRONG: it is in no
    auto mapping, so it leaves the processor set and downgrades to AutoTokenizer."""
    assert _resolve_omni_auto_model(transformers.LlamaConfig()) is None
    assert _resolve_omni_auto_model(object()) is None


def test_every_class_the_resolver_can_return_takes_a_processor():
    """The resolver's answer decides processor selection, so every possible
    answer must be in the set that selects AutoProcessor."""
    import unsloth.models.loader as loader

    processor_classes = _multimodal_auto_classes()
    for name in loader._OMNI_AUTO_CLASS_NAMES:
        auto_class = getattr(transformers, name, None)
        if auto_class is None:
            continue
        assert auto_class in processor_classes, name


def test_speech_seq2seq_is_not_treated_as_a_vision_model():
    """Including it made loading Whisper call
    _construct_vlm_processor_fallback("openai/whisper-tiny", "whisper"): an image
    processor build for an audio model, whose WhisperProcessor has none."""
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


class _NoEmbeddings:
    """A model that cannot answer for its embeddings, like Qwen3-Omni."""

    def get_input_embeddings(self):
        raise NotImplementedError("composite model, no single embedding")

    def get_output_embeddings(self):
        raise NotImplementedError("composite model, no single embedding")


@pytest.mark.parametrize("requested", [True, "auto"])
def test_offload_embedding_declines_a_model_it_cannot_inspect(requested, capsys):
    """Returning the explicit True is WRONG: the caller then calls
    get_input_embeddings() unguarded, failing the load over a VRAM optimisation
    that cannot be applied anyway."""
    from unsloth.models.vision import _resolve_offload_embedding

    assert _resolve_offload_embedding(_NoEmbeddings(), requested) is False
    printed = capsys.readouterr().out
    if requested == "auto":
        # the default declines silently: nobody asked for it
        assert "Not offloading embeddings" not in printed
    else:
        assert "Not offloading embeddings" in printed


def test_omni_reaches_the_vllm_guard_rather_than_the_language_model_path():
    """Without the needs_processor term, is_vlm_config is False for an omni
    checkpoint (its vision config hides under thinker_config), which skips the
    guard and calls load_vllm with is_vision_model=False for an unsupported model."""
    from unsloth.models.vision import VLLM_SUPPORTED_VLM

    config = _omni_config()
    assert not hasattr(config, "vision_config"), "premise: vision lives under thinker_config"
    assert "qwen3_omni_moe" not in VLLM_SUPPORTED_VLM, "premise: vLLM does not support it"

    # what is_vlm_config now computes for it
    resolved = _resolve_omni_auto_model(config)
    is_vlm = resolved in [IMAGE_TEXT_CLASS]
    needs_processor = is_vlm or resolved in _multimodal_auto_classes()
    is_vlm_config = is_vlm or needs_processor or hasattr(config, "vision_config")
    assert is_vlm_config, "must reach the fast_inference guard"


def _tiny(cls):
    from transformers import LlamaConfig

    return cls(
        LlamaConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=8,
            intermediate_size=8,
        )
    )


def _pretrained_base():
    import torch.nn as nn
    from transformers import LlamaConfig, PreTrainedModel

    class Base(PreTrainedModel):
        config_class = LlamaConfig

        def __init__(self, config):
            super().__init__(config)
            self.embed = nn.Embedding(8, 4)

    return Base


def test_a_getter_that_fails_internally_is_not_read_as_a_bad_signature():
    """Skipping it would cost the module its input-gradient hook, so under PEFT
    with gradient checkpointing backward fails or the adapter trains on nothing."""
    Base = _pretrained_base()

    class RaisesInside(Base):
        def get_input_embeddings(self):
            raise TypeError("genuine bug inside a valid zero-arg getter")

    with pytest.raises(TypeError, match="genuine bug inside"):
        _tiny(RaisesInside).enable_input_require_grads()


def test_a_getter_that_cannot_take_zero_arguments_is_skipped():
    """stepfun-ai/Step-3.7-Flash declares get_input_embeddings(self, input_ids)."""
    Base = _pretrained_base()

    class WrongSignature(Base):
        def get_input_embeddings(self, input_ids):
            raise AssertionError("must not be reached")

    _tiny(WrongSignature).enable_input_require_grads()  # must not raise


def test_a_normal_model_still_gets_its_hook():
    """The negative control: neither guard may swallow the ordinary case."""
    Base = _pretrained_base()

    class Normal(Base):
        def get_input_embeddings(self):
            return self.embed

    model = _tiny(Normal)
    model.enable_input_require_grads()
    assert model.embed._forward_hooks, "the input-gradient hook must be registered"
