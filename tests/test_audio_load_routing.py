# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Audio models FastModel failed to load: SDPA-less towers (MusicGen's EnCodec, VibeVoice-ASR's
tokenizers), AudioFlamingo3 / MusicFlamingo routing."""

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from unsloth.models._utils import (  # noqa: E402
    _scoped_sdpa,
    _transformers_supports_attn_impl_mapping,
    resolve_attention_implementation,
    resolve_model_class,
)
from unsloth.models.loader import (  # noqa: E402
    _resolve_audio_lm_auto_model,
    _resolve_non_vlm_auto_model,
    _resolve_omni_auto_model,
    _resolve_speech_seq2seq_auto_model,
)
from unsloth.models.vision import _audio_lm_class  # noqa: E402

try:
    from transformers import AutoModelForImageTextToText as IMAGE_TEXT_CLASS
except ImportError:  # transformers 4.x
    from transformers import AutoModelForVision2Seq as IMAGE_TEXT_CLASS


def _vibevoice_asr():
    config_class = getattr(transformers, "VibeVoiceAsrConfig", None)
    model_class = getattr(transformers, "VibeVoiceAsrForConditionalGeneration", None)
    if config_class is None or model_class is None:
        pytest.skip("this transformers has no VibeVoice-ASR")
    if not _transformers_supports_attn_impl_mapping():
        pytest.skip("this transformers takes no per-sub-config attn_implementation")
    return config_class(), model_class


def test_plain_sdpa_fails_on_a_tower_without_it():
    """Without this the next test would prove nothing on a build where every tower has SDPA."""
    config, model_class = _vibevoice_asr()
    with torch.device("meta"), pytest.raises(ValueError, match = "scaled_dot_product_attention"):
        model_class._from_config(config, attn_implementation = "sdpa")


def test_sdpa_is_scoped_away_from_towers_without_it():
    config, model_class = _vibevoice_asr()
    impl = resolve_attention_implementation(model_class, config, supports_sdpa = True)
    assert isinstance(impl, dict) and impl[""] == "sdpa"
    assert impl["acoustic_tokenizer_encoder_config"] == "eager"
    assert impl.get("text_config", "sdpa") == "sdpa"
    with torch.device("meta"):
        model_class._from_config(config, attn_implementation = impl)


@pytest.mark.parametrize(
    "config_name", ["Qwen2AudioConfig", "LlavaConfig", "Gemma3Config", "LlamaConfig"]
)
def test_models_whose_towers_all_take_sdpa_keep_plain_sdpa(config_name):
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"this transformers has no {config_name}")
    assert _scoped_sdpa(config_class()) == "sdpa"


def _audio_flamingo():
    config_class = getattr(transformers, "AudioFlamingo3Config", None)
    if config_class is None:
        pytest.skip("this transformers has no AudioFlamingo3")
    return config_class()


def test_audio_flamingo_has_no_image_text_omni_or_speech_class():
    """The defect: every class the loader tried before falls through to ImageTextToText, which raises."""
    config = _audio_flamingo()
    assert resolve_model_class(IMAGE_TEXT_CLASS, config) is None
    assert _resolve_omni_auto_model(config) is None
    assert _resolve_speech_seq2seq_auto_model(config) is None


def test_audio_flamingo_routes_to_the_lm_class_that_builds_the_full_model():
    config = _audio_flamingo()
    auto_class = _resolve_audio_lm_auto_model(config)
    assert auto_class is not None
    assert (
        resolve_model_class(auto_class, config)
        is transformers.AudioFlamingo3ForConditionalGeneration
    )
    assert (
        _audio_lm_class(auto_class, config) is not None
    )  # so the processor, not the tokenizer, loads


def test_audio_lm_routing_never_takes_text_or_vision_models():
    for config_name in ("LlamaConfig", "Gemma3Config", "Qwen2VLConfig"):
        config_class = getattr(transformers, config_name, None)
        if config_class is None:
            continue
        assert _resolve_audio_lm_auto_model(config_class()) is None
        assert _audio_lm_class(transformers.AutoModelForCausalLM, config_class()) is None


def test_config_without_architectures_loads_its_speech_seq2seq_class():
    """moonshine-ai/moonshine-streaming-* ship a config with no `architectures`, and no causal-LM class maps it."""
    config_class = getattr(transformers, "MoonshineStreamingConfig", None)
    if config_class is None:
        pytest.skip("this transformers has no Moonshine Streaming")
    config = config_class()
    assert not getattr(config, "architectures", None)
    assert _resolve_non_vlm_auto_model(config) is transformers.AutoModelForSpeechSeq2Seq


@pytest.mark.parametrize(
    "config, expected",
    [
        (lambda: transformers.LlamaConfig(), "AutoModelForCausalLM"),
        (lambda: transformers.WhisperConfig(), "AutoModelForCausalLM"),
        (lambda: transformers.Wav2Vec2Config(), "AutoModelForCausalLM"),
        (lambda: transformers.Wav2Vec2Config(architectures = ["Wav2Vec2ForCTC"]), "AutoModelForCTC"),
    ],
)
def test_non_vlm_routing_keeps_every_other_route(config, expected):
    assert _resolve_non_vlm_auto_model(config()).__name__ == expected
