# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Text encoder-decoders load via AutoModelForSeq2SeqLM; multimodal models also mapped there keep their class."""

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models.vision import _is_text_seq2seq_config  # noqa: E402


def _config(name):
    cls = getattr(transformers, name, None)
    if cls is None:
        pytest.skip(f"this transformers has no {name}")
    return cls()


@pytest.mark.parametrize(
    "name",
    ["T5Config", "MT5Config", "BartConfig", "MarianConfig", "PegasusConfig", "T5GemmaConfig"],
)
def test_text_seq2seq_configs_route_to_seq2seq(name):
    config = _config(name)
    assert _is_text_seq2seq_config(config)
    assert type(config) in transformers.AutoModelForSeq2SeqLM._model_mapping


@pytest.mark.parametrize(
    "name",
    [
        "VoxtralConfig",
        "Qwen2AudioConfig",
        "GraniteSpeechConfig",
        "T5Gemma2Config",
        "LlamaConfig",
        "Qwen3Config",
        "Gemma3Config",
        "WhisperConfig",
    ],
)
def test_other_configs_keep_their_route(name):
    assert not _is_text_seq2seq_config(_config(name))


def test_no_config():
    assert not _is_text_seq2seq_config(None)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("T5Config", True),
        ("BartConfig", True),
        ("T5GemmaConfig", True),
        ("T5Gemma2Config", True),
        ("VoxtralConfig", False),
        ("Qwen2AudioConfig", False),
        ("GraniteSpeechConfig", False),
        ("WhisperConfig", False),
        ("LlamaConfig", False),
        ("Gemma3Config", False),
    ],
)
def test_seq2seq_lm_config_picks_lora_task_and_ga_count(name, expected):
    from unsloth.models._utils import _is_seq2seq_lm_config
    assert _is_seq2seq_lm_config(_config(name)) is expected


def test_get_batch_samples_dispatch():
    from types import SimpleNamespace
    from unsloth.models import _utils

    calls = []
    dispatch = _utils._make_seq2seq_aware_get_batch_samples(
        lambda self, *a, **k: calls.append("stock") or "stock"
    )
    assert dispatch.__name__ == "_unsloth_get_batch_samples"
    original = _utils._unsloth_get_batch_samples
    _utils._unsloth_get_batch_samples = lambda self, *a, **k: calls.append("unsloth") or "unsloth"
    try:
        for name, want in (
            ("T5Gemma2Config", "stock"),
            ("LlamaConfig", "unsloth"),
            ("WhisperConfig", "unsloth"),
        ):
            trainer = SimpleNamespace(model = SimpleNamespace(config = _config(name)))
            assert dispatch(trainer, iter([]), 1) == want
    finally:
        _utils._unsloth_get_batch_samples = original


@pytest.mark.parametrize(
    "name, expected",
    [
        ("T5Config", "right"),
        ("BartConfig", "right"),
        ("WhisperConfig", "right"),
        ("SeamlessM4Tv2Config", "right"),
        ("T5Gemma2Config", "right"),
        ("LlamaConfig", "left"),
        ("Qwen3Config", "left"),
        ("Gemma3Config", "left"),
        ("VoxtralConfig", "left"),
        ("Qwen2AudioConfig", "left"),
    ],
)
def test_encoder_decoders_pad_right(name, expected):
    from unsloth.models.vision import _generation_padding_side
    assert _generation_padding_side(_config(name)) == expected


@pytest.mark.parametrize("name", ["SeamlessM4TConfig", "SeamlessM4Tv2Config"])
def test_seamless_has_a_seq2seq_class_and_no_causal_class(name):
    from unsloth.models._utils import _is_seq2seq_lm_config, resolve_model_class

    config = _config(name)
    assert not _is_text_seq2seq_config(config)
    assert _is_seq2seq_lm_config(config)
    assert resolve_model_class(transformers.AutoModelForCausalLM, config) is None
    assert resolve_model_class(transformers.AutoModelForSeq2SeqLM, config).__name__.endswith(
        "ForTextToText"
    )
