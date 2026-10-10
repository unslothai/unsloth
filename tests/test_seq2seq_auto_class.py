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


def test_a_marked_head_is_counted_by_unsloth_zoo(monkeypatch):
    from types import SimpleNamespace
    from unsloth.models import _utils

    calls = []
    dispatch = _utils._make_seq2seq_aware_get_batch_samples(
        lambda self, *a, **k: calls.append("stock") or "stock"
    )
    monkeypatch.setattr(
        _utils,
        "_unsloth_get_batch_samples",
        lambda self, *a, **k: calls.append("unsloth") or "unsloth",
    )

    class MarkedHead:
        _unsloth_counts_unshifted_labels = True

        def __init__(self, name):
            self.config = _config(name)

    class Peft:
        def __init__(self, head):
            self.config = head.config
            self._head = head

        def get_base_model(self):
            return self._head

    zoo = pytest.importorskip("unsloth_zoo.loss_utils")
    if not hasattr(zoo, "counts_unshifted_labels"):
        pytest.skip(reason = "unsloth_zoo predates the unshifted-label marker")
    for name in ("T5Gemma2Config", "WhisperConfig", "LlamaConfig"):
        assert dispatch(SimpleNamespace(model = MarkedHead(name)), iter([]), 1) == "unsloth"
        assert dispatch(SimpleNamespace(model = Peft(MarkedHead(name))), iter([]), 1) == "unsloth"
    unmarked = SimpleNamespace(config = _config("T5Gemma2Config"))
    assert dispatch(SimpleNamespace(model = unmarked), iter([]), 1) == "stock"


def test_an_older_zoo_without_the_marker_keeps_stock_seq2seq(monkeypatch):
    import builtins
    from types import SimpleNamespace
    from unsloth.models import _utils

    real_import = builtins.__import__

    def no_marker(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == "unsloth_zoo.loss_utils" and fromlist and "counts_unshifted_labels" in fromlist:
            raise ImportError("old unsloth_zoo")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_marker)
    head = SimpleNamespace(config = _config("T5Gemma2Config"), _unsloth_counts_unshifted_labels = True)
    assert _utils._head_counts_unshifted_labels(head) is False


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


@pytest.mark.parametrize("name", ["WhisperConfig", "MoonshineConfig", "Speech2TextConfig"])
def test_speech_seq2seq_configs_resolve_without_auto_model(name):
    # FastModel.from_pretrained(whisper_lora) without auto_model used to ask AutoModelForImageTextToText (#2726).
    from unsloth.models.loader import _resolve_speech_seq2seq_auto_model
    assert (
        _resolve_speech_seq2seq_auto_model(_config(name)) is transformers.AutoModelForSpeechSeq2Seq
    )


@pytest.mark.parametrize("name", ["LlamaConfig", "Gemma3Config", "T5Config"])
def test_non_speech_configs_do_not_resolve_to_speech_seq2seq(name):
    from unsloth.models.loader import _resolve_speech_seq2seq_auto_model
    assert _resolve_speech_seq2seq_auto_model(_config(name)) is None
