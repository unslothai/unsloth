# Unsloth - 2x faster, 70% less memory LLM finetuning
# Tests for the `finetune_last_n_layers` parity knob (translation helper only,
# no CUDA / real checkpoint); mirrors unsloth-zoo's MLX layers_to_transform path.

from __future__ import annotations

import pytest


def test_get_total_transformer_layers_reads_num_hidden_layers():
    from unsloth.models.vision import _get_total_transformer_layers

    class FakeConfig:
        num_hidden_layers = 18

    class FakeModel:
        config = FakeConfig()

    assert _get_total_transformer_layers(FakeModel()) == 18


def test_get_total_transformer_layers_reads_text_config():
    from unsloth.models.vision import _get_total_transformer_layers

    class TextConfig:
        num_hidden_layers = 24

    class FakeConfig:
        text_config = TextConfig()

    class FakeModel:
        config = FakeConfig()

    # No num_hidden_layers at top level — should fall through to text_config.
    assert _get_total_transformer_layers(FakeModel()) == 24


def test_get_total_transformer_layers_handles_alternative_attr_names():
    from unsloth.models.vision import _get_total_transformer_layers
    for attr in ("n_layer", "n_layers", "num_layers"):
        cfg = type("Cfg", (), {attr: 12})()
        model = type("M", (), {"config": cfg})()
        assert _get_total_transformer_layers(model) == 12


def test_get_total_transformer_layers_returns_none_when_unknown():
    from unsloth.models.vision import _get_total_transformer_layers

    class FakeConfig:
        pass

    class FakeModel:
        config = FakeConfig()

    assert _get_total_transformer_layers(FakeModel()) is None


def test_get_total_transformer_layers_returns_none_for_missing_config():
    from unsloth.models.vision import _get_total_transformer_layers
    class FakeModel:
        pass

    assert _get_total_transformer_layers(FakeModel()) is None


def test_finetune_last_n_layers_signature_present_on_llama_and_vision():
    """Both entry points expose finetune_last_n_layers with default None."""
    import inspect
    from unsloth.models.llama import FastLlamaModel
    from unsloth.models.vision import FastBaseModel

    for cls in (FastLlamaModel, FastBaseModel):
        sig = inspect.signature(cls.get_peft_model)
        assert (
            "finetune_last_n_layers" in sig.parameters
        ), f"{cls.__name__}.get_peft_model missing finetune_last_n_layers"
        assert sig.parameters["finetune_last_n_layers"].default is None, (
            f"{cls.__name__}.get_peft_model: finetune_last_n_layers default "
            f"must be None to preserve historical behavior"
        )
