# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Training under auto padding-free must never build a KV cache: transformers then drops its packed-sequence
mask and packed documents attend across each other. Trainer __init__ ends in for_inference (restoring
use_cache), so for_training has to disable it again even without gradient checkpointing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
training_utils = pytest.importorskip("unsloth_zoo.training_utils")
if not hasattr(training_utils, "disable_use_cache"):
    pytest.skip("unsloth_zoo predates disable_use_cache", allow_module_level = True)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        use_cache = True,
    )
    return LlamaForCausalLM(config)


def _fast_models():
    from unsloth.models.llama import FastLlamaModel
    from unsloth.models.vision import FastBaseModel
    return [FastLlamaModel, FastBaseModel]


@pytest.mark.parametrize("use_gradient_checkpointing", [False, True, "unsloth"])
@pytest.mark.parametrize("fast_model", range(2))
def test_for_training_disables_the_cache_with_or_without_checkpointing(
    fast_model, use_gradient_checkpointing
):
    cls = _fast_models()[fast_model]
    model = _tiny_llama()
    # Mirrors an Unsloth trainer: prepare disabled it, Trainer __init__ ended in for_inference (restored it).
    training_utils.disable_use_cache(model)
    training_utils.restore_use_cache(model)
    assert model.config.use_cache is True
    try:
        cls.for_training(model, use_gradient_checkpointing = use_gradient_checkpointing)
    except Exception as error:  # for_training arms checkpointing machinery some hosts lack
        if model.config.use_cache is True:
            pytest.skip(f"{cls.__name__}.for_training cannot run here: {error}")
    assert model.config.use_cache is False


@pytest.mark.parametrize("fast_model", range(2))
def test_a_model_never_prepared_still_trains_without_a_cache_and_infers_with_one(fast_model):
    cls = _fast_models()[fast_model]
    model = _tiny_llama()
    assert getattr(model, "_unsloth_use_cache_originals", None) is None
    try:
        cls.for_training(model, use_gradient_checkpointing = False)
    except Exception as error:
        if model.config.use_cache is True:
            pytest.skip(f"{cls.__name__}.for_training cannot run here: {error}")
    assert model.config.use_cache is False
    # Recorded on the way in, so generation gets its cache back.
    training_utils.restore_use_cache(model)
    assert model.config.use_cache is True


class _Trainer:
    args = SimpleNamespace(gradient_accumulation_steps = 1)
    model_accepts_loss_kwargs = True

    def __init__(self):
        self.seen = []

    def _old_compute_loss(self, model, inputs, *args, **kwargs):
        self.seen.append(dict(inputs))
        return torch.tensor(0.0)


def _compute_loss(inputs):
    from unsloth.models._utils import _unsloth_pre_compute_loss

    trainer = _Trainer()
    model = SimpleNamespace(config = SimpleNamespace(model_type = "llama"), training = True)
    _unsloth_pre_compute_loss(trainer, model, inputs, num_items_in_batch = torch.tensor(4))
    return trainer.seen[0]


def test_a_packed_training_batch_runs_without_a_cache():
    inputs = {
        "input_ids": torch.zeros(1, 4, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([2, 2]),
    }
    assert _compute_loss(inputs)["use_cache"] is False
    assert "use_cache" not in inputs, "the collated batch was mutated"


def test_a_padded_training_batch_is_left_alone():
    assert "use_cache" not in _compute_loss({"input_ids": torch.zeros(2, 4, dtype = torch.long)})


def test_an_explicit_use_cache_in_the_batch_wins():
    inputs = {
        "input_ids": torch.zeros(1, 4, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([4]),
        "use_cache": True,
    }
    assert _compute_loss(inputs)["use_cache"] is True
