# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from types import SimpleNamespace

import pytest

import unsloth  # noqa: F401
from peft import LoraConfig
from unsloth.models._utils import reject_alora
from unsloth.models.llama import FastLlamaModel
from unsloth.models.vision import FastBaseModel


def _model(**extra):
    config = LoraConfig(r = 8, target_modules = ["q_proj"], task_type = "CAUSAL_LM", **extra)
    return SimpleNamespace(peft_config = {"default": config})


ALORA = dict(alora_invocation_tokens = [151644, 77091, 198])


def test_alora_is_rejected():
    with pytest.raises(NotImplementedError, match = "alora_invocation_tokens"):
        reject_alora(_model(**ALORA))


def test_alora_in_any_adapter_is_rejected():
    model = _model()
    model.peft_config["second"] = _model(**ALORA).peft_config["default"]
    with pytest.raises(NotImplementedError):
        reject_alora(model)


@pytest.mark.parametrize(
    "model", [_model(), _model(use_dora = True), SimpleNamespace(), SimpleNamespace(peft_config = None)]
)
def test_other_models_pass(model):
    reject_alora(model)


@pytest.mark.parametrize(
    "patch",
    [FastLlamaModel.patch_peft_model, FastBaseModel.post_patch_model],
    ids = ["patch_peft_model", "post_patch_model"],
)
def test_patch_entry_points_reject_alora(patch):
    with pytest.raises(NotImplementedError, match = "alora_invocation_tokens"):
        patch(_model(**ALORA))


def test_prewrapped_peft_model_is_rejected():
    from peft import get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    model = get_peft_model(LlamaForCausalLM(config), _model(**ALORA).peft_config["default"])
    with pytest.raises(NotImplementedError, match = "alora_invocation_tokens"):
        FastLlamaModel.get_peft_model(model, r = 8, target_modules = ["q_proj"])
