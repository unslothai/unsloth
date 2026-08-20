# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from pydantic import ValidationError

from models import TrainingStartRequest


def _request(**overrides):
    fields = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
    }
    fields.update(overrides)
    return TrainingStartRequest(**fields)


def test_lora_targets_default_on():
    request = _request()

    assert request.finetune_language_layers is True
    assert request.finetune_attention_modules is True
    assert request.finetune_mlp_modules is True
    assert request.finetune_vision_layers is False


def test_vision_lora_with_no_targets_is_rejected():
    with pytest.raises(ValidationError, match = "Nothing to train"):
        _request(
            is_dataset_image = True,
            finetune_vision_layers = False,
            finetune_language_layers = False,
            finetune_attention_modules = False,
            finetune_mlp_modules = False,
        )


def test_text_lora_with_no_targets_is_allowed():
    # Only the VLM path reads these four selectors; a text LoRA run builds its
    # adapters from target_modules, so an all-false request still trains.
    request = _request(
        finetune_vision_layers = False,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.finetune_language_layers is False


def test_continued_pretraining_with_no_targets_is_allowed():
    request = _request(
        training_type = "Continued Pretraining",
        finetune_vision_layers = False,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.training_type == "Continued Pretraining"


def test_vision_only_target_is_enough():
    request = _request(
        is_dataset_image = True,
        finetune_vision_layers = True,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.finetune_vision_layers is True


def test_full_finetuning_ignores_targets():
    request = _request(
        training_type = "Full Finetuning",
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.training_type == "Full Finetuning"
