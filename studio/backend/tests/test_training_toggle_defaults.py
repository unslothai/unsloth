# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from models.training import TrainingStartRequest, TokenPreviewRequest
from utils.models.training_defaults import model_training_recommendations


def test_training_and_token_preview_default_to_assistant_targets_and_packing():
    start = TrainingStartRequest(
        model_name = "unlisted/model", training_type = "LoRA/QLoRA", format_type = "auto"
    )
    preview = TokenPreviewRequest(model_name = "unlisted/model")
    assert start.packing is True
    assert start.train_on_completions is True
    assert preview.train_on_completions is True


@pytest.mark.parametrize("value", [True, False])
def test_worker_config_preserves_explicit_training_toggles(monkeypatch, value):
    from core.training import training

    monkeypatch.setattr(training, "get_device", lambda: SimpleNamespace(value = "cuda"))
    request = TrainingStartRequest(
        model_name = "unlisted/model",
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        packing = value,
        train_on_completions = value,
    )
    config = training._build_training_worker_config(request.model_dump())
    assert config["packing"] is value
    assert config["train_on_completions"] is value


def test_internal_worker_config_defaults_match_api_defaults(monkeypatch):
    from core.training import training

    monkeypatch.setattr(training, "get_device", lambda: SimpleNamespace(value = "cuda"))
    config = training._build_training_worker_config({"model_name": "unlisted/model"})
    assert config["packing"] is True
    assert config["train_on_completions"] is True


@pytest.mark.parametrize(
    "flags, enabled",
    [
        ({}, True),
        ({"is_vision": True}, True),
        ({"is_vision": True, "is_audio": True}, True),
        ({"is_audio": True}, False),
        ({"is_embedding": True}, False),
        ({"is_decision": True}, False),
    ],
)
def test_recommendations_ignore_legacy_recipe_opt_outs_but_respect_training_objective(
    flags, enabled
):
    config = {"training": {"packing": False, "train_on_completions": False, "batch_size": 2}}
    defaults = {"is_vision": False, "is_audio": False, "is_embedding": False, "is_decision": False}
    result = model_training_recommendations(config, **{**defaults, **flags})
    assert result["training"]["packing"] is enabled
    assert result["training"]["train_on_completions"] is enabled
    assert result["training"]["batch_size"] == 2
    # Imported YAML stays literal; only the model-recommendation response is normalized.
    assert config["training"]["packing"] is False
    assert config["training"]["train_on_completions"] is False


def test_vision_and_text_diffusion_packing_guards_remain_effective():
    from core.training.dataset_bounds import effective_packing
    from core.training.worker import _apply_diffusion_lm_overrides

    assert effective_packing({"packing": True}, branch_never_packs = True) is False
    config = {"packing": True}
    _apply_diffusion_lm_overrides(SimpleNamespace(diffusion_lm_profile = "dream"), config)
    assert config["packing"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset_streaming": True, "max_steps": 2},
        {"training_type": "Continued Pretraining"},
        {"format_type": "raw"},
        {"is_decision": True},
        {"is_embedding": True},
    ],
)
def test_unsupported_objectives_do_not_enable_an_omitted_completion_option(overrides):
    values = {
        "model_name": "unlisted/model",
        "training_type": "LoRA/QLoRA",
        "format_type": "auto",
        **overrides,
    }
    assert TrainingStartRequest(**values).train_on_completions is False
    # Explicit values are preserved; the existing route/trainer checks remain authoritative.
    assert TrainingStartRequest(**values, train_on_completions = True).train_on_completions is True
