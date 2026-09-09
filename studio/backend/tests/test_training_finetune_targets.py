# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

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


def test_request_layer_does_not_guess_the_branch():
    # Whether these four are read at all depends on the model, which the request cannot see,
    # so every combination is accepted here and settled in the worker after detection.
    for flags in (
        {},
        {"is_dataset_image": True},
        {"is_dataset_audio": True},
        {"is_dataset_image": True, "is_dataset_audio": True},
        {"training_type": "Continued Pretraining", "is_dataset_image": True},
        {"training_type": "Full Finetuning"},
    ):
        request = _request(
            finetune_vision_layers = False,
            finetune_language_layers = False,
            finetune_attention_modules = False,
            finetune_mlp_modules = False,
            **flags,
        )

        assert request.finetune_language_layers is False


# --- worker-level check, after detection has settled which branch the run takes ---


class _Trainer:
    def __init__(
        self,
        is_vlm = False,
        is_audio_vlm = False,
    ):
        self.is_vlm = is_vlm
        self.is_audio_vlm = is_audio_vlm


def _config(**overrides):
    config = {"training_type": "LoRA/QLoRA"}
    config.update(overrides)
    return config


_ALL_OFF = {
    "finetune_vision_layers": False,
    "finetune_language_layers": False,
    "finetune_attention_modules": False,
    "finetune_mlp_modules": False,
}


def test_worker_rejects_audio_vlm_with_no_targets():
    from core.training.worker import _check_finetune_targets_after_detect
    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), _config(**_ALL_OFF))


def test_worker_allows_codec_audio_with_no_targets():
    # csm / snac / whisper / bicodec / dac leave is_audio_vlm False and build adapters from
    # target_modules, so an all-false request is valid and must not be rejected.
    from core.training.worker import _check_finetune_targets_after_detect
    _check_finetune_targets_after_detect(_Trainer(), _config(**_ALL_OFF))


def test_worker_rejects_vision_vlm_with_no_targets():
    from core.training.worker import _check_finetune_targets_after_detect
    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer(is_vlm = True), _config(**_ALL_OFF))


def test_worker_rejects_audio_vlm_with_a_module_type_but_no_layer_family():
    # get_peft_regex's first guard: mlp alone is not enough, some family must be on.
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_mlp_modules": True})

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), config)


def test_worker_allows_audio_vlm_with_a_family_and_a_module_type():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_language_layers": True, "finetune_mlp_modules": True})
    _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), config)


def test_worker_rejects_vision_family_with_no_module_type():
    # get_peft_regex's second guard: a family with neither attention nor mlp still raises,
    # so "at least one of the four" would have been too loose a rule here.
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_vision_layers": True})

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer(is_vlm = True), config)


def test_worker_allows_vision_family_with_a_module_type():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(
        **{**_ALL_OFF, "finetune_vision_layers": True, "finetune_attention_modules": True}
    )
    _check_finetune_targets_after_detect(_Trainer(is_vlm = True), config)


def test_worker_defaults_count_as_selected():
    # An omitted selector defaults on for the three language-side flags, so a config that
    # simply does not mention them must not be read as "nothing selected".
    from core.training.worker import _check_finetune_targets_after_detect
    _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), _config())


def test_worker_exempts_continued_pretraining():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(training_type = "Continued Pretraining", **_ALL_OFF)
    _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), config)


def test_worker_exempts_full_finetuning():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(training_type = "Full Finetuning", **_ALL_OFF)
    _check_finetune_targets_after_detect(_Trainer(is_vlm = True), config)


def test_worker_rejection_is_not_mistaken_for_a_cache_problem():
    # The caller funnels exceptions through the incomplete-cache fallback for a local-only
    # model, so a nothing-to-train run must not read as a corrupt download and get retried.
    from core.training.worker import _is_model_cache_artifact_error
    error = ValueError(
        "Nothing to train: select at least one layer family (finetune_language_layers or "
        "finetune_vision_layers) and at least one module type (finetune_attention_modules "
        "or finetune_mlp_modules)."
    )

    assert _is_model_cache_artifact_error(error) is False


# --- MLX path: selectors are read for text models too, and before any model load ---


def test_mlx_rejects_no_module_types():
    from core.training.worker import _check_mlx_finetune_targets
    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_finetune_targets(_config(**_ALL_OFF))


def test_mlx_rejects_text_run_with_no_module_types():
    # No is_vlm gate on this path: FastMLXModel.get_peft_model is handed the selectors for
    # text models too, so an all-false text run fails there where CUDA would ignore them.
    from core.training.worker import _check_mlx_finetune_targets
    config = _config(**{**_ALL_OFF, "finetune_language_layers": True})

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_finetune_targets(config)


def test_mlx_allows_empty_layer_family_when_a_module_type_is_on():
    # The caller back-fills finetune_language_layers when a module type is selected, so this
    # trains fine and must not be rejected -- the CUDA guard would reject the same config.
    from core.training.worker import _check_mlx_finetune_targets
    config = _config(**{**_ALL_OFF, "finetune_attention_modules": True})
    _check_mlx_finetune_targets(config)


def test_mlx_allows_defaults():
    from core.training.worker import _check_mlx_finetune_targets
    _check_mlx_finetune_targets(_config())


def test_cuda_rejects_empty_layer_family():
    # get_peft_regex's first guard, which the MLX back-fill makes unreachable there.
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_attention_modules": True})

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer(is_vlm = True), config)


def test_cuda_text_run_is_untouched_by_either_guard():
    from core.training.worker import _check_finetune_targets_after_detect
    _check_finetune_targets_after_detect(_Trainer(), _config(**_ALL_OFF))
