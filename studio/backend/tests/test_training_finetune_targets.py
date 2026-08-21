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


def test_image_tagged_continued_pretraining_with_no_targets_is_allowed():
    # worker.py takes its `if is_cpt` branch BEFORE the LoRA one and passes only
    # target_modules, so a CPT run never reads these four however its dataset is
    # tagged. Gating the exemption on is_dataset_image rejected this valid config;
    # the case above leaves the flag at its default, so it never covered this.
    request = _request(
        training_type = "Continued Pretraining",
        is_dataset_image = True,
        finetune_vision_layers = False,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.training_type == "Continued Pretraining"
    assert request.is_dataset_image is True


def test_audio_lora_with_no_targets_passes_request_validation():
    # An audio request is ambiguous at this layer: the audio-VLM branch reads all four
    # selectors, the codec/ASR branches ignore them, and only pre_detect's probe separates
    # them. So the request is accepted here and settled in the worker (tests below).
    request = _request(
        is_dataset_audio = True,
        finetune_vision_layers = False,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.is_dataset_audio is True


def test_audio_lora_with_one_target_is_allowed():
    request = _request(
        is_dataset_audio = True,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = False,
        finetune_mlp_modules = True,
    )

    assert request.is_dataset_audio is True


def test_audio_tagged_continued_pretraining_with_no_targets_is_allowed():
    # training_type exempts CPT before the dataset gate, for audio as for image.
    request = _request(
        training_type = "Continued Pretraining",
        is_dataset_audio = True,
        finetune_vision_layers = False,
        finetune_language_layers = False,
        finetune_attention_modules = False,
        finetune_mlp_modules = False,
    )

    assert request.is_dataset_audio is True


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


def test_worker_allows_audio_vlm_with_one_target():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_mlp_modules": True})
    _check_finetune_targets_after_detect(_Trainer(is_audio_vlm = True), config)


def test_worker_allows_vision_only_target():
    from core.training.worker import _check_finetune_targets_after_detect
    config = _config(**{**_ALL_OFF, "finetune_vision_layers": True})
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
    # _pre_detect_training_model's caller funnels exceptions through the incomplete-cache
    # fallback when the model is local-only. This error must not look like a cache artifact
    # error, or a nothing-to-train run would be retried as a corrupt-download instead.
    from core.training.worker import _is_model_cache_artifact_error
    error = ValueError(
        "Nothing to train: enable at least one of finetune_language_layers, "
        "finetune_attention_modules, finetune_mlp_modules, or finetune_vision_layers."
    )

    assert _is_model_cache_artifact_error(error) is False
