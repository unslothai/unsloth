# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for ``_get_local_weight_size_bytes``.

The sizer must charge one copy of the weights and no torch bookkeeping. It used
to sum every ``.safetensors``/``.bin``/``.pt``/``.pth`` under the model dir, so a
repo shipping both formats (root safetensors plus ``original/consolidated.*.pth``)
priced two models, and a full-finetune output dir also charged
``training_args.bin`` / ``optimizer.pt``. The inflated bytes reach
``estimate_fp16_model_size_bytes`` (which prefers the larger of local vs config)
and are then billed again as optimizer + gradient state under full fine-tuning.

Run:
    python -m pytest studio/backend/tests/test_local_weight_size_one_format.py -q
"""

from pathlib import Path

from utils.hardware.hardware import _get_local_weight_size_bytes


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)


def test_dual_format_repo_counts_safetensors_only(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "consolidated.00.pth", 1000)
    _write(tmp_path / "training_args.bin", 500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_full_finetune_output_dir_ignores_bookkeeping(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 4000)
    _write(tmp_path / "model-00002-of-00002.safetensors", 3000)
    _write(tmp_path / "training_args.bin", 500)
    _write(tmp_path / "optimizer.pt", 8000)
    _write(tmp_path / "scheduler.pt", 100)
    _write(tmp_path / "rng_state.pth", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_bin_only_repo_still_sums_shards(tmp_path):
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 4000)
    _write(tmp_path / "pytorch_model-00002-of-00002.bin", 3000)
    _write(tmp_path / "training_args.bin", 500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_nested_checkpoints_still_skipped(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "checkpoint-60" / "model.safetensors", 5000)
    _write(tmp_path / "global_step10" / "model.safetensors", 7000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_diffusers_subfolder_bin_components_are_counted(tmp_path):
    _write(tmp_path / "unet" / "diffusion_pytorch_model.bin", 3440)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.bin", 320)
    _write(tmp_path / "text_encoder" / "pytorch_model.bin", 460)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4220


def test_mixed_format_components_are_all_counted(tmp_path):
    _write(tmp_path / "unet" / "diffusion_pytorch_model.safetensors", 9900)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.bin", 320)
    _write(tmp_path / "text_encoder_2" / "pytorch_model.bin", 9500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 19720


def test_torch_only_tower_beside_safetensors_language_model(tmp_path):
    _write(tmp_path / "model-00001-of-00001.safetensors", 13400)
    _write(tmp_path / "vision_tower" / "pytorch_model.bin", 1700)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 15100


def test_freely_named_torch_checkpoints_are_counted(tmp_path):
    _write(tmp_path / "open_clip" / "open_clip_pytorch_model.bin", 3900)
    _write(tmp_path / "lora" / "pytorch_lora_weights.bin", 150)
    _write(tmp_path / "esm" / "esm2_t33_650M_UR50D.pt", 2600)
    _write(tmp_path / "legacy" / "weights.pth", 1200)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7850


def test_original_copy_counts_when_it_is_the_only_copy(tmp_path):
    _write(tmp_path / "original" / "consolidated.00.pth", 16000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 16000


def test_same_directory_dual_format_still_charges_one_copy(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "pytorch_model.bin", 1000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000
