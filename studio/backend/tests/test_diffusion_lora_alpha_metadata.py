# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import types

import pytest
import torch

pytest.importorskip("diffusers")
from peft import LoraConfig  # noqa: E402
from safetensors import safe_open  # noqa: E402

from core.training import diffusion_dit_trainer as dit  # noqa: E402
from core.training.diffusion_train_common import DEFAULT_LORA_FILENAME  # noqa: E402

_LAYERS = {
    "blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 16),
    "blocks.0.attn.to_q.lora_B.weight": torch.zeros(16, 4),
}


def _adapter():
    return LoraConfig(r = 4, lora_alpha = 8, target_modules = ["to_q"])


@pytest.mark.parametrize("family", sorted(dit._SPECS))
def test_every_dit_family_saves_its_lora_alpha(family, tmp_path):
    spec = dit._SPECS[family]
    assert spec.save_takes_metadata, family
    trained = types.SimpleNamespace(peft_config = {"default": _adapter()})
    try:
        dit._adapter_saver(spec, trained)(None, str(tmp_path), dict(_LAYERS))
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"{family}: this diffusers has no pipeline for it ({exc})")
    with safe_open(str(tmp_path / DEFAULT_LORA_FILENAME), "pt") as f:
        meta = json.loads(f.metadata()["lora_adapter_metadata"])
    assert meta["transformer.r"] == 4 and meta["transformer.lora_alpha"] == 8


def test_sdxl_saves_its_lora_alpha(tmp_path):
    from diffusers import StableDiffusionXLPipeline

    from core.training.diffusion_lora_trainer import _unet_lora_metadata_kwargs

    unet = types.SimpleNamespace(peft_config = {"default": _adapter()})
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory = str(tmp_path),
        unet_lora_layers = dict(_LAYERS),
        safe_serialization = True,
        weight_name = DEFAULT_LORA_FILENAME,
        **_unet_lora_metadata_kwargs(StableDiffusionXLPipeline.save_lora_weights, unet),
    )
    with safe_open(str(tmp_path / DEFAULT_LORA_FILENAME), "pt") as f:
        meta = json.loads(f.metadata()["lora_adapter_metadata"])
    assert meta["unet.lora_alpha"] == 8
