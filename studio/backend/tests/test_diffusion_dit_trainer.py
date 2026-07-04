# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the flow-matching DiT LoRA trainer (FLUX.1 / Qwen-Image / Z-Image).

CPU-only: cover family resolution, the per-family spec table, the QLoRA prequant
heuristic, the bf16-only guard, and the gated-repo name check. The full training loop is
exercised by the live GPU smokes, not here."""

from __future__ import annotations

import pytest

from core.training.diffusion_dit_trainer import (
    _FLUX_TARGETS,
    _GATED_TRAIN_REPOS,
    _QWEN_TARGETS,
    _SPECS,
    _ZIMAGE_TARGETS,
    _assert_gated_access,
    _repo_is_prequantized,
    _select_lora_targets,
    run_dit_lora_training,
)
from core.training.diffusion_train_common import (
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    family_train_infos,
)


def test_specs_cover_the_three_dit_families():
    assert set(_SPECS) == {"flux.1", "qwen-image", "z-image"}
    # FLUX / Qwen share the added-kv attention target set; Z-Image is single-stream.
    assert "add_q_proj" in _SPECS["flux.1"].lora_targets
    assert "add_q_proj" in _SPECS["qwen-image"].lora_targets
    assert "add_q_proj" not in _SPECS["z-image"].lora_targets
    # Z-Image and Qwen are bf16-only.
    assert _SPECS["z-image"].force_bf16 is True
    assert _SPECS["qwen-image"].force_bf16 is True


def test_select_lora_targets_uses_family_default_for_generic_config():
    # normalized() leaves lora_target_modules empty when a caller doesn't set it, so an empty
    # tuple must resolve to the family's targets (which add the DiT-specific joint-attention
    # projections), not the generic SDXL list.
    assert _select_lora_targets((), _FLUX_TARGETS) == _FLUX_TARGETS
    assert _select_lora_targets((), _QWEN_TARGETS) == _QWEN_TARGETS
    assert _select_lora_targets((), _ZIMAGE_TARGETS) == _ZIMAGE_TARGETS
    # The generic SDXL default is NOT treated as unset here: an explicit list wins.
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _FLUX_TARGETS) == DEFAULT_LORA_TARGETS


def test_select_lora_targets_explicit_override_wins():
    # Any explicit tuple is a deliberate override and must win over the family spec.
    override = ("to_q", "to_k")
    assert _select_lora_targets(override, _FLUX_TARGETS) == override
    # The default request path (config leaving targets unset) reaches the spec.
    cfg = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg.lora_target_modules == ()
    assert (
        _select_lora_targets(cfg.lora_target_modules, _SPECS["flux.1"].lora_targets)
        == _FLUX_TARGETS
    )


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", True),
        ("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", True),
        ("some/model-int4", True),
        ("black-forest-labs/FLUX.1-dev", False),
        ("Tongyi-MAI/Z-Image-Turbo", False),
    ],
)
def test_prequant_heuristic(repo, expected):
    assert _repo_is_prequantized(repo) is expected


def test_zimage_rejects_fp16_before_loading():
    # bf16-only families must refuse an explicit fp16 request up front (no model load).
    cfg = DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image-Turbo",
        data_dir = "does-not-exist",
        output_dir = "o",
        mixed_precision = "fp16",
    )
    with pytest.raises(ValueError, match = "bf16"):
        run_dit_lora_training(cfg)


def test_gated_access_requires_token():
    assert "black-forest-labs/flux.1-dev" in _GATED_TRAIN_REPOS
    # No token -> clear, actionable error before any download.
    with pytest.raises(ValueError, match = "gated"):
        _assert_gated_access("black-forest-labs/FLUX.1-dev", None)
    with pytest.raises(ValueError, match = "gated"):
        _assert_gated_access("black-forest-labs/FLUX.1-dev", "   ")
    # With a token, or for a non-gated repo, it is a no-op.
    _assert_gated_access("black-forest-labs/FLUX.1-dev", "hf_realtoken")
    _assert_gated_access("Tongyi-MAI/Z-Image-Turbo", None)


def test_family_train_infos_lists_dit_families():
    infos = {i["name"]: i for i in family_train_infos()}
    for fam in ("sdxl", "flux.1", "qwen-image", "z-image"):
        assert fam in infos, f"{fam} missing from family_train_infos"
        assert infos[fam]["default_base"]
        assert infos[fam]["base_repos"]
        assert "resolution" in infos[fam]["defaults"]
    # FLUX default base is the gated dev repo; its note flags the license requirement.
    assert infos["flux.1"]["default_base"] == "black-forest-labs/FLUX.1-dev"
    assert "gated" in infos["flux.1"]["vram_note"].lower()
    # Z-Image defaults to the prequant nf4 repo for QLoRA.
    assert "4bit" in infos["z-image"]["default_base"].lower()
