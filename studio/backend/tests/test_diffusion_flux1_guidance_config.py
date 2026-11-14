# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1 dev / Krea-dev single files assembled on the schnell base keep their guidance embedder."""

from __future__ import annotations

import json

import pytest

from core.inference.diffusion_content import inspect_checkpoint
from core.inference.diffusion_families import (
    detect_family,
    transformer_config_overrides_for,
    transformer_variant_differs_from_base,
)

_GUIDED = {"guidance_embeds": True}
_UNGUIDED = {"guidance_embeds": False}
_SCHNELL = "black-forest-labs/FLUX.1-schnell"
_DEV = "black-forest-labs/FLUX.1-dev"


@pytest.mark.parametrize(
    "identifiers, expected",
    [
        (("flux.1-dev", "my-model.gguf", "/models/unet", _SCHNELL), _GUIDED),
        (("flux.1-dev", "flux1-schnell-fp8.safetensors", "/models", _SCHNELL), _GUIDED),
        (("flux.1-schnell", "flux1-dev-Q4_K_M.gguf", "/models", _SCHNELL), _UNGUIDED),
        ((None, "flux1-dev-Q4_K_M.gguf", "/models", _SCHNELL), _GUIDED),
        ((None, "flux1_krea_dev_fp8_scaled.safetensors", "/models", _SCHNELL), _GUIDED),
        ((None, "FLUX.1-Krea-dev.safetensors", "/models", _SCHNELL), _GUIDED),
        ((None, "flux1-schnell-Q4_K_S.gguf", "unsloth/FLUX.1-schnell-GGUF", _SCHNELL), _UNGUIDED),
        ((None, None, "unsloth/FLUX.1-dev-GGUF", _DEV), _GUIDED),
        (("flux.1-schnell", "my-model.gguf", "/models", _DEV), _UNGUIDED),
        ((None, "model.gguf", "/models/unet", _SCHNELL), _UNGUIDED),
    ],
)
def test_flux1_transformer_config_per_variant(identifiers, expected):
    fam = detect_family("flux.1")
    assert fam.base_repo == _SCHNELL
    assert transformer_config_overrides_for(fam, *identifiers) == expected


def test_flux1_variant_mismatch_detection_is_unchanged():
    fam = detect_family("flux.1")
    assert transformer_variant_differs_from_base(fam, _SCHNELL, "flux1-dev-Q4_K_M.gguf")
    assert not transformer_variant_differs_from_base(fam, _SCHNELL, "flux1-schnell-Q4_K_S.gguf")
    assert not transformer_variant_differs_from_base(fam, None, "flux1-dev-Q4_K_M.gguf")


_TINY = dict(
    patch_size = 1,
    in_channels = 64,  # what the header classifier reads as plain FLUX.1 (not Fill / Depth)
    num_layers = 1,
    num_single_layers = 1,
    attention_head_dim = 8,
    num_attention_heads = 2,
    joint_attention_dim = 16,
    pooled_projection_dim = 8,
    axes_dims_rope = [2, 2, 4],
)


def _forward(model, torch, guidance):
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        return model(
            hidden_states = torch.randn(1, 4, 64, generator = gen),
            encoder_hidden_states = torch.randn(1, 3, 16, generator = gen),
            pooled_projections = torch.randn(1, 8, generator = gen),
            timestep = torch.tensor([0.5]),
            img_ids = torch.zeros(4, 3),
            txt_ids = torch.zeros(3, 3),
            # FluxPipeline passes guidance only when the config enables the embedder.
            guidance = torch.tensor([guidance]) if model.config.guidance_embeds else None,
        ).sample


@pytest.mark.parametrize("file_guided", [True, False])
def test_the_config_follows_the_file_not_the_base(tmp_path, file_guided):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    save_file = pytest.importorskip("safetensors.torch").save_file

    base = tmp_path / "base"
    (base / "transformer").mkdir(parents = True)
    (base / "transformer" / "config.json").write_text(
        json.dumps(
            {"_class_name": "FluxTransformer2DModel", "guidance_embeds": not file_guided, **_TINY}
        )
    )
    torch.manual_seed(0)
    src = diffusers.FluxTransformer2DModel(guidance_embeds = file_guided, **_TINY)
    path = tmp_path / "renamed.safetensors"
    save_file({k: v.contiguous() for k, v in src.state_dict().items()}, str(path))

    variant = inspect_checkpoint(str(path)).variant
    assert variant == ("flux.1-dev" if file_guided else "flux.1-schnell")
    fam = detect_family("flux.1")
    overrides = transformer_config_overrides_for(
        fam, variant, path.name, str(tmp_path), _SCHNELL if file_guided else _DEV
    )
    model = diffusers.FluxTransformer2DModel.from_single_file(
        str(path),
        config = str(base),
        subfolder = "transformer",
        torch_dtype = torch.float32,
        **overrides,
    )
    assert model.config.guidance_embeds == file_guided
    for (name, got), want in zip(model.state_dict().items(), src.state_dict().values()):
        assert torch.equal(got, want), name
    if file_guided:
        assert not torch.allclose(_forward(model, torch, 1.0), _forward(model, torch, 3.5))
