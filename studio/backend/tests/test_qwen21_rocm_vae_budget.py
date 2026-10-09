# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference import diffusion_vae_tiling as vt

torch = pytest.importorskip("torch")


@pytest.fixture
def decoder(monkeypatch):
    monkeypatch.delenv(vt.MAX_TILE_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", "7.14")
    vae = type("AutoencoderKLQwenImage21", (), {})()
    vae._unsloth_wide_geometry = (16, 32, 16)
    vae._unsloth_wide_stock_tile = 16
    z = SimpleNamespace(device = SimpleNamespace(type = "cuda"), shape = (1, 64, 1, 96, 96))
    monkeypatch.setattr(vt, "_free_mib", lambda *a, **kw: (12000, 1))
    return vae, z


@pytest.mark.parametrize("side", [64, 96, 128])
def test_rocm_does_not_grow_qwen_tiles_into_unmeasured_workspace(decoder, side):
    vae, z = decoder
    for _ in range(2):
        budget = vt.decode_tile_budget(vae, z)
        assert vt.choose_tiles(side, side, budget) == (32, 32)


@pytest.mark.parametrize("other_path", ["cuda", "cpu", "fused", "other_vae"])
def test_other_decoders_retain_their_adaptive_budget(monkeypatch, decoder, other_path):
    vae, z = decoder
    if other_path == "cuda":
        monkeypatch.setattr(torch.version, "hip", None)
    elif other_path == "cpu":
        z.device.type = "cpu"
    elif other_path == "fused":
        vae._unsloth_vae_fused_installed = True
    else:
        other = type("AutoencoderKLQwenImage", (), {})()
        other.__dict__.update(vae.__dict__)
        vae = other
    assert vt.decode_tile_budget(vae, z) > 32**2


def test_failed_fused_decoder_uses_the_unfused_limit(decoder):
    vae, z = decoder
    vae._unsloth_vae_fused_installed = True
    vae._unsloth_vae_fused_failed = True
    assert vt.decode_tile_budget(vae, z) == 32**2


def test_explicit_tile_override_still_wins(monkeypatch, decoder):
    vae, z = decoder
    monkeypatch.setenv(vt.MAX_TILE_ENV, "64")
    assert vt.decode_tile_budget(vae, z) == 64**2


def test_floor_memory_refusal_is_preserved(monkeypatch, decoder):
    vae, z = decoder
    monkeypatch.setattr(vt, "_free_mib", lambda *a, **kw: (512, 1))
    assert vt.floor_shortfall(vae, z) is not None
    assert vt.decode_tile_budget(vae, z) == 32**2
