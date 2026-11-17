# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2-class VAEs decode a would-be-tiled latent with whole-map GroupNorm stats (diffusion_vae_strips.full_map_decode,
torch fallback off CUDA): tiles cut from a large latent take their own stats and shift colour (2048 px FLUX.3 Klein:
25.6 dB vs the untiled decode). CPU, tiny random VAEs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_strips as strips  # noqa: E402
from core.inference import diffusion_vae_tiling as vt  # noqa: E402

LATENT = 48  # > the tiny VAEs' 32-latent stock tile, so diffusers tiles it
FULL_MAP_ENV = "UNSLOTH_FLUX_VAE_FULL_MAP"


def _flux2(seed: int = 0):
    cls = getattr(diffusers, "AutoencoderKLFlux2", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLFlux2")
    torch.manual_seed(seed)
    vae = cls(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        layers_per_block = 1,
        latent_channels = 8,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()
    # Random GroupNorm affines and a latent with a strong spatial trend: per-tile stats then differ from the map's.
    with torch.no_grad():
        for m in vae.modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    return vae


def _latent(c: int = 8, side: int = LATENT):
    g = torch.Generator().manual_seed(1)
    ramp = torch.linspace(-2, 2, side).view(1, 1, side, 1)
    return torch.randn(1, c, side, side, generator = g) + ramp


def _decode(vae, z, tiled: bool):
    vae.use_tiling = tiled
    with torch.no_grad():
        return vae.decode(z, return_dict = False)[0]


@pytest.fixture
def roomy(monkeypatch):
    """A host reading with room for the whole-map decode (CPU: the planner's host budget)."""
    monkeypatch.setattr(vt, "_host_free_mib", lambda vae, z: (64 * 1024.0, 1.0))
    monkeypatch.delenv(FULL_MAP_ENV, raising = False)
    monkeypatch.delenv(vt.MAX_TILE_ENV, raising = False)


@pytest.mark.parametrize("tile_budget", [None, 32 * 32])
def test_tiled_flux2_decode_matches_the_untiled_decode(roomy, monkeypatch, tile_budget):
    """Every would-be-tiled FLUX.2 decode equals the untiled one, whatever tiles the planner would pick (a 32-latent
    tile budget: the unfused tiles of a low-memory card, where the whole-map strips still fit)."""
    if tile_budget:
        monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: tile_budget)
    vae = _flux2()
    vae.enable_tiling()
    assert vt.install(vae)
    z = _latent()
    ref = _decode(vae, z, tiled = False)
    out = _decode(vae, z, tiled = True)
    assert torch.allclose(out, ref, atol = 2e-5, rtol = 0), float((out - ref).abs().max())


def test_stock_tiles_shift_the_flux2_decode(roomy, monkeypatch):
    """The defect the whole-map decode removes: diffusers' tiles of the same latent differ from the untiled decode."""
    monkeypatch.setenv(FULL_MAP_ENV, "0")
    monkeypatch.setattr(
        vt, "decode_tile_budget", lambda vae, z: 32 * 32
    )  # a low-memory card's tiles
    vae = _flux2()
    vae.enable_tiling()
    vt.install(vae)
    z = _latent()
    ref = _decode(vae, z, tiled = False)
    out = _decode(vae, z, tiled = True)
    assert "_unsloth_vae_full_map_calls" not in vae.__dict__
    assert float((out - ref).abs().max()) > 1e-2


def test_kill_switch_keeps_the_tiles(roomy, monkeypatch):
    monkeypatch.setenv(FULL_MAP_ENV, "0")
    vae = _flux2()
    vae.enable_tiling()
    assert not strips.full_map_ok(vae)
    assert vt.full_map_decode(vae, _latent()) is None


def test_no_room_keeps_the_tiles(monkeypatch):
    monkeypatch.delenv(FULL_MAP_ENV, raising = False)
    monkeypatch.setattr(vt, "_host_free_mib", lambda vae, z: (1.0, 1.0))
    vae = _flux2()
    vae.enable_tiling()
    assert vt.full_map_decode(vae, _latent()) is None
    monkeypatch.setattr(vt, "_host_free_mib", lambda vae, z: None)
    assert vt.full_map_decode(vae, _latent()) is None


def test_an_untiled_decode_is_untouched(roomy):
    """A latent under the stock tile never reaches the tiled path: bit-identical to diffusers."""
    vae = _flux2()
    z = _latent(side = 24)
    ref = _decode(vae, z, tiled = False)
    vae.enable_tiling()
    vt.install(vae)
    out = _decode(vae, z, tiled = True)
    assert torch.equal(out, ref)
    assert "_unsloth_vae_full_map_calls" not in vae.__dict__


def test_other_vaes_keep_their_tiled_decode(roomy, monkeypatch):
    """Only FLUX.2-class VAEs take the whole-map decode; AutoencoderKL's tiled decode is bit-identical to the one
    without it (the kill switch is main's path)."""
    torch.manual_seed(0)
    kl = diffusers.AutoencoderKL(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        latent_channels = 4,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()
    assert not strips.full_map_ok(kl)
    z = _latent(c = 4)
    kl.enable_tiling()
    vt.install(kl)
    monkeypatch.setenv(vt.MAX_TILE_ENV, "32")
    with_full_map = _decode(kl, z, tiled = True)
    monkeypatch.setenv(FULL_MAP_ENV, "0")
    assert torch.equal(_decode(kl, z, tiled = True), with_full_map)
    monkeypatch.delenv(vt.MAX_TILE_ENV)
    monkeypatch.delenv(FULL_MAP_ENV)
    with_full_map = _decode(kl, z, tiled = True)
    monkeypatch.setenv(FULL_MAP_ENV, "0")
    assert torch.equal(_decode(kl, z, tiled = True), with_full_map)
    assert "_unsloth_vae_full_map_calls" not in kl.__dict__


@pytest.mark.parametrize("in_bias", [False, True])
def test_torch_group_stats_match_group_norm(in_bias):
    norm = torch.nn.GroupNorm(8, 64, eps = 1e-6)
    x = torch.randn(1, 64, 37, 23) * 3 + 1
    bias = torch.randn(64) if in_bias else None
    mean, rstd = strips._stats_torch(x, norm, bias)
    xb = x + bias.view(1, -1, 1, 1) if in_bias else x
    g = xb.reshape(8, -1)
    assert torch.allclose(mean, g.mean(1), atol = 1e-5)
    assert torch.allclose(rstd, torch.rsqrt(g.var(1, unbiased = False) + 1e-6), rtol = 1e-4)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(("info", msg % a))

    def warning(self, msg, *a):
        self.lines.append(("warning", msg % a))

    debug = info


def test_a_peak_just_over_the_budget_keeps_main_s_tiles_and_says_so_once(monkeypatch):
    """Free memory a hair under what the whole-map decode needs: the tiled decode is bit-identical to main's (the kill
    switch path), and one warning names the tiles and the numbers."""
    monkeypatch.delenv(FULL_MAP_ENV, raising = False)
    monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: 32 * 32)
    vae = _flux2()
    vae.enable_tiling()
    log = _Log()
    vt.install(vae, log)
    z = _latent()
    need = strips.full_map_decode_mib(vae, z)
    monkeypatch.setattr(vt, "_host_free_mib", lambda vae, z: (need / vt.FREE_FRACTION - 1.0, 1.0))
    first = _decode(vae, z, tiled = True)
    _decode(vae, z, tiled = True)
    assert "_unsloth_vae_full_map_calls" not in vae.__dict__
    monkeypatch.setenv(FULL_MAP_ENV, "0")
    assert torch.equal(first, _decode(vae, z, tiled = True))
    warned = [m for lvl, m in log.lines if lvl == "warning" and "decodes in tiles" in m]
    assert len(warned) == 1 and "exceeds" in warned[0], log.lines
    # a hair over: the whole map, logged once as engaged
    monkeypatch.delenv(FULL_MAP_ENV)
    monkeypatch.setattr(vt, "_host_free_mib", lambda vae, z: (need / vt.FREE_FRACTION + 1.0, 1.0))
    _decode(vae, z, tiled = True)
    _decode(vae, z, tiled = True)
    assert vae._unsloth_vae_full_map_calls == 2
    assert len([m for lvl, m in log.lines if "decodes the whole map" in m]) == 1


def test_the_mid_block_attention_is_chunked_where_no_fused_kernel_takes_it(roomy, monkeypatch):
    """ROCm / MPS: no fused SDPA takes the VAE's head dim, and a whole-map math SDPA would hold every score (8 GiB per
    bf16 copy at 2048 px). The whole-map decode runs under Studio's chunked mode: chunked, and still exact."""
    from core.inference import diffusion_vae_attn_chunked as chunked

    calls = []
    real = chunked.chunked_sdpa
    monkeypatch.setattr(chunked, "fused_sdpa_available", lambda q, k, v: False)
    monkeypatch.setattr(chunked, "score_budget_bytes", lambda default_mb = 0: 64 * 1024)
    monkeypatch.setattr(chunked, "half_math_enabled", lambda: False, raising = False)
    monkeypatch.setattr(chunked, "_half_math_call", lambda q: False)
    monkeypatch.setattr(chunked, "chunked_sdpa", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: 32 * 32)
    vae = _flux2()
    vae.enable_tiling()
    vt.install(vae)
    z = _latent()
    ref = _decode(vae, z, tiled = False)
    out = _decode(vae, z, tiled = True)
    assert vae._unsloth_vae_full_map_calls == 1 and calls
    assert torch.allclose(out, ref, atol = 2e-5, rtol = 0), float((out - ref).abs().max())


def test_a_failing_kernel_path_still_decodes_the_whole_map(roomy, monkeypatch):
    """The Triton strip kernels raise (a card / Triton build they fail on): the whole map decodes in torch ops instead of
    falling back to colour-shifting tiles, and the next request skips the failing kernels."""
    monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: 32 * 32)
    raised = []
    real = strips._stats

    def kernel_stats(
        x,
        norm,
        in_bias = None,
    ):
        if not strips._portable():
            raised.append(1)
            raise RuntimeError("triton: no kernel image for this device")
        return real(x, norm, in_bias)

    monkeypatch.setattr(strips, "_kernels_ok", lambda decoder, z: True)
    monkeypatch.setattr(strips, "_stats", kernel_stats)
    vae = _flux2()
    vae.enable_tiling()
    vt.install(vae)
    z = _latent()
    ref = _decode(vae, z, tiled = False)
    for _ in range(2):
        out = _decode(vae, z, tiled = True)
        assert torch.allclose(out, ref, atol = 2e-5, rtol = 0), float((out - ref).abs().max())
    assert vae._unsloth_vae_full_map_calls == 2 and len(raised) == 1
