# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""KL VAEs on diffusers' Decoder (FLUX.2 / FLUX.3 and AutoencoderKL: FLUX.1, Z-Image, HiDream, Lumina-2, Chroma,
SDXL) decode a would-be-tiled latent with whole-map GroupNorm stats (diffusion_vae_strips.full_map_decode, torch
fallback off CUDA): tiles cut from a large latent take their own stats and shift colour (2048 px FLUX.3 Klein: 25.6 dB
vs the untiled decode). Eligibility is read off the structure, so a new family on the same pieces qualifies. CPU, tiny
random VAEs."""

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


def _kl(post_quant: bool = True, seed: int = 0):
    """AutoencoderKL as FLUX.1 / Z-Image / HiDream / Lumina-2 (no post_quant_conv) or SDXL (with one) build it."""
    torch.manual_seed(seed)
    vae = diffusers.AutoencoderKL(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        latent_channels = 4,
        norm_num_groups = 8,
        sample_size = 64,
        use_quant_conv = post_quant,
        use_post_quant_conv = post_quant,
    ).eval()
    with torch.no_grad():
        for m in vae.modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    return vae


@pytest.mark.parametrize("post_quant", [False, True])
@pytest.mark.parametrize("tile_budget", [None, 32 * 32])
def test_tiled_autoencoder_kl_decode_matches_the_untiled_decode(
    roomy, monkeypatch, post_quant, tile_budget
):
    """AutoencoderKL (FLUX.1, Z-Image, HiDream, Lumina-2, Chroma, SDXL) shares FLUX.2's Decoder, so the whole-map
    decode is the untiled one there too; before, its tiles shifted colour the same way."""
    if tile_budget:
        monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: tile_budget)
    vae = _kl(post_quant)
    assert strips.full_map_ok(vae)
    vae.enable_tiling()
    assert vt.install(vae)
    z = _latent(c = 4)
    ref = _decode(vae, z, tiled = False)
    out = _decode(vae, z, tiled = True)
    assert vae.__dict__.get("_unsloth_vae_full_map_calls") == 1
    assert torch.allclose(out, ref, atol = 2e-5, rtol = 0), float((out - ref).abs().max())
    if tile_budget:  # the low-memory tiles it replaces shift colour
        monkeypatch.setenv(FULL_MAP_ENV, "0")
        assert float((_decode(vae, z, tiled = True) - ref).abs().max()) > 1e-2


def test_a_decoder_with_non_power_of_two_widths_decodes_the_whole_map(roomy, monkeypatch):
    """MOSS-Image's FLUX.2-class VAE has a 96 / 192 / 384-channel decoder: the Triton primitives need power-of-two
    channels, the torch fallback does not, so it still decodes the whole map (before, its tiles shifted colour)."""
    monkeypatch.setattr(vt, "decode_tile_budget", lambda vae, z: 32 * 32)
    cls = getattr(diffusers, "AutoencoderKLFlux2", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLFlux2")
    torch.manual_seed(0)
    vae = cls(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        decoder_block_out_channels = (24, 48),
        layers_per_block = 1,
        latent_channels = 8,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()
    with torch.no_grad():
        for m in vae.modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    assert not strips.decoder_ok(vae.decoder) and strips.full_map_ok(vae)
    vae.enable_tiling()
    assert vt.install(vae)
    z = _latent()
    ref = _decode(vae, z, tiled = False)
    out = _decode(vae, z, tiled = True)
    assert vae.__dict__.get("_unsloth_vae_full_map_calls") == 1
    assert torch.allclose(out, ref, atol = 2e-5, rtol = 0), float((out - ref).abs().max())


def test_a_vae_whose_decode_differs_keeps_its_tiles(roomy, monkeypatch):
    """Structure, not names: a KL VAE that does anything else between the latent and the decoder (here a scale) is not
    reproduced by the strips, so it keeps its tiled decode bit for bit."""

    class ScaledKL(diffusers.AutoencoderKL):
        def _decode(
            self,
            z,
            return_dict = True,
        ):
            return super()._decode(z * 2.0, return_dict = return_dict)

    torch.manual_seed(0)
    vae = ScaledKL(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        latent_channels = 4,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()
    assert not strips.full_map_ok(vae)
    assert strips.full_map_ok(_kl())  # the same pieces with the stock decode do


def test_a_renamed_copy_of_the_kl_vae_qualifies(roomy):
    """A new family whose VAE class is a copy of AutoencoderKL under another name (diffusers' "Copied from") gets the
    whole-map decode without being listed anywhere."""

    class NewFamilyVAE(diffusers.AutoencoderKL):
        def _decode(
            self,
            z,
            return_dict = True,
        ):
            if self.use_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.tiled_decode(z, return_dict = return_dict)

            if self.post_quant_conv is not None:
                z = self.post_quant_conv(z)

            dec = self.decoder(z)

            if not return_dict:
                return (dec,)

            return DecoderOutput(sample = dec)

    from diffusers.models.autoencoders.vae import DecoderOutput  # noqa: F401 - the copied body's name

    torch.manual_seed(0)
    vae = NewFamilyVAE(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        latent_channels = 4,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()
    assert strips.full_map_ok(vae)


def _fixture_classes():
    import json
    from pathlib import Path

    data = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "vae_tile_configs.json").read_text(
            encoding = "utf-8"
        )
    )["families"]
    return {name: entry["config"] for name, entry in data.items()}


# VAE classes the whole-map strips do not reproduce yet, with what they need; a family on any other class must qualify.
NOT_YET_EXACT = {
    "AutoencoderKLQwenImage": "causal Conv3d decoder with per-pixel RMS norm: needs the 3D strip decode",
    "AutoencoderKLQwenImage21": "Qwen-Image family decoder: needs the 3D strip decode",
    "AutoencoderKLHunyuanImage": "32x decoder with its own blocks: needs its strip decode",
    "AutoencoderKLMiniMaxH3Image": "decodes with the release's own tiles (a ViT decoder trained at that tile)",
    "AutoencoderKLMiniMaxH3": "decodes with the release's own tiles (a ViT decoder trained at that tile)",
    "AutoencoderKLWan": "video VAE: needs the 3D strip decode",
    "AutoencoderKLLTX2Video": "video VAE: needs the 3D strip decode",
    "AutoencoderKLHunyuanVideo15": "video VAE: needs the 3D strip decode",
}


def test_every_family_vae_decodes_the_whole_map_or_says_why_not():
    """Each family's real VAE config (the seam guard's pinned vae/config.json), built on the meta device: a class the
    rule reads as eligible must be, and any other is listed in NOT_YET_EXACT with the reason. A new family on a
    KL Decoder needs no entry anywhere; one on new pieces fails here until it is either reproduced or listed."""
    seen = {}
    for family, config in _fixture_classes().items():
        name = config["_class_name"]
        cls = getattr(diffusers, name, None)
        if cls is None:
            continue  # this diffusers has no such class: the family cannot load here either
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        try:
            with torch.device("meta"):
                vae = cls.from_config(kwargs)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{family}: {name} did not build from its pinned config: {exc}")
        seen[family] = (name, strips.full_map_ok(vae))
    assert seen, "no family VAE built"
    for family, (name, ok) in seen.items():
        assert ok or name in NOT_YET_EXACT, (family, name)
        assert not (
            ok and name in NOT_YET_EXACT
        ), f"{name} now qualifies: drop it from NOT_YET_EXACT ({family})"
    for name in ("AutoencoderKL", "AutoencoderKLFlux2"):
        assert any(n == name and ok for n, ok in seen.values()), name


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
