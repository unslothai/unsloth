# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for the generate-time activation guard's tiled look and its per-request override.

Reported from Unsloth Desktop on Windows: Upscale was refused as too big for the card while the
same upscale ran in ComfyUI on the same GPU, and the refusal's only way through was an environment
variable a desktop install has no terminal to set. The refusal priced the VAE decoding the whole
frame at once; ComfyUI decodes in tiles when that does not fit. Measured at 2048x2048 on diffusers
img2img (peak allocated minus resident weights, MiB): Qwen-Image 16,910 untiled against 1,456
denoise / 304 tiled decode; Z-Image-Turbo 9,730 untiled against 1,899 / 2,455; FLUX.1-schnell 9,730
against 1,701 / 2,455; SDXL 19,459 against 645 / 4,908; Qwen-Image-2.1 edit with a 1024 condition
image 26,864 against 4,957 / 473. The estimator said 32,768 for Qwen-Image.
"""

from __future__ import annotations

import pytest

from core.inference import diffusion_memory as dm
from core.inference.diffusion_memory import (
    ACTIVATION_REFUSE,
    ACTIVATION_RUN,
    ACTIVATION_TILE,
    DeviceMemory,
    OVERSIZED_GENERATE_ENV,
    OVERSIZED_GENERATE_SETTING_LABEL,
    estimate_image_runtime_mib,
    estimate_tiled_image_runtime_mib,
    image_activation_shortfall_message,
    image_activation_verdict,
)

_QWEN_HINT = "qwen-image|unsloth/Qwen-Image-2512-GGUF|Qwen/Qwen-Image"
_ZIMAGE_HINT = "z-image|unsloth/Z-Image-Turbo-GGUF|Tongyi-MAI/Z-Image-Turbo"
_Q21_HINT = "qwen-image-2.1|unsloth/Qwen-Image-2.1-GGUF|Qwen/Qwen-Image-2.1"
_Q21_COND_WEIGHT = 0.32  # diffusion_families: qwen-image-2.1 condition_pixel_weight


def _card(gigabytes: float, free_fraction: float = 0.97) -> DeviceMemory:
    total = int(gigabytes * 1024)
    return DeviceMemory("cuda", "cuda", "discrete_vram", int(total * free_fraction), total)


def _verdict(width, height, memory, **kw):
    kw.setdefault("family", _QWEN_HINT)
    return image_activation_verdict(device_memory = memory, width = width, height = height, **kw)


@pytest.fixture(autouse = True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(OVERSIZED_GENERATE_ENV, raising = False)


# -- the case that was reported -----------------------------------------------------------------


@pytest.mark.parametrize("gigabytes", [8, 12, 16])
def test_a_2048_upscale_that_was_refused_now_runs_tiled(gigabytes):
    card = _card(gigabytes)
    # Before: no tiled look, so the untiled 32 GB figure refuses on every consumer card.
    before = _verdict(2048, 2048, card, source_driven = True)
    assert before.action == ACTIVATION_REFUSE
    # After: the loaded VAE tiles (Qwen-Image VAEs tile at 256), so the decode is bounded and the
    # remaining cost is the denoiser, which fits.
    after = _verdict(2048, 2048, card, source_driven = True, vae_tile_side = 256)
    assert after.action == ACTIVATION_TILE
    assert after.message is None
    assert after.tiled_needed_mib < after.needed_mib


@pytest.mark.parametrize(
    "hint,tile", [(_ZIMAGE_HINT, 1024), ("flux|x|black-forest-labs/FLUX.1-dev", 1024)]
)
def test_autoencoderkl_families_run_tiled_on_an_8gb_card(hint, tile):
    # FLUX / Z-Image / SDXL VAEs tile at 1024, so the tile costs what a 1024 frame costs, which is
    # exactly what the load already budgeted.
    v = _verdict(2048, 2048, _card(8), family = hint, source_driven = True, vae_tile_side = tile)
    assert v.action == ACTIVATION_TILE


def test_sizes_that_fit_untiled_are_left_alone():
    # Nothing is tiled that did not need it: a single tile is bit-identical, several are blended,
    # so a request that fits keeps decoding exactly as before.
    for gigabytes in (8, 12, 16, 24, 48, 80):
        assert _verdict(1024, 1024, _card(gigabytes), vae_tile_side = 256).action == ACTIVATION_RUN
    assert _verdict(2048, 2048, _card(80), vae_tile_side = 256).action == ACTIVATION_RUN


# -- what still refuses -------------------------------------------------------------------------


def _ten_2048_references() -> int:
    return int(10 * 2048 * 2048 * _Q21_COND_WEIGHT)


def test_a_request_that_cannot_run_even_tiled_still_refuses_with_an_in_app_action():
    # Qwen-Image-2.1 edit at its largest output with ten 2048 condition images: the condition
    # tokens alone are ~100 GB of working memory by the calibrated weight, so tiling cannot help.
    v = _verdict(
        2400,
        1792,
        _card(16),
        family = _Q21_HINT,
        source_driven = True,
        condition_pixels = _ten_2048_references(),
        vae_tile_side = 256,
    )
    assert v.action == ACTIVATION_REFUSE
    message = v.message
    assert "2400x1792" in message
    assert "even with tiled VAE decoding" in message
    assert "fewer input images" in message
    # The in-app action comes first; the env var is kept, but only as the server spelling.
    assert OVERSIZED_GENERATE_SETTING_LABEL in message
    assert OVERSIZED_GENERATE_ENV in message
    assert message.index(OVERSIZED_GENERATE_SETTING_LABEL) < message.index(OVERSIZED_GENERATE_ENV)
    assert "server" in message[message.index(OVERSIZED_GENERATE_SETTING_LABEL) :]
    # And the number it quotes is the tiled one it was finally decided on.
    assert f"{(v.tiled_needed_mib + dm.DEFAULT_BASE_OVERHEAD_MIB) / 1024:.2f} GB" in message


def test_the_sdpa_math_fallback_gets_no_tiled_relief():
    # Math-only attention materialises the score matrix, which grows with the SQUARE of the token
    # count: there the denoiser, not the VAE, is what overruns, and the linear per-megapixel figure
    # does not hold. Same refusal as before.
    v = _verdict(2048, 2048, _card(16), vae_tile_side = 256, quadratic_attention = True)
    assert v.action == ACTIVATION_REFUSE
    assert "even with tiled VAE decoding" not in v.message


def test_a_vae_that_cannot_tile_keeps_the_old_verdict():
    v = _verdict(2048, 2048, _card(16), vae_tile_side = None)
    assert v.action == ACTIVATION_REFUSE
    assert "even with tiled VAE decoding" not in v.message


# -- the override -------------------------------------------------------------------------------


def test_allow_oversized_turns_a_refusal_into_a_tiled_attempt():
    kw = dict(
        family = _Q21_HINT,
        condition_pixels = _ten_2048_references(),
        vae_tile_side = 256,
    )
    v = _verdict(2400, 1792, _card(16), allow_oversized = True, **kw)
    assert v.action == ACTIVATION_TILE
    assert v.overridden
    assert v.message is None
    # With no tiling available the attempt runs as loaded.
    kw["vae_tile_side"] = None
    v = _verdict(2400, 1792, _card(16), allow_oversized = True, **kw)
    assert v.action == ACTIVATION_RUN and v.overridden


def test_an_override_under_quadratic_attention_still_tiles_the_vae():
    # The tiled estimate is not trusted under the SDPA math fallback, so it cannot lift the refusal,
    # but the attempt the caller asked for should still decode tile by tile.
    v = _verdict(
        2048, 2048, _card(16), vae_tile_side = 256, quadratic_attention = True, allow_oversized = True
    )
    assert v.action == ACTIVATION_TILE and v.overridden
    assert v.tiled_needed_mib is None


def test_the_env_var_still_overrides_for_server_installs(monkeypatch):
    monkeypatch.setenv(OVERSIZED_GENERATE_ENV, "1")
    v = _verdict(2048, 2048, _card(16), vae_tile_side = None)
    assert v.action == ACTIVATION_RUN and v.overridden
    assert (
        image_activation_shortfall_message(
            device_memory = _card(16), width = 2048, height = 2048, family = _QWEN_HINT
        )
        is None
    )


def test_the_override_does_not_change_requests_that_fit():
    v = _verdict(1024, 1024, _card(16), vae_tile_side = 256, allow_oversized = True)
    assert v.action == ACTIVATION_RUN and not v.overridden


def test_raiser_returns_the_verdict_and_raises_only_on_refuse():
    tile = dm.raise_on_image_activation_shortfall(
        device_memory = _card(16), width = 2048, height = 2048, family = _QWEN_HINT, vae_tile_side = 256
    )
    assert tile.action == ACTIVATION_TILE
    with pytest.raises(dm.ImageActivationShortfallError):
        dm.raise_on_image_activation_shortfall(
            device_memory = _card(16), width = 2048, height = 2048, family = _QWEN_HINT
        )


# -- the estimate against what was measured -----------------------------------------------------

# (hint, tile side, side, measured denoise peak, measured tiled VAE peak), MiB, diffusers img2img.
_MEASURED = [
    (_QWEN_HINT, 256, 1024, 369, 275),
    (_QWEN_HINT, 256, 1536, 819, 285),
    (_QWEN_HINT, 256, 2048, 1456, 304),
    (_ZIMAGE_HINT, 1024, 1024, 478, 2434),
    (_ZIMAGE_HINT, 1024, 1536, 1071, 2434),
    (_ZIMAGE_HINT, 1024, 2048, 1899, 2455),
    ("flux|x|black-forest-labs/FLUX.1-schnell", 1024, 2048, 1701, 2455),
    ("sdxl|x|stabilityai/sdxl-turbo", 1024, 2048, 645, 4908),
]


@pytest.mark.parametrize("hint,tile,side,denoise,vae", _MEASURED)
def test_the_tiled_estimate_covers_every_measured_peak(hint, tile, side, denoise, vae):
    est = estimate_tiled_image_runtime_mib(width = side, height = side, family = hint, tile_side = tile)
    assert est >= 1.2 * max(denoise, vae), (hint, side, est)


def test_the_tiled_estimate_covers_the_measured_edit_with_a_condition_image():
    # Qwen-Image-2.1 unified edit, one 1024 condition image, output 1024 / 2048: 3,228 / 4,957 MiB.
    cond = int(1024 * 1024 * _Q21_COND_WEIGHT)
    for side, measured in ((1024, 3228), (2048, 4957)):
        est = estimate_tiled_image_runtime_mib(
            width = side, height = side, family = _Q21_HINT, condition_pixels = cond, tile_side = 256
        )
        assert est >= 1.1 * measured, (side, est)


def test_the_tiled_estimate_never_undercuts_the_untiled_one_at_or_below_a_tile():
    # One tile IS the untiled decode, so below the tile side tiling buys nothing and the two agree
    # on the VAE; the denoise term can only add.
    for side in (256, 512, 1024):
        untiled = estimate_image_runtime_mib(width = side, height = side, family = _ZIMAGE_HINT)
        tiled = estimate_tiled_image_runtime_mib(
            width = side, height = side, family = _ZIMAGE_HINT, tile_side = 1024
        )
        assert tiled >= untiled


def test_the_tiled_estimate_scales_with_batch_and_area():
    one = estimate_tiled_image_runtime_mib(width = 2048, height = 2048, tile_side = 256)
    four = estimate_tiled_image_runtime_mib(width = 2048, height = 2048, batch_size = 4, tile_side = 256)
    assert four >= 4 * dm.DENOISE_MIB_PER_MEGAPIXEL * 4
    assert four > one


# -- the VAE helpers ----------------------------------------------------------------------------


class _Vae:
    def __init__(self, **attrs):
        self.use_tiling = False
        self.use_slicing = False
        self.calls: list[str] = []
        for k, v in attrs.items():
            setattr(self, k, v)

    def enable_tiling(self):
        self.calls.append("enable_tiling")
        self.use_tiling = True

    def disable_tiling(self):
        self.calls.append("disable_tiling")
        self.use_tiling = False

    def enable_slicing(self):
        self.calls.append("enable_slicing")
        self.use_slicing = True

    def disable_slicing(self):
        self.calls.append("disable_slicing")
        self.use_slicing = False


class _NoTileVae:
    pass


def test_vae_tile_side_reads_diffusers_attributes():
    assert dm.vae_tile_side(_Vae(tile_sample_min_size = 1024)) == 1024
    assert dm.vae_tile_side(_Vae(tile_sample_min_height = 256, tile_sample_min_width = 256)) == 256
    assert dm.vae_tile_side(_Vae()) == dm.DEFAULT_VAE_TILE_SIDE
    assert dm.vae_tile_side(_Vae(tile_sample_min_size = (512, 768))) == 768
    assert dm.vae_tile_side(_NoTileVae()) is None
    assert dm.vae_tile_side(None) is None


def test_tiling_is_engaged_for_one_call_and_undone():
    vae = _Vae()

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.vae = vae
    restore = dm.engage_vae_tiling_for_call(pipe)
    assert vae.use_tiling and vae.use_slicing
    restore()
    assert not vae.use_tiling and not vae.use_slicing
    assert vae.calls == ["enable_tiling", "enable_slicing", "disable_tiling", "disable_slicing"]


def test_a_vae_the_load_already_tiled_is_left_tiled():
    # The low-VRAM tiers tile at load: undoing that after the call would change every later decode.
    vae = _Vae()
    vae.use_tiling = vae.use_slicing = True

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.vae = vae
    assert dm.engage_vae_tiling_for_call(pipe) is None
    assert vae.use_tiling and vae.calls == []


def test_a_vae_whose_tiling_fails_reports_it_is_not_tiled():
    # enable_tiling() is best-effort; the caller must learn it failed, not just get a slicing undo.
    class _BrokenTilingVae(_Vae):
        def enable_tiling(self):
            raise RuntimeError("no tiling on this build")

    vae = _BrokenTilingVae()

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.vae = vae
    restore, tiled = dm.engage_vae_tiling(pipe)
    assert tiled is False
    assert vae.use_slicing and not vae.use_tiling
    restore()
    assert not vae.use_slicing
    # Already tiled at load, or tiled now: both count.
    vae = _Vae()
    vae.use_tiling = True
    pipe.vae = vae
    assert dm.engage_vae_tiling(pipe)[1] is True
    pipe.vae = _Vae()
    assert dm.engage_vae_tiling(pipe)[1] is True
