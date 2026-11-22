# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reference tokens are priced at the denoise rate they cost, not at the flat whole-render rate.

FLUX.2 Klein reference images never pass through the VAE decoder: each is resized to at most 1 MP and its 4096 latent
tokens only lengthen the denoiser's sequence. The request guard charged every extra reference at the flat estimator's
8192 MiB per megapixel (``condition_pixel_weight`` 1.0, the untiled-decode rate) and left the primary one unpriced, so a
2048^2 edit with four references on a 24 GB L4 read 30 GB needed (tiled) against 16.5 GB usable and was refused, or with
allow_oversized streamed the whole transformer from the flat 57 GB untiled figure, while it runs resident.

The tables below are measured points; every priced figure must cover its measured peak, and every non-reference
estimate is pinned to the numbers before the change.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference import diffusion as diff
from core.inference.diffusion_conditioning import EDIT_EXTRA_IMAGE_AREA
from core.inference.diffusion_families import _FAMILIES
from core.inference.diffusion_flux3 import REFERENCE_AREA as FLUX3_REFERENCE_AREA
from core.inference.diffusion_memory import DeviceMemory, calibrated_image_activation

MP = 1024 * 1024
GIB = 1024
L4_TOTAL = 22_563
_FAM = {f.name: f for f in _FAMILIES}


def _verdict(
    *,
    free,
    total,
    family,
    cond,
    width = 2048,
    height = 2048,
    calibrated = False,
    measured = None,
):
    return dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        width = width,
        height = height,
        family = family,
        condition_pixels = cond,
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = calibrated,
        measured_activation = measured,
    )


# ---------------------------------------------------------------- non-reference estimates are unchanged
# (flat, tiled, verdict on a 16_865 MiB L4 budget) at 1024^2, 2048^2, 1536x1024, recorded before the change.
T2I_PINS = {
    "flux.1": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "flux.1 schnell": [(6963, 6963, "run"), (27852, 6963, "tile"), (10444, 6963, "run")],
    "flux.2-klein": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "flux.2-klein black-forest-labs/flux.2-klein-9b": [
        (8192, 8192, "run"),
        (32768, 8192, "tile"),
        (12288, 8192, "run"),
    ],
    "flux.2-dev": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "flux.3-klein": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "flux.3-klein-base": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "qwen-image": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "qwen-image-edit": [(11059, 11059, "run"), (44236, 11059, "tile"), (16588, 11059, "tile")],
    "qwen-image-2.1": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "z-image turbo": [(6963, 6963, "run"), (27852, 6963, "tile"), (10444, 6963, "run")],
    "sdxl": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "lumina-2": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "hidream-i1": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "minimax-h3-image": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "moss-image": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
    "chroma": [(8192, 8192, "run"), (32768, 8192, "tile"), (12288, 8192, "run")],
}
_SIZES = ((1024, 1024), (2048, 2048), (1536, 1024))


@pytest.mark.parametrize("hint", sorted(T2I_PINS))
def test_txt2img_estimates_are_pinned(hint):
    got = []
    for w, h in _SIZES:
        flat = dm.estimate_image_runtime_mib(width = w, height = h, family = hint)
        tiled = dm.estimate_tiled_image_runtime_mib(
            width = w, height = h, family = hint, tile_side = dm.DEFAULT_VAE_TILE_SIDE, vae_sliced = True
        )
        action = _verdict(
            free = 16_865 + 2048, total = L4_TOTAL, family = hint, cond = 0, width = w, height = h
        )
        got.append((flat, tiled, action.action))
    assert got == T2I_PINS[hint]


@pytest.mark.parametrize("name", sorted(_FAM))
def test_txt2img_charges_no_condition_pixels(name):
    # no source image: every family's guard charge is zero, so its txt2img verdict is the pinned one above
    assert diff._guard_condition_pixels(_FAM[name], False, 0, None) == 0


def _pre_change_condition_pixels(fam, has_source, extra, ref_resolution):
    """The guard's inline expression before ``reference_area``."""
    w = getattr(fam, "condition_pixel_weight", 1.0)
    if ref_resolution is not None:
        return int((1 + extra) * ref_resolution * ref_resolution * w)
    if diff.is_flux3_family(fam) and has_source:
        return int((1 + extra) * FLUX3_REFERENCE_AREA * w)
    return int(extra * EDIT_EXTRA_IMAGE_AREA * w)


@pytest.mark.parametrize("name", sorted(n for n, f in _FAM.items() if not f.reference_area))
@pytest.mark.parametrize("extra", [0, 1, 3, 9])
@pytest.mark.parametrize("has_source", [False, True])
def test_other_families_condition_charge_is_unchanged(name, extra, has_source):
    fam = _FAM[name]
    res = (tuple(fam.reference_resolutions) or (None,))[0]
    assert diff._guard_condition_pixels(
        fam, has_source, extra, res
    ) == _pre_change_condition_pixels(fam, has_source, extra, res)


def test_only_flux2_klein_prices_references_by_area():
    assert {n for n, f in _FAM.items() if f.reference_area} == {"flux.2-klein"}
    assert _FAM["flux.2-klein"].reference_area == MP
    assert _FAM["flux.2-klein"].condition_pixel_weight == 0.135
    assert _FAM["flux.3-klein"].condition_pixel_weight == 0.11
    assert _FAM["qwen-image-2.1"].condition_pixel_weight == 0.32


def test_flux2_klein_prices_the_primary_reference_too():
    fam = _FAM["flux.2-klein"]
    assert diff._guard_condition_pixels(fam, True, 0, None) == int(MP * 0.135)
    assert diff._guard_condition_pixels(fam, True, 3, None) == int(4 * MP * 0.135)
    assert diff._guard_condition_pixels(fam, False, 3, None) == int(
        3 * EDIT_EXTRA_IMAGE_AREA * 0.135
    )


def test_generate_marks_area_priced_references_priced():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    at = src.index("guard_unpriced_refs = bool(")
    assert 'and not getattr(fam, "reference_area", 0)' in src[at : at + 500]
    call = src.index("guard_condition_pixels = _guard_condition_pixels(")
    assert "fam, init_pil is not None, len(ref_extra), ref_resolution" in src[call : call + 200]


def test_flux2_weight_covers_its_measured_denoise_slope():
    # flux.2-klein's measured denoise (max tier): 1_205 MiB at 1024^2, 3_924 at 2048^2 -> MiB per MP of tokens
    lo, hi = (
        dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.2-klein"][1][1],
        dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.2-klein"][1][4],
    )
    slope = (hi - lo) / 3.0
    per_ref = 8192 * _FAM["flux.2-klein"].condition_pixel_weight
    assert per_ref >= slope * dm._ACTIVATION_MARGIN


# ---------------------------------------------------------------- streaming credit for area-priced references
def _v(
    action,
    needed = 37_191,
    tiled = 8_519,
    overridden = False,
):
    return dm.ImageActivationVerdict(
        action, needed_mib = needed, tiled_needed_mib = tiled, overridden = overridden
    )


@pytest.mark.parametrize(
    "verdict, short, extra, want, asked",
    [
        # tiles: the tiled estimate is what must fit now; 0 short streams nothing
        (_v(dm.ACTIVATION_TILE), 0, 14_303, 0, 8_519),
        (_v(dm.ACTIVATION_TILE), 4_000, 14_303, 4_000, 8_519),
        # never more than the placement's own shortfall
        (_v(dm.ACTIVATION_TILE), 20_000, 14_303, 14_303, 8_519),
        # runs untiled: the untiled estimate
        (_v(dm.ACTIVATION_RUN, needed = 9_000), 100, 5_000, 100, 9_000),
        # refused, overridden or unreadable: unchanged
        (_v(dm.ACTIVATION_REFUSE), 0, 14_303, 14_303, None),
        (_v(dm.ACTIVATION_TILE, overridden = True), 0, 14_303, 14_303, None),
        (_v(dm.ACTIVATION_TILE), None, 14_303, 14_303, 8_519),
        (None, 0, 14_303, 14_303, None),
    ],
)
def test_area_reference_streams_only_the_live_shortfall(
    monkeypatch, verdict, short, extra, want, asked
):
    seen = []

    def fake(
        pipe,
        need,
        device = None,
    ):
        seen.append(need)
        return short

    monkeypatch.setattr(diff, "live_reference_shortfall_mib", fake)
    assert diff._area_reference_extra_mib(object(), extra, verdict, "cuda:0") == want
    assert seen == ([] if asked is None else [asked])


def test_generate_credits_live_memory_only_for_area_priced_references():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    at = src.index("extra_mib = _area_reference_extra_mib(")
    gate = src[src.rindex("if (", 0, at) : at]
    for part in (
        "extra_mib > 0",
        "guard_condition_pixels > 0",
        "not guard_reference_pixels",
        "not layer_count",
        'getattr(fam, "reference_area", 0)',
    ):
        assert part in gate, part
    # before the release, after the measured / reference pricing
    assert (
        src.index("extra_mib = split_aware_request_extra(")
        < at
        < src.index("restore_resident = release_resident_groups(state.pipe, extra_mib, logger)")
    )


# ---------------------------------------------------------------- measured points
# (card, family hint, condition images, measured MiB above the resident weights at the render's highest phase, card
# total MiB, free MiB the guard read). 2048^2 output, every reference 1 MP; Colab, Studio defaults (int8 auto tier).
# FLUX.2 rows: this change's head, resident (A100) or streamed (T4: the peak includes the streamed-group window, an
# upper bound). FLUX.3 rows: origin/main with allow_oversized (pricing unchanged here, kept as a regression floor).
MEASURED = [
    # FLUX.2 klein 9B int8, A100-40: denoise 23.686 GiB alloc over 19.834 resident (transformer + bf16 encoder)
    ("a100", "flux.2-klein", 4, 3_944, 40_442, 16_343),
    # FLUX.2 klein base 9B int8 (CFG 4.0, branches not batched): 23.706 over 19.834
    ("a100", "flux.2-klein", 4, 3_965, 40_442, 16_327),
    # FLUX.2 klein 4B int8, A100-40: decode 13.052 / denoise 12.780 over 9.865
    ("a100", "flux.2-klein", 4, 3_263, 40_442, 27_365),
    # FLUX.2 klein 9B int8, T4 fp16, transformer streamed: denoise 6.945 over 0.193
    ("t4", "flux.2-klein", 4, 6_913, 14_912, 14_458),
    # FLUX.3 klein int8, L4, 4 and 6 references: decode 13.994 / 14.885 over 10.822 / 10.545 (the eager arm)
    ("l4", "flux.3-klein", 4, 3_248, L4_TOTAL, 9_288),
    ("l4", "flux.3-klein", 6, 4_444, L4_TOTAL, 7_782),
]


@pytest.mark.parametrize("card, hint, refs, measured, total, free", MEASURED)
def test_priced_reference_activation_covers_the_measured_peak(
    card, hint, refs, measured, total, free
):
    fam = _FAM[hint.split()[0]]
    cond = diff._guard_condition_pixels(fam, True, refs - 1, None)
    calibrated = hint.startswith("flux.3")
    act = calibrated_image_activation(hint, max_speed = False) if calibrated else None
    tiled = dm.estimate_tiled_image_runtime_mib(
        width = 2048,
        height = 2048,
        family = hint,
        condition_pixels = cond,
        tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        measured = act,
    )
    assert tiled >= measured, (card, hint, refs, tiled, measured)


@pytest.mark.parametrize(
    "card, hint, refs, measured, total, free", [m for m in MEASURED if m[1] == "flux.2-klein"]
)
def test_flux2_reference_requests_that_ran_are_admitted(card, hint, refs, measured, total, free):
    cond = diff._guard_condition_pixels(_FAM["flux.2-klein"], True, refs - 1, None)
    before = _verdict(free = free, total = total, family = hint, cond = 3 * EDIT_EXTRA_IMAGE_AREA)
    after = _verdict(free = free, total = total, family = hint, cond = cond)
    assert after.action != dm.ACTIVATION_REFUSE, after.message
    # the pre-change charge (three extra images at 8192 MiB/MP, the primary unpriced) needed 28_672 MiB tiled
    assert before.tiled_needed_mib == 28_672 and after.tiled_needed_mib == 8_519
