# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 Klein references are priced at their denoise cost; non-reference estimates stay pinned."""

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


# (flat, tiled, verdict on a 16_865 MiB L4 budget) per _SIZES, recorded before the change.
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
    lo, hi = (
        dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.2-klein"][1][1],
        dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.2-klein"][1][4],
    )
    slope = (hi - lo) / 3.0
    per_ref = 8192 * _FAM["flux.2-klein"].condition_pixel_weight
    assert per_ref >= slope * dm._ACTIVATION_MARGIN


def _v(
    action,
    needed = 37_191,
    tiled = 8_519,
    overridden = False,
    budget = None,
):
    return dm.ImageActivationVerdict(
        action,
        needed_mib = needed,
        tiled_needed_mib = tiled,
        overridden = overridden,
        budget_mib = budget,
    )


@pytest.mark.parametrize(
    "verdict, short, extra, want, asked",
    [
        (_v(dm.ACTIVATION_TILE), 0, 14_303, 0, 8_519),
        (_v(dm.ACTIVATION_TILE), 4_000, 14_303, 4_000, 8_519),
        (_v(dm.ACTIVATION_REFUSE), 4_945, 14_303, 4_945, 8_519),
        (_v(dm.ACTIVATION_TILE), 20_000, 14_303, 14_303, 8_519),
        (_v(dm.ACTIVATION_RUN, needed = 9_000, tiled = None), 100, 5_000, 100, 9_000),
        (_v(dm.ACTIVATION_TILE, overridden = True), 0, 14_303, 14_303, None),
        (_v(dm.ACTIVATION_TILE), None, 14_303, 14_303, 8_519),
        (_v(dm.ACTIVATION_TILE, needed = None, tiled = None), 0, 14_303, 14_303, None),
        (None, 0, 14_303, 14_303, None),
        (_v(dm.ACTIVATION_REFUSE, budget = 4_700), 4_945, 14_303, 8_519 + 2_048 - 4_700, 8_519),
        (_v(dm.ACTIVATION_TILE, budget = 30_000), 0, 14_303, 0, 8_519),
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
    assert (
        src.index("extra_mib = split_aware_request_extra(")
        < at
        < src.index("restore_resident = release_resident_groups(state.pipe, extra_mib, logger)")
    )


# (card, hint, condition images, measured MiB above resident weights, total MiB, free MiB); 2048^2, 1 MP refs, int8.
MEASURED = [
    # klein 9B, 9B base (CFG), 4B on A100; 9B streamed on T4
    ("a100", "flux.2-klein", 4, 3_944, 40_442, 16_343),
    ("a100", "flux.2-klein", 4, 3_965, 40_442, 16_327),
    ("a100", "flux.2-klein", 4, 3_263, 40_442, 27_365),
    ("t4", "flux.2-klein", 4, 6_913, 14_912, 14_458),
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
    assert before.tiled_needed_mib == 28_672 and after.tiled_needed_mib == 8_519
