# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-GPU placement of a dense fp16 FLUX.2 Klein seeded on the default / max tiers.

The tier seed was measured on text-to-image. A reference request (the source image joins the denoiser's sequence
unpriced) is sized from the flat estimate, and on one card the resident DiT groups stream back to pay for it. A
companion or layer split keeps no such groups, so pricing the split from the seed packed the primary with weights a
reference request could never get back: on two T4s a two-image edit at 1024x1024 was refused where the flat placement
tiled the decode and ran (measured on FLUX.2-klein-4B: the reference images add under 1.6 GiB at 1024x1024). The
split is now priced as before the seed for such a family; text-to-image-only families keep the seeded split.
CPU-only: fake cards, loaded sizes stubbed.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm  # noqa: E402
import core.inference.diffusion_multigpu as mg  # noqa: E402
import core.inference.diffusion_layer_split_plan as lsp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_auto_placement import Card, Host, _Pipe, _env  # noqa: E402, F401
from test_diffusion_dense_tier_activation import (  # noqa: E402
    ENVS,
    KLEIN_FLAT,
    KLEIN_LOADED,
    _flat_plan,
)

T4 = [Card(0, 15095, 289), Card(1, 15095, 289)]
TURING_BUSY = [Card(0, 24576, 6119, name = "Quadro RTX 6000"), Card(1, 15095, 289)]
TWO_REFS = 1024 * 1024  # the extra image beside the source, in condition_pixels


@pytest.fixture(autouse = True)
def _no_learned_store(tmp_path, monkeypatch):
    import core.inference.diffusion_measured_peaks as mp
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    mp._cache.update(path = None, mtime = None, data = None)


def _place(
    monkeypatch,
    cards,
    unpriced: bool,
    speed: str = "default",
    tier_seeds: bool = True,
    torchao: bool = False,
):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    if not tier_seeds:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE_TIERS", "0")
    Host(cards).install(monkeypatch)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda _: dict(KLEIN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda _: torchao)
    monkeypatch.setattr(
        lsp,
        "_denoiser_layouts",
        lambda _: {"transformer": lsp.DenoiserLayout(193, (180,) * 40)},
    )
    pipe = _Pipe()
    pipe.transformer = torch.nn.Linear(2, 2, dtype = torch.float16)
    pipe.transformer.dtype = torch.float16
    pipe.components = {"transformer": pipe.transformer}
    pipe._unsloth_dense_tier_seeds = True
    pipe._unsloth_phase_seeds = torchao
    pipe._unsloth_unpriced_references = unpriced
    primary = cards[0]
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", primary.free_mib, primary.total_mib)
    flat = _flat_plan(
        KLEIN_FLAT,
        "flux.2-klein",
        budget_mib = dm._safe_device_budget_mib(memory),
        total_mib = primary.total_mib,
    )
    measured = dm.refine_plan_from_loaded_weights(
        pipe, flat, family = "flux.2-klein", speed_mode = speed
    )
    with mg.selection_scope(mg.MultiGpuSelection(0, (1,), automatic = True)):
        plan = mg.refine_plan_for_multigpu(pipe, measured)
    return measured, plan


def _primary_weights(plan) -> int:
    if plan.layer_split is not None:
        return dict(plan.layer_split.card_weights_mib).get(0, 0)
    assert plan.secondary_ordinal is not None
    return sum(m for name, (m, _) in KLEIN_LOADED.items() if name not in plan.secondary_components)


def _two_reference_verdict(cards, plan) -> str:
    primary = cards[0]
    memory = dm.DeviceMemory(
        "cuda",
        "cuda",
        "discrete_vram",
        primary.free_mib - _primary_weights(plan),
        primary.total_mib,
    )
    return dm.image_activation_verdict(
        device_memory = memory,
        width = 1024,
        height = 1024,
        family = "flux.2-klein",
        condition_pixels = TWO_REFS,
        vae_tile_side = 1024,
    ).action


def _shape(plan):
    split = plan.layer_split
    return (
        plan.secondary_ordinal,
        tuple(plan.secondary_components or ()),
        None if split is None else tuple(split.card_weights_mib),
    )


@pytest.mark.parametrize("cards", [T4, TURING_BUSY], ids = ["2xT4", "turing24_busy+T4"])
def test_a_reference_family_split_is_priced_as_before_the_seed(monkeypatch, cards):
    measured, plan = _place(monkeypatch, cards, unpriced = True)
    _, flat = _place(monkeypatch, cards, unpriced = True, tier_seeds = False)
    # the split matches the flat placement, so a two-image edit runs instead of being refused ...
    assert _two_reference_verdict(cards, flat) != dm.ACTIVATION_REFUSE
    assert _two_reference_verdict(cards, plan) != dm.ACTIVATION_REFUSE
    assert _shape(plan) == _shape(flat)
    # ... while one card still places from the seed (resident groups a reference request can stream back)
    assert measured.estimates["measured_runtime_headroom_mib"] == 3072
    assert measured.estimates["measured_omits_references"] is True
    assert measured.resident_transformer_mib


@pytest.mark.parametrize("cards", [T4, TURING_BUSY], ids = ["2xT4", "turing24_busy+T4"])
def test_a_text_to_image_family_keeps_the_seeded_split(monkeypatch, cards):
    measured, plan = _place(monkeypatch, cards, unpriced = False)
    assert not measured.estimates.get("measured_omits_references")
    _, flat = _place(monkeypatch, cards, unpriced = False, tier_seeds = False)
    # the seed still packs more of the model onto the primary than the flat estimate
    assert _primary_weights(plan) > _primary_weights(flat)


def test_the_eager_tier_is_unchanged(monkeypatch):
    # the eager dense table never sized reference requests from the flat estimate (measured_request_extra_mib)
    measured, plan = _place(monkeypatch, T4, unpriced = True, speed = "off")
    assert not measured.estimates.get("measured_omits_references")
    _, text_only = _place(monkeypatch, T4, unpriced = False, speed = "off")
    assert _shape(plan) == _shape(text_only)


def test_klein_is_the_family_whose_references_go_unpriced():
    import core.inference.diffusion as diffusion
    from core.inference.diffusion_families import _FAMILIES

    by_name = {f.name: f for f in _FAMILIES}
    assert diffusion._references_unpriced(by_name["flux.2-klein"]) is True
    for name in ("flux.1", "z-image"):
        if name in by_name:
            assert diffusion._references_unpriced(by_name[name]) is False
    for name, fam in by_name.items():
        if diffusion.is_flux3_family(fam) or tuple(getattr(fam, "reference_resolutions", ()) or ()):
            assert diffusion._references_unpriced(fam) is False, name
    assert diffusion._references_unpriced(types.SimpleNamespace(name = "x")) is False
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "pipe._unsloth_unpriced_references = _references_unpriced(fam)" in src


def test_a_learned_peak_above_the_flat_estimate_still_prices_the_split(monkeypatch):
    # the reference fallback raises the split's headroom to the flat estimate, never below a larger measured one
    measured, _ = _place(monkeypatch, T4, unpriced = True)
    flat = int(measured.estimates["runtime_headroom_mib"])
    seen = []
    real = lsp.refine_plan_for_layer_split

    def spy(*args, **kwargs):
        seen.append(kwargs["headroom_mib"])
        return real(*args, **kwargs)

    monkeypatch.setattr(lsp, "refine_plan_for_layer_split", spy)
    for learned in (flat + 4096, 3072):
        boosted = dataclasses.replace(
            measured, estimates = {**measured.estimates, "measured_runtime_headroom_mib": learned}
        )
        seen.clear()
        with mg.selection_scope(mg.MultiGpuSelection(0, (1,), automatic = True)):
            mg.refine_plan_for_multigpu(_Pipe(), boosted)
        assert seen and seen[0] == max(learned, flat)


@pytest.mark.parametrize("unpriced", [True, False])
def test_a_phase_seeded_reference_family_is_flagged_like_the_request_guard(monkeypatch, unpriced):
    # a quantized (non-dense) denoiser places from the phase seed, also measured on text-to-image: the request guard
    # sizes its references flat too, so the split must not be priced from that seed either
    measured, _ = _place(monkeypatch, T4, unpriced = unpriced, torchao = True)
    assert measured.estimates["measured_dense_transformer_mib"] is None
    assert bool(measured.estimates.get("measured_omits_references")) is unpriced
