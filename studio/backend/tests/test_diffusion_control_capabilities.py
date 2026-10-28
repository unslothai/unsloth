# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the Images page is told about the loaded model's Guidance, Negative prompt and Live preview controls: each
status field must agree with what the engine actually does with the value."""

from __future__ import annotations

import pytest

from core.inference import diffusion, diffusion_preview, sd_cpp_backend
from core.inference.diffusion_families import _FAMILIES


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_diffusers_negative_threshold_is_where_the_negative_prompt_engages(fam):
    threshold = diffusion._negative_prompt_guidance_threshold(fam)
    if not fam.uses_negative_prompt:
        assert not diffusion._negative_prompt_engaged(fam, 10.0)
    elif threshold is None:
        assert diffusion._negative_prompt_engaged(fam, 0.0)
    else:
        assert not diffusion._negative_prompt_engaged(fam, threshold)
        assert diffusion._negative_prompt_engaged(fam, threshold + 0.5)


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_native_negative_threshold_is_where_sd_cpp_runs_cfg(fam):
    # sd.cpp drops the negative prompt unless _map_guidance gives a cfg above 1.
    threshold = sd_cpp_backend._negative_prompt_guidance_threshold(fam)
    if threshold is None:
        assert fam.name in sd_cpp_backend._EMBEDDED_GUIDANCE_FAMILIES
        return
    assert sd_cpp_backend._map_guidance(fam, threshold)[0] <= 1.0
    assert sd_cpp_backend._map_guidance(fam, threshold + 0.5)[0] > 1.0


def test_flux3_thresholds_are_one():
    by_name = {f.name: f for f in _FAMILIES}
    for name in ("flux.3-klein", "flux.3-klein-base"):
        assert sd_cpp_backend._negative_prompt_guidance_threshold(by_name[name]) == 1.0
    assert diffusion._negative_prompt_guidance_threshold(by_name["flux.3-klein-base"]) == 1.0


def test_only_the_distilled_flux3_graph_ignores_guidance_as_a_family():
    # Flux3DistilledModularPipeline has no guider, so diffusion_flux3._set_guidance drops the value.
    assert [f.name for f in _FAMILIES if not f.uses_guidance] == ["flux.3-klein"]


@pytest.mark.parametrize(
    "family, is_distilled, expected",
    [
        ("flux.3-klein", False, False),
        ("flux.3-klein-base", False, True),
        # Flux2KleinPipeline.do_classifier_free_guidance is False whenever config.is_distilled.
        ("flux.2-klein", True, False),
        ("flux.2-klein", False, True),
        # Krea 2 Turbo's is_distilled only moves the scheduler shift; it still runs CFG.
        ("krea-2", True, True),
    ],
)
def test_guidance_support_follows_the_loaded_checkpoint(family, is_distilled, expected):
    from types import SimpleNamespace

    fam = next(f for f in _FAMILIES if f.name == family)
    state = SimpleNamespace(
        family = fam, pipe = SimpleNamespace(config = SimpleNamespace(is_distilled = is_distilled))
    )
    assert diffusion._guidance_supported(state) is expected


@pytest.mark.parametrize("name", ["flux.3-klein", "flux.3-klein-base", "flux.2-klein"])
def test_flux3_has_a_live_preview_map_like_flux2(name, monkeypatch):
    import torch

    monkeypatch.delenv(diffusion_preview.PREVIEW_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert diffusion_preview.factors_for(name) == diffusion_preview.factors_for("flux.2-klein")
    assert diffusion_preview.preview_available(name, "cuda")
    assert not diffusion_preview.preview_available(name, "cpu")
    monkeypatch.setenv(diffusion_preview.PREVIEW_ENV, "0")
    assert not diffusion_preview.preview_available(name, "cuda")


def test_a_family_without_a_map_has_no_live_preview(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    without = [f.name for f in _FAMILIES if diffusion_preview.factors_for(f.name) is None]
    assert without, "every family has a map; drop this test"
    assert not diffusion_preview.preview_available(without[0], "cuda")
