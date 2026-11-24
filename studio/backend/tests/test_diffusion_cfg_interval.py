# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fast CFG (limited-interval CFG) on FLUX.3 Klein base: the guider the pipeline installs, its state reset between
calls, and the route / request surface."""

import types

import pytest

pytest.importorskip("diffusers")

from core.inference.diffusion_flux3 import _Flux3KleinPipelineMixin


class _MP:
    def __init__(self, guider):
        self.guider = guider

    def update_components(self, guider):
        self.guider = guider


def _guider(scale = 3.5, stop = 1.0):
    from diffusers import ClassifierFreeGuidance
    return ClassifierFreeGuidance(guidance_scale = scale, stop = stop)


def _set(mp, scale, stop):
    _Flux3KleinPipelineMixin._set_guidance(types.SimpleNamespace(), mp, scale, stop)


def test_stop_installs_an_interval_guider():
    mp = _MP(_guider())
    _set(mp, 3.5, 0.8)
    assert mp.guider.config.stop == 0.8
    assert mp.guider.config.guidance_scale == 3.5
    # steps 0-39 of 50 run both branches, 40-49 the conditional one only
    mp.guider.set_state(step = 39, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 2
    mp.guider.set_state(step = 40, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 1


def test_default_keeps_full_cfg_and_reverts_an_interval():
    mp = _MP(_guider(stop = 0.8))
    _set(mp, 3.5, None)
    assert mp.guider.config.stop == 1.0
    mp.guider.set_state(step = 49, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 2


def test_unchanged_guider_is_kept():
    g = _guider(stop = 0.8)
    mp = _MP(g)
    _set(mp, 3.5, 0.8)
    assert mp.guider is g


def test_state_is_reset_before_each_call():
    # The previous render ended past the stop: without a reset the next text step would see one condition and
    # skip the negative embeddings the CFG steps need.
    g = _guider(stop = 0.8)
    g.set_state(step = 49, num_inference_steps = 50, timestep = None)
    assert g.num_conditions == 1
    mp = _MP(g)
    _set(mp, 3.5, 0.8)
    assert mp.guider is g
    assert mp.guider.num_conditions == 2


def test_call_refuses_an_out_of_range_stop():
    pipe = types.SimpleNamespace(modular_pipeline = lambda: _MP(_guider()), is_distilled = False)
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match = "guidance_stop"):
            _Flux3KleinPipelineMixin.__call__(
                pipe, prompt = "a sloth", guidance_scale = 3.5, guidance_stop = bad
            )


def test_route_passes_or_refuses_cfg_interval_end():
    from routes.inference import _cfg_interval_kwargs

    def diffusers_generate(*, prompt, cfg_interval_end = None):
        return None

    def native_generate(*, prompt):
        return None

    unset = types.SimpleNamespace(cfg_interval_end = None)
    req = types.SimpleNamespace(cfg_interval_end = 0.8)
    assert _cfg_interval_kwargs(native_generate, unset) == {}
    assert _cfg_interval_kwargs(diffusers_generate, unset) == {}
    assert _cfg_interval_kwargs(diffusers_generate, req) == {"cfg_interval_end": 0.8}
    with pytest.raises(ValueError, match = "Fast CFG"):
        _cfg_interval_kwargs(native_generate, req)


def test_request_model_bounds():
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest

    assert DiffusionGenerateRequest(prompt = "a").cfg_interval_end is None
    assert DiffusionGenerateRequest(prompt = "a", cfg_interval_end = 0.8).cfg_interval_end == 0.8
    for bad in (0.0, 1.01, -1):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "a", cfg_interval_end = bad)
