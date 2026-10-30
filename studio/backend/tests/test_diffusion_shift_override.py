# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request Shift override: the ComfyUI node mapping per family, the schedule it builds, validation."""

import math

import pytest

from core.inference.diffusion_families import _FAMILIES, comfy_shift_node_for
from core.inference.diffusion_flow_shift import (
    COMFY_FLUX_BASE_SHIFT,
    COMFY_FLUX_MAX_SHIFT,
    SHIFT_NODE_FLUX,
    SHIFT_NODE_SD3,
    comfy_model_sampling_flux_mu,
    requested_flow_shift,
    scheduler_with_flow_shift,
    shift_control,
)


# ComfyUI comfy/model_sampling.py: time_snr_shift (ModelSamplingDiscreteFlow, set by ModelSamplingSD3 /
# ModelSamplingAuraFlow) and flux_time_shift (ModelSamplingFlux), transcribed.
def _comfy_time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _comfy_flux_time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# comfy_extras/nodes_model_advanced.py ModelSamplingFlux.patch, transcribed.
def _comfy_model_sampling_flux_shift(max_shift, base_shift, width, height):
    x1 = 256
    x2 = 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    return (width * height / (8 * 8 * 2 * 2)) * mm + b


# Node each ComfyUI workflow uses for shift: native ModelSampling per model type, but
# Qwen-Image/Edit/Layered patch ModelSamplingAuraFlow.
EXPECTED_NODES = {
    "flux.1": SHIFT_NODE_FLUX,
    "flux.1-kontext": SHIFT_NODE_FLUX,
    "flux.2-klein": SHIFT_NODE_FLUX,
    "flux.2-dev": SHIFT_NODE_FLUX,
    "flux.3-klein": SHIFT_NODE_FLUX,
    "flux.3-klein-base": SHIFT_NODE_FLUX,
    "krea-2": SHIFT_NODE_FLUX,
    "qwen-image-2.1": SHIFT_NODE_FLUX,
    "qwen-image": SHIFT_NODE_SD3,
    "qwen-image-edit": SHIFT_NODE_SD3,
    "qwen-image-layered": SHIFT_NODE_SD3,
    "z-image": SHIFT_NODE_SD3,
    "lumina-2": SHIFT_NODE_SD3,
    "hunyuanimage-2.1": SHIFT_NODE_SD3,
    "hidream-i1": SHIFT_NODE_SD3,
    "moss-image": SHIFT_NODE_SD3,
    # Not offered: Ideogram 4 runs its own mu/std preset, SDXL is not flow matching.
    "ideogram-4": None,
    "sdxl": None,
}


def test_every_family_has_its_comfyui_shift_node():
    assert {f.name: f.comfy_shift_node for f in _FAMILIES} == EXPECTED_NODES


@pytest.mark.parametrize(
    "ident, node",
    [
        (
            "black-forest-labs/FLUX.1-schnell",
            SHIFT_NODE_SD3,
        ),  # ComfyUI FluxSchnell is ModelType.FLOW
        ("flux1-schnell-Q4_K_M.gguf", SHIFT_NODE_SD3),
        ("black-forest-labs/FLUX.1-dev", SHIFT_NODE_FLUX),
        ("black-forest-labs/FLUX.1-Krea-dev", SHIFT_NODE_FLUX),
        ("someone/my-flux-finetune", SHIFT_NODE_FLUX),
    ],
)
def test_flux1_checkpoint_variants(ident, node):
    fam = next(f for f in _FAMILIES if f.name == "flux.1")
    assert comfy_shift_node_for(fam, ident) == node


@pytest.mark.parametrize("hi, lo", [(1.15, 0.5), (3.0, 0.5), (1.0, 1.0), (0.0, 2.0)])
@pytest.mark.parametrize("w, h", [(1024, 1024), (512, 768), (1344, 768), (64, 64)])
def test_flux_mu_is_comfyui_model_sampling_flux(hi, lo, w, h):
    assert comfy_model_sampling_flux_mu(hi, lo, w, h) == pytest.approx(
        _comfy_model_sampling_flux_shift(hi, lo, w, h), abs = 1e-12
    )


def test_flux_mu_matches_diffusers_calculate_shift():
    """Same line as diffusers' FluxPipeline mu at its default anchors (image_seq_len = w*h/256)."""
    pytest.importorskip("diffusers")
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    for w, h in ((1024, 1024), (768, 1344)):
        seq = (w // 16) * (h // 16)
        assert comfy_model_sampling_flux_mu(1.15, 0.5, w, h) == pytest.approx(
            calculate_shift(seq, 256, 4096, 0.5, 1.15), abs = 1e-9
        )


def test_unset_requests_nothing():
    for node in (None, SHIFT_NODE_SD3, SHIFT_NODE_FLUX):
        assert requested_flow_shift(node, width = 1024, height = 1024) is None


def test_sd3_node_is_the_static_shift():
    assert requested_flow_shift(SHIFT_NODE_SD3, shift = 3.1) == 3.1


def test_flux_node_resolves_mu_at_the_output_size():
    got = requested_flow_shift(
        SHIFT_NODE_FLUX, base_shift = 0.5, max_shift = 1.15, width = 1024, height = 1024
    )
    assert got == pytest.approx(math.exp(1.15))
    only_max = requested_flow_shift(SHIFT_NODE_FLUX, max_shift = 2.0, width = 512, height = 512)
    assert only_max == pytest.approx(
        math.exp(_comfy_model_sampling_flux_shift(2.0, COMFY_FLUX_BASE_SHIFT, 512, 512))
    )
    only_base = requested_flow_shift(SHIFT_NODE_FLUX, base_shift = 0.8, width = 512, height = 512)
    assert only_base == pytest.approx(
        math.exp(_comfy_model_sampling_flux_shift(COMFY_FLUX_MAX_SHIFT, 0.8, 512, 512))
    )


@pytest.mark.parametrize(
    "node, kwargs, needle",
    [
        (None, {"shift": 3.0}, "flow-matching"),
        (SHIFT_NODE_SD3, {"base_shift": 0.5}, "single Shift"),
        (SHIFT_NODE_SD3, {"shift": 0.0}, "above 0"),
        (SHIFT_NODE_SD3, {"shift": 101.0}, "above 0"),
        (SHIFT_NODE_SD3, {"shift": float("nan")}, "above 0"),
        (SHIFT_NODE_FLUX, {"shift": 3.0}, "base / max"),
        (SHIFT_NODE_FLUX, {"max_shift": -1.0}, "between 0"),
        (SHIFT_NODE_FLUX, {"base_shift": 89.0, "max_shift": 89.0}, "no usable"),
        (SHIFT_NODE_FLUX, {"base_shift": 0.0, "max_shift": 100.0}, "no usable"),
    ],
)
def test_invalid_requests_are_refused(node, kwargs, needle):
    with pytest.raises(ValueError, match = needle):
        requested_flow_shift(node, width = 1024, height = 1024, **kwargs)


@pytest.mark.parametrize(
    "hi, lo, w, h",
    [
        (80.0, 80.0, 1024, 1024),
        (100.0, 0.0, 4096, 4096),
        (100.0, 0.0, 512, 512),
        (100.0, 0.0, 128, 128),
        (0.0, 100.0, 1024, 1216),
        (0.0, 100.0, 1024, 1056),
    ],
)
def test_extreme_flux_shift_is_refused_or_builds_a_sane_schedule(hi, lo, w, h):
    resident = _flow_euler(shift = 3.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 1.15)
    try:
        shift = requested_flow_shift(
            SHIFT_NODE_FLUX, base_shift = lo, max_shift = hi, width = w, height = h
        )
    except ValueError as exc:
        assert "no usable" in str(exc)
        return
    built = scheduler_with_flow_shift(resident, shift)
    built.set_timesteps(8, device = "cpu")
    sigmas = built.sigmas.tolist()
    assert all(0.0 <= x <= 1.0 + 1e-4 for x in sigmas), sigmas
    assert all(a >= b for a, b in zip(sigmas, sigmas[1:])), sigmas


def _flow_euler(**config):
    diffusers = pytest.importorskip("diffusers")
    return diffusers.FlowMatchEulerDiscreteScheduler(**config)


@pytest.mark.parametrize(
    "config",
    [
        dict(shift = 3.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 1.15),
        dict(
            shift = 1.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 0.9, shift_terminal = 0.02
        ),
        dict(shift = 6.0, use_dynamic_shifting = False),
    ],
)
@pytest.mark.parametrize("shift", [1.0, 3.1, 6.0])
def test_built_schedule_is_comfyui_time_snr_shift(config, shift):
    import numpy as np

    resident = _flow_euler(**config)
    built = scheduler_with_flow_shift(resident, shift)
    unchanged = not config["use_dynamic_shifting"] and shift == config["shift"]
    assert (built is resident) == unchanged
    assert resident.config.get("shift") == config["shift"]  # the loaded scheduler is not touched
    steps = 8
    t = np.linspace(1.0, 1.0 / steps, steps)
    # The pipelines pass this grid (FluxPipeline / QwenImagePipeline); mu is ignored once dynamic shifting is off.
    built.set_timesteps(sigmas = t, mu = 0.123, device = "cpu")
    got = built.sigmas.double().numpy()[:-1]
    want = np.array([_comfy_time_snr_shift(shift, x) for x in t])
    assert np.abs(got - want).max() < 1e-6


@pytest.mark.parametrize("hi, lo", [(1.15, 0.5), (3.0, 0.5)])
def test_flux_node_schedule_is_comfyui_flux_time_shift(hi, lo):
    import numpy as np

    resident = _flow_euler(shift = 3.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 1.15)
    shift = requested_flow_shift(
        SHIFT_NODE_FLUX, base_shift = lo, max_shift = hi, width = 768, height = 1024
    )
    built = scheduler_with_flow_shift(resident, shift)
    steps = 6
    t = np.linspace(1.0, 1.0 / steps, steps)
    built.set_timesteps(sigmas = t, mu = 9.0, device = "cpu")
    mu = _comfy_model_sampling_flux_shift(hi, lo, 768, 1024)
    want = np.array([_comfy_flux_time_shift(mu, 1.0, x) for x in t])
    assert np.abs(built.sigmas.double().numpy()[:-1] - want).max() < 1e-6


def test_flow_shift_schedulers_take_flow_shift():
    """HiDream ships UniPC with ``flow_shift``: the override sets that key."""
    diffusers = pytest.importorskip("diffusers")
    resident = diffusers.UniPCMultistepScheduler(
        prediction_type = "flow_prediction", use_flow_sigmas = True, flow_shift = 3.0
    )
    built = scheduler_with_flow_shift(resident, 6.0)
    assert built.config.flow_shift == 6.0 and resident.config.flow_shift == 3.0


def test_scheduler_without_flow_shift_is_refused():
    diffusers = pytest.importorskip("diffusers")
    with pytest.raises(ValueError, match = "no flow shift"):
        scheduler_with_flow_shift(diffusers.EulerDiscreteScheduler(), 3.0)


def test_shift_control_reports_the_node_and_defaults():
    static = _flow_euler(shift = 3.0, use_dynamic_shifting = False)
    dynamic = _flow_euler(shift = 1.0, use_dynamic_shifting = True)
    assert shift_control(SHIFT_NODE_SD3, static) == {"node": SHIFT_NODE_SD3, "shift": 3.0}
    assert shift_control(SHIFT_NODE_SD3, dynamic) == {"node": SHIFT_NODE_SD3, "shift": None}
    assert shift_control(SHIFT_NODE_FLUX, dynamic) == {
        "node": SHIFT_NODE_FLUX,
        "base_shift": 0.5,
        "max_shift": 1.15,
    }
    assert shift_control(None, static) is None
    diffusers = pytest.importorskip("diffusers")
    assert shift_control(SHIFT_NODE_SD3, diffusers.EulerDiscreteScheduler()) is None


def test_route_refuses_shift_on_an_engine_without_it():
    from types import SimpleNamespace

    from routes.inference import _shift_kwargs

    def native_generate(*, prompt, sampler = None):
        return None

    def diffusers_generate(
        *,
        prompt,
        shift = None,
        base_shift = None,
        max_shift = None,
    ):
        return None

    unset = SimpleNamespace(shift = None, base_shift = None, max_shift = None)
    assert _shift_kwargs(native_generate, unset) == {}
    assert _shift_kwargs(diffusers_generate, unset) == {}
    req = SimpleNamespace(shift = 3.0, base_shift = None, max_shift = None)
    assert _shift_kwargs(diffusers_generate, req) == {
        "shift": 3.0,
        "base_shift": None,
        "max_shift": None,
    }
    with pytest.raises(ValueError, match = "diffusers engine"):
        _shift_kwargs(native_generate, req)


def test_request_model_bounds():
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest

    assert DiffusionGenerateRequest(prompt = "x").shift is None
    assert DiffusionGenerateRequest(prompt = "x", shift = 3.1).shift == 3.1
    for bad in ({"shift": 0.0}, {"shift": 101.0}, {"max_shift": -0.1}, {"base_shift": 200.0}):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "x", **bad)


def test_flux_node_defaults_to_the_resolved_fixed_mu():
    """A load that resolved ComfyUI's fixed mu (Kontext 1.15, Qwen-Image-2.1 0.69) reports base = max = mu, so
    a request that leaves one field out starts from it, and sending the defaults back changes nothing."""
    from core.inference.diffusion_flow_shift import flux_mu_shift

    resolved = _flow_euler(shift = flux_mu_shift(0.69), use_dynamic_shifting = False)
    control = shift_control(SHIFT_NODE_FLUX, resolved)
    assert control["base_shift"] == pytest.approx(0.69) and control["max_shift"] == pytest.approx(
        0.69
    )
    same = requested_flow_shift(
        SHIFT_NODE_FLUX,
        base_shift = control["base_shift"],
        max_shift = control["max_shift"],
        width = 1344,
        height = 768,
    )
    assert scheduler_with_flow_shift(resolved, same) is resolved
    only_max = requested_flow_shift(
        SHIFT_NODE_FLUX, max_shift = 2.0, width = 1024, height = 1024, defaults = control
    )
    assert only_max == pytest.approx(
        math.exp(_comfy_model_sampling_flux_shift(2.0, 0.69, 1024, 1024))
    )


def test_dynamic_flux_reports_its_config_line():
    krea2_like = _flow_euler(shift = 1.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 1.15)
    assert shift_control(SHIFT_NODE_FLUX, krea2_like)["max_shift"] == 1.15
    qwen21_like = _flow_euler(shift = 1.0, use_dynamic_shifting = True, base_shift = 0.5, max_shift = 0.9)
    assert shift_control(SHIFT_NODE_FLUX, qwen21_like)["max_shift"] == 0.9


def test_the_resolved_shift_is_not_rebuilt():
    """Sending the value the load resolved keeps the loaded scheduler object (no rebuild, identical render)."""
    resolved = _flow_euler(shift = 3.1, use_dynamic_shifting = False)
    assert scheduler_with_flow_shift(resolved, 3.1) is resolved
    assert scheduler_with_flow_shift(resolved, 3.2) is not resolved


def test_parameters_text_carries_the_shift_the_run_used():
    """The A1111 parameters text names a requested static shift, so pasting it back restores Shift."""
    from core.inference import image_gallery as gallery

    meta = {
        "prompt": "a cat",
        "width": 64,
        "height": 64,
        "steps": 4,
        "guidance": 3.5,
        "seed": 1,
        "model": "m",
    }
    assert "Shift:" not in gallery._params_text(meta)
    assert ", Shift: 3.1" in gallery._params_text({**meta, "shift": 3.1})
    assert (
        "shift" not in gallery._params_text({**meta, "base_shift": 0.5, "max_shift": 1.15}).lower()
    )
