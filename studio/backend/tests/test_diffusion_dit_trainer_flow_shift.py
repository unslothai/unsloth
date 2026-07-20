# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the DiT trainer's timestep-shift / CFG-dropout / loss-weighting levers.

CPU-only: cover the flow_shift config resolution (qwen-image defaults to "auto", every
other family stays on the identity 1.0), the exact sigma transform for the auto and
numeric modes, the shifted sampling distribution, and the bell weight table. The full
training loop is exercised by the live GPU smokes, not here."""

from __future__ import annotations

import math

import pytest

from core.training.diffusion_dit_trainer import (
    _bell_loss_weights,
    _gather_sigmas,
    _sample_timesteps,
    _training_sigma_table,
)
from core.training.diffusion_train_common import DiffusionLoraConfig

QWEN_SHIFT_TERMINAL = 0.02


def _qwen_scheduler():
    # The Qwen/Qwen-Image scheduler config: shift=1.0 is SKIPPED at init because
    # use_dynamic_shifting is true, base_shift = max_shift = log 3 (constant inference mu),
    # exponential time shift, terminal stretch to 0.02.
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps = 1000,
        shift = 1.0,
        use_dynamic_shifting = True,
        base_shift = math.log(3.0),
        max_shift = math.log(3.0),
        shift_terminal = QWEN_SHIFT_TERMINAL,
        time_shift_type = "exponential",
    )


def _flux_static_scheduler():
    # A static-shift scheduler (shift baked into sigmas at init, no dynamic shifting).
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(num_train_timesteps = 1000, shift = 3.0)


# ── config resolution ─────────────────────────────────────────────────────────
def test_flow_shift_defaults_per_family():
    qwen = DiffusionLoraConfig(
        base_model = "Qwen/Qwen-Image", data_dir = "d", output_dir = "o"
    ).normalized()
    assert qwen.resolved_family == "qwen-image"
    assert qwen.flow_shift == "auto"
    flux = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev", data_dir = "d", output_dir = "o"
    ).normalized()
    assert flux.flow_shift == 1.0
    zimg = DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image-Turbo", data_dir = "d", output_dir = "o"
    ).normalized()
    assert zimg.flow_shift == 1.0


def test_flow_shift_explicit_values_and_validation():
    cfg = DiffusionLoraConfig(
        base_model = "Qwen/Qwen-Image", data_dir = "d", output_dir = "o", flow_shift = 2.2
    ).normalized()
    assert cfg.flow_shift == 2.2
    # String numerics from the Studio config path coerce; "auto" passes through.
    assert (
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", flow_shift = "3.0"
        )
        .normalized()
        .flow_shift
        == 3.0
    )
    assert (
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", flow_shift = "AUTO"
        )
        .normalized()
        .flow_shift
        == "auto"
    )
    with pytest.raises(ValueError, match = "flow_shift"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", flow_shift = 0.0
        ).normalized()
    with pytest.raises(ValueError, match = "flow_shift"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", flow_shift = "bogus"
        ).normalized()


def test_cfg_dropout_and_weighting_scheme_validation():
    cfg = DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o").normalized()
    assert cfg.cfg_dropout == 0.0
    assert cfg.weighting_scheme == "none"
    on = DiffusionLoraConfig(
        base_model = "b",
        data_dir = "d",
        output_dir = "o",
        cfg_dropout = 0.1,
        weighting_scheme = "bell",
    ).normalized()
    assert on.cfg_dropout == 0.1
    assert on.weighting_scheme == "bell"
    with pytest.raises(ValueError, match = "cfg_dropout"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", cfg_dropout = 1.5
        ).normalized()
    with pytest.raises(ValueError, match = "weighting_scheme"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", weighting_scheme = "sigma_sqrt"
        ).normalized()


def test_config_from_dict_plumbs_the_new_fields():
    from core.training.diffusion_train_common import _config_from_dict

    cfg = _config_from_dict(
        {
            "base_model": "Qwen/Qwen-Image",
            "data_dir": "d",
            "output_dir": "o",
            "flow_shift": "auto",
            "cfg_dropout": 0.05,
            "weighting_scheme": "bell",
        }
    )
    assert cfg.flow_shift == "auto"
    assert cfg.cfg_dropout == 0.05
    assert cfg.weighting_scheme == "bell"


# ── sigma table transforms ────────────────────────────────────────────────────
def test_auto_table_matches_the_exact_qwen_transform():
    import torch

    sched = _qwen_scheduler()
    table = _training_sigma_table(sched, "auto")
    base = sched.sigmas
    # Exponential shift at mu = log 3 with sigma exponent 1 is exp(mu)/(exp(mu) + 1/u - 1)
    # = 3u/(1 + 2u), then the terminal stretch maps the schedule's last sigma to 0.02.
    shifted = 3.0 * base / (1.0 + 2.0 * base)
    scale = (1.0 - shifted[-1]) / (1.0 - QWEN_SHIFT_TERMINAL)
    expected = 1.0 - (1.0 - shifted) / scale
    assert torch.allclose(table, expected, atol = 1e-6)
    # Fixed-point spot checks: sigma 1.0 stays 1.0, the terminal sigma lands on 0.02, and
    # the midpoint u = 0.5 rises to ~0.754 (3u/(1+2u) = 0.75 before the stretch).
    assert abs(float(table[0]) - 1.0) < 1e-6
    assert abs(float(table[-1]) - QWEN_SHIFT_TERMINAL) < 1e-6
    assert abs(float(table[499]) - 0.75427) < 1e-3
    # The table stays a valid descending schedule in (0, 1].
    assert bool((table[:-1] > table[1:]).all())


def test_numeric_table_applies_the_linear_shift():
    import torch

    sched = _qwen_scheduler()
    table = _training_sigma_table(sched, 2.2)
    base = sched.sigmas
    assert torch.allclose(table, 2.2 * base / (1.0 + 1.2 * base), atol = 1e-6)
    # u = 0.5 under shift s maps to s/(s+1).
    assert abs(float(table[499]) - 2.2 / 3.2) < 1e-3


def test_identity_and_static_families_are_untouched():
    # flow_shift 1.0 must return the scheduler's own table object (no numeric drift for
    # FLUX / Z-Image / Krea 2), and "auto" on a static-shift scheduler is a no-op too:
    # its init already baked the shift into sigmas.
    sched = _qwen_scheduler()
    assert _training_sigma_table(sched, 1.0) is sched.sigmas
    static = _flux_static_scheduler()
    assert _training_sigma_table(static, "auto") is static.sigmas
    assert _training_sigma_table(static, 1.0) is static.sigmas


def test_sampled_sigma_distribution_shifts_under_auto():
    import torch

    torch.manual_seed(0)
    sched = _qwen_scheduler()
    auto_table = _training_sigma_table(sched, "auto")
    _, idx = _sample_timesteps(sched, 4096, "cpu")
    base = _gather_sigmas(sched.sigmas, idx, "cpu", torch.float32, 1)
    shifted = _gather_sigmas(auto_table, idx, "cpu", torch.float32, 1)
    # Unshifted logit-normal draws center at 0.5; the mu = log 3 shift + terminal stretch
    # pushes the mass toward high noise (mean ~0.72). Shift raises EVERY sample.
    assert abs(float(base.mean()) - 0.5) < 0.03
    assert float(shifted.mean()) > 0.68
    assert bool((shifted >= base - 1e-6).all())


def test_gather_sigmas_broadcasts_to_ndim():
    import torch

    sched = _qwen_scheduler()
    sig = _gather_sigmas(sched.sigmas, torch.tensor([0, 499, 999]), "cpu", torch.float32, 4)
    assert sig.shape == (3, 1, 1, 1)
    assert abs(float(sig[0].flatten()) - 1.0) < 1e-6


# ── bell weighting ────────────────────────────────────────────────────────────
def test_bell_weights_shape_peak_and_normalization():
    w = _bell_loss_weights(1000)
    assert w.shape == (1000,)
    assert float(w.min()) >= 0.0
    # Peak at mid-schedule, mean 1 so the expected loss scale is unchanged.
    assert int(w.argmax()) == 500
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert float(w[500]) > float(w[0])
    assert float(w[500]) > float(w[999])
