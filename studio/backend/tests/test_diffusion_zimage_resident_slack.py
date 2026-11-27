# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    METAL_WORKING_SET,
    OFFLOAD_GROUP,
    OFFLOAD_NONE,
    DeviceMemory,
    calibrated_image_activation,
    plan_diffusion_memory,
)

# Dense bf16 Z-Image-Turbo as logged on a Windows RTX 5090 (32 GB): 30861 of 32579 MiB free at plan time.
ZIMAGE = dict(model_dense_mib = 19_571, companion_dense_mib = 7_820, text_encoder_dense_mib = 7_629)
FLAT_HEADROOM = 6_963  # estimate_image_runtime_mib for a turbo variant
RTX5090 = (30_861, 32_579)


def _target():
    return types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)


def _plan(
    free,
    total,
    act,
    kind = "discrete_vram",
    backend = "cuda",
    sizes = ZIMAGE,
    mode = None,
):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = DeviceMemory(backend, backend, kind, free, total),
        runtime_headroom_mib = FLAT_HEADROOM,
        requested_mode = mode,
        calibrated_activation = act,
        **sizes,
    )


@pytest.fixture(params = [False, True], ids = ["eager_tier", "max_tier"])
def act(request):
    return calibrated_image_activation("z-image", max_speed = request.param)


def test_flat_plan_streams_the_text_encoder_on_the_5090():
    plan = _plan(*RTX5090, None)
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_text_encoders is True and plan.stream_transformer is False
    assert plan.estimates["resident_required_mib"] == 28_582
    assert plan.estimates["safe_device_budget_mib"] == 28_813


def test_measured_zimage_stays_fully_resident_on_the_5090(act, monkeypatch):
    for platform in ("win32", "linux"):
        monkeypatch.setattr(dm.sys, "platform", platform)
        plan = _plan(*RTX5090, act)
        assert plan.offload_policy == OFFLOAD_NONE, plan.reasons
        assert plan.stream_text_encoders is False
        assert plan.vae_tiling is False and plan.vae_slicing is False
        # The #14951 reserve and the base overhead are still charged; the slack replaces only the 15% margin.
        assert plan.estimates["safe_device_budget_mib"] == RTX5090[0] - 2_048
        slack = plan.estimates["calibrated_resident_slack_mib"]
        assert slack == int(RTX5090[1] * 0.10)
        need = ZIMAGE["model_dense_mib"] + act.headroom(False) + dm.DEFAULT_BASE_OVERHEAD_MIB
        assert need + slack <= plan.estimates["safe_device_budget_mib"]
        assert need > int(plan.estimates["safe_device_budget_mib"] * 0.85)
        assert "slack" in plan.reasons[-1]


@pytest.mark.parametrize(
    "free, total",
    [
        (22_500, 23_034),  # L4
        (23_000, 24_564),  # RTX 4090 / 3090
        (22_000, 24_564),  # 24 GB card with a desktop on it
        (27_000, 28_672),  # 28 GB: below the slack line
    ],
)
def test_24gb_cards_still_stream_the_text_encoder(act, free, total):
    plan = _plan(free, total, act)
    assert plan.offload_policy == OFFLOAD_GROUP, plan.reasons
    assert plan.stream_text_encoders is True and plan.stream_transformer is False


def test_the_slack_line_is_exact(act):
    total = RTX5090[1]
    slack = int(total * 0.10)
    need = ZIMAGE["model_dense_mib"] + act.headroom(False) + dm.DEFAULT_BASE_OVERHEAD_MIB + slack
    fits = need + 2_048  # free that leaves exactly the reserve after the slack
    assert _plan(fits, total, act).offload_policy == OFFLOAD_NONE
    short = _plan(fits - 1, total, act)
    assert short.offload_policy == OFFLOAD_GROUP and short.stream_text_encoders is True


def test_a_vram_budget_setting_widens_the_slack(act, monkeypatch):
    # A user VRAM Budget that keeps 8 GiB free is the slack when larger, so the 5090 streams again.
    monkeypatch.setattr(dm, "_device_user_budget_reserve_mib", lambda memory, base: 8_192)
    plan = _plan(*RTX5090, act)
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders is True


@pytest.mark.parametrize(
    "family",
    ["qwen-image-2.1", "flux.1", "flux.2-klein", "flux.3-klein", "moss-image", "lumina-2", "sdxl"],
)
@pytest.mark.parametrize("max_speed", [False, True])
def test_other_calibrated_families_keep_the_15_percent_rule(family, max_speed):
    act = calibrated_image_activation(family, max_speed = max_speed)
    assert act is not None and act.slack_resident is False
    for free in range(18_000, RTX5090[1] + 1, 256):
        assert "calibrated_resident_slack_mib" not in _plan(free, RTX5090[1], act).estimates


def test_uncalibrated_families_are_unchanged():
    assert calibrated_image_activation("hidream-i1") is None
    for free, total in (RTX5090, (22_500, 23_034), (46_000, 49_140), (78_000, 81_559)):
        flat = _plan(free, total, None)
        if free == RTX5090[0]:
            assert flat.offload_policy == OFFLOAD_GROUP and flat.stream_text_encoders is True
        assert "calibrated_resident_slack_mib" not in flat.estimates


@pytest.mark.parametrize(
    "kind, backend",
    [
        ("unified_memory", "cuda"),
        ("unified_memory", "mps"),
        (METAL_WORKING_SET, "mps"),
        ("system_memory", "cpu"),
    ],
)
def test_unified_memory_is_unchanged(act, kind, backend):
    for free, total in (RTX5090, (60_000, 65_536), (100_000, 131_072)):
        assert _plan(free, total, act, kind, backend) == _plan(free, total, None, kind, backend)


def test_amd_and_other_vendors_get_no_calibration(monkeypatch):
    import core.inference.diffusion as d

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    fam = types.SimpleNamespace(name = "z-image")
    for backend, vendor in (("cuda", "amd"), ("cuda", None), ("xpu", "intel"), ("mps", "apple")):
        assert (
            d._calibrated_activation(fam, types.SimpleNamespace(backend = backend, vendor = vendor))
            is None
        )


@pytest.mark.parametrize("mode", ["fast", "balanced", "low_vram"])
def test_explicit_modes_are_unchanged(act, mode):
    assert _plan(*RTX5090, act, mode = mode) == _plan(*RTX5090, None, mode = mode)


def test_more_vram_never_leaves_full_residency(act):
    total = RTX5090[1]
    resident = False
    for free in range(20_000, total + 1, 64):
        plan = _plan(free, total, act)
        if resident:
            assert plan.offload_policy == OFFLOAD_NONE, free
        resident = plan.offload_policy == OFFLOAD_NONE
    assert resident
