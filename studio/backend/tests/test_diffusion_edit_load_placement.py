# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An edit-capable load places for a reference's denoise, so the first edit does not hook the resident transformer;
CPU only.

From a Qwen-Image-2.1 int8 edit under a 16 GiB budget: the load kept the 6922 MiB transformer resident and 1488 MiB of
the 8917 MiB encoder, and every edit then released the flat 8192 MiB/MP x 0.32 x 1.15 = 3014 MiB, more than the
encoder held, so the transformer took offload hooks, pinned and streamed mid-render."""

from __future__ import annotations

import pathlib
import types

import pytest

import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm

Q21_LOADED = {
    "transformer": (6922, "dit"),
    "text_encoder": (8959, "text_encoder"),
    "vae": (644, "other"),
}
Q21_FLAT = dict(model_dense_mib = 19630, companion_dense_mib = 12182, text_encoder_dense_mib = 10847)
REF = 1024 * 1024
COND = int(REF * 0.32)
FLAT_EXTRA = int(8192 * 0.32 * dm._MEASURED_PEAK_MARGIN)  # 3014, the logged release


class _Qwen21:
    def __init__(self):
        self.config = types.SimpleNamespace(
            num_layers = 32,
            num_attention_heads = 32,
            attention_head_dim = 128,
            patch_size = 1,
            causal_condition = True,
        )

    def forward(
        self,
        hidden_states,
        kv_cache = None,
        kv_cache_mode = None,
    ):
        return hidden_states


def _pipe():
    return types.SimpleNamespace(
        transformer = _Qwen21(), vae_scale_factor = 16, _unsloth_phase_seeds = True
    )


@pytest.fixture
def q21(tmp_path, monkeypatch):
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    for env in (
        mp.LEARNED_PEAKS_ENV,
        mp.PHASE_PEAKS_ENV,
        dm.MEASURED_ACTIVATION_ENV,
        dm.PARTIAL_RESIDENT_ENV,
        dm.DERIVED_REFERENCE_PRICE_ENV,
        "UNSLOTH_DIFFUSION_EDIT_LOAD_PLACEMENT",
    ):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(Q21_LOADED), raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    return _pipe()


def _plan(budget_mib: int, total_mib: int):
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = memory,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image-2.1"
        ),
        requested_mode = None,
        **Q21_FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


def _refine(pipe, plan):
    return dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "qwen-image-2.1", speed_mode = "default"
    )


def _reserve(plan):
    """The reserve diffusion.py stores from the refined plan."""
    est = plan.estimates
    covered = (
        (int(est["measured_reference_phase_mib"]), int(est["measured_reference_reserve_mib"]))
        if est.get("measured_reference_reserve_mib")
        else None
    )
    return (
        int(est["measured_runtime_headroom_mib"]),
        "qwen-image-2.1",
        "default",
        est.get("measured_dense_transformer_mib"),
        est.get("measured_compute_bytes"),
        est.get("measured_learned_peak_mib"),
        est.get("measured_learned_peak_pixels"),
        est.get("measured_learned_peak_quadratic"),
        est.get("measured_static_seed", True),
        est.get("measured_learned_cfg_off", False),
        est.get("measured_dense_bf16", False),
        est.get("measured_small_host_int8", False),
        bool(est.get("measured_streamed_headroom_mib")),
        est.get("measured_denoise_phase_headroom_mib"),
        covered,
    )


def _edit_extra(pipe, plan, **kw):
    pipe._unsloth_measured_reserve = _reserve(plan)
    args = dict(
        width = 1024,
        height = 1024,
        condition_pixels = COND,
        reference_pixels = REF,
        cfg = False,
        workflow = "edit",
    )
    args.update(kw)
    return dm.measured_request_extra_mib(pipe, **args)


# 24 GiB (the encoder whole beside the transformer), an L4 (most of it), 16 GiB (part of it)
@pytest.mark.parametrize("budget,total", [(21432, 24576), (20000, 23034), (14500, 16376)])
def test_an_edit_needs_no_mid_render_release_on_a_group_load(q21, budget, total):
    plan = _refine(q21, _plan(budget, total))
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_text_encoders
    assert plan.resident_transformer_mib in (None, 6922)
    # the guidance-free 1024 edit: 768 x 2 joint sequence + 2048 MiB condition cache x 1.15 -> 4096
    assert plan.estimates["measured_reference_need_mib"] == 4096
    assert plan.estimates["measured_reference_reserve_mib"] >= 4096
    assert _edit_extra(q21, plan) == 0
    # the reference's room comes out of the encoder, never the transformer
    room = budget - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644 - 6922
    assert int(plan.resident_text_encoder_mib or 0) == min(8959, room - (4096 - 2304))


def test_text_to_image_requests_are_priced_as_before(q21):
    plan = _refine(q21, _plan(14500, 16376))
    q21._unsloth_measured_reserve = _reserve(plan)
    assert dm.measured_request_extra_mib(q21, width = 1024, height = 1024) == 0
    assert dm.measured_request_extra_mib(q21, width = 2048, height = 2048) == 8704 - 2304


def test_a_guidance_edit_is_still_priced_past_the_placement(q21):
    plan = _refine(q21, _plan(14500, 16376))
    # 2 x (1536 + 2355) = 7782 -> 7936 against the 4096 placed
    assert _edit_extra(q21, plan, cfg = None) == 7936 - 4096


def test_kill_switch_keeps_the_text_to_image_placement(q21, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_EDIT_LOAD_PLACEMENT", "0")
    plan = _refine(q21, _plan(14500, 16376))
    room = 14500 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644 - 6922
    assert plan.resident_text_encoder_mib == room
    assert "measured_reference_reserve_mib" not in plan.estimates
    assert _edit_extra(q21, plan) == FLAT_EXTRA > room


def test_the_derived_price_switch_also_turns_it_off(q21, monkeypatch):
    monkeypatch.setenv(dm.DERIVED_REFERENCE_PRICE_ENV, "0")
    plan = _refine(q21, _plan(14500, 16376))
    assert "measured_reference_reserve_mib" not in plan.estimates


def test_other_families_are_untouched(q21):
    plan = _plan(14500, 16376)
    for family in ("qwen-image-2.1-turbo", "flux.1", "qwen-image"):
        refined = dm.refine_plan_from_loaded_weights(q21, plan, family = family, speed_mode = "default")
        assert "measured_reference_reserve_mib" not in refined.estimates


def test_a_streaming_tier_load_is_untouched(q21):
    """8 / 12 GiB stream the transformer: the denoise-phase placement already prices references (#17233)."""
    for budget, total in ((5600, 8188), (9319, 12288)):
        plan = _refine(q21, _plan(budget, total))
        assert "measured_reference_reserve_mib" not in plan.estimates


def test_the_encoders_give_way_down_to_none(q21):
    """Just past the 16 GiB budget the whole encoder streams; the reserve still records what the load covered."""
    plan = _refine(q21, _plan(13407, 16376))
    assert plan.resident_text_encoder_mib is None
    assert plan.estimates["measured_reference_te_cut_mib"] > 0
    assert 0 < _edit_extra(q21, plan) < FLAT_EXTRA


def test_generate_releases_only_the_live_shortfall_for_a_priced_edit():
    src = (pathlib.Path(dm.__file__).parent / "diffusion.py").read_text(encoding = "utf-8")
    at = src.index("reference_need = load_covered_reference_need_mib(")
    hook = src.index("hook_resident_denoiser(state.pipe, guard_target.torch_device, logger)")
    assert at < hook
    assert "extra_mib = min(extra_mib, short)" in src[at:hook]
    stored = src.index("MEASURED_RESERVE_REFERENCE_INDEX: the edit the placement priced")
    assert "measured_reference_reserve_mib" in src[stored : stored + 400]
