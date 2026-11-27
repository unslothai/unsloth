# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A denoise-phase placement prices a reference image from its token count, not the flat 8192 MiB/MP allowance; CPU only.

Numbers from Qwen-Image-2.1 int8 edits on an A100 under an 8 GiB budget: 768 MiB denoise reserve, 2304 MiB whole-render
reserve, one 1024 reference (4096 tokens), condition K/V cache 2061 MiB, learned edit peak 3271-3447 MiB."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm

QUANT = "torchao:Int8Tensor"
PHASE = 768
REF = 1024 * 1024
# condition_pixels the guard passes for that reference (Qwen-Image-2.1's condition_pixel_weight)
COND = int(REF * 0.32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "peaks.json"
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(path))
    for env in (
        mp.LEARNED_PEAKS_ENV,
        mp.PHASE_PEAKS_ENV,
        dm.MEASURED_ACTIVATION_ENV,
        dm.DENOISE_PHASE_RESIDENCY_ENV,
        dm.DERIVED_REFERENCE_PRICE_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA A100-SXM4-40GB")
    return path


class _Qwen21:
    def __init__(self, causal = True):
        self.config = types.SimpleNamespace(
            num_layers = 32,
            num_attention_heads = 32,
            attention_head_dim = 128,
            patch_size = 1,
            causal_condition = causal,
        )

    def forward(
        self,
        hidden_states,
        kv_cache = None,
        kv_cache_mode = None,
    ):
        return hidden_states


class _NoCache(_Qwen21):
    def forward(self, hidden_states):
        return hidden_states


def _pipe(
    phase = PHASE,
    family = "qwen-image-2.1",
    transformer = None,
    identity = None,
):
    reserve = (
        2304,
        family,
        "default",
        None,
        None,
        None,
        None,
        None,
        True,
        False,
        False,
        False,
        False,
        phase,
    )
    return types.SimpleNamespace(
        _unsloth_measured_reserve = reserve,
        _unsloth_phase_seeds = True,
        _unsloth_peak_identity = identity,
        transformer = transformer if transformer is not None else _Qwen21(),
        vae_scale_factor = 16,
    )


def _edit(pipe, **kw):
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


def _flat_extra(phase = PHASE) -> int:
    allowance = int(8192 * COND / float(1024 * 1024) * dm._MEASURED_PEAK_MARGIN)
    return 2304 + allowance - phase


def test_kv_cache_size_matches_the_logged_release():
    """32 blocks x K and V x 4096 dims x bf16 per token; 4096 reference tokens = 2048 MiB (logged: 2061 with the prompt)."""
    per_token = dm._prefix_kv_bytes_per_token(_pipe(), 2)
    assert per_token == 32 * 2 * 32 * 128 * 2
    assert dm._reference_pixels_per_token(_pipe()) == 256
    assert (REF // 256) * per_token >> 20 == 2048


def test_one_1024_reference_is_priced_from_its_tokens(store):
    # (768 x 2 joint sequence + 2048 MiB cache x 1.15), rounded up to 256: 4096, against the 768 denoise reserve
    assert _edit(_pipe()) == 4096 - PHASE
    assert _flat_extra() == 4550
    assert _edit(_pipe()) < _flat_extra()


def test_the_derived_need_covers_the_measured_edit_peak():
    need = dm.derived_reference_need_mib(
        _pipe(),
        family = "qwen-image-2.1",
        phase_mib = PHASE,
        width = 1024,
        height = 1024,
        reference_pixels = REF,
        cfg = False,
    )
    # learned edit peak (window included) and the condition cache with the margin
    assert need >= 3447 and need >= int(2061 * dm._MEASURED_PEAK_MARGIN)


def test_kill_switch_keeps_the_flat_allowance(store, monkeypatch):
    monkeypatch.setenv(dm.DERIVED_REFERENCE_PRICE_ENV, "0")
    assert _edit(_pipe()) == _flat_extra()


def test_the_price_scales_with_the_reference_tokens(store):
    small = _edit(_pipe(), reference_pixels = 512 * 512, condition_pixels = int(512 * 512 * 0.32))
    two = _edit(_pipe(), reference_pixels = 2 * REF, condition_pixels = 2 * COND)
    one = _edit(_pipe())
    assert small < one < two
    # 512^2: 768 x 1.25 + 512 MiB x 1.15 = 1549 -> 1792
    assert small == 1792 - PHASE


def test_a_guidance_request_doubles_the_cache_and_the_branches(store):
    on = _edit(_pipe(), cfg = None)
    assert on == 7936 - PHASE  # 2 x (1536 + 2355) = 7782 -> 7936
    assert on > _edit(_pipe())


def test_a_larger_learned_workflow_peak_wins(store):
    pipe = _pipe(identity = ("qwen-image-2.1", QUANT, "default"))
    assert _edit(pipe) == 4096 - PHASE
    mp.record(
        family = "qwen-image-2.1",
        quant = dm.with_workflow(QUANT, "edit"),
        speed_mode = "default",
        width = 1024,
        height = 1024,
        batch_size = 1,
        peak_mib = 4000,
        device_name = "NVIDIA A100-SXM4-40GB",
    )
    mp._cache.update(path = None, mtime = None, data = None)
    learned = dm.measured_image_runtime_mib(
        "qwen-image-2.1",
        "default",
        quant = dm.with_workflow(QUANT, "edit"),
        width = 1024,
        height = 1024,
        phase_seeds = True,
        static_seed = True,
    )
    assert learned is not None and learned > 4096
    assert _edit(pipe) == learned - PHASE


def test_no_prefix_cache_prices_only_the_joint_sequence(store):
    assert _edit(_pipe(transformer = _NoCache())) == 1536 - PHASE
    assert _edit(_pipe(transformer = _Qwen21(causal = False))) == 1536 - PHASE


@pytest.mark.parametrize(
    "case",
    ["quadratic", "family", "no_phase", "no_reference_pixels", "large_canvas", "no_vae_scale"],
)
def test_unpriceable_requests_keep_the_flat_allowance(store, case):
    pipe, kw = _pipe(), {}
    if case == "quadratic":
        kw["quadratic_attention"] = True
    elif case == "family":
        pipe = _pipe(family = "flux.3-klein")
    elif case == "no_phase":
        pipe = _pipe(phase = None)
    elif case == "no_reference_pixels":
        kw["reference_pixels"] = 0
    elif case == "large_canvas":
        kw.update(width = 2048, height = 2048)
    elif case == "no_vae_scale":
        pipe.vae_scale_factor = None
    old = dict(kw, reference_pixels = 0, quadratic_attention = False)
    if case == "quadratic":
        old.pop("quadratic_attention")
    assert _edit(pipe, **kw) == _edit(pipe, **old)
    if case in ("quadratic", "no_reference_pixels", "no_vae_scale"):
        assert _edit(pipe, **kw) == _flat_extra()


def test_text_to_image_is_unchanged(store):
    assert dm.measured_request_extra_mib(_pipe(), width = 1024, height = 1024, reference_pixels = 0) == 0
    assert (
        dm.measured_request_extra_mib(_pipe(), width = 1024, height = 1024, reference_pixels = REF) == 0
    )
