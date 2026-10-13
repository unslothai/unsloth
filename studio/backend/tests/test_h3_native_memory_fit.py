# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 native (sd.cpp) memory fit: the Vulkan GTT pool, the host RAM floor, and the failure text."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend
from core.inference.video import video_failure_detail
from core.inference.video_minimax_h3 import (
    H3_NATIVE_FIT_ENV,
    amdgpu_gtt_limit_bytes,
    h3_native_memory_fit,
)

GIB = 1024**3
# The published unsloth/MiniMax-H3-GGUF sizes: Q4_K denoiser, Q4_K_M encoder, video + audio VAE.
Q4 = dict(dit_bytes = int(10.64 * GIB), te_bytes = int(16.97 * GIB), vae_bytes = int(5.41 * GIB))
AUTO = ("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1")
LOW_VRAM = ("--offload-to-cpu", "--clip-on-cpu", "--diffusion-fa", "--vae-tiling", "--max-vram", "-1")
# A 48 GiB host: TTM's default pages_limit is half of RAM.
GTT_48 = 24 * GIB


def _fit(flags, mode, *, accelerator = "vulkan", gtt = GTT_48, host = 40 * GIB, sizes = Q4, env = None):
    return h3_native_memory_fit(
        flags,
        accelerator = accelerator,
        memory_mode = mode,
        pinned_limit_bytes = gtt,
        host_available_bytes = host,
        env = env or {},
        **sizes,
    )


def test_auto_on_a_48gb_radv_host_runs_the_encoder_on_the_cpu():
    fit = _fit(AUTO, "auto")
    assert fit.refusal is None
    assert fit.flags == AUTO + ("--clip-on-cpu",)
    assert "GTT" in fit.note and "33.0 GiB" in fit.note


@pytest.mark.parametrize("mode", ["balanced", "fast"])
def test_an_explicit_mode_that_overflows_gtt_is_refused_naming_the_one_that_fits(mode):
    fit = _fit(AUTO, mode)
    assert fit.flags == AUTO
    assert "Low VRAM" in fit.refusal and "16.0 GiB" in fit.refusal


def test_low_vram_fits_a_48gb_radv_host_unchanged():
    fit = _fit(LOW_VRAM, "low_vram")
    assert fit == type(fit)(LOW_VRAM)


def test_a_pool_too_small_even_for_low_vram_names_the_smaller_quant_and_the_kernel_knob():
    fit = _fit(LOW_VRAM, "low_vram", gtt = 16 * GIB)
    assert "Q2_K" in fit.refusal and "ttm.pages_limit" in fit.refusal


def test_a_large_pool_changes_nothing():
    assert _fit(AUTO, "auto", gtt = 64 * GIB).flags == AUTO


@pytest.mark.parametrize("accelerator", ["cuda", "rocm", "cpu", None])
def test_the_gtt_pool_only_binds_a_vulkan_build(accelerator):
    fit = _fit(AUTO, "auto", accelerator = accelerator)
    assert fit.flags == AUTO and fit.refusal is None and fit.note is None


def test_no_amdgpu_reading_changes_nothing():
    assert _fit(AUTO, "auto", gtt = None).flags == AUTO


def test_a_resident_render_is_not_judged():
    resident = ("--diffusion-fa", "--max-vram", "-1")
    assert _fit(resident, "fast", gtt = 1 * GIB, host = 1 * GIB).flags == resident


def test_a_host_without_room_for_the_weights_is_refused_naming_q2():
    fit = _fit(AUTO, "auto", accelerator = "cuda", host = 30 * GIB)
    assert "system RAM" in fit.refusal and "Q2_K" in fit.refusal


def test_the_kill_switch_restores_the_old_flags():
    fit = _fit(AUTO, "auto", host = 1 * GIB, env = {H3_NATIVE_FIT_ENV: "0"})
    assert fit.flags == AUTO and fit.refusal is None


def test_the_gtt_reading_takes_the_smallest_amdgpu_pool(tmp_path):
    for card, value in (("card0", "25769803776"), ("card1", "8589934592"), ("card2", "junk")):
        dev = tmp_path / "class" / "drm" / card / "device"
        dev.mkdir(parents = True)
        (dev / "mem_info_gtt_total").write_text(value + "\n", encoding = "utf-8")
    assert amdgpu_gtt_limit_bytes(str(tmp_path)) == 8 * GIB
    assert amdgpu_gtt_limit_bytes(str(tmp_path / "none")) is None


_RADV_TAIL = (
    "sd-cli exited 1. Last output:\n"
    "[V] model manager prepared params backend buffers (295.12 MB, 11 tensors, 1 blocks, RAM) on Vulkan_Host"
    " --- model_manager.cpp:490\n"
    "radv/amdgpu: Not enough memory for command submission.\n"
    "[E] ggml: ggml_vulkan: device lost on Vulkan0\n"
)


def test_the_radv_failure_reads_as_memory_with_the_settings_that_help():
    detail = video_failure_detail(RuntimeError(_RADV_TAIL))
    assert "ran out of GPU or system memory" in detail
    assert "Low VRAM" in detail and "864x480" in detail and "quant" in detail


def test_a_bare_device_loss_is_explained_as_likely_memory():
    detail = video_failure_detail(RuntimeError("sd-cli exited 1. Last output:\nggml_vulkan: device lost"))
    assert "reset the device" in detail and "Low VRAM" in detail


def test_an_unrelated_native_crash_keeps_its_old_text():
    detail = video_failure_detail(RuntimeError("sd-cli exited -11. Last output:\nSegmentation fault"))
    assert "renderer stopped unexpectedly" in detail


def test_a_torch_oom_keeps_its_old_text():
    detail = video_failure_detail(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert "smaller resolution, fewer frames, or fewer steps" in detail


def test_the_fallback_reason_names_the_missing_rocm_libraries(monkeypatch):
    import ctypes

    def fake_cdll(name, *a, **k):
        if name == "libhipblas.so.3":
            raise OSError(name)
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(sd_cpp_backend.sys, "platform", "linux")
    assert sd_cpp_backend.rocm_runtime_missing_sonames() == ["libhipblas.so.3"]
    reason = sd_cpp_backend.accelerator_fallback_reason("rocm")
    assert "libhipblas.so.3" in reason and "ROCm 7" in reason
    assert "diffusion-accelerator-fallback" in sd_cpp_backend.ACCELERATOR_FALLBACK_CLEAR_HINT


def test_the_fallback_reason_without_missing_libraries_points_at_the_card(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend, "rocm_runtime_missing_sonames", lambda: [])
    assert "gfx target" in sd_cpp_backend.accelerator_fallback_reason("rocm")
