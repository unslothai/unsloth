# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 native (sd.cpp) memory fit: the Vulkan GTT pool, the host RAM floor, and the failure text."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend
from core.inference.video import video_failure_detail
from core.inference.video_minimax_h3 import (
    H3_NATIVE_FIT_ENV,
    H3NativeMemoryRefusal,
    amdgpu_gtt_free_bytes,
    h3_native_memory_fit,
    h3_native_offloaded_render_flags,
    vulkan_device_is_amd,
)
from utils.host_memory import cgroup_swap_allowance_bytes

GIB = 1024**3
# The published unsloth/MiniMax-H3-GGUF sizes: Q4_K denoiser, Q4_K_M encoder, video + audio VAE.
Q4 = dict(dit_bytes = int(10.64 * GIB), te_bytes = int(16.97 * GIB), vae_bytes = int(5.41 * GIB))
AUTO = ("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1")
LOW_VRAM = (
    "--offload-to-cpu",
    "--clip-on-cpu",
    "--diffusion-fa",
    "--vae-tiling",
    "--max-vram",
    "-1",
)
# A 48 GiB host: TTM's default pages_limit is half of RAM.
GTT_48 = 24 * GIB


def _fit(
    flags,
    mode,
    *,
    accelerator = "vulkan",
    gtt = GTT_48,
    host = 40 * GIB,
    sizes = Q4,
    env = None,
):
    return h3_native_memory_fit(
        flags,
        accelerator = accelerator,
        memory_mode = mode,
        pinned_free_bytes = gtt,
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


def _card(root, card, slot, *, total, used = None, vendor = "0x1002"):
    dev = root / "class" / "drm" / card / "device"
    dev.mkdir(parents = True)
    (dev / "vendor").write_text(vendor + "\n", encoding = "utf-8")
    (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n", encoding = "utf-8")
    (dev / "mem_info_gtt_total").write_text(f"{total}\n", encoding = "utf-8")
    if used is not None:
        (dev / "mem_info_gtt_used").write_text(f"{used}\n", encoding = "utf-8")


def test_the_gtt_reading_subtracts_what_is_already_used(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 5 * GIB)
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path)) == 19 * GIB


def test_the_gtt_reading_follows_the_selected_card(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 1 * GIB)
    _card(tmp_path, "card1", "0000:c5:00.0", total = 24 * GIB, used = 20 * GIB)  # the APU
    assert amdgpu_gtt_free_bytes("0000:03:00", sysfs_root = str(tmp_path)) == 23 * GIB
    assert amdgpu_gtt_free_bytes("0000:03:00.0", sysfs_root = str(tmp_path)) == 23 * GIB
    # Unknown slot: the tightest card, since TTM's limit is shared.
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path)) == 4 * GIB
    # A slot that is no amdgpu card (an NVIDIA or Intel device) has no pool.
    assert amdgpu_gtt_free_bytes("0000:81:00", sysfs_root = str(tmp_path)) is None


def test_a_non_amd_drm_card_is_never_read(tmp_path):
    _card(tmp_path, "card0", "0000:00:02.0", total = 8 * GIB, vendor = "0x8086")
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path)) is None
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path / "none")) is None


@pytest.mark.parametrize(
    "description, amd",
    [
        ("AMD Radeon RX 7800 XT (RADV NAVI32)", True),
        ("AMD Radeon(TM) Graphics (RADV GFX1151)", True),
        ("Radeon RX 6800", True),
        ("Intel(R) Arc(tm) A770 Graphics (DG2)", False),
        ("NVIDIA GeForce RTX 4070 Ti", False),
        ("llvmpipe (LLVM 17.0.6, 256 bits)", False),
        (None, False),
    ],
)
def test_only_an_amd_vulkan_device_is_bound_by_gtt(description, amd):
    assert vulkan_device_is_amd(description) is amd


OFFLOADED_FIT = AUTO + ("--clip-on-cpu", "--backend", "diffusion=Vulkan0,te=cpu,vae=Vulkan0")


def test_a_resident_auto_render_drops_nothing_and_adds_no_encoder_move():
    resident = ["--diffusion-fa", "--max-vram", "-1"]
    assert h3_native_offloaded_render_flags(
        resident, render_resident = True, offloaded_flags = OFFLOADED_FIT, offload_refusal = "no"
    ) == resident


def test_an_offloaded_auto_render_takes_the_fitted_flags():
    assert h3_native_offloaded_render_flags(
        list(AUTO), render_resident = False, offloaded_flags = OFFLOADED_FIT, offload_refusal = None
    ) == list(OFFLOADED_FIT)
    assert h3_native_offloaded_render_flags(
        list(AUTO), render_resident = False, offloaded_flags = None, offload_refusal = None
    ) == list(AUTO)


def test_an_offloaded_auto_render_that_cannot_fit_is_refused_with_its_reason():
    with pytest.raises(H3NativeMemoryRefusal) as caught:
        h3_native_offloaded_render_flags(
            list(AUTO), render_resident = False, offloaded_flags = None, offload_refusal = "too big for GTT"
        )
    detail = video_failure_detail(caught.value)
    assert detail.endswith("too big for GTT")


def _cgroup(tmp_path, files, proc = "0::/user.slice/app.scope\n"):
    root = tmp_path / "cg"
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(value + "\n", encoding = "utf-8")
    root.mkdir(exist_ok = True)
    (tmp_path / "self").write_text(proc, encoding = "utf-8")
    return dict(cgroup_root = str(root), proc_self_cgroup = str(tmp_path / "self"))


def test_swap_is_bounded_by_the_cgroup_swap_allowance(tmp_path):
    paths = _cgroup(
        tmp_path,
        {
            "user.slice/app.scope/memory.max": str(48 * GIB),
            "user.slice/app.scope/memory.swap.max": str(4 * GIB),
            "user.slice/app.scope/memory.swap.current": str(1 * GIB),
        },
    )
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == 3 * GIB


def test_swap_max_zero_admits_no_swap(tmp_path):
    paths = _cgroup(
        tmp_path,
        {
            "user.slice/app.scope/memory.max": str(48 * GIB),
            "user.slice/app.scope/memory.swap.max": "0",
            "user.slice/app.scope/memory.swap.current": "0",
        },
    )
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == 0


def test_an_unlimited_cgroup_keeps_host_free_swap(tmp_path):
    paths = _cgroup(
        tmp_path,
        {"user.slice/app.scope/memory.max": "max", "user.slice/app.scope/memory.swap.max": "max"},
    )
    assert cgroup_swap_allowance_bytes(8 * GIB, **paths) == 8 * GIB


def test_a_ram_limit_without_swap_accounting_admits_no_swap(tmp_path):
    paths = _cgroup(tmp_path, {"user.slice/memory.max": str(48 * GIB)})
    assert cgroup_swap_allowance_bytes(8 * GIB, **paths) == 0


def test_cgroup_v1_memsw_headroom_bounds_swap(tmp_path):
    paths = _cgroup(
        tmp_path,
        {
            "memory/docker/x/memory.limit_in_bytes": str(32 * GIB),
            "memory/docker/x/memory.usage_in_bytes": str(10 * GIB),
            "memory/docker/x/memory.memsw.limit_in_bytes": str(40 * GIB),
            "memory/docker/x/memory.memsw.usage_in_bytes": str(12 * GIB),
        },
        proc = "4:memory:/docker/x\n",
    )
    # RAM+swap room 28 GiB minus RAM room 22 GiB.
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == 6 * GIB


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
    detail = video_failure_detail(
        RuntimeError("sd-cli exited 1. Last output:\nggml_vulkan: device lost")
    )
    assert "reset the device" in detail and "Low VRAM" in detail


def test_an_unrelated_native_crash_keeps_its_old_text():
    detail = video_failure_detail(
        RuntimeError("sd-cli exited -11. Last output:\nSegmentation fault")
    )
    assert "renderer stopped unexpectedly" in detail


def test_a_torch_oom_keeps_its_old_text():
    detail = video_failure_detail(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert "smaller resolution, fewer frames, or fewer steps" in detail


def test_nested_cgroup_v1_pairs_the_tightest_combined_and_ram_headroom(tmp_path):
    """Child: 2 GiB RAM room, 12 GiB RAM+swap room. Parent: 20 GiB RAM room, 21 GiB RAM+swap room. The process can
    use min(12, 21) - min(2, 20) = 10 GiB of swap; per-level differences would say min(10, 1) = 1."""
    paths = _cgroup(
        tmp_path,
        {
            "memory/p/memory.limit_in_bytes": str(30 * GIB),
            "memory/p/memory.usage_in_bytes": str(10 * GIB),
            "memory/p/memory.memsw.limit_in_bytes": str(40 * GIB),
            "memory/p/memory.memsw.usage_in_bytes": str(19 * GIB),
            "memory/p/c/memory.limit_in_bytes": str(8 * GIB),
            "memory/p/c/memory.usage_in_bytes": str(6 * GIB),
            "memory/p/c/memory.memsw.limit_in_bytes": str(20 * GIB),
            "memory/p/c/memory.memsw.usage_in_bytes": str(8 * GIB),
        },
        proc = "4:memory:/p/c\n",
    )
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == 10 * GIB


def test_cgroup_v1_ram_only_levels_still_bound_the_ram_side(tmp_path):
    paths = _cgroup(
        tmp_path,
        {
            "memory/p/memory.limit_in_bytes": str(4 * GIB),
            "memory/p/memory.usage_in_bytes": str(1 * GIB),
            "memory/p/memory.memsw.limit_in_bytes": str(1 << 62),
            "memory/p/memory.memsw.usage_in_bytes": str(1 * GIB),
            "memory/p/c/memory.limit_in_bytes": str(16 * GIB),
            "memory/p/c/memory.usage_in_bytes": str(1 * GIB),
            "memory/p/c/memory.memsw.limit_in_bytes": str(20 * GIB),
            "memory/p/c/memory.memsw.usage_in_bytes": str(1 * GIB),
        },
        proc = "4:memory:/p/c\n",
    )
    # RAM room min(3, 15) = 3 GiB, combined room 19 GiB: 16 GiB of swap.
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == 16 * GIB


def test_native_host_ram_does_not_add_back_what_studio_holds(monkeypatch):
    from core.inference import video_minimax_h3
    from utils import host_memory

    monkeypatch.setattr(host_memory, "usable_host_ram_mib", lambda: 30 * 1024)
    monkeypatch.setattr(video_minimax_h3, "h3_process_held_host_bytes", lambda: 40 * GIB)
    assert video_minimax_h3.h3_native_host_available_bytes() == 30 * GIB


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


def test_an_unaskable_rocm_runtime_is_not_read_as_loading(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend, "rocm_runtime_missing_sonames", lambda: None)
    monkeypatch.setattr(sd_cpp_backend.os, "name", "posix")
    reason = sd_cpp_backend.accelerator_fallback_reason("rocm")
    assert "could not be checked" in reason and "/dev/kfd" not in reason


def test_the_windows_fallback_reason_names_the_hip_dlls(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend.os, "name", "nt")
    reason = sd_cpp_backend.accelerator_fallback_reason("rocm")
    assert "amdhip64_7.dll" in reason and "0xC0000135" in reason and "/dev/kfd" not in reason


def test_rocm_runtime_check_is_unknown_on_windows(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend.os, "name", "nt")
    assert sd_cpp_backend.rocm_runtime_missing_sonames() is None


def test_the_listed_device_names_an_unowned_build(monkeypatch):
    listing = "CPU\tAMD Ryzen 9\nVulkan0\tAMD Radeon RX 7800 XT (RADV NAVI32)\nVulkan1\tIntel(R) UHD\n"
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: listing)
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli") == ("vulkan", "AMD Radeon RX 7800 XT (RADV NAVI32)")
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli", "Vulkan1") == ("vulkan", "Intel(R) UHD")
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: "CPU\tAMD Ryzen 9\n")
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli") == (None, None)


def test_the_fallback_reason_without_missing_libraries_points_at_the_card(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend.os, "name", "posix")
    monkeypatch.setattr(sd_cpp_backend, "rocm_runtime_missing_sonames", lambda: [])
    assert "gfx target" in sd_cpp_backend.accelerator_fallback_reason("rocm")
