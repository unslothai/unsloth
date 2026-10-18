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
    amdgpu_vram_free_bytes,
    h3_native_compute_bytes,
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
    swap = 0,
    spill = 0,
    uma = False,
):
    return h3_native_memory_fit(
        flags,
        accelerator = accelerator,
        memory_mode = mode,
        pinned_free_bytes = gtt,
        host_ram_bytes = host,
        host_swap_bytes = swap,
        device_spill_bytes = spill,
        reserve_pinned = uma,
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


# Denoiser 10, encoder 4, VAEs 2 GiB: 12 GiB pinned with the encoder on the CPU, 15 with sd-cli's overhead.
SMALL = dict(dit_bytes = 10 * GIB, te_bytes = 4 * GIB, vae_bytes = 2 * GIB)
# Denoiser 10, encoder 17, VAEs 5 GiB: 15 GiB pinned with the encoder on the CPU, 32 with it on the GPU.
BIG_TE = dict(dit_bytes = 10 * GIB, te_bytes = 17 * GIB, vae_bytes = 5 * GIB)


def test_swap_never_covers_the_pinned_set():
    """Page-locked host buffers cannot be swapped: 15 GiB locked against 12 GiB of RAM is refused however much swap
    the cgroup allows."""
    fit = _fit(LOW_VRAM, "low_vram", accelerator = "cuda", host = 12 * GIB, swap = 500 * GIB, sizes = SMALL)
    assert "cannot be swapped" in fit.refusal and "12.0 GiB of RAM" in fit.refusal


def test_swap_covers_the_pageable_cpu_run_encoder():
    fit = _fit(AUTO, "auto", accelerator = "cuda", host = 25 * GIB, swap = 20 * GIB, sizes = BIG_TE)
    assert fit.refusal is None
    assert fit.flags == AUTO + ("--clip-on-cpu",)
    assert "page-locked" in fit.note


def test_an_explicit_mode_whose_locked_set_overflows_ram_names_low_vram():
    fit = _fit(AUTO, "balanced", accelerator = "cuda", host = 25 * GIB, swap = 20 * GIB, sizes = BIG_TE)
    assert "Low VRAM" in fit.refusal and "cannot be swapped" in fit.refusal


def test_without_swap_the_whole_set_must_fit_ram():
    fit = _fit(AUTO, "auto", accelerator = "cuda", host = 25 * GIB, swap = 0, sizes = BIG_TE)
    assert "RAM plus 0.0 GiB of swap" in fit.refusal


def test_the_16gb_card_48gb_host_decision_is_unchanged():
    """About 42 GiB of a 48 GB host available, no swap, 23.4 GiB of GTT: the encoder still moves for GTT."""
    fit = _fit(AUTO, "auto", gtt = int(23.4 * GIB), host = 42 * GIB, swap = 0)
    assert fit.flags == AUTO + ("--clip-on-cpu",) and "GTT" in fit.note


def test_a_uma_card_reserves_its_pinned_weights_from_the_device_budget():
    """gfx1151 with a 0.5 GiB carve-out and 22.5 GiB of GTT free: sd.cpp reads that free memory at startup, before it
    pins the denoiser and VAEs in the same pool, so --max-vram -1 planned one segment and the 864x480x124 render
    failed its capacity check. Reserving the pinned set leaves the budget the compute buffer really has."""
    spill = h3_native_compute_bytes(864, 480, 124) - (GIB // 2)
    fit = _fit(AUTO, "auto", gtt = int(22.5 * GIB), spill = spill, host = 100 * GIB, uma = True)
    assert fit.refusal is None
    pinned = Q4["dit_bytes"] + Q4["vae_bytes"] + GIB
    assert fit.max_vram == f"-{pinned / GIB:.2f}"
    assert fit.flags == (
        "--offload-to-cpu",
        "--diffusion-fa",
        "--max-vram",
        fit.max_vram,
        "--clip-on-cpu",
    )
    assert fit.max_vram == "-17.05"  # measured: sd.cpp budgeted 5.73 GiB and rendered 864x480x124
    # Measured with the same reserve: 20 GiB free leaves no room for the compute buffer, 30 GiB renders 1344x768x124.
    assert _fit(AUTO, "auto", gtt = 20 * GIB, spill = spill, host = 100 * GIB, uma = True).refusal
    spill = h3_native_compute_bytes(1344, 768, 124) - (GIB // 2)
    fit = _fit(AUTO, "auto", gtt = int(22.5 * GIB), spill = spill, host = 100 * GIB, uma = True)
    assert "weights and activations into GPU-accessible system memory (GTT)" in fit.refusal
    assert _fit(AUTO, "auto", gtt = 30 * GIB, spill = spill, host = 100 * GIB, uma = True).refusal is None


def test_an_explicit_mode_on_a_uma_card_keeps_its_placement_and_gets_the_reserve():
    fit = _fit(LOW_VRAM, "low_vram", gtt = 30 * GIB, host = 100 * GIB, uma = True)
    assert fit.refusal is None and fit.max_vram is not None
    assert [f for f in fit.flags if f != fit.max_vram] == [f for f in LOW_VRAM if f != "-1"]


def test_a_uma_card_without_a_gtt_reading_keeps_max_vram_minus_one():
    """Windows has no amdgpu sysfs, so no GTT reading: the reserve stays off and the committed -1 is kept."""
    fit = _fit(AUTO, "auto", gtt = None, host = 100 * GIB, uma = True)
    assert fit.refusal is None and fit.max_vram is None and fit.flags == AUTO


def test_a_discrete_card_keeps_max_vram_minus_one():
    fit = _fit(AUTO, "auto")
    assert fit.max_vram is None and fit.flags[:4] == AUTO


def test_the_compute_estimate_covers_the_measured_buffers():
    """sd-cli's minimax_h3 compute buffer on gfx1151: 5175.60 MB at 864x480x124, 10595.71 MB at 1344x768x124."""
    assert 5175.60 * 2**20 <= h3_native_compute_bytes(864, 480, 124) <= 6 * GIB
    assert 10595.71 * 2**20 <= h3_native_compute_bytes(1344, 768, 124) <= 11 * GIB


def test_a_spill_only_binds_a_vulkan_gtt_pool():
    fit = _fit(AUTO, "auto", accelerator = "rocm", spill = 64 * GIB)
    assert fit.refusal is None and fit.flags == AUTO


def test_the_kill_switch_restores_the_old_flags():
    fit = _fit(AUTO, "auto", host = 1 * GIB, env = {H3_NATIVE_FIT_ENV: "0"})
    assert fit.flags == AUTO and fit.refusal is None


def _card(
    root,
    card,
    slot,
    *,
    total,
    used = None,
    vendor = "0x1002",
    vram = None,
):
    dev = root / "class" / "drm" / card / "device"
    dev.mkdir(parents = True)
    (dev / "vendor").write_text(vendor + "\n", encoding = "utf-8")
    (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n", encoding = "utf-8")
    (dev / "mem_info_gtt_total").write_text(f"{total}\n", encoding = "utf-8")
    if used is not None:
        (dev / "mem_info_gtt_used").write_text(f"{used}\n", encoding = "utf-8")
    if vram is not None:
        (dev / "mem_info_vram_total").write_text(f"{vram[0]}\n", encoding = "utf-8")
        (dev / "mem_info_vram_used").write_text(f"{vram[1]}\n", encoding = "utf-8")


def test_the_gtt_reading_subtracts_what_is_already_used(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 5 * GIB)
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path)) == 19 * GIB


def test_the_gtt_reading_follows_the_selected_card(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 1 * GIB)
    _card(tmp_path, "card1", "0000:c5:00.0", total = 24 * GIB, used = 20 * GIB)  # the APU
    # TTM's page limit is shared, so the APU's 20 GiB counts against the selected card's pool too.
    assert amdgpu_gtt_free_bytes("0000:03:00", sysfs_root = str(tmp_path)) == 3 * GIB
    assert amdgpu_gtt_free_bytes("0000:03:00.0", sysfs_root = str(tmp_path)) == 3 * GIB
    # Unknown slot: the smallest pool, less what every card uses.
    assert amdgpu_gtt_free_bytes(sysfs_root = str(tmp_path)) == 3 * GIB
    # A slot that is no amdgpu card (an NVIDIA or Intel device) has no pool.
    assert amdgpu_gtt_free_bytes("0000:81:00", sysfs_root = str(tmp_path)) is None


def test_gtt_used_on_another_amd_card_shrinks_the_selected_cards_pool(tmp_path):
    """APU plus discrete card: TTM's page limit is one pool, so the APU's GTT use is not free for the dGPU."""
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 2 * GIB)
    _card(tmp_path, "card1", "0000:c5:00.0", total = 24 * GIB, used = 12 * GIB)
    assert amdgpu_gtt_free_bytes("0000:03:00", sysfs_root = str(tmp_path)) == 10 * GIB
    # Only the selected card at work: the per-device bound still applies.
    (tmp_path / "class" / "drm" / "card1" / "device" / "mem_info_gtt_used").write_text(
        "0\n", encoding = "utf-8"
    )
    assert amdgpu_gtt_free_bytes("0000:03:00", sysfs_root = str(tmp_path)) == 22 * GIB


def test_the_vram_reading_takes_the_selected_card_or_the_only_one(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, vram = (16 * GIB, 1 * GIB))
    assert amdgpu_vram_free_bytes(sysfs_root = str(tmp_path)) == 15 * GIB
    _card(tmp_path, "card1", "0000:c5:00.0", total = 24 * GIB, vram = (GIB // 2, GIB // 4))
    assert amdgpu_vram_free_bytes("0000:c5:00", sysfs_root = str(tmp_path)) == GIB // 4
    assert amdgpu_vram_free_bytes("0000:03:00.0", sysfs_root = str(tmp_path)) == 15 * GIB
    # Two cards and no selection: which one sd.cpp renders on is unknown.
    assert amdgpu_vram_free_bytes(sysfs_root = str(tmp_path)) is None
    assert amdgpu_vram_free_bytes("0000:81:00", sysfs_root = str(tmp_path)) is None


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
    assert (
        h3_native_offloaded_render_flags(
            resident, render_resident = True, offloaded_flags = OFFLOADED_FIT, offload_refusal = "no"
        )
        == resident
    )


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
            list(AUTO),
            render_resident = False,
            offloaded_flags = None,
            offload_refusal = "too big for GTT",
        )
    detail = video_failure_detail(caught.value)
    assert detail.endswith("too big for GTT")


def _cgroup(
    tmp_path,
    files,
    proc = "0::/user.slice/app.scope\n",
):
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


def test_sd_cpps_segment_capacity_failure_reads_as_memory():
    """sd.cpp's graph-cut executor reports a device it cannot fit this way, without any generic OOM wording."""
    tail = (
        "sd-cli exited 1. Last output:\n"
        "[W] model manager cannot make enough memory available on Vulkan0: need 5687.60 MB device / 5175.60 MB "
        "budget, available 1356.88 MB device / 11346.88 MB budget --- model_manager.cpp:1944\n"
        "[E] minimax_h3 segment 1/1 (graph) failed during workspace capacity check --- ggml_runner.cpp:898\n"
        "[E] generate failed --- main.cpp:939\n"
    )
    detail = video_failure_detail(RuntimeError(tail))
    assert "ran out of GPU or system memory" in detail and "864x480" in detail
    assert not sd_cpp_backend.output_shows_accelerator_failure(tail)
    assert sd_cpp_backend.output_shows_capacity_failure(
        "[E] minimax_h3 segment 2/9 (graph) failed during workspace capacity check"
    )
    assert sd_cpp_backend.output_shows_capacity_failure(
        "[E] minimax_h3 segment 1/4 (graph) failed during allocated capacity check"
    )


# sd.cpp's model manager logs this when one eviction pass falls short, then recovers: renders that complete print it
# (gfx1151, 864x480x124 with the pinned reserve: rc 0, a full video, two of these lines in its log).
_RECOVERED_CAPACITY_WARNING = (
    "[W] model manager cannot make enough memory available on Vulkan0: need 1038.05 MB device / 5701.21 MB "
    "budget, available 17290.44 MB device / 5690.66 MB budget --- model_manager.cpp:1944\n"
)


def test_a_recovered_capacity_warning_is_not_a_capacity_failure():
    """Only the failing segment's [E] line is a capacity failure. Read as one, the warning would hide what really
    stopped the render: the build-level failure below must still be recorded, and an unrelated crash must not be
    explained as memory."""
    assert not sd_cpp_backend.output_shows_capacity_failure(_RECOVERED_CAPACITY_WARNING)
    build_failure = (
        "sd-cli exited 1. Last output:\n"
        + _RECOVERED_CAPACITY_WARNING
        + "ggml_cuda_compute_forward: error: invalid device function\n"
    )
    assert sd_cpp_backend.output_shows_accelerator_failure(build_failure)
    assert sd_cpp_backend.output_shows_decisive_accelerator_failure(build_failure)
    crash = (
        "sd-cli exited -11. Last output:\n" + _RECOVERED_CAPACITY_WARNING + "Segmentation fault\n"
    )
    assert "renderer stopped unexpectedly" in video_failure_detail(RuntimeError(crash))


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


_V1_SWAP_ROOM = {
    "memory/p/memory.limit_in_bytes": str(30 * GIB),
    "memory/p/memory.usage_in_bytes": str(10 * GIB),
    "memory/p/memory.memsw.limit_in_bytes": str(40 * GIB),
    "memory/p/memory.memsw.usage_in_bytes": str(10 * GIB),
}


_ENFORCING_CHILD = {
    "memory/p/c/memory.limit_in_bytes": str(28 * GIB),
    "memory/p/c/memory.usage_in_bytes": str(8 * GIB),
    "memory/p/c/memory.memsw.limit_in_bytes": str(100 * GIB),
    "memory/p/c/memory.memsw.usage_in_bytes": str(8 * GIB),
}


@pytest.mark.parametrize(
    "swappiness, expected",
    [
        # The child sets no limit: its value is irrelevant; the enforcing parent reports none.
        ({"memory/p/c/memory.swappiness": "0"}, 10 * GIB),
        # The enforcing parent's 0 governs its own limit reclaim whatever the child says.
        ({"memory/p/memory.swappiness": "0"}, 0),
        ({"memory/p/c/memory.swappiness": "60", "memory/p/memory.swappiness": "0"}, 0),
        ({"memory/p/c/memory.swappiness": "60", "memory/p/memory.swappiness": "60"}, 10 * GIB),
        # The process's own cgroup counts once it enforces.
        ({**_ENFORCING_CHILD, "memory/p/c/memory.swappiness": "0"}, 0),
        ({**_ENFORCING_CHILD, "memory/p/c/memory.swappiness": "60"}, 10 * GIB),
    ],
)
def test_cgroup_v1_swappiness_zero_on_an_enforcing_level_admits_no_swap(
    tmp_path, swappiness, expected
):
    """https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html#swappiness: 0 keeps a level's limit
    reclaim from swapping. Any level with a finite limit counts; one without a limit reclaims nothing."""
    (tmp_path / "cg" / "memory" / "p" / "c").mkdir(parents = True)
    paths = _cgroup(tmp_path, {**_V1_SWAP_ROOM, **swappiness}, proc = "4:memory:/p/c\n")
    # RAM room 20 GiB, RAM+swap room 30 GiB: 10 GiB of swap unless swappiness forbids it.
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == expected


@pytest.mark.parametrize(
    "limit, expected",
    [(str(30 * GIB), 0), ("9223372036854771712", 64 * GIB)],
)
def test_cgroup_v1_swappiness_zero_on_a_namespaced_root_counts_when_it_enforces(
    tmp_path, limit, expected
):
    """A v1 container sees its own cgroup as the mount root (/proc/self/cgroup reads "/"); its swappiness 0 counts
    once that root enforces a limit, while an unlimited host root's value stays irrelevant."""
    files = {
        "memory/memory.limit_in_bytes": limit,
        "memory/memory.usage_in_bytes": str(10 * GIB),
        "memory/memory.memsw.limit_in_bytes": str(40 * GIB) if expected == 0 else limit,
        "memory/memory.memsw.usage_in_bytes": str(10 * GIB),
        "memory/memory.swappiness": "0",
    }
    paths = _cgroup(tmp_path, files, proc = "4:memory:/\n")
    assert cgroup_swap_allowance_bytes(64 * GIB, **paths) == expected


def test_the_kill_switch_turns_off_the_per_render_overhead_guard(monkeypatch):
    from core.inference import video_minimax_h3

    monkeypatch.setattr(video_minimax_h3, "h3_native_host_budgets", lambda: (1 << 20, 0))
    assert video_minimax_h3.h3_native_render_overhead_refusal()
    monkeypatch.setenv(H3_NATIVE_FIT_ENV, "0")
    assert video_minimax_h3.h3_native_render_overhead_refusal() is None
    assert video_minimax_h3.h3_native_fit_check_enabled() is False


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


def test_unreadable_membership_with_a_root_ram_limit_admits_no_swap(tmp_path):
    root = tmp_path / "cg"
    root.mkdir()
    (root / "memory.max").write_text(str(48 * GIB) + "\n", encoding = "utf-8")
    missing = str(tmp_path / "no-such-proc-self-cgroup")
    assert (
        cgroup_swap_allowance_bytes(64 * GIB, cgroup_root = str(root), proc_self_cgroup = missing) == 0
    )


def test_unreadable_membership_reads_the_root_swap_files(tmp_path):
    root = tmp_path / "cg"
    root.mkdir()
    for name, value in (
        ("memory.max", str(48 * GIB)),
        ("memory.swap.max", str(8 * GIB)),
        ("memory.swap.current", str(3 * GIB)),
    ):
        (root / name).write_text(value + "\n", encoding = "utf-8")
    missing = str(tmp_path / "no-such-proc-self-cgroup")
    assert (
        cgroup_swap_allowance_bytes(64 * GIB, cgroup_root = str(root), proc_self_cgroup = missing)
        == 5 * GIB
    )


def test_unreadable_membership_without_any_limit_keeps_host_swap(tmp_path):
    root = tmp_path / "cg"
    root.mkdir()
    missing = str(tmp_path / "no-such-proc-self-cgroup")
    assert (
        cgroup_swap_allowance_bytes(8 * GIB, cgroup_root = str(root), proc_self_cgroup = missing)
        == 8 * GIB
    )


def test_cgroup_v1_swap_is_measured_against_the_ram_the_fit_actually_budgets(tmp_path):
    """40 GiB cgroup RAM room, 70 GiB RAM+swap room, but only 25 GiB MemAvailable: the fit budgets 25 GiB of RAM, so
    45 GiB of swap brings the charge to the 70 GiB the memsw limit allows (not 30 + 25 = 55)."""
    paths = _cgroup(
        tmp_path,
        {
            "memory/x/memory.limit_in_bytes": str(50 * GIB),
            "memory/x/memory.usage_in_bytes": str(10 * GIB),
            "memory/x/memory.memsw.limit_in_bytes": str(80 * GIB),
            "memory/x/memory.memsw.usage_in_bytes": str(10 * GIB),
        },
        proc = "4:memory:/x\n",
    )
    assert cgroup_swap_allowance_bytes(100 * GIB, **paths, ram_budget = 25 * GIB) == 45 * GIB
    assert cgroup_swap_allowance_bytes(100 * GIB, **paths) == 30 * GIB


def test_cgroup_v1_memsw_headroom_is_the_tightest_level(tmp_path):
    from utils.host_memory import cgroup_v1_memsw_headroom_bytes
    paths = _cgroup(
        tmp_path,
        {
            "memory/p/memory.memsw.limit_in_bytes": str(40 * GIB),
            "memory/p/memory.memsw.usage_in_bytes": str(35 * GIB),
            "memory/p/c/memory.memsw.limit_in_bytes": str(100 * GIB),
            "memory/p/c/memory.memsw.usage_in_bytes": str(10 * GIB),
        },
        proc = "4:memory:/p/c\n",
    )
    assert cgroup_v1_memsw_headroom_bytes(**paths) == 5 * GIB


def test_cgroup_v1_memsw_headroom_fails_closed_and_is_absent_without_a_limit(tmp_path):
    from utils.host_memory import cgroup_v1_memsw_headroom_bytes

    unreadable = _cgroup(
        tmp_path / "a",
        {"memory/x/memory.memsw.limit_in_bytes": str(40 * GIB)},
        proc = "4:memory:/x\n",
    )
    assert cgroup_v1_memsw_headroom_bytes(**unreadable) == 0
    (tmp_path / "b").mkdir()
    v2_only = _cgroup(tmp_path / "b", {"user.slice/memory.max": "max"})
    assert cgroup_v1_memsw_headroom_bytes(**v2_only) is None


def test_ram_is_capped_by_the_v1_memsw_headroom(monkeypatch):
    """80 GiB of RAM room but only 5 GiB RAM+swap room (the process swapped a lot): RAM and swap together may not
    exceed the 5 GiB the cgroup can still charge."""
    from core.inference import video_minimax_h3
    from utils import host_memory

    monkeypatch.setattr(video_minimax_h3, "h3_native_host_available_bytes", lambda: 80 * GIB)
    monkeypatch.setattr(host_memory, "cgroup_v1_memsw_headroom_bytes", lambda *a, **k: 5 * GIB)
    monkeypatch.setattr(
        host_memory,
        "cgroup_swap_allowance_bytes",
        lambda free, *a, ram_budget = None, **k: min(free, max(5 * GIB - ram_budget, 0)),
    )
    ram, swap = video_minimax_h3.h3_native_host_budgets()
    assert (ram, swap) == (5 * GIB, 0)


def test_an_unreadable_swap_reading_keeps_the_ram_check(monkeypatch):
    psutil = pytest.importorskip("psutil")

    from core.inference import video_minimax_h3
    from utils import host_memory

    def broken():
        raise OSError("no /proc/swaps")

    monkeypatch.setattr(video_minimax_h3, "h3_native_host_available_bytes", lambda: 30 * GIB)
    monkeypatch.setattr(host_memory, "cgroup_v1_memsw_headroom_bytes", lambda *a, **k: None)
    monkeypatch.setattr(psutil, "swap_memory", broken)
    assert video_minimax_h3.h3_native_host_budgets() == (30 * GIB, 0)


def test_an_unreadable_gtt_use_is_reported(tmp_path):
    from core.inference.video_minimax_h3 import amdgpu_gtt_used_unknown

    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB)
    _card(tmp_path, "card1", "0000:c5:00.0", total = 24 * GIB, used = 1 * GIB)
    assert amdgpu_gtt_used_unknown("0000:03:00", sysfs_root = str(tmp_path)) is True
    # The shared pool includes card0's unreadable use, so even the readable card's answer is unknown.
    assert amdgpu_gtt_used_unknown("0000:c5:00", sysfs_root = str(tmp_path)) is True
    assert amdgpu_gtt_used_unknown(sysfs_root = str(tmp_path)) is True
    dev = tmp_path / "class" / "drm" / "card1" / "device"
    (dev / "mem_info_gtt_used").write_text("garbage\n", encoding = "utf-8")
    assert amdgpu_gtt_used_unknown("0000:c5:00", sysfs_root = str(tmp_path)) is True


def test_one_gtt_pass_returns_free_and_readability_together(tmp_path):
    from core.inference.video_minimax_h3 import amdgpu_gtt_reading

    _card(tmp_path, "card0", "0000:03:00.0", total = 24 * GIB, used = 4 * GIB)
    _card(tmp_path, "card1", "0000:c5:00.0", total = 16 * GIB)
    assert amdgpu_gtt_reading("0000:03:00", sysfs_root = str(tmp_path)) == (20 * GIB, False)
    assert amdgpu_gtt_reading("0000:c5:00", sysfs_root = str(tmp_path)) == (12 * GIB, False)
    assert amdgpu_gtt_reading(sysfs_root = str(tmp_path)) == (12 * GIB, False)
    assert amdgpu_gtt_reading("0000:81:00", sysfs_root = str(tmp_path)) == (None, True)


def test_a_slot_matching_no_amdgpu_card_falls_back_to_the_tightest_card(monkeypatch):
    from core.inference import video_minimax_h3
    from core.inference.video_minimax_h3 import H3NativeFitInputs, h3_native_live_fit

    def reading(
        slot = None,
        *a,
        **k,
    ):
        return (None, True) if slot else (0, True)

    monkeypatch.setattr(video_minimax_h3, "amdgpu_gtt_reading", reading)
    monkeypatch.setattr(video_minimax_h3, "h3_native_host_budgets", lambda: (1 << 50, 0))
    inputs = H3NativeFitInputs(
        base_flags = AUTO,
        accelerator = "vulkan",
        check_gtt = True,
        gtt_pci_slot = "0000:81:00",
        dit_bytes = GIB,
        te_bytes = GIB,
        vae_bytes = GIB,
        device_name = "Vulkan0",
    )
    assert "GTT" in h3_native_live_fit(inputs, "auto").refusal


class _FakeServer:
    started: list = []

    def __init__(self, binary):
        self.alive = False

    def start(
        self,
        files,
        *,
        offload,
        env = None,
        extra_args = None,
    ):
        self.started.append(tuple(offload))
        self.alive = True

    def is_alive(self):
        return self.alive

    def stop(self):
        self.alive = False


@pytest.fixture
def fake_server(monkeypatch):
    from core.inference import sd_cpp_server

    _FakeServer.started = []
    monkeypatch.setattr(sd_cpp_server, "SdCppServer", _FakeServer)
    return _FakeServer


def test_a_server_start_is_fitted_and_started_with_the_fitted_flags(fake_server):
    from core.inference.video_minimax_h3 import H3NativeServerSlot

    slot = H3NativeServerSlot("/x/sd-server", object())
    calls = []

    def fit():
        calls.append(1)
        return ("--offload-to-cpu", "--clip-on-cpu")

    slot.get(("--offload-to-cpu",), (), pre_spawn = fit)
    assert calls == [1] and fake_server.started == [("--offload-to-cpu", "--clip-on-cpu")]
    # The live server is reused with no new fit.
    slot.get(("--offload-to-cpu", "--clip-on-cpu"), (), pre_spawn = fit)
    assert calls == [1] and len(fake_server.started) == 1


def test_a_server_that_stopped_after_the_snapshot_is_fitted_again_and_can_be_refused(fake_server):
    """alive_signature() said reuse, but the server exited before get(): the start is fitted, and refused."""
    from core.inference.video_minimax_h3 import H3NativeMemoryRefusal, H3NativeServerSlot

    slot = H3NativeServerSlot("/x/sd-server", object())
    slot.get(("--offload-to-cpu",), ())
    slot._server.alive = False

    def refuse():
        raise H3NativeMemoryRefusal("no longer fits")

    with pytest.raises(H3NativeMemoryRefusal):
        slot.get(("--offload-to-cpu",), (), pre_spawn = refuse)
    assert len(fake_server.started) == 1 and slot._server is None


def test_a_refused_server_start_is_not_turned_into_a_disabled_slot():
    import inspect

    from core.inference import video as video_mod

    source = inspect.getsource(video_mod.VideoBackend._h3_native_server_render)
    assert "pre_spawn = functools.partial(" in source
    assert "except (SdCppCancelled, H3NativeMemoryRefusal):" in source


def test_cgroup_v1_usage_leaves_out_the_inactive_file_cache(tmp_path):
    """https://docs.kernel.org/admin-guide/cgroup-v1/memory.html: usage counts page cache the kernel reclaims
    first; total_inactive_file is subtracted, as for RAM."""
    from utils.host_memory import cgroup_v1_memsw_headroom_bytes

    paths = _cgroup(
        tmp_path,
        {
            "memory/x/memory.limit_in_bytes": str(32 * GIB),
            "memory/x/memory.usage_in_bytes": str(30 * GIB),
            "memory/x/memory.memsw.limit_in_bytes": str(40 * GIB),
            "memory/x/memory.memsw.usage_in_bytes": str(38 * GIB),
            "memory/x/memory.stat": f"cache {20 * GIB}\ntotal_inactive_file {20 * GIB}\n",
        },
        proc = "4:memory:/x\n",
    )
    assert cgroup_v1_memsw_headroom_bytes(**paths) == 22 * GIB
    # RAM room 22 GiB (32 - (30 - 20)), RAM+swap room 22 GiB: no swap beyond the RAM.
    assert cgroup_swap_allowance_bytes(100 * GIB, **paths) == 0
    assert cgroup_swap_allowance_bytes(100 * GIB, **paths, ram_budget = 10 * GIB) == 12 * GIB


def _fit_unknown_use(flags, mode, *, gtt):
    return h3_native_memory_fit(
        flags,
        accelerator = "vulkan",
        memory_mode = mode,
        pinned_free_bytes = gtt,
        pinned_used_unknown = True,
        host_ram_bytes = 60 * GIB,
        env = {},
        **Q4,
    )


@pytest.mark.parametrize("mode", ["auto", "balanced", "fast"])
def test_unknown_gtt_use_fails_closed_unless_the_user_chose_low_vram(mode):
    """A 24 GiB pool whose use cannot be read may have 12 GiB taken elsewhere: neither the 33 GiB set nor the 16 GiB
    one is placed on it automatically. Low VRAM is the user's explicit choice and still runs."""
    fit = _fit_unknown_use(AUTO, mode, gtt = 24 * GIB)
    assert fit.flags == AUTO
    assert "could not read how much" in fit.refusal and "Low VRAM" in fit.refusal


def test_unknown_gtt_use_leaves_low_vram_alone_and_refuses_past_the_total():
    assert _fit_unknown_use(LOW_VRAM, "low_vram", gtt = 64 * GIB).flags == LOW_VRAM
    assert "ttm.pages_limit" in _fit_unknown_use(LOW_VRAM, "low_vram", gtt = 12 * GIB).refusal


def test_a_transiently_unreadable_listing_keeps_the_last_good_one(tmp_path, monkeypatch):
    import os

    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"x")
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    sd_cpp_backend._remember_device_listing(
        str(binary), "Vulkan0\tAMD Radeon RX 7800 XT (RADV NAVI32)\n"
    )
    sd_cpp_backend._remember_device_listing(str(binary), None)
    assert sd_cpp_backend.sd_cpp_listed_device(str(binary), probe = False) == (
        "vulkan",
        "AMD Radeon RX 7800 XT (RADV NAVI32)",
    )
    # A replaced binary is never answered from the old listing.
    binary.write_bytes(b"replaced build")
    os.utime(binary, ns = (1, 1))
    assert sd_cpp_backend.sd_cpp_listed_device(str(binary), probe = False) == (None, None)


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
    monkeypatch.setattr(sd_cpp_backend.os, "name", "posix")
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
    listing = (
        "CPU\tAMD Ryzen 9\nVulkan0\tAMD Radeon RX 7800 XT (RADV NAVI32)\nVulkan1\tIntel(R) UHD\n"
    )
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: listing)
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli") == (
        "vulkan",
        "AMD Radeon RX 7800 XT (RADV NAVI32)",
    )
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli", "Vulkan1") == ("vulkan", "Intel(R) UHD")
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: "CPU\tAMD Ryzen 9\n")
    assert sd_cpp_backend.sd_cpp_listed_device("/x/sd-cli") == (None, None)


def test_the_fallback_reason_without_missing_libraries_points_at_the_card(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend.os, "name", "posix")
    monkeypatch.setattr(sd_cpp_backend, "rocm_runtime_missing_sonames", lambda: [])
    assert "gfx target" in sd_cpp_backend.accelerator_fallback_reason("rocm")
