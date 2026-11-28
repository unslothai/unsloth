# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows integrated CUDA (RTX Spark N1X): the driver counts the carve-out plus shared RAM
(112,350 MiB on a 64 GB part) and every GPU byte is charged to Windows commit, so llama.cpp rows
are capped at commit available less the iGPU reserve. Discrete cards and Linux are unchanged."""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.llama_cpp as llama_cpp
from core.inference import diffusion_memory
from core.inference.llama_cpp import LlamaCppBackend, _IGPU_HOST_RESERVE_MIB
from utils import host_memory

GIB_MIB = 1024
# N1X device readings (MiB): CUDA pool, nvidia-smi carve-out, commit limit (64 GB RAM + pagefile).
N1X_CUDA_TOTAL = 112350
N1X_CUDA_FREE_IDLE = 112150
N1X_SMI_TOTAL = 64000
N1X_SMI_FREE_IDLE = 64000 - 1033
N1X_COMMIT_LIMIT = 74732
N1X_COMMIT_IDLE = int(58.98 * GIB_MIB)  # ullAvailPageFile at idle
N1X_PHYS_AVAIL = int(50.6 * GIB_MIB)  # ullAvailPhys / MemAvailable, unmoved by GPU allocations
# 40 GiB of GPU memory held: commit falls 1:1, OS available does not move, the carve-out fills.
HELD_COMMIT = int(17.70 * GIB_MIB)
HELD_CUDA_FREE = int(69.48 * GIB_MIB)
HELD_SMI_FREE = max(0, N1X_SMI_FREE_IDLE - 40 * GIB_MIB)


def _host(
    monkeypatch,
    *,
    platform,
    integrated,
    commit = None,
    pool = N1X_CUDA_TOTAL,
    limit = N1X_COMMIT_LIMIT,
    order = "PCI_BUS_ID",
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FIT_HEADROOM_MIB", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FREE_MIB", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_POOL_MIB", {})
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: N1X_PHYS_AVAIL)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_gpu_ids",
        staticmethod(lambda: {0} if integrated else set()),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_pool_total_mib",
        staticmethod(lambda: {0: pool} if integrated else {}),
    )
    monkeypatch.setattr(
        host_memory,
        "windows_commit_mib",
        lambda: None if commit is None else (commit, limit),
    )
    monkeypatch.delenv(diffusion_memory._WIN_IGPU_COMMIT_CAP_ENV, raising = False)
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)


def _torch_rows(monkeypatch, free_mib, total_mib):
    torch = types.SimpleNamespace(
        version = types.SimpleNamespace(hip = None),
        cuda = types.SimpleNamespace(
            is_available = lambda: True, mem_get_info = None, device_count = lambda: 1
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set())
    )
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware, "trusted_mem_get_info", lambda ordinal = 0: (free_mib << 20, total_mib << 20)
    )
    return LlamaCppBackend._get_gpu_memory_torch(None, True)


@pytest.mark.parametrize(
    ("commit", "smi_free", "cuda_free"),
    [
        (N1X_COMMIT_IDLE, N1X_SMI_FREE_IDLE, N1X_CUDA_FREE_IDLE),
        (HELD_COMMIT, HELD_SMI_FREE, HELD_CUDA_FREE),
    ],
    ids = ["idle", "40GiB_held"],
)
def test_n1x_rows_are_capped_at_commit(monkeypatch, commit, smi_free, cuda_free):
    _host(monkeypatch, platform = "win32", integrated = True, commit = commit)
    expected = (0, commit - _IGPU_HOST_RESERVE_MIB, N1X_COMMIT_LIMIT)

    smi = LlamaCppBackend._widen_integrated_cuda_rows([(0, smi_free, N1X_SMI_TOTAL)])
    torch_rows = _torch_rows(monkeypatch, cuda_free, N1X_CUDA_TOTAL)

    assert smi == [expected]
    assert torch_rows == [expected]
    # The child's --fit reads CUDA free, so the gap to commit is kept free on top of its margin.
    assert LlamaCppBackend._WINDOWS_COMMIT_FIT_HEADROOM_MIB == {0: cuda_free - commit}


def test_the_cap_never_credits_more_than_the_driver_has_free(monkeypatch):
    # A large page file: commit available is above CUDA free, so CUDA free is the ceiling.
    _host(monkeypatch, platform = "win32", integrated = True, commit = 90000, limit = 150000)
    smi = LlamaCppBackend._widen_integrated_cuda_rows([(0, HELD_SMI_FREE, N1X_SMI_TOTAL)])
    torch_rows = _torch_rows(monkeypatch, HELD_CUDA_FREE, N1X_CUDA_TOTAL)
    assert torch_rows == [(0, HELD_CUDA_FREE - _IGPU_HOST_RESERVE_MIB, N1X_CUDA_TOTAL)]
    # nvidia-smi sees the carve-out only; its used part is counted out of the pool.
    assert smi[0][1] <= HELD_CUDA_FREE - _IGPU_HOST_RESERVE_MIB


def test_a_single_n1x_is_capped_under_fastest_first(monkeypatch):
    _host(monkeypatch, platform = "win32", integrated = True, commit = HELD_COMMIT, order = "")
    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, HELD_SMI_FREE, N1X_SMI_TOTAL)])
    assert rows == [(0, HELD_COMMIT - _IGPU_HOST_RESERVE_MIB, N1X_COMMIT_LIMIT)]


def test_fastest_first_still_refuses_the_join_off_windows(monkeypatch):
    _host(monkeypatch, platform = "linux", integrated = True, order = "")
    row = [(0, HELD_SMI_FREE, N1X_SMI_TOTAL)]
    assert LlamaCppBackend._widen_integrated_cuda_rows(list(row)) == row


def test_an_unsized_pool_keeps_its_free_reading_under_the_cap(monkeypatch):
    _host(monkeypatch, platform = "win32", integrated = True, commit = HELD_COMMIT, pool = 0)
    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, 8000, 0)])
    assert rows == [(0, 8000 - _IGPU_HOST_RESERVE_MIB, 0)]


def test_the_fit_target_keeps_the_commit_gap_free(tmp_path, monkeypatch):
    from core.inference.llama_cpp import GgufLoadIntent

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    targets = []
    for gap in (0, 53023):
        backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),))
        monkeypatch.setattr(
            LlamaCppBackend, "_WINDOWS_COMMIT_FIT_HEADROOM_MIB", {0: gap} if gap else {}
        )
        intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
        targets.append(h.last(_spawn_cmd(backend, intent), "--fit-target"))
    assert targets[0] is None  # llama.cpp's default margin, unchanged
    assert targets[1] == str(int(llama_cpp._LLAMA_FIT_TARGET_DEFAULT_MIB) + 53023)


def test_n1x_without_a_commit_reading_keeps_the_shared_pool_sizing(monkeypatch):
    _host(monkeypatch, platform = "win32", integrated = True, commit = None)
    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, N1X_SMI_FREE_IDLE, N1X_SMI_TOTAL)])
    # Unchanged rule: max(free, MemAvailable) less the reserve, floored at the driver free.
    assert rows == [(0, N1X_SMI_FREE_IDLE, N1X_CUDA_TOTAL)]


def test_the_diffusion_opt_out_also_turns_the_llama_cap_off(monkeypatch):
    _host(monkeypatch, platform = "win32", integrated = True, commit = N1X_COMMIT_IDLE)
    monkeypatch.setenv(diffusion_memory._WIN_IGPU_COMMIT_CAP_ENV, "0")
    rows = _torch_rows(monkeypatch, N1X_CUDA_FREE_IDLE, N1X_CUDA_TOTAL)
    assert rows == [(0, N1X_CUDA_FREE_IDLE - _IGPU_HOST_RESERVE_MIB, N1X_CUDA_TOTAL)]


def test_a_windows_discrete_card_is_unchanged(monkeypatch):
    _host(monkeypatch, platform = "win32", integrated = False, commit = HELD_COMMIT)
    smi_rows = [(0, 30000, 32607)]
    assert LlamaCppBackend._widen_integrated_cuda_rows(list(smi_rows)) == smi_rows
    assert _torch_rows(monkeypatch, 30000, 32607) == smi_rows


def test_linux_gb10_is_unchanged(monkeypatch):
    # DGX Spark GB10: 128 GB unified, no commit concept; the commit reader must not be consulted.
    def _forbidden():
        raise AssertionError("windows_commit_mib read on Linux")

    _host(monkeypatch, platform = "linux", integrated = True, pool = 122570)
    monkeypatch.setattr(host_memory, "windows_commit_mib", _forbidden)
    avail = N1X_PHYS_AVAIL
    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, 40000, 122570)])
    assert rows == [(0, max(avail - _IGPU_HOST_RESERVE_MIB, 40000), 122570)]
    torch_rows = _torch_rows(monkeypatch, 40000, 122570)
    assert torch_rows == [(0, avail - _IGPU_HOST_RESERVE_MIB, 122570)]
    assert llama_cpp._windows_integrated_cuda_commit_mib(40000, 122570) is None


def test_diffusion_fallback_never_credits_past_the_device_total():
    """diffusion_memory's fallback max(free, min(available, total)) runs only without a commit
    reading; with free <= total it cannot exceed the device total."""
    from unittest.mock import patch
    for free, avail, total in [
        (40000, 51814, 122570),
        (112150, 51814, 112350),
        (2000, 200000, 8192),
    ]:
        with (
            patch.object(diffusion_memory, "_available_system_memory_mib", return_value = avail),
            patch.object(diffusion_memory, "_cgroup_available_memory_mib", return_value = None),
            patch.object(diffusion_memory, "_cgroup_memory_limit_mib", return_value = None),
        ):
            credited, capacity = diffusion_memory._unified_reclaimable_memory_mib(free, total)
        assert credited <= total and capacity == total


def test_a_vulkan_launch_ignores_a_gap_a_cuda_probe_left_behind(tmp_path, monkeypatch):
    """After a CUDA load and a switch to Vulkan, ordinal 0 is a Vulkan device, not the N1X."""
    from core.inference.llama_cpp import GgufLoadIntent

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    targets = []
    for gap in (0, 53023):
        backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),), vulkan = True)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_WINDOWS_COMMIT_FIT_HEADROOM_MIB",
            {0: gap} if gap else {},
            raising = False,
        )
        intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
        targets.append(h.last(_spawn_cmd(backend, intent), "--fit-target"))
    assert targets[1] == targets[0]


def _mixed_host(
    tmp_path,
    monkeypatch,
    *,
    order = "PCI_BUS_ID",
):
    """An N1X (id 0) beside a discrete card (id 1), the default memory slider."""
    from . import _sidecar_fit_helpers as h

    B = LlamaCppBackend
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    for k in ("CUDA_VISIBLE_DEVICES", "LLAMA_ARG_DEVICE", "LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_RPC"):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setattr(llama_cpp, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: False))
    monkeypatch.setattr(B, "_SMI_MIG_INDICES", frozenset())
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    monkeypatch.setattr(B, "_cuda_device_info", staticmethod(lambda: []))
    # The iGPU makes this a shared-memory host, so the trim block that would seed a per-device map is skipped.
    monkeypatch.setattr(
        B, "_fit_trim_host_has_shared_memory_gpu", staticmethod(lambda *_a, **_k: True)
    )
    from utils.hardware import nvidia

    monkeypatch.setattr(nvidia, "cuda_device_info_or_torch", lambda *_a, **_k: [], raising = False)
    backend, gguf = h.cards(tmp_path, memory = ((0, 17_100, 74_732), (1, 30_000, 32_607)))
    monkeypatch.setattr(
        LlamaCppBackend, "_WINDOWS_COMMIT_FIT_HEADROOM_MIB", {0: 53023}, raising = False
    )
    return backend, gguf


def test_a_discrete_card_beside_the_n1x_keeps_its_own_margin(tmp_path, monkeypatch):
    """llama.cpp broadcasts a single --fit-target, so the iGPU's gap goes in a per-device list."""
    from core.inference.llama_cpp import GgufLoadIntent

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    runs = []
    for gap in (None, 53023):
        backend, gguf = _mixed_host(tmp_path, monkeypatch)
        if gap is None:
            monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FIT_HEADROOM_MIB", {})
        intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
        runs.append(h.margins(_spawn_cmd(backend, intent)))
    without, with_gap = runs
    if len(without) == 1:
        without = without * 2  # one broadcast value: every card had the same margin
    # The N1X keeps its gap free; the discrete card's margin does not move.
    assert with_gap == [without[0] + 53023, without[1]]


def test_no_gap_is_broadcast_when_the_device_order_is_unprovable(tmp_path, monkeypatch):
    from core.inference.llama_cpp import GgufLoadIntent

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    backend, gguf = _mixed_host(tmp_path, monkeypatch, order = "FASTEST_FIRST")
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
    targets = h.fit_targets(_spawn_cmd(backend, intent))
    assert all(int(t) < 53023 for t in targets), targets


@pytest.mark.parametrize("vram_fraction", [1.0, 0.97])
def test_a_full_slider_cannot_eat_into_the_commit_gap(tmp_path, monkeypatch, vram_fraction):
    """The slider's margin is clamped at llama.cpp's floor first; the commit gap sits outside it."""
    from core.inference.llama_cpp import GgufLoadIntent

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    gap = N1X_CUDA_FREE_IDLE - N1X_COMMIT_IDLE  # 51,755 MiB on the idle N1X
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.setattr(llama_cpp, "_active_vram_fraction", lambda: vram_fraction)
    margins = []
    for g in (0, gap):
        backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_WINDOWS_COMMIT_FIT_HEADROOM_MIB",
            {0: g} if g else {},
            raising = False,
        )
        intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
        margins.append(h.margin(_spawn_cmd(backend, intent)))
    without, with_gap = margins
    # The fitter keeps the slider's own (floored) margin plus the whole gap free.
    assert with_gap == without + gap


def test_the_launch_reads_cuda_free_after_an_nvidia_smi_probe(tmp_path, monkeypatch):
    """Real order: nvidia-smi answers first, and its CUDA-free estimate (pool total less the
    carve-out's used) sits under CUDA's own 112,150 MiB, so the launch reads CUDA free itself."""
    from core.inference.llama_cpp import GgufLoadIntent
    from utils.hardware import nvidia

    from . import _sidecar_fit_helpers as h
    from .test_llama_cpp_gpu_oom_respawn import _spawn_cmd

    _host(monkeypatch, platform = "win32", integrated = True, commit = N1X_COMMIT_IDLE)
    LlamaCppBackend._widen_integrated_cuda_rows([(0, N1X_SMI_FREE_IDLE, N1X_SMI_TOTAL)])
    gaps = dict(LlamaCppBackend._WINDOWS_COMMIT_FIT_HEADROOM_MIB)
    commit_free = dict(getattr(LlamaCppBackend, "_WINDOWS_COMMIT_FREE_MIB", {}))
    assert gaps[0] < N1X_CUDA_FREE_IDLE - N1X_COMMIT_IDLE  # the estimate errs small
    # Back to this host's platform and GPU mask for the launch harness; keep what the probe recorded.
    monkeypatch.undo()
    monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FIT_HEADROOM_MIB", gaps)
    monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FREE_MIB", commit_free, raising = False)
    reads = []
    monkeypatch.setattr(
        nvidia,
        "get_cuda_free_mib",
        lambda: reads.append(1) or [N1X_CUDA_FREE_IDLE],
        raising = False,
    )
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),))
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
    target = h.last(_spawn_cmd(backend, intent), "--fit-target")
    assert int(target) == int(llama_cpp._LLAMA_FIT_TARGET_DEFAULT_MIB) + (
        N1X_CUDA_FREE_IDLE - N1X_COMMIT_IDLE
    )
    assert reads


def test_an_unread_cuda_free_keeps_the_recorded_gap(monkeypatch):
    from utils.hardware import nvidia

    monkeypatch.setattr(LlamaCppBackend, "_WINDOWS_COMMIT_FREE_MIB", {0: N1X_COMMIT_IDLE})
    monkeypatch.setattr(nvidia, "get_cuda_free_mib", lambda: None)
    assert LlamaCppBackend._live_commit_fit_headroom_mib({0: 50922}) == {0: 50922}
    # A live reading below the estimate never shrinks the gap.
    monkeypatch.setattr(nvidia, "get_cuda_free_mib", lambda: [100_000])
    assert LlamaCppBackend._live_commit_fit_headroom_mib({0: 50922}) == {0: 50922}


def test_a_zero_estimated_gap_is_still_read_live(monkeypatch):
    """Commit 111,500 MiB sits above the nvidia-smi estimate (111,317) but under CUDA free (112,150)."""
    from utils.hardware import nvidia

    _host(monkeypatch, platform = "win32", integrated = True, commit = 111_500, limit = 150_000)
    LlamaCppBackend._widen_integrated_cuda_rows([(0, N1X_SMI_FREE_IDLE, N1X_SMI_TOTAL)])
    assert LlamaCppBackend._WINDOWS_COMMIT_FIT_HEADROOM_MIB == {0: 0}
    monkeypatch.setattr(nvidia, "get_cuda_free_mib", lambda: [N1X_CUDA_FREE_IDLE], raising = False)
    gap = LlamaCppBackend._live_commit_fit_headroom_mib({0: 0})[0]
    # At least CUDA free less commit available; the row's credit (capped at the estimate) can
    # make it larger, never smaller.
    assert gap >= N1X_CUDA_FREE_IDLE - 111_500
    assert gap == N1X_CUDA_FREE_IDLE - LlamaCppBackend._WINDOWS_COMMIT_FREE_MIB[0]
