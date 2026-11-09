# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU / MPS wide-tile decodes size their tiles from the live host budget instead of always taking the 32-latent
floor (64x64 latents: 9 floor tiles, 9216 decoded latents, vs one 4096-latent tile)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_memory as dm  # noqa: E402
from core.inference import diffusion_vae_tiling as vt  # noqa: E402

MIB = 2**20


class _Decoder(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype = dtype))


def _vae(dtype = torch.float32, cls_name = "AutoencoderKLQwenImage"):
    vae = type(cls_name, (), {})()
    vae.decoder = _Decoder(dtype)
    vae._unsloth_wide_geometry = (8, 32, 16)
    vae._unsloth_wide_stock_tile = 32
    return vae


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(vt.MAX_TILE_ENV, raising = False)
    monkeypatch.delenv(vt.HOST_BUDGET_ENV, raising = False)


def _memory(
    monkeypatch,
    kind,
    free_mib,
    total_mib,
    working_set = None,
    in_use = 0,
):
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (total_mib, free_mib))
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: free_mib)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: total_mib)
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: None)
    monkeypatch.setattr(
        dm, "_metal_working_set_mib", lambda: (working_set, in_use) if working_set else (None, 0)
    )


def _expected(free_mib, total_mib, z, elem, reserve):
    out = 4 * 4 * z.shape[0] * z.shape[-2] * z.shape[-1] * 64 / MIB
    per_latent = vt.decode_mib_per_latent(8, False, "AutoencoderKLQwenImage") * elem / 2
    return max(0, int(vt.FREE_FRACTION * (free_mib - reserve - out) / per_latent))


def test_cpu_budget_engages_and_grows_the_tiles(monkeypatch):
    _memory(monkeypatch, "cpu", free_mib = 48 * 1024, total_mib = 64 * 1024)
    vae, z = _vae(torch.float32), torch.zeros(1, 16, 1, 64, 64)
    budget = vt.decode_tile_budget(vae, z)
    # system memory keeps max(1 GiB, 10% of total) back; fp32 prices 2x the bf16 coefficient
    assert budget == _expected(48 * 1024, 64 * 1024, z, 4, max(1024, int(64 * 1024 * 0.10)))
    assert budget >= 64 * 64
    assert vt.choose_tiles(64, 64, budget) == (64, 64)
    assert vt.choose_tiles(64, 64, None) == (32, 32)


def test_tight_cpu_memory_stays_at_the_floor(monkeypatch):
    _memory(monkeypatch, "cpu", free_mib = 1500, total_mib = 8 * 1024)
    vae, z = _vae(torch.float32), torch.zeros(1, 16, 1, 64, 64)
    budget = vt.decode_tile_budget(vae, z)
    assert budget is not None and budget < 32 * 32
    assert vt.choose_tiles(64, 64, budget) == (32, 32)


def test_kill_switch_restores_the_floor(monkeypatch):
    _memory(monkeypatch, "cpu", free_mib = 48 * 1024, total_mib = 64 * 1024)
    monkeypatch.setenv(vt.HOST_BUDGET_ENV, "0")
    vae, z = _vae(), torch.zeros(1, 16, 1, 64, 64)
    assert vt.decode_tile_budget(vae, z) is None
    assert vt.choose_tiles(64, 64, vt.decode_tile_budget(vae, z)) == (32, 32)


def test_explicit_max_tile_still_wins(monkeypatch):
    _memory(monkeypatch, "cpu", free_mib = 48 * 1024, total_mib = 64 * 1024)
    monkeypatch.setenv(vt.MAX_TILE_ENV, "40")
    assert vt.decode_tile_budget(_vae(), torch.zeros(1, 16, 1, 64, 64)) == 1600


def test_mps_reads_the_planners_unified_budget(monkeypatch):
    """MPS takes snapshot_device_memory's reading, as the load planner does: Metal's working set less what Metal
    holds, widened to the RAM reading only when that leaves more."""
    z = types.SimpleNamespace(device = torch.device("mps"), shape = (1, 16, 1, 64, 64), dim = lambda: 5)
    vae = _vae(torch.bfloat16)
    # 64 GB Mac, 20 GiB RAM free, 48 GiB working set of which Metal holds 30: the working set (18 GiB) wins.
    _memory(monkeypatch, "mps", 20 * 1024, 64 * 1024, working_set = 48 * 1024, in_use = 30 * 1024)
    assert vt.decode_tile_budget(vae, z) == _expected(
        18 * 1024, 48 * 1024, z, 2, int(48 * 1024 * 0.15)
    )
    # No working-set reading: RAM less the unified reserve.
    _memory(monkeypatch, "mps", 40 * 1024, 64 * 1024)
    assert vt.decode_tile_budget(vae, z) == _expected(
        40 * 1024, 64 * 1024, z, 2, max(2048, int(64 * 1024 * 0.20))
    )


def test_cpu_budget_is_capped_by_the_cgroup_and_windows_commit(monkeypatch):
    """A container's enforcing limit, not host RAM: past it the process is killed, with no OOM to retry on."""
    _memory(monkeypatch, "cpu", free_mib = 48 * 1024, total_mib = 64 * 1024)
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 3 * 1024)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: 4 * 1024)
    vae, z = _vae(torch.float32), torch.zeros(1, 16, 1, 64, 64)
    capped = vt.decode_tile_budget(vae, z)
    assert capped == _expected(3 * 1024, 4 * 1024, z, 4, 1024)
    assert capped < 64 * 64 and vt.choose_tiles(64, 64, capped) != (64, 64)
    _memory(monkeypatch, "cpu", free_mib = 48 * 1024, total_mib = 64 * 1024)
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: 3 * 1024)
    assert vt.decode_tile_budget(vae, z) == _expected(3 * 1024, 64 * 1024, z, 4, 64 * 1024 // 10)


def test_no_reading_keeps_the_floor(monkeypatch):
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (None, None))
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: None)
    assert vt.decode_tile_budget(_vae(), torch.zeros(1, 16, 1, 64, 64)) is None


def test_other_devices_never_take_the_host_reading(monkeypatch):
    called = []
    monkeypatch.setattr(dm, "snapshot_device_memory", lambda t: called.append(t))
    for kind in ("xpu", "meta"):
        z = types.SimpleNamespace(
            device = torch.device(kind), shape = (1, 16, 1, 64, 64), dim = lambda: 5
        )
        assert vt._host_free_mib(_vae(), z) is None
    assert not called


def test_cuda_path_is_unchanged(monkeypatch):
    """A CUDA latent never reaches the host reading; its budget is the mem_get_info formula."""
    monkeypatch.setattr(vt, "_host_free_mib", lambda *a: pytest.fail("host reading on CUDA"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (20 * 2**30, 24 * 2**30))
    monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: 0)
    monkeypatch.setattr(dm, "user_budget_reserve_bytes", lambda device, total: 0)
    z = types.SimpleNamespace(
        device = torch.device("cuda", 0), shape = (1, 16, 1, 64, 64), dim = lambda: 5
    )
    vae = _vae(torch.bfloat16)
    out = 4 * 4 * 64 * 64 * 64 / MIB
    assert vt.decode_tile_budget(vae, z) == int(
        vt.FREE_FRACTION
        * (20 * 1024 - out)
        / vt.decode_mib_per_latent(8, False, "AutoencoderKLQwenImage")
    )


def test_floor_shortfall_and_untiled_release_stay_cuda_only(monkeypatch):
    """The floor is still always tried off CUDA, and the untiled-decode release never fires on CPU."""
    _memory(monkeypatch, "cpu", free_mib = 100, total_mib = 8 * 1024)
    vae, z = _vae(), torch.zeros(1, 16, 1, 64, 64)
    assert vt.floor_shortfall(vae, z) is None
    assert vt._free_mib(vae, z) is None
