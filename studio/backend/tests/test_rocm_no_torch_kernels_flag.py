# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system flags a GPU the installed torch has no kernels for.

Training already drops it from automatic selection (#8792), but the UI summed its memory into
the VRAM total and ticked it by default. Hermetic: the torch inventory and the arch gate are
stubbed, so this never reads the host's GPUs.
"""

from __future__ import annotations

import pytest
import utils.hardware.hardware as hw


def _row(
    index,
    name,
    total_gb,
    *,
    unified = False,
):
    return {
        "index": index,
        "visible_ordinal": index,
        "name": name,
        "total_gb": total_gb,
        "used_gb": None,
        "shared_memory": unified,
        "shared_memory_host_backed_gb": None,
        "_rocm_known_unified": unified,
        "_rocm_gfx": "",
        "_cuda_integrated": False,
    }


# 2x R9700 (gfx1201) plus the Ryzen 9950X iGPU (gfx1036), as Windows ROCm reports them.
_ROWS = [
    _row(0, "AMD Radeon AI PRO R9700", 31.86),
    _row(1, "AMD Radeon AI PRO R9700", 31.86),
    _row(2, "AMD Radeon(TM) Graphics", 36.23, unified = True),
]


@pytest.fixture
def rocm_host(monkeypatch):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1, 2])
    monkeypatch.setattr(hw, "_torch_get_device_inventory", lambda ids: [dict(r) for r in _ROWS])
    monkeypatch.setattr(hw, "_rocm_linux_shared_pool_host_gb_by_index", lambda devices: {})
    monkeypatch.setattr(hw, "_windows_rocm_shared_pool_host_gb_by_index", lambda devices: {})
    monkeypatch.setattr(hw, "_backend_visible_devices_env", lambda: None)
    return monkeypatch


def _devices():
    return {
        d["index"]: d for d in hw._probe_backend_visible_gpu_info(hw.DeviceType.CUDA)["devices"]
    }


def test_uncovered_igpu_is_flagged(rocm_host):
    rocm_host.setattr(hw, "IS_ROCM", True)
    rocm_host.setattr(hw, "rocm_gpu_ids_without_torch_kernels", lambda: {2})

    devices = _devices()

    assert devices[2]["no_torch_kernels"] is True
    assert "no_torch_kernels" not in devices[0]
    assert "no_torch_kernels" not in devices[1]


def test_covered_host_carries_no_flag(rocm_host):
    rocm_host.setattr(hw, "IS_ROCM", True)
    rocm_host.setattr(hw, "rocm_gpu_ids_without_torch_kernels", lambda: set())

    assert not any("no_torch_kernels" in d for d in _devices().values())


def test_non_rocm_host_never_asks_the_arch_gate(rocm_host):
    def _forbidden():
        raise AssertionError("the ROCm arch gate ran on a non-ROCm host")

    rocm_host.setattr(hw, "IS_ROCM", False)
    rocm_host.setattr(hw, "rocm_gpu_ids_without_torch_kernels", _forbidden)

    assert not any("no_torch_kernels" in d for d in _devices().values())
