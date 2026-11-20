# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Apple Silicon host detected as CPU (GGUF-only install, or MLX missing) still runs GGUFs on
Metal, so /api/system's inference inventory must list the unified pool instead of no GPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.hardware import DeviceType
from utils.hardware import hardware as hw

GiB = 1024**3


@pytest.fixture(autouse = True)
def _fresh_working_set_cache():
    hw._apple_metal_working_set_bytes = None
    yield
    hw._apple_metal_working_set_bytes = None


def _metal_info(
    *,
    apple = True,
    device = DeviceType.CPU,
    working_set = lambda: int(10.67 * GiB),
    agx = None,
    total = 16 * GiB,
):
    vm = SimpleNamespace(total = total, available = 6 * GiB)
    with (
        patch.object(hw, "is_apple_silicon", return_value = apple),
        patch.object(hw, "get_device", return_value = device),
        patch.object(hw, "_metal_recommended_working_set_bytes", working_set),
        patch.object(hw, "_read_apple_gpu_stats", return_value = agx or {}),
        patch("psutil.virtual_memory", return_value = vm),
    ):
        return hw.get_apple_metal_inference_gpu_info()


def test_cpu_classified_mac_lists_one_shared_metal_device():
    info = _metal_info(agx = {"vram_used_bytes": int(1.5 * GiB)})
    assert info["available"] is True
    assert info["backend"] == "metal"
    assert info["gguf_gpu_ids_supported"] is False
    (device,) = info["devices"]
    assert device["memory_total_gb"] == 16.0
    assert device["shared_memory"] is True
    assert device["shared_memory_host_backed_gb"] == 16.0
    assert device["metal_working_set_gb"] == pytest.approx(10.67)
    assert device["vram_used_gb"] == 1.5


def test_unknown_working_set_falls_back_to_total_ram_like_the_loader():
    (device,) = _metal_info(working_set = lambda: 0)["devices"]
    assert device["metal_working_set_gb"] == 16.0
    assert "vram_used_gb" not in device


def test_working_set_probe_failure_falls_back_to_total_ram():
    def boom():
        raise RuntimeError("no metal")

    (device,) = _metal_info(working_set = boom)["devices"]
    assert device["metal_working_set_gb"] == 16.0


def test_working_set_is_probed_once():
    calls = []

    def probe():
        calls.append(1)
        return int(10.67 * GiB)

    _metal_info(working_set = probe)
    _metal_info(working_set = probe)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "apple, device",
    [(False, DeviceType.CPU), (True, DeviceType.MLX), (False, DeviceType.CUDA)],
)
def test_other_hosts_keep_their_inventory(apple, device):
    # MLX already publishes the device through the training inventory; off Apple there is no Metal.
    assert _metal_info(apple = apple, device = device) is None


def test_unreadable_total_memory_reports_nothing():
    assert _metal_info(total = 0) is None
