# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from unittest.mock import patch

import pytest

from utils.hardware import DeviceType
from utils.hardware import hardware as hw

_MEM = {"available": True, "total_gb": 64.0, "device_name": "Apple M3 Max"}


def _mlx_device(working_set):
    with (
        patch.object(hw, "get_gpu_memory_info", return_value = dict(_MEM)),
        patch.object(hw, "_metal_recommended_working_set_bytes", working_set),
    ):
        info = hw._probe_backend_visible_gpu_info(DeviceType.MLX)
    assert info["available"] is True
    (device,) = info["devices"]
    return device


def test_mlx_device_reports_metal_working_set():
    device = _mlx_device(lambda: int(51.84 * 1024**3))
    assert device["metal_working_set_gb"] == pytest.approx(51.84)
    assert device["memory_total_gb"] == 64.0
    assert device["shared_memory"] is True


@pytest.mark.parametrize("value", [0, None])
def test_mlx_device_working_set_unknown_is_none(value):
    assert _mlx_device(lambda: value)["metal_working_set_gb"] is None


def test_mlx_device_working_set_probe_failure_is_none():
    def boom():
        raise RuntimeError("no metal")

    device = _mlx_device(boom)
    assert device["metal_working_set_gb"] is None
    assert device["memory_total_gb"] == 64.0
