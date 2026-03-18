# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace
from unittest.mock import patch

from utils.hardware.hardware import DeviceType, get_gpu_utilization


class TestGetGpuUtilization:
    def test_cpu_returns_empty_gpu_list(self):
        with patch(
            "utils.hardware.hardware.get_device", return_value = DeviceType.CPU
        ):
            result = get_gpu_utilization()

        assert result["available"] is False
        assert result["backend"] == "cpu"
        assert result["gpu_count"] == 0
        assert result["gpus"] == []

    def test_cuda_nvidia_smi_returns_all_gpus(self):
        smi_output = "\n".join(
            [
                "0, NVIDIA RTX 4090, 91, 67, 12000, 24564, 305.12, 450.00",
                "1, NVIDIA RTX 4090, 12, 49, 2048, 24564, 82.50, 450.00",
            ]
        )
        completed = SimpleNamespace(returncode = 0, stdout = smi_output)

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("subprocess.run", return_value = completed),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 2),
            patch("utils.hardware.hardware.get_visible_gpu_count", return_value = 2),
        ):
            result = get_gpu_utilization()

        assert result["available"] is True
        assert result["backend"] == "cuda"
        assert result["gpu_count"] == 2
        assert result["physical_gpu_count"] == 2
        assert result["visible_gpu_count"] == 2
        assert len(result["gpus"]) == 2

        first, second = result["gpus"]
        assert first["index"] == 0
        assert first["name"] == "NVIDIA RTX 4090"
        assert first["gpu_utilization_pct"] == 91.0
        assert first["temperature_c"] == 67.0
        assert first["vram_used_gb"] == 11.72
        assert first["vram_total_gb"] == 23.99
        assert first["power_draw_w"] == 305.12

        assert second["index"] == 1
        assert second["gpu_utilization_pct"] == 12.0
        assert second["temperature_c"] == 49.0
        assert second["vram_used_gb"] == 2.0
        assert second["power_draw_w"] == 82.5

        # Legacy top-level fields still mirror GPU 0 for compatibility.
        assert result["gpu_utilization_pct"] == first["gpu_utilization_pct"]
        assert result["temperature_c"] == first["temperature_c"]
        assert result["vram_used_gb"] == first["vram_used_gb"]
        assert result["power_draw_w"] == first["power_draw_w"]
