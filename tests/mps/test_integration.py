# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for MPS (Apple Silicon) support.

These tests verify the MPS integration works correctly across the Unsloth codebase.
MPS-specific tests are skipped on non-MPS systems.
"""

import pytest
import torch


def is_mps_system():
    """Check if running on an MPS-capable system."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


class TestUnslothImport:
    """Tests that Unsloth can be imported on any system."""

    def test_device_type_import(self):
        """device_type module should import without errors."""
        from unsloth.device_type import (
            is_hip,
            is_mps,
            get_device_type,
            DEVICE_TYPE,
            DEVICE_TYPE_TORCH,
            DEVICE_COUNT,
            ALLOW_PREQUANTIZED_MODELS,
            ALLOW_BITSANDBYTES,
        )

        # All exports should be available
        assert is_hip is not None
        assert is_mps is not None
        assert get_device_type is not None
        assert DEVICE_TYPE is not None
        assert DEVICE_TYPE_TORCH is not None
        assert DEVICE_COUNT is not None
        assert ALLOW_PREQUANTIZED_MODELS is not None
        assert ALLOW_BITSANDBYTES is not None

    def test_mps_module_import(self):
        """MPS module should import without errors."""
        from unsloth.kernels.mps import (
            is_mps_available,
            get_mps_device_info,
            get_mps_memory_info,
            get_mps_capabilities,
        )

        # All exports should be available
        assert is_mps_available is not None
        assert get_mps_device_info is not None
        assert get_mps_memory_info is not None
        assert get_mps_capabilities is not None


class TestKernelUtilsImport:
    """Tests that kernel utils imports work on all systems."""

    def test_kernels_utils_import(self):
        """kernels/utils.py should import without errors on any device."""
        # This tests the conditional imports work correctly
        from unsloth.kernels.utils import (
            MAX_FUSED_SIZE,
            next_power_of_2,
            torch_amp_custom_fwd,
            torch_amp_custom_bwd,
            calculate_settings,
        )

        assert MAX_FUSED_SIZE == 65536
        assert callable(next_power_of_2)
        assert torch_amp_custom_fwd is not None
        assert torch_amp_custom_bwd is not None
        assert callable(calculate_settings)

    def test_next_power_of_2(self):
        """next_power_of_2 should work correctly."""
        from unsloth.kernels.utils import next_power_of_2

        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(16) == 16
        assert next_power_of_2(17) == 32

    def test_calculate_settings(self):
        """calculate_settings should return valid block size and warps."""
        from unsloth.kernels.utils import calculate_settings

        block_size, num_warps = calculate_settings(128)
        assert block_size >= 128
        assert block_size & (block_size - 1) == 0  # Power of 2
        assert num_warps >= 4


@pytest.mark.skipif(not is_mps_system(), reason = "MPS not available")
class TestMPSIntegration:
    """Integration tests that only run on MPS systems."""

    def test_mps_tensor_creation(self):
        """Should be able to create tensors on MPS device."""
        tensor = torch.tensor([1.0, 2.0, 3.0], device = "mps")
        assert tensor.device.type == "mps"
        del tensor

    def test_mps_float16_support(self):
        """MPS should support float16."""
        tensor = torch.tensor([1.0], dtype = torch.float16, device = "mps")
        result = tensor + tensor
        assert result.dtype == torch.float16
        del tensor, result

    def test_mps_device_info_on_mps(self):
        """get_mps_device_info should return full info on MPS."""
        from unsloth.kernels.mps import get_mps_device_info

        info = get_mps_device_info()
        assert info["available"] is True
        assert "chip" in info
        assert "mac_version" in info
        assert "pytorch_version" in info

    def test_mps_memory_info_on_mps(self):
        """get_mps_memory_info should return memory info on MPS."""
        from unsloth.kernels.mps import get_mps_memory_info

        info = get_mps_memory_info()
        assert info["available"] is True
        assert info["memory_type"] == "unified"
        assert "total_system_memory_gb" in info

    def test_mps_capabilities_on_mps(self):
        """get_mps_capabilities should return full capabilities on MPS."""
        from unsloth.kernels.mps import get_mps_capabilities

        caps = get_mps_capabilities()
        assert caps["available"] is True
        assert "supports_float16" in caps
        assert "supports_bfloat16" in caps
        assert caps["supports_quantization"] is False
        assert caps["supports_triton"] is False

    def test_unsloth_device_type_is_mps(self):
        """On MPS, DEVICE_TYPE should be 'mps'."""
        from unsloth.device_type import DEVICE_TYPE

        assert DEVICE_TYPE == "mps"

    def test_triton_is_none_on_mps(self):
        """On MPS, triton should be None in kernels/utils."""
        from unsloth.kernels.utils import triton

        assert triton is None


@pytest.mark.skipif(is_mps_system(), reason = "Only run on non-MPS systems")
class TestNonMPSSystems:
    """Tests that verify behavior on non-MPS systems."""

    def test_mps_utilities_return_unavailable(self):
        """MPS utilities should indicate unavailable on non-MPS."""
        from unsloth.kernels.mps import (
            is_mps_available,
            get_mps_device_info,
            get_mps_memory_info,
            get_mps_capabilities,
        )

        assert is_mps_available() is False
        assert get_mps_device_info()["available"] is False
        assert get_mps_memory_info()["available"] is False
        assert get_mps_capabilities()["available"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
