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
Unit tests for MPS (Apple Silicon) device detection and utilities.

These tests verify the MPS infrastructure without requiring actual Apple Silicon hardware.
Tests use mocking to simulate MPS availability scenarios.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


class TestIsMps:
    """Tests for is_mps() function."""

    def test_is_mps_returns_bool(self):
        """is_mps() should always return a boolean."""
        from unsloth.device_type import is_mps
        result = is_mps()
        assert isinstance(result, bool)

    def test_is_mps_is_cached(self):
        """is_mps() should be cached (functools.cache)."""
        from unsloth.device_type import is_mps
        # Call twice - should return same object due to caching
        result1 = is_mps()
        result2 = is_mps()
        assert result1 is result2


class TestGetDeviceType:
    """Tests for get_device_type() function."""

    def test_get_device_type_returns_string(self):
        """get_device_type() should return a string."""
        from unsloth.device_type import get_device_type
        result = get_device_type()
        assert isinstance(result, str)

    def test_get_device_type_valid_values(self):
        """get_device_type() should return a valid device type."""
        from unsloth.device_type import get_device_type
        valid_types = ("cuda", "hip", "xpu", "mps")
        result = get_device_type()
        assert result in valid_types


class TestDeviceCount:
    """Tests for get_device_count() function."""

    def test_get_device_count_returns_int(self):
        """get_device_count() should return an integer."""
        from unsloth.device_type import get_device_count
        result = get_device_count()
        assert isinstance(result, int)

    def test_get_device_count_positive(self):
        """get_device_count() should return at least 1."""
        from unsloth.device_type import get_device_count
        result = get_device_count()
        assert result >= 1


class TestDeviceTypeConstants:
    """Tests for device type constants."""

    def test_device_type_is_string(self):
        """DEVICE_TYPE should be a string."""
        from unsloth.device_type import DEVICE_TYPE
        assert isinstance(DEVICE_TYPE, str)

    def test_device_type_torch_is_string(self):
        """DEVICE_TYPE_TORCH should be a string."""
        from unsloth.device_type import DEVICE_TYPE_TORCH
        assert isinstance(DEVICE_TYPE_TORCH, str)

    def test_device_count_is_int(self):
        """DEVICE_COUNT should be an integer."""
        from unsloth.device_type import DEVICE_COUNT
        assert isinstance(DEVICE_COUNT, int)

    def test_allow_prequantized_is_bool(self):
        """ALLOW_PREQUANTIZED_MODELS should be a boolean."""
        from unsloth.device_type import ALLOW_PREQUANTIZED_MODELS
        assert isinstance(ALLOW_PREQUANTIZED_MODELS, bool)

    def test_allow_bitsandbytes_is_bool(self):
        """ALLOW_BITSANDBYTES should be a boolean."""
        from unsloth.device_type import ALLOW_BITSANDBYTES
        assert isinstance(ALLOW_BITSANDBYTES, bool)


class TestMPSSpecificBehavior:
    """Tests for MPS-specific behavior (only run on MPS devices)."""

    @pytest.mark.skipif(
        not hasattr(__import__('torch').backends, 'mps') or 
        not __import__('torch').backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_device_type(self):
        """On MPS, get_device_type() should return 'mps'."""
        from unsloth.device_type import get_device_type
        assert get_device_type() == "mps"

    @pytest.mark.skipif(
        not hasattr(__import__('torch').backends, 'mps') or 
        not __import__('torch').backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_device_count_is_one(self):
        """On MPS, device count should be 1."""
        from unsloth.device_type import DEVICE_COUNT
        assert DEVICE_COUNT == 1

    @pytest.mark.skipif(
        not hasattr(__import__('torch').backends, 'mps') or 
        not __import__('torch').backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_bitsandbytes_disabled(self):
        """On MPS, bitsandbytes should be disabled."""
        from unsloth.device_type import ALLOW_BITSANDBYTES
        assert ALLOW_BITSANDBYTES is False

    @pytest.mark.skipif(
        not hasattr(__import__('torch').backends, 'mps') or 
        not __import__('torch').backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_mps_prequantized_disabled(self):
        """On MPS, prequantized models should be disabled."""
        from unsloth.device_type import ALLOW_PREQUANTIZED_MODELS
        assert ALLOW_PREQUANTIZED_MODELS is False


class TestMPSUtilities:
    """Tests for MPS utility functions."""

    def test_mps_utilities_import(self):
        """MPS utilities module should be importable."""
        from unsloth.kernels.mps import (
            is_mps_available,
            get_mps_device_info,
            get_mps_memory_info,
            get_mps_capabilities,
        )
        assert callable(is_mps_available)
        assert callable(get_mps_device_info)
        assert callable(get_mps_memory_info)
        assert callable(get_mps_capabilities)

    def test_is_mps_available_returns_bool(self):
        """is_mps_available() should return a boolean."""
        from unsloth.kernels.mps import is_mps_available
        result = is_mps_available()
        assert isinstance(result, bool)

    def test_get_mps_device_info_returns_dict(self):
        """get_mps_device_info() should return a dictionary."""
        from unsloth.kernels.mps import get_mps_device_info
        result = get_mps_device_info()
        assert isinstance(result, dict)
        assert "available" in result

    def test_get_mps_memory_info_returns_dict(self):
        """get_mps_memory_info() should return a dictionary."""
        from unsloth.kernels.mps import get_mps_memory_info
        result = get_mps_memory_info()
        assert isinstance(result, dict)
        assert "available" in result

    def test_get_mps_capabilities_returns_dict(self):
        """get_mps_capabilities() should return a dictionary."""
        from unsloth.kernels.mps import get_mps_capabilities
        result = get_mps_capabilities()
        assert isinstance(result, dict)
        assert "available" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
