# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for Apple Silicon hardware detection.

Run:
    python -m pytest tests/test_mps_hardware.py -v
"""

import sys
import os
import pytest
import platform

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Hardware detection tests only run on macOS"
)


class TestGetAppleHardwareInfo:
    """Tests for get_apple_hardware_info() function."""
    
    def test_returns_dict(self):
        """Verify function returns a well-structured dict."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        assert isinstance(info, dict)
        
        # Verify all expected keys are present
        expected_keys = [
            "is_apple_silicon",
            "chip_name",
            "chip_family",
            "chip_variant",
            "total_memory_bytes",
            "total_memory_gb",
            "usable_memory_gb",
            "cpu_cores_total",
            "cpu_cores_performance",
            "cpu_cores_efficiency",
            "gpu_cores",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_is_cached(self):
        """Verify function is cached (lru_cache)."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info1 = get_apple_hardware_info()
        info2 = get_apple_hardware_info()
        
        # Should be the exact same object due to caching
        assert info1 is info2
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_detects_apple_silicon(self):
        """Verify Apple Silicon is correctly detected on ARM Macs."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        assert info["is_apple_silicon"] is True
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_chip_name_detected(self):
        """Verify chip name is detected (M1/M2/M3/M4 variants)."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        chip_name = info["chip_name"]
        
        # Should contain at least "Apple" or "M1"/"M2"/"M3"/"M4"
        assert chip_name != "Unknown", "Chip name should be detected"
        assert "Apple" in chip_name or any(f"M{i}" in chip_name for i in range(1, 10))
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_chip_family_parsed(self):
        """Verify chip family is correctly parsed from chip name."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        family = info["chip_family"]
        
        # Should be one of the known families
        assert family in ["M1", "M2", "M3", "M4", "Unknown"]
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_chip_variant_valid(self):
        """Verify chip variant is one of expected values."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        variant = info["chip_variant"]
        
        valid_variants = ["base", "Pro", "Max", "Ultra"]
        assert variant in valid_variants
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_memory_detected(self):
        """Verify total and usable memory are reasonable values."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        
        total_gb = info["total_memory_gb"]
        usable_gb = info["usable_memory_gb"]
        total_bytes = info["total_memory_bytes"]
        
        # Macs have at least 8GB, at most 192GB
        assert 8 <= total_gb <= 200, f"Total memory {total_gb}GB out of range"
        assert total_bytes > 0, "Total memory bytes should be positive"
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_usable_memory_less_than_total(self):
        """Verify usable memory is less than total (reserved for system)."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        
        total_gb = info["total_memory_gb"]
        usable_gb = info["usable_memory_gb"]
        
        assert usable_gb < total_gb, "Usable memory should be less than total"
        assert usable_gb > 0, "Usable memory should be positive"
        
        # Usable should be at least 60% of total
        ratio = usable_gb / total_gb
        assert ratio >= 0.60, f"Usable ratio {ratio} is too low"
    
    @pytest.mark.skipif(
        platform.processor() != "arm",
        reason="Apple Silicon specific test"
    )
    def test_gpu_cores_estimated(self):
        """Verify GPU cores are estimated."""
        from unsloth.kernels.mps import get_apple_hardware_info
        
        info = get_apple_hardware_info()
        gpu_cores = info["gpu_cores"]
        
        # Should be at least 7 (base M1) and at most 80 (Ultra)
        assert 7 <= gpu_cores <= 100, f"GPU cores {gpu_cores} out of range"


class TestMPSDeviceInfo:
    """Tests for get_mps_device_info() function."""
    
    def test_returns_dict(self):
        """Verify function returns a dict with expected keys."""
        from unsloth.kernels.mps import get_mps_device_info
        
        info = get_mps_device_info()
        assert isinstance(info, dict)
        
        if info.get("available"):
            assert "chip" in info
            assert "mac_version" in info
            assert "pytorch_version" in info


class TestMPSMemoryInfo:
    """Tests for get_mps_memory_info() function."""
    
    def test_returns_dict(self):
        """Verify function returns a dict with expected keys."""
        from unsloth.kernels.mps import get_mps_memory_info
        
        info = get_mps_memory_info()
        assert isinstance(info, dict)
        
        if info.get("available"):
            assert "total_memory_gb" in info
            assert "usable_memory_gb" in info
            assert info["memory_type"] == "unified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
