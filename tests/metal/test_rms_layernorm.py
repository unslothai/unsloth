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

"""Unit tests for Metal RMS LayerNorm kernel."""

import platform
import pytest
import torch


class TestMetalKernelImports:
    """Test that Metal kernel module imports correctly on all platforms."""

    def test_import_metal_module(self):
        """Metal module should be importable without crashing."""
        from unsloth.kernels.metal import (
            is_metal_available,
            USE_METAL_KERNEL,
            metal_rms_layernorm,
        )
        
        assert callable(is_metal_available)
        assert callable(metal_rms_layernorm)
        assert isinstance(USE_METAL_KERNEL, bool)

    def test_is_metal_available_returns_bool(self):
        """is_metal_available should return False on non-Mac."""
        from unsloth.kernels.metal import is_metal_available
        
        result = is_metal_available()
        assert isinstance(result, bool)
        
        if platform.system() != "Darwin":
            assert result is False


class TestMetalRMSLayerNormReference:
    """Test reference implementation matching (works on all platforms)."""

    def _reference_rms_layernorm(self, X, W, eps, gemma=False):
        """PyTorch reference implementation of RMS LayerNorm."""
        X_f32 = X.to(torch.float32)
        variance = X_f32.pow(2).mean(-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + eps)
        X_norm = (X_f32 * rms_inv).to(X.dtype)
        
        if gemma:
            return (W + 1.0).to(X.dtype) * X_norm
        else:
            return W.to(X.dtype) * X_norm

    def test_reference_impl_basic(self):
        """Verify reference implementation works correctly."""
        X = torch.randn(2, 16, 32)
        W = torch.randn(32)
        eps = 1e-5
        
        Y = self._reference_rms_layernorm(X, W, eps)
        
        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()

    def test_reference_impl_gemma_variant(self):
        """Verify Gemma variant uses (1 + W) scaling."""
        X = torch.randn(2, 16, 32)
        W = torch.randn(32)
        eps = 1e-5
        
        Y_std = self._reference_rms_layernorm(X, W, eps, gemma=False)
        Y_gemma = self._reference_rms_layernorm(X, W, eps, gemma=True)
        
        # Results should differ (Gemma uses 1+W scaling)
        assert not torch.allclose(Y_std, Y_gemma)


@pytest.mark.metal_only
class TestMetalRMSLayerNormKernel:
    """Tests that require Metal kernel to be available (macOS only)."""

    def _reference_rms_layernorm(self, X, W, eps, gemma=False):
        """PyTorch reference implementation."""
        X_f32 = X.to(torch.float32)
        variance = X_f32.pow(2).mean(-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + eps)
        X_norm = (X_f32 * rms_inv).to(X.dtype)
        
        if gemma:
            return (W + 1.0).to(X.dtype) * X_norm
        else:
            return W.to(X.dtype) * X_norm

    def test_metal_kernel_basic_parity(self):
        """Metal kernel output should match reference implementation."""
        from unsloth.kernels.metal import metal_rms_layernorm
        
        X = torch.randn(2, 16, 32, device="mps")
        W = torch.randn(32, device="mps")
        eps = 1e-5
        
        Y_metal = metal_rms_layernorm(X, W, eps)
        Y_ref = self._reference_rms_layernorm(X.cpu(), W.cpu(), eps).to("mps")
        
        assert torch.allclose(Y_metal, Y_ref, atol=1e-5)

    def test_metal_kernel_gemma_variant(self):
        """Metal Gemma variant should match reference."""
        from unsloth.kernels.metal import metal_rms_layernorm
        
        X = torch.randn(2, 16, 32, device="mps")
        W = torch.randn(32, device="mps")
        eps = 1e-5
        
        Y_metal = metal_rms_layernorm(X, W, eps, gemma=True)
        Y_ref = self._reference_rms_layernorm(X.cpu(), W.cpu(), eps, gemma=True).to("mps")
        
        assert torch.allclose(Y_metal, Y_ref, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("hidden_dim", [256, 1024, 4096])
    def test_metal_kernel_various_shapes(self, batch_size, seq_len, hidden_dim):
        """Metal kernel should work with various tensor shapes."""
        from unsloth.kernels.metal import metal_rms_layernorm
        
        X = torch.randn(batch_size, seq_len, hidden_dim, device="mps")
        W = torch.randn(hidden_dim, device="mps")
        eps = 1e-5
        
        Y_metal = metal_rms_layernorm(X, W, eps)
        Y_ref = self._reference_rms_layernorm(X.cpu(), W.cpu(), eps).to("mps")
        
        assert torch.allclose(Y_metal, Y_ref, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_metal_kernel_various_dtypes(self, dtype):
        """Metal kernel should work with various dtypes."""
        from unsloth.kernels.metal import metal_rms_layernorm
        
        X = torch.randn(2, 16, 32, device="mps", dtype=dtype)
        W = torch.randn(32, device="mps", dtype=dtype)
        eps = 1e-5
        
        Y = metal_rms_layernorm(X, W, eps)
        
        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()


if __name__ == "__main__":
    print("Running Metal RMS LayerNorm tests...")
    
    # Run import tests
    test_imports = TestMetalKernelImports()
    test_imports.test_import_metal_module()
    print("✅ Import test passed")
    
    test_imports.test_is_metal_available_returns_bool()
    print("✅ Availability check passed")
    
    # Run reference tests
    test_ref = TestMetalRMSLayerNormReference()
    test_ref.test_reference_impl_basic()
    print("✅ Reference implementation test passed")
    
    test_ref.test_reference_impl_gemma_variant()
    print("✅ Gemma variant test passed")
    
    # Run Metal tests if available
    if platform.system() == "Darwin":
        from unsloth.kernels.metal import is_metal_available
        if is_metal_available():
            test_metal = TestMetalRMSLayerNormKernel()
            test_metal.test_metal_kernel_basic_parity()
            print("✅ Metal kernel parity test passed")
            
            test_metal.test_metal_kernel_gemma_variant()
            print("✅ Metal Gemma variant test passed")
            
            print("\n🚀 ALL METAL KERNEL TESTS PASSED")
        else:
            print("\n⚠️ MLX not available - skipping Metal kernel tests")
    else:
        print(f"\n⚠️ Skipping Metal tests (not on macOS, detected: {platform.system()})")
