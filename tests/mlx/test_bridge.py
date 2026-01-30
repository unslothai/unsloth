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

"""Tests for MLX bridge tensor conversion functions."""

import sys
import pytest


class TestBridgeImports:
    """Test that bridge module imports correctly on all platforms."""

    def test_import_bridge_functions(self):
        """Bridge functions should be importable without crashing."""
        from unsloth.kernels.mlx.bridge import (
            torch_to_mlx,
            mlx_to_torch,
            mlx_context,
            with_mlx_context,
            synchronize_mps,
            synchronize_mlx,
        )
        
        assert callable(torch_to_mlx)
        assert callable(mlx_to_torch)
        assert callable(synchronize_mps)
        assert callable(synchronize_mlx)

    def test_synchronize_mps_safe_on_non_mac(self):
        """synchronize_mps should not crash on non-Mac platforms."""
        from unsloth.kernels.mlx.bridge import synchronize_mps
        
        # Should run without error on any platform
        synchronize_mps()

    def test_synchronize_mlx_safe_when_unavailable(self):
        """synchronize_mlx should not crash when MLX unavailable."""
        from unsloth.kernels.mlx.bridge import synchronize_mlx
        
        # Should run without error even if MLX not installed
        synchronize_mlx()


@pytest.mark.mlx_only
class TestBridgeConversion:
    """Tests that require MLX to be available."""

    def test_torch_to_mlx_basic(self):
        """Basic torch to MLX conversion should work."""
        import torch
        import mlx.core as mx
        from unsloth.kernels.mlx import torch_to_mlx
        
        tensor = torch.randn(4, 4)
        arr = torch_to_mlx(tensor)
        
        assert isinstance(arr, mx.array)
        assert arr.shape == (4, 4)

    def test_mlx_to_torch_basic(self):
        """Basic MLX to torch conversion should work."""
        import torch
        import mlx.core as mx
        from unsloth.kernels.mlx import mlx_to_torch
        
        arr = mx.ones((4, 4))
        tensor = mlx_to_torch(arr, device="cpu")
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 4)

    def test_roundtrip_conversion(self):
        """Data should survive torch→mlx→torch roundtrip."""
        import torch
        import mlx.core as mx
        from unsloth.kernels.mlx import torch_to_mlx, mlx_to_torch
        
        original = torch.randn(8, 8)
        arr = torch_to_mlx(original)
        recovered = mlx_to_torch(arr, device="cpu")
        
        assert torch.allclose(original, recovered, atol=1e-5)

    def test_mlx_context_manager(self):
        """mlx_context should work as context manager."""
        import torch
        from unsloth.kernels.mlx import mlx_context, torch_to_mlx, mlx_to_torch
        
        tensor = torch.randn(4, 4)
        
        with mlx_context():
            arr = torch_to_mlx(tensor)
            result = mlx_to_torch(arr, device="cpu")
        
        assert torch.allclose(tensor, result, atol=1e-5)

    def test_with_mlx_context_decorator(self):
        """with_mlx_context decorator should work."""
        import torch
        from unsloth.kernels.mlx import with_mlx_context, torch_to_mlx, mlx_to_torch
        
        @with_mlx_context
        def process(t):
            arr = torch_to_mlx(t)
            return mlx_to_torch(arr, device="cpu")
        
        tensor = torch.randn(4, 4)
        result = process(tensor)
        
        assert torch.allclose(tensor, result, atol=1e-5)


class TestBridgeErrorHandling:
    """Test error handling when MLX unavailable."""

    def test_torch_to_mlx_raises_without_mlx(self):
        """torch_to_mlx should raise UnslothMLXError when MLX unavailable."""
        from unsloth.kernels.mlx import is_mlx_available, UnslothMLXError
        from unsloth.kernels.mlx.bridge import torch_to_mlx
        
        if not is_mlx_available():
            import torch
            tensor = torch.randn(4, 4)
            
            with pytest.raises(UnslothMLXError):
                torch_to_mlx(tensor)

    def test_mlx_context_raises_without_mlx(self):
        """mlx_context should raise UnslothMLXError when MLX unavailable."""
        from unsloth.kernels.mlx import is_mlx_available, UnslothMLXError
        from unsloth.kernels.mlx.bridge import mlx_context
        
        if not is_mlx_available():
            with pytest.raises(UnslothMLXError):
                with mlx_context():
                    pass
