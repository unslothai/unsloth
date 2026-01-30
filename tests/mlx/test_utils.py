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

"""Tests for MLX utilities."""

import sys
import pytest


class TestMLXUtils:
    """Test MLX utility functions."""

    def test_is_mlx_available_returns_bool(self):
        """is_mlx_available should return a boolean."""
        from unsloth.kernels.mlx import is_mlx_available
        
        result = is_mlx_available()
        assert isinstance(result, bool)

    def test_is_mlx_available_false_on_non_darwin(self):
        """is_mlx_available should return False on non-macOS platforms."""
        from unsloth.kernels.mlx.utils import is_mlx_available
        
        # Clear the cache to test fresh
        is_mlx_available.cache_clear()
        
        if sys.platform != "darwin":
            assert is_mlx_available() is False

    def test_get_mlx_version_type(self):
        """get_mlx_version should return str or None."""
        from unsloth.kernels.mlx import get_mlx_version
        
        result = get_mlx_version()
        assert result is None or isinstance(result, str)

    def test_unsloth_mlx_error_is_exception(self):
        """UnslothMLXError should be a proper exception."""
        from unsloth.kernels.mlx import UnslothMLXError
        
        assert issubclass(UnslothMLXError, RuntimeError)
        
        # Should be raiseable
        with pytest.raises(UnslothMLXError):
            raise UnslothMLXError()

    def test_unsloth_mlx_error_default_message(self):
        """UnslothMLXError should have a helpful default message."""
        from unsloth.kernels.mlx import UnslothMLXError
        
        error = UnslothMLXError()
        assert "MLX is not available" in str(error)
        assert "unsloth[apple]" in str(error)

    def test_unsloth_mlx_error_custom_message(self):
        """UnslothMLXError should accept custom messages."""
        from unsloth.kernels.mlx import UnslothMLXError
        
        custom_msg = "Custom error message"
        error = UnslothMLXError(custom_msg)
        assert str(error) == custom_msg

    def test_require_mlx_decorator_exists(self):
        """require_mlx decorator should be importable."""
        from unsloth.kernels.mlx import require_mlx
        
        assert callable(require_mlx)

    def test_require_mlx_raises_when_mlx_unavailable(self):
        """require_mlx should raise UnslothMLXError when MLX is not available."""
        from unsloth.kernels.mlx import require_mlx, UnslothMLXError, is_mlx_available
        
        if not is_mlx_available():
            @require_mlx
            def dummy_function():
                pass
            
            with pytest.raises(UnslothMLXError):
                dummy_function()


class TestMLXModuleImports:
    """Test that the MLX module imports correctly."""

    def test_import_from_kernels_mlx(self):
        """Should be able to import from unsloth.kernels.mlx."""
        from unsloth.kernels.mlx import (
            is_mlx_available,
            get_mlx_version,
            UnslothMLXError,
            require_mlx,
        )
        
        # All should be importable
        assert is_mlx_available is not None
        assert get_mlx_version is not None
        assert UnslothMLXError is not None
        assert require_mlx is not None

    def test_import_does_not_crash(self):
        """Importing mlx module should not crash on any platform."""
        try:
            import unsloth.kernels.mlx
            success = True
        except Exception:
            success = False
        
        assert success, "Import of unsloth.kernels.mlx should not crash"
