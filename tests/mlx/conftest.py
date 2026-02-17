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

"""Pytest configuration for MLX tests."""

import sys
import pytest


def pytest_configure(config):
    """Configure pytest markers for MLX tests."""
    config.addinivalue_line(
        "markers", "mlx_only: mark test to run only when MLX is available"
    )


def _apply_mac_patches():
    """Apply Mac patches BEFORE any tests run to avoid import errors."""
    if sys.platform == "darwin":
        try:
            from patcher import patch_for_mac
            patch_for_mac(verbose=False)
        except Exception:
            pass


# Apply patches at module import time (before any test imports unsloth)
_apply_mac_patches()


def pytest_collection_modifyitems(config, items):
    """Skip MLX-only tests on non-Mac platforms."""
    if sys.platform != "darwin":
        skip_mlx = pytest.mark.skip(reason="MLX only available on macOS")
        for item in items:
            if "mlx_only" in item.keywords:
                item.add_marker(skip_mlx)
