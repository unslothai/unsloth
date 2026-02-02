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

"""Pytest configuration for Metal kernel tests."""

import sys
import types
from importlib.machinery import ModuleSpec

import pytest


def perfect_mock(name):
    """Create a perfect mock module for non-Mac platforms."""
    m = types.ModuleType(name)
    m.__spec__ = ModuleSpec(name, None)
    m.__file__ = f"{name}.py"
    m.__path__ = []
    if name == "bitsandbytes":
        m.__version__ = "0.42.0"
        m.functional = types.ModuleType("bitsandbytes.functional")
        m.functional.__spec__ = ModuleSpec("bitsandbytes.functional", None)
    return m


# Mock Triton/bitsandbytes on non-CUDA platforms
if "triton" not in sys.modules:
    sys.modules["triton"] = perfect_mock("triton")
    sys.modules["triton.language"] = perfect_mock("triton.language")
    sys.modules["triton.jit"] = perfect_mock("triton.jit")
    sys.modules["triton.runtime"] = perfect_mock("triton.runtime")
    sys.modules["triton.runtime.jit"] = perfect_mock("triton.runtime.jit")

if "bitsandbytes" not in sys.modules:
    sys.modules["bitsandbytes"] = perfect_mock("bitsandbytes")
    sys.modules["bitsandbytes.functional"] = sys.modules["bitsandbytes"].functional


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "metal_only: mark test to run only on macOS with Metal support"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Metal tests on non-Mac platforms."""
    import platform

    if platform.system() != "Darwin":
        skip_metal = pytest.mark.skip(reason="Metal tests require macOS")
        for item in items:
            if "metal_only" in item.keywords:
                item.add_marker(skip_metal)
