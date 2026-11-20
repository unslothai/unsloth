# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On a real Apple Silicon Mac, MLX, torch.mps and Metal.framework report the same Metal working set.

torch covers hosts without MLX (macOS 13) and Metal.framework a GGUF-only install with neither,
so both must be the number the MLX path has always budgeted from. Skipped anywhere MLX and torch
cannot both be imported.
"""

import platform
import sys

import pytest

from utils.hardware.hardware import (
    _metal_framework_working_set_bytes as _REAL_FRAMEWORK_WORKING_SET_BYTES,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason = "needs Apple Silicon",
)


def test_torch_mps_matches_mlx_working_set(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    if not (mx.metal.is_available() and torch.backends.mps.is_available()):
        pytest.skip("Metal unavailable to MLX or torch")
    if not hasattr(torch.mps, "recommended_max_memory"):
        pytest.skip("torch < 2.5")

    from utils.hardware import hardware

    with_mlx = hardware._metal_recommended_working_set_bytes()
    assert with_mlx == int(
        hardware._mlx_device_info(mx).get("max_recommended_working_set_size") or 0
    )
    assert with_mlx > 0

    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert hardware._metal_recommended_working_set_bytes() == with_mlx

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(hardware, "_metal_framework_working_set", None)
    monkeypatch.setattr(
        hardware, "_metal_framework_working_set_bytes", _REAL_FRAMEWORK_WORKING_SET_BYTES
    )
    assert hardware._metal_recommended_working_set_bytes() == with_mlx
