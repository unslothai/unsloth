# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ROCm/RDNA spoof: present torch as an AMD Radeon (RDNA 2/3/4) card on a
GPU-less host, so hip paths (device_type -> "hip", llama.cpp ROCm bundle) are
testable in CPU-only CI with no AMD hardware. The ROCm sibling of
_zoo_aggressive_cuda_spoof.py: it reuses that spoof's torch.cuda no-op machinery
and overlays the AMD identity (torch.version.hip, gcnArchName, Radeon name).
Apply BEFORE importing unsloth/unsloth_zoo, since DEVICE_TYPE is cached there.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# gfx -> (marketing name, (capability major, minor), torch.version.hip). hip is
# the ROCm build torch was made against (RDNA2/3 ship 6.x; gfx1102/115x/RDNA4 7.2).
_PROFILES: dict[str, tuple[str, tuple[int, int], str]] = {
    "gfx1030": ("AMD Radeon RX 6900 XT", (10, 3), "6.4.43483"),  # RDNA2
    "gfx1031": ("AMD Radeon RX 6700 XT", (10, 3), "6.4.43483"),
    "gfx1032": ("AMD Radeon RX 6600", (10, 3), "6.4.43483"),
    "gfx1034": ("AMD Radeon RX 6400", (10, 3), "6.4.43483"),
    "gfx1100": ("AMD Radeon RX 7900 XTX", (11, 0), "6.4.43483"),  # RDNA3
    "gfx1101": ("AMD Radeon RX 7800 XT", (11, 0), "6.4.43483"),
    "gfx1102": ("AMD Radeon RX 7600", (11, 0), "7.2.1"),
    "gfx1150": ("AMD Radeon 890M", (11, 5), "7.2.1"),  # RDNA3.5 APU
    "gfx1151": ("AMD Radeon 8060S", (11, 5), "7.2.1"),
    "gfx1200": ("AMD Radeon RX 9060 XT", (12, 0), "7.2.1"),  # RDNA4
    "gfx1201": ("AMD Radeon RX 9070 XT", (12, 0), "7.2.1"),
}


def _cuda_spoof():
    """Load the sibling CUDA spoof by path (robust to sys.path), so we reuse its
    torch.cuda machinery instead of duplicating it."""
    if "_zoo_aggressive_cuda_spoof" in sys.modules:
        return sys.modules["_zoo_aggressive_cuda_spoof"]
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_zoo_aggressive_cuda_spoof.py"
    )
    spec = importlib.util.spec_from_file_location("_zoo_aggressive_cuda_spoof", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["_zoo_aggressive_cuda_spoof"] = mod
    return mod


def apply(gfx: str = "gfx1100", device_count: int = 1) -> None:
    """Present torch as `gfx`. Re-callable to switch arch (identity is overlaid;
    the underlying no-op machinery is applied once)."""
    import torch

    if gfx not in _PROFILES:
        raise KeyError(f"Unknown gfx {gfx!r}; known: {', '.join(_PROFILES)}")
    name, cap, hip = _PROFILES[gfx]

    _cuda_spoof().apply()  # is_available/device_count/streams/rng/amp/...

    # Overlay the AMD identity on top of the (NVIDIA-shaped) CUDA spoof.
    torch.version.hip = hip
    torch.version.cuda = None
    torch.cuda.device_count = lambda: device_count
    torch.cuda.get_device_name = lambda *a, **k: name
    torch.cuda.get_device_capability = lambda *a, **k: cap
    torch.cuda.get_arch_list = lambda: [gfx]

    class _Props:
        pass

    _p = _Props()
    _p.name = name
    _p.gcnArchName = f"{gfx}:sramecc-:xnack-"  # ROCm advertises feature flags
    _p.major, _p.minor = cap
    _p.total_memory = 16 * 1024**3
    _p.multi_processor_count = 40
    _p.warp_size = 32  # RDNA wavefront (CDNA is 64)
    _p.is_integrated = gfx in ("gfx1150", "gfx1151")
    _p.is_multi_gpu_board = False
    torch.cuda.get_device_properties = lambda *a, **k: _p


if __name__ == "__main__":
    apply()
    import torch
    print(
        "ROCm spoof applied:",
        torch.version.hip,
        torch.cuda.get_device_properties(0).gcnArchName,
    )
