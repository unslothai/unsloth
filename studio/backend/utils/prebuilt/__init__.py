# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared prebuilt-consumer core for the llama.cpp and whisper.cpp installers.

Currently exposes coverage-aware artifact selection (`selection`) and GPU
host-capability detection (`hosts`). See the module docstrings for how the
selection algorithm is lifted from install_llama_prebuilt.py and generalised.
"""

from __future__ import annotations

from .hosts import (
    NvidiaCaps,
    detect_nvidia_caps,
    detect_torch_cuda_runtime_line,
    parse_cuda_visible_devices,
)
from .selection import (
    SelArtifact,
    artifact_covers_sms,
    blackwell_capable_runtime_lines,
    blackwell_min_toolkit_for_caps,
    compatible_runtime_lines_for_driver,
    cuda_runtime_lines_for_major,
    host_is_blackwell,
    match_rocm_artifact,
    normalize_compute_cap,
    normalize_compute_caps,
    rank_cuda_attempts,
    select_cuda_attempts,
    sm_range,
)

__all__ = [
    "NvidiaCaps",
    "SelArtifact",
    "artifact_covers_sms",
    "blackwell_capable_runtime_lines",
    "blackwell_min_toolkit_for_caps",
    "compatible_runtime_lines_for_driver",
    "cuda_runtime_lines_for_major",
    "detect_nvidia_caps",
    "detect_torch_cuda_runtime_line",
    "host_is_blackwell",
    "match_rocm_artifact",
    "normalize_compute_cap",
    "normalize_compute_caps",
    "parse_cuda_visible_devices",
    "rank_cuda_attempts",
    "select_cuda_attempts",
    "sm_range",
]
