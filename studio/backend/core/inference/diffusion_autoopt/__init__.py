# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion/video auto-optimizer orchestrator; import is torch-free, submodules load torch lazily."""

from __future__ import annotations

from .apply import (
    VIDEO_COMPILE_CACHE_DECLINE,
    Audit,
    finalize,
    install_patches,
    new_audit,
    record_load_steps,
    run_speed_layer,
)
from .patches import (
    GROUP_EAGER,
    GROUP_FUSED_DIT,
    GROUP_VIDEO_FUSED,
    STAGE_LOAD,
    STAGE_LOAD_EAGER,
    PatchContext,
    PatchSpec,
    body_has,
    body_has_once,
    register,
    source_digest,
    splice_source,
    uninstall_group,
)
from .plan import build_plan, compile_cache_kwargs

__all__ = [
    "VIDEO_COMPILE_CACHE_DECLINE",
    "Audit",
    "GROUP_EAGER",
    "GROUP_FUSED_DIT",
    "GROUP_VIDEO_FUSED",
    "PatchContext",
    "PatchSpec",
    "STAGE_LOAD",
    "STAGE_LOAD_EAGER",
    "body_has",
    "body_has_once",
    "build_plan",
    "compile_cache_kwargs",
    "finalize",
    "install_patches",
    "new_audit",
    "record_load_steps",
    "register",
    "run_speed_layer",
    "source_digest",
    "splice_source",
    "uninstall_group",
]
