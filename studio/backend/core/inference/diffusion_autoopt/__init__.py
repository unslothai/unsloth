# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion / video auto-optimizer: one orchestrator over Studio's speed primitives.

* ``targets``  denoiser and runner resolution (attribute contracts + by-type discovery).
* ``profile``  static introspection of a denoiser class (repeated blocks, graph-break screen), cached by digest.
* ``patches``  the PatchSpec registry: one install / uninstall lifecycle and one fingerprint util for every
               per-architecture source / forward patch.
* ``plan``     the ordered steps, their predicted decision + reason, the canonical compile-cache kwargs.
* ``apply``    ``run_speed_layer``: the single image + video entry, per-step audit.
* ``runtime``  call-time checks against the runner that actually executes a request.

Import is torch-free; every submodule imports torch / diffusers lazily.
"""

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
