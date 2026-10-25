# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared attribute surface for llama.cpp backend doubles; new attributes go here, not re-declared."""

from __future__ import annotations

from typing import Optional

from models.inference import _InferenceRuntimeFields

# From the model so new Optional fields need no edit here.
_RUNTIME_FIELDS = frozenset(_InferenceRuntimeFields.model_fields)


class FakeLlamaCppBackend:
    """Attributes routes/inference.py reads off a loaded GGUF backend; subclasses override per scenario."""

    is_loaded = True
    model_identifier = "test/model.gguf"
    is_vision = False
    supports_tools = False
    # None matches the real property before a model is loaded.
    context_length: Optional[int] = None

    # Runtime fields /status mirrors, at LlamaCppBackend.__init__ values (None is rejected).
    is_diffusion = False
    supports_reasoning = False
    reasoning_always_on = False
    reasoning_style = "enable_thinking"
    reasoning_effort_levels: list = []
    reasoning_budget = -1
    reasoning_budget_message = ""
    supports_preserve_thinking = False
    tensor_parallel = False
    gpu_memory_mode = "auto"
    gpu_layers = -1
    n_cpu_moe = 0
    n_moe_layers = 0
    gpu_backend_unavailable = False
    offload_overridden = False
    _is_audio = False
    _has_audio_input = False
    _has_video_input = False
    _disable_vision = False
    _vision_disabled_by_user = False
    _requested_reasoning_budget = -1
    _requested_reasoning_budget_message = ""
    # Read directly, so an absent one is an AttributeError before the drift check.
    requested_spec_mode = None
    requested_parallel_slots = 1
    effective_parallel_slots = 1
    requested_n_ctx = 0
    requested_extra_args = None
    spec_fallback_reason = None
    spec_drafter_kind = None
    last_load_warning = None

    def __getattr__(self, name):
        """Only runtime fields answer: a blanket None stops callers' defaults, since None is not absent."""
        if name not in _RUNTIME_FIELDS:
            raise AttributeError(name)
        try:
            return object.__getattribute__(self, f"_{name}")
        except AttributeError:
            return None
