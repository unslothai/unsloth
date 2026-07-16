# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the GGUF mmproj load toggle."""

from core.inference.llama_cpp import LlamaCppBackend


def _loaded_backend(*, load_mmproj: bool = True) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "unsloth/Qwen3.6-35B-Vision-MTP-GGUF"
    backend._gguf_path = None
    backend._hf_variant = "IQ4_NL"
    backend._requested_n_ctx = 0
    backend._cache_type_kv = "q8_0"
    backend._requested_spec_mode = "auto"
    backend._spec_fallback_reason = None
    backend._speculative_type = "draft-mtp"
    backend._spec_draft_n_max = 2
    backend._tensor_parallel = False
    backend._layer_preserves_tensor_intent = False
    backend._chat_template_override = None
    backend._extra_args = []
    backend._load_mmproj = load_mmproj
    return backend


def test_mmproj_toggle_participates_in_loaded_state_match():
    backend = _loaded_backend(load_mmproj = True)

    assert backend._already_in_target_state(
        model_identifier = "unsloth/Qwen3.6-35B-Vision-MTP-GGUF",
        hf_variant = "IQ4_NL",
        n_ctx = 0,
        cache_type_kv = "q8_0",
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = [],
        is_vision = True,
        load_mmproj = True,
    )

    assert not backend._already_in_target_state(
        model_identifier = "unsloth/Qwen3.6-35B-Vision-MTP-GGUF",
        hf_variant = "IQ4_NL",
        n_ctx = 0,
        cache_type_kv = "q8_0",
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = [],
        is_vision = True,
        load_mmproj = False,
    )
