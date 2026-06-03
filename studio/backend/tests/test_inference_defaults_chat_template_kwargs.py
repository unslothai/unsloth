# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the ``chat_template_kwargs`` extension to the per-family
inference defaults registry.

Adding nested template kwargs to ``inference_defaults.json`` lets the
family resolver push values like ``reasoning_effort`` and
``enable_thinking`` for modern reasoning models without requiring every
client to set ``extra_body.chat_template_kwargs`` explicitly.
"""

from utils.inference.inference_config import (
    get_family_inference_params,
    load_inference_config,
)


class TestRegistry:
    def test_gpt_oss_carries_reasoning_effort(self):
        params = get_family_inference_params("unsloth/gpt-oss-120b-GGUF")
        assert "chat_template_kwargs" in params
        assert params["chat_template_kwargs"]["reasoning_effort"] == "medium"

    def test_nemotron_carries_enable_thinking(self):
        params = get_family_inference_params(
            "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF"
        )
        assert "chat_template_kwargs" in params
        assert params["chat_template_kwargs"]["enable_thinking"] is True

    def test_family_without_kwargs_returns_no_field(self):
        # qwen3 family has scalar sampling fields but no chat_template_kwargs.
        params = get_family_inference_params("unsloth/Qwen3-8B-GGUF")
        assert "chat_template_kwargs" not in params

    def test_unknown_family_returns_empty(self):
        params = get_family_inference_params("unsloth/CompletelyMadeUp-99B")
        assert params == {}


class TestLoadInferenceConfig:
    def test_gpt_oss_load_surfaces_dict(self):
        cfg = load_inference_config("unsloth/gpt-oss-120b-GGUF")
        assert cfg["chat_template_kwargs"] == {"reasoning_effort": "medium"}

    def test_nemotron_load_surfaces_dict(self):
        cfg = load_inference_config("unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF")
        assert cfg["chat_template_kwargs"] == {"enable_thinking": True}

    def test_qwen3_load_returns_none_for_kwargs(self):
        cfg = load_inference_config("unsloth/Qwen3-8B-GGUF")
        assert cfg["chat_template_kwargs"] is None

    def test_load_returns_a_fresh_dict_per_call(self):
        # Mutating the returned dict must not poison the registry.
        a = load_inference_config("unsloth/gpt-oss-120b-GGUF")
        a["chat_template_kwargs"]["reasoning_effort"] = "high"
        b = load_inference_config("unsloth/gpt-oss-120b-GGUF")
        assert b["chat_template_kwargs"]["reasoning_effort"] == "medium"

    def test_scalar_sampling_fields_unchanged_when_kwargs_added(self):
        # Adding a nested key must not regress the existing scalar field
        # extraction for families that gained chat_template_kwargs.
        cfg = load_inference_config("unsloth/gpt-oss-120b-GGUF")
        assert cfg["temperature"] == 1.0
        assert cfg["top_p"] == 1.0
        assert cfg["top_k"] == 0

    def test_repeated_calls_reuse_cached_result(self):
        """The passthrough request path calls ``load_inference_config``
        on every chat-completion / Responses request. Cache the heavy
        work (YAML reads + recursive ``rglob`` inside
        ``_has_specific_yaml``) so the hot path doesn't pay the full
        scan each time."""
        from utils.inference.inference_config import (
            _has_specific_yaml,
            _load_inference_config_cached,
        )

        _load_inference_config_cached.cache_clear()
        _has_specific_yaml.cache_clear()

        ident = "unsloth/gpt-oss-120b-GGUF"
        load_inference_config(ident)
        load_inference_config(ident)
        load_inference_config(ident)

        # Three calls, one underlying miss; the rest served from cache.
        info = _load_inference_config_cached.cache_info()
        assert info.misses == 1, info
        assert info.hits >= 2, info


class TestTemperatureBumps:
    def test_devstral_temperature_lowered_to_card_value(self):
        # Devstral-Small-2 card recommends T=0.15.
        cfg = load_inference_config("unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF")
        assert cfg["temperature"] == 0.15

    def test_ministral_temperature_below_one_tenth(self):
        # Ministral-3 card says "temperature below 0.1 for production".
        cfg = load_inference_config("unsloth/Ministral-3-8B-Instruct-2512-GGUF")
        assert cfg["temperature"] < 0.1
