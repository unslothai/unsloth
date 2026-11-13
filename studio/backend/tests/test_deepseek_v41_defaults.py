# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DeepSeek-V4.1-Flash-1017 (llama.cpp arch ``deepseek41``).

Sampling comes from the ``deepseek-v4`` family in ``inference_defaults.json``
(temperature 1.0, top_p 1.0), which every V4.1 id shape matches by substring.
The template gates thinking on ``enable_thinking`` / ``thinking_mode`` and maps
``reasoning_effort`` low / medium / high / max to a budget, so the Think menu
offers all four. Like deepseek4, the arch has its own KV cache and no
kv_lora_rank, so the K == V rule is matched by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Faithful slice of the DeepSeek-V4.1 GGUF template: the thinking gate and the
# reasoning_effort budget map.
DEEPSEEK_V41_TEMPLATE = """
{%- if thinking_mode is not defined or thinking_mode is none -%}
    {%- if enable_thinking is defined and enable_thinking is not none -%}
        {%- set thinking_mode = "thinking" if enable_thinking else "chat" -%}
    {%- else -%}
        {%- set thinking_mode = "thinking" -%}
    {%- endif -%}
{%- endif -%}
{%- if reasoning_effort is not defined or reasoning_effort is none -%}{%- set reasoning_effort = "high" -%}{%- endif -%}
{%- set effort_map = {"low": 50, "medium": 62, "high": 75, "max": 100} -%}
{%- if reasoning_effort is string and reasoning_effort in effort_map -%}
    {%- set budget = effort_map[reasoning_effort] -%}
{%- endif -%}
{{- bos_token -}}
{%- for message in messages -%}
    {%- if message.role == "user" -%}{{- "<｜User｜>" + message.content -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<｜Assistant｜>" -}}{%- if thinking_mode == "thinking" -%}{{- "<think>" -}}{%- else -%}{{- "</think>" -}}{%- endif -%}
{%- endif -%}
"""

_IDS = [
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-Q4_K_XL",
    "/scan/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/0123abcd",
    "/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf",
]


@pytest.mark.parametrize("model_id", _IDS)
def test_sampling_defaults(model_id):
    from utils.inference.inference_config import get_family_inference_params

    params = get_family_inference_params(model_id)
    assert params["temperature"] == 1.0
    assert params["top_p"] == 1.0


@pytest.mark.parametrize("model_id", _IDS[:1] + _IDS[2:3])
def test_reasoning_effort_ladder(model_id):
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(DEEPSEEK_V41_TEMPLATE, model_id)
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["low", "medium", "high", "max"]


def test_requires_symmetric_kv():
    from core.inference.llama_cpp import LlamaCppBackend
    assert LlamaCppBackend._requires_symmetric_kv(None, "deepseek41") is True


def test_listed_as_default_gguf():
    from core.inference.defaults import DEFAULT_MODELS_GGUF, DEFAULT_MODELS_STANDARD
    assert "unsloth/DeepSeek-V4.1-Flash-1017-GGUF" in DEFAULT_MODELS_GGUF
    assert "unsloth/DeepSeek-V4.1-Flash-1017-GGUF" in DEFAULT_MODELS_STANDARD
