# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen3.8 reasoning detection and published sampling defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


QWEN38_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
  {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
  {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('unsupported') }}
  {%- endif %}
{%- endif %}
"""


def test_template_exposes_low_medium_xhigh_and_off():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(QWEN38_TEMPLATE, "Qwen/Qwen3.8-27B")
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["low", "medium", "xhigh"]


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3.8-27B",
        "unsloth/Qwen3.8-27B-GGUF:Q5_K_M",
        "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q5_K_M.gguf",
    ],
)
def test_published_thinking_sampling_defaults_cover_model_id_shapes(model_id):
    from utils.inference.inference_config import load_inference_config

    config = load_inference_config(model_id)
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 20
    assert config["min_p"] == 0.0
    assert config["presence_penalty"] == 0.0


def test_older_qwen3_8b_keeps_qwen3_defaults():
    from utils.inference.inference_config import load_inference_config
    config = load_inference_config("Qwen/Qwen3-8B")
    assert config["temperature"] == 0.6
