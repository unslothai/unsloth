# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detection now surfaces 'high' as the plain thinking tier, so the UI offers None / High / Max."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Faithful slice of the DeepSeek-V4-Flash GGUF template: only 'max' escalates.
DEEPSEEK_V4_TEMPLATE = """
{%- if not thinking is defined -%}
  {%- if enable_thinking is defined -%}
    {%- set thinking = enable_thinking -%}
  {%- else -%}
    {%- set thinking = false -%}
  {%- endif -%}
{%- endif -%}
{%- if not reasoning_effort is defined -%}
  {%- set reasoning_effort = none -%}
{%- endif -%}
{{- bos_token -}}
{%- if thinking and reasoning_effort == 'max' -%}
  {{- 'Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n\\n' -}}
{%- endif -%}
{%- for message in messages -%}
  {{- '<|User|>' + (message['content'] or '') -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- '<|Assistant|>' -}}
  {%- if thinking -%}{{- '<think>' -}}{%- else -%}{{- '</think>' -}}{%- endif -%}
{%- endif -%}
"""


GLM_STYLE_TEMPLATE = """
{%- if enable_thinking -%}
  {%- if reasoning_effort == 'high' -%}{{- 'H' -}}
  {%- elif reasoning_effort == 'max' -%}{{- 'M' -}}
  {%- endif -%}
{%- endif -%}
"""


NON_DEEPSEEK_MAX_ONLY_TEMPLATE = DEEPSEEK_V4_TEMPLATE


HIGH_ONLY_TEMPLATE = """
{%- if enable_thinking and reasoning_effort == 'high' -%}{{- 'H' -}}{%- endif -%}
"""


def _render(template: str, **kwargs) -> str:
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment()
    tmpl = env.from_string(template)
    return tmpl.render(bos_token = "<BOS>", add_generation_prompt = True, **kwargs)


def test_deepseek_v4_surfaces_high_as_plain_tier():
    """Sole 'max' escalation expands to ['high', 'max'] so None/High/Max show."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, "unsloth/DeepSeek-V4-Flash")
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["high", "max"]


def test_deepseek_v4_snapshot_path_keeps_high():
    """A copy loaded from its HF snapshot dir is still DeepSeek-V4."""
    from core.inference.llama_cpp import detect_reasoning_flags

    path = "/scan/models--unsloth--DeepSeek-V4-Flash-GGUF/snapshots/0123abcd"
    flags = detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, path)
    assert flags["reasoning_effort_levels"] == ["high", "max"]


def test_glm_style_two_level_template_unchanged():
    """A template that already names a sub-'max' tier is left as-is."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(GLM_STYLE_TEMPLATE, "unsloth/GLM-5.2")
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["high", "max"]


def test_synthetic_high_scoped_to_deepseek_v4():
    """The same ['max']-only template under a non-deepseek id keeps ['max']."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(NON_DEEPSEEK_MAX_ONLY_TEMPLATE, "vendor/OtherHybrid-GGUF")
    assert flags["reasoning_effort_levels"] == ["max"]


def test_guard_does_not_fire_for_sub_max_singleton():
    """The expansion targets only ['max']; a lone 'high' stays a singleton."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(HIGH_ONLY_TEMPLATE, "custom/high-only")
    assert flags["reasoning_effort_levels"] == ["high"]


def _kwargs_for(flags: dict, enable_thinking, reasoning_effort):
    """Drive the real backend method with a shim carrying the detected flags."""
    from core.inference.llama_cpp import LlamaCppBackend

    shim = object.__new__(LlamaCppBackend)
    shim._supports_reasoning = flags["supports_reasoning"]
    shim._reasoning_always_on = flags["reasoning_always_on"]
    shim._reasoning_style = flags["reasoning_style"]
    shim._reasoning_effort_levels = flags["reasoning_effort_levels"]
    shim._supports_preserve_thinking = flags["supports_preserve_thinking"]
    return shim._request_reasoning_kwargs(enable_thinking, reasoning_effort, None) or {}


def _flags():
    from core.inference.llama_cpp import detect_reasoning_flags
    return detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, "unsloth/DeepSeek-V4-Flash")


def test_none_state_renders_non_thinking():
    """UI 'None' -> enable_thinking=false -> closed </think>, no preamble."""
    kwargs = _kwargs_for(_flags(), enable_thinking = False, reasoning_effort = None)
    assert kwargs == {"enable_thinking": False}
    out = _render(DEEPSEEK_V4_TEMPLATE, messages = [{"role": "user", "content": "hi"}], **kwargs)
    assert out.endswith("</think>")
    assert "Absolute maximum" not in out


def test_high_state_renders_plain_thinking():
    """UI 'High' -> et=true, effort=high -> open <think>, no max preamble."""
    kwargs = _kwargs_for(_flags(), enable_thinking = True, reasoning_effort = "high")
    assert kwargs == {"enable_thinking": True, "reasoning_effort": "high"}
    out = _render(DEEPSEEK_V4_TEMPLATE, messages = [{"role": "user", "content": "hi"}], **kwargs)
    assert out.endswith("<think>")
    assert "Absolute maximum" not in out


def test_max_state_injects_max_preamble():
    """UI 'Max' -> et=true, effort=max -> open <think> plus the max preamble."""
    kwargs = _kwargs_for(_flags(), enable_thinking = True, reasoning_effort = "max")
    assert kwargs == {"enable_thinking": True, "reasoning_effort": "max"}
    out = _render(DEEPSEEK_V4_TEMPLATE, messages = [{"role": "user", "content": "hi"}], **kwargs)
    assert out.endswith("<think>")
    assert "Absolute maximum" in out


def test_high_effort_alone_enables_thinking():
    """API caller sending only reasoning_effort='high' (no enable_thinking) still
    gets thinking on, so the newly exposed High mode renders correctly."""
    kwargs = _kwargs_for(_flags(), enable_thinking = None, reasoning_effort = "high")
    assert kwargs == {"enable_thinking": True, "reasoning_effort": "high"}
    out = _render(DEEPSEEK_V4_TEMPLATE, messages = [{"role": "user", "content": "hi"}], **kwargs)
    assert out.endswith("<think>")
    assert "Absolute maximum" not in out
