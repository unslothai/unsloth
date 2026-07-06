# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DeepSeek-V4-Flash reasoning toggle: None / High / Max.

The GGUF template gates thinking with ``enable_thinking`` and only branches
``reasoning_effort`` on ``'max'`` (an escalation layered over plain thinking).
Detection used to return the single level ``['max']``, so the UI collapsed to
None / Max and the plain-thinking tier was unreachable. Detection now surfaces
``'high'`` as that plain tier, giving None / High / Max. These tests pin the
classifier, the GLM-style parity case, and the full request-kwargs -> rendered
prompt path for all three states (the model itself is too large to load here).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jinja2

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Faithful slice of the DeepSeek-V4-Flash GGUF template: the enable_thinking
# gate, the sole ``reasoning_effort == 'max'`` escalation, and the plain-think
# fallback. Any non-'max' effort renders as ordinary thinking.
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


# GLM-5.2-style: branches on two effort literals, so 'high' already exists as
# the sub-'max' tier and detection must leave the pair untouched.
GLM_STYLE_TEMPLATE = """
{%- if enable_thinking -%}
  {%- if reasoning_effort == 'high' -%}{{- 'H' -}}
  {%- elif reasoning_effort == 'max' -%}{{- 'M' -}}
  {%- endif -%}
{%- endif -%}
"""


# A template whose sole effort literal is a sub-'max' level: the guard must not
# fire (it targets only the ['max']-alone case).
HIGH_ONLY_TEMPLATE = """
{%- if enable_thinking and reasoning_effort == 'high' -%}{{- 'H' -}}{%- endif -%}
"""


def _render(template: str, **kwargs) -> str:
    env = jinja2.Environment()
    tmpl = env.from_string(template)
    return tmpl.render(bos_token = "<BOS>", add_generation_prompt = True, **kwargs)


# ── Classifier ───────────────────────────────────────────────────────


def test_deepseek_v4_surfaces_high_as_plain_tier():
    """Sole 'max' escalation expands to ['high', 'max'] so None/High/Max show."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, "unsloth/DeepSeek-V4-Flash")
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["high", "max"]


def test_glm_style_two_level_template_unchanged():
    """A template that already names a sub-'max' tier is left as-is."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(GLM_STYLE_TEMPLATE, "unsloth/GLM-5.2")
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["high", "max"]


def test_guard_does_not_fire_for_sub_max_singleton():
    """The expansion targets only ['max']; a lone 'high' stays a singleton."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(HIGH_ONLY_TEMPLATE, "custom/high-only")
    assert flags["reasoning_effort_levels"] == ["high"]


# ── Request kwargs -> rendered prompt, for each UI state ──────────────


def _kwargs_for(flags: dict, enable_thinking, reasoning_effort):
    """Drive the real backend method with a shim carrying the detected flags."""
    from core.inference.llama_cpp import LlamaCppBackend

    shim = SimpleNamespace(
        _supports_reasoning = flags["supports_reasoning"],
        _reasoning_always_on = flags["reasoning_always_on"],
        _reasoning_style = flags["reasoning_style"],
        _reasoning_effort_levels = flags["reasoning_effort_levels"],
        _supports_preserve_thinking = flags["supports_preserve_thinking"],
    )
    build = LlamaCppBackend._request_reasoning_kwargs.__get__(shim)
    return build(enable_thinking, reasoning_effort, None) or {}


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
