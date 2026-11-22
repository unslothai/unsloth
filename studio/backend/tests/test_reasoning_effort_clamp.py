# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsupported effort levels clamp to the template's offered levels, as in the UI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


QWEN38_EARLY_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}
    {%- endif %}
    {{- 'effort=' ~ resolved_reasoning_effort }}
{%- else %}
    {{- 'effort=off' }}
{%- endif %}
"""


def _backend(template):
    from core.inference.llama_cpp import LlamaCppBackend, detect_reasoning_flags

    flags = detect_reasoning_flags(template, "vendor/Qwen3.8-27B-GGUF")
    backend = object.__new__(LlamaCppBackend)
    backend._supports_reasoning = flags["supports_reasoning"]
    backend._reasoning_always_on = flags["reasoning_always_on"]
    backend._reasoning_style = flags["reasoning_style"]
    backend._reasoning_effort_levels = flags["reasoning_effort_levels"]
    backend._supports_preserve_thinking = False
    backend._architecture = None
    return backend


def _render(template, kwargs):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(**(kwargs or {})).strip()


def test_early_qwen38_ladder_is_detected_without_high():
    backend = _backend(QWEN38_EARLY_TEMPLATE)
    assert backend._reasoning_style == "enable_thinking_effort"
    assert backend._reasoning_effort_levels == ["low", "medium", "xhigh"]


@pytest.mark.parametrize(
    ("requested", "rendered"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "medium"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
    ],
)
def test_every_request_renders_the_nearest_offered_level(requested, rendered):
    backend = _backend(QWEN38_EARLY_TEMPLATE)
    kwargs = backend._request_reasoning_kwargs(True, requested)
    assert kwargs == {"enable_thinking": True, "reasoning_effort": rendered}
    assert _render(QWEN38_EARLY_TEMPLATE, kwargs) == f"effort={rendered}"


def test_an_effort_alone_still_turns_thinking_on():
    backend = _backend(QWEN38_EARLY_TEMPLATE)
    assert backend._request_reasoning_kwargs(None, "high") == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


@pytest.mark.parametrize(("enable_thinking", "effort"), [(False, "high"), (None, "none")])
def test_off_stays_off_without_an_effort(enable_thinking, effort):
    backend = _backend(QWEN38_EARLY_TEMPLATE)
    kwargs = backend._request_reasoning_kwargs(enable_thinking, effort)
    assert kwargs == {"enable_thinking": False}
    assert _render(QWEN38_EARLY_TEMPLATE, kwargs) == "effort=off"


def test_no_effort_leaves_the_template_default():
    backend = _backend(QWEN38_EARLY_TEMPLATE)
    assert backend._request_reasoning_kwargs(True, None) == {"enable_thinking": True}
    assert backend._request_reasoning_kwargs(None, None) is None


@pytest.mark.parametrize(
    ("requested", "levels", "expected"),
    [
        # GLM-5.2: high is the weakest offered level.
        ("low", ["high", "max"], "high"),
        ("medium", ["high", "max"], "high"),
        # Claude 4.6: xhigh aliases to max.
        ("xhigh", ["low", "medium", "high", "max"], "max"),
        # Kimi-K3.
        ("medium", ["low", "high", "max"], "low"),
        ("xhigh", ["low", "high", "max"], "max"),
        ("high", [], None),
        (None, ["low", "high"], None),
        ("turbo", ["low", "high"], None),
    ],
)
def test_clamp_matches_the_think_menu(requested, levels, expected):
    from core.inference.llama_cpp import _clamp_reasoning_effort_to_levels
    assert _clamp_reasoning_effort_to_levels(requested, levels) == expected


def test_substitution_is_logged_once_per_ladder(monkeypatch):
    from core.inference import llama_cpp

    backend = _backend(QWEN38_EARLY_TEMPLATE)
    messages = []
    monkeypatch.setattr(llama_cpp, "_REASONING_EFFORT_CLAMPS_LOGGED", set())
    monkeypatch.setattr(llama_cpp.logger, "info", lambda message, *a, **k: messages.append(message))
    for _ in range(3):
        backend._request_reasoning_kwargs(True, "high")
    backend._request_reasoning_kwargs(True, "medium")

    assert messages == [
        "reasoning_effort 'high' is not offered by this template "
        "['low', 'medium', 'xhigh']; using 'medium'"
    ]
