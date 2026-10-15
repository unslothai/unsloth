# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advertised reasoning_effort levels widen the allowlist; without a list the four levels stay."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


WIDE_LADDER_TEMPLATE = """
{%- if reasoning_effort == 'none' -%}{{- 'N' -}}
{%- elif reasoning_effort == 'minimal' -%}{{- 'm' -}}
{%- elif reasoning_effort == 'low' -%}{{- 'l' -}}
{%- elif reasoning_effort == 'medium' -%}{{- 'M' -}}
{%- elif reasoning_effort == 'high' -%}{{- 'H' -}}
{%- elif reasoning_effort == 'xhigh' -%}{{- 'X' -}}
{%- elif reasoning_effort == 'max' -%}{{- 'Z' -}}
{%- endif -%}
"""


# gpt-oss-style: levels named in prose, not literals, so detection publishes none.
GPT_OSS_TEMPLATE = """
{%- set effort = reasoning_effort or 'medium' -%}
{{- 'Reasoning: ' + effort -}}
"""


WIDE_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _shim(levels, architecture = "inkling"):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "reasoning_effort"
    backend._reasoning_effort_levels = levels
    backend._supports_preserve_thinking = False
    backend._architecture = architecture
    return backend


def test_wide_ladder_is_published_by_detection():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(WIDE_LADDER_TEMPLATE, "vendor/Wide-GGUF")
    assert flags["reasoning_style"] == "reasoning_effort"
    assert flags["reasoning_effort_levels"] == WIDE_LEVELS


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", 0.0),
        ("minimal", 0.1),
        ("low", 0.2),
        ("medium", 0.7),
        ("high", 0.9),
        ("xhigh", 0.99),
        ("max", 0.99),
    ],
)
def test_every_advertised_level_reaches_the_template(effort, expected):
    assert _shim(WIDE_LEVELS)._request_reasoning_kwargs(None, effort, None) == {
        "reasoning_effort": expected
    }
    assert _shim(WIDE_LEVELS)._request_reasoning_kwargs(True, effort, None) == {
        "reasoning_effort": expected
    }


def test_wide_ladder_still_disables_on_the_off_path():
    assert _shim(WIDE_LEVELS)._request_reasoning_kwargs(False, None, None) == {
        "reasoning_effort": 0.2
    }


def test_a_model_advertising_no_levels_keeps_the_four_level_behaviour():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(GPT_OSS_TEMPLATE, "unsloth/gpt-oss-20b-GGUF")
    assert flags["reasoning_style"] == "reasoning_effort"
    assert flags["reasoning_effort_levels"] == []

    backend = _shim([], architecture = None)
    assert backend._request_reasoning_kwargs(None, "low", None) == {"reasoning_effort": "low"}
    assert backend._request_reasoning_kwargs(None, "medium", None) == {"reasoning_effort": "medium"}
    assert backend._request_reasoning_kwargs(None, "high", None) == {"reasoning_effort": "high"}
    assert backend._request_reasoning_kwargs(None, "none", None) == {"reasoning_effort": "none"}
    assert backend._request_reasoning_kwargs(None, "minimal", None) == {"reasoning_effort": "low"}
    assert backend._request_reasoning_kwargs(True, "max", None) == {"reasoning_effort": "high"}
    assert backend._request_reasoning_kwargs(None, "max", None) is None
    assert backend._request_reasoning_kwargs(True, None, None) == {"reasoning_effort": "high"}
    assert backend._request_reasoning_kwargs(False, None, None) == {"reasoning_effort": "low"}
    assert backend._request_reasoning_kwargs(None, None, None) is None


def test_a_narrow_ladder_does_not_lose_the_gpt_oss_levels():
    # Advertised levels widen the allowlist rather than replace it, so a level
    # the scan missed is still forwarded.
    backend = _shim(["high", "max"], architecture = None)
    assert backend._request_reasoning_kwargs(None, "low", None) == {"reasoning_effort": "low"}
    assert backend._request_reasoning_kwargs(None, "max", None) == {"reasoning_effort": "max"}


def test_missing_levels_attribute_is_tolerated():
    backend = _shim(WIDE_LEVELS)
    del backend._reasoning_effort_levels
    assert backend._request_reasoning_kwargs(None, "high", None) == {"reasoning_effort": 0.9}


def test_the_level_renders_in_the_template():
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment().from_string(WIDE_LADDER_TEMPLATE)
    assert template.render(reasoning_effort = "max") == "Z"
    assert template.render(reasoning_effort = "xhigh") == "X"
    assert template.render(reasoning_effort = "minimal") == "m"


def test_other_reasoning_styles_are_untouched():
    backend = _shim(["low", "high", "max"], architecture = None)
    backend._reasoning_style = "enable_thinking"
    assert backend._request_reasoning_kwargs(True, "max", None) == {"enable_thinking": True}
    assert backend._request_reasoning_kwargs(False, None, None) == {"enable_thinking": False}

    backend._reasoning_style = "enable_thinking_effort"
    assert backend._request_reasoning_kwargs(None, "max", None) == {
        "enable_thinking": True,
        "reasoning_effort": "max",
    }

    backend._reasoning_style = "reasoning_effort"
    backend._reasoning_always_on = True
    assert backend._request_reasoning_kwargs(True, "max", None) is None
