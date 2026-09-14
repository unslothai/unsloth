# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A ``reasoning_effort`` template's own ladder reaches the request kwargs.

Detection publishes every effort literal a ``reasoning_effort``-style template
branches on, and the Think menu offers exactly those. The request builder,
though, forwarded only gpt-oss's ``none`` / ``low`` / ``medium`` / ``high``:
anything wider fell through to the ``enable_thinking`` fallback, so picking
'xhigh' or 'max' sent 'high' (or, with no ``enable_thinking`` alongside, no
``chat_template_kwargs`` at all), and 'minimal' was rewritten to 'low'. On a
model whose named levels are coerced to a numeric dial that put 'max' and
'minimal' on their neighbours' values.

The advertised levels now widen the allowlist rather than replace it, so a
template that exposes no list keeps the four-level behaviour exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Slice of a wide-ladder template: reasoning_effort only (no enable_thinking
# gate), branching on the whole 'none'..'max' scale.
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


# gpt-oss-style: reasoning_effort only, low/medium/high named in prose rather
# than compared as literals, so detection publishes no level list.
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
    # The Think menu sends the level alone, with no enable_thinking alongside.
    assert _shim(WIDE_LEVELS)._request_reasoning_kwargs(None, effort, None) == {
        "reasoning_effort": expected
    }
    # An API caller may pair it with enable_thinking; the level still wins.
    assert _shim(WIDE_LEVELS)._request_reasoning_kwargs(True, effort, None) == {
        "reasoning_effort": expected
    }


def test_wide_ladder_still_disables_on_the_off_path():
    # The retry/off call sites pass (False, None): no level, so the fallback
    # runs and thinking goes off at the ladder's low end.
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
    # Not on the ladder: downgraded, exactly as before.
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
    # Duck-typed engine stand-ins bind this method without the attribute.
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
