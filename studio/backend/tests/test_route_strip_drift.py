# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins the known divergence between the route display strip and the canonical one.

``routes/inference.py::_strip_tool_xml_for_display`` predates
``core/inference/tool_call_parser.py`` (it goes back to the February route refactor,
before the parser existed) and never picked up the markdown-code gating that landed in
``core/tool_healing.py``. The GGUF, safetensors and ``strip_tool_markup`` paths all share
one scan order now; this one still has its own.

The difference is not cosmetic: the route strip deletes a rehearsal that sits inside a
fenced code block, so an assistant answer that *shows* what a tool call looks like has the
example removed. The canonical strip keeps it.

Changing the route path changes what users see on a shipped streaming route, so it is left
alone here and pinned instead: this test fails if either side moves, which makes the
follow-up a deliberate decision rather than an accident.
"""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.tool_call_parser import strip_tool_markup  # noqa: E402

ENABLED = {"get_weather", "search"}


@pytest.fixture(scope = "module")
def route_strip():
    routes_inference = pytest.importorskip("routes.inference")
    return routes_inference._strip_tool_xml_for_display


def test_route_strip_deletes_a_fenced_rehearsal_canonical_keeps_it(route_strip):
    """The concrete user-visible symptom of the drift."""
    text = '```\nget_weather[ARGS]{"city": "Paris"}\n```'

    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == "```\n\n```"
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text


def test_route_strip_deletes_an_inline_code_rehearsal(route_strip):
    text = "`get_weather[ARGS]{}`"

    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == "``"
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text


def test_route_strip_still_removes_a_real_call(route_strip):
    """Whatever else differs, the route strip must keep doing its actual job."""
    text = 'Here you go. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    stripped = route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)

    assert "<tool_call>" not in stripped
    assert "Here you go." in stripped


def test_route_strip_is_disabled_when_healing_is_off(route_strip):
    text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    assert route_strip(text, auto_heal_tool_calls = False, enabled_tool_names = ENABLED) == text
