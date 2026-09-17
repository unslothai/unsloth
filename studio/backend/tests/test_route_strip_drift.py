# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins the known divergence between the route display strip and the canonical one.

``routes/inference.py::_strip_tool_xml_for_display`` predates the parser and never picked
up the markdown-code gating in ``core/tool_healing.py``, so it deletes a rehearsal inside
a fenced code block that the canonical strip keeps.

Changing a shipped streaming route changes what users see, so it is pinned rather than
fixed here: this test fails if either side moves, making the follow-up deliberate.
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
