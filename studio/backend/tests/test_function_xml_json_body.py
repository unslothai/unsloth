# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen3.5-style ``<function=NAME>`` with no ``<parameter>`` blocks: a JSON object body is the arguments
(models mix the Hermes and XML forms). Reading it as zero arguments ran the tool with ``{}`` (web_search:
"No query provided"). Every other body parses as before."""

from __future__ import annotations

import json

import pytest

from core.inference.tool_call_parser import parse_tool_calls_from_text, strip_tool_markup

ENABLED = {"web_search", "search_docs"}


def _calls(text):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    ]


@pytest.mark.parametrize("wrap", [True, False])
def test_json_body_is_the_arguments(wrap):
    call = '<function=search_docs>\n{"query": "rope scaling", "limit": 5}\n</function>'
    text = f"<tool_call>\n{call}\n</tool_call>" if wrap else call
    assert _calls(text) == [("search_docs", {"query": "rope scaling", "limit": 5})]


def test_non_json_body_is_unchanged():
    text = "<tool_call>\n<function=web_search>\nmars rover news\n</function>\n</tool_call>"
    assert _calls(text) == [("web_search", {})]
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == ""


def test_unreadable_json_body_is_unchanged():
    text = '<tool_call>\n<function=web_search>\n{"query": "mars\n</function>\n</tool_call>'
    assert _calls(text) == [("web_search", {})]


def test_zero_argument_and_parameter_calls_are_unchanged():
    assert _calls("<tool_call>\n<function=web_search>\n</function>\n</tool_call>") == [
        ("web_search", {})
    ]
    text = "<tool_call>\n<function=web_search>\n<parameter=query>\nmars\n</parameter>\n</function>\n</tool_call>"
    assert _calls(text) == [("web_search", {"query": "mars"})]


@pytest.mark.parametrize("wrap", [True, False])
def test_json_body_is_left_alone_with_auto_heal_off(wrap):
    # Auto-Heal off (allow_incomplete=False) accepts only well-formed calls, so the mixed form keeps
    # its strict reading in both copies of the reader.
    from core.tool_healing import parse_tool_calls_from_text as heal_parse

    call = '<function=python>\n{"code": "print(1)"}\n</function>'
    text = f"<tool_call>\n{call}\n</tool_call>" if wrap else call
    for parse in (parse_tool_calls_from_text, heal_parse):
        strict = parse(text, allow_incomplete = False, enabled_tool_names = ENABLED)
        assert [
            (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in strict
        ] == [("python", {})]
        healed = parse(text, allow_incomplete = True, enabled_tool_names = ENABLED)
        assert [
            (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in healed
        ] == [("python", {"code": "print(1)"})]
