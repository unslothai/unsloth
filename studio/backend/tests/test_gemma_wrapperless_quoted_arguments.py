# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gemma 4 sometimes drops its ``<|tool_call>`` wrapper. The healer promotes that bare ``call:NAME{...}``,
and its ``<|"|>``-quoted values must decode as they do inside the wrapper, not keep the delimiters."""

from __future__ import annotations

import json

import pytest

from core.inference.tool_call_parser import parse_tool_calls_from_text

ENABLED = {"web_search", "search_docs"}


def _calls(text):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    ]


@pytest.mark.parametrize("lead", ["", "Let me look that up.\n"])
def test_bare_call_decodes_gemma_quotes_like_the_wrapped_one(lead):
    body = 'call:search_docs{query:<|"|>a, b {c}<|"|>,limit:5,exact:true}'
    want = [("search_docs", {"query": "a, b {c}", "limit": 5, "exact": True})]
    assert _calls(lead + body) == want
    assert _calls("<|tool_call>" + body + "<tool_call|>") == want


@pytest.mark.parametrize("body", ["call:web_search{query:x}", 'call:web_search{query:"x"}'])
def test_unquoted_bare_bodies_are_unchanged(body):
    assert _calls(body) == [("web_search", {"query": "x"})]


def test_literal_marker_inside_a_json_quoted_value_keeps_its_arguments():
    # ``, b:<|"|>`` sits inside a JSON string, so the Gemma decoder rejects the body: keep the
    # quote-less reading rather than running the tool with no arguments.
    assert _calls('call:web_search{query:"a, b:<|"|>c"}') == [
        ("web_search", {"query": 'a, b:<|"|>c'})
    ]


@pytest.mark.parametrize(
    "body, want",
    [
        ("query:<|\"|>mars<|\"|>,unit:'celsius'", {"query": "mars", "unit": "celsius"}),
        (
            'query:<|"|>mars<|"|>,opts:{note:\'<|"|>kept<|"|>\'}',
            {"query": "mars", "opts": {"note": '<|"|>kept<|"|>'}},
        ),
        (
            "query:<|\"|>a, b:c<|\"|>,opts:{unit:'c'},tags:['x','y']",
            {"query": "a, b:c", "opts": {"unit": "c"}, "tags": ["x", "y"]},
        ),
        ('code:<|"|>print("hi")\nx = \'y\'<|"|>', {"code": "print(\"hi\")\nx = 'y'"}),
    ],
)
def test_other_values_keep_their_quote_less_reading(body, want):
    assert _calls("call:web_search{" + body + "}") == [("web_search", want)]
