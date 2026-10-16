# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A reply made of many JSON lines that quote tool markup (a Hermes-format dataset, say) is data:
the parser steps over each line instead of recursing once per line."""

import sys
import time

from core.inference.tool_call_parser import parse_tool_calls_from_text

LINE = '{"text": "use <tool_call> tags"}\n'


def test_many_json_lines_quoting_tool_markup_parse_to_no_calls():
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        start = time.monotonic()
        calls = parse_tool_calls_from_text(LINE * 1600, enabled_tool_names = {"web_search"})
        elapsed = time.monotonic() - start
    finally:
        sys.setrecursionlimit(limit)
    assert calls == []
    # Base took about a minute before the RecursionError on this input.
    assert elapsed < 10, elapsed


def test_a_real_call_after_quoting_lines_is_still_found():
    text = (
        LINE * 3
        + '<tool_call>{"name": "web_search", "arguments": {"query": "unsloth"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls] == ["web_search"]
