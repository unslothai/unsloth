# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``<function=NAME><parameter=KEY>`` parsing asked ``_inside_open_parameter`` once per candidate tag and
each call could rescan the text, so a long value full of tag literals took about a minute per megabyte.
The parser, the strip and the healer now ask one ``_OpenParameterIndex`` per text."""

from __future__ import annotations

import hashlib
import json
import random
import time

import pytest

from core import tool_healing
from core.inference import tool_call_parser
from core.inference.tool_call_parser import (
    StreamingMarkupStripper,
    parse_tool_calls_from_text,
    strip_tool_markup,
)

ENABLED = {"web_search", "write_file", "a", "f"}
PIECES = (
    "<function=a>",
    '<function name="f">',
    "</function>",
    "<tool_call>",
    "</tool_call>",
    "<parameter=p>",
    '<parameter name="q">',
    "<param=r>",
    '<param name="s">',
    "</parameter>",
    "</param>",
    "<param",
    "<parameter=",
    "<function=",
    "{",
    "}",
    '"x"',
    " ",
    "\n",
    "a",
    "<",
    ">",
)


def _random_texts(
    count,
    seed,
    max_pieces = 24,
):
    rng = random.Random(seed)
    return [
        "".join(rng.choice(PIECES) for _ in range(rng.randint(0, max_pieces))) for _ in range(count)
    ]


class _HelperIndex:
    """The pre-index behaviour: every query is a fresh ``_inside_open_parameter`` scan of the text."""

    def __init__(self, content, **kwargs):
        self._content = content
        self._kwargs = kwargs

    def inside(self, pos):
        return tool_healing._inside_open_parameter(self._content, pos, **self._kwargs)


def test_the_index_answers_like_the_shared_helper_at_every_offset(monkeypatch):
    # No warm-up scans, so every query below is answered by the index.
    monkeypatch.setattr(tool_healing, "_OPEN_PARAMETER_SCAN_LIMIT", 0)
    for text in _random_texts(400, seed = 0) + _random_texts(100, seed = 1, max_pieces = 80):
        parser_inside = tool_healing._OpenParameterIndex(
            text, **tool_call_parser._OPEN_PARAMETER_TAGS
        ).inside
        healer_inside = tool_healing._OpenParameterIndex(text).inside
        for pos in range(len(text) + 1):
            assert parser_inside(pos) == tool_call_parser._inside_open_parameter(text, pos), (
                text,
                pos,
            )
            assert healer_inside(pos) == tool_healing._inside_open_parameter(text, pos), (text, pos)


PROSE = (
    "Each call is `<function=NAME>` with `<parameter=KEY>` blocks closed by `</parameter>`, then `</function>`.\n"
    "```python\nout = [f'<function={name}>', '<parameter=path>a.py</parameter>', '</function>']\n```\n"
)
CALL = "<function=web_search>\n<parameter=query>\nweather in Paris\n</parameter>\n</function>"
WRITE = (
    "<function=write_file>\n<parameter=path>\nparser.py\n</parameter>\n<parameter=content>\n"
    + "t = '<parameter=path>' + p + '</parameter>'  # </function> <function=a>\n" * 20
)
CORPUS = _random_texts(200, seed = 2, max_pieces = 40) + [
    PROSE * 20 + CALL,
    WRITE + "\n</parameter>\n</function>",
    WRITE,
    "<tool_call>\n" + CALL + "\n</tool_call>\nDone.",
    "<function=f><parameter=p>" + "<function=a>" * 50 + "</parameter></function>" + CALL,
    ("<tool_call>\n" + CALL + "\n</tool_call>\n") * 5,
    ('<tool_call>{"name": "a", "arguments": {}}</tool_call>\n<function=a></function>\n') * 5,
    '<function name="f"><param name="p">' + '<function name="a">x' * 20 + "</param></function>",
    '<function name="f"><param name="p">1</param></function>\n<function name="a">',
]
# sha256 of the outputs below for CORPUS, recorded on the per-tag scan implementation before the index.
RECORDED_DIGEST = "b0b70a5748fe61c622cf53d1247a53bf78f3150c017940bee3f83b0298ecec2c"


def _outputs(text):
    return (
        parse_tool_calls_from_text(text, enabled_tool_names = ENABLED),
        parse_tool_calls_from_text(text, enabled_tool_names = ENABLED, allow_incomplete = False),
        strip_tool_markup(text, final = True, enabled_tool_names = ENABLED),
        strip_tool_markup(text, final = False, enabled_tool_names = ENABLED),
        tool_healing.parse_tool_calls_from_text(text, enabled_tool_names = ENABLED),
        tool_healing.parse_tool_calls_from_text(
            text, enabled_tool_names = ENABLED, allow_incomplete = False
        ),
        tool_healing.parse_tool_calls_from_text(text, with_spans = True),
    )


def _streamed(text, chunk = 5):
    stripper = StreamingMarkupStripper(ENABLED)
    return [stripper.strip(text[:end]) for end in range(chunk, len(text) + chunk, chunk)]


def _digest(results):
    return hashlib.sha256(json.dumps(results, sort_keys = True).encode()).hexdigest()


def test_outputs_match_the_per_tag_scan(monkeypatch):
    """Calls, strips (final and mid-stream), heals with Auto-Heal on and off: identical to the helper-backed run."""
    default = [(_outputs(text), _streamed(text)) for text in CORPUS]
    monkeypatch.setattr(tool_healing, "_OPEN_PARAMETER_SCAN_LIMIT", 0)
    indexed = [(_outputs(text), _streamed(text)) for text in CORPUS]
    monkeypatch.setattr(tool_healing, "_OpenParameterIndex", _HelperIndex)
    scanned = [(_outputs(text), _streamed(text)) for text in CORPUS]
    assert indexed == scanned
    assert default == scanned
    # Also covers the bounded wrapper and end-tag searches, which the helper swap above does not reach.
    assert _digest(default) == RECORDED_DIGEST
    # The corpus must reach the function parser, not only fall through it.
    assert sum(len(out[0]) for out, _stream in indexed) >= 50


@pytest.mark.parametrize(
    "text",
    [
        "<function=f><parameter=p>" + "<function=a>" * 80000,
        "<function=a><parameter=p>x</parameter></function>" * 20000,
        "<function=f><parameter=p>" + '<tool_call>{"a":1}' * 55000,
        "<parameter=p><function=a>" * 40000,
        '<function name="f"><param name="p">' + '<function name="a">' * 50000,
        ('<tool_call>{"name":"a","arguments":{}}</tool_call>\n<function=a></function>\n') * 6500,
    ],
    ids = [
        "openers_in_a_value",
        "many_calls",
        "json_markers_in_a_value",
        "unclosed_params_and_openers",
        "attribute_openers_in_a_value",
        "json_and_function_calls",
    ],
)
def test_a_megabyte_of_adversarial_markup_parses_in_linear_time(text):
    start = time.perf_counter()
    parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    strip_tool_markup(text, final = True, enabled_tool_names = ENABLED)
    tool_healing.parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    # Each took 20 s or more with a per-tag scan of the text and runs in 0.2 to 2 s here; 15 s leaves room for slow CI runners.
    assert time.perf_counter() - start < 15.0
