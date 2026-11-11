# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference import tool_call_parser
from routes.inference import _strip_tool_xml_for_display

_NAMES = {"python", "get_weather"}
_PROSE = (
    "# Results\n\nThe mean of column `a` is 3.5 and x < y > z holds. A list [1, 2] and a dict {'k': 1}.\n\n"
    "```python\nprint('recall: done')\n```\n\n  trailing spaces and unicode: café 日本語 ｜ 🙂  \n\n\n"
)


def test_plain_prose_skips_the_strip_chain(monkeypatch):
    calls = []
    real = tool_call_parser._mask_blocked_bodies

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(tool_call_parser, "_mask_blocked_bodies", spy)
    for names in (None, set(), _NAMES):
        out = _strip_tool_xml_for_display(
            _PROSE, auto_heal_tool_calls = True, enabled_tool_names = names
        )
        assert out is _PROSE
    assert calls == []
    _strip_tool_xml_for_display(
        _PROSE + "call:python{}", auto_heal_tool_calls = True, enabled_tool_names = _NAMES
    )
    _strip_tool_xml_for_display(
        _PROSE + "<tool_call>", auto_heal_tool_calls = True, enabled_tool_names = _NAMES
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            'Hi <tool_call>{"name": "python", "arguments": {}}</tool_call> there',
            "Hi  there",
        ),
        ("Hi <function=python><parameter=code>1</parameter></function> there", "Hi  there"),
        ('Hi [TOOL_CALLS]python[ARGS]{"code": "1"}', "Hi "),
        ("Hi <|python_tag|>print(1)", "Hi "),
        ("<think>a <tool_call></think>b <tool_call>x", "<think>a <tool_call></think>b "),
        ("Mention `<tool_call>` in prose.", "Mention `<tool_call>` in prose."),
    ],
)
def test_markup_still_stripped(text, expected):
    assert (
        _strip_tool_xml_for_display(text, auto_heal_tool_calls = True, enabled_tool_names = _NAMES)
        == expected
    )


def test_gemma_call_re_matches_the_word_boundary_form():
    from itertools import product

    from routes.inference import _DISPLAY_GEMMA_CALL_RE

    reference = __import__("re").compile(r"(?<!\w)call\s*:")
    for parts in product(
        ("", "x", "_", "é", " ", "(", "re"), ("call", "Call"), ("", " ", "\t\n"), (":", "x")
    ):
        text = "".join(parts) + " call:"
        for pos in range(len(text) + 1):
            got, want = _DISPLAY_GEMMA_CALL_RE.search(text, pos), reference.search(text, pos)
            assert (got and got.span()) == (want and want.span()), (text, pos)
