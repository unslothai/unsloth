# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_StreamingDisplayStrip must return exactly what _strip_tool_xml_for_display returns for every prefix of a
streamed turn, while skipping the full rescan on turns with no tool markup."""

import random

import pytest

from routes import inference
from routes.inference import _StreamingDisplayStrip, _strip_tool_xml_for_display

_NAMES = {"web_search", "python", "terminal"}
_PROSE = (
    "A hash map stores keys in buckets chosen by a hash function. Collisions chain or probe; when the load factor "
    "grows past a threshold the table resizes and rehashes. `dict[str, int]` lookups stay O(1) on average. "
)
_TRANSCRIPTS = [
    "<think>" + _PROSE * 3 + "</think>" + _PROSE * 4,
    _PROSE
    + '<tool_call>{"name": "python", "arguments": {"code": "print(1)"}}</tool_call>'
    + _PROSE,
    _PROSE + "<function=python><parameter=code>print(2)</parameter></function>" + _PROSE,
    _PROSE + '[TOOL_CALLS]python[ARGS]{"code": "print(3)"}' + _PROSE,
    _PROSE + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>python" + _PROSE,
    _PROSE
    + '<|tool_calls_section_begin|><|tool_call_begin|>functions.python:0 {"code": "x"}'
    + _PROSE,
    _PROSE + "<|python_tag|>print(4)" + _PROSE,
    _PROSE + 'call:python{code: "print(5)"}' + _PROSE,
    _PROSE + "so we call : python with args" + _PROSE,
    _PROSE + "recall: this is prose, not a call" + _PROSE,
    "<think>" + _PROSE + '<tool_call>{"name": "python"}</tool_call></think>' + _PROSE,
]


def _replay(strip, text, rng):
    cut = 0
    while cut < len(text):
        cut = min(len(text), cut + rng.randint(1, 16))
        yield text[:cut], strip.strip(text[:cut])


@pytest.mark.parametrize("names", [None, set(), _NAMES])
@pytest.mark.parametrize("auto_heal", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_every_prefix_matches_the_full_strip(names, auto_heal, seed):
    rng = random.Random(seed)
    strip = _StreamingDisplayStrip(auto_heal_tool_calls = auto_heal, enabled_tool_names = names)
    # One instance across turns, as a tool loop restarts its cumulative text every iteration.
    for text in rng.sample(_TRANSCRIPTS, len(_TRANSCRIPTS)):
        for prefix, got in _replay(strip, text, rng):
            assert got == _strip_tool_xml_for_display(
                prefix, auto_heal_tool_calls = auto_heal, enabled_tool_names = names
            ), prefix


def test_a_call_split_across_tokens_is_still_stripped():
    strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = _NAMES)
    text = _PROSE + 'call:python{code: "print(5)"}'
    for prefix in (_PROSE + "ca", _PROSE + "call", _PROSE + "call :", text):
        assert strip.strip(prefix) == _strip_tool_xml_for_display(
            prefix, auto_heal_tool_calls = True, enabled_tool_names = _NAMES
        )


def test_prose_turns_skip_the_full_rescan(monkeypatch):
    calls = []
    original = inference._strip_tool_xml_for_display

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "_strip_tool_xml_for_display", counting)
    strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = _NAMES)
    text = _TRANSCRIPTS[0]
    for cut in range(1, len(text) + 1, 4):
        assert strip.strip(text[:cut]) == text[:cut]
    assert calls == []
    marked = _TRANSCRIPTS[1]
    strip.strip(marked)
    assert len(calls) == 1
