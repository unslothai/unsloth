# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streaming healer's reparse gate is invisible: every chunking emits what an always-reparsing healer emits."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.passthrough_healing as ph  # noqa: E402
from core.tool_healing import _balanced_bracket_end, _balanced_json_span  # noqa: E402


class _AlwaysParse:
    def release(self):
        pass

    def may_parse(self, buffer):
        return True


class _ReparsingHealer(ph.StreamToolCallHealer):
    """The pre-gate behaviour: a full strict reparse on every held chunk."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._strict_gate = _AlwaysParse()


NAMES = ("web_search", "terminal", "python")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    for name in NAMES
]
GQ = '<|"|>'
DOCS = [
    'Hi <tool_call>{"name": "web_search", "arguments": {"query": "a \\"}\\" {b}"}}</tool_call> after',
    '<tool_call>\n{"name": "terminal", "parameters": {"query": "ls", "n": [1, {"a": "}"}]}}\n</tool_call>',
    '<tool_call>{"name": "undeclared", "arguments": {}}</tool_call><tool_call>{"name": "python", "arguments": {"query": "x"',
    "<|tool_call>call:web_search{query:" + GQ + "x {y} 中文" + GQ + ",n:3}<tool_call|> tail",
    "<|tool_call>call:web_search{query:" + GQ + "never closed {",
    "<function=terminal>\n<parameter=query>\nls {a} </function> x\n</parameter>\n</function> ok",
    "<tool_call>\n<function=python>\n<parameter=query>print({})</parameter>\n</function>\n</tool_call>",
    'Sure.[TOOL_CALLS]web_search[ARGS]{"query": "x = {1}\\\\\\" ]"} more',
    '[TOOL_CALLS] terminal {"query": "a", "o": {"k": [1, "]"]}}',
    '[TOOL_CALLS]python[CALL_ID]abc123[ARGS]{"query": "q"}[/TOOL_CALLS]',
    '[TOOL_CALLS][{"name": "web_search", "arguments": {"query": "]}"}}{"name": "python", "arguments": {}}] x',
    '[TOOL_CALLS]web_search[ARGS]{"query": "truncated \\',
    'Prose python[ARGS]  \n{"query": "1"} and ```\npython[ARGS]{"query": "2"}\n```',
    'web_search<|content_invoke_tool_json|>{"name": "web_search", "args": {"query": "z"}}<|end_message|>',
    '<think>plan <tool_call>{"name": "python", "arguments": {}}</tool_call></think><tool_call>{"name": "python", "arguments": {"query": "y"}}</tool_call>',
    "Prose with {braces} [and] brackets, a < b, 🙂 and no call at all.",
]


def _events(healer_cls, chunks):
    healer = healer_cls(set(NAMES), TOOLS)
    events = [healer.feed(chunk) for chunk in chunks]
    events.append(healer.finalize())
    return json.dumps(events, ensure_ascii = False)


def _chunkings(doc, rng):
    yield [doc[i : i + 1] for i in range(len(doc))]
    for i in range(1, len(doc)):
        yield [doc[:i], doc[i:]]
    for high in (3, 9, 40):
        chunks, i = [], 0
        while i < len(doc):
            step = rng.randint(1, high)
            chunks.append(doc[i : i + step])
            i += step
        yield chunks


@pytest.mark.parametrize("doc", DOCS)
def test_every_chunking_matches_always_reparsing_healer(doc):
    rng = random.Random(len(doc))
    for chunks in _chunkings(doc, rng):
        assert _events(ph.StreamToolCallHealer, chunks) == _events(_ReparsingHealer, chunks), chunks


def test_resumed_scans_match_balanced_scans_at_every_prefix():
    rng = random.Random(0)
    alphabet = '{}[]"\\a 中'
    for _ in range(400):
        body = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        for opener, scan, resume in (
            ("{", _balanced_json_span, ph._resume_json_span),
            ("[", _balanced_bracket_end, ph._resume_bracket_end),
        ):
            text = opener + body
            state, end = [0, 0, False, False], None
            for length in range(1, len(text) + 1):
                prefix = text[:length]
                if end is None:
                    end = resume(prefix, state)
                assert end == scan(prefix, 0), (text, length)


@pytest.mark.parametrize(
    "opening, closing",
    [
        ('<tool_call>{"name": "python", "arguments": {"query": "', '"}}</tool_call>'),
        ("<|tool_call>call:python{query:" + GQ, GQ + "}<tool_call|>"),
        ("<function=python>\n<parameter=query>\n", "\n</parameter>\n</function>"),
        ('[TOOL_CALLS]python[ARGS]{"query": "', '"}'),
        ('[TOOL_CALLS][{"name": "python", "arguments": {"query": "', '"}}]'),
    ],
)
def test_long_held_call_is_not_reparsed_per_chunk(monkeypatch, opening, closing):
    calls = []
    real_parse = ph.parse_tool_calls_from_text
    monkeypatch.setattr(
        ph,
        "parse_tool_calls_from_text",
        lambda *a, **k: calls.append(1) or real_parse(*a, **k),
    )
    text = opening + "x = {1}; y = [2]\\n" * 200 + closing
    healer = ph.StreamToolCallHealer({"python"}, TOOLS)
    events = []
    for i in range(0, len(text), 4):
        events += healer.feed(text[i : i + 4])
    events += healer.finalize()
    assert [kind for kind, _ in events] == ["tool_call"]
    assert len(calls) <= 2


_FUZZ_PIECES = (
    "[TOOL_CALLS]",
    "[ARGS]",
    "[CALL_ID]",
    "python",
    "web_search",
    "-",
    " ",
    "{",
    "}",
    "[",
    "]",
    '"',
    "\\",
    ":",
    "x",
)


def test_random_marker_soup_matches_always_reparsing_healer():
    rng = random.Random(15618)
    for _ in range(300):
        doc = "".join(rng.choice(_FUZZ_PIECES) for _ in range(rng.randint(1, 24)))
        for chunks in (
            [doc],
            [doc[i : i + 1] for i in range(len(doc))],
            [doc[i : i + 5] for i in range(0, len(doc), 5)],
        ):
            assert _events(ph.StreamToolCallHealer, chunks) == _events(
                _ReparsingHealer, chunks
            ), doc


def _count_scans(monkeypatch):
    scans = []
    for name in ("_resume_json_span", "_resume_bracket_end"):
        real = getattr(ph, name)
        monkeypatch.setattr(
            ph, name, lambda text, state, _real = real: scans.append(1) or _real(text, state)
        )
    return scans


@pytest.mark.parametrize(
    "text",
    [
        "[TOOL_CALLS]f" + "python[ARGS]{" * 2000,
        "[TOOL_CALLS]python{" * 2000,
        '[TOOL_CALLS]python{"query": "' + "[ARGS]{" * 2000 + "}",
    ],
    ids = ["rehearsal_no_closer", "bracket_no_closer", "nameless_quoted"],
)
def test_openers_that_cannot_parse_are_not_scanned(monkeypatch, text):
    scans = _count_scans(monkeypatch)
    for chunks in ([text], [text[i : i + 64] for i in range(0, len(text), 64)]):
        scans.clear()
        assert _events(ph.StreamToolCallHealer, chunks) == _events(_ReparsingHealer, chunks)
        assert len(scans) <= len(chunks), len(scans)


def test_gate_lets_go_of_released_text():
    healer = ph.StreamToolCallHealer({"python"}, TOOLS)
    healer.feed("[TOOL_CALLS]python[ARGS]{" + "x" * (ph._MAX_HOLD_CHARS + 10))
    assert healer._buffer == "" and healer._strict_gate._buf == ""
    healer.feed('<tool_call>{"name": "python"')
    assert healer._strict_gate._buf
    healer.finalize()
    assert healer._strict_gate._buf == ""
