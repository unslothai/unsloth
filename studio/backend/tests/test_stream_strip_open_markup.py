# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming display strips stay output-identical to the full strip on every prefix, without re-stripping
the whole turn per token once an open call or a markup mention is in it.

Three per-token strips are covered: the generator's ``StreamingMarkupStripper`` (its open-call gate now also
takes ``<function=NAME>`` bodies and non-JSON ``<tool_call>`` bodies), the route's ``_StreamingDisplayStrip``
(holds ``P + <tool_call>|<function=NAME> + B`` at P), and ``_ReasoningSpanGuard`` (the Messages API strip, which
used to run the full strip on every content event)."""

import random
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference import tool_call_parser as parser  # noqa: E402
from core.inference.tool_call_parser import StreamingMarkupStripper  # noqa: E402
from routes import inference  # noqa: E402
from routes.inference import (  # noqa: E402
    _ReasoningSpanGuard,
    _StreamingDisplayStrip,
    _strip_tool_xml_for_display,
)

ENABLED = {"web_search", "python", "terminal", "write_file"}
NAME_GATES = (ENABLED, None, set())


class _FullStripEveryToken(StreamingMarkupStripper):
    """The generator stripper with its open-call gate off: every token strips the open tail."""

    __slots__ = ()

    def _gate_try_arm(self, *args):
        return None


class _FullSpanGuard(_ReasoningSpanGuard):
    """The span guard as it was: the full display strip on every event."""

    def _display_strip(self, auto_heal_tool_calls, enabled_tool_names):
        class _Full:
            @staticmethod
            def strip(text):
                return _strip_tool_xml_for_display(
                    text,
                    auto_heal_tool_calls = auto_heal_tool_calls,
                    enabled_tool_names = enabled_tool_names,
                )

        return _Full


_CODE = (
    "def area(r):\n    return 3.14159 * r * r  # a < b and c > d\n\n"
    "for i in range(10):\n    print(i, area(i), {'k': [i]})\n"
)
_PREFIXES = (
    "",
    "I'll write the file now.\n\n",
    "Sure. ",
    "To call a tool the model wraps JSON in a `",
    "x `y` z\n",
    "call me maybe ",
    "  \n",
    "a [link](x) ",
    "brace {x} ",
    "<think>plan it</think>\n",
    "</think>stray ",
    "<parameter=x>v\n",
    "call :",
    "```xml\n",
    "中文 ",
)
_OPENERS = (
    "<tool_call>",
    "<tool_call>\n",
    "<tool_call>\n<function=write_file>\n",
    "<tool_call><function=python>",
    "<function=write_file>\n",
    '<function name="python">',
    "<function=nope>\n",
    "<tool_call>web_search\n",
    '<tool_call>\n{"name": "write_file", "arguments": {"content": "',
    "` tag and ",
)
_BODY_PIECES = (
    "<parameter=path>\nsrc/a.py\n</parameter>\n",
    "<parameter=content>\n",
    _CODE,
    "word ",
    "\n",
    "<",
    "</",
    "<div>",
    "a < b ",
    "{",
    "}",
    "[",
    "]",
    ":",
    "call",
    "call ",
    " :",
    "recall: ",
    "`",
    "```",
    "\\n",
    '\\"',
    "<arg_key>q</arg_key>\n<arg_value>x</arg_value>\n",
    "</parameter>",
    "</param>",
    "<param",
    "中文",
)
_TRIGGER_PIECES = (
    "</function>",
    "</tool_call>",
    "<tool_call>",
    "<function=python>",
    "<|tool_call>call:web_search{q:1}<tool_call|>",
    "[TOOL_CALLS]",
    '[TOOL_CALLS]web_search{"q": 1}',
    "web_search[ARGS]{}",
    "x[ARGS]{",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>python",
    "<｜tool▁calls▁end｜>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.python:0 {}<|tool_call_end|>",
    "<|python_tag|>print(1)",
    "<|content_invoke_tool_json|>",
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    "call:web_search{q:1}",
    "call \n\t :python{code:1}",
    "<|channel|>commentary to=functions.python <|message|>{}<|call|>",
    '<minimax:tool_call>\n<invoke name="python">\n<parameter name="code">1</parameter>\n</invoke>\n',
    "</minimax:tool_call>",
    "[/TOOL_CALLS]",
)
_ENDINGS = (
    "",
    "\n</function>\n</tool_call>",
    "\n</parameter>\n</function>\n</tool_call>\nDone.",
    '"}}\n</tool_call> then prose',
    "</tool_call> and <tool_call>\n<function=python>\n<parameter=code>\nx",
    " the end.",
)


def _corpus(rng: random.Random, n: int):
    for _ in range(n):
        text = rng.choice(_PREFIXES) + rng.choice(_OPENERS)
        trigger_rate = rng.choice((0.0, 0.0, 0.02, 0.1))
        for _ in range(rng.randint(2, 30)):
            pool = _TRIGGER_PIECES if rng.random() < trigger_rate else _BODY_PIECES
            text += rng.choice(pool)
        yield text + rng.choice(_ENDINGS)


def _prefixes(text: str, rng: random.Random):
    cut = 0
    while cut < len(text):
        cut = min(len(text), cut + rng.choice((1, 2, 3, 4, 5, 6, 8, 16)))
        yield text[:cut]


def test_generator_strip_matches_on_every_prefix():
    rng = random.Random(20261030)
    engaged = 0
    for text in _corpus(rng, 120):
        for names in NAME_GATES:
            for seg_final in (True, False):
                gated = StreamingMarkupStripper(names, seg_final = seg_final)
                full = _FullStripEveryToken(names, seg_final = seg_final)
                for prefix in _prefixes(text, rng):
                    was_on = gated._gate_on
                    assert gated.strip(prefix) == full.strip(prefix), (text, len(prefix))
                    engaged += was_on and gated._gate_on
    # The gate must answer many tokens here, or this proves nothing.
    assert engaged > 3000, engaged


def test_route_display_strip_matches_on_every_prefix():
    rng = random.Random(20261031)
    held = 0
    for text in _corpus(rng, 150):
        for names in NAME_GATES:
            strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = names)
            for prefix in _prefixes(text, rng):
                was_held = strip._hold_out is not None
                assert strip.strip(prefix) == _strip_tool_xml_for_display(
                    prefix, auto_heal_tool_calls = True, enabled_tool_names = names
                ), (text, len(prefix))
                held += was_held and strip._hold_out is not None
    assert held > 2000, held


def test_span_guard_matches_the_full_strip_across_turns():
    rng = random.Random(20261101)
    texts = list(_corpus(rng, 40))
    for names in NAME_GATES:
        for prov in (None, {"wrapped": 2, "wraps": [{"len": 9}, {"len": 4}]}):
            guard, ref = _ReasoningSpanGuard(prov), _FullSpanGuard(prov)
            for turn, text in enumerate(texts[:6]):
                if turn:
                    guard.tool_end()
                    ref.tool_end()
                if prov is not None and turn < 2:
                    text = "<think>" + "r" * (9 if turn == 0 else 4) + "</think>" + text
                for prefix in _prefixes(text, rng):
                    kwargs = {"auto_heal_tool_calls": True, "enabled_tool_names": names}
                    assert guard.strip(prefix, **kwargs) == ref.strip(prefix, **kwargs), (
                        text,
                        len(prefix),
                    )
    # Auto-Heal off returns the text untouched on both.
    assert _ReasoningSpanGuard().strip("a <tool_call>b", auto_heal_tool_calls = False) == (
        "a <tool_call>b"
    )


def _count_route_strips(monkeypatch):
    calls = [0]
    real = inference._strip_tool_xml_for_display

    def _counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(inference, "_strip_tool_xml_for_display", _counting)
    return calls


def _write_file_stream(tokens: int, prefix: str = "I'll write it.\n\n"):
    text = prefix + "<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n"
    text += "<parameter=content>\n" + _CODE * (tokens // 20 + 1)
    out, cut = [], len(prefix) + 1
    while len(out) < tokens:
        cut += 4
        out.append(text[:cut])
    return out


def test_function_xml_write_file_is_not_restripped_per_token():
    seen = {}
    for tokens in (300, 2000):
        calls = [0, 0]

        class _Counting(StreamingMarkupStripper):
            __slots__ = ()

            def _full_strip(self, text):
                calls[0] += 1
                calls[1] += len(text)
                return super()._full_strip(text)

        for seg_final in (True, False):
            stripper = _Counting(ENABLED, seg_final = seg_final)
            full = _FullStripEveryToken(ENABLED, seg_final = seg_final)
            for prefix in _write_file_stream(tokens):
                assert stripper.strip(prefix) == full.strip(prefix)
        seen[tokens] = tuple(calls)
    assert seen[300] == seen[2000], seen
    assert seen[2000][0] < 80, seen


@pytest.mark.parametrize(
    "prefix",
    ["To call a tool the model wraps JSON in a `", "I'll write it.\n\n", ""],
)
def test_route_holds_an_open_call_or_mention_to_the_end(monkeypatch, prefix):
    calls = _count_route_strips(monkeypatch)
    seen = {}
    for tokens in (300, 2000):
        calls[0] = 0
        strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
        for text in _write_file_stream(tokens, prefix):
            got = strip.strip(text)
        seen[tokens] = calls[0]
        assert got == prefix
    assert seen[300] == seen[2000] < 40, seen


def test_route_hold_releases_on_the_close_split_at_every_offset():
    body = "Writing.\n<tool_call>\n<function=write_file>\n<parameter=content>\n" + _CODE * 3
    close = "</parameter>\n</function>\n</tool_call>\nDone."
    for split in range(1, len(close)):
        for names in NAME_GATES:
            strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = names)
            text = body
            for piece in list("pad words ") + [close[:split], close[split:]]:
                text += piece
                assert strip.strip(text) == _strip_tool_xml_for_display(
                    text, auto_heal_tool_calls = True, enabled_tool_names = names
                ), split
    assert strip.strip(text) == "Writing.\n\nDone."


def test_span_guard_skips_the_full_strip_on_plain_prose(monkeypatch):
    calls = _count_route_strips(monkeypatch)
    guard = _ReasoningSpanGuard()
    text = ""
    for i in range(2000):
        text += f"word{i} "
        assert guard.strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == text
    assert calls[0] == 0


def test_gemma_scan_finds_what_finditer_finds():
    """Literal-anchored scan used by ``_needs_whole_buffer``; same first promotable match as the regex sweep."""

    def _reference(text, start, names):
        start = text.find(parser._GEMMA_BARE_SENTINEL, start)
        if start < 0:
            return -1
        for m in parser._GEMMA_BARE_TC_RE.finditer(text, start):
            if parser._markerless_promotable(m.group(1), names):
                return m.start()
        return -1

    rng = random.Random(7)
    alphabet = (
        "call",
        "call:",
        "call :",
        ":",
        "x",
        "web_search",
        "python",
        "{",
        " ",
        "re",
        ".",
        "-",
        "\n",
    )
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        start = rng.randint(0, len(text))
        for names in NAME_GATES:
            assert parser._promotable_gemma_call_pos(text, start, names) == _reference(
                text, start, names
            ), (text, start)
