# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An open ``<tool_call>`` body is not re-stripped per token, and the strip results do not change."""

import json
import random
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.tool_call_parser import StreamingMarkupStripper  # noqa: E402

ENABLED = {"web_search", "python", "terminal"}


class _FullStripEveryToken(StreamingMarkupStripper):
    """The stripper before the open-call gate: the gate never arms, so every token strips the open tail."""

    __slots__ = ()

    def _gate_try_arm(self, *args):
        return None


def _code_tokens(count):
    line = 'def f(x):\n    return {"a": [x, "b\\\\c"], "d": f"{x}"}\n'
    escaped = json.dumps(line)[1:-1]
    return [escaped[i : i + 4] for i in range(0, len(escaped), 4)] * (count // 16 + 1)


def _replay(stripper, chunks):
    text, outs = "", []
    for chunk in chunks:
        text += chunk
        outs.append(stripper.strip(text))
    return outs


class _CountingStripper(StreamingMarkupStripper):
    __slots__ = ()
    seen = [0, 0]

    def _full_strip(self, text):
        _CountingStripper.seen[0] += 1
        _CountingStripper.seen[1] += len(text)
        return super()._full_strip(text)


@pytest.mark.parametrize("seg_final", [True, False])
@pytest.mark.parametrize(
    "preamble", ["I'll search. ", "", "Running it:\n", "<think>plan</think>\n\n"]
)
def test_open_call_work_does_not_grow_with_the_body(seg_final, preamble):
    """Counts characters handed to the full strip, not time: re-stripping the open tail per token is quadratic."""
    opener = preamble + '<tool_call>{"name": "python", "arguments": {"code": "'
    seen = {}
    for count in (250, 2000):
        chunks = [opener] + _code_tokens(count)[:count]
        _CountingStripper.seen[:] = [0, 0]
        outs = _replay(_CountingStripper(ENABLED, seg_final = seg_final), chunks)
        seen[count] = tuple(_CountingStripper.seen)
        assert outs == _replay(_FullStripEveryToken(ENABLED, seg_final = seg_final), chunks)
    # The few calls left fall in the first ~128 characters, before ``_is_extension`` samples a middle.
    assert seen[250] == seen[2000], seen
    assert seen[2000][0] < 40, seen


def test_open_call_engages_and_matches_on_close(monkeypatch):
    """The close tag, split at every offset, still ends the gate on the token that completes it."""
    body = (
        'Searching.\n<tool_call>{"name": "web_search", "arguments": {"query": "a \\"quoted\\" {x}'
    )
    close = '"}}</tool_call>\nDone.'
    for split in range(1, len(close)):
        chunks = [body] + list("pad words ") + [close[:split], close[split:]]
        for names in (ENABLED, None):
            got = _replay(StreamingMarkupStripper(names), chunks)
            want = _replay(_FullStripEveryToken(names), chunks)
            assert got == want, split
    assert got[-1] == want[-1] == "Searching.\n\nDone."


_TRIGGERS = (
    "</tool_call>",
    "<tool_call>",
    "<|tool_call>",
    "<function=python>",
    "[TOOL_CALLS]",
    "x[ARGS]{",
    "web_search[ARGS]{}",
    "<｜tool▁calls▁begin｜>",
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    "call:web_search{q:1}",
    "call \n\t :python{code:1}",
    "<|python_tag|>",
    "<param",
    # Closed forms: the non-final strip removes these, so a missed trigger changes its result.
    "<function=web_search>x</function>",
    '[TOOL_CALLS]web_search{"q": 1}',
    "[TOOL_CALLS][]",
    "<|tool_call>call:web_search{q:1}<tool_call|>",
    "<｜tool▁calls▁begin｜>x<｜tool▁calls▁end｜>",
    # Kimi K2: only ``<|tool_call`` sees these, as neither holds ``<tool_call``.
    "<|tool_calls_section_begin|><|tool_call_begin|>x<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_call_begin|>x<|tool_call_end|>",
    "<think>t</think>",
    "\n</think>\n",
    "\ncall:web_search{q:1}\n",
)


def test_every_strip_sentinel_is_an_open_call_trigger():
    """A new strip arm adds its literal to ``_STRIP_SENTINELS``; the gate must see it too. The two
    left out have no arm a non-final strip runs, and a final strip cuts them with the open call."""
    from core.inference import tool_call_parser as parser

    missing = (
        set(parser._STRIP_SENTINELS)
        - {parser._GEMMA_BARE_SENTINEL}
        - set(parser._OPEN_CALL_TRIGGERS)
    )
    assert missing == {"<|python_tag|>", "<|content_invoke_tool_json|>"}, missing


@pytest.mark.parametrize("trigger", _TRIGGERS)
@pytest.mark.parametrize("seg_final", [True, False])
def test_every_trigger_split_one_char_at_a_time_matches(trigger, seg_final):
    # Past the ~128 characters ``_is_extension`` needs, so the gate answers later tokens.
    prefix = 'ok <tool_call>{"name": "web_search", "arguments": {"query": "' + "words " * 40
    chunks = [prefix] + list(trigger + ' more" }}') + ["</tool_call>", " after"]
    for names in (ENABLED, None):
        assert _replay(StreamingMarkupStripper(names, seg_final = seg_final), chunks) == _replay(
            _FullStripEveryToken(names, seg_final = seg_final), chunks
        )


_PREFIXES = (
    "",
    "I'll search. ",
    "a [link](x) ",
    "brace {x} ",
    "call me ",
    "x `y` z\n",
    "<think>p</think>\n",
    "</think>stray ",
    "<parameter=x>v\n",
    "call :",
    "  　",
    "line\n",
)
_PIECES = (
    "word ",
    '\\"',
    "\\\\",
    "\\n",
    "{",
    "}",
    "[",
    "]",
    ":",
    "call",
    "call ",
    " :",
    "recall:",
    "<",
    "</",
    "<tool_call>",
    "</tool_call>",
    "<|tool_call>",
    "<function=python>",
    "[ARGS]",
    "[TOOL_CALLS]",
    "<think>",
    "</think>",
    "[/THINK]",
    "<｜",
    "　",
    "\n",
    "`",
    "<param",
    "中文",
)


def test_gate_matches_full_strip_on_seeded_open_calls():
    rng = random.Random(20261027)
    engaged = 0
    for _ in range(250):
        text = rng.choice(_PREFIXES) + rng.choice(
            ("<tool_call>", "<tool_call>\n", "<tool_call>name\n")
        )
        text += (
            '{"name": "' + rng.choice(("web_search", "python", "nope")) + '", "arguments": {"q": "'
        )
        text += "filler " * 25
        plain = rng.random()
        text += "".join(
            rng.choice(_PIECES) if rng.random() > plain else "word "
            for _ in range(rng.randint(0, 40))
        )
        text += rng.choice(
            ("", '"}}', '"}}</tool_call>', '"}}</tool_call> then <tool_call>{"name": "x", "a')
        )
        for names in (ENABLED, None):
            for seg_final in (True, False):
                gated = StreamingMarkupStripper(names, seg_final = seg_final)
                plain_stripper = _FullStripEveryToken(names, seg_final = seg_final)
                pos = 0
                while pos < len(text):
                    pos = min(len(text), pos + rng.randint(1, 4))
                    was_on = gated._gate_on
                    assert gated.strip(text[:pos]) == plain_stripper.strip(text[:pos]), (text, pos)
                    engaged += was_on and gated._gate_on
    # The gate must actually answer calls here, or this test proves nothing.
    assert engaged > 1000, engaged


def test_open_call_keeps_no_reference_to_the_buffer():
    stripper = StreamingMarkupStripper(ENABLED)
    text = 'Sure. <tool_call>{"name": "web_search", "arguments": {"query": "' + "w " * 50
    stripper.strip(text)
    text += "more"
    stripper.strip(text)
    assert stripper._gate_on
    assert all(getattr(stripper, slot) is not text for slot in StreamingMarkupStripper.__slots__)


def test_repeated_open_call_strip_is_idempotent():
    stripper = StreamingMarkupStripper(ENABLED)
    text = 'Sure. <tool_call>{"name": "web_search", "arguments": {"query": "x'
    first = stripper.strip(text)
    assert stripper.strip(text) == stripper.strip(text) == first == "Sure. "
