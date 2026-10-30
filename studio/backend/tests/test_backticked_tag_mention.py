# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool tag alone in an inline code span is prose: shown by every display strip, never executed.

A model explaining tool calling writes "Use `<tool_call>` tags." The display strips read the
backticked opener as an unclosed call at end of turn and cut the answer to "Use `", and a
backticked `<function=NAME>` was even run as a zero-argument call. Real calls (after a mention,
with backticks in their arguments, inside a code fence) must still be hidden and executed.
"""

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core import tool_healing  # noqa: E402
from core.inference.tool_call_parser import (  # noqa: E402
    _BLOCKED_BODY_MASK,
    StreamingMarkupStripper,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from routes.inference import (  # noqa: E402
    _hold_pending_mention,
    _ReasoningSpanGuard,
    _StreamingDisplayStrip,
    _strip_tool_xml,
    _strip_tool_xml_for_display,
)

NAMES = {"web_search", "python"}

MENTIONS = [
    "`<tool_call>`",
    "`</tool_call>`",
    "``<tool_call>``",
    "` <tool_call> `",
    "`<tool_call>web_search`",
    "`<function=web_search>`",
    '`<function name="web_search">`',
    "`</function>`",
    "`<parameter=query>`",
    "`</parameter>`",
    "`<|tool_call>`",
    "`<tool_call|>`",
    "`[TOOL_CALLS]`",
    "`[TOOL_CALLS]web_search`",
    "`[/TOOL_CALLS]`",
    "`[ARGS]`",
    "`<|python_tag|>`",
    "`<｜tool▁calls▁begin｜>`",
    "`<|tool_calls_section_begin|>`",
    "`<|tool_call_begin|>`",
]

PROSE = [
    "Use {m} tags.\nok",
    "Use {m}",
    "Open with {m} and close with `</tool_call>`, then stop.",
    "Wrap the JSON in {m} and `</function>` pairs.\n\nThat is all.",
]

JSON_CALL = '<tool_call>\n{"name": "web_search", "arguments": {"query": "unsloth"}}\n</tool_call>'
FUNC_CALL = "<function=web_search>\n<parameter=query>\nunsloth\n</parameter>\n</function>"


def _route(text, heal = True):
    return _strip_tool_xml_for_display(text, auto_heal_tool_calls = heal, enabled_tool_names = NAMES)


def _calls(text, heal = True):
    out = parse_tool_calls_from_text(text, allow_incomplete = heal, enabled_tool_names = NAMES)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in out]


def _prose_cases():
    return [p.format(m = m) for m in MENTIONS for p in PROSE]


@pytest.mark.parametrize("text", _prose_cases())
def test_backticked_tag_is_shown_by_every_display_strip(text):
    assert _route(text) == text
    assert _route(text, heal = False) == text
    assert _strip_tool_xml(text, NAMES) == text
    assert strip_tool_markup(text, final = True, enabled_tool_names = NAMES) == text.strip()
    assert strip_tool_markup(text, final = False, enabled_tool_names = NAMES) == text
    assert (
        _ReasoningSpanGuard(None).strip(text, auto_heal_tool_calls = True, enabled_tool_names = NAMES)
        == text
    )


@pytest.mark.parametrize("text", _prose_cases())
def test_backticked_tag_is_never_executed(text):
    assert _calls(text, heal = True) == []
    assert _calls(text, heal = False) == []


def test_reported_example():
    text = "Use `<tool_call>` tags.\nok"
    assert _route(text) == text
    assert strip_tool_markup(text, final = True, enabled_tool_names = NAMES) == text


def test_backticked_function_opener_no_longer_runs_a_zero_arg_call():
    # Before: Auto-Heal ran web_search({}) for the first, strict mode for the second.
    assert _calls("Use `<function=web_search>` tags.\nok") == []
    assert _calls("Use `<function=web_search>` and `</function>` tags.", heal = False) == []


def _assert_streams(text, stripper_factory, expect_final):
    stripper = stripper_factory()
    prev = ""
    for i in range(1, len(text) + 1):
        shown = stripper.strip(text[:i])
        assert shown.startswith(prev), (text[:i], prev, shown)
        prev = shown
    assert prev == expect_final


PROSE_STREAM_CASES = [
    "Use `<tool_call>` tags.\nok",
    "Use ``<tool_call>`` and `</tool_call>`, or `<function=web_search>`.\nok",
    "Emit `[TOOL_CALLS]web_search` then `<|python_tag|>`.\nok",
    "Open `<tool_call>` and close `</tool_call>`.\nok",
    "Close with `</tool_call>`.\nok",
    "Run `ls`<b>`x`, then `<|tool_call>` and ` <｜tool▁calls▁begin｜> `.\nok",
]
STREAM_CASES = PROSE_STREAM_CASES + [
    "Use `<tool_call>` tags.\n" + JSON_CALL + "\nDone.",
    "Use `<function=web_search>` tags.\n" + FUNC_CALL,
]


class _RouteStream:
    """What the chat route runs per cumulative content event."""

    def __init__(self, heal = True):
        self._heal = heal
        self._strip = _StreamingDisplayStrip(auto_heal_tool_calls = heal, enabled_tool_names = NAMES)

    def strip(self, text):
        return self._strip.strip(_hold_pending_mention(text, self._heal))


@pytest.mark.parametrize("text", PROSE_STREAM_CASES)
def test_route_stream_is_append_only_and_ends_on_the_one_shot_strip(text):
    # The route sends clean[len(prev):] per event, so a display that shrinks and regrows duplicates text.
    _assert_streams(text, _RouteStream, _route(text))
    assert _route(text) == text


class _GuardStream:
    """The Messages API emitter: one span guard fed every cumulative content event."""

    def __init__(self):
        self._guard = _ReasoningSpanGuard(None)

    def strip(self, text):
        return self._guard.strip(text, auto_heal_tool_calls = True, enabled_tool_names = NAMES)


@pytest.mark.parametrize("text", PROSE_STREAM_CASES)
def test_messages_api_stream_is_append_only(text):
    _assert_streams(text, _GuardStream, text)


@pytest.mark.parametrize("text", STREAM_CASES)
def test_route_stream_ends_on_the_one_shot_strip(text):
    stripper = _RouteStream()
    for i in range(1, len(text) + 1):
        shown = stripper.strip(text[:i])
    assert shown == _route(text)
    assert shown.startswith(text.split("\n", 1)[0])


@pytest.mark.parametrize("seg_final", [True, False])
@pytest.mark.parametrize("text", STREAM_CASES)
def test_generator_stream_matches_a_fresh_strip_at_every_prefix(text, seg_final):
    stripper = StreamingMarkupStripper(NAMES, seg_final = seg_final)
    for i in range(1, len(text) + 1):
        prefix = text[:i]
        assert stripper.strip(prefix) == StreamingMarkupStripper(NAMES, seg_final = seg_final).strip(
            prefix
        ), prefix


@pytest.mark.parametrize("call", [JSON_CALL, FUNC_CALL])
def test_real_call_after_a_mention_is_hidden_and_executed(call):
    text = "Use `<tool_call>` or `<function=x>` tags.\n" + call
    assert _calls(text) == [("web_search", {"query": "unsloth"})]
    assert _route(text) == "Use `<tool_call>` or `<function=x>` tags.\n"
    assert strip_tool_markup(text, final = True, enabled_tool_names = NAMES) == (
        "Use `<tool_call>` or `<function=x>` tags."
    )


@pytest.mark.parametrize(
    "query",
    ["run `ls` then `pwd`", "explain `<tool_call>` tags", "`</function>` closes it"],
)
def test_real_call_with_backticks_in_its_arguments(query):
    for call in (
        '<tool_call>\n{"name": "web_search", "arguments": {"query": %s}}\n</tool_call>'
        % json.dumps(query),
        f"<function=web_search>\n<parameter=query>\n{query}\n</parameter>\n</function>",
    ):
        text = "Searching.\n" + call
        calls = _calls(text)
        assert calls == [("web_search", {"query": query})], call
        assert _BLOCKED_BODY_MASK not in calls[0][1]["query"]
        assert strip_tool_markup(text, final = True, enabled_tool_names = NAMES) == "Searching."
        if "</function>" in query and call.startswith("<tool_call>"):
            # Pre-existing and unchanged: the route regex closes a <tool_call> at a quoted </function>.
            assert _route(text) == _strip_tool_xml_for_display(
                text.replace("`", "\x00"), auto_heal_tool_calls = True, enabled_tool_names = NAMES
            ).replace("\x00", "`")
            continue
        assert _route(text) == "Searching.\n"


@pytest.mark.parametrize("call", [JSON_CALL, FUNC_CALL])
def test_fenced_call_keeps_current_behaviour(call):
    # A fenced call is still run (current behaviour), so its markup stays hidden.
    text = "Running it:\n```\n" + call + "\n```\nok"
    assert _calls(text) == [("web_search", {"query": "unsloth"})]
    assert "<tool_call>" not in _route(text) and "<function=" not in _route(text)


def test_auto_heal_off_route_is_verbatim_and_strict_parse_ignores_mentions():
    text = "Use `<function=web_search>` and `</function>`.\n`<tool_call>` too."
    assert _route(text, heal = False) == text
    # Nothing is stripped with Auto-Heal off, so nothing is held either.
    stream = _RouteStream(heal = False)
    assert [stream.strip(text[:i]) for i in range(1, len(text) + 1)] == [
        text[:i] for i in range(1, len(text) + 1)
    ]
    assert _calls(text, heal = False) == []


def test_stream_display_strip_itself_still_equals_the_full_strip():
    # The hold lives in the callers, so the incremental strip keeps its exact contract.
    for text in STREAM_CASES:
        strip = _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = NAMES)
        for i in range(1, len(text) + 1):
            assert strip.strip(text[:i]) == _route(text[:i])


def test_mention_spans_exclude_code_that_holds_more_than_a_tag():
    assert tool_healing.tag_mention_spans('a `<tool_call>{"x": 1}` b') == []
    assert tool_healing.tag_mention_spans("a `<think>` b") == []
    assert tool_healing.tag_mention_spans("no backticks <tool_call>") == []
    text = "a `<tool_call>` b"
    (span,) = tool_healing.tag_mention_spans(text)
    assert text[span[0] : span[1]] == "<tool_call>"
    assert tool_healing.in_tag_mention([span], span[0])
    assert not tool_healing.in_tag_mention([span], span[1])


@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize(
    "answer",
    [
        "Use `<tool_call>` tags.\nok",
        "Use `<function=web_search>` tags.\nok",
        "Open `<|tool_call>` and close `</tool_call>`.\nok",
    ],
)
def test_gguf_tool_loop_to_route_stream_end_to_end(monkeypatch, answer, chunk):
    """Real GGUF tool loop on a fake llama-server stream, then the route's per-event strip and delta send."""
    sys.path.insert(0, str(BACKEND_ROOT / "tests"))
    from test_llama_cpp_tool_loop import _done, _finish, _make_backend, _sse, _web_search_tool

    executed = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_k: executed.append((name, arguments)) or "ok",
    )
    tokens = [answer[i : i + chunk] for i in range(0, len(answer), chunk)]
    streams = [
        [_sse({"content": t}) for t in tokens] + [_finish("stop"), _done()],
        [_sse({"content": "unexpected second turn"}), _finish("stop"), _done()],
    ]
    backend = _make_backend(monkeypatch, streams, [])
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "How do I call tools?"}],
            tools = [_web_search_tool()],
            max_tool_iterations = 2,
            auto_heal_tool_calls = True,
        )
    )
    strip = _RouteStream()
    prev, client = "", ""
    for event in events:
        if event.get("type") == "content":
            clean = strip.strip(event.get("text", ""))
            client += clean[len(prev) :]
            prev = clean
    assert executed == []
    assert client == answer
