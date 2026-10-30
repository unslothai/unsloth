# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-M2 / M2.1 ``<minimax:tool_call>`` calls that reach Studio as text: llama-server leaves them
in content whenever its own tool parser does not run (tool_choice "none", the final answer pass)."""

import json
import random
import sys
import time
from pathlib import Path

import pytest

# The tool-loop harness sits alongside and imports as a top-level module.
_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from core.inference.chat_template_helpers import neutralize_control_markup  # noqa: E402
from core.inference.tool_call_parser import (  # noqa: E402
    StreamingMarkupStripper,
    TOOL_XML_SIGNALS,
    _inside_minimax_value,
    _MinimaxValueIndex,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
)

ENABLED = {"web_search", "python"}
SEARCH = (
    '<minimax:tool_call>\n<invoke name="web_search">\n'
    '<parameter name="query">weather in Paris</parameter>\n</invoke>\n</minimax:tool_call>'
)


def _calls(text, **kwargs):
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED, **kwargs)
    ]


def _streamed(text, enabled = ENABLED):
    stripper = StreamingMarkupStripper(enabled)
    out = ""
    for end in range(1, len(text) + 1):
        out = stripper.strip(text[:end])
    return out


def test_a_well_formed_block_is_a_call():
    assert _calls(SEARCH) == [("web_search", {"query": "weather in Paris"})]
    assert _calls(SEARCH, allow_incomplete = False) == [("web_search", {"query": "weather in Paris"})]


def test_the_block_opener_wakes_the_stream_buffer():
    assert "<minimax:tool_call>" in TOOL_XML_SIGNALS


def test_the_compact_spelling_is_a_call():
    text = (
        '<minimax:tool_call><invoke name="web_search"><parameter name="query">weather in Paris'
        "</parameter></invoke></minimax:tool_call>"
    )
    assert _calls(text) == [("web_search", {"query": "weather in Paris"})]


@pytest.mark.parametrize(
    "text",
    [
        SEARCH[: -len("</minimax:tool_call>")],
        SEARCH[: -len("</invoke>\n</minimax:tool_call>")],
        SEARCH[: -len("</parameter>\n</invoke>\n</minimax:tool_call>")],
    ],
    ids = ["no_block_close", "no_invoke_close", "no_parameter_close"],
)
def test_a_truncated_block_heals_and_strict_mode_keeps_only_closed_invokes(text):
    assert _calls(text) == [("web_search", {"query": "weather in Paris"})]
    strict = _calls(text, allow_incomplete = False)
    if text.rstrip().endswith("</invoke>"):
        assert strict == [("web_search", {"query": "weather in Paris"})]
    else:
        assert strict == []


def test_two_invokes_and_two_blocks_are_two_calls():
    second = '<invoke name="python">\n<parameter name="code">print(1 + 1)</parameter>\n</invoke>\n'
    one_block = SEARCH.replace("</invoke>\n", "</invoke>\n" + second)
    two_blocks = SEARCH + "\n<minimax:tool_call>\n" + second + "</minimax:tool_call>"
    expected = [
        ("web_search", {"query": "weather in Paris"}),
        ("python", {"code": "print(1 + 1)"}),
    ]
    assert _calls(one_block) == expected
    assert _calls(two_blocks) == expected


def test_a_multiline_code_value_keeps_its_text():
    code = 'a, b = 1, 2\nif a < b:\n    print("a < b </invoke>")'
    text = (
        '<minimax:tool_call>\n<invoke name="python">\n<parameter name="code">'
        + code
        + "\n</parameter>\n</invoke>\n</minimax:tool_call>"
    )
    assert _calls(text) == [("python", {"code": code})]
    assert strip_tool_markup("Running it.\n" + text, final = True, enabled_tool_names = ENABLED) == (
        "Running it."
    )


def test_values_stay_strings_like_the_function_xml_form():
    text = SEARCH.replace("</invoke>", '<parameter name="max_results">5</parameter>\n</invoke>')
    assert _calls(text) == [("web_search", {"query": "weather in Paris", "max_results": "5"})]


def test_a_block_inside_reasoning_is_not_run():
    rehearsal = SEARCH.replace("web_search", "python").replace("query", "code")
    assert _calls("<think>I could do " + rehearsal + " instead</think>\n" + SEARCH) == [
        ("web_search", {"query": "weather in Paris"})
    ]
    # The template opens <think> in the prompt, so only a stray closer marks the reasoning.
    assert _calls("I could do " + rehearsal + "\n</think>\n\n" + SEARCH) == [
        ("web_search", {"query": "weather in Paris"})
    ]
    assert _calls("<think>I could do " + rehearsal) == []


def test_the_displayed_answer_drops_the_block_and_keeps_the_prose():
    text = "Let me look that up.\n" + SEARCH
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == "Let me look that up."
    assert _streamed(text).strip() == "Let me look that up."
    # An unclosed block at the end of the turn is a truncated call, not prose.
    truncated = text[: -len("</invoke>\n</minimax:tool_call>")]
    assert strip_tool_markup(truncated, final = True, enabled_tool_names = ENABLED) == (
        "Let me look that up."
    )
    # Mid-stream (final answer pass) a closed block goes and an open one stays buffered.
    assert strip_tool_markup(text + "\nDone.") == "Let me look that up.\n\nDone."


def test_a_quoted_block_inside_another_call_stays_data():
    qwen = (
        '<tool_call>{"name": "python", "arguments": {"code": '
        + json.dumps("s = '" + SEARCH + "'")
        + "}}</tool_call>"
    )
    assert [name for name, _ in _calls(qwen)] == ["python"]
    function_xml = (
        "<function=python><parameter=code>s = '''" + SEARCH + "'''</parameter></function>"
    )
    assert _calls(function_xml) == [("python", {"code": "s = '''" + SEARCH + "'''"})]


def test_prose_naming_the_format_is_left_alone():
    text = 'MiniMax writes calls as `<invoke name="x">` blocks.'
    assert _calls(text) == []
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text
    assert _streamed(text) == text


def test_a_pasted_block_opener_is_neutralized():
    assert "<minimax:tool_call>" not in neutralize_control_markup("a " + SEARCH + " b")
    assert "</minimax:tool_call>" not in neutralize_control_markup("a " + SEARCH + " b")


def _content_snapshots(events):
    """Content events are cumulative snapshots of the visible answer."""
    return [event.get("text", "") for event in events if event.get("type") == "content"]


@pytest.mark.parametrize("chunk", [None, 7], ids = ["one_delta", "seven_char_deltas"])
def test_the_gguf_tool_loop_runs_a_text_block_and_hides_it(monkeypatch, chunk):
    reply = "Let me look that up.\n" + SEARCH
    pieces = (
        [reply] if chunk is None else [reply[i : i + chunk] for i in range(0, len(reply), chunk)]
    )
    streams = [
        [_sse({"content": piece}) for piece in pieces] + [_done()],
        [_sse({"content": "It is sunny."}), _done()],
    ]
    backend, _payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather in Paris?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert calls == [("web_search", {"query": "weather in Paris"})]
    snapshots = _content_snapshots(events)
    # A half-streamed opener can show for one snapshot, as with every other format; a complete one never does.
    assert not any(
        "<minimax:tool_call>" in text or "<invoke" in text for text in snapshots
    ), snapshots
    assert "Let me look that up." in snapshots[-2]
    assert snapshots[-1] == "It is sunny."


def test_the_value_index_answers_like_the_shared_helper():
    """The parser asks the index, not the per-tag scan; both must agree on every offset."""
    rng = random.Random(0)
    pieces = (
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        '<invoke name="python">',
        "</invoke>",
        '<parameter name="code">',
        "<parameter=code>",
        '<param name="x">',
        "</parameter>",
        "</param>",
        "<param",
        " ",
        "\n",
        "a",
        "<",
        ">",
    )
    for _ in range(400):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 24)))
        index = _MinimaxValueIndex(text)
        for pos in range(len(text) + 1):
            assert index.inside(pos) == _inside_minimax_value(text, pos), (text, pos)


@pytest.mark.parametrize(
    "text",
    [
        '<parameter name="p">' + "<minimax:tool_call>" * 50000,
        '<minimax:tool_call><invoke name="python"><parameter name="code">'
        + '<invoke name="a">' * 50000,
        '<minimax:tool_call><invoke name="python">' + '<parameter name="p">' * 50000,
        "<think>a</think>" * 30000 + SEARCH * 3000,
    ],
    ids = ["openers_in_a_value", "invokes_in_a_value", "unclosed_params", "think_spans"],
)
def test_a_megabyte_of_adversarial_markup_parses_in_linear_time(text):
    start = time.perf_counter()
    parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    strip_tool_markup(text, final = True, enabled_tool_names = ENABLED)
    # Each took 35-90 s with a per-tag scan of the text.
    assert time.perf_counter() - start < 5.0


def _client_view(events, strip):
    """What the GGUF streaming route sends: each cumulative event goes through the display strip and only the
    growth is forwarded (append-only); an empty status or a tool_start starts a new turn."""
    shown, prev = "", ""
    for event in events:
        if event.get("type") == "tool_start" or (
            event.get("type") == "status" and not event.get("text")
        ):
            prev = ""
        elif event.get("type") == "content":
            clean = strip.strip(event.get("text", ""))
            shown += clean[len(prev) :]
            prev = clean
    return shown


@pytest.mark.parametrize("closed", [True, False], ids = ["closed", "unclosed"])
def test_the_route_display_strip_hides_a_block(closed):
    from routes.inference import _strip_tool_xml_for_display

    text = "Let me look that up.\n" + (SEARCH if closed else SEARCH[: -len("</minimax:tool_call>")])
    shown = _strip_tool_xml_for_display(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
    assert shown == "Let me look that up.\n"
    prose = 'MiniMax writes calls as `<invoke name="x">` blocks.'
    assert (
        _strip_tool_xml_for_display(prose, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
        == prose
    )


def test_a_block_in_the_final_answer_pass_never_reaches_the_client(monkeypatch):
    """The capped final pass sends tool_choice "none", so llama-server leaves the block in content; its
    streaming stripper keeps an open block and the route forwards growth only, so the route must hide it."""
    from routes.inference import _StreamingDisplayStrip

    reply = "Let me look that up.\n" + SEARCH
    streams = [
        [
            _sse(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "weather in Paris"}),
                            },
                        }
                    ]
                }
            ),
            _done(),
        ],
        [_sse({"content": reply[i : i + 7]}) for i in range(0, len(reply), 7)] + [_done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "weather in Paris?"}],
        [{"type": "function", "function": {"name": "web_search"}}],
    )

    assert payloads[-1].get("tool_choice") == "none"
    assert calls == [("web_search", {"query": "weather in Paris"})]
    shown = _client_view(
        events, _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
    )
    assert "Let me look that up." in shown
    assert "<invoke" not in shown and "<parameter" not in shown, shown
    assert "</minimax:tool_call>" not in shown, shown
