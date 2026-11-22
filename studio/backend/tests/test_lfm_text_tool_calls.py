# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM2 / LFM2.5 text tool calls (#18802). Blocks are real llama-server b12588 outputs (LFM2.5-1.2B-Thinking
Q4_K_M, tool_choice "none", seed 3407; the last is the reporter's LFM2.5-24B-A2B)."""

import json
import sys
import time
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
for _entry in (_BACKEND_DIR, _TESTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from core.inference.chat_template_helpers import neutralize_control_markup  # noqa: E402
from core.inference.native_tool_tokens import closes_an_open_envelope  # noqa: E402
from core.inference.tool_call_parser import (  # noqa: E402
    StreamingMarkupStripper,
    TOOL_XML_SIGNALS,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from core.inference.tool_loop_controller import ToolLoopController  # noqa: E402
from core.inference.tools import WEB_SEARCH_TOOL  # noqa: E402
from test_llama_cpp_tool_loop import (  # noqa: E402
    _backend_and_payloads,
    _done,
    _finish,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
)

START = "<|tool_call_start|>"
END = "<|tool_call_end|>"
ENABLED = {"web_search", "python"}

SINGLE = (
    "\n<|tool_call_start|>[web_search(query='', url='https://www.liquid.ai/blog/lfm2-5-24b-a2b')]"
    "<|tool_call_end|>"
)
MULTI = (
    "\n<|tool_call_start|>[web_search(query='LFM2.5-24B-A2B benchmarks'), "
    "web_search(query='LFM2.5-24B-A2B license')]<|tool_call_end|>"
)
QUOTES_AND_COMMAS = (
    '\n<|tool_call_start|>[web_search(query="Liquid AI\'s LFM2.5, 24B-A2B (license, benchmarks)")]'
    "<|tool_call_end|>"
)
# Sampled after a web_search result was replayed: the model asks to read the top hit.
AFTER_RESULTS = (
    "\n<|tool_call_start|>[web_search(query='LFM2.5-24B-A2B', "
    "url='https://www.liquid.ai/blog/lfm2-5-24b-a2b')]<|tool_call_end|>"
)
# The same quote inside a single-quoted string, unescaped.
UNESCAPED_QUOTE = (
    "\n<|tool_call_start|>[web_search(query='what's new in LFM2.5, and why it matters')]"
    "<|tool_call_end|>"
)
# Neither value can be recovered: inner quotes end the string early, or it never closes.
MALFORMED_INNER_QUOTES = (
    '\n<|tool_call_start|>[web_search(query="it\'s "LFM2.5", 24B, A2B, url="")]<|tool_call_end|>'
)
MALFORMED_UNCLOSED = (
    "\n<|tool_call_start|>[web_search(query='LFM2.5 'A2B', MoE, \"24B\" release)]<|tool_call_end|>"
)
REPORTED = (
    "<|tool_call_start|>[web_fetch(url='https://localai.stats.com/models/hf-liquidai-lfm2-5-24b-a2b')]"
    "<|tool_call_end|>"
)


def _calls(text, **kwargs):
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED, **kwargs)
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            SINGLE,
            [("web_search", {"query": "", "url": "https://www.liquid.ai/blog/lfm2-5-24b-a2b"})],
        ),
        (
            MULTI,
            [
                ("web_search", {"query": "LFM2.5-24B-A2B benchmarks"}),
                ("web_search", {"query": "LFM2.5-24B-A2B license"}),
            ],
        ),
        (
            QUOTES_AND_COMMAS,
            [("web_search", {"query": "Liquid AI's LFM2.5, 24B-A2B (license, benchmarks)"})],
        ),
        (
            AFTER_RESULTS,
            [
                (
                    "web_search",
                    {"query": "LFM2.5-24B-A2B", "url": "https://www.liquid.ai/blog/lfm2-5-24b-a2b"},
                )
            ],
        ),
        (
            REPORTED,
            [("web_fetch", {"url": "https://localai.stats.com/models/hf-liquidai-lfm2-5-24b-a2b"})],
        ),
        # The chat template's own rendering: quotes, backslashes and newlines escaped, mappings as JSON.
        (
            START
            + '[python(code=\'print(\\\'a, b\\\')\\nx = "\\\\\\\\"\', opts={"n": 2, "ok": true})]'
            + END,
            [("python", {"code": "print('a, b')\nx = \"\\\\\"", "opts": {"n": 2, "ok": True}})],
        ),
        (
            START + "[f(a=True, b=None, c=-1.5, d=[1, 'two'], e=null, g=false)]" + END,
            [("f", {"a": True, "b": None, "c": -1.5, "d": [1, "two"], "e": None, "g": False})],
        ),
        # Older LFM2 templates asked for JSON inside the same markers.
        (
            START + '[{"name": "web_search", "arguments": {"query": "mars rover"}}]' + END,
            [("web_search", {"query": "mars rover"})],
        ),
        (START + "[get_time()]" + END, [("get_time", {})]),
        # A sampled string can hold raw line breaks the template would have escaped.
        (
            START + "[python(code='print(1)\nprint(2)')]" + END,
            [("python", {"code": "print(1)\nprint(2)"})],
        ),
        # A closer quoted inside an argument is data, not the end of the block.
        (
            START + "[python(code='print(\"<|tool_call_end|>\")')]" + END,
            [("python", {"code": 'print("<|tool_call_end|>")'})],
        ),
    ],
    ids = [
        "single",
        "multiple_in_one_list",
        "quotes_and_commas",
        "after_tool_results",
        "reported_24b",
        "template_escapes",
        "python_and_json_literals",
        "json_spelling",
        "no_arguments",
        "raw_newline_in_string",
        "quoted_closer",
    ],
)
def test_a_text_block_parses_to_calls(text, expected):
    assert _calls(text) == expected
    # Prose before the block is not part of the call.
    assert _calls("Let me look that up." + text) == expected


def test_a_closed_block_parses_with_auto_heal_off_and_an_unclosed_one_does_not():
    assert _calls(MULTI, allow_incomplete = False) == _calls(MULTI)
    unclosed = MULTI[: -len(END)]
    assert _calls(unclosed) == _calls(MULTI)
    assert _calls(unclosed, allow_incomplete = False) == []


def test_auto_heal_reads_a_string_holding_its_own_quote():
    assert _calls(UNESCAPED_QUOTE) == [
        ("web_search", {"query": "what's new in LFM2.5, and why it matters"})
    ]
    assert _calls(UNESCAPED_QUOTE, allow_incomplete = False) == []
    # Escapes the model did write still decode.
    assert _calls(START + "[web_search(query='it's a \\'test\\'\\nnext')]" + END) == [
        ("web_search", {"query": "it's a 'test'\nnext"})
    ]
    # A second ``key=`` inside means the string's end is unknowable.
    assert _calls(START + "[web_search(query='it's a', url='b')]" + END) == []


@pytest.mark.parametrize(
    "text",
    [
        MALFORMED_INNER_QUOTES,
        MALFORMED_UNCLOSED,
        START + "[web_search('positional only')]" + END,
        START + "[web_search(query=os.system('x'))]" + END,
        START + "[web_search(query=f'{x}')]" + END,
        START + "[web_search(query='a') trailing]" + END,
        START + "[web_search(query='a')" + END,
        START + "[not a call]" + END,
        START + "[]" + END,
        START + END,
        "LFM writes " + START + " before a call list and " + END + " after it.",
        "The `<|tool_call_start|>` token opens a call.",
    ],
    ids = [
        "inner_quotes",
        "unclosed_string",
        "positional",
        "expression",
        "f_string",
        "trailing_junk",
        "missing_bracket",
        "not_a_call",
        "empty_list",
        "empty_block",
        "prose_mention",
        "backticked_mention",
    ],
)
def test_malformed_text_is_never_a_call(text):
    assert _calls(text) == []
    assert _calls(text, allow_incomplete = False) == []


@pytest.mark.parametrize(
    "text",
    [
        START + "[not a call]" + END + " tail",
        START + "[]" + END,
        "LFM writes " + START + " before a call list and " + END + " after it.",
        "The marker " + START + " opens a call list.",
    ],
)
def test_text_that_is_not_a_call_list_is_shown_as_typed(text):
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text.strip()


def test_a_block_inside_reasoning_is_a_rehearsal():
    text = "<think>I could call " + SINGLE + " here.</think>The page says it is a MoE."
    assert _calls(text) == []
    assert _calls("<think>Plan.</think>" + SINGLE) == _calls(SINGLE)


@pytest.mark.parametrize(
    "text", [SINGLE, MULTI, QUOTES_AND_COMMAS, AFTER_RESULTS, REPORTED, MALFORMED_INNER_QUOTES]
)
def test_the_display_strip_hides_a_call_block_and_keeps_the_prose(text):
    assert START in TOOL_XML_SIGNALS
    assert has_tool_signal("Let me check." + text)
    assert strip_tool_markup("Let me check." + text, final = True, enabled_tool_names = ENABLED) == (
        "Let me check."
    )
    # Mid-stream an open block is held, not shown and not dropped.
    open_tail = "Let me check." + text[: -len(END)]
    assert strip_tool_markup(open_tail, final = True, enabled_tool_names = ENABLED) == "Let me check."


def test_the_streaming_stripper_never_shows_a_complete_block():
    """An open block is held as with MiniMax (the route strip hides it, below); once closed it is gone."""
    reply = "Let me look that up." + MULTI + " Done."
    stripper = StreamingMarkupStripper(ENABLED, seg_final = False)
    snapshots = [stripper.strip(reply[:i]) for i in range(1, len(reply) + 1)]
    assert not any(END in snap for snap in snapshots), snapshots
    assert snapshots[-1] == "Let me look that up.\n Done."
    assert strip_tool_markup(reply, final = True, enabled_tool_names = ENABLED) == (
        "Let me look that up.\n Done."
    )


def test_the_route_display_strip_hides_a_block():
    from routes.inference import _strip_tool_xml_for_display

    for text in ("Let me look that up." + REPORTED, "Let me look that up." + REPORTED[: -len(END)]):
        shown = _strip_tool_xml_for_display(
            text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED
        )
        assert shown == "Let me look that up."
    prose = "LFM opens calls with `<|tool_call_start|>`."
    assert (
        _strip_tool_xml_for_display(prose, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
        == prose
    )


def test_pasted_markup_is_neutralized_and_the_closer_is_kept_for_the_parser():
    assert START not in neutralize_control_markup("see " + SINGLE)
    assert END not in neutralize_control_markup("see " + SINGLE)
    # A runtime stop on the closer is kept when the LFM opener is in the text.
    assert closes_an_open_envelope(SINGLE, END)


@pytest.mark.parametrize(
    "text",
    [
        START * 50000,
        START + "[f(a='" + "x" * 1_000_000,
        START + "[" + "f(a=[" * 20000,
        (START + "[f(a='b')]" + END) * 20000,
        START + "[f(a='" + "\\'" * 300_000 + "')]" + END,
    ],
    ids = ["openers", "unclosed_string", "deep_nesting", "many_blocks", "escapes"],
)
def test_a_megabyte_of_adversarial_markup_parses_in_linear_time(text):
    start = time.perf_counter()
    parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    strip_tool_markup(text, final = True, enabled_tool_names = ENABLED)
    assert time.perf_counter() - start < 10.0


# Page-read names other harnesses ship; LFM2.5 calls web_fetch for what web_search's url argument does.


def _decide(
    name,
    arguments,
    *,
    tools = (WEB_SEARCH_TOOL,),
    auto_heal = True,
):
    controller = ToolLoopController(tools = list(tools), auto_heal_tool_calls = auto_heal)
    return controller.prepare_call(
        {"id": "call_0", "function": {"name": name, "arguments": json.dumps(arguments)}}
    )


@pytest.mark.parametrize("name", ["web_fetch", "WebFetch", "fetch_url", "web-fetch", "fetch"])
def test_a_page_read_call_runs_as_web_search_on_its_url(name):
    decision = _decide(
        name, {"url": "https://www.liquid.ai/blog/lfm2-5-24b-a2b", "prompt": "summarize"}
    )
    assert decision.action == "execute"
    assert decision.tool_name == "web_search"
    assert decision.arguments == {"url": "https://www.liquid.ai/blog/lfm2-5-24b-a2b"}
    assert decision.provenance.get("healed") is True


def test_a_page_read_call_otherwise_stays_an_unknown_tool():
    url = {"url": "https://example.com"}
    assert _decide("web_fetch", {"query": "lfm"}).action == "disabled"
    # Auto-Heal off: names run exactly as sent.
    assert _decide("web_fetch", url, auto_heal = False).action == "disabled"
    # Search is off for this request.
    python_tool = {"type": "function", "function": {"name": "python", "parameters": {}}}
    assert _decide("web_fetch", url, tools = (python_tool,)).action == "disabled"
    assert _decide("open_browser", url).action == "disabled"
    # A real web_fetch tool (an MCP server's, say) wins over the alias.
    web_fetch_tool = {"type": "function", "function": {"name": "web_fetch", "parameters": {}}}
    decision = _decide("web_fetch", url, tools = (WEB_SEARCH_TOOL, web_fetch_tool))
    assert (decision.action, decision.tool_name) == ("execute", "web_fetch")


# The GGUF tool loop end to end, with llama-server streams shaped like the captured ones.


def _content_snapshots(events):
    return [event.get("text", "") for event in events if event.get("type") == "content"]


def test_a_web_fetch_call_keeps_the_research_going(monkeypatch):
    """#18802: an undeclared web_fetch call ended the loop and the final pass showed the raw block."""
    url = "https://www.liquid.ai/blog/lfm2-5-24b-a2b"
    streams = [
        [
            _tool_call_sse("web_search", {"query": "LFM2.5-24B-A2B model"}, "c1"),
            _finish("tool_calls"),
            _done(),
        ],
        [_tool_call_sse("web_fetch", {"url": url}, "c2"), _finish("tool_calls"), _done()],
        [_sse({"content": "It is a 24B MoE with 2B active parameters."}), _finish("stop"), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5-24B-A2B and summarize it."}],
        [WEB_SEARCH_TOOL],
        max_tool_iterations = 50,
    )

    assert calls == [
        ("web_search", {"query": "LFM2.5-24B-A2B model"}),
        ("web_search", {"url": url}),
    ]
    assert [payload.get("tool_choice") for payload in payloads] == ["auto", "auto", "auto"]
    assert _content_snapshots(events)[-1] == "It is a 24B MoE with 2B active parameters."


@pytest.mark.parametrize("chunk", [None, 7], ids = ["one_delta", "seven_char_deltas"])
@pytest.mark.parametrize(
    "block, expected",
    [
        (
            AFTER_RESULTS,
            [
                ("web_search", {"query": "LFM2.5-24B-A2B model"}),
                ("web_search", {"url": "https://www.liquid.ai/blog/lfm2-5-24b-a2b"}),
            ],
        ),
        (
            MULTI,
            [
                ("web_search", {"query": "LFM2.5-24B-A2B model"}),
                ("web_search", {"query": "LFM2.5-24B-A2B benchmarks"}),
                ("web_search", {"query": "LFM2.5-24B-A2B license"}),
            ],
        ),
    ],
    ids = ["after_results", "multiple_in_one_list"],
)
def test_the_gguf_tool_loop_runs_a_text_block_after_tool_results_and_hides_it(
    monkeypatch, chunk, block, expected
):
    reply = "Reading the top hit." + block
    pieces = (
        [reply] if chunk is None else [reply[i : i + chunk] for i in range(0, len(reply), chunk)]
    )
    streams = [
        [
            _tool_call_sse("web_search", {"query": "LFM2.5-24B-A2B model"}, "c1"),
            _finish("tool_calls"),
            _done(),
        ],
        [_sse({"content": piece}) for piece in pieces] + [_finish("stop"), _done()],
        [_sse({"content": "It is a 24B MoE."}), _finish("stop"), _done()],
    ]
    backend, _payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Research LFM2.5-24B-A2B."}],
        [WEB_SEARCH_TOOL],
        max_tool_iterations = 50,
    )

    assert calls == expected
    snapshots = _content_snapshots(events)
    assert not any("web_search(" in text or END in text for text in snapshots), snapshots
    assert snapshots[-1] == "It is a 24B MoE."


def test_malformed_text_in_a_tool_round_stays_the_answer(monkeypatch):
    text = "LFM writes " + START + " before a call list."
    streams = [[_sse({"content": text}), _finish("stop"), _done()]]
    backend, _payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "How does LFM call tools?"}], [WEB_SEARCH_TOOL]
    )

    assert calls == []
    assert _content_snapshots(events)[-1] == text


def _client_view(events, strip):
    """Mimics the GGUF streaming route: strip each cumulative event, forward growth; status/tool_start resets."""
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


def test_a_block_in_the_final_answer_pass_never_reaches_the_client(monkeypatch):
    """The capped final pass sends tool_choice "none", so llama-server returns the block as content."""
    from routes.inference import _StreamingDisplayStrip

    reply = "Here is what I found so far." + REPORTED
    streams = [
        [
            _tool_call_sse("web_search", {"query": "LFM2.5-24B-A2B"}, "c1"),
            _finish("tool_calls"),
            _done(),
        ],
        [_sse({"content": reply[i : i + 7]}) for i in range(0, len(reply), 7)]
        + [_finish("stop"), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    calls = _record_tool_calls(monkeypatch, "result")

    events = _run_tool_loop(
        backend, [{"role": "user", "content": "Research LFM2.5-24B-A2B."}], [WEB_SEARCH_TOOL]
    )

    assert payloads[-1].get("tool_choice") == "none"
    assert calls == [("web_search", {"query": "LFM2.5-24B-A2B"})]
    shown = _client_view(
        events, _StreamingDisplayStrip(auto_heal_tool_calls = True, enabled_tool_names = ENABLED)
    )
    assert "Here is what I found so far." in shown
    assert "web_fetch" not in shown and START not in shown and END not in shown, shown


def test_markup_quoted_across_formats_stays_argument_data():
    qwen = (
        '<tool_call>{"name": "python", "arguments": {"code": "print(\\"'
        + START
        + "[web_search(query='x')]"
        + END
        + '\\")"}}</tool_call>'
    )
    assert _calls(qwen) == [
        ("python", {"code": 'print("' + START + "[web_search(query='x')]" + END + '")'})
    ]
    assert strip_tool_markup("ok " + qwen, final = True, enabled_tool_names = ENABLED) == "ok"
    lfm = (
        START
        + '[python(code=\'print("<tool_call>{\\"name\\": \\"web_search\\"}</tool_call>")\')]'
        + END
    )
    assert _calls(lfm) == [
        ("python", {"code": 'print("<tool_call>{"name": "web_search"}</tool_call>")'})
    ]
    assert strip_tool_markup("ok " + lfm, final = True, enabled_tool_names = ENABLED) == "ok"


# The final answer pass: tool_choice "none" leaves the tools in the prompt, so LFM still writes a call. Banning its
# one opener token, with the budget notice as a user turn, keeps that pass prose (#18802).


class _FakeTokenizeClient:
    calls: list = []

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def post(
        self,
        url,
        json = None,
    ):
        type(self).calls.append((url, json))
        tokens = (
            [10]
            if json == {"content": START, "add_special": False, "parse_special": True}
            else [1, 2]
        )
        return type("R", (), {"status_code": 200, "json": lambda _self: {"tokens": tokens}})()


def test_the_opener_id_is_looked_up_once_and_only_for_an_lfm_template(monkeypatch):
    from core.inference import llama_cpp

    backend, _payloads = _backend_and_payloads(monkeypatch, [])
    _FakeTokenizeClient.calls = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", _FakeTokenizeClient)
    backend._chat_template = "{{ '<|im_start|>' }}"
    assert backend._final_pass_call_opener_id() is None
    assert _FakeTokenizeClient.calls == []
    backend._chat_template = "{{ '<|tool_call_start|>[' }}"
    assert backend._final_pass_call_opener_id() == 10
    assert backend._final_pass_call_opener_id() == 10
    assert len(_FakeTokenizeClient.calls) == 1


@pytest.mark.parametrize(
    "user_bias, expected",
    [
        (None, [[10, False]]),
        ({"42": -5}, {"42": -5, "10": False}),
        ([[42, -5]], [[42, -5], [10, False]]),
    ],
)
def test_the_ban_keeps_the_callers_logit_bias(user_bias, expected):
    from core.inference.llama_cpp import _with_banned_token
    assert _with_banned_token(user_bias, 10) == expected


@pytest.mark.parametrize("opener_id", [None, 10], ids = ["other_template", "lfm_template"])
def test_the_lfm_final_pass_bans_the_opener_and_nudges_as_a_user_turn(monkeypatch, opener_id):
    streams = [
        [
            _tool_call_sse("web_search", {"query": "LFM2.5-24B-A2B"}, "c1"),
            _finish("tool_calls"),
            _done(),
        ],
        [_sse({"content": "It is a 24B MoE."}), _finish("stop"), _done()],
    ]
    backend, payloads = _backend_and_payloads(monkeypatch, streams)
    monkeypatch.setattr(backend, "_final_pass_call_opener_id", lambda: opener_id)
    _record_tool_calls(monkeypatch, "result")

    _run_tool_loop(
        backend, [{"role": "user", "content": "Research LFM2.5-24B-A2B."}], [WEB_SEARCH_TOOL]
    )

    final = payloads[-1]
    assert final["tool_choice"] == "none"
    last = final["messages"][-1]
    if opener_id is None:
        # Unchanged for every other template: no ban, the notice rides the tool result.
        assert "logit_bias" not in final
        assert last["role"] == "tool"
    else:
        assert final["logit_bias"] == [[10, False]]
        assert last["role"] == "user"
    assert "tool calls" in json.dumps(last["content"]).lower()
