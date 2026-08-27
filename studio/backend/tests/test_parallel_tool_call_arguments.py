# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel tool calls that share one delta index (issue #9807).

A server that streams parallel calls as id-less, index-based deltas reuses one
slot for several calls. Appending their arguments glues them into one
unparseable string, which then rides into the next request verbatim and the
provider answers 400. The fork on a differing call id cannot catch this one:
there are no ids to differ.

The boundary between two calls is the end of a top-level JSON object, not a
change of function name -- the reported stream calls the same tool three times.
"""

from __future__ import annotations

import json

import pytest

from core.inference.studio_tool_loop import (
    _Turn,
    _split_top_level_json_objects,
)


# --------------------------------------------------------------------- scanner


@pytest.mark.parametrize(
    "text,complete,tail",
    [
        ("", [], ""),
        ("{}", ["{}"], ""),
        ('{"a":1}', ['{"a":1}'], ""),
        ('{"a":1}{"b":2}', ['{"a":1}', '{"b":2}'], ""),
        (
            '{"a":1} {"b":2}\n{"c":3}\r\n{"d":4}',
            ['{"a":1}', '{"b":2}', '{"c":3}', '{"d":4}'],
            "",
        ),
        ('{"a":', [], '{"a":'),
        ('{"a":1}{"b":', ['{"a":1}'], '{"b":'),
        # Braces that are data, not structure.
        ('{"a":"}{"}{"b":2}', ['{"a":"}{"}', '{"b":2}'], ""),
        ('{"a":"say \\"}{\\" ok"}{"b":2}', ['{"a":"say \\"}{\\" ok"}', '{"b":2}'], ""),
        ('{"p":"C:\\\\Users\\\\me"}{"b":2}', ['{"p":"C:\\\\Users\\\\me"}', '{"b":2}'], ""),
        ('{"a":{"b":{"c":1}}}', ['{"a":{"b":{"c":1}}}'], ""),
        ('{"a":[{"b":1},{"c":2}]}', ['{"a":[{"b":1},{"c":2}]}'], ""),
        (
            '{"q":"caf\u00e9 \u65e5\u672c\u8a9e \U0001f680"}',
            ['{"q":"caf\u00e9 \u65e5\u672c\u8a9e \U0001f680"}'],
            "",
        ),
        # Not a run of objects: left alone rather than cut somewhere meaningless.
        ('[{"a":1}]', [], '[{"a":1}]'),
        ('"hello"', [], '"hello"'),
        ("42", [], "42"),
        ("null", [], "null"),
        ('{"a":1}junk{"b":2}', [], '{"a":1}junk{"b":2}'),
        ('{"a":1}}', [], '{"a":1}}'),
        ('{"a":1,}{"b":2}', [], '{"a":1,}{"b":2}'),
        ('{"a":"unterminated', [], '{"a":"unterminated'),
    ],
)
def test_split_top_level_json_objects(text, complete, tail):
    assert _split_top_level_json_objects(text) == (complete, tail)


# ----------------------------------------------------------------- accumulator


def _delta(
    index,
    name = None,
    arguments = "",
    call_id = None,
):
    function: dict = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    call: dict = {"index": index, "function": function}
    if call_id is not None:
        call["id"] = call_id
    return call


def _shape(turn: _Turn):
    return [
        (turn.by_index[key]["function"]["name"], turn.by_index[key]["function"]["arguments"])
        for key in turn.order
    ]


def test_the_stream_from_9807_becomes_one_call_per_object():
    # Three fetches and a search, all id-less at index 0. The same tool repeats,
    # so there is no name change to fork on.
    turn = _Turn()
    turn.merge_structured([_delta(0, "url", '{"url":"a"}')])
    turn.merge_structured([_delta(0, "url", '{"url":"b"}')])
    turn.merge_structured([_delta(0, "query", '{"q":"c"}')])
    turn.merge_structured([_delta(0, "url", '{"url":"d"}')])

    assert _shape(turn) == [
        ("url", '{"url":"a"}'),
        ("url", '{"url":"b"}'),
        ("query", '{"q":"c"}'),
        ("url", '{"url":"d"}'),
    ]
    for _name, arguments in _shape(turn):
        assert isinstance(json.loads(arguments), dict)


def test_every_forked_call_gets_an_id_of_its_own():
    turn = _Turn()
    turn.merge_structured([_delta(0, "url", '{"url":"a"}{"url":"b"}')])
    ids = [call["id"] for call in turn.calls()]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert all(ids)


def test_a_call_at_another_index_survives_a_fork():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(1, "beta", '{"b":2}')])
    turn.merge_structured([_delta(0, "gamma", '{"c":3}')])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}'), ("gamma", '{"c":3}')]


def test_an_ordinary_fragmented_call_is_still_one_call():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":')])
    turn.merge_structured([_delta(0, None, "1")])
    turn.merge_structured([_delta(0, None, "}")])

    assert _shape(turn) == [("alpha", '{"a":1}')]


def test_a_fragment_continues_the_call_still_being_written():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}{"b":')])
    turn.merge_structured([_delta(0, None, "2}")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("alpha", '{"b":2}')]


def test_a_stream_that_carries_ids_forks_on_the_id_as_before():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":', call_id = "call_a")])
    turn.merge_structured([_delta(1, "beta", '{"b":', call_id = "call_b")])
    turn.merge_structured([_delta(0, None, "1}", call_id = "call_a")])
    turn.merge_structured([_delta(1, None, "2}", call_id = "call_b")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    assert [call["id"] for call in turn.calls()] == ["call_a", "call_b"]


def test_a_name_arriving_in_fragments_is_unaffected():
    # llama-server resends the whole name as it grows; OpenAI streams it in
    # pieces. Both still land on one call.
    turn = _Turn()
    turn.merge_structured([_delta(0, "web", '{"q":')])
    turn.merge_structured([_delta(0, "web_search", '"x"}')])

    assert _shape(turn) == [("web_search", '{"q":"x"}')]


def test_a_name_only_delta_does_not_rename_the_finished_call():
    # The name arrives before its arguments, so the accumulated text is still
    # one whole object and there is nothing to fork on yet. Merging it into the
    # finished call gives "alphabeta", which matches no enabled tool.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", "")])
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]


def test_an_id_after_one_closed_object_opens_a_call_not_a_claim():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", '{"b":2}', call_id = "call_b")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    assert [call["id"] for call in turn.calls()] == ["call_0_0", "call_b"]


def test_an_id_after_a_closed_fork_opens_a_third_call():
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}{"b":2}')])
    turn.merge_structured([_delta(0, "gamma", '{"c":3}', call_id = "call_c")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("alpha", '{"b":2}'), ("gamma", '{"c":3}')]
    ids = [call["id"] for call in turn.calls()]
    assert len(set(ids)) == len(ids)


def test_nesting_deep_enough_to_exhaust_the_stack_is_unsplittable():
    # json.loads raises RecursionError, which is a RuntimeError and not a
    # ValueError. Uncaught it escapes the streaming loop mid-response, so the
    # segment counts as unsplittable and _normalized_call downgrades it as
    # it always has.
    payload = '{"x":' * 20000 + "null" + "}" * 20000
    assert _split_top_level_json_objects(payload) == ([], payload)

    turn = _Turn()
    turn.merge_structured([_delta(0, "deep", payload)])
    assert _shape(turn) == [("deep", payload)]


def test_a_fragment_repeating_the_slots_id_continues_that_call():
    # llama-server grows the name across deltas. An id names its call, so
    # forking here would give two calls one id, with the arguments stranded on
    # the abandoned name and the grown one running empty.
    turn = _Turn()
    turn.merge_structured([_delta(0, "web", '{"q":"x"}', call_id = "call_a")])
    turn.merge_structured([_delta(0, "web_search", "", call_id = "call_a")])

    assert _shape(turn) == [("web_search", '{"q":"x"}')]
    assert [call["id"] for call in turn.calls()] == ["call_a"]


def test_whitespace_chunked_after_a_closing_brace_is_not_a_new_call():
    # Trailing whitespace is legal JSON and says nothing about another call.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha", " ")])

    assert _shape(turn) == [("alpha", '{"a":1} ')]

    # ... and the call that really does follow it still forks.
    turn.merge_structured([_delta(0, "alpha", '{"b":2}')])
    assert _shape(turn) == [("alpha", '{"a":1} '), ("alpha", '{"b":2}')]


def test_the_metadata_a_delta_carries_goes_to_the_call_it_closes():
    # Gemini checks the opaque signature against the exact call it is replayed
    # on, so it belongs to the last object this delta closed, not to the one
    # the slot already held. Same placement as the frontend split.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":')])
    turn.merge_structured(
        [_delta(0, None, '1}{"b":2}') | {"extra_content": {"google": {"thought_signature": "SIG"}}}]
    )

    entries = [turn.by_index[key] for key in turn.order]
    assert [e["function"]["arguments"] for e in entries] == ['{"a":1}', '{"b":2}']
    assert entries[0].get("extra_content") is None
    assert entries[1]["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_nonstandard_numeric_constants_are_not_object_boundaries():
    # json.loads takes NaN and Infinity; JSON.parse does not. Accepting them
    # here would cut text the frontend leaves whole, so the backend would run
    # two calls where the UI shows one.
    assert _split_top_level_json_objects('{"a":NaN}{"b":2}') == ([], '{"a":NaN}{"b":2}')
    assert _split_top_level_json_objects('{"a":Infinity}{"b":2}') == (
        [],
        '{"a":Infinity}{"b":2}',
    )
    # Ordinary numbers, exponents included, still split.
    assert _split_top_level_json_objects('{"a":1.5e3}{"b":2}') == (
        ['{"a":1.5e3}', '{"b":2}'],
        "",
    )
