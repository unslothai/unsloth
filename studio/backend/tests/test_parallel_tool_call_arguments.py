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
