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


def _reported(turn: _Turn):
    """The calls the turn ends up making, which is what the loop executes.

    Not every slot becomes one: a name that arrived with no arguments of its
    own opened a slot mid-stream, and it is dropped here rather than run.
    """
    return [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]


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
    payload = '{"x":' * 20000 + "null" + "}" * 20000
    assert _split_top_level_json_objects(payload) == ([], payload)

    turn = _Turn()
    turn.merge_structured([_delta(0, "deep", payload)])
    assert _shape(turn) == [("deep", payload)]


def test_a_fragment_repeating_the_slots_id_continues_that_call():
    # llama-server grows the name across deltas; forking gives two calls one id.
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
    # It belongs to the last object this delta closed, which is what Gemini
    # validates the signature against.
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
    # json.loads takes NaN and Infinity; JSON.parse does not.
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


def test_an_opening_delta_after_a_closed_call_does_not_claim_it():
    # Landing the opening delta on the finished call glues what follows.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", "", call_id = "call_b")])
    turn.merge_structured([_delta(0, None, '{"b":2}', call_id = "call_b")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    ids = [call["id"] for call in turn.calls()]
    assert ids[1] == "call_b"
    assert len(set(ids)) == len(ids)


def test_a_name_held_for_the_next_call_grows_across_deltas():
    # OpenAI streams "web" then "_search"; llama-server resends the whole.
    for fragments in (("web", "_search"), ("web", "web_search")):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        for fragment in fragments:
            turn.merge_structured([_delta(0, fragment, "")])
        turn.merge_structured([_delta(0, None, '{"q":"x"}')])

        assert _shape(turn) == [("alpha", '{"a":1}'), ("web_search", '{"q":"x"}')]


def test_whitespace_carrying_the_repeated_name_is_not_the_next_call():
    # The name is that call's, resent; parking it gave "alphabeta".
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha", " ")])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1} '), ("beta", '{"b":2}')]


def test_a_repeated_id_reaches_its_own_call_across_a_later_split():
    # The index points at the newer call, so matching a repeated id there
    # renamed it and gave it a second copy of the id.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}', call_id = "call_a")])
    turn.merge_structured([_delta(0, "beta", '{"b":')])
    turn.merge_structured([_delta(0, "alpha_long", "", call_id = "call_a")])

    assert _shape(turn) == [("alpha_long", '{"a":1}'), ("beta", '{"b":')]
    ids = [call["id"] for call in turn.calls()]
    assert ids[0] == "call_a"
    assert len(set(ids)) == len(ids)


def test_metadata_announced_with_a_name_waits_for_that_call():
    # The signature is for the call being announced, and native replay rejects
    # a call wearing another's.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured(
        [_delta(0, "beta", None) | {"extra_content": {"google": {"thought_signature": "SIG"}}}]
    )
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    entries = [turn.by_index[key] for key in turn.order]
    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    assert entries[0].get("extra_content") is None
    assert entries[1]["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_the_resumable_scan_agrees_with_scanning_from_the_start():
    # Only safe while it answers as restarting does, for every chunking.
    import random

    from core.inference.studio_tool_loop import _BoundaryScan

    pieces = list('{}"\\ abc:,1[]') + ['\\"', '"a"', "NaN", "\n", "\r\n", "\t", '{"a":1}', "}{"]
    rng = random.Random(20260827)
    for _ in range(4000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 16)))
        scan = _BoundaryScan()
        cut = 0
        result = scan.feed("")
        while cut < len(text):
            cut = min(len(text), cut + rng.randint(1, 4))
            result = scan.feed(text[:cut])
        assert result == _split_top_level_json_objects(text), text


def test_one_argument_streamed_a_character_at_a_time_stays_linear():
    # Rescanning per fragment made a 10 KB argument cost seconds. The bound is
    # generous: the quadratic term is asserted, not a machine's speed.
    import time

    payload = '{"code":"' + "x" * 20000 + '"}'
    turn = _Turn()
    turn.merge_structured([_delta(0, "write", "")])
    started = time.perf_counter()
    for ch in payload:
        turn.merge_structured([_delta(0, None, ch)])
    assert time.perf_counter() - started < 2.0
    assert _shape(turn) == [("write", payload)]


def test_metadata_arriving_alone_stays_on_the_call_that_closed():
    # No name, so nothing announces another call: the signature is this call's.
    # Parking it lost it outright when no further call followed.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([{"index": 0, "extra_content": {"google": {"thought_signature": "SIG"}}}])

    assert _shape(turn) == [("alpha", '{"a":1}')]
    entries = [turn.by_index[key] for key in turn.order]
    assert entries[0]["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_a_call_announced_by_name_only_is_still_reported():
    # A tool that takes no parameters can be announced and then simply end, and
    # _normalized_call already reads empty arguments as {}.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", "")])

    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("alpha", '{"a":1}'), ("beta", "{}")]
    ids = [call["id"] for call in turn.calls()]
    assert len(set(ids)) == len(ids)

    # And a later fragment that does open it opens it, rather than adding a
    # second call beside the one this reported.
    turn.merge_structured([_delta(0, None, '{"b":2}')])
    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]


def test_a_name_resent_or_grown_after_a_call_closed_invents_nothing():
    # Indistinguishable from a second no-argument call to the same tool, so
    # take the reading that does not run a tool twice. A grown name opens a
    # slot (the prefix is no proof) but `calls` drops it unfilled.
    for resent in ("alpha", "alpha_long"):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        turn.merge_structured([_delta(0, resent, "")])

        reported = [
            (call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()
        ]
        assert reported == [("alpha", '{"a":1}')]


def test_a_pending_call_runs_where_the_stream_announced_it():
    # The budget is spent down this list in order.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(0, "B", "")])
    turn.merge_structured([_delta(1, "C", '{"c":3}')])

    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("A", '{"a":1}'), ("B", "{}"), ("C", '{"c":3}')]


def test_a_fragment_that_does_not_open_an_object_does_not_open_a_call():
    # A next call begins with its own "{"; forking on anything else made a
    # stray scalar suffix run the tool twice.
    turn = _Turn()
    turn.merge_structured([_delta(0, "q", '{"query":"a"}')])
    turn.merge_structured([_delta(0, "q", '"b"')])

    assert _shape(turn) == [("q", '{"query":"a"}"b"')]
    assert _split_top_level_json_objects('{"query":"a"}"b"') == ([], '{"query":"a"}"b"')


def test_an_integer_too_long_to_convert_is_still_a_boundary():
    # json.loads raises past the 4300-digit int cap where JSON.parse does not,
    # and 4301 digits is 4 KB. Validation never reads them, so keep them text.
    big = '{"a":' + "1" * 4301 + "}"
    assert _split_top_level_json_objects(big + '{"b":2}') == ([big, '{"b":2}'], "")

    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", big)])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert _shape(turn) == [("alpha", big), ("beta", '{"b":2}')]


def test_two_calls_announced_at_once_do_not_break_the_sort():
    # Two announcements can share a moment. Breaking the tie by comparing the
    # calls themselves raises TypeError and takes the whole turn with it.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(1, "B", '{"b":1}')])
    turn.merge_structured([_delta(0, "C", "")])
    turn.merge_structured([_delta(1, "D", "")])

    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("A", '{"a":1}'), ("B", '{"b":1}'), ("C", "{}"), ("D", "{}")]


def test_a_pending_call_keeps_its_place_when_it_later_opens():
    # A call takes the moment it was announced at.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(0, "B", "")])
    turn.merge_structured([_delta(1, "C", '{"c":3}')])
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    # calls() is the list the loop spends its budget down.
    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("A", '{"a":1}'), ("B", '{"b":2}'), ("C", '{"c":3}')]


def test_a_resent_name_does_not_rename_the_call_it_closed():
    # A name extending the closed call's is most likely it, resent, so a delta
    # naming its own call wins: merging gave "alpha_longbeta".
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha_long", "")])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    assert _reported(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]

    # The second no-argument call to the same tool still works, and so does a
    # name streamed in fragments: neither has an opening name that disagrees.
    same = _Turn()
    same.merge_structured([_delta(0, "no_args", "{}")])
    same.merge_structured([_delta(0, "no_args", "")])
    same.merge_structured([_delta(0, None, "{}")])
    assert _reported(same) == [("no_args", "{}"), ("no_args", "{}")]

    grown = _Turn()
    grown.merge_structured([_delta(0, "alpha", '{"a":1}')])
    grown.merge_structured([_delta(0, "web", "")])
    grown.merge_structured([_delta(0, "_search", '{"q":1}')])
    assert _reported(grown) == [("alpha", '{"a":1}'), ("web_search", '{"q":1}')]


def test_metadata_on_a_resent_name_stays_with_the_closed_call():
    # Once the name is read as the closed call's, so is the metadata.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}') | {"extra_content": {"own": "A"}}])
    turn.merge_structured([_delta(0, "alpha_long", None) | {"extra_content": {"resent": 1}}])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    reported = turn.calls()
    assert _reported(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    assert reported[0]["extra_content"] == {"own": "A", "resent": 1}
    assert reported[1].get("extra_content") is None


def test_a_discarded_resend_does_not_lend_its_place_to_the_next_call():
    # The moment belongs to the announcement, and a name read as a resent
    # announced nothing, so the call that opens takes its own arrival.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(0, "A_long", None)])
    turn.merge_structured([_delta(1, "C", '{"c":3}')])
    turn.merge_structured([_delta(0, "B", '{"b":2}')])

    assert [call["function"]["name"] for call in turn.calls()] == ["A", "C", "B"]

    # An announcement that is accepted still keeps its place.
    kept = _Turn()
    kept.merge_structured([_delta(0, "A", '{"a":1}')])
    kept.merge_structured([_delta(0, "B", None)])
    kept.merge_structured([_delta(1, "C", '{"c":3}')])
    kept.merge_structured([_delta(0, None, '{"b":2}')])
    assert [call["function"]["name"] for call in kept.calls()] == ["A", "B", "C"]


def test_an_id_stamped_after_the_object_closed_claims_that_call():
    # Reading the late id as another call left the finished one minted.
    for late in ({}, {"name": "alpha"}):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        turn.merge_structured([{"index": 0, "id": "call_a", "function": late}])

        reported = [
            (call["id"], call["function"]["name"], call["function"]["arguments"])
            for call in turn.calls()
        ]
        assert reported == [("call_a", "alpha", '{"a":1}')]

    # An id that names a different call is still the next call opening.
    opened = _Turn()
    opened.merge_structured([_delta(0, "alpha", '{"a":1}')])
    opened.merge_structured([_delta(0, "beta", "", call_id = "call_b")])
    opened.merge_structured([_delta(0, None, '{"b":2}', call_id = "call_b")])
    assert _shape(opened) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]


def test_a_new_name_arriving_with_whitespace_opens_its_own_call():
    # A different name opens the next call rather than renaming the finished
    # one, which gave "alphabeta" and an unnamed second call. The whitespace
    # rides the announcing delta and is valid JSON, so both still parse.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", " ")])
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", ' {"b":2}')]

    # And a name repeated with the whitespace is still that call's, resent, so
    # the call that really opens next takes its own name.
    resent = _Turn()
    resent.merge_structured([_delta(0, "alpha", '{"a":1}')])
    resent.merge_structured([_delta(0, "alpha", " ")])
    resent.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert _shape(resent) == [("alpha", '{"a":1} '), ("beta", '{"b":2}')]


def test_metadata_on_a_resent_name_reaches_the_closed_call():
    # A name read as a resent invents no call, so its metadata has nowhere
    # else to go.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}') | {"extra_content": {"own": "A"}}])
    turn.merge_structured([_delta(0, "alpha_long", None) | {"extra_content": {"sig": "S"}}])

    reported = turn.calls()
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in reported] == [
        ("alpha", '{"a":1}')
    ]
    assert reported[0]["extra_content"] == {"own": "A", "sig": "S"}
    # Read twice, so a second read cannot double anything or lose it.
    assert turn.calls()[0]["extra_content"] == {"own": "A", "sig": "S"}


def test_a_catalog_holding_both_web_and_web_search_splits_either_way_round():
    # Both are in Studio's own catalog, so a shared prefix is no evidence
    # either way; reading it as evidence swallowed the second announcement.
    for first, second in (("web_search", "web"), ("web", "web_search")):
        turn = _Turn()
        turn.merge_structured([_delta(0, first, '{"a":1}')])
        turn.merge_structured([_delta(0, second)])
        turn.merge_structured([_delta(0, None, '{"b":2}')])

        assert _reported(turn) == [(first, '{"a":1}'), (second, '{"b":2}')]


def test_a_name_bringing_an_object_over_an_announcement_is_the_next_call():
    # An announcement has no object to close, so the next name grew into it.
    for announced in ("alpha_long", "zeta"):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        # No `arguments` field: an empty string means the call has started,
        # and a name after one is the fragment dialect.
        turn.merge_structured([_delta(0, announced, None)])
        turn.merge_structured([_delta(0, "beta", '{"b":2}')])

        assert _reported(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]


def test_an_unfinished_fork_does_not_reserve_a_card_id():
    # The client releases the card it drew for the fork this turn filters out.
    taken: set[str] = set()
    cards: set[str] = set()
    first = _Turn()
    first.merge_structured([_delta(0, "alpha", '{"a":1}{')])
    assert [call.get("card_id") for call in first.calls(taken, cards)] == ["tool_call_0"]

    second = _Turn()
    second.round = 1
    second.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert [call.get("card_id") for call in second.calls(taken, cards)] == ["tool_call_1"]


def test_a_provider_claiming_a_minted_id_displaces_the_id_less_call():
    # Provider ids are reserved before any card id is minted, so the id-less
    # call moves aside. The client displaces once the claim lands.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(1, "beta", '{"b":2}', call_id = "tool_call_0")])

    reported = [
        (call.get("card_id") or call["id"], call["function"]["name"]) for call in turn.calls()
    ]
    assert reported == [("tool_call_1", "alpha"), ("tool_call_0", "beta")]


def test_a_claim_on_a_split_born_card_leaves_every_call_its_own():
    # The client renumbers its minted cards when a provider claims one of their
    # spellings, so the numbering here is what it has to land on: the claim
    # keeps tool_call_1 and the three id-less calls take 0, 2 and 3 in order.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}{"b":2}{"c":3}')])
    turn.merge_structured([_delta(1, "beta", '{"d":4}', call_id = "tool_call_1")])

    reported = [
        (call.get("card_id") or call["id"], call["function"]["arguments"]) for call in turn.calls()
    ]
    assert reported == [
        ("tool_call_0", '{"a":1}'),
        ("tool_call_2", '{"b":2}'),
        ("tool_call_3", '{"c":3}'),
        ("tool_call_1", '{"d":4}'),
    ]


def test_a_nameless_call_reserves_no_card_id():
    # The client drops the card it drew for a call that never got a name, so
    # the next round has to land on the number this one leaves free.
    taken: set[str] = set()
    cards: set[str] = set()
    first = _Turn()
    first.merge_structured([_delta(0, None, '{"a":1}')])
    assert first.calls(taken, cards) == []

    second = _Turn()
    second.round = 1
    second.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert [call.get("card_id") for call in second.calls(taken, cards)] == ["tool_call_0"]


def test_a_card_ledger_is_append_only_across_rounds():
    # A round that carries tool_call_0 as a provider id does not take the card
    # an earlier round already drew under that spelling, so the third round
    # numbers from what both are holding.
    taken: set[str] = set()
    cards: set[str] = set()
    first = _Turn()
    first.merge_structured([_delta(0, "alpha", '{"a":1}')])
    assert [call.get("card_id") for call in first.calls(taken, cards)] == ["tool_call_0"]

    second = _Turn()
    second.round = 1
    second.merge_structured([_delta(0, "beta", '{"b":2}', call_id = "tool_call_0")])
    assert [call.get("card_id") for call in second.calls(taken, cards)] == [None]

    third = _Turn()
    third.round = 2
    third.merge_structured([_delta(0, "gamma", '{"c":3}')])
    assert [call.get("card_id") for call in third.calls(taken, cards)] == ["tool_call_1"]


def test_a_rejected_call_reserves_no_card_id():
    # A provider id on a call that never gets a name draws no card: the client
    # releases the id it had minted, so holding it here would put the two out of
    # step from the next round on.
    taken: set[str] = set()
    cards: set[str] = set()
    first = _Turn()
    first.merge_structured([_delta(0, None, '{"a":1}', call_id = "tool_call_0")])
    first.merge_structured([_delta(1, "beta", '{"b":2}')])
    assert [call.get("card_id") for call in first.calls(taken, cards)] == ["tool_call_1"]

    second = _Turn()
    second.round = 1
    second.merge_structured([_delta(0, "gamma", '{"c":3}')])
    assert [call.get("card_id") for call in second.calls(taken, cards)] == ["tool_call_0"]


def test_a_nameless_claim_does_not_take_a_valid_call_s_card():
    # The client displaces the card holding a spelling the moment a provider
    # claims it, so a claim that turns out not to be a call has to give the
    # number back. Nothing is reserved for it here, so the valid call keeps it.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(1, None, '{"b":2}', call_id = "tool_call_0")])

    reported = [(call.get("card_id"), call["function"]["name"]) for call in turn.calls()]
    assert reported == [("tool_call_0", "alpha")]


def test_a_repeated_names_metadata_waits_for_the_call_it_announced():
    # The same tool twice on one slot, the second announced by a name-only
    # delta carrying its own signature. Merging it where it landed overwrote
    # the closed call's and left the new call unsigned, and Gemini validates a
    # signature against the call it is replayed on.
    turn = _Turn()
    turn.merge_structured([_delta(0, "lookup", '{"q":"a"}') | {"extra_content": {"sig": "A"}}])
    turn.merge_structured([_delta(0, "lookup", None) | {"extra_content": {"sig": "B"}}])
    turn.merge_structured([_delta(0, None, '{"q":"b"}')])

    reported = turn.calls()
    assert _reported(turn) == [("lookup", '{"q":"a"}'), ("lookup", '{"q":"b"}')]
    assert reported[0]["extra_content"] == {"sig": "A"}
    assert reported[1]["extra_content"] == {"sig": "B"}

    # No object followed, so the repeated name really was that call's resent.
    kept = _Turn()
    kept.merge_structured([_delta(0, "lookup", '{"q":"a"}') | {"extra_content": {"own": 1}}])
    kept.merge_structured([_delta(0, "lookup", None) | {"extra_content": {"sig": "B"}}])
    assert kept.calls()[0]["extra_content"] == {"own": 1, "sig": "B"}


def test_parked_metadata_survives_a_late_id_landing_on_the_call():
    # The slot key does not change when an id lands here, so the parked
    # signature is still found. Pinned because the frontend keys the same wait
    # by card id, which a late id does rename, and the two have to agree.
    turn = _Turn()
    turn.merge_structured(
        [_delta(0, "lookup", '{"q":"a"}') | {"extra_content": {"sig": "A"}}]
    )
    turn.merge_structured([_delta(0, "lookup", None) | {"extra_content": {"sig": "B"}}])
    turn.merge_structured([_delta(0, None, "") | {"id": "call_x"}])

    reported = turn.calls()
    assert _reported(turn) == [("lookup", '{"q":"a"}')]
    assert reported[0]["extra_content"] == {"sig": "B"}


def test_a_stable_id_naming_a_longer_tool_opens_its_own_call():
    # Reading "web_search" as "web" grown gave the id to the completed call.
    turn = _Turn()
    turn.merge_structured([_delta(0, "web", '{"a":1}')])
    turn.merge_structured([_delta(0, "web_search", "", call_id = "call_b")])
    turn.merge_structured([_delta(0, None, '{"b":2}', call_id = "call_b")])

    reported = [
        (call["id"], call["function"]["name"], call["function"]["arguments"])
        for call in turn.calls()
    ]
    assert reported == [("call_0_0", "web", '{"a":1}'), ("call_b", "web_search", '{"b":2}')]


def test_an_unfinished_split_tail_is_not_reported_as_a_call():
    # Nothing marks this truncated, and running the tool on half an argument
    # is the worse failure.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}{')])

    assert [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()] == [
        ("alpha", '{"a":1}')
    ]

    # It is held, not discarded: the rest of the object still opens the call.
    turn.merge_structured([_delta(0, None, '"b":2}')])
    assert _shape(turn) == [("alpha", '{"a":1}'), ("alpha", '{"b":2}')]


def test_a_second_call_to_the_same_tool_keeps_that_tools_name():
    # The index is reused with arguments alone, and nameless calls are dropped.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, None, '{"a":2}')])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("alpha", '{"a":2}')]


def test_a_snapshot_repeated_to_carry_the_id_claims_the_call():
    # The id arrives on a verbatim repeat; a second call runs the tool twice.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha", '{"a":1}', call_id = "call_a")])

    reported = [
        (call["id"], call["function"]["name"], call["function"]["arguments"])
        for call in turn.calls()
    ]
    assert reported == [("call_a", "alpha", '{"a":1}')]


def test_a_second_call_that_differs_anywhere_still_opens_its_own():
    # The claim above is an exact repeat only: a call whose arguments differ is
    # a parallel call of its own however alike the two look.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha", '{"a":2}', call_id = "call_b")])

    reported = [(call["id"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("call_0_0", '{"a":1}'), ("call_b", '{"a":2}')]


# ------------------------------------------------------------------- card ids


def test_an_id_less_call_carries_the_card_id_the_client_painted():
    # Without the id the client minted, tool_start finds no card: eight cards
    # for the four calls in #9807.
    turn = _Turn()
    for url in ("a", "b", "c", "d"):
        turn.merge_structured([_delta(0, "fetch", '{"url":"%s"}' % url)])

    reported = [(call["id"], call.get("card_id")) for call in turn.calls()]
    assert reported == [
        ("call_0_0", "tool_call_0"),
        ("call_0_1", "tool_call_1"),
        ("call_0_2", "tool_call_2"),
        ("call_0_3", "tool_call_3"),
    ]


def test_the_conversation_id_is_not_the_card_id():
    # What is stored and replayed stays call_<round>_<position>, so a thread
    # written before this still resolves.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])

    call = turn.calls()[0]
    assert call["id"] == "call_0_0"
    assert call["card_id"] == "tool_call_0"


def test_a_call_the_provider_named_gets_no_card_id():
    # An id on the wire is already the card's id on both sides, so minting a
    # second spelling for it would be the very mismatch this exists to close.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}', call_id = "call_a")])

    call = turn.calls()[0]
    assert call["id"] == "call_a"
    assert "card_id" not in call


def test_a_second_round_keeps_numbering_where_the_first_stopped():
    # One list of cards spans the whole response, so restarting at tool_call_0
    # would address the second round's events to the first round's cards.
    first = _Turn()
    first.merge_structured([_delta(0, "alpha", '{"a":1}')])
    painted: set[str] = set()
    taken: set[str] = set()
    assert [call["card_id"] for call in first.calls(taken, painted)] == ["tool_call_0"]

    second = _Turn()
    second.round = 1
    second.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert [call["card_id"] for call in second.calls(taken, painted)] == ["tool_call_1"]


def test_a_provider_id_in_the_minted_namespace_is_not_handed_out_twice():
    # tool_call_<n> is not reserved to Unsloth.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}', call_id = "tool_call_0")])
    turn.merge_structured([_delta(1, "beta", '{"b":2}')])

    reported = [(call["id"], call.get("card_id")) for call in turn.calls()]
    assert reported == [("tool_call_0", None), ("call_0_1", "tool_call_1")]


def test_a_slot_that_opens_late_keeps_its_own_index():
    # The slot names the card before any split has happened.
    turn = _Turn()
    turn.merge_structured([_delta(3, "alpha", '{"a":1}')])

    assert [call.get("card_id") for call in turn.calls()] == ["tool_call_3"]
