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


def test_an_opening_delta_after_a_closed_call_does_not_claim_it():
    # The conventional opening delta carries the id and the name with empty
    # arguments. Landing it on the finished call would put that call's id on it,
    # so the arguments delta that follows would match and glue on, losing the
    # call the delta was announcing.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", "", call_id = "call_b")])
    turn.merge_structured([_delta(0, None, '{"b":2}', call_id = "call_b")])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    ids = [call["id"] for call in turn.calls()]
    assert ids[1] == "call_b"
    assert len(set(ids)) == len(ids)


def test_a_name_held_for_the_next_call_grows_across_deltas():
    # Both dialects the accumulator reconciles reach the held name too: OpenAI
    # streams "web" then "_search", llama-server resends "web" then
    # "web_search". Last-write-wins would open the call as "_search", which
    # matches no enabled tool and silently never runs.
    for fragments in (("web", "_search"), ("web", "web_search")):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        for fragment in fragments:
            turn.merge_structured([_delta(0, fragment, "")])
        turn.merge_structured([_delta(0, None, '{"q":"x"}')])

        assert _shape(turn) == [("alpha", '{"a":1}'), ("web_search", '{"q":"x"}')]


def test_whitespace_carrying_the_repeated_name_is_not_the_next_call():
    # A provider that repeats the name on every delta and chunks the trailing
    # whitespace separately is still writing to the call that closed, so its
    # name is that call's resent. Parking it merged the two names into
    # "alphabeta", which matches no enabled tool and silently never runs.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha", " ")])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1} '), ("beta", '{"b":2}')]


def test_a_repeated_id_reaches_its_own_call_across_a_later_split():
    # An id-less call opening after a stable-id call leaves the index pointing
    # at the newer one. Matching the repeated id against only that call renamed
    # it, gave it a second copy of the id and stranded the growth fragment.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}', call_id = "call_a")])
    turn.merge_structured([_delta(0, "beta", '{"b":')])
    turn.merge_structured([_delta(0, "alpha_long", "", call_id = "call_a")])

    assert _shape(turn) == [("alpha_long", '{"a":1}'), ("beta", '{"b":')]
    ids = [call["id"] for call in turn.calls()]
    assert ids[0] == "call_a"
    assert len(set(ids)) == len(ids)


def test_metadata_announced_with_a_name_waits_for_that_call():
    # Gemini stows the thoughtSignature for the call being announced, so a
    # name-only delta carrying one describes the next call, not the closed one.
    # Left behind it gives the closed call another call's signature and the new
    # call none, and native replay rejects both.
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
    # The accumulator resumes the boundary scan instead of restarting it, which
    # is only safe while the two give the same answer for every string and every
    # set of chunk boundaries.
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
    # Rescanning the whole accumulation per fragment made a 10 KB argument cost
    # seconds, which stalls the response for any tool that takes code or file
    # content. Generous bound: this asserts the quadratic term is gone, not a
    # particular machine's speed.
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
    # Indistinguishable from a second no-argument call to the same tool, so the
    # conservative reading is the one that does not run a tool twice.
    for resent in ("alpha", "alpha_long"):
        turn = _Turn()
        turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
        turn.merge_structured([_delta(0, resent, "")])

        assert _shape(turn) == [("alpha", '{"a":1}')]


def test_a_pending_call_runs_where_the_stream_announced_it():
    # The loop spends its budget down this list in order, so a call announced
    # second must not run third: a finite budget would run the later call and
    # reject the earlier one.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(0, "B", "")])
    turn.merge_structured([_delta(1, "C", '{"c":3}')])

    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("A", '{"a":1}'), ("B", "{}"), ("C", '{"c":3}')]


def test_a_fragment_that_does_not_open_an_object_does_not_open_a_call():
    # A next call begins with the "{" of its own arguments object. Forking on
    # any non-whitespace text cut where the scanner deliberately leaves the text
    # whole, so a stray scalar suffix ran the tool a second time.
    turn = _Turn()
    turn.merge_structured([_delta(0, "q", '{"query":"a"}')])
    turn.merge_structured([_delta(0, "q", '"b"')])

    assert _shape(turn) == [("q", '{"query":"a"}"b"')]
    assert _split_top_level_json_objects('{"query":"a"}"b"') == ([], '{"query":"a"}"b"')


def test_an_integer_too_long_to_convert_is_still_a_boundary():
    # json.loads raises ValueError past the 4300-digit cap on int conversion,
    # where JSON.parse does not, and a 4301-digit literal is about 4 KB and fits
    # in an ordinary payload. Validation never reads the numbers, so keeping
    # them as text keeps the two scanners cutting in the same place.
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
    # The call takes the moment it was announced at, not the moment its
    # arguments arrived, so a finite budget cannot run the later call and
    # reject the earlier one.
    turn = _Turn()
    turn.merge_structured([_delta(0, "A", '{"a":1}')])
    turn.merge_structured([_delta(0, "B", "")])
    turn.merge_structured([_delta(1, "C", '{"c":3}')])
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    # calls() is the list the loop spends its budget down.
    reported = [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()]
    assert reported == [("A", '{"a":1}'), ("B", '{"b":2}'), ("C", '{"c":3}')]


def test_an_opening_name_beats_a_parked_resend():
    # A parked name that extends the closed call's own is most likely that
    # call's name resent, so a delta that names its call outright wins: seeding
    # "alpha_long" and merging "beta" onto it gave "alpha_longbeta", which
    # matches no enabled tool and never runs.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "alpha_long", "")])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]

    # The second no-argument call to the same tool still works, and so does a
    # name streamed in fragments: neither has an opening name that disagrees.
    same = _Turn()
    same.merge_structured([_delta(0, "no_args", "{}")])
    same.merge_structured([_delta(0, "no_args", "")])
    same.merge_structured([_delta(0, None, "{}")])
    assert _shape(same) == [("no_args", "{}"), ("no_args", "{}")]

    grown = _Turn()
    grown.merge_structured([_delta(0, "alpha", '{"a":1}')])
    grown.merge_structured([_delta(0, "web", "")])
    grown.merge_structured([_delta(0, "_search", '{"q":1}')])
    assert _shape(grown) == [("alpha", '{"a":1}'), ("web_search", '{"q":1}')]


def test_metadata_parked_with_a_resent_name_stays_with_the_closed_call():
    # The metadata was announced alongside the name, so once that name is read
    # as the closed call's resent, the metadata is the closed call's too.
    # Handing it to the new call leaves one wearing another's signature.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}') | {"extra_content": {"own": "A"}}])
    turn.merge_structured([_delta(0, "alpha_long", None) | {"extra_content": {"resent": 1}}])
    turn.merge_structured([_delta(0, "beta", '{"b":2}')])

    entries = [turn.by_index[key] for key in turn.order]
    assert _shape(turn) == [("alpha", '{"a":1}'), ("beta", '{"b":2}')]
    assert entries[0]["extra_content"] == {"own": "A", "resent": 1}
    assert entries[1].get("extra_content") is None


def test_a_discarded_resend_does_not_lend_its_place_to_the_next_call():
    # The moment belongs to the announcement. Once the parked name is read as
    # the closed call's resent, the call that opens here was not announced
    # then, so keeping that moment would run it ahead of a call the stream
    # really did open first, and a finite budget would reject the earlier one.
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
    # A provider that opens id-less and stamps the real id on a later delta.
    # Reading the id as proof of another call left the finished call under a
    # minted id, and when the delta repeated the name it ran a second empty
    # invocation of the tool.
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


def test_a_new_name_arriving_with_whitespace_is_still_held():
    # The whitespace belongs to the object that just closed, but the name on
    # that delta may be the next call announced early. Merging it left the
    # closed call named "alphabeta" and the new call unnamed, so neither ran.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}')])
    turn.merge_structured([_delta(0, "beta", " ")])
    turn.merge_structured([_delta(0, None, '{"b":2}')])

    assert _shape(turn) == [("alpha", '{"a":1} '), ("beta", '{"b":2}')]

    # And a name repeated with the whitespace is still that call's, resent, so
    # the call that really opens next takes its own name.
    resent = _Turn()
    resent.merge_structured([_delta(0, "alpha", '{"a":1}')])
    resent.merge_structured([_delta(0, "alpha", " ")])
    resent.merge_structured([_delta(0, "beta", '{"b":2}')])
    assert _shape(resent) == [("alpha", '{"a":1} '), ("beta", '{"b":2}')]


def test_metadata_on_a_resent_name_reaches_the_closed_call():
    # No call is invented for a name read as the closed call's resent, so
    # metadata announced on that delta has nowhere else to go. Dropping it cost
    # the call its thought signature, and the provider rejects the replay.
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


def test_a_stable_id_naming_a_longer_tool_opens_its_own_call():
    # A catalog can hold both "web" and "web_search". Reading the second name as
    # a growth of the first gave the id to the completed call, and the arguments
    # that followed glued onto it, losing the second intent entirely.
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
    # Only "length" and "content_filter" mark a turn truncated, so a stream that
    # stops after the start of a second object looks complete. Running the tool
    # again on that lone brace is worse than dropping a call the model never
    # finished writing.
    turn = _Turn()
    turn.merge_structured([_delta(0, "alpha", '{"a":1}{')])

    assert [(call["function"]["name"], call["function"]["arguments"]) for call in turn.calls()] == [
        ("alpha", '{"a":1}')
    ]

    # It is held, not discarded: the rest of the object still opens the call.
    turn.merge_structured([_delta(0, None, '"b":2}')])
    assert _shape(turn) == [("alpha", '{"a":1}'), ("alpha", '{"b":2}')]
