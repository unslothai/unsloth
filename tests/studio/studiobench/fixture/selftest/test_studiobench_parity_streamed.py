# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A message that is still being written has no defined moment, so it is refused rather than scored.

THE DEFECT. studiobench drives both arms through the same scripted actions, and the parity digest
is taken at the CLOSE of each action window -- a wall-clock offset in the film. The two arms are
two cells run back to back against one pacer: the bytes on the wire are identical by construction,
but each arm has its own send click, its own `t0` and its own paint clock. So when a slot lands
inside a live reply the two digests are taken at two different points in the same stream, and the
difference that comes back is wall clock wearing the shape of a UI change. It is the same mistake
as every entry in the instrument-defect list this file is a response to: MEASURING AT A MOMENT
WHOSE MEANING IS NOT STABLE ACROSS THE THINGS BEING COMPARED.

WHY YOU CANNOT RECOGNISE IT BY ITS SIZE, which is the part that has misled people. Mid-stream,
Studio does not show a prefix of the finished reply. `parseIncompleteMarkdown` runs remend over the
tail and closes whatever construct is half-arrived, KaTeX renders the repaired formula and, while
it will not parse, writes the parse error and its character offset into a `title`, Shiki
re-tokenises the repaired fence, and the trailing code block carries `data-incomplete`. None of
that is monotonic in how much text has arrived.

MEASURED, out of tree, on the shipped frozen corpus (the streamed unit, 4,238 characters) driven
through the real remend, the real KaTeX and the real Shiki into the shipped `signature()`:

  stepping by the pacer's own 24-character chunk   175 of 175 adjacent pairs differ
                                                   0 of them at the same serialised length
  stepping one character at a time                 52 of 4,237 steps make the signature SHORTER
                                                   34 pairs of distinct stream positions serialise
                                                   to the same length with different digests
                                                   398 of 4,237 steps move the digest not at all

So at the shipped cadence the drift is total -- one chunk of skew fails a stable action outright --
and the same-length variant, while real, needs sub-chunk skew to reach. Both are the same defect
and the fix is deliberately blind to length: what is refused is the message that was in flight, not
a difference of a particular size.

WHAT THE FIX MUST NOT DO. A normaliser that quietened this by widening what it erases would be
worse than the bug: it would pass a null control perfectly and detect nothing. So this file scores
both directions on every change, as a MUTANT SCORE and a NULL SCORE rather than as "the tests pass".

  NULL   pairs that are the same build at two points in one stream. Every one must be refused.
  MUTANT real, visible rendering differences, injected WHILE A REPLY IS IN FLIGHT. Every one must
         still be caught, because a change that only shows up during a stream is still a change.

The before/after comparison is free and needs no second checkout: a capture WITHOUT the streaming
fields is exactly what the previous instrument produced, and `compare` falls back to the plain
digest for it. `test_the_null_battery_scores_the_old_instrument_too` runs the same battery both
ways and pins both numbers.

THE SCORES THIS FILE HOLDS, and the same two scores taken out of tree against a real jsdom document
built by the real renderers and digested by the shipped `capture()` rather than by these fixtures:

  in tree     NULL 0 of 6 reported as a difference (all 6 refused); MUTANT 13 of 13 detected, and
              0 of 13 reported as a MATCH.
  live DOM    NULL 15 of 15 -> 0 of 15 reported as a difference; MUTANT 10 of 11 still detected,
              11 of 11 never a MATCH. The one demotion is the reorder pinned below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.analysis import parity as P  # noqa: E402

from tests.studio.studiobench.fixture.selftest.test_studiobench_parity_digest import (  # noqa: E402
    run_js,
)

# ── building a capture out of the SHIPPED signature ──────────────────
#
# Not a hand-written digest. The whole point of the node harness is that the thing under test is
# `scene/parity.js` as it ships, so these fixtures are DOM trees and every digest below is produced
# by walking them with the real `signature()`.


def message(
    index: int,
    *,
    role: str = "assistant",
    body: str = "settled text",
    streaming: bool = False,
    extra: list | None = None,
    attrs: dict | None = None,
) -> dict:
    """One thread message, in the shape Studio renders.

    `streaming` puts `data-status="running"` on the text part, which is assistant-ui's own
    published state and what `scene/dom.js::streamingMessages` reads. A settled part reads
    `"complete"`, so the attribute is present either way and its VALUE is what moves.
    """
    children: list = [
        {
            "tag": "div",
            "attrs": {"data-status": "running" if streaming else "complete"},
            "children": [{"tag": "p", "attrs": {"class": "aui-md-p"}, "children": [body]}],
        }
    ]
    children.extend(extra or [])
    return {
        "tag": "div",
        "attrs": dict({"data-role": role, "class": "aui-message-root"}, **(attrs or {})),
        "children": children,
        "_i": index,
        "_streaming": streaming,
    }


def thread(messages: list[dict], overlays: list[dict] | None = None) -> dict:
    return {
        "tag": "div",
        "attrs": {"class": "aui-thread-root"},
        "children": [
            {"tag": "div", "attrs": {"class": "aui-thread-viewport"}, "children": list(messages)}
        ],
        "_overlays": list(overlays or []),
    }


def _mark_elided(node: dict) -> dict:
    """The same tree with every message marked for elision, which is what the scaffold digest is.

    Recursive and keyed on `data-role`, because that is what `capture()` elides: `dom.messages()`
    returns every `[data-role]` in the document, wherever it sits. A helper that only reached the
    first branch would build a scaffold that quietly agreed about the branches it never walked.
    """
    if not isinstance(node, dict):
        return node
    if (node.get("attrs") or {}).get("data-role"):
        return dict(node, elide = True)
    return dict(node, children = [_mark_elided(c) for c in node.get("children") or []])


def capture(tree: dict, *, streaming_fields: bool = True) -> dict:
    """A parity capture for one arm, every digest taken by the shipped `signature()`.

    `streaming_fields=False` produces exactly what the instrument produced before the streamed
    message was named: digests only, no `in_flight`, no settled digest. That is the before-picture
    and it is not a mock of one.
    """
    messages = tree["children"][0]["children"]
    overlays = tree.get("_overlays") or []
    # `elide` marks EVERY message, which is what `capture()` in parity.js does: the scaffold has to
    # be the same walk on both arms, and which message is in flight is not.
    scaffold_tree = _mark_elided(tree)
    got = run_js(
        {
            "trees": [tree] + messages + [o["tree"] for o in overlays],
            "elided": [scaffold_tree],
        }
    )
    sigs = got["signatures"]
    whole, msg_sigs = sigs[0], sigs[1 : 1 + len(messages)]
    overlay_sigs = sigs[1 + len(messages) :]
    settled = got["elided"][0]
    hashes = run_js({"hashes": [whole, settled] + msg_sigs + overlay_sigs})["hashes"]
    out: dict = {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": hashes[0],
        "chars": len(whole),
        "messages": [
            {"i": m["_i"], "role": m["attrs"]["data-role"], "digest": h, "chars": len(s)}
            for m, h, s in zip(messages, hashes[2 : 2 + len(messages)], msg_sigs)
        ],
        "overlays": [
            {"sel": o["sel"], "digest": h, "chars": len(s)}
            for o, h, s in zip(overlays, hashes[2 + len(messages) :], overlay_sigs)
        ],
        "styles": {"digest": "s0", "chars": 5, "elements": 4, "capped": False},
        "mounted_messages": len(messages),
        "thread_total": len(messages),
    }
    if streaming_fields:
        in_flight = [m["_i"] for m in messages if m["_streaming"]]
        for row, m in zip(out["messages"], messages):
            if m["_streaming"]:
                row["in_flight"] = True
        out.update(
            digest_scaffold = hashes[1],
            chars_scaffold = len(settled),
            in_flight = in_flight,
            streaming = bool(in_flight),
            in_flight_unplaced = False,
        )
    return out


# The body of the reply, at four points in one stream. Not four different documents: the same one,
# with the trailing construct repaired differently at each point, which is what the renderer
# actually does. The `title` is the KaTeX parse error rehype-katex writes while the formula will
# not parse, and its character offset moves with every arriving character.
def streamed_body(chars: int) -> list:
    text = "The bounded shard coalesces the retained layout, except that the fibre stays inter"
    return [
        {"tag": "p", "children": [text[:chars]]},
        {
            "tag": "span",
            "attrs": {
                "class": "katex-error",
                "title": f"ParseError: KaTeX parse error: Expected 'EOF' at position {chars}",
            },
            "children": [f"$$\\lambda_{{a{chars:04d}}}$$"],
        },
    ]


def streaming_arm(
    chars: int,
    *,
    settled_body: str = "settled text",
    streaming_fields: bool = True,
    overlays: list | None = None,
) -> dict:
    return capture(
        thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(1, body = settled_body),
                message(2, streaming = True, body = "reply so far", extra = streamed_body(chars)),
            ],
            overlays,
        ),
        streaming_fields = streaming_fields,
    )


#: Four points in one stream. Adjacent pairs are what two arms one paint apart look like.
STREAM_POINTS = (12, 24, 48, 81)


# ── 1. the reproduction, both cases ──────────────────────────────────


def test_the_same_document_at_two_points_in_one_stream_moves_the_raw_digest():
    """CASE ONE: the drift is real, and it is not the comparison layer inventing it.

    Two arms, one build, one pacer, the same logical document -- and the digests disagree, because
    the digest was taken at two different points in the same reply. This is the reading that has
    been read as "this pull request changed the UI".
    """
    a, b = streaming_arm(24), streaming_arm(48)
    assert a["digest"] != b["digest"], "no drift to fix; the fixture is not reproducing the defect"
    # ...and the settled thread is byte-identical, which is what makes it a false alarm rather
    # than a finding.
    assert a["digest_scaffold"] == b["digest_scaffold"]


def test_the_drift_is_refused_rather_than_reported_as_a_ui_change():
    got = P.compare(streaming_arm(24), streaming_arm(48))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []
    assert got["not_digested"] == [2]
    assert "STILL BEING WRITTEN" in got["reason"]


def test_two_genuinely_different_documents_still_differ_while_a_reply_streams():
    """CASE TWO, and it is the one a bad fix breaks.

    Same point in the stream on both arms, but a SETTLED message renders different text. That is a
    real difference and it has to survive the streamed message being excused.
    """
    got = P.compare(
        streaming_arm(24, settled_body = "settled text"),
        streaming_arm(24, settled_body = "settled text, rewritten"),
    )
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == ["msg1(assistant):120->131c"], got["moved"]


def test_a_real_difference_survives_the_streamed_message_drifting_at_the_same_time():
    # The realistic case: the arms are at different points in the stream AND something real
    # changed. Excusing the first must not excuse the second, and the streamed message must not
    # appear in `moved` and drown the finding.
    got = P.compare(
        streaming_arm(24, settled_body = "settled text"),
        streaming_arm(81, settled_body = "settled text, rewritten"),
    )
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == ["msg1(assistant):120->131c"], got["moved"]
    assert got["in_flight"] == [2]


# ── 2. the null score ────────────────────────────────────────────────


def null_battery(*, streaming_fields: bool) -> list[dict]:
    """Every ordered pair of stream points: one build, one document, two moments."""
    arms = {n: streaming_arm(n, streaming_fields = streaming_fields) for n in STREAM_POINTS}
    return [
        P.compare(arms[a], arms[b])
        for i, a in enumerate(STREAM_POINTS)
        for b in STREAM_POINTS[i + 1 :]
    ]


def test_the_null_score_is_zero():
    results = null_battery(streaming_fields = True)
    differing = [r for r in results if r["verdict"] == P.DIFFER]
    assert len(results) == 6
    assert not differing, f"NULL SCORE {len(differing)}/{len(results)}: {differing}"
    # And they are refused, not passed. A null that reports MATCH on a pair whose digests plainly
    # disagree would be the instrument certifying a surface it could not look at.
    assert all(r["verdict"] == P.NOT_COMPARABLE for r in results)


def test_the_null_battery_scores_the_old_instrument_too():
    """The before-picture, on the same fixtures, in the same run.

    A capture without the streaming fields is exactly what the previous instrument recorded, and
    `compare` falls back to the plain digest for it. So this is not a claim about what used to
    happen; it is the old behaviour, measured.
    """
    before = null_battery(streaming_fields = False)
    assert all(r["verdict"] == P.DIFFER for r in before), before
    # Every one of them localised to the streamed message and to nothing else, which is what made
    # them read as a UI change rather than as a clock.
    for r in before:
        assert [m for m in r["moved"] if m.startswith("msg2(")] == r["moved"], r["moved"]


def test_two_arms_that_landed_on_the_same_point_in_the_stream_still_match():
    # The coverage this must NOT cost. Two arms at the same point serialise identically, which is
    # exactly the claim this mode makes, so it stays a pass.
    got = P.compare(streaming_arm(24), streaming_arm(24))
    assert got["verdict"] == P.MATCH, got
    assert got["in_flight"] == [2]


def test_a_settled_thread_is_scored_exactly_as_it_was():
    settled = capture(thread([message(0, role = "user", body = "hi"), message(1)]))
    assert settled["in_flight"] == [] and settled["streaming"] is False
    assert P.compare(settled, settled)["verdict"] == P.MATCH
    # The scaffold is a SHORTER walk than the whole thread, not an equal one, and it has to be:
    # every message is elided from it whether or not anything was streaming. What makes the reading
    # complete again is the per-message rows, and the mutant battery is what shows they do.
    assert settled["chars_scaffold"] < settled["chars"]


# ── 3. the mutant score ──────────────────────────────────────────────
#
# Every mutant is a real, visible rendering difference, injected WHILE A REPLY IS IN FLIGHT and at
# the SAME point in the stream on both arms, so the only thing the comparison can be reacting to is
# the mutation. A normaliser that reports "identical" without ever having been shown a difference
# is worthless, and one that stopped seeing these because a stream happened to be running would be
# worse than the drift it was written to fix.


def mutants() -> list[tuple[str, dict, dict]]:
    at = 24

    def base_tree(**kw) -> dict:
        return thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(1, **kw),
                message(2, streaming = True, body = "reply so far", extra = streamed_body(at)),
            ]
        )

    out: list[tuple[str, dict, dict]] = []
    base = capture(base_tree())

    def add(name: str, tree: dict) -> None:
        out.append((name, base, capture(tree)))

    add(
        "a settled message gains an element",
        base_tree(extra = [{"tag": "span", "children": ["new"]}]),
    )
    add("a settled message's text changes", base_tree(body = "settled text, rewritten"))
    add(
        "a reasoning pane silently collapses",
        base_tree(
            extra = [{"tag": "div", "attrs": {"data-slot": "reasoning-root", "data-state": "closed"}}]
        ),
    )
    add("a class list changes", base_tree(attrs = {"class": "aui-message-root flex-col"}))
    add(
        "a control becomes disabled",
        base_tree(extra = [{"tag": "button", "attrs": {"disabled": ""}}]),
    )
    add("a message changes role", base_tree(role = "user"))

    # Structural mutants that are not expressible as one message's attributes.
    add(
        "a settled message disappears",
        thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(2, streaming = True, body = "reply so far", extra = streamed_body(at)),
            ]
        ),
    )
    add(
        "the STREAMING message disappears",
        thread([message(0, role = "user", body = "the prompt"), message(1)]),
    )
    add(
        "the thread gains scaffolding outside every message",
        {
            "tag": "div",
            "attrs": {"class": "aui-thread-root"},
            "children": [
                {
                    "tag": "div",
                    "attrs": {"class": "aui-thread-viewport"},
                    "children": base_tree()["children"][0]["children"],
                },
                {"tag": "div", "attrs": {"class": "aui-empty-state"}},
            ],
        },
    )
    add(
        "siblings are reordered inside a settled message",
        base_tree(extra = [{"tag": "b", "children": ["x"]}, {"tag": "i", "children": ["y"]}]),
    )
    out.append(
        (
            "siblings reordered the other way",
            capture(
                base_tree(extra = [{"tag": "b", "children": ["x"]}, {"tag": "i", "children": ["y"]}])
            ),
            capture(
                base_tree(extra = [{"tag": "i", "children": ["y"]}, {"tag": "b", "children": ["x"]}])
            ),
        )
    )
    # An overlay lives outside the thread root, so it has its own walk and its own way of going
    # unnoticed. Both shapes, because a menu that mounts and a menu that is rewritten are
    # different regressions.
    menu = {
        "sel": '[role="menu"]',
        "tree": {
            "tag": "div",
            "attrs": {"role": "menu"},
            "children": [{"tag": "span", "children": ["Copy"]}],
        },
    }
    other = {
        "sel": '[role="menu"]',
        "tree": {
            "tag": "div",
            "attrs": {"role": "menu"},
            "children": [{"tag": "span", "children": ["Delete"]}],
        },
    }
    at_tree = base_tree()
    out.append(
        (
            "an overlay mounts on one arm only",
            capture(at_tree),
            capture(dict(at_tree, _overlays = [menu])),
        )
    )
    out.append(
        (
            "an overlay's contents are rewritten",
            capture(dict(at_tree, _overlays = [menu])),
            capture(dict(at_tree, _overlays = [other])),
        )
    )
    return out


def test_the_mutant_score_is_total():
    caught, missed = [], []
    for name, before, after in mutants():
        got = P.mutation_detected(before, after)
        (caught if got["detected"] else missed).append(name)
    assert not missed, f"MUTANT SCORE {len(caught)}/{len(caught) + len(missed)}, missed: {missed}"
    assert len(caught) == 13, f"the battery shrank to {len(caught)}; mutants were removed"


def test_no_mutant_is_ever_reported_as_a_match():
    # The weaker bar, held separately, because it is the one that decides whether a change can ship
    # green. DIFFER is the outcome that names the change; NOT COMPARABLE is the outcome that
    # refuses to say; only MATCH lets it through, and nothing here may reach it.
    for name, before, after in mutants():
        assert P.compare(before, after)["verdict"] != P.MATCH, name


def test_the_mutant_score_is_unchanged_by_the_streaming_fields():
    # The same battery scored as the OLD instrument would score it. Identical, which is the claim:
    # nothing that used to be caught stopped being caught. If this ever diverges from the test
    # above, the elision has started hiding something.
    for name, before, after in mutants():
        old_before = {
            k: v
            for k, v in before.items()
            if k
            not in (
                "digest_scaffold",
                "chars_scaffold",
                "in_flight",
                "streaming",
                "in_flight_unplaced",
            )
        }
        old_after = {
            k: v
            for k, v in after.items()
            if k
            not in (
                "digest_scaffold",
                "chars_scaffold",
                "in_flight",
                "streaming",
                "in_flight_unplaced",
            )
        }
        assert P.mutation_detected(old_before, old_after)["detected"], name


# ── 4. what the fix gives up, pinned rather than left implicit ───────


def test_reordering_the_streamed_message_past_a_sibling_is_refused_not_passed():
    """THE SECOND COST, and the one that had to be found by a mutant rather than by reading.

    The per-message rows are keyed by mounted index. Swap the streamed message with the settled one
    and the streaming row sits at index 2 on one arm and index 1 on the other, so BOTH indices are
    in flight on one side or the other and both are withheld -- and the scaffold markers that
    survive carry the same `assistant` role in either order. So a reorder of two same-role messages
    involving the streamed one is demoted from a difference to a refusal.

    Demoted, NOT hidden: NOT COMPARABLE is not a pass and nothing goes green on it. Pinned here so
    the demotion is a known cost rather than something a later reader discovers.
    """
    a = streaming_arm(24)
    b = capture(
        thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(1, streaming = True, body = "reply so far", extra = streamed_body(24)),
                message(2),
            ]
        )
    )
    got = P.compare(a, b)
    assert got["verdict"] != P.MATCH
    assert got["verdict"] == P.NOT_COMPARABLE, got


def test_a_real_change_inside_the_streamed_message_is_refused_not_caught():
    """THE COST, stated as a test so it cannot be forgotten.

    A rendering regression that lands inside the message that happens to be streaming is no longer
    distinguishable from stream progress, and comes back NOT COMPARABLE. It was not distinguishable
    before either -- every action that lands in a stream is on the declared unstable list, so the
    difference used to print under "expected to vary" and the run exited 0. The outcome moves from
    a pass to "not measured", which is the honest one, but it is a give-up and it is written down.
    """
    a = streaming_arm(24)
    b = capture(
        thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(1),
                message(
                    2,
                    streaming = True,
                    body = "reply so far",
                    extra = streamed_body(24) + [{"tag": "span", "children": ["a real change"]}],
                ),
            ]
        )
    )
    got = P.compare(a, b)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert got["verdict"] != P.MATCH


def test_a_message_that_is_in_flight_on_one_arm_only_is_still_excused():
    # The ordinary case, and it must not read as a build difference: one arm finished the reply
    # before its digest was taken and the other did not.
    a = streaming_arm(81)
    b = capture(
        thread(
            [
                message(0, role = "user", body = "the prompt"),
                message(1),
                message(2, streaming = False, body = "reply so far", extra = streamed_body(81)),
            ]
        )
    )
    assert P.compare(a, b)["verdict"] == P.NOT_COMPARABLE


def test_a_message_that_vanished_is_never_excused_by_being_in_flight():
    # The elision withholds a subtree, never an element. A streamed message that is absent on one
    # arm is a lost message, and no amount of "it was still arriving" makes that not a difference.
    a = streaming_arm(24)
    b = capture(thread([message(0, role = "user", body = "the prompt"), message(1)]))
    got = P.compare(a, b)
    assert got["verdict"] == P.DIFFER
    # Caught one level earlier than `localise`, by the mount-count check: neither arm is windowing
    # and they mounted different numbers of messages, which is a lost conversation and is reported
    # as such rather than as a row that moved.
    assert "different numbers of messages (3 vs 2)" in got["reason"], got


# ── 5. the positive control on the streaming probe itself ────────────


def test_a_running_reply_the_probe_could_not_place_refuses_the_pair():
    """A scan that can return zero needs a control, and this one can.

    `streamingMessages()` walks selectors written against Studio's markup. Rename `data-status` and
    it matches nothing, every capture reads as "no reply was in flight", and the instrument returns
    the strongest claim it has about the stream on the strength of never having looked. The app
    publishes the same fact through the Stop button, so the disagreement is carried out of the page
    and refused here rather than resolved into the reassuring answer.
    """
    a = streaming_arm(24)
    blind = dict(streaming_arm(24), in_flight = [], in_flight_unplaced = True)
    got = P.compare(a, blind)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert "could not be identified" in got["reason"]
    assert P.compare(blind, a)["verdict"] == P.NOT_COMPARABLE


def _blind_arm(chars: int, *, prompt: str = "the prompt") -> dict:
    """A streaming arm whose `data-status` hook went quiet, so it could not place its own stream.

    The shape the treatment side of a hook-renaming build has: the app says a reply is running and
    not one message published a streaming state.
    """
    arm = capture(
        thread(
            [
                message(0, role = "user", body = prompt),
                message(1, body = "settled text"),
                message(2, streaming = True, body = "reply so far", extra = streamed_body(chars)),
            ]
        )
    )
    return dict(arm, in_flight = [], in_flight_unplaced = True)


def test_a_settled_user_row_survives_the_blind_probe_refusal():
    """A USER ROW IS NEVER THE REPLY BEING WRITTEN, so the refusal may not take it out.

    The reachable pair is one build against another that renamed the `data-status` hook -- an
    assistant-ui bump does exactly that -- and the same bump can restyle the user bubble. The
    stream cannot be placed, so every assistant row is withheld correctly; the USER row is not,
    because a stream writes into an assistant message and both arms agree this row is the user's.

    Without this it left as NOT COMPARABLE with an empty `moved`, and `report` buckets a refusal as
    blind and never consults it for the exit code, so a real rendering regression exited 0 with no
    row on screen naming it. `compare_visible` already applies this rule to its own rows.
    """
    base = _blind_arm(24)
    treat = _blind_arm(24, prompt = "the prompt, rendered differently")
    assert treat["in_flight_unplaced"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == [
        "msg0(user):%d->%dc" % (base["messages"][0]["chars"], treat["messages"][0]["chars"])
    ], got["moved"]
    assert "could not be identified" in got["reason"]
    # And it is symmetric: which arm went blind cannot decide whether the finding is reported.
    assert P.compare(treat, base)["verdict"] == P.DIFFER


def test_a_role_that_changed_survives_the_blind_probe_refusal():
    """The role is captured BESIDE the digest, so how far a reply has arrived cannot explain it.

    A treatment that renders the live assistant row as `data-role="user"` is a structural
    regression, and it is exactly the row the refusal is otherwise about.
    """
    base = _blind_arm(24)
    treat = dict(
        capture(
            thread(
                [
                    message(0, role = "user", body = "the prompt"),
                    message(1, body = "settled text"),
                    message(
                        2,
                        role = "user",
                        streaming = True,
                        body = "reply so far",
                        extra = streamed_body(24),
                    ),
                ]
            )
        ),
        in_flight = [],
        in_flight_unplaced = True,
    )
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    # The scaffold carries the role marker too, and both arms agree about generation here, so the
    # existing scaffold rule quotes it as well. The row itself is what must not be lost.
    assert "msg2:role assistant->user" in got["moved"], got["moved"]


def test_the_blind_scaffold_rule_needs_run_state_evidence_the_composer_cannot_forge():
    """SUPPRESSING A COMPOSER DIFFERENCE WITH THE COMPOSER IS ARGUING IN A CIRCLE.

    The blind branch withheld the scaffold whenever `generation_disagrees` said the arms were at
    different points in the turn -- and that predicate reads `composer_control`, the token naming
    which control the composer rendered. So a treatment that DROPS the Stop button, renames it, or
    selects the wrong control makes the tokens differ FOR THAT REASON, and the scaffold carrying
    the regression went out with the refusal. `report` files a refusal under `blind` and takes its
    exit code from `stable_bad or one_sided`, so the run went green on it.

    `streaming` and `queued_idle` are read off the thread's run state, not off the composer, so
    they are evidence the composer cannot manufacture. Here they AGREE -- both arms are generating
    -- so there is no run-state explanation for the scaffold moving and it is a finding. This is
    the same corroboration the settled-pair composer suppression already requires.
    """
    base = dict(_blind_arm(24), composer_control = "Stop generating", streaming = True)
    treat = dict(
        _blind_arm(24),
        # The regression: the control is gone, so the composer renders a different subtree and the
        # scaffold moves with it.
        composer_control = "",
        streaming = True,
        digest_scaffold = "scaffold-with-no-stop-button",
        chars_scaffold = base["chars_scaffold"] - 21,
    )
    # The run state itself says the two arms were doing the same thing.
    assert P._run_state_disagrees(base, treat) is False
    # ...while the composer-derived predicate says they were not, which is the circle.
    assert P.generation_disagrees(base, treat) is True

    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("thread scaffolding outside any message" in m for m in got["moved"]), got["moved"]
    # Symmetric: which arm lost the button cannot decide whether the finding is reported.
    assert P.compare(treat, base)["verdict"] == P.DIFFER


def test_the_blind_scaffold_rule_still_withholds_a_corroborated_run_state_difference():
    """THE CONTROL. When the run state independently says one arm was generating and the other was
    not, the composer differs BECAUSE of that, and quoting it would manufacture the wall-clock
    false alarm this file exists to remove. The refusal is right there and must survive."""
    base = dict(_blind_arm(24), composer_control = "Stop generating", streaming = True)
    treat = dict(
        _blind_arm(24),
        composer_control = "Send message",
        streaming = False,
        digest_scaffold = "scaffold-with-send-button",
        chars_scaffold = base["chars_scaffold"] + 8,
    )
    assert P._run_state_disagrees(base, treat) is True

    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert not any("thread scaffolding" in m for m in got["moved"]), got["moved"]


def test_the_composer_refusal_says_the_scaffold_reading_is_an_aggregate():
    """A REFUSAL THAT NAMES ONE CAUSE READS AS HAVING RULED OUT THE OTHERS.

    `digest_scaffold` is one digest over the viewport, the composer dock and the empty state
    together. The suppression is justified by the composer swap alone, but the reading it acts on
    cannot tell that swap apart from a change to another scaffold surface in the same
    cross-run-state capture, so such a change rides along inside the refusal.

    That is a real limit of this capture and separating it needs a composer-scoped digest the
    payload does not carry. What must not happen meanwhile is the sentence claiming more than the
    reading supports: unqualified, it reads as "only the composer differed", which is what a reader
    would act on. So the refusal states the aggregate and states what it could not separate.
    """
    settled = thread([message(0, role = "user", body = "the prompt"), message(1)])
    base = dict(capture(settled), composer_control = "Stop generating", streaming = True)
    treat = dict(
        capture(settled),
        composer_control = "Send message",
        streaming = False,
        digest_scaffold = "scaffold-with-send-button",
        chars_scaffold = base["chars_scaffold"] + 8,
    )
    got = P.compare(base, treat)
    # The pair really does take the composer suppression, not the blind one.
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"], got["reason"]
    assert "ONE AGGREGATE digest" in got["reason"], got["reason"]
    assert "cannot separate the composer swap" in got["reason"], got["reason"]


def test_the_streamed_row_itself_is_still_withheld_when_the_probe_is_blind():
    """The narrowing is not a hole in the other direction.

    An assistant row that differs is precisely the reading with no defined moment, and it stays
    refused. Only rows that provably cannot be the reply being written are reported.
    """
    got = P.compare(_blind_arm(24), _blind_arm(48))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []
    # A SETTLED ASSISTANT row is withheld too: with the stream unplaceable, this instrument cannot
    # prove which assistant row the reply is being written into.
    base = _blind_arm(24)
    treat = dict(
        capture(
            thread(
                [
                    message(0, role = "user", body = "the prompt"),
                    message(1, body = "a different settled text"),
                    message(2, streaming = True, body = "reply so far", extra = streamed_body(24)),
                ]
            )
        ),
        in_flight = [],
        in_flight_unplaced = True,
    )
    assert P.compare(base, treat)["verdict"] == P.NOT_COMPARABLE


def test_a_user_row_flagged_in_flight_by_the_other_arm_is_still_withheld():
    """The arm that COULD place its stream is believed about which rows have no defined moment."""
    base = _blind_arm(24)
    # The non-blind arm names index 0 as in flight. Whatever role it carries, its digest is a point
    # in a stream on that arm, so it is not a settled row.
    treat = dict(_blind_arm(24, prompt = "a different prompt"), in_flight = [0])
    treat["in_flight_unplaced"] = True
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []


def test_an_old_payload_without_the_streaming_fields_is_scored_as_it_always_was():
    # Not silently refused, and not silently excused: a capture recorded before the fields existed
    # carries no claim about a stream, and falls back to the digest it does carry.
    a = streaming_arm(24, streaming_fields = False)
    b = streaming_arm(24, streaming_fields = False)
    assert P.compare(a, b)["verdict"] == P.MATCH
    assert P.compare(a, streaming_arm(48, streaming_fields = False))["verdict"] == P.DIFFER


# ── 6. the elision itself, at the level of the shipped signature ─────


def test_eliding_a_subtree_keeps_its_presence_its_position_and_its_role():
    got = run_js(
        {
            "elided": [
                _mark_elided(
                    thread(
                        [message(0, role = "user", body = "hi"), message(1, streaming = True, body = "a")]
                    )
                ),
                _mark_elided(
                    thread(
                        [
                            message(0, role = "user", body = "hi"),
                            message(1, streaming = True, body = "a completely different reply"),
                        ]
                    )
                ),
                _mark_elided(thread([message(0, role = "user", body = "hi")])),
                _mark_elided(
                    thread(
                        [
                            message(0, role = "user", body = "hi"),
                            message(1, role = "user", streaming = True, body = "a"),
                        ]
                    )
                ),
            ]
        }
    )["elided"]
    short, long_, absent, other_role = got
    # The content inside is withheld...
    assert short == long_
    # ...but the element being there at all is not, and neither is what it is.
    assert short != absent
    assert short != other_role
    assert "<!in-flight div role=assistant>" in short


def test_elision_is_off_unless_asked_for():
    tree = thread([message(0, role = "user", body = "hi"), message(1, streaming = True, body = "a")])
    plain = run_js({"trees": [tree]})["signatures"][0]
    assert "<!in-flight" not in plain
    assert "a" in plain
