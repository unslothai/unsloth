# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Streamed replies are refused, not scored: the two arms sample the same stream at different points."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.analysis import parity as P  # noqa: E402

from tests.studio.studiobench.fixture.selftest.test_studiobench_parity_digest import (  # noqa: E402
    run_js,
)

# Fixtures are DOM trees digested by the shipped signature(), not hand-written digests.


def message(
    index: int,
    *,
    role: str = "assistant",
    body: str = "settled text",
    streaming: bool = False,
    extra: list | None = None,
    attrs: dict | None = None,
) -> dict:
    """The text part's data-status is running while streaming and complete once settled; the value moves."""
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
    """Recursive and keyed on data-role like capture(); a shallow walk would skip nested branches."""
    if not isinstance(node, dict):
        return node
    if (node.get("attrs") or {}).get("data-role"):
        return dict(node, elide = True)
    return dict(node, children = [_mark_elided(c) for c in node.get("children") or []])


def capture(tree: dict, *, streaming_fields: bool = True) -> dict:
    """Per-arm parity capture; streaming_fields=False gives the pre-streaming instrument's exact output."""
    messages = tree["children"][0]["children"]
    overlays = tree.get("_overlays") or []
    # Elide every message, as capture() does, so the scaffold walk is identical on both arms.
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


# The KaTeX error title's char offset moves with every arriving character.
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


STREAM_POINTS = (12, 24, 48, 81)


def test_the_same_document_at_two_points_in_one_stream_moves_the_raw_digest():
    """Same build sampled at two points in one reply moves the raw digest: stream drift, not a UI change."""
    a, b = streaming_arm(24), streaming_arm(48)
    assert a["digest"] != b["digest"], "no drift to fix; the fixture is not reproducing the defect"
    assert a["digest_scaffold"] == b["digest_scaffold"]


def test_the_drift_is_refused_rather_than_reported_as_a_ui_change():
    got = P.compare(streaming_arm(24), streaming_arm(48))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []
    assert got["not_digested"] == [2]
    assert "STILL BEING WRITTEN" in got["reason"]


def test_two_genuinely_different_documents_still_differ_while_a_reply_streams():
    """A real difference in a settled message must still be reported while a reply streams."""
    got = P.compare(
        streaming_arm(24, settled_body = "settled text"),
        streaming_arm(24, settled_body = "settled text, rewritten"),
    )
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == ["msg1(assistant):120->131c"], got["moved"]


def test_a_real_difference_survives_the_streamed_message_drifting_at_the_same_time():
    got = P.compare(
        streaming_arm(24, settled_body = "settled text"),
        streaming_arm(81, settled_body = "settled text, rewritten"),
    )
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == ["msg1(assistant):120->131c"], got["moved"]
    assert got["in_flight"] == [2]


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
    assert all(r["verdict"] == P.NOT_COMPARABLE for r in results)


def test_the_null_battery_scores_the_old_instrument_too():
    """Runs the null battery without streaming fields, so compare falls back to the plain digest."""
    before = null_battery(streaming_fields = False)
    assert all(r["verdict"] == P.DIFFER for r in before), before
    for r in before:
        assert [m for m in r["moved"] if m.startswith("msg2(")] == r["moved"], r["moved"]


def test_two_arms_that_landed_on_the_same_point_in_the_stream_still_match():
    got = P.compare(streaming_arm(24), streaming_arm(24))
    assert got["verdict"] == P.MATCH, got
    assert got["in_flight"] == [2]


def test_a_settled_thread_is_scored_exactly_as_it_was():
    settled = capture(thread([message(0, role = "user", body = "hi"), message(1)]))
    assert settled["in_flight"] == [] and settled["streaming"] is False
    assert P.compare(settled, settled)["verdict"] == P.MATCH
    # The scaffold is shorter by design; per-message rows restore completeness.
    assert settled["chars_scaffold"] < settled["chars"]


# Mutants are injected at the same stream point on both arms, so only the mutation can differ.


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
    # Only MATCH lets a change ship green; DIFFER and NOT COMPARABLE both block.
    for name, before, after in mutants():
        assert P.compare(before, after)["verdict"] != P.MATCH, name


def test_the_mutant_score_is_unchanged_by_the_streaming_fields():
    # Must match the old instrument's score; divergence means elision hides something.
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


def test_reordering_the_streamed_message_past_a_sibling_is_refused_not_passed():
    """Reordering the streamed message past a same-role sibling is refused as NOT COMPARABLE, not passed."""
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
    """A real change inside the streaming message is NOT COMPARABLE, a known give-up rather than a pass."""
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
    a = streaming_arm(24)
    b = capture(thread([message(0, role = "user", body = "the prompt"), message(1)]))
    got = P.compare(a, b)
    assert got["verdict"] == P.DIFFER
    assert "different numbers of messages (3 vs 2)" in got["reason"], got


def test_a_running_reply_the_probe_could_not_place_refuses_the_pair():
    """A running reply the probe cannot place refuses the pair, rather than reading as no reply in
    flight."""
    a = streaming_arm(24)
    blind = dict(streaming_arm(24), in_flight = [], in_flight_unplaced = True)
    got = P.compare(a, blind)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert "could not be identified" in got["reason"]
    assert P.compare(blind, a)["verdict"] == P.NOT_COMPARABLE


def _blind_arm(chars: int, *, prompt: str = "the prompt") -> dict:
    """A streaming arm whose data-status hook published nothing, so its own stream cannot be placed."""
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
    """A user row is never the reply being written, so the blind-probe refusal must not withhold it."""
    base = _blind_arm(24)
    treat = _blind_arm(24, prompt = "the prompt, rendered differently")
    assert treat["in_flight_unplaced"] is True
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert got["moved"] == [
        "msg0(user):%d->%dc" % (base["messages"][0]["chars"], treat["messages"][0]["chars"])
    ], got["moved"]
    assert "could not be identified" in got["reason"]
    assert P.compare(treat, base)["verdict"] == P.DIFFER


def test_a_role_that_changed_survives_the_blind_probe_refusal():
    """The role is captured beside the digest, so a role change survives the blind-probe refusal."""
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
    assert "msg2:role assistant->user" in got["moved"], got["moved"]


def test_the_blind_scaffold_rule_needs_run_state_evidence_the_composer_cannot_forge():
    """Scaffold suppression needs run-state evidence the composer cannot forge, not composer_control."""
    base = dict(_blind_arm(24), composer_control = "Stop generating", streaming = True)
    treat = dict(
        _blind_arm(24),
        composer_control = "",
        streaming = True,
        digest_scaffold = "scaffold-with-no-stop-button",
        chars_scaffold = base["chars_scaffold"] - 21,
    )
    assert P._run_state_disagrees(base, treat) is False
    assert P.generation_disagrees(base, treat) is True

    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert any("thread scaffolding outside any message" in m for m in got["moved"]), got["moved"]
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
    """digest_scaffold covers viewport, composer dock and empty state together, so it is an aggregate."""
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
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "composer dock is inside the thread root" in got["reason"], got["reason"]
    assert "ONE AGGREGATE digest" in got["reason"], got["reason"]
    assert "cannot separate the composer swap" in got["reason"], got["reason"]


def test_the_streamed_row_itself_is_still_withheld_when_the_probe_is_blind():
    """Assistant rows stay withheld while the probe is blind; an unplaceable stream may write into any."""
    got = P.compare(_blind_arm(24), _blind_arm(48))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []
    # Settled assistant rows are withheld too: an unplaceable stream could be writing into any.
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
    treat = dict(_blind_arm(24, prompt = "a different prompt"), in_flight = [0])
    treat["in_flight_unplaced"] = True
    got = P.compare(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []


def test_an_old_payload_without_the_streaming_fields_is_scored_as_it_always_was():
    a = streaming_arm(24, streaming_fields = False)
    b = streaming_arm(24, streaming_fields = False)
    assert P.compare(a, b)["verdict"] == P.MATCH
    assert P.compare(a, streaming_arm(48, streaming_fields = False))["verdict"] == P.DIFFER


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
    assert short == long_
    assert short != absent
    assert short != other_role
    assert "<!in-flight div role=assistant>" in short


def test_elision_is_off_unless_asked_for():
    tree = thread([message(0, role = "user", body = "hi"), message(1, streaming = True, body = "a")])
    plain = run_js({"trees": [tree]})["signatures"][0]
    assert "<!in-flight" not in plain
    assert "a" in plain
