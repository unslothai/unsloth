# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The UI-parity digest, tested in both directions without a browser.

A digest is a claim with two failure modes and they pull in opposite directions:

    FALSE POSITIVE  something volatile survives normalisation, two runs of ONE build disagree, and
                    within a day nobody opens the report. The live proof is the null control; what
                    is testable here is that each normalisation rule does what it says.
    FALSE NEGATIVE  something that matters is normalised away or never walked, the check passes
                    and the UI changed anyway. This is the worse one because it is silent, and it
                    is what most of this file is about: every KEPT property gets a test that the
                    signature moves when it moves.

WHY THE REAL JAVASCRIPT AND NOT A PYTHON PORT. The normaliser that ships is `scene/parity.js`. A
Python re-implementation tested here would pass forever while the shipped regexes drifted away
from it, and the test would be measuring itself. So the fixtures are Python, the evaluator is
node running the actual file, and if node is missing the test SKIPS with that reason stated rather
than passing on a substitute. A skip says "not measured"; a pass would say "measured, fine".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.analysis import parity as P  # noqa: E402

PARITY_JS = Path(__file__).resolve().parents[2] / "scene" / "parity.js"

# A DOM shim, not a DOM. `signature()` touches exactly six things on an element, so those six are
# what the harness provides; anything richer would be a second browser to keep correct. `capture()`
# is deliberately NOT exercised here because it needs querySelectorAll, getComputedStyle and the
# real selector adapter -- that is the live null and spike controls' job, and pretending to cover
# it with a mock is how a test suite comes to assert its own fixtures.
HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const window = { };
const document = { body: { tagName: "BODY", attributes: [], childNodes: [],
                           getAttribute: () => null },
                   querySelectorAll: () => [] };
window.getComputedStyle = () => ({ getPropertyValue: () => "" });
(new Function("window", "document", src))(window, document);

const build = (spec) => {
  if (typeof spec === "string") return { nodeType: 3, nodeValue: spec };
  const attrs = spec.attrs || {};
  return {
    nodeType: 1,
    tagName: (spec.tag || "div").toUpperCase(),
    attributes: Object.keys(attrs).map((name) => ({ name })),
    getAttribute: (name) => (name in attrs ? attrs[name] : null),
    childNodes: (spec.children || []).map(build),
  };
};

const spec = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const parity = window.__sb.parity;
console.log(JSON.stringify({
  texts: (spec.texts || []).map((t) => parity.normText(t)),
  urls: (spec.urls || []).map((u) => parity.normUrl(u)),
  signatures: (spec.trees || []).map((t) => parity.signature(build(t))),
  hashes: (spec.hashes || []).map((s) => parity.hash(s)),
}));
"""


def _node() -> str:
    exe = shutil.which("node") or shutil.which("nodejs")
    if exe is None:
        pytest.skip(
            "node is not installed, so the shipped parity.js could not be evaluated; "
            "this is NOT MEASURED rather than passing"
        )
    return exe


def run_js(spec: dict) -> dict:
    exe = _node()
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.js"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        payload = Path(tmp) / "spec.json"
        payload.write_text(json.dumps(spec), encoding = "utf-8")
        got = subprocess.run(
            [exe, str(harness), str(PARITY_JS), str(payload)],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    if got.returncode != 0:
        raise AssertionError(f"the parity.js harness failed: {got.stderr.strip()[-800:]}")
    return json.loads(got.stdout)


def norm_text(*values: str) -> list[str]:
    return run_js({"texts": list(values)})["texts"]


def sig(tree: dict) -> str:
    return run_js({"trees": [tree]})["signatures"][0]


def sigs(*trees: dict) -> list[str]:
    return run_js({"trees": list(trees)})["signatures"]


# ── the normaliser: things that MUST be erased ──────────────────────


def test_rendered_durations_collapse():
    # unslothai/unsloth#9054: a 295 vs 310 ms difference in the action bar, which is wall clock.
    got = norm_text("copied in 295ms", "copied in 310ms", "took 1.2 s", "ran for 3 min")
    assert got[0] == got[1], got
    assert "#T" in got[2] and "#T" in got[3], got


def test_relative_and_absolute_times_collapse():
    got = norm_text("sent just now", "sent yesterday", "sent at 14:05", "sent at 2:05 pm")
    assert all("#T" in g for g in got), got


def test_backend_minted_uuids_collapse():
    # The volatile that made the FIRST null control fail on all eighteen actions: every message
    # root carries `data-message-id`, and the two arms are two installs with two databases.
    a, b = norm_text(
        "id 71ad5735-ede4-464d-a36b-44309ef67624", "id f44017dd-f5f7-45dd-9f53-475c115e61ac"
    )
    assert a == b == "id #ID", (a, b)


def test_long_hex_ids_collapse():
    a, b = norm_text("build 3a816e656eb74295aa11", "build 9f0d1c2b7e6a4d38bb22")
    assert a == b, (a, b)


def test_urls_lose_their_origin_but_keep_their_path():
    got = run_js(
        {
            "urls": [
                "http://127.0.0.1:5830/assets/index.js",
                "http://127.0.0.1:5831/assets/index.js",
                "http://127.0.0.1:5830/assets/other.js",
                "blob:http://127.0.0.1:5830/8f2c-11",
                "data:image/png;base64,AAAA",
            ]
        }
    )
    urls = got["urls"]
    # The two arms of an A/B are two ports by construction, so the origin cannot be signal.
    assert urls[0] == urls[1], urls
    # ...but a DIFFERENT asset at the same origin still has to move the digest.
    assert urls[0] != urls[2], urls
    assert urls[3] == "#BLOB" and urls[4].startswith("#DATA:"), urls


# ── the normaliser: things that MUST SURVIVE ────────────────────────
#
# Every test above widens the set of things the digest cannot see. These are the counterweight:
# a normaliser that erased them would pass a null control perfectly and detect nothing.


def test_a_bare_number_is_not_a_duration():
    a, b = norm_text("3 files changed", "4 files changed")
    assert a != b, (a, b)


def test_a_word_beginning_with_a_unit_letter_is_not_a_unit():
    a, b = norm_text("5 stars", "6 stars")
    assert a != b, (a, b)


def test_a_short_hex_string_is_not_an_id():
    # Colours, error codes and the like are eight characters or fewer and are content.
    a, b = norm_text("code deadbeef", "code cafebabe")
    assert a != b, (a, b)


def test_text_content_moves_the_signature():
    one, two = sigs(
        {"tag": "p", "children": ["hello world"]}, {"tag": "p", "children": ["hello worlds"]}
    )
    assert one != two


# ── the signature: every KEPT property, tested as kept ──────────────


@pytest.mark.parametrize(
    "attr,before,after",
    [
        ("data-state", "open", "closed"),  # a reasoning pane that silently collapses
        ("data-slot", "reasoning-root", "tool-root"),
        ("data-role", "assistant", "user"),
        ("aria-hidden", "false", "true"),  # content gone from the accessibility tree
        ("class", "flex gap-2", "flex-col gap-8"),  # a layout class swap
        ("title", "Copy code", "Copy"),
        ("role", "menu", "listbox"),
    ],
)
def test_a_changed_attribute_value_moves_the_signature(attr, before, after):
    one, two = sigs({"tag": "div", "attrs": {attr: before}}, {"tag": "div", "attrs": {attr: after}})
    assert one != two, attr


def test_adding_or_removing_a_boolean_attribute_moves_the_signature():
    # `disabled` has no value to compare, so only its PRESENCE can carry it.
    one, two = sigs({"tag": "button", "attrs": {}}, {"tag": "button", "attrs": {"disabled": ""}})
    assert one != two


def test_a_volatile_attribute_keeps_its_presence_even_though_its_value_is_dropped():
    # Dropping the value must not drop the fact that the attribute is there: an element that
    # gains an `id` has changed, even though which id it gained is noise.
    plain, with_id = sigs(
        {"tag": "div", "attrs": {}}, {"tag": "div", "attrs": {"id": "radix-:r1a:"}}
    )
    assert plain != with_id
    # Two different generated ids, however, must read the same.
    a, b = sigs(
        {"tag": "div", "attrs": {"id": "radix-:r1a:"}},
        {"tag": "div", "attrs": {"id": "radix-:r9z:"}},
    )
    assert a == b


def test_the_shared_signature_still_sees_virtualization_bookkeeping():
    """WHERE THE `aria-posinset` EXCLUSION LIVES, and where it does not.

    The VISIBLE-region digest drops `aria-posinset` and `aria-setsize`, because readiness.py lets a
    windowed arm publish them on the message itself and the fully mounted arm publishes neither --
    so comparing them reports every message as changed while the content is identical. That
    exclusion is passed in by that one caller. The shared `signature`, which the whole-thread
    digest, the per-message rows and the overlays all use, keeps them: those pairs are scored only
    when NEITHER arm is windowing, and there an ordinal that appears or moves is a real difference.
    """
    plain, numbered = sigs(
        {"tag": "div", "attrs": {}}, {"tag": "div", "attrs": {"aria-posinset": "3"}}
    )
    assert plain != numbered
    three, four = sigs(
        {"tag": "div", "attrs": {"aria-posinset": "3"}},
        {"tag": "div", "attrs": {"aria-posinset": "4"}},
    )
    assert three != four
    small, large = sigs(
        {"tag": "div", "attrs": {"aria-setsize": "18"}},
        {"tag": "div", "attrs": {"aria-setsize": "180"}},
    )
    assert small != large


def test_added_and_removed_elements_move_the_signature():
    small, large = sigs(
        {"tag": "div", "children": [{"tag": "span"}]},
        {"tag": "div", "children": [{"tag": "span"}, {"tag": "b"}]},
    )
    assert small != large


def test_reordered_siblings_move_the_signature():
    # Two elements with identical content in the other order. A digest built from a SET rather
    # than a sequence would read these as equal, and a list that renders backwards is a real bug.
    one, two = sigs(
        {
            "tag": "ul",
            "children": [{"tag": "li", "children": ["a"]}, {"tag": "li", "children": ["b"]}],
        },
        {
            "tag": "ul",
            "children": [{"tag": "li", "children": ["b"]}, {"tag": "li", "children": ["a"]}],
        },
    )
    assert one != two


def test_nesting_moves_the_signature():
    # Same tags, same text, different tree. Closing tags are what make this detectable.
    flat, nested = sigs(
        {"tag": "div", "children": [{"tag": "span", "children": ["x"]}, {"tag": "b"}]},
        {"tag": "div", "children": [{"tag": "span", "children": ["x", {"tag": "b"}]}]},
    )
    assert flat != nested


def test_attribute_order_does_not_move_the_signature():
    # React can emit attributes in either order for the same render. Sorting them is what makes
    # the digest a property of the DOM rather than of the serialiser.
    one, two = sigs(
        {"tag": "div", "attrs": {"class": "a", "data-state": "open"}},
        {"tag": "div", "attrs": {"data-state": "open", "class": "a"}},
    )
    assert one == two


def test_whitespace_only_text_nodes_do_not_move_the_signature():
    one, two = sigs(
        {"tag": "p", "children": ["hello"]}, {"tag": "p", "children": ["hello", "   ", "\n\t"]}
    )
    assert one == two


def test_the_depth_cap_leaves_a_visible_marker():
    # A truncated signature that reads like a complete one is the silent false negative this file
    # exists to rule out, so the cap has to be legible in the output.
    deep = {"tag": "div"}
    for _ in range(60):
        deep = {"tag": "div", "children": [deep]}
    assert "<!depth-cap>" in sig(deep)


def test_content_below_the_depth_cap_is_not_compared():
    # The honest statement of the limit: past 40 levels the digest stops looking, and this test
    # records that as a KNOWN hole rather than leaving somebody to discover it.
    def wrap(inner, n):
        for _ in range(n):
            inner = {"tag": "div", "children": [inner]}
        return inner

    one, two = sigs(
        wrap({"tag": "p", "children": ["alpha"]}, 60), wrap({"tag": "p", "children": ["omega"]}, 60)
    )
    assert one == two, "if this now fails the cap moved and the docstring must be updated"


# ── the comparison layer, in pure Python ────────────────────────────


def capture(
    digest = "aaaa",
    *,
    messages = None,
    overlays = None,
    root = "thread",
    styles = None,
    chars = 100,
) -> dict:
    return {
        "parity_attempted": True,
        "root_kind": root,
        "digest": digest,
        "chars": chars,
        "messages": messages
        if messages is not None
        else [
            {"i": 0, "role": "user", "digest": "m0", "chars": 10},
            {"i": 1, "role": "assistant", "digest": "m1", "chars": 20},
        ],
        "overlays": overlays if overlays is not None else [],
        "styles": styles
        if styles is not None
        else {"digest": "s0", "chars": 5, "elements": 4, "capped": False},
    }


def test_identical_captures_match():
    assert P.compare(capture(), capture())["verdict"] == P.MATCH


def test_a_failed_capture_is_never_a_match():
    # The single most dangerous confusion in the whole instrument: a capture that threw and a
    # capture that agreed both produce no complaint unless they are told apart here.
    failed = {"parity_attempted": False, "reason": "threadRoot is not a function"}
    got = P.compare(failed, capture())
    assert got["verdict"] == P.NOT_COMPARABLE
    assert "threadRoot" in got["reason"]
    assert P.compare(None, capture())["verdict"] == P.NOT_COMPARABLE


def test_two_different_roots_are_not_comparable():
    # A body-root capture carries the sidebar and its relative timestamps. Comparing it with a
    # thread-root one produces two plausible hashes and a meaningless verdict.
    got = P.compare(capture(root = "thread"), capture(root = "body"))
    assert got["verdict"] == P.NOT_COMPARABLE
    assert "different roots" in got["reason"]


def test_a_capture_from_an_older_instrument_is_not_silently_compared():
    old = capture()
    del old["root_kind"]
    assert P.compare(old, capture())["verdict"] == P.NOT_COMPARABLE
    # Both sides old is an old payload, which IS comparable; it just predates the field.
    other = capture("bbbb")
    del other["root_kind"]
    assert P.compare(old, other)["verdict"] == P.DIFFER


def test_a_difference_is_localised_to_the_message_that_moved():
    moved = capture(
        "zzzz",
        messages = [
            {"i": 0, "role": "user", "digest": "m0", "chars": 10},
            {"i": 1, "role": "assistant", "digest": "CHANGED", "chars": 33},
        ],
    )
    got = P.compare(capture(), moved)
    assert got["verdict"] == P.DIFFER
    assert got["moved"] == ["msg1(assistant):20->33c"], got["moved"]


def test_an_added_message_is_localised_as_one_sided():
    extra = capture(
        "zzzz",
        messages = [
            {"i": 0, "role": "user", "digest": "m0", "chars": 10},
            {"i": 1, "role": "assistant", "digest": "m1", "chars": 20},
            {"i": 2, "role": "assistant", "digest": "m2", "chars": 5},
        ],
    )
    assert P.compare(capture(), extra)["moved"] == ["msg2(assistant):only treatment"]


def test_an_overlay_that_changes_without_changing_count_is_still_localised():
    # The bug this pins: comparing only the NUMBER of overlays passes an open menu whose contents
    # were rewritten, which is exactly the popover regression the overlay walk was added for.
    one = capture("aaaa", overlays = [{"sel": '[role="menu"]', "digest": "o1", "chars": 40}])
    two = capture("zzzz", overlays = [{"sel": '[role="menu"]', "digest": "o2", "chars": 44}])
    got = P.compare(one, two)
    assert got["moved"] == ['overlay0[[role="menu"]]:40->44c'], got["moved"]


def test_an_overlay_change_alone_is_a_difference():
    # THE FALSE NEGATIVE THE SPIKE CONTROL FOUND. An overlay lives outside the thread root, so a
    # menu that mounts when it should not leaves the whole-thread digest untouched. Testing only
    # that digest made the entire overlay walk unreachable and reported a clean pass.
    one = capture("aaaa", overlays = [])
    two = capture("aaaa", overlays = [{"sel": '[role="menu"]', "digest": "o1", "chars": 40}])
    got = P.compare(one, two)
    assert got["verdict"] == P.DIFFER, "an overlay appearing on one arm only is a difference"
    assert got["moved"] == ["overlays 0->1"], got["moved"]


def test_a_message_change_alone_is_a_difference():
    # The same shape one level down: if a per-message digest moves while the whole-thread digest
    # somehow does not, the pair still differs. Belt and braces on the aggregate hash.
    moved = capture(
        "aaaa",
        messages = [
            {"i": 0, "role": "user", "digest": "m0", "chars": 10},
            {"i": 1, "role": "assistant", "digest": "CHANGED", "chars": 20},
        ],
    )
    assert P.compare(capture(), moved)["verdict"] == P.DIFFER


def test_a_difference_outside_every_message_is_reported_as_such():
    # An empty `moved` list would read as "nothing differs" next to a DIFFER verdict.
    got = P.compare(capture("aaaa"), capture("zzzz"))
    assert got["verdict"] == P.DIFFER
    assert got["moved"] and "scaffolding" in got["moved"][0]


def test_the_style_probe_is_a_separate_verdict_from_the_structural_one():
    styled = capture(styles = {"digest": "OTHER", "chars": 5, "elements": 4, "capped": False})
    got = P.compare(capture(), styled)
    # Structure identical, style moved: a stylesheet change is exactly this shape, and folding it
    # into the structural verdict would put the hard signal's credibility on the soft reading.
    assert got["verdict"] == P.MATCH
    assert got["style_verdict"] == P.DIFFER


def test_a_capped_style_probe_is_not_comparable_rather_than_equal():
    capped = capture(styles = {"digest": "s0", "chars": 5, "elements": 64, "capped": True})
    got = P.compare(capture(), capped)
    assert got["style_verdict"] == P.NOT_COMPARABLE


# ── mutation detection and the derived unstable set ─────────────────


def test_mutation_detected_reports_a_real_change():
    got = P.mutation_detected(capture(), capture("zzzz"))
    assert got["detected"] is True


def test_mutation_detected_does_not_claim_a_detection_it_did_not_make():
    assert P.mutation_detected(capture(), capture())["detected"] is False
    # And a capture that FAILED is not a detection either, in either direction.
    got = P.mutation_detected(capture(), {"parity_attempted": False, "reason": "gone"})
    assert got["detected"] is False and got["verdict"] == P.NOT_COMPARABLE


def test_an_action_that_never_ran_is_not_a_matching_surface():
    # MEASURED, not imagined: on a 100K fast-tier null control, five of the eighteen actions did
    # not run on either arm -- no attachments button, no Copy button, a missed slot. The window
    # still closes and the digest is still captured, so both arms agreed and `image_upload` was
    # reported as a stable, matching surface. Nobody had opened it.
    idle = {"ran": False, "reason": "no visible attachments button", "parity": capture()}
    got = P.compare_rows(idle, idle)
    assert got["verdict"] == P.NOT_EXERCISED
    assert "nothing touched" in got["reason"]
    # NEITHER arm ran it, so nobody opened the surface on either build and the only thing lost is
    # coverage. `one_sided` says so, and the caller needs it to keep that apart from the case below.
    assert got["one_sided"] == ""


def test_an_action_only_one_arm_could_perform_is_named_as_such():
    # A control that stops opening leaves NO digest to differ: the arm that cannot reach it
    # records `ran: false` and the pair carries no comparison at all. Folding that into the
    # missed-slot case is how a button that no longer works reads as lost coverage.
    idle = {"ran": False, "reason": "the control never became visible", "parity": capture()}
    got = P.compare_rows({"ran": True, "parity": capture()}, idle)
    assert got["verdict"] == P.NOT_EXERCISED
    assert got["one_sided"] == "base"
    assert "did not behave the same way" in got["reason"]
    # And in the other direction, named after the arm that DID run it.
    assert P.compare_rows(idle, {"ran": True, "parity": capture()})["one_sided"] == "treatment"


def test_a_pair_that_ran_on_both_arms_is_compared_normally():
    got = P.compare_rows(
        {"ran": True, "parity": capture()}, {"ran": True, "parity": capture("zzzz")}
    )
    assert got["verdict"] == P.DIFFER


def test_an_unexercised_action_contributes_no_evidence_of_stability():
    got = P.derive_unstable(
        [
            ("image_upload", {"verdict": P.NOT_EXERCISED}),
            ("image_upload", {"verdict": P.NOT_EXERCISED}),
        ]
    )
    assert got["image_upload"]["observations"] == 0
    assert got["image_upload"]["undetermined"] is True
    assert got["image_upload"]["unstable"] is False


def test_instability_needs_more_than_one_observation():
    once = P.derive_unstable([("copy_markdown", {"verdict": P.DIFFER})])
    assert once["copy_markdown"]["undetermined"] is True
    assert once["copy_markdown"]["unstable"] is False


def test_an_action_that_differs_against_itself_is_derived_as_unstable():
    got = P.derive_unstable(
        [
            ("stop_generation", {"verdict": P.DIFFER}),
            ("stop_generation", {"verdict": P.MATCH}),
            ("settings", {"verdict": P.MATCH}),
            ("settings", {"verdict": P.MATCH}),
        ]
    )
    assert got["stop_generation"]["unstable"] is True
    assert got["settings"]["unstable"] is False


def test_a_blind_action_is_counted_as_blind_and_not_as_stable():
    # An action whose digest could never be captured has an observation count of zero. Reporting
    # it as "stable" would be the instrument certifying a surface it never looked at.
    got = P.derive_unstable(
        [
            ("image_upload", {"verdict": P.NOT_COMPARABLE}),
            ("image_upload", {"verdict": P.NOT_COMPARABLE}),
        ]
    )
    assert got["image_upload"]["observations"] == 0
    assert got["image_upload"]["not_comparable"] == 2
    assert got["image_upload"]["unstable"] is False
    assert got["image_upload"]["undetermined"] is True


def test_the_cross_check_reports_both_directions_of_disagreement():
    derived = {
        "stop_generation": {"unstable": True, "undetermined": False},
        "settings": {"unstable": True, "undetermined": False},
        "scroll_after": {"unstable": False, "undetermined": False},
    }
    got = P.cross_check(derived, ["stop_generation", "scroll_after", "never_ran"])
    assert got["unstable_but_not_declared"] == ["settings"]
    assert got["declared_stable_in_practice"] == ["scroll_after"]
    assert got["declared_but_never_observed"] == ["never_ran"]


def test_every_declared_unstable_action_carries_a_mechanism():
    # An action silenced without a stated reason is a hole nobody can audit later.
    assert P.UNSTABLE_ACTIONS
    for action, mechanism in P.UNSTABLE_ACTIONS.items():
        assert len(mechanism) > 40, f"{action} is silenced without a real mechanism"


def test_the_verdict_tally_counts_not_comparable_as_its_own_outcome():
    got = P.summarise(
        [{"verdict": P.MATCH}, {"verdict": P.DIFFER}, {"verdict": P.NOT_COMPARABLE}, {}]
    )
    assert got == {P.MATCH: 1, P.DIFFER: 1, P.NOT_COMPARABLE: 2}
