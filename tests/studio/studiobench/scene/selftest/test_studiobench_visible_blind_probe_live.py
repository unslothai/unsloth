# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""THE STREAMING PROBE'S POSITIVE CONTROL IN VISIBLE MODE, in a real browser.

`compare_visible` scores windowed pairs from `parityVisible.capture()` alone. It never sees the
structural capture, so `in_flight_unplaced` -- the control that catches a `data-status` hook that
has gone quiet -- did not reach it, while the per-row `in_flight` it DOES read walks those exact
selectors.

That gap is asymmetric-by-construction, which is what makes it reachable rather than theoretical.
The two arms of an A/B are two different builds: the merge base and the head. A head that renames
or drops the status hook is precisely the change the control exists to catch, and it is renamed on
ONE arm only, so nothing cancels out. Both rows then read `in_flight: false`, one arm's reply is
mid-tail while the other's has finished, and the per-ordinal loop scores those two points in one
stream as a rendering difference. That is the wall-clock false alarm this change exists to remove,
arriving through the one door it had left open.

Read GLOBALLY and not over the visible rows, which is the other half of the decision: "a reply is
being written" and "the message it is being written into is on screen" are different questions, and
a reply streaming below the fold is an ordinary state that must refuse nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_STUDIO_TESTS = _HERE.parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import parity as P  # noqa: E402

_DOM_JS = _STUDIO_TESTS / "studiobench" / "scene" / "dom.js"
_PARITY_JS = _STUDIO_TESTS / "studiobench" / "scene" / "parity.js"

_STOP_BUTTON = (
    '<button class="aui-composer-cancel" aria-label="Stop generating">'
    '<span class="aui-sr-only">Stop generating</span></button>'
)
_SEND_BUTTON = (
    '<button class="aui-composer-send" aria-label="Send message">'
    '<span class="aui-sr-only">Send message</span></button>'
)


def _page(
    *,
    tail: str,
    tail_running: bool,
    generating: bool,
    hook: str = "data-status",
    running_value: str = "running",
    tall: bool = False,
    user_body: str = "message 3",
    live_role: str = "assistant",
    drop_tail: bool = False,
    tail_has_parts: bool = True,
) -> str:
    """A four-message thread in a viewport that shows all of it.

    `hook` is the attribute the last assistant message publishes its status through. The shipped
    name is `data-status`; anything else is a build whose hook this instrument does not know, which
    is what going blind looks like from the outside.
    """
    rows = []
    for i in range(1, 5):
        if drop_tail and i == 4:
            # The windowed case: the arm still DECLARES four messages through `aria-setsize`, and
            # has unmounted the one it is writing into. That is a thread the instrument can see
            # three quarters of, not an instrument that has stopped working.
            continue
        role = "user" if i % 2 else "assistant"
        last = i == 4
        if last:
            # `live_role` is what the arm CALLS the row it is writing into. The shipped build says
            # assistant; a build that says otherwise is the regression ordinal-role parity exists
            # to catch, and it is asked here rather than assumed.
            role = live_role
        body = tail if last else f"message {i}"
        if i == 3:
            body = user_body
        if last and not tail_has_parts:
            # The gap between the send being accepted and the reply's first part arriving. The
            # assistant message is mounted and publishes nothing, because it has nothing to
            # publish: thread.tsx renders "Generating..." in place of any part.
            rows.append(
                f'<div class="row" aria-posinset="{i}" aria-setsize="4">'
                f'<div data-role="{role}"><span>Generating...</span></div></div>'
            )
            continue
        status = running_value if (last and tail_running) else "complete"
        # THE HOOK IS RENAMED ON EVERY MESSAGE, not only the streamed one. `data-status` is a single
        # line in markdown-text.tsx and it is rendered for `complete` parts as well as `running`
        # ones, so a build that renames it renames it everywhere. A fixture that renamed it on one
        # message would be describing a build that does not exist, and would then be the only
        # evidence that the control fires.
        attr = f'{hook}="{status}"'
        rows.append(
            f'<div class="row" aria-posinset="{i}" aria-setsize="4">'
            f'<div data-role="{role}"><div {attr}>{body}</div></div></div>'
        )
    height = "600px" if tall else "60px"
    return f"""<!doctype html><meta charset="utf-8">
<style>
  body {{ margin: 0; }}
  .aui-thread-viewport {{ height: 400px; overflow-y: auto; }}
  [data-role] {{ height: {height}; }}
</style>
<div class="aui-thread-root">
  <div class="aui-thread-viewport">{"".join(rows)}</div>
  <div class="aui-composer-root">
    <textarea aria-label="Message input"></textarea>
    {_STOP_BUTTON if generating else _SEND_BUTTON}
  </div>
</div>"""


def _skip_reason() -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"playwright is not installed: {exc}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason = _skip_reason() or "")


@pytest.fixture(scope = "module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args = ["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"chromium could not be launched: {exc}")
        yield b
        b.close()


def _capture(browser, **kw) -> dict:
    """One capture, on a PAGE OF ITS OWN.

    `set_content` rewrites the document through `document.write`, which keeps the JS context, so a
    second call finds `window.__sb` already there, skips re-initialising it, and leaves the
    observer bound to the previous document's viewport. The symptom is a capture that reports the
    PREVIOUS page's visible set -- a fixture bug that would read as a finding.
    """
    page = browser.new_page(viewport = {"width": 900, "height": 700})
    try:
        page.set_content(_page(**kw))
        page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        got = page.evaluate("() => window.__sb.parityVisible.watch()")
        assert got.get("visible_attempted") is True, got
        # IntersectionObserver's first delivery is asynchronous.
        page.wait_for_timeout(150)
        cap = page.evaluate("async () => await window.__sb.parityVisible.capture()")
        assert cap.get("visible_attempted") is True, cap
        return cap
    finally:
        page.close()


#: The base arm: an ordinary settled thread on the shipped build.
_SETTLED = dict(tail = "the whole reply, arrived", tail_running = False, generating = False)
#: THE TREATMENT THAT WENT BLIND, and the shape matters. `data-status` is one line in
#: markdown-text.tsx and it is rendered for `complete` parts as well as `running` ones, so a build
#: that renames the ATTRIBUTE renames it on every message and every settled row differs too -- which
#: is a rendering difference in its own right and is reported as one. The interesting blindness is a
#: build that changed the status VOCABULARY: `complete` still reads `complete`, so every settled row
#: is byte-identical, and the only thing lost is the ability to see that a reply is being written.
_BLIND = dict(
    tail = "the whole reply, arr", running_value = "streaming", tail_running = True, generating = True
)
#: The cruder form: the attribute itself is gone. Kept because it is the one blindness a WINDOWED
#: arm can still be caught at, where a missing row is otherwise an ordinary state.
_BLIND_ATTR = dict(
    tail = "the whole reply, arr", hook = "data-state", tail_running = True, generating = True
)
#: The same mid-stream moment on a build whose hook IS known. Already handled, as residue.
_MIDSTREAM = dict(tail = "the whole reply, arr", tail_running = True, generating = True)


def test_a_blinded_treatment_is_refused_not_scored_as_a_rendering_difference(browser):
    """THE REGRESSION, end to end: two points in one stream, scored as a difference."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_BLIND)
    # Same messages on screen, so nothing above the digest comparison can refuse this pair.
    assert base["ever_visible"] == treat["ever_visible"] == [1, 2, 3, 4]
    # And every row on both arms claims to be settled, which is the false statement.
    assert not any(r["in_flight"] for r in base["messages"].values())
    assert not any(r["in_flight"] for r in treat["messages"].values())
    assert base["messages"]["4"]["digest"] != treat["messages"]["4"]["digest"]

    assert treat["in_flight_unplaced"] is True, treat
    assert base["in_flight_unplaced"] is False, base
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "could not be identified" in got["reason"]
    assert P.compare_visible(treat, base)["verdict"] == P.NOT_COMPARABLE


def test_the_same_moment_on_a_build_with_the_hook_is_still_residue(browser):
    """The path that already worked must keep working, and by the route it already took."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_MIDSTREAM)
    assert treat["messages"]["4"]["in_flight"] is True
    assert treat["in_flight_unplaced"] is False, treat
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert got["not_digested"] == [4], got


def test_a_settled_pair_is_not_refused(browser):
    """The coverage this must not cost: nothing is generating, so nothing is withheld."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_SETTLED)
    assert base["streaming"] is False and treat["streaming"] is False
    assert P.compare_visible(base, treat)["verdict"] == P.MATCH


def test_a_reply_streaming_below_the_fold_refuses_nothing(browser):
    """Why the control is read globally rather than over the visible rows.

    Tall messages, so the streamed last one is off screen. It is placed -- the hook is intact and
    `streamingMessages()` finds it -- so the probe is not blind, and a capture that read only the
    rows it could see would refuse a pair for being unable to see something it was never claiming
    to have seen.
    """
    cap = _capture(browser, **dict(_MIDSTREAM, tall = True))
    assert 4 not in cap["ever_visible"], cap["ever_visible"]
    assert cap["streaming"] is True
    assert cap["in_flight_unplaced"] is False, cap


def test_a_lost_conversation_is_still_a_finding_while_a_reply_runs(browser):
    """The ordering, held: the refusal sits AFTER the two lost-conversation findings.

    A treatment that puts different messages on screen is a difference whether or not its stream
    could be placed, exactly as `compare` keeps `mount_count_mismatch` ahead of the same refusal.
    """
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **dict(_BLIND, tall = True))
    assert treat["in_flight_unplaced"] is True
    assert base["ever_visible"] != treat["ever_visible"]
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "DIFFERENT MESSAGES on screen" in got["reason"]


# ── what the refusal may NOT take out with it ────────────────────────
#
# The blind-probe refusal is about rows whose meaning depends on where the stream had got to. Two
# kinds of row in this payload provably do not: a row both arms call the user's, because a reply is
# written into an assistant message; and a row whose ROLE changed, because a role is captured beside
# the digest and how far a reply has arrived says nothing about whose it is. Both used to leave here
# as NOT COMPARABLE with an empty `moved`, and `visible_report` buckets a refusal as blind and never
# consults it for the exit code, so the run went green on them.


def test_a_changed_user_row_survives_the_blind_refusal(browser):
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **dict(_BLIND, user_body = "the user message, rewritten"))
    assert treat["in_flight_unplaced"] is True, treat
    # Row 3 is the user's on both arms and differs; row 4 is the one that cannot be read.
    assert base["messages"]["3"]["role"] == treat["messages"]["3"]["role"] == "user"
    assert base["messages"]["3"]["digest"] != treat["messages"]["3"]["digest"]
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert [m for m in got["moved"] if m.startswith("ordinal 3(user)")] == got["moved"], got[
        "moved"
    ]
    assert "cannot be the reply being written" in got["reason"]


def test_the_blind_refusal_still_covers_the_assistant_rows(browser):
    """The narrowing must not become a hole: only the provable rows come through."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_BLIND)
    assert treat["in_flight_unplaced"] is True
    assert base["messages"]["4"]["digest"] != treat["messages"]["4"]["digest"]
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["moved"] == []


def test_a_role_change_on_the_live_row_is_reported_not_elided(browser):
    """The row is in flight on the treatment arm, so its DIGEST is withheld. Its role is not.

    The digest and the role are two separate readings of the same row, and only one of them names a
    point in a stream. Eliding both meant a treatment that renders the live assistant row as the
    user's came back NOT COMPARABLE.
    """
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **dict(_MIDSTREAM, live_role = "user"))
    assert treat["messages"]["4"]["in_flight"] is True, treat["messages"]["4"]
    assert base["messages"]["4"]["role"] == "assistant"
    assert treat["messages"]["4"]["role"] == "user"
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "ordinal 4:role assistant->user" in got["moved"], got["moved"]


def test_the_same_row_with_the_same_role_is_still_residue(browser):
    """The control on the test above: without the role change it stays a refusal."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_MIDSTREAM)
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE
    assert got["not_digested"] == [4], got


# ── a row that is not there is not an instrument that stopped working ─


def test_a_windowed_arm_that_unmounted_the_live_row_is_not_read_as_blind(browser):
    """THE FALSE POSITIVE THE CONTROL USED TO HAVE, and the one this mode exists for.

    `streamingMessages()` scans MOUNTED DOM. A windowed arm scrolled away from the tail unmounts the
    message it is writing into -- that is the whole point of windowing -- so the scan returns
    nothing on a build whose hooks are perfectly intact. Refusing there discards the settled rows
    that WERE on screen, which is the coverage visible mode is for.

    The arm still declares four messages through `aria-setsize`, so the capture can tell the
    difference between a thread it can see part of and a thread that has nothing to say.
    """
    cap = _capture(browser, **dict(_MIDSTREAM, drop_tail = True))
    assert cap["streaming"] is True
    assert cap["status_hook_present"] is True
    assert 4 not in cap["ever_visible"], cap["ever_visible"]
    assert cap["in_flight_unplaced"] is False, cap


def test_a_windowed_arm_is_still_caught_when_the_hook_itself_is_gone(browser):
    """The narrowing is not a hole. A missing ROW explains a quiet scan; a missing ATTRIBUTE does
    not, because the settled rows would still be publishing it."""
    cap = _capture(browser, **dict(_BLIND_ATTR, drop_tail = True))
    assert cap["status_hook_present"] is False
    assert cap["in_flight_unplaced"] is True, cap


def test_a_full_mount_is_still_caught_when_only_the_STATUS_VALUE_changed(browser):
    """And the reverse: on a full mount the row cannot be missing, so a quiet scan is blindness
    even though the attribute is still there. This is the case a hook-presence test alone would
    have lost, and it is the likelier build change of the two."""
    cap = _capture(browser, **_BLIND)
    assert cap["status_hook_present"] is True
    assert cap["in_flight_unplaced"] is True, cap


def test_what_the_windowed_narrowing_gives_up(browser):
    """Pinned rather than left implicit: a WINDOWED arm whose status VALUE changed is not caught.

    The row it is writing into may legitimately be absent, and the attribute is still published by
    the settled rows, so nothing distinguishes it from an ordinary windowed capture. It under-claims
    rather than over-claims, and the same build compared on any full-mount pair still trips.
    """
    cap = _capture(browser, **dict(_BLIND, drop_tail = True))
    assert cap["status_hook_present"] is True
    assert cap["in_flight_unplaced"] is False, cap


def test_the_gap_before_the_first_part_arrives_is_not_a_blind_probe(browser):
    """THE THIRD CAUSE, and `send_turn` returns exactly into it.

    `send_turn` breaks the instant `isRunning()` flips (scene/actions.py), the window closes there
    and the capture follows within milliseconds. At that moment the assistant message is mounted
    with ZERO content parts -- thread.tsx renders "Generating..." in place of any part -- so it
    publishes no status because it has none to publish. The older assistant messages still publish
    theirs, which is what says the hook is intact and this is an ordinary interval.
    """
    cap = _capture(browser, **dict(_MIDSTREAM, tail_has_parts = False))
    assert cap["streaming"] is True
    assert cap["status_hook_present"] is True
    assert cap["in_flight_unplaced"] is False, cap


def test_that_gap_is_still_caught_if_no_message_publishes_a_status_at_all(browser):
    """The other half: nothing to publish on the LAST message is ordinary, nothing to publish
    ANYWHERE is a hook that is gone, and a settled message would still be carrying it."""
    cap = _capture(browser, **dict(_BLIND_ATTR, tail_has_parts = False))
    assert cap["status_hook_present"] is False
    assert cap["in_flight_unplaced"] is True, cap
