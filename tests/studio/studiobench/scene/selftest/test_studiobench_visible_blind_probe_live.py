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
    hook: str,
    tail_running: bool,
    generating: bool,
    tall: bool = False,
) -> str:
    """A four-message thread in a viewport that shows all of it.

    `hook` is the attribute the last assistant message publishes its status through. The shipped
    name is `data-status`; anything else is a build whose hook this instrument does not know, which
    is what going blind looks like from the outside.
    """
    rows = []
    for i in range(1, 5):
        role = "user" if i % 2 else "assistant"
        last = i == 4
        body = tail if last else f"message {i}"
        status = "running" if (last and tail_running) else "complete"
        attr = f'{hook}="{status}"' if last else f'data-status="{status}"'
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
_SETTLED = dict(
    tail = "the whole reply, arrived", hook = "data-status", tail_running = False, generating = False
)
#: The treatment arm: its reply is still being written and its status hook is not one this
#: instrument knows, so every row reads settled.
_BLIND = dict(tail = "the whole reply, arr", hook = "data-state", tail_running = True, generating = True)
#: The same mid-stream moment on a build whose hook IS known. Already handled, as residue.
_MIDSTREAM = dict(
    tail = "the whole reply, arr", hook = "data-status", tail_running = True, generating = True
)


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
