# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""compare_visible must check in_flight_unplaced globally, since per-row checks miss a renamed hook."""

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
    """The status hook the last assistant message publishes; a renamed hook is what blindness looks like."""
    rows = []
    for i in range(1, 5):
        if drop_tail and i == 4:
            # Windowed: the arm declares four messages via `aria-setsize` but unmounted the live one.
            continue
        role = "user" if i % 2 else "assistant"
        last = i == 4
        if last:
            role = live_role
        body = tail if last else f"message {i}"
        if i == 3:
            body = user_body
        if last and not tail_has_parts:
            # Before the first part, thread.tsx renders "Generating...", so nothing is published.
            rows.append(
                f'<div class="row" aria-posinset="{i}" aria-setsize="4">'
                f'<div data-role="{role}"><span>Generating...</span></div></div>'
            )
            continue
        status = running_value if (last and tail_running) else "complete"
        # `data-status` is rendered for complete parts too, so a rename applies to every message.
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
    """Fresh page per capture: set_content keeps window.__sb, which pins the old observer's viewport."""
    page = browser.new_page(viewport = {"width": 900, "height": 700})
    try:
        page.set_content(_page(**kw))
        page.add_script_tag(content = _DOM_JS.read_text(encoding = "utf-8"))
        page.add_script_tag(content = _PARITY_JS.read_text(encoding = "utf-8"))
        got = page.evaluate("() => window.__sb.parityVisible.watch()")
        assert got.get("visible_attempted") is True, got
        page.wait_for_timeout(150)
        cap = page.evaluate("async () => await window.__sb.parityVisible.capture()")
        assert cap.get("visible_attempted") is True, cap
        return cap
    finally:
        page.close()


_SETTLED = dict(tail = "the whole reply, arrived", tail_running = False, generating = False)
#: Blind via a changed status vocabulary: settled rows stay identical, only in-flight is lost.
_BLIND = dict(
    tail = "the whole reply, arr", running_value = "streaming", tail_running = True, generating = True
)
#: Attribute removed: the one blindness still detectable on a windowed arm.
_BLIND_ATTR = dict(
    tail = "the whole reply, arr", hook = "data-state", tail_running = True, generating = True
)
_MIDSTREAM = dict(tail = "the whole reply, arr", tail_running = True, generating = True)


def test_a_blinded_treatment_is_refused_not_scored_as_a_rendering_difference(browser):
    """THE REGRESSION, end to end: two points in one stream, scored as a difference."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **_BLIND)
    assert base["ever_visible"] == treat["ever_visible"] == [1, 2, 3, 4]
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
    """Streaming below the fold must refuse nothing: a capture is not blamed for rows it never claimed."""
    cap = _capture(browser, **dict(_MIDSTREAM, tall = True))
    assert 4 not in cap["ever_visible"], cap["ever_visible"]
    assert cap["streaming"] is True
    assert cap["in_flight_unplaced"] is False, cap


def test_a_lost_conversation_is_still_a_finding_while_a_reply_runs(browser):
    """A lost conversation is still a finding while a reply runs; the blind refusal comes after it."""
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **dict(_BLIND, tall = True))
    assert treat["in_flight_unplaced"] is True
    assert base["ever_visible"] != treat["ever_visible"]
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "DIFFERENT MESSAGES on screen" in got["reason"]


# User rows and rows whose role changed do not depend on stream progress, so they must still
# be reported instead of being swallowed by the blind-probe refusal.


def test_a_changed_user_row_survives_the_blind_refusal(browser):
    base = _capture(browser, **_SETTLED)
    treat = _capture(browser, **dict(_BLIND, user_body = "the user message, rewritten"))
    assert treat["in_flight_unplaced"] is True, treat
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
    """A live row's digest is withheld but its role is still compared, so a role flip is not hidden."""
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


def test_a_windowed_arm_that_unmounted_the_live_row_is_not_read_as_blind(browser):
    """An unmounted live row is not a blind probe; the streaming scan only sees mounted DOM."""
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
    """A windowed capture whose status value changed is not caught, since its live row may be unmounted."""
    cap = _capture(browser, **dict(_BLIND, drop_tail = True))
    assert cap["status_hook_present"] is True
    assert cap["in_flight_unplaced"] is False, cap


def test_the_gap_before_the_first_part_arrives_is_not_a_blind_probe(browser):
    """A new reply with no parts yet has no status to publish, so the gap is not a blind probe."""
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
