# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two `thread_reopen` defects left behind by workspace task #102, held closed.

DEFECT ONE: REFUSING THE READING BUT NOT THE DAMAGE. Task #102 taught the action to report NOT RUN
when the New chat control could not be clicked and `page.goto` was substituted, because a document
navigation is not the client-side subtree rebuild this action exists to time. It detected that
AFTER the fact, though, by reading `path == "navigate"` once the goto had already run -- so the row
was honest and the scene was wrecked: every slot after this one carried on from an empty new chat,
and `delete_message`, the last slot of every film, found no messages and went unexercised for a
reason that had nothing to do with deleting. The substitution is now declined BEFORE it happens.

DEFECT TWO: A DECLARATION READ AS A REBUILD. The completion condition was
`threadTotal() >= before`. On a windowed arm `threadTotal()` is `aria-setsize`, the store's claim
about how long the conversation is, and the FIRST reopened row publishes it -- so the condition was
satisfied with three of eighteen messages mounted, no final assistant content and no syntax
highlighting. The action recorded that as `reopen_ms`, took its census off a half-built DOM, and
passed its own assertion because `after == before` compared the same declared total with itself.
The wait is now runtime/readiness.py's own gate: the end of the conversation mounted, and the mount
settled.

Both are conditions on what the action DOES with what the page reports, so both are testable
against a scripted page and neither needs a browser.

    python -m pytest tests/studio/studiobench/scene/selftest/test_studiobench_thread_reopen_gate.py -q
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.runtime.types import ActionContext, Cell  # noqa: E402
from studiobench.scene import actions as A  # noqa: E402

NEW_CHAT = 'button[aria-label="New chat"]'
SIDEBAR_ROW = '[data-thread-id="t1"]'
BASE_URL = "http://127.0.0.1:1"
THREAD_URL = f"{BASE_URL}/chat?thread=t1"
NEW_CHAT_URL = f"{BASE_URL}/chat?new=studiobench"

#: The length of the seeded thread, and the marker the seeder wrote into its last user turn.
TOTAL = 18
MARKER = "studiobench turn 8: continue with unit 3"


@dataclass(frozen = True)
class _Frame:
    """One moment of the rebuilt thread, as the page would report it.

    A script of these IS the defect: `setsize` is the store's declared length and it is published
    on the very first frame, while `mounted`, `elements` and `spans` are what has actually been
    built and arrive over the following ones.
    """

    mounted: int
    elements: int
    scroll_height: int
    spans: int
    marker: bool
    setsize: int = TOTAL


#: A rebuild that declares its full length immediately and then takes four more frames to become a
#: thread. Frame 3 has the whole conversation mounted but is still highlighting, so `elements` is
#: still moving and the mount is not settled until frame 4 repeats it.
REBUILD = (
    _Frame(mounted = 3, elements = 1_200, scroll_height = 2_000, spans = 0, marker = False),
    _Frame(mounted = 9, elements = 4_800, scroll_height = 6_400, spans = 120, marker = False),
    _Frame(mounted = 18, elements = 9_700, scroll_height = 12_000, spans = 4_210, marker = True),
    _Frame(mounted = 18, elements = 11_900, scroll_height = 12_400, spans = 8_940, marker = True),
    _Frame(mounted = 18, elements = 11_900, scroll_height = 12_400, spans = 8_940, marker = True),
)

#: A rebuild that never arrives: the store keeps declaring eighteen messages and the thread stops
#: at three of them, settled, with the end of the conversation nowhere on screen. The exact shape
#: the old condition scored as a fast, successful re-open.
STALLED = (_Frame(mounted = 3, elements = 1_200, scroll_height = 2_000, spans = 0, marker = False),)

#: A correctly windowed rebuild: a window of six rows out of eighteen, anchored at the end, with
#: the last user turn among them. `full` conditions would refuse this forever, which is why the
#: action picks its mode from the mount it left rather than assuming one.
WINDOWED = (
    _Frame(mounted = 2, elements = 900, scroll_height = 11_800, spans = 0, marker = False),
    _Frame(mounted = 6, elements = 3_400, scroll_height = 12_400, spans = 2_100, marker = True),
    _Frame(mounted = 6, elements = 3_400, scroll_height = 12_400, spans = 2_100, marker = True),
)


class _ThreadPage:
    """A page with a thread on it that can be left and reopened, one frame per poll.

    Three phases, because the action's whole job is to tell them apart: `thread` (the seeded thread
    is on screen), `gone` (the New chat route, nothing mounted) and `rebuild` (walking `frames`).
    Every way of moving between them -- a click, or a `page.goto` -- is recorded, so a test can ask
    what the action DID to the scene and not only what it reported.
    """

    def __init__(
        self,
        *,
        frames = REBUILD,
        unclickable = (),
        mounted = TOTAL,
        total = TOTAL,
    ):
        self.frames = tuple(frames)
        self.unclickable = set(unclickable)
        self.mounted = mounted
        self.total = total
        self.phase = "thread"
        self.step = 0
        self.goto_calls: list[str] = []
        self.probes = 0

    # ── what the page would report ──────────────────────────────────

    @property
    def frame(self) -> _Frame:
        return self.frames[min(self.step, len(self.frames) - 1)]

    def thread_total(self) -> int:
        return {"thread": self.total, "gone": 0}.get(self.phase, self.frame.setsize)

    def message_count(self) -> int:
        return {"thread": self.mounted, "gone": 0}.get(self.phase, self.frame.mounted)

    def _probe(self) -> dict:
        """One reading in the shape runtime/readiness.py's PROBE_JS returns.

        The mounted rows are numbered as a window ANCHORED AT THE END of the conversation --
        `setsize - mounted + 1` through `setsize` -- because that is what the gate requires of a
        windowed arm, and numbering them any other way would be modelling a virtualizer bug rather
        than the thread this action leaves and comes back to.
        """
        self.probes += 1
        f = self.frame
        mounted = self.message_count()
        return {
            "probe_attempted": True,
            "mounted": mounted,
            "elements": f.elements,
            "composer": True,
            "running": False,
            "setsize": f.setsize,
            "setsize_values": [f.setsize],
            "posinset_count": mounted,
            "posinset_distinct": mounted,
            "min_posinset": max(1, f.setsize - mounted + 1) if mounted else None,
            "max_posinset": f.setsize if mounted else None,
            "marker_found": f.marker,
            "marker_from_end": 1 if f.marker else None,
            "last_role": "assistant",
            "last_tail": "...",
            "scroll_height": f.scroll_height,
            "client_height": 900,
            "scroll_top": f.scroll_height - 900,
            "from_bottom": 0,
            "viewport_present": True,
            "jump_button_present": True,
            "app_says_at_bottom": True,
            "pinning": False,
        }

    # ── the Playwright surface the action uses ──────────────────────

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "probe_attempted" in script:  # readiness.PROBE_JS
            return self._probe()
        if "threadTotal" in script:
            return self.thread_total()
        if "messageCount" in script:
            return self.message_count()
        if '[data-role="user"]' in script:
            return MARKER if self.phase != "gone" else None
        if "pre span" in script:
            return self.frame.spans if self.phase == "rebuild" else 0
        # The hit-test spread and the hover target, both of which report "no reachable point" for
        # an unclickable control. Returning None here is what sends `_click_or_navigate` down the
        # branch under test rather than into an off-centre click.
        return None

    def query_selector(self, selector):
        page = self

        class _Handle:
            def click(self, timeout = None):
                if selector in page.unclickable:
                    raise TimeoutError(f"{selector} is not clickable")
                page._route(selector)

        return _Handle()

    def goto(self, url, **_kwargs):
        self.goto_calls.append(url)
        self._route(url)

    def wait_for_timeout(self, _ms):
        # A REAL, TINY SLEEP. The readiness wait is bounded on the monotonic clock, so a poll that
        # returned instantly would spin thousands of times inside the timeout instead of walking
        # the scripted frames.
        time.sleep(0.002)
        if self.phase == "rebuild":
            self.step += 1

    # ── phases ──────────────────────────────────────────────────────

    def _route(self, target: str) -> None:
        if "New chat" in target or "new=" in target:
            self.phase = "gone"
        elif "thread-id" in target or "thread=" in target:
            self.phase = "rebuild"
            self.step = 0


def _ctx(
    page,
    log = None,
    budget_ms = 30_000,
) -> ActionContext:
    return ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r100K.base.rep0", rung = "100K", rung_tokens = 100_000),
        window = None,
        args = {"thread_id": "t1", "base_url": BASE_URL},
        budget_ms = budget_ms,
        dom = None,
        log = log or (lambda _m: None),
    )


# ── defect one: the fallback is declined, not detected ──────────────


def test_a_refused_reopen_leaves_the_thread_where_it_found_it():
    """THE COLLATERAL DAMAGE, asserted as damage rather than as a row.

    The action still reports NOT RUN, and the page must still be showing the thread afterwards.
    Before the fix `page.goto` had already run by the time the refusal was decided, so the scene
    continued from an empty new chat and every later slot measured that instead.
    """
    page = _ThreadPage(unclickable = {NEW_CHAT})
    result = A.thread_reopen(_ctx(page))

    assert result.ran is False
    assert result.timings == {}
    assert page.goto_calls == [], "the invalid fallback navigation was performed anyway"
    assert (
        page.phase == "thread"
    ), "the scene was left on the new-chat page for the slots that follow"
    assert page.message_count() == TOTAL, "the following slots inherited an empty thread"


def test_the_refusal_still_says_why_in_the_row_and_in_the_log():
    """Declining the substitution must not make the refusal quieter than it was."""
    said: list[str] = []
    page = _ThreadPage(unclickable = {NEW_CHAT})
    result = A.thread_reopen(_ctx(page, said.append))

    assert "not a thread rebuild" in (result.reason or "")
    assert "no navigation was performed" in (result.reason or "")
    assert any("NOT MEASURED" in line for line in said), said


def test_click_or_navigate_declines_the_substitute_when_the_caller_refuses_it():
    """The contract the caller relies on: no goto, `ok = False`, and the click failure explained."""
    page = _ThreadPage(unclickable = {NEW_CHAT})
    got = A._click_or_navigate(_ctx(page), NEW_CHAT, NEW_CHAT_URL, allow_navigate = False)

    assert got.ok is False
    assert got.path == "failed"
    assert got.navigated is False
    assert "not clickable" in (got.reason or "")
    assert page.goto_calls == []


def test_refusing_the_substitute_does_not_refuse_the_click():
    """`allow_navigate` governs the FALLBACK only. A control that can be clicked is still clicked,
    which is the path every successful run takes."""
    page = _ThreadPage()
    got = A._click_or_navigate(_ctx(page), NEW_CHAT, NEW_CHAT_URL, allow_navigate = False)

    assert got.ok is True
    assert got.path == "click"
    assert page.phase == "gone", "the app's own route change never happened"


def test_the_default_still_navigates_for_every_other_caller():
    """The signature gained a keyword and must not have changed what anyone else gets. The reopen
    half of `thread_reopen` depends on this: from an empty new chat the navigation is what puts the
    thread back for the slots that follow."""
    page = _ThreadPage(unclickable = {NEW_CHAT})
    got = A._click_or_navigate(_ctx(page), NEW_CHAT, NEW_CHAT_URL)

    assert got.ok is True
    assert got.path == "navigate"
    assert page.goto_calls == [NEW_CHAT_URL]


def test_a_substituted_navigation_on_the_way_back_repairs_the_scene_but_is_not_timed():
    """THE DELIBERATE ASYMMETRY. Leaving, a navigation is the thing that breaks the scene and is
    refused. Returning, it is what puts the thread back on screen for the slots that follow, so it
    is allowed to stand -- and the action still reports NOT RUN, with no timing, because a document
    reload is still not a rebuild."""
    page = _ThreadPage(unclickable = {SIDEBAR_ROW})
    result = A.thread_reopen(_ctx(page))

    assert result.ran is False
    assert result.timings == {}
    assert "never timed" in (result.reason or "")
    assert page.goto_calls == [THREAD_URL]
    assert page.phase == "rebuild", "the thread was not put back for the slots that follow"


# ── defect two: a declared total is not a finished rebuild ──────────


def test_the_scripted_rebuild_declares_its_total_before_it_has_built_anything():
    """WITHOUT THIS THE TEST BELOW PROVES NOTHING. If the first frame did not already publish
    `aria-setsize = 18`, the old condition would have waited too and both would pass."""
    first = REBUILD[0]
    assert first.setsize == TOTAL
    assert first.mounted == 3
    assert first.marker is False and first.spans == 0


def test_reopen_waits_for_the_thread_to_be_rebuilt_not_for_it_to_be_declared():
    """The reading the old condition produced came off frame 0: three of eighteen rows, no end of
    conversation, no highlighting. Every observation below is taken after the wait, so they are the
    assertion that the wait outlasted the declaration."""
    page = _ThreadPage()
    result = A.thread_reopen(_ctx(page))

    assert result.ran is True
    assert result.expect_ok is True
    assert result.expect["mounted_after"] == 18, "the census was taken off a partly built thread"
    assert result.expect["highlight_spans_after"] == 8_940, "the fences had not been highlighted"
    # Frame 4 is the earliest one that is both end-present and settled, so nothing short of five
    # probes can have satisfied the gate.
    assert page.probes >= 5, page.probes
    assert result.timings["reopen_ms"] is not None


def test_the_row_carries_the_gate_the_timing_was_taken_against():
    """`reopen_ms` means "until ready", so the row has to say which definition of ready and how it
    was reached. Two arms in different modes are otherwise silently incomparable."""
    page = _ThreadPage()
    result = A.thread_reopen(_ctx(page))

    assert result.expect["reopen_ready_mode"] == "full"
    readiness = result.expect["reopen_readiness"]
    assert readiness["ready"] is True
    assert readiness["conditions"]["end_present"] is True
    assert readiness["conditions"]["settled"] is True
    assert readiness["conditions"]["all_messages_mounted"] is True


def test_a_thread_that_declares_its_total_and_never_rebuilds_gets_no_timing(monkeypatch):
    """The old condition's worst case, and the one that made the defect invisible: the store
    publishes eighteen, three rows mount, nothing else ever arrives. That was scored as a fast
    re-open. It is now a run with no timing and the outstanding condition named."""
    monkeypatch.setattr(A, "_REOPEN_READY_CEILING_S", 0.4)
    page = _ThreadPage(frames = STALLED)
    result = A.thread_reopen(_ctx(page))

    assert result.ran is True
    assert result.timings["reopen_ms"] is None, "a half-built thread reported a rebuild time"
    assert result.expect_ok is False
    readiness = result.expect["reopen_readiness"]
    assert readiness["ready"] is False
    assert readiness["conditions"]["end_present"] is False
    assert readiness["conditions"]["all_messages_mounted"] is False
    assert "never reached a ready state" in (result.reason or "")


def test_a_windowed_arm_is_held_to_the_windowed_gate_and_not_to_a_full_mount():
    """The other way this fix could have been wrong. A windowed arm never mounts every message, so
    waiting for a full mount would time out on the arm the whole comparison exists to score. The
    mode is read from the mount the action LEFT, exactly as the cell's opening gate read it."""
    page = _ThreadPage(frames = WINDOWED, mounted = 6, total = TOTAL)
    result = A.thread_reopen(_ctx(page))

    assert result.expect["reopen_ready_mode"] == "windowed"
    assert result.ran is True
    assert result.expect_ok is True
    assert result.expect["mounted_after"] == 6
    assert result.expect["messages_after"] == TOTAL
    conditions = result.expect["reopen_readiness"]["conditions"]
    assert conditions["end_present"] is True
    assert conditions["anchored_at_end"] is True
    assert conditions["total_matches_seeded"] is True


def test_a_thread_whose_end_cannot_be_identified_is_refused_before_it_is_touched():
    """No marker, no way to tell a rebuilt thread from a half-rebuilt one -- so the action refuses,
    and refuses early enough that the thread it cannot verify is also one it has not disturbed."""

    class _NoMarker(_ThreadPage):
        def evaluate(
            self,
            script,
            arg = None,
        ):
            if '[data-role="user"]' in script:
                return None
            return super().evaluate(script, arg)

    page = _NoMarker()
    result = A.thread_reopen(_ctx(page))

    assert result.ran is False
    assert "identify the end of the thread" in (result.reason or "")
    assert page.phase == "thread" and page.goto_calls == []


# ── defect three: the harness's own retry, billed to the rebuild ────

#: How long the centre click burns before it gives up. `_click_or_navigate` passes
#: `timeout = 2000` to Playwright, and the hit-target check retries for the whole of it against a
#: control it cannot hit at the centre. Scaled down here so the test is fast; what is asserted is
#: that the timings do not contain it, at whatever size it is.
RETRY_MS = 400


class _HoverRevealedPage(_ThreadPage):
    """A page whose controls behave the way the sidebar's actually do.

    `.sidebar-header-action` ships `opacity-0 pointer-events-none` and is revealed by its group's
    `:hover`, so every hit test at rest falls through to the group underneath: `handle.click` waits
    out its whole actionability timeout, and the off-centre/hover path is the one that works. That
    is not an edge case, it is the documented behaviour of the New chat button on every run.
    """

    def __init__(
        self,
        *,
        slow = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.slow = set(slow)
        self.moves: list[tuple[float, float]] = []
        self._pending: str | None = None

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "elementFromPoint" in script or "getBoundingClientRect" in script:
            selector = arg[0] if isinstance(arg, list) else arg
            self._pending = selector
            return {"x": 12.0, "y": 34.0}
        return super().evaluate(script, arg)

    def query_selector(self, selector):
        page = self

        class _Handle:
            def click(self, timeout = None):
                if selector in page.slow:
                    # The retry, faithfully: time passes and then it fails.
                    time.sleep(RETRY_MS / 1000)
                    raise TimeoutError(f"{selector} was not clickable in {timeout}ms")
                page._route(selector)

        return _Handle()

    @property
    def mouse(self):
        page = self

        class _Mouse:
            def move(self, x, y):
                page.moves.append((x, y))

            def click(self, x, y):
                page.moves.append((x, y))
                if page._pending is not None:
                    page._route(page._pending)

        return _Mouse()


def test_the_failed_click_retry_is_not_charged_to_the_close_or_the_rebuild():
    """THE DEFECT. `close_ms` and `reopen_ms` were clocked from before the FIRST click attempt, so
    Playwright's 2,000 ms hit-target retry against the hover-revealed New chat button landed inside
    them -- on every run, since that control is hover-revealed by design. sweep/floor_table.py
    harvests both as quotable metrics, so two seconds of harness retry was being compared against
    the other arm as though it were the cost of tearing down and rebuilding a thread.

    Both clocks now start at the click that WORKED. The retry is not thrown away, it is named.
    """
    page = _HoverRevealedPage(slow = {NEW_CHAT, SIDEBAR_ROW})
    result = A.thread_reopen(_ctx(page))

    assert result.ran is True, result.reason
    assert page.goto_calls == [], "the click path was available and should not have been replaced"
    close_ms = result.timings["close_ms"]
    reopen_ms = result.timings["reopen_ms"]
    assert close_ms < RETRY_MS, f"the retry is still inside close_ms ({close_ms}ms)"
    assert reopen_ms < RETRY_MS, f"the retry is still inside reopen_ms ({reopen_ms}ms)"
    # ...and it is still in the payload, as the harness's own cost rather than the app's.
    assert result.expect["left_click_retry_ms"] >= RETRY_MS
    assert result.expect["reopen_click_retry_ms"] >= RETRY_MS


def test_a_control_that_clicks_first_time_reports_no_retry_at_all():
    """The control. Nothing about the ordinary path moves, and the new fields read zero rather than
    a small plausible number that a reader would have to interpret."""
    page = _HoverRevealedPage()
    result = A.thread_reopen(_ctx(page))

    assert result.ran is True, result.reason
    assert result.expect["left_click_retry_ms"] < 50
    assert result.expect["reopen_click_retry_ms"] < 50
    assert result.timings["reopen_ms"] > 0
