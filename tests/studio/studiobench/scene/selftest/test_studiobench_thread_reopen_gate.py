# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""thread_reopen declines page.goto before it runs and waits on the readiness gate, not threadTotal."""

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

TOTAL = 18
MARKER = "studiobench turn 8: continue with unit 3"


@dataclass(frozen = True)
class _Frame:
    """One rebuild frame; setsize is published on the first frame, while mounted rows arrive later."""

    mounted: int
    elements: int
    scroll_height: int
    spans: int
    marker: bool
    setsize: int = TOTAL


#: Frame 3 is fully mounted but still highlighting, so the mount only settles at frame 4.
REBUILD = (
    _Frame(mounted = 3, elements = 1_200, scroll_height = 2_000, spans = 0, marker = False),
    _Frame(mounted = 9, elements = 4_800, scroll_height = 6_400, spans = 120, marker = False),
    _Frame(mounted = 18, elements = 9_700, scroll_height = 12_000, spans = 4_210, marker = True),
    _Frame(mounted = 18, elements = 11_900, scroll_height = 12_400, spans = 8_940, marker = True),
    _Frame(mounted = 18, elements = 11_900, scroll_height = 12_400, spans = 8_940, marker = True),
)

#: A rebuild that never arrives: total declared as 18 but only 3 mounted.
STALLED = (_Frame(mounted = 3, elements = 1_200, scroll_height = 2_000, spans = 0, marker = False),)

#: Windowed rebuild; `full` conditions would refuse it forever.
WINDOWED = (
    _Frame(mounted = 2, elements = 900, scroll_height = 11_800, spans = 0, marker = False),
    _Frame(mounted = 6, elements = 3_400, scroll_height = 12_400, spans = 2_100, marker = True),
    _Frame(mounted = 6, elements = 3_400, scroll_height = 12_400, spans = 2_100, marker = True),
)


class _ThreadPage:
    """Fake page cycling thread, gone and rebuild phases; each click or goto is recorded."""

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

    @property
    def frame(self) -> _Frame:
        return self.frames[min(self.step, len(self.frames) - 1)]

    def thread_total(self) -> int:
        return {"thread": self.total, "gone": 0}.get(self.phase, self.frame.setsize)

    def message_count(self) -> int:
        return {"thread": self.mounted, "gone": 0}.get(self.phase, self.frame.mounted)

    def _probe(self) -> dict:
        """Mounted rows are numbered as a window anchored at the end, setsize - mounted + 1 through
        setsize."""
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

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "probe_attempted" in script:
            return self._probe()
        if "threadTotal" in script:
            return self.thread_total()
        if "messageCount" in script:
            return self.message_count()
        if '[data-role="user"]' in script:
            return MARKER if self.phase != "gone" else None
        if "pre span" in script:
            return self.frame.spans if self.phase == "rebuild" else 0
        # None sends `_click_or_navigate` down the fallback branch under test.
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
        # Real sleep: the wait is bounded on the monotonic clock, so an instant poll would spin.
        time.sleep(0.002)
        if self.phase == "rebuild":
            self.step += 1

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


def test_a_refused_reopen_leaves_the_thread_where_it_found_it():
    """A refused reopen must leave the thread on screen, since later slots would run on an empty chat."""
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
    """On the way back a substituted navigation repairs the scene but is still reported NOT RUN."""
    page = _ThreadPage(unclickable = {SIDEBAR_ROW})
    result = A.thread_reopen(_ctx(page))

    assert result.ran is False
    assert result.timings == {}
    assert "never timed" in (result.reason or "")
    assert page.goto_calls == [THREAD_URL]
    assert page.phase == "rebuild", "the thread was not put back for the slots that follow"


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
    # Frame 4 is the first end-present and settled frame, so at least five probes ran.
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


#: Scaled-down stand-in for the 2000 ms centre-click retry in `_click_or_navigate`.

RETRY_MS = 400


class _HoverRevealedPage(_ThreadPage):
    """New chat is hover-revealed, so at rest every hit test misses it and only the hover path clicks."""

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
    """Close and reopen clocks start at the click that worked, so Playwright's hover retry is not timed."""
    page = _HoverRevealedPage(slow = {NEW_CHAT, SIDEBAR_ROW})
    result = A.thread_reopen(_ctx(page))

    assert result.ran is True, result.reason
    assert page.goto_calls == [], "the click path was available and should not have been replaced"
    close_ms = result.timings["close_ms"]
    reopen_ms = result.timings["reopen_ms"]
    assert close_ms < RETRY_MS, f"the retry is still inside close_ms ({close_ms}ms)"
    assert reopen_ms < RETRY_MS, f"the retry is still inside reopen_ms ({reopen_ms}ms)"
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
