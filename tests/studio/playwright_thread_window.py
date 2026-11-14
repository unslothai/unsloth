# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A long thread mounts only the rows near the reader, and nothing the reader can do shows it.

smoke-thread-weight.html?window=1 is the thread as the app opens it (without the flag the page
holds every row mounted). Structure only, never a timing:

  1. The DOM does not grow with the chat: a 400-message thread mounts about what an 80-message
     one does, and its rows publish the thread's length and their place (aria-setsize,
     aria-posinset), so assistive technology does not count only what is mounted.
  2. The scroll range is the fully mounted thread's, within EXTENT_TOLERANCE, before a single row
     above the fold has rendered.
  3. Wheeling to the top, the window catches up with the reader after every step, passes every
     message once, and ends on the first one.
  4. A reader at rest is not moved by the window settling behind them.
  5. A jump (a dragged scrollbar) lands on mounted rows.
  6. A row the reader is working in stays mounted, with its focus, wherever they jump to, apart
     from the window and without the rows between; it is let go once they leave it.
  7. Copying a selection that runs over unmounted rows carries the whole conversation, the same
     text the fully mounted thread selects, and the window comes back once the selection is gone.
  8. A thread hidden behind another route mounts nothing while it has no boxes to measure, and a
     full mount it was in the middle of does not count as pending for the threads on screen.
  9. A reply streamed into the windowed thread lands at the tail with the reader following it.
 10. Mounting everything (select all) goes in passes bounded by height, so very long replies do
     not land thirty-two to a commit.
 11. The window at the tail holds the last prompt too, and a prompt being edited (a row with no
     `data-role`) keeps its editor, focus and draft across jumps to both ends.
 12. The leading spacer carries the KaTeX equation numbers of the rows it stands for, so the
     page-wide numbering of the mounted rows is the fully mounted thread's.
 13. A code block's own word wrap survives its row being unmounted and mounted again.
 14. A reader who scrolls away while a reply streams has rows under them before it ends: the
     stream does not hold the window's commits back.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5219"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
ENGINES = [e for e in os.environ.get("SMOKE_ENGINES", "chromium").split(",") if e]
SMALL = 80
LARGE = 400
STREAM_CHUNKS = 20
LONG_STREAM_CHUNKS = 600
AWAY_WHEEL_STEPS = 12
# Rows slower to render than the gap between two streamed chunks.
AWAY_SEED = {"agent": {"rounds": 5, "compactionEvery": 10, "sources": True, "rich": True}}
# Frames, not a time: a row the reader reaches is committed inside the frame that shows it.
AWAY_FRAMES = 20
DRAFT = "unsaved draft for the windowed thread"
SEED = {"agent": {"rounds": 2, "compactionEvery": 10, "sources": True, "rich": True}}
# Same allowance studiobench gives a windowed arm (analysis/behaviour.py).
EXTENT_TOLERANCE = 0.10
# Rows the larger thread may mount beyond the smaller one: a row of slack either side.
ROW_SLACK = 4
WHEEL_PX = 700
WHEEL_GAP_MS = 40
# Rows are separated by gaps, so a covered viewport is not rows from edge to edge.
MIN_COVERAGE = 0.5
# How long the window may take to catch up with one wheel step before it counts as lost. Not a
# pace: a shared CI runner serving unminified modules takes whole frames for a commit.
CATCH_UP_MS = 30_000
# Very long replies, for the full mount: a few thousand px each.
TALL_MESSAGES = 60
TALL_SEED = {"agent": {"rounds": 12, "compactionEvery": 10, "sources": True, "rich": True}}
# thread-window.ts FILL_BUDGET_PX and progressive-mount-controller.ts CHUNK_MESSAGES. The budget
# is spent on estimated heights, so a pass may come out somewhat over it once rendered.
FILL_BUDGET_PX = 20_000
FILL_BUDGET_SLACK = 1.5
FILL_CHUNK_ROWS = 32

STATE = """() => {
  const api = window.__threadWeight;
  const viewport = api.viewport();
  const rows = [...document.querySelectorAll('[data-role]')];
  return {
    length: api.threadLength(),
    mounted: rows.length,
    positions: rows.map((row) => Number(row.getAttribute('aria-posinset'))),
    sizes: [...new Set(rows.map((row) => row.getAttribute('aria-setsize')))],
    roles: [...new Set(rows.map((row) => row.getAttribute('role')))],
    scrollTop: viewport.scrollTop,
    scrollHeight: viewport.scrollHeight,
    clientHeight: viewport.clientHeight,
    topSpacer: document.querySelector('[data-thread-spacer="top"]')?.offsetHeight ?? null,
    nodes: document.getElementsByTagName('*').length,
  };
}"""

# One sample per frame: how much of the viewport rows cover, and which messages are mounted.
RECORD = """() => {
  const viewport = window.__threadWeight.viewport();
  const record = { frames: 0, minCoverage: 1, seen: new Set(), live: true };
  window.__windowRecord = record;
  const tick = () => {
    if (!record.live) return;
    const box = viewport.getBoundingClientRect();
    let covered = 0;
    for (const row of document.querySelectorAll('[data-role]')) {
      const at = row.getBoundingClientRect();
      covered += Math.max(0, Math.min(at.bottom, box.bottom) - Math.max(at.top, box.top));
      record.seen.add(Number(row.getAttribute('aria-posinset')));
    }
    record.frames += 1;
    record.minCoverage = Math.min(record.minCoverage, covered / box.height);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}"""

# Rows over at least `min` of the viewport: the window is where the reader is.
COVERED = """(min) => {
  const box = window.__threadWeight.viewport().getBoundingClientRect();
  let covered = 0;
  for (const row of document.querySelectorAll('[data-role]')) {
    const at = row.getBoundingClientRect();
    covered += Math.max(0, Math.min(at.bottom, box.bottom) - Math.max(at.top, box.top));
  }
  return covered / box.height >= min;
}"""

COVERED_AFTER = """async ([frames, min]) => {
  for (let frame = 0; frame < frames; frame += 1) {
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  const box = window.__threadWeight.viewport().getBoundingClientRect();
  let covered = 0;
  for (const row of document.querySelectorAll('[data-role]')) {
    const at = row.getBoundingClientRect();
    covered += Math.max(0, Math.min(at.bottom, box.bottom) - Math.max(at.top, box.top));
  }
  return covered / box.height;
}"""

STOP_RECORD = """() => {
  const record = window.__windowRecord;
  record.live = false;
  return { frames: record.frames, minCoverage: record.minCoverage, seen: [...record.seen] };
}"""

# The first row the reader can see, and where it sits, over a run of frames.
REST = """async (frames) => {
  const viewport = window.__threadWeight.viewport();
  const fold = viewport.getBoundingClientRect().top;
  const row = [...document.querySelectorAll('[data-role]')].find(
    (candidate) => candidate.getBoundingClientRect().bottom > fold + 1,
  );
  if (!row) return null;
  const id = row.getAttribute('data-message-id');
  const offsets = [];
  for (let frame = 0; frame < frames; frame += 1) {
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const now = document.querySelector(`[data-message-id="${CSS.escape(id)}"]`);
    offsets.push(now ? now.getBoundingClientRect().top - fold : null);
  }
  return offsets;
}"""

# Focus a control in the first row the reader can see, and mark its node: a row that was unmounted
# and mounted again has a new node, which the mark is not on.
PIN = """() => {
  const viewport = window.__threadWeight.viewport();
  const fold = viewport.getBoundingClientRect().top;
  for (const row of document.querySelectorAll('[data-role]')) {
    if (row.getBoundingClientRect().bottom <= fold + 1) continue;
    const control = row.querySelector('button, [tabindex]');
    if (!control) continue;
    control.focus({ preventScroll: true });
    if (document.activeElement !== control) continue;
    control.setAttribute('data-pin-probe', '');
    return { position: Number(row.getAttribute('aria-posinset')), scrollTop: viewport.scrollTop };
  }
  return null;
}"""

PINNED = """() => {
  const probe = document.querySelector('[data-pin-probe]');
  const row = probe?.closest('[data-role]') ?? null;
  return {
    mounted: probe !== null,
    focused: probe !== null && document.activeElement === probe,
    position: row ? Number(row.getAttribute('aria-posinset')) : null,
    gaps: document.querySelectorAll('[data-thread-spacer="gap"]').length,
  };
}"""

# The first mounted row's ordinal and the counter increments of the spacer standing in for the rest.
SPACER_EQUATIONS = """() => ({
  first: Number(document.querySelector('[data-role][aria-posinset]').getAttribute('aria-posinset')),
  increment: document.querySelector('[data-thread-spacer="top"]').style.counterIncrement,
})"""

EQUATIONS_BEFORE = """(first) => [...document.querySelectorAll('[data-role]')]
  .slice(0, first - 1)
  .reduce((sum, row) => sum + row.querySelectorAll('.eqn-num').length, 0)"""

JUMP = """(top) => {
  const viewport = window.__threadWeight.viewport();
  viewport.scrollTo({ top: top < 0 ? viewport.scrollHeight : top, behavior: 'instant' });
}"""

# One entry per commit that added rows: how many, and how tall they came out.
WATCH_PASSES = """() => {
  const viewport = window.__threadWeight.viewport();
  const passes = [];
  window.__fillPasses = passes;
  new MutationObserver((records) => {
    let rows = 0;
    let px = 0;
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node.nodeType !== 1 || !node.matches('[data-role]')) continue;
        rows += 1;
        px += node.getBoundingClientRect().height;
      }
    }
    if (rows > 0) passes.push({ rows, px: Math.round(px) });
  }).observe(viewport, { childList: true });
}"""

SELECT_ALL = """() => {
  document.body.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'a', ctrlKey: true, bubbles: true, cancelable: true }),
  );
}"""

SELECT_THREAD = """() => {
  const viewport = window.__threadWeight.viewport();
  const selection = window.getSelection();
  selection.removeAllRanges();
  const range = document.createRange();
  range.selectNodeContents(viewport);
  selection.addRange(range);
}"""

# A dispatched event runs the app's copy handlers in every engine; a key press does not copy in
# WebKit. What the selection holds once they return is what the clipboard would take.
COPY = """() => {
  window.__threadWeight.viewport().dispatchEvent(
    new ClipboardEvent('copy', { bubbles: true, cancelable: true }),
  );
  return {
    chars: window.__selectedChars(),
    mounted: document.querySelectorAll('[data-role]').length,
  };
}"""

# Without whitespace: a code fence serialises its line breaks differently once it has been
# highlighted (Firefox), and which fences have been is a matter of where each page has scrolled.
SELECTED_CHARS = """
window.__selectedChars = () => window.getSelection().toString().replace(/\\s+/g, '').length;
"""


def info(message: str) -> None:
    print(f"[thread-window] {message}", flush = True)


def open_thread(
    context,
    messages: int,
    *,
    windowed: bool,
    stream: bool | int = False,
    seed: dict = SEED,
):
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    query = (
        "shell=1"
        + ("&window=1" if windowed else "")
        + (f"&stream={STREAM_CHUNKS if stream is True else stream}" if stream else "")
    )
    page.goto(f"{BASE}/smoke-thread-weight.html?{query}", wait_until = "load")
    page.wait_for_function("() => window.__threadWeight !== undefined", timeout = 120_000)
    page.evaluate("([n, o]) => window.__threadWeight.seed(n, o)", [messages, seed])
    page.wait_for_function(
        "(n) => window.__threadWeight.threadLength() >= n", arg = messages, timeout = 120_000
    )
    if not windowed:
        page.wait_for_function(
            "(n) => window.__threadWeight.messageCount() >= n", arg = messages, timeout = 180_000
        )
    settle(page)
    return page, errors


def settle(page) -> dict:
    """Waits until the mounted rows and the scroll range have held for half a second."""
    last, stable = None, 0
    for _ in range(300):
        state = page.evaluate(STATE)
        key = (state["mounted"], state["scrollHeight"], round(state["scrollTop"]))
        stable = stable + 1 if key == last else 0
        if stable >= 5:
            return state
        last = key
        page.wait_for_timeout(100)
    raise AssertionError(f"the thread never settled: {last}")


def check_ordinals(
    state: dict,
    label: str,
    pinned: int | None = None,
) -> None:
    """The mounted rows are one run of consecutive messages, plus `pinned` when it is outside it."""
    positions = state["positions"]
    assert state["sizes"] == [str(state["length"])], (
        f"{label}: rows must all publish the thread's length {state['length']} as aria-setsize, "
        f"got {state['sizes']}"
    )
    assert state["roles"] == ["article"], (
        f"{label}: every windowed row must carry role=article, the role its ordinals need, got "
        f"{state['roles']}"
    )
    assert positions == sorted(positions), f"{label}: rows are out of order: {positions}"
    if pinned is not None:
        assert pinned in positions, f"{label}: the pinned row {pinned} is not mounted: {positions}"
        run = [position for position in positions if position != pinned]
        if run and (pinned == run[0] - 1 or pinned == run[-1] + 1):
            run = positions
    else:
        run = positions
    assert run == list(
        range(run[0], run[0] + len(run))
    ), f"{label}: mounted rows are not consecutive messages: {positions}"


def check_engine(playwright, engine: str) -> None:
    launcher = getattr(playwright, engine)
    browser = launcher.launch(
        headless = True, args = chromium_launch_args() if engine == "chromium" else []
    )
    try:
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script(SELECTED_CHARS)

        # 1. The DOM does not grow with the chat.
        page, errors = open_thread(context, SMALL, windowed = True)
        small = page.evaluate(STATE)
        page.close()
        page, errors = open_thread(context, LARGE, windowed = True)
        opened = page.evaluate(STATE)
        info(
            f"{engine}: {SMALL} messages mount {small['mounted']} rows ({small['nodes']} elements), "
            f"{LARGE} mount {opened['mounted']} ({opened['nodes']})"
        )
        assert opened["length"] == LARGE, opened["length"]
        assert opened["mounted"] <= small["mounted"] + ROW_SLACK, (
            f"a {LARGE}-message thread mounts {opened['mounted']} rows against {small['mounted']} "
            f"for {SMALL}: the mounted rows grow with the chat"
        )
        check_ordinals(opened, "on open")
        assert (
            opened["positions"][-1] == LARGE
        ), f"the thread did not open on its last message: {opened['positions']}"
        bottom = opened["scrollHeight"] - opened["clientHeight"] - opened["scrollTop"]
        assert abs(bottom) <= 2, f"the thread opened {bottom}px above its end"

        # 2. The scroll range is the fully mounted thread's.
        full_page, full_errors = open_thread(context, LARGE, windowed = False)
        full = full_page.evaluate(STATE)
        full_page.evaluate(SELECT_THREAD)
        full_chars = full_page.evaluate("() => window.__selectedChars()")
        full_page.close()
        assert full["mounted"] == LARGE and not full_errors, (full["mounted"], full_errors)
        drift = abs(opened["scrollHeight"] - full["scrollHeight"]) / full["scrollHeight"]
        info(
            f"{engine}: scroll range {opened['scrollHeight']} windowed, {full['scrollHeight']} "
            f"mounted ({drift:.1%} apart)"
        )
        assert drift <= EXTENT_TOLERANCE, (
            f"the windowed thread is {opened['scrollHeight']}px tall and the mounted one "
            f"{full['scrollHeight']}px: {drift:.1%} apart, {EXTENT_TOLERANCE:.0%} allowed"
        )

        # 3. Wheeling to the top, the window keeps up and passes every message.
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.evaluate(RECORD)
        rest_checked = False
        for step in range(6_000):
            page.mouse.wheel(0, -WHEEL_PX)
            page.wait_for_timeout(WHEEL_GAP_MS)
            try:
                page.wait_for_function(COVERED, arg = MIN_COVERAGE, timeout = CATCH_UP_MS)
            except PlaywrightTimeoutError:
                raise AssertionError(
                    f"rows cover under {MIN_COVERAGE:.0%} of the viewport {CATCH_UP_MS}ms after "
                    f"wheel step {step}: the window lost the reader"
                ) from None
            if step == 60 and not rest_checked:
                # 4. A reader at rest is not moved by the window settling behind them.
                rest_checked = True
                page.wait_for_timeout(700)
                offsets = page.evaluate(REST, 40)
                assert (
                    offsets and None not in offsets
                ), f"the reader's row unmounted at rest: {offsets}"
                moved = max(offsets) - min(offsets)
                assert moved <= 1, f"a reader at rest moved {moved}px while the window settled"
            if (
                step % 10 == 9
                and page.evaluate("() => window.__threadWeight.viewport().scrollTop") <= 0
            ):
                break
        page.wait_for_timeout(600)
        record = page.evaluate(STOP_RECORD)
        top = page.evaluate(STATE)
        # The least coverage on any one frame is how far the window fell behind in passing. It
        # is reported and not asserted: it is this machine's speed, not the window's structure.
        info(
            f"{engine}: wheel to top, {record['frames']} frames, least coverage on a frame "
            f"{record['minCoverage']:.0%}, {len(record['seen'])} of {LARGE} messages passed"
        )
        missing = sorted(set(range(1, LARGE + 1)) - set(record["seen"]))
        assert not missing, f"messages never mounted on the way up: {missing[:20]}"
        check_ordinals(top, "at the top")
        assert top["positions"][0] == 1 and top["scrollTop"] <= 0 and top["topSpacer"] == 0, (
            f"the top of the thread is not its first message: {top['positions'][:3]}, "
            f"scrollTop {top['scrollTop']}, spacer {top['topSpacer']}"
        )
        assert (
            top["mounted"] <= small["mounted"] + 2 * ROW_SLACK
        ), f"{top['mounted']} rows are still mounted after the sweep: rows are not unmounting"

        # 5. A jump lands on mounted rows.
        page.evaluate(RECORD)
        page.evaluate(
            "() => { const v = window.__threadWeight.viewport();"
            " v.scrollTo({ top: v.scrollHeight / 2, behavior: 'instant' }); }"
        )
        settle(page)
        middle = page.evaluate(STATE)
        record = page.evaluate(STOP_RECORD)
        check_ordinals(middle, "after a jump")
        landed = page.evaluate(REST, 1)
        assert landed and landed[0] is not None, "a jump to the middle left the viewport empty"
        assert 1 < middle["positions"][0] and middle["positions"][-1] < LARGE, middle["positions"]

        # 6. A row the reader is working in survives every jump, apart from the window.
        pin = page.evaluate(PIN)
        assert pin, "no mounted row has a control to focus"
        bound = small["mounted"] + 2 * ROW_SLACK + 1
        for label, target in (("the top", 0), ("the end", -1)):
            page.evaluate(JUMP, target)
            state = settle(page)
            held = page.evaluate(PINNED)
            assert held["mounted"] and held["focused"], (
                f"a jump to {label} unmounted the row the reader had focus in "
                f"(message {pin['position']}): {held}, rows {state['positions']}"
            )
            check_ordinals(state, f"pinned, at {label}", pinned = pin["position"])
            assert held["gaps"] >= 1, f"the pinned row has no spacer of its own at {label}: {held}"
            assert state["mounted"] <= bound, (
                f"{state['mounted']} rows mounted at {label} with one pinned: the window "
                f"stretched to the pinned row instead of leaving it where it is"
            )
            drift = abs(state["scrollHeight"] - full["scrollHeight"]) / full["scrollHeight"]
            assert drift <= EXTENT_TOLERANCE, (
                f"the scroll range is {drift:.1%} off the mounted thread at {label}: the "
                f"spacers around the pinned row do not stand in for the rows left out"
            )
        info(
            f"{engine}: message {pin['position']} kept its focus across jumps to both ends, "
            f"{state['mounted']} rows mounted"
        )
        # Back onto it: the window takes the row in, without moving its node.
        page.evaluate(JUMP, pin["scrollTop"])
        state = settle(page)
        held = page.evaluate(PINNED)
        assert (
            held["mounted"] and held["focused"] and held["gaps"] == 0
        ), f"coming back to the pinned row lost it: {held}, rows {state['positions']}"
        check_ordinals(state, "back on the pinned row")
        # Let go: the row unmounts once the reader is elsewhere and no longer in it.
        page.evaluate(JUMP, 0)
        settle(page)
        page.evaluate("() => document.activeElement?.blur()")
        page.wait_for_function(
            "() => document.querySelector('[data-pin-probe]') === null", timeout = 30_000
        )
        check_ordinals(settle(page), "after the pin is let go")
        page.evaluate(JUMP, pin["scrollTop"])
        settle(page)

        # 7. A copy over unmounted rows carries the whole conversation.
        page.evaluate(SELECT_THREAD)
        copied = page.evaluate(COPY)
        info(
            f"{engine}: copy holds {copied['chars']} characters with {copied['mounted']} rows "
            f"mounted; the mounted thread selects {full_chars}"
        )
        assert (
            copied["mounted"] == LARGE
        ), f"a copy over the whole thread ran with {copied['mounted']} of {LARGE} rows mounted"
        assert copied["chars"] == full_chars, (
            f"the copy holds {copied['chars']} characters and the mounted thread {full_chars}: "
            f"text was lost or added"
        )
        page.evaluate("() => window.getSelection().removeAllRanges()")
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length <= n",
            arg = small["mounted"] + 2 * ROW_SLACK,
            timeout = 30_000,
        )
        check_ordinals(settle(page), "after the copy")
        assert not errors, errors
        page.close()

        # 8. A hidden thread stays as it is, and holds nothing else up.
        page, errors = open_thread(context, LARGE, windowed = True)
        shown = page.evaluate(STATE)
        page.evaluate(
            "() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'none'; }"
        )
        page.wait_for_timeout(1_500)
        hidden = page.evaluate("() => document.querySelectorAll('[data-role]').length")
        assert hidden == shown["mounted"], (
            f"{hidden} rows mounted while the thread was hidden, {shown['mounted']} before: "
            f"a thread with no boxes is walking itself"
        )
        page.evaluate(
            "() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'flex'; }"
        )
        back = settle(page)
        check_ordinals(back, "shown again")
        drift = abs(back["scrollHeight"] - shown["scrollHeight"]) / shown["scrollHeight"]
        assert back["positions"][-1] == LARGE and drift <= 0.01, (
            f"the thread came back changed: rows {back['positions']}, scroll range "
            f"{shown['scrollHeight']} -> {back['scrollHeight']}"
        )
        # A reader asks for every row (the find bar), and the thread is hidden part way through.
        page.evaluate("() => document.documentElement.setAttribute('data-find-bar-open', '')")
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length > n",
            arg = back["mounted"],
            timeout = 30_000,
        )
        assert page.evaluate(
            "() => window.__threadWeight.pendingMounts()"
        ), "a full mount under way on a visible thread must count as pending"
        page.evaluate(
            "() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'none'; }"
        )
        page.wait_for_timeout(500)
        part = page.evaluate("() => document.querySelectorAll('[data-role]').length")
        assert part < LARGE, f"the full mount finished before the thread was hidden ({part} rows)"
        assert not page.evaluate("() => window.__threadWeight.pendingMounts()"), (
            f"a hidden thread stopped {part} rows into a full mount still counts as pending: "
            f"the turn rail of every visible thread waits on it"
        )
        page.evaluate("() => document.documentElement.removeAttribute('data-find-bar-open')")
        page.evaluate(
            "() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'flex'; }"
        )
        # Nobody is waiting for the rest of it any more, so the window comes back.
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length <= n",
            arg = small["mounted"] + 2 * ROW_SLACK,
            timeout = 30_000,
        )
        check_ordinals(settle(page), "after the abandoned full mount")
        assert not errors, errors
        page.close()

        # 9. A streamed reply lands at the tail with the reader following it.
        page, errors = open_thread(context, LARGE, windowed = True, stream = True)
        page.evaluate("() => window.__threadWeight.send('go')")
        page.wait_for_function("() => window.__threadWeight.isRunning()", timeout = 60_000)
        page.wait_for_function(
            "() => !window.__threadWeight.isRunning()", timeout = 120_000, polling = 100
        )
        sent = settle(page)
        check_ordinals(sent, "after a reply")
        text = page.evaluate("() => window.__threadWeight.lastAssistantMessage()?.innerText ?? ''")
        assert (
            sent["length"] == LARGE + 2 and sent["positions"][-1] == LARGE + 2
        ), f"the reply is not the mounted tail: {sent['length']} messages, {sent['positions']}"
        assert f"word{STREAM_CHUNKS - 1}" in text, "the stream did not finish in the mounted row"
        assert (
            sent["mounted"] <= small["mounted"] + 2 * ROW_SLACK
        ), f"{sent['mounted']} rows mounted after a reply"
        assert not errors, errors
        page.close()

        # 10. A full mount goes in passes bounded by height.
        page, errors = open_thread(context, TALL_MESSAGES, windowed = True, seed = TALL_SEED)
        tall = page.evaluate(STATE)
        assert (
            tall["mounted"] < TALL_MESSAGES
        ), f"the tall thread is not windowed: {tall['mounted']}"
        page.evaluate(WATCH_PASSES)
        page.evaluate(SELECT_ALL)
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length >= n",
            arg = TALL_MESSAGES,
            timeout = 120_000,
        )
        passes = page.evaluate("() => window.__fillPasses")
        tallest = max(passes, key = lambda entry: entry["px"])
        info(
            f"{engine}: select all mounted {TALL_MESSAGES} long replies in {len(passes)} passes, "
            f"the largest {tallest['rows']} rows and {tallest['px']}px"
        )
        for entry in passes:
            assert entry["rows"] <= 2 * FILL_CHUNK_ROWS, f"a pass mounted {entry['rows']} rows"
            assert entry["rows"] == 1 or entry["px"] <= FILL_BUDGET_PX * FILL_BUDGET_SLACK, (
                f"one pass of the full mount added {entry['rows']} rows, {entry['px']}px of them: "
                f"the height budget ({FILL_BUDGET_PX}px) is not bounding it"
            )
        assert not errors, errors
        page.close()

        # 11. The last prompt is mounted with its reply, and an open prompt editor keeps its draft.
        page, errors = open_thread(context, LARGE, windowed = True)
        last_user = page.evaluate(
            "() => [...document.querySelectorAll('[data-role]')].filter("
            "(row) => row.dataset.role === 'user').length"
        )
        assert last_user >= 1, "the window at the tail left out the prompt its reply answers"
        edited = page.evaluate("() => window.__threadWeight.editLastPrompt()")
        editor = page.locator(f'[data-message-id="{edited}"] textarea')
        editor.fill(DRAFT)
        # Wheeled, not scrolled by script: a programmatic scroll is not the reader leaving the tail.
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for _ in range(20):
            page.mouse.wheel(0, -WHEEL_PX)
            page.wait_for_timeout(WHEEL_GAP_MS)
        away = settle(page)
        assert editor.count() == 1 and editor.evaluate("(e) => e === document.activeElement"), (
            f"a prompt being edited lost its editor or focus {away['scrollHeight'] - away['scrollTop']}"
            f"px up the thread"
        )
        assert editor.input_value() == DRAFT, "a prompt being edited lost its draft"
        assert not errors, errors
        page.close()

        # 12. Equation numbers continue across the rows the leading spacer stands for.
        numbered = {"numbered": True}
        page, errors = open_thread(context, LARGE, windowed = True, seed = numbered)
        spacer = page.evaluate(SPACER_EQUATIONS)
        page.close()
        page, errors = open_thread(context, LARGE, windowed = False, seed = numbered)
        expected = page.evaluate(EQUATIONS_BEFORE, spacer["first"])
        page.close()
        info(
            f"{engine}: the leading spacer counts {spacer['increment']!r}, rows above hold {expected}"
        )
        assert spacer["first"] > 1 and expected > 0, spacer
        assert spacer["increment"] == f"katexEqnNo {expected} mmlEqnNo {expected}", (
            f"the leading spacer counts {spacer['increment']!r} for rows that render {expected} "
            f"numbered equations: equation numbers below it shift with the window"
        )

        # 13. A block's wrap choice outlives its row.
        page, errors = open_thread(context, LARGE, windowed = True, seed = numbered)
        row = page.evaluate(
            "() => window.__threadWeight.lastAssistantMessage().getAttribute('data-message-id')"
        )
        wrap = f'[data-message-id="{row}"] button[aria-label="Enable word wrap"], [data-message-id="{row}"] button[aria-label="Disable word wrap"]'
        before = page.locator(wrap).first.get_attribute("aria-pressed")
        page.locator(wrap).first.click()
        page.evaluate("() => window.__threadWeight.composer()?.focus()")
        # A loaded host can hold the window still for a settle; the row must leave within a few.
        for _ in range(5):
            page.evaluate(JUMP, 0)
            settle(page)
            if page.locator(f'[data-message-id="{row}"]').count() == 0:
                break
        else:
            raise AssertionError("the row never left")
        page.evaluate(JUMP, -1)
        settle(page)
        after = page.locator(wrap).first.get_attribute("aria-pressed")
        assert before != after, f"a block's word wrap reset to {after} when its row came back"
        assert not errors, errors
        page.close()

        # 14. A bound Fork Chat chord still forks at the tip once the last row has left.
        context.add_init_script(
            "localStorage.setItem('unsloth_keyboard_shortcuts', "
            "JSON.stringify({forkChat: {primary: 'Mod+Alt+Shift+KeyY'}}))"
        )
        page, errors = open_thread(context, LARGE, windowed = True)
        last = page.evaluate(
            "() => window.__threadWeight.lastAssistantMessage().getAttribute('data-message-id')"
        )
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for _ in range(400):
            page.mouse.wheel(0, -WHEEL_PX)
            page.wait_for_timeout(30)
            if page.evaluate("() => document.querySelector('.aui-thread-viewport').scrollTop <= 0"):
                break
        settle(page)
        assert page.locator(f'[data-message-id="{last}"]').count() == 0, "the last row never left"
        page.keyboard.press("ControlOrMeta+Alt+Shift+KeyY")
        page.wait_for_timeout(1000)
        forks = [
            json.loads(write["body"] or "{}")
            for write in page.evaluate("() => window.__threadWeight.apiWrites()")
            if write["url"].endswith("/fork")
        ]
        info(f"{engine}: the fork chord at the top posts {forks!r}")
        assert [fork.get("messageId") for fork in forks] == [
            last
        ], f"the fork chord posted {forks!r} with the last row unmounted, not one fork of {last}"
        page.close()

        # 14. Scrolling away from a streaming reply.
        page, errors = open_thread(
            context, TALL_MESSAGES, windowed = True, stream = LONG_STREAM_CHUNKS, seed = AWAY_SEED
        )
        page.evaluate("() => window.__threadWeight.send('go')")
        page.wait_for_function("() => window.__threadWeight.isRunning()", timeout = 60_000)
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for _ in range(AWAY_WHEEL_STEPS):
            page.mouse.wheel(0, -WHEEL_PX)
            page.wait_for_timeout(WHEEL_GAP_MS)
        coverage = page.evaluate(COVERED_AFTER, [AWAY_FRAMES, MIN_COVERAGE])
        away = page.evaluate(STATE)
        assert page.evaluate(
            "() => window.__threadWeight.isRunning()"
        ), "the reply ended before the check: it proves nothing about a stream"
        assert coverage >= MIN_COVERAGE, (
            f"rows cover {coverage:.0%} of the viewport {AWAY_FRAMES} frames after the reader "
            f"wheeled away from a streaming reply (mounted {away['positions']}): the window is "
            "waiting for the stream"
        )
        check_ordinals(away, "away from a streaming reply", pinned = TALL_MESSAGES + 2)
        info(
            f"{engine}: wheeled away from a streaming reply, rows {away['positions']} cover the "
            "reader while it is still being written"
        )
        page.wait_for_function(
            "() => !window.__threadWeight.isRunning()", timeout = 300_000, polling = 100
        )
        text = page.evaluate("() => window.__threadWeight.lastAssistantMessage()?.innerText ?? ''")
        assert (
            f"word{LONG_STREAM_CHUNKS - 1}" in text
        ), "the reply the reader left did not finish in its mounted row"
        assert not errors, errors
        page.close()
        info(f"{engine}: ok " + json.dumps({"small": small["mounted"], "large": opened["mounted"]}))
    finally:
        browser.close()


def main() -> int:
    proc = None
    try:
        if not _EXTERNAL:
            proc = start_vite(PORT)
            wait_for_smoke_page(
                f"{BASE}/smoke-thread-weight.html",
                "smoke-thread-weight-main.tsx",
                proc = proc,
                info = info,
            )
        with sync_playwright() as playwright:
            for engine in ENGINES:
                check_engine(playwright, engine)
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
