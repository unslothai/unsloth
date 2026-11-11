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
  3. Wheeling to the top never shows an empty viewport, passes every message once, and ends on
     the first one.
  4. A reader at rest is not moved by the window settling behind them.
  5. A jump (a dragged scrollbar) lands on mounted rows.
  6. Copying a selection that runs over unmounted rows carries the whole conversation, the same
     text the fully mounted thread selects, and the window comes back once the selection is gone.
  7. A thread hidden behind another route mounts nothing while it has no boxes to measure.
  8. A reply streamed into the windowed thread lands at the tail with the reader following it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
SEED = {"agent": {"rounds": 2, "compactionEvery": 10, "sources": True, "rich": True}}
# Same allowance studiobench gives a windowed arm (analysis/behaviour.py).
EXTENT_TOLERANCE = 0.10
# Rows the larger thread may mount beyond the smaller one: a row of slack either side.
ROW_SLACK = 4
WHEEL_PX = 700
WHEEL_GAP_MS = 40
# Rows are separated by gaps, so a covered viewport is not rows from edge to edge.
MIN_COVERAGE = 0.5

STATE = """() => {
  const api = window.__threadWeight;
  const viewport = api.viewport();
  const rows = [...document.querySelectorAll('[data-role]')];
  return {
    length: api.threadLength(),
    mounted: rows.length,
    positions: rows.map((row) => Number(row.getAttribute('aria-posinset'))),
    sizes: [...new Set(rows.map((row) => row.getAttribute('aria-setsize')))],
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


def open_thread(context, messages: int, *, windowed: bool, stream: bool = False):
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    query = "shell=1" + ("&window=1" if windowed else "") + (f"&stream={STREAM_CHUNKS}" if stream else "")
    page.goto(f"{BASE}/smoke-thread-weight.html?{query}", wait_until = "load")
    page.wait_for_function("() => window.__threadWeight !== undefined", timeout = 120_000)
    page.evaluate("([n, o]) => window.__threadWeight.seed(n, o)", [messages, SEED])
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


def check_ordinals(state: dict, label: str) -> None:
    positions = state["positions"]
    assert state["sizes"] == [str(state["length"])], (
        f"{label}: rows must all publish the thread's length {state['length']} as aria-setsize, "
        f"got {state['sizes']}"
    )
    assert positions == list(range(positions[0], positions[0] + len(positions))), (
        f"{label}: mounted rows are not consecutive messages: {positions}"
    )


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
        assert opened["positions"][-1] == LARGE, (
            f"the thread did not open on its last message: {opened['positions']}"
        )
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

        # 3. Wheeling to the top never shows an empty viewport and passes every message.
        box = page.locator(".aui-thread-viewport").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.evaluate(RECORD)
        rest_checked = False
        for step in range(6_000):
            page.mouse.wheel(0, -WHEEL_PX)
            page.wait_for_timeout(WHEEL_GAP_MS)
            if step == 60 and not rest_checked:
                # 4. A reader at rest is not moved by the window settling behind them.
                rest_checked = True
                page.wait_for_timeout(700)
                offsets = page.evaluate(REST, 40)
                assert offsets and None not in offsets, f"the reader's row unmounted at rest: {offsets}"
                moved = max(offsets) - min(offsets)
                assert moved <= 1, f"a reader at rest moved {moved}px while the window settled"
            if step % 10 == 9 and page.evaluate("() => window.__threadWeight.viewport().scrollTop") <= 0:
                break
        page.wait_for_timeout(600)
        record = page.evaluate(STOP_RECORD)
        top = page.evaluate(STATE)
        info(
            f"{engine}: wheel to top, {record['frames']} frames, least coverage "
            f"{record['minCoverage']:.0%}, {len(record['seen'])} of {LARGE} messages passed"
        )
        assert record["minCoverage"] >= MIN_COVERAGE, (
            f"a frame showed rows over only {record['minCoverage']:.0%} of the viewport while "
            f"scrolling: the window fell behind the reader"
        )
        missing = sorted(set(range(1, LARGE + 1)) - set(record["seen"]))
        assert not missing, f"messages never mounted on the way up: {missing[:20]}"
        check_ordinals(top, "at the top")
        assert top["positions"][0] == 1 and top["scrollTop"] <= 0 and top["topSpacer"] == 0, (
            f"the top of the thread is not its first message: {top['positions'][:3]}, "
            f"scrollTop {top['scrollTop']}, spacer {top['topSpacer']}"
        )
        assert top["mounted"] <= small["mounted"] + 2 * ROW_SLACK, (
            f"{top['mounted']} rows are still mounted after the sweep: rows are not unmounting"
        )

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

        # 6. A copy over unmounted rows carries the whole conversation.
        page.evaluate(SELECT_THREAD)
        copied = page.evaluate(COPY)
        info(
            f"{engine}: copy holds {copied['chars']} characters with {copied['mounted']} rows "
            f"mounted; the mounted thread selects {full_chars}"
        )
        assert copied["mounted"] == LARGE, (
            f"a copy over the whole thread ran with {copied['mounted']} of {LARGE} rows mounted"
        )
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

        # 7. A hidden thread stays as it is.
        page, errors = open_thread(context, LARGE, windowed = True)
        shown = page.evaluate(STATE)
        page.evaluate("() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'none'; }")
        page.wait_for_timeout(1_500)
        hidden = page.evaluate("() => document.querySelectorAll('[data-role]').length")
        assert hidden == shown["mounted"], (
            f"{hidden} rows mounted while the thread was hidden, {shown['mounted']} before: "
            f"a thread with no boxes is walking itself"
        )
        page.evaluate("() => { document.querySelector('[data-smoke=\"thread\"]').style.display = 'flex'; }")
        back = settle(page)
        check_ordinals(back, "shown again")
        drift = abs(back["scrollHeight"] - shown["scrollHeight"]) / shown["scrollHeight"]
        assert back["positions"][-1] == LARGE and drift <= 0.01, (
            f"the thread came back changed: rows {back['positions']}, scroll range "
            f"{shown['scrollHeight']} -> {back['scrollHeight']}"
        )
        assert not errors, errors
        page.close()

        # 8. A streamed reply lands at the tail with the reader following it.
        page, errors = open_thread(context, LARGE, windowed = True, stream = True)
        page.evaluate("() => window.__threadWeight.send('go')")
        page.wait_for_function("() => window.__threadWeight.isRunning()", timeout = 60_000)
        page.wait_for_function(
            "() => !window.__threadWeight.isRunning()", timeout = 120_000, polling = 100
        )
        sent = settle(page)
        check_ordinals(sent, "after a reply")
        text = page.evaluate("() => window.__threadWeight.lastAssistantMessage()?.innerText ?? ''")
        assert sent["length"] == LARGE + 2 and sent["positions"][-1] == LARGE + 2, (
            f"the reply is not the mounted tail: {sent['length']} messages, {sent['positions']}"
        )
        assert f"word{STREAM_CHUNKS - 1}" in text, "the stream did not finish in the mounted row"
        assert sent["mounted"] <= small["mounted"] + 2 * ROW_SLACK, (
            f"{sent['mounted']} rows mounted after a reply"
        )
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
