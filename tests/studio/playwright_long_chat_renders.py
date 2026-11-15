# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check rendering during streaming and scrolling in a long chat with tools and web sources.

A streamed token must not re-render the reply's source badges, and a run starting or ending
must not re-render the python / terminal cards of earlier replies. Counts come from a React
DevTools hook installed before the page loads, using DevTools' rule for which fibers rendered.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5217"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
# The harness page holds every row mounted, so the whole history is in the DOM.
MESSAGES = 30
CHUNKS = 60

RENDER_COUNTER = r"""
(() => {
  window.__renders = null;
  const nameOf = (fiber) => {
    const type = fiber.type;
    if (!type) return null;
    if (fiber.tag === 0 || fiber.tag === 1 || fiber.tag === 15) return type.displayName || type.name || null;
    if (fiber.tag === 11) return type.displayName || type.render?.displayName || type.render?.name || null;
    if (fiber.tag === 14) return type.displayName || type.type?.displayName || type.type?.name || null;
    return null;
  };
  const walk = (fiber, counts) => {
    // Children that were not cloned this commit were not visited by React.
    if (fiber.alternate && fiber.alternate.child === fiber.child) return;
    for (let child = fiber.child; child; child = child.sibling) {
      const name = nameOf(child);
      // PerformedWork (1) is set only on fibers whose component function ran this commit.
      if (name && child.alternate && (child.flags & 1)) counts[name] = (counts[name] || 0) + 1;
      walk(child, counts);
    }
  };
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    supportsFiber: true,
    isDisabled: false,
    renderers: new Map(),
    inject(renderer) { const id = this.renderers.size + 1; this.renderers.set(id, renderer); return id; },
    checkDCE() {},
    onScheduleFiberRoot() {},
    onCommitFiberUnmount() {},
    onPostCommitFiberRoot() {},
    onCommitFiberRoot(_id, root) {
      if (window.__renders) walk(root.current, window.__renders);
    },
  };
})();
"""


def info(message: str) -> None:
    print(f"[long-chat-renders] {message}", flush = True)


def run(page) -> dict[str, int]:
    page.goto(f"{BASE}/smoke-thread-weight.html?stream={CHUNKS}&sources=1")
    page.wait_for_function("() => window.__threadWeight !== undefined", timeout = 120_000)
    page.evaluate(
        "(n) => window.__threadWeight.seed(n, { agent: { rounds: 1, sources: true } })",
        MESSAGES,
    )
    page.wait_for_function(
        "(n) => window.__threadWeight.messageCount() >= n", arg = MESSAGES, timeout = 120_000
    )
    seeded = page.evaluate(
        '() => document.querySelectorAll(\'[data-role="assistant"] a[href^="https://example"]\').length'
    )
    assert seeded > 0, "the seeded history shows no source badges"
    page.wait_for_timeout(500)
    page.evaluate("() => { window.__renders = {}; }")
    page.evaluate("() => window.__threadWeight.send('go')")
    page.wait_for_function("() => window.__threadWeight.isRunning()", timeout = 60_000)
    page.wait_for_function("() => !window.__threadWeight.isRunning()", timeout = 120_000, polling = 100)
    page.wait_for_timeout(300)
    counts = page.evaluate(
        "() => { const c = window.__renders; window.__renders = null; return c; }"
    )
    live = page.evaluate(
        "() => window.__threadWeight.lastAssistantMessage()?.querySelectorAll('a[href$=\"/live\"]').length ?? 0"
    )
    text = page.evaluate("() => window.__threadWeight.lastAssistantMessage()?.innerText ?? ''")
    assert live > 0, "the streamed reply shows no source badges, so nothing was measured"
    assert f"word{CHUNKS - 1}" in text, "the stream did not finish"
    return counts


def check_reply_paint(page) -> None:
    viewport = page.locator(".aui-thread-viewport")
    replies = page.locator('[data-role="assistant"]')
    # The first two replies are neighbors. One will contain a wide unwrapped block.
    first, second = replies.nth(0), replies.nth(1)
    scroll = "(v, end) => v.scrollTo({top: end ? v.scrollHeight : 0, behavior: 'instant'})"
    viewport.evaluate(scroll, True)
    expect(second).to_have_css("contain", "paint")
    page.emulate_media(media = "print")
    expect(second).to_have_css("contain", "none")
    page.emulate_media(media = "screen")

    viewport.evaluate(scroll, False)
    expect(first).to_have_css("contain", "none")
    expect(second).to_have_css("contain", "none")
    first.locator(".aui-assistant-message-content").evaluate("e => e.style.minWidth = '200vw'")
    viewport.evaluate(scroll, True)
    expect(second).to_have_css("contain", "paint")
    assert first.evaluate("e => e.scrollWidth > e.clientWidth"), "fixture has no wide reply"
    expect(first).to_have_css("contain", "none")
    assert viewport.evaluate(
        "v => v.scrollWidth > v.clientWidth"
    ), "wide reply lost its horizontal scroll range"


ROUNDS = 12
# With KaTeX's MathML on the page, each removal from the middle of the reply restyles all of it.
CHILD_LIST_WATCH = r"""
() => {
  window.__tail = {midRemovals: 0, statuses: 0};
  const seen = new WeakSet();
  const watch = () => document.querySelectorAll('.aui-assistant-message-content').forEach((el) => {
    if (seen.has(el)) return;
    seen.add(el);
    new MutationObserver((records) => records.forEach((r) => {
      if (r.target === el && r.nextSibling) window.__tail.midRemovals += r.removedNodes.length;
      r.addedNodes.forEach((n) => {
        if (n.nodeType === 1 && (n.matches('[data-slot="next-round-indicator"]') || n.querySelector('[data-slot="next-round-indicator"]'))) window.__tail.statuses++;
      });
    })).observe(el, {childList: true, subtree: true});
  });
  new MutationObserver(watch).observe(document.body, {childList: true, subtree: true});
}
"""


def check_round_tail(page) -> dict[str, int]:
    page.goto(f"{BASE}/smoke-thread-weight.html?stream=5&rounds={ROUNDS}")
    page.wait_for_function("() => window.__threadWeight !== undefined", timeout = 120_000)
    page.evaluate(CHILD_LIST_WATCH)
    page.evaluate("() => window.__threadWeight.send('go')")
    page.wait_for_function("() => window.__threadWeight.isRunning()", timeout = 60_000)
    page.wait_for_function("() => !window.__threadWeight.isRunning()", timeout = 120_000, polling = 100)
    tail = page.evaluate("() => window.__tail")
    text = page.evaluate("() => window.__threadWeight.lastAssistantMessage()?.innerText ?? ''")
    assert "word4" in text and f"Round {ROUNDS - 1}" in text, "the agent stream did not finish"
    assert page.locator(".katex-mathml").count() >= ROUNDS, "no KaTeX MathML on the page"
    assert tail["statuses"] >= ROUNDS - 1, f"the next-round status showed {tail['statuses']} times"
    # One for the first "Working..." status, which the first part replaces.
    assert tail["midRemovals"] <= 1, (
        f"{tail['midRemovals']} nodes left the middle of the reply over {ROUNDS} tool rounds: "
        f"a round's tail is being removed after the next part is appended"
    )
    return tail


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
            browser = playwright.chromium.launch(headless = True, args = chromium_launch_args())
            try:
                context = browser.new_context(viewport = {"width": 1280, "height": 900})
                context.add_init_script(RENDER_COUNTER)
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                counts = run(page)
                check_reply_paint(page)
                tail = check_round_tail(context.new_page())
            finally:
                browser.close()
        badges = counts.get("SourceBadge", 0)
        cards = counts.get("PythonToolUIImpl", 0) + counts.get("TerminalToolUIImpl", 0)
        info(
            json.dumps(
                {
                    "SourceBadge": badges,
                    "python/terminal cards": cards,
                    "chunks": CHUNKS,
                    "round_tail": tail,
                }
            )
        )
        assert not errors, errors
        # The live reply's six badges may render a couple of times (mount, status change), not per token.
        assert badges <= 2 * 6, (
            f"source badges re-rendered {badges} times over a {CHUNKS}-chunk stream: "
            f"a streamed token is re-rendering them"
        )
        assert cards <= 2, (
            f"python / terminal cards re-rendered {cards} times: the run start or end is "
            f"re-rendering every card in the history"
        )
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
