# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate: typing in a long chat must not cost more as the chat grows.

The production build of smoke-thread-weight.html (vite.long-thread.config.ts) is seeded with an
agent-style history (open reasoning, tool cards, web sources, compaction markers, code, KaTeX,
tables) inside the app's own ancestors (`?shell=1`). The same page is measured at a small and a
large size, interleaved, and the gate compares Chromium's style-recalc time per keystroke and per
wheel step between them. A ratio, not an absolute budget, so a slow runner does not fail it: work
that is independent of the chat's length stays near 1x, while a selector that walks the whole
thread (#16603: a descendant `:has()` on the workspace) grows with it.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import statistics
import subprocess
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
DIST = Path(os.environ.get("LONG_THREAD_DIST", REPO / ".playwright-cli" / "long-thread-production"))
SMALL = int(os.environ.get("LONG_THREAD_SMALL", "40"))
LARGE = int(os.environ.get("LONG_THREAD_LARGE", "200"))
REPEATS = int(os.environ.get("LONG_THREAD_REPEATS", "3"))
# Measured on #16603: a whole-thread `:has()` makes per-keystroke style cost grow with the chat
# (about 4x at 4x the turns); without it the cost stays flat, within the noise of a loaded host.
# A floor keeps sub-millisecond noise from turning into a large ratio. Scroll is reported, not
# gated: wheel steps already restyle every hover-styled control in the thread on main (separate,
# older cost), so its ratio cannot tell a new regression apart.
MAX_TYPE_RATIO = float(os.environ.get("LONG_THREAD_MAX_TYPE_RATIO", "2.5"))
TYPE_FLOOR_MS = 1.0
SCROLL_FLOOR_MS = 2.0
KEYS = "the quick brown fox jumps over"
SEED_OPTIONS = {"agent": {"rounds": 2, "compactionEvery": 10, "sources": True, "rich": True}}


def build() -> None:
    if os.environ.get("LONG_THREAD_DIST"):
        return
    subprocess.run(
        ["npx", "vite", "build", "--config", "vite.long-thread.config.ts"], cwd = FRONTEND, check = True
    )


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(_QuietHandler, directory = str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: D401
        pass


def metrics(cdp) -> dict:
    return {m["name"]: m["value"] for m in cdp.send("Performance.getMetrics")["metrics"]}


def measure(browser, base: str, turns: int) -> dict:
    ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(f"{base}/smoke-thread-weight.html?shell=1", wait_until = "load")
    page.wait_for_function("() => !!window.__threadWeight", timeout = 60_000)
    count = turns * 2
    page.evaluate("([n, o]) => window.__threadWeight.seed(n, o)", [count, SEED_OPTIONS])
    page.wait_for_function(
        "(n) => window.__threadWeight.messageCount() >= n", arg = count, timeout = 120_000
    )
    page.wait_for_timeout(2500)
    shape = page.evaluate("""() => ({
      messages: document.querySelectorAll('[data-role]').length,
      nodes: document.getElementsByTagName('*').length,
      sources: document.querySelectorAll('[data-role="assistant"] a[href^="https://example"]').length,
      reasoning: document.querySelectorAll('.aui-reasoning-root, [data-reasoning]').length,
      code: document.querySelectorAll('pre').length,
      katex: document.querySelectorAll('.katex').length,
      tables: document.querySelectorAll('table').length,
      workspace: !!document.querySelector('[data-wallpaper-workspace] .thread-bottom-fade'),
    })""")
    cdp = ctx.new_cdp_session(page)
    cdp.send("Performance.enable")
    box = page.locator(".aui-composer-input").first
    box.click()
    m0 = metrics(cdp)
    for ch in KEYS:
        page.keyboard.press("Space" if ch == " " else ch)
        page.wait_for_timeout(60)
    page.wait_for_timeout(300)
    m1 = metrics(cdp)
    typed_ok = page.evaluate("() => window.__threadWeight.composerText()") == KEYS
    # Scroll from the middle, so both sizes move through the same kind of content instead of the
    # short chat running into its top and turning the remaining wheel steps into no-ops.
    page.evaluate(
        "() => { const v = document.querySelector('.aui-thread-viewport'); v.scrollTop = v.scrollHeight / 2; }"
    )
    page.wait_for_timeout(1500)
    page.mouse.move(720, 420)
    m1 = metrics(cdp)
    steps = 24
    for i in range(steps):
        page.mouse.wheel(0, -400 if i < steps // 2 else 400)
        page.wait_for_timeout(50)
    page.wait_for_timeout(400)
    m2 = metrics(cdp)
    ctx.close()
    return {
        "turns": turns,
        "shape": shape,
        "typed_ok": typed_ok,
        "type_style_ms": 1000 * (m1["RecalcStyleDuration"] - m0["RecalcStyleDuration"]) / len(KEYS),
        "type_layout_ms": 1000 * (m1["LayoutDuration"] - m0["LayoutDuration"]) / len(KEYS),
        "scroll_style_ms": 1000 * (m2["RecalcStyleDuration"] - m1["RecalcStyleDuration"]) / steps,
        "scroll_layout_ms": 1000 * (m2["LayoutDuration"] - m1["LayoutDuration"]) / steps,
    }


def main() -> int:
    build()
    if not (DIST / "smoke-thread-weight.html").exists():
        print(f"FAIL: no production build at {DIST}")
        return 1
    server, base = serve(DIST)
    runs: dict[int, list[dict]] = {SMALL: [], LARGE: []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args = chromium_launch_args())
            for r in range(REPEATS):
                for turns in (SMALL, LARGE) if r % 2 == 0 else (LARGE, SMALL):
                    out = measure(browser, base, turns)
                    runs[turns].append(out)
                    print(json.dumps(out), flush = True)
            browser.close()
    finally:
        server.shutdown()

    def med(turns: int, key: str) -> float:
        return statistics.median(x[key] for x in runs[turns])

    failures = []
    for turns, results in runs.items():
        shape = results[0]["shape"]
        if (
            shape["messages"] < turns * 2
            or not shape["workspace"]
            or not all(shape[k] for k in ("reasoning", "sources", "code", "katex", "tables"))
        ):
            failures.append(f"fixture did not render what it claims at {turns} turns: {shape}")
        if not all(x["typed_ok"] for x in results):
            failures.append(f"typed text was lost at {turns} turns")
    type_ratio = max(med(LARGE, "type_style_ms"), TYPE_FLOOR_MS) / max(
        med(SMALL, "type_style_ms"), TYPE_FLOOR_MS
    )
    scroll_ratio = max(med(LARGE, "scroll_style_ms"), SCROLL_FLOOR_MS) / max(
        med(SMALL, "scroll_style_ms"), SCROLL_FLOOR_MS
    )
    print(
        f"style per keystroke: {SMALL} turns {med(SMALL, 'type_style_ms'):.2f} ms, {LARGE} turns "
        f"{med(LARGE, 'type_style_ms'):.2f} ms, ratio {type_ratio:.2f} (max {MAX_TYPE_RATIO})"
    )
    print(
        f"style per wheel step: {SMALL} turns {med(SMALL, 'scroll_style_ms'):.2f} ms, {LARGE} turns "
        f"{med(LARGE, 'scroll_style_ms'):.2f} ms, ratio {scroll_ratio:.2f} (reported only)"
    )
    if type_ratio > MAX_TYPE_RATIO:
        failures.append(
            f"typing style cost grows with the chat: {type_ratio:.2f}x at {LARGE // SMALL}x the turns"
        )
    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print("PASS: typing cost does not grow with the chat")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
