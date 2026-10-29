# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""exercise ProgressiveRows paging, cleanup, remounting, and observer rearming in browsers."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

from playwright.async_api import async_playwright

FRONTEND = Path(__file__).resolve().parents[2] / "studio" / "frontend"
ENGINES = [e for e in os.environ.get("SMOKE_ENGINES", "chromium").split(",") if e]

# track live observers to verify that completed or unmounted lists release them
COUNT_OBSERVERS = """
  window.liveObservers = 0;
  const Original = window.IntersectionObserver;
  window.IntersectionObserver = class extends Original {
    constructor(...args) { super(...args); this.live = false; }
    observe(target) { if (!this.live) { this.live = true; window.liveObservers++; } return super.observe(target); }
    disconnect() { if (this.live) { this.live = false; window.liveObservers--; } return super.disconnect(); }
  };
"""

STATE = """() => {
  const items = [...document.querySelectorAll('[data-sidebar="content"] li')];
  const last = items[items.length - 1];
  return {
    rows: document.querySelectorAll('[data-row]').length,
    end: !!document.querySelector('[data-end]'),
    sentinelLast: !!last && !last.hasAttribute('data-row') && !last.hasAttribute('data-end'),
    renders: window.rowRenders,
    observers: window.liveObservers,
  };
}"""


async def settle(page):
    """wait until the row count has held for half a second."""
    last, stable = None, 0
    for _ in range(200):
        state = await page.evaluate(STATE)
        stable = stable + 1 if state["rows"] == last else 0
        if stable >= 5:
            return state
        last = state["rows"]
        await page.wait_for_timeout(100)
    raise AssertionError(f"row count never settled: {state}")


async def scroll_to_bottom(page):
    await page.evaluate(
        "() => { const s = document.querySelector('[data-sidebar=\"content\"]'); s.scrollTop = s.scrollHeight; }"
    )
    return await settle(page)


async def check(browser, url, engine):
    page = await browser.new_page(viewport = {"width": 600, "height": 700})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.add_init_script(COUNT_OBSERVERS)

    await page.goto(url + "?count=1000")
    await page.locator("[data-row]").first.wait_for()
    state = await settle(page)
    assert state["rows"] == 50 and state["sentinelLast"] and not state["end"], state
    assert state["renders"] == 50 and state["observers"] == 1, state

    # prior rows must remain mounted without rerendering as each page adds 50 rows
    for expected in (100, 150, 200):
        state = await scroll_to_bottom(page)
        assert state["rows"] == expected, state
        assert state["renders"] == expected, state
        assert state["observers"] == 1, state

    # refetching preserves page depth but rerenders each mounted row once
    await page.evaluate("window.fixture.refresh()")
    state = await settle(page)
    assert state["rows"] == 200 and state["renders"] == 400, state

    # unmounting disconnects the observer, and remounting restarts from the first page
    assert await page.evaluate("window.liveObservers") == 1
    await page.evaluate("window.fixture.setMounted(false)")
    await page.wait_for_timeout(200)
    assert await page.evaluate("window.liveObservers") == 0
    await page.evaluate("window.fixture.setMounted(true)")
    state = await settle(page)
    assert state["rows"] == 50 and state["observers"] == 1, state

    # shrinking within a page releases the observer, while regrowth restarts paging
    await page.evaluate("window.fixture.setCount(30)")
    state = await settle(page)
    assert state["rows"] == 30 and state["end"] and state["observers"] == 0, state
    await page.evaluate("window.fixture.setCount(1000)")
    state = await settle(page)
    assert state["rows"] == 50 and state["sentinelLast"] and state["observers"] == 1, state

    # the final page leaves only rows and the end marker, with no sentinel or observer
    for _ in range(40):
        state = await scroll_to_bottom(page)
        if state["end"]:
            break
    assert state["rows"] == 1000 and state["end"] and not state["sentinelLast"], state
    assert state["observers"] == 0, state

    # a sentinel in range must rearm each page; 100 rows fit in the 3000 px viewport plus 800 px margin, but 150 do not
    await page.goto(url + "?count=1000&height=3000")
    await page.locator("[data-row]").first.wait_for()
    state = await settle(page)
    assert state["rows"] == 150 and state["sentinelLast"], state

    assert not errors, errors
    await page.close()
    print(
        f"{engine}: paging, render reuse, refresh, end, cleanup, remount, shrink and refill passed",
        flush = True,
    )


async def run_checks(url):
    async with async_playwright() as playwright:
        for engine in ENGINES:
            launch = {"headless": True}
            executable = os.environ.get(f"PW_{engine.upper()}_EXECUTABLE")
            if executable:
                launch["executable_path"] = executable
            browser = await getattr(playwright, engine).launch(**launch)
            try:
                await check(browser, url, engine)
            finally:
                await browser.close()


async def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    node = shutil.which("node")
    assert node, "Node.js is required"
    with tempfile.TemporaryFile(mode = "w+") as log:
        server = subprocess.Popen(
            [
                node,
                "node_modules/vite/bin/vite.js",
                "--config",
                "tests/fixtures/progressive-rows/vite.config.ts",
                "--port",
                str(port),
                "--strictPort",
            ],
            cwd = FRONTEND,
            stdout = log,
            stderr = subprocess.STDOUT,
        )
        try:
            for _ in range(150):
                if server.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    with urlopen(url, timeout = 1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    await asyncio.sleep(0.2)
            else:
                raise RuntimeError("Vite did not start")
            await run_checks(url)
        finally:
            server.terminate()
            try:
                server.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    asyncio.run(main())
