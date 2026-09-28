# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Timestamp behavior against real assistant-ui primitives and application CSS.

Run: python tests/studio/playwright_user_message_time.py
Requires frontend npm dependencies and Python Playwright/Chromium. No backend
or GPU is needed. PW_CHROMIUM_EXECUTABLE may select an installed browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

from playwright.async_api import async_playwright

FRONTEND = Path(__file__).resolve().parents[2] / "studio" / "frontend"


async def paint(page):
    await page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


async def check(url):
    async with async_playwright() as playwright:
        launch = {"headless": True}
        if os.environ.get("PW_CHROMIUM_EXECUTABLE"):
            launch["executable_path"] = os.environ["PW_CHROMIUM_EXECUTABLE"]
        browser = await playwright.chromium.launch(**launch)
        try:
            page = await browser.new_page(viewport = {"width": 800, "height": 650})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.add_init_script("""
              window.formatCalls = 0;
              for (const name of ['toLocaleString', 'toLocaleTimeString']) {
                const original = Date.prototype[name];
                Date.prototype[name] = function(...args) {
                  window.formatCalls++;
                  return original.apply(this, args);
                };
              }
            """)
            await page.goto(url + "?count=500&today")
            await page.locator(".aui-user-message-root").first.wait_for()
            await page.mouse.move(799, 640)
            await paint(page)
            assert await page.locator("time").count() == 0
            assert await page.evaluate("window.formatCalls") == 0
            first = page.locator(".aui-user-message-root").first
            await first.hover()
            await page.locator("time").wait_for()
            print("Hover mounted; checking streaming", flush = True)
            calls = await page.evaluate("window.formatCalls")
            assert calls == 2, calls
            for _ in range(120):
                await page.evaluate("window.messageTimeFixture.updateReply()")
                await paint(page)
            assert await page.evaluate("window.formatCalls") == calls
            assert await page.locator("time").count() == 1
            await page.mouse.move(799, 640)
            await paint(page)
            assert await page.locator("time").count() == 0

            print("Streaming passed; checking keyboard", flush = True)
            # Keyboard focus mounts the bar and exposes the full date. Leaving
            # the message with the mouse must not unmount a focused control.
            await page.goto(url + "?branches")
            await page.locator("#before").focus()
            await page.keyboard.press("Tab")
            await page.locator("time").wait_for()
            await page.keyboard.press("Tab")
            trigger = page.locator(".aui-user-message-time-trigger")
            assert await trigger.evaluate("(e) => e === document.activeElement")
            await page.get_by_role("tooltip").wait_for()
            assert (
                await trigger.get_attribute("aria-label")
                == await page.get_by_role("tooltip").inner_text()
            )
            await page.locator(".aui-user-message-root").hover(position = {"x": 4, "y": 4})
            await page.mouse.move(799, 640)
            await paint(page)
            assert await trigger.evaluate("(e) => e === document.activeElement")
            await page.locator("#after").focus()
            await paint(page)
            assert await page.locator("time").count() == 0

            # Reverse traversal first reveals the message, then enters the
            # controls instead of skipping them for the preceding page button.
            await page.goto(url)
            await page.locator(".aui-user-message-root").wait_for()
            await page.locator("#after").focus()
            await page.keyboard.press("Shift+Tab")
            sentinel = page.locator(".aui-user-reveal-sentinel")
            assert await sentinel.evaluate("(e) => e === document.activeElement")
            sentinel_style = await sentinel.evaluate(
                "e => ({ display: getComputedStyle(e).display, "
                "width: getComputedStyle(e).width, "
                "height: getComputedStyle(e).height, "
                "outlineStyle: getComputedStyle(e).outlineStyle, "
                "outlineWidth: getComputedStyle(e).outlineWidth })"
            )
            expected_sentinel_style = {
                "display": "block",
                "width": "0px",
                "height": "0px",
                "outlineStyle": "solid",
                "outlineWidth": "1px",
            }
            assert sentinel_style == expected_sentinel_style, sentinel_style
            delete = page.get_by_role("button", name = "Delete", exact = True)
            await delete.wait_for()
            await page.keyboard.press("Shift+Tab")
            assert await delete.evaluate("(e) => e === document.activeElement")

            print("Keyboard passed; checking layout", flush = True)
            # Long dates, translations and font scaling must fit the viewport;
            # Copy/Edit/Fork/Delete and branch targets must retain their size.
            layouts = []
            for width, locale, scale in [
                (375, "en", ".9375"),
                (320, "en", ".9375"),
                (375, "ru", "1.25"),
                (320, "ar", "1.25"),
            ]:
                await page.set_viewport_size({"width": width, "height": 650})
                await page.goto(f"{url}?branches&locale={locale}&scale={scale}")
                await page.locator(".aui-user-message-root").hover(position = {"x": 4, "y": 4})
                await page.locator("time").wait_for()
                await page.evaluate("document.fonts.ready")
                geometry = await page.evaluate("""() => {
                  const viewport = document.querySelector('.aui-thread-viewport').getBoundingClientRect();
                  const rects = [...document.querySelectorAll('.aui-user-message-footer button')].map(e => {
                    const r = e.getBoundingClientRect();
                    return {x:r.x, right:r.right, width:r.width, timestamp:e.classList.contains('aui-user-message-time-trigger')};
                  });
                  return {left:viewport.left, right:viewport.right, rects};
                }""")
                for rect in geometry["rects"]:
                    assert rect["x"] >= geometry["left"] - 1, geometry
                    assert rect["right"] <= geometry["right"] + 1, geometry
                    if not rect["timestamp"]:
                        assert rect["width"] >= 24, geometry
                layouts.append({"width": width, "locale": locale, "scale": scale})

            # Invalid or synthetic timestamps leave the controls usable.
            for query in ["invalid", "estimated"]:
                await page.goto(url + "?" + query)
                await page.locator(".aui-user-message-root").hover(position = {"x": 4, "y": 4})
                await page.get_by_role("button", name = "Copy", exact = True).wait_for()
                assert await page.locator("time").count() == 0

            touch = await browser.new_page(
                viewport = {"width": 375, "height": 650}, has_touch = True, is_mobile = True
            )
            touch.on("pageerror", lambda error: errors.append(str(error)))
            await touch.goto(url + "?branches")
            await touch.get_by_text("Hello", exact = True).tap()
            await touch.locator("time").wait_for()
            await touch.locator(".aui-user-message-time-trigger").tap()
            await touch.get_by_role("tooltip").wait_for()
            assert await touch.get_by_role("tooltip").inner_text() == await touch.locator(
                ".aui-user-message-time-trigger"
            ).get_attribute("aria-label")
            await touch.locator("#after").tap()
            await paint(touch)
            assert await touch.locator("time").count() == 0
            assert not errors, errors
            print(
                json.dumps(
                    {
                        "streamUpdates": 120,
                        "messages": 500,
                        "extraFormatCalls": 0,
                        "layouts": layouts,
                        "keyboard": "passed",
                        "touch": "passed",
                        "invalidAndEstimated": "passed",
                    },
                    indent = 2,
                )
            )
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
                "tests/fixtures/user-message-time/vite.config.ts",
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
            await check(url)
        finally:
            server.terminate()
            try:
                server.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    asyncio.run(main())
