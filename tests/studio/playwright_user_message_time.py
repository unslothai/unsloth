# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""User-message timestamps on real assistant-ui primitives and app CSS; no backend needed."""

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
            # Leaving the message with the mouse must not unmount a focused control.
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

            # Reverse traversal must enter the controls, not skip to the preceding page button.
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

            await page.goto(url + "?popup")
            await page.locator(".aui-assistant-message-root").focus()
            await page.get_by_role("button", name = "More", exact = True).click()
            await page.get_by_role("menuitem", name = "Menu action").wait_for()
            await page.mouse.move(799, 640)
            await paint(page)
            await page.evaluate("""async () => {
              const root = document.querySelector('.aui-assistant-message-root');
              const original = root.querySelector.bind(root);
              window.popupQueries = 0;
              root.querySelector = (...args) => {
                window.popupQueries++;
                return original(...args);
              };
              for (let i = 0; i < 8; i++) await new Promise(requestAnimationFrame);
            }""")
            assert await page.evaluate("window.popupQueries") == 0
            await page.keyboard.press("Escape")
            await paint(page)
            assert await page.get_by_role("button", name = "More", exact = True).count() == 0

            print("Keyboard passed; checking layout", flush = True)
            layouts = []
            # (viewport width, locale, UI font scale, browser Interface Scale, narrow count)
            for width, locale, scale, interface, *narrow in [
                (375, "en", ".9375", 1),
                # At the 12px minimum --ui-space-scale is 0.8, so a target sized as a multiple of it drops under 24px.
                (375, "en", ".75", 1),
                (320, "en", ".9375", 1),
                (375, "ru", "1.25", 1),
                (320, "ar", "1.25", 1),
                # 50% Interface Scale: a target clamped at 24px reaches past the gap and over its neighbour.
                (375, "en", ".75", 0.5),
                # 200% Interface Scale: once the picker wraps the target reaches across the row gap.
                (375, "en", ".75", 2),
                (320, "en", ".75", 2),
                # A narrow chat font is where each chevron's target reaches furthest toward the count.
                (375, "en", ".75", 1, "narrow count"),
            ]:
                await page.set_viewport_size({"width": width, "height": 650})
                await page.goto(f"{url}?branches&locale={locale}&scale={scale}")
                if narrow:
                    await page.add_style_tag(
                        content = ".aui-user-branch-picker > span { font-size: 2px !important; }"
                    )
                if interface != 1:
                    await page.evaluate(
                        "s => document.documentElement.style.setProperty('--ui-interface-scale', s)",
                        str(interface),
                    )
                await page.locator(".aui-user-message-root").hover(position = {"x": 4, "y": 4})
                await page.locator("time").wait_for()
                await page.evaluate("document.fonts.ready")
                geometry = await page.evaluate("""() => {
                  const viewport = document.querySelector('.aui-thread-viewport').getBoundingClientRect();
                  const rects = [...document.querySelectorAll('.aui-user-message-footer button')].map(e => {
                    const r = e.getBoundingClientRect();
                    // The target is the box plus a positioned ::before, if the control extends it with one.
                    let t = {x:r.x, right:r.right, y:r.y, bottom:r.bottom};
                    const before = getComputedStyle(e, '::before');
                    if (before.content !== 'none' && before.position === 'absolute') {
                      const left = r.x + e.clientLeft, top = r.y + e.clientTop;
                      t = {
                        x: Math.min(t.x, left + parseFloat(before.left)),
                        right: Math.max(t.right, left + e.clientWidth - parseFloat(before.right)),
                        y: Math.min(t.y, top + parseFloat(before.top)),
                        bottom: Math.max(t.bottom, top + e.clientHeight - parseFloat(before.bottom)),
                      };
                    }
                    // The middle of every edge of the target has to reach this control when pressed, not
                    // whatever is painted over it there. Not the corners: hit testing follows a round
                    // control's border-radius.
                    const missed = [];
                    for (const [fx, fy] of [[0.5, 0.5], [0, 0.5], [1, 0.5], [0.5, 0], [0.5, 1]]) {
                      const px = t.x + 0.5 + fx * (t.right - t.x - 1), py = t.y + 0.5 + fy * (t.bottom - t.y - 1);
                      const hit = document.elementFromPoint(px, py);
                      if (!hit || hit.closest('button') !== e) missed.push([px, py, hit && hit.className]);
                    }
                    const cs = getComputedStyle(e);
                    return {...t, width:t.right - t.x, height:t.bottom - t.y, missed,
                      box:{width:r.width, height:r.height, padding:[cs.paddingTop, cs.paddingRight, cs.paddingBottom, cs.paddingLeft]},
                      chevron:e.classList.contains('aui-branch-chevron-btn'),
                      timestamp:e.classList.contains('aui-user-message-time-trigger')};
                  });
                  return {left:viewport.left, right:viewport.right, rects};
                }""")
                for rect in geometry["rects"]:
                    assert rect["x"] >= geometry["left"] - 1, geometry
                    assert rect["right"] <= geometry["right"] + 1, geometry
                    assert not rect["missed"], (
                        "part of this target does not reach it",
                        rect,
                        geometry,
                    )
                    if not rect["timestamp"]:
                        assert rect["width"] >= 24 * interface - 0.01, geometry
                        assert rect["height"] >= 24 * interface - 0.01, geometry
                    if rect["chevron"]:
                        # The visible box draws the focus ring; square and unpadded keeps it a centred circle.
                        box = rect["box"]
                        assert abs(box["width"] - box["height"]) <= 0.01, rect
                        assert set(box["padding"]) == {"0px"}, rect
                # In an overlap the later target wins the click; wrapped layouts mean compare boxes, not x alone.
                rects = geometry["rects"]
                for i, a in enumerate(rects):
                    for b in rects[i + 1 :]:
                        overlap_x = min(a["right"], b["right"]) - max(a["x"], b["x"])
                        overlap_y = min(a["bottom"], b["bottom"]) - max(a["y"], b["y"])
                        assert overlap_x <= 0.01 or overlap_y <= 0.01, (a, b, geometry)
                layouts.append(
                    {
                        "width": width,
                        "locale": locale,
                        "scale": scale,
                        "interface": interface,
                        "narrow": bool(narrow),
                    }
                )

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
