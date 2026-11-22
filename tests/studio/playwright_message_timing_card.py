# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reply stats card stays open under the pointer so its figures can be copied (#18755)."""

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
CARD = "[data-radix-popper-content-wrapper] > *"
TRIGGER = '[data-slot="message-timing-trigger"]'


async def check(url):
    async with async_playwright() as playwright:
        launch = {"headless": True}
        if os.environ.get("PW_CHROMIUM_EXECUTABLE"):
            launch["executable_path"] = os.environ["PW_CHROMIUM_EXECUTABLE"]
        browser = await playwright.chromium.launch(**launch)
        try:
            page = await browser.new_page(viewport = {"width": 900, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(url)
            # The older reply's bar is autohidden, the path that unmounted its card.
            for reply, prompt in (("older", "457ms"), ("newest", "458ms")):
                await page.mouse.move(890, 50)
                root = page.locator(f'[data-message-id="{reply}"]')
                box = await root.bounding_box()
                await page.mouse.move(box["x"] + 20, box["y"] + 40, steps = 3)
                trigger = root.locator(TRIGGER)
                await trigger.wait_for()
                t = await trigger.bounding_box()
                x = t["x"] + t["width"] / 2
                await page.mouse.move(x, t["y"] + t["height"] / 2, steps = 3)
                card = page.locator(CARD)
                await card.wait_for()
                c = await card.bounding_box()
                await page.mouse.move(x, c["y"] + c["height"] / 2, steps = 8)
                await page.wait_for_timeout(300)
                assert await card.is_visible(), f"{reply}: card closed when the pointer reached it"
                row = await card.locator("div.grid > div").first.bounding_box()
                await page.mouse.move(row["x"] + 2, row["y"] + row["height"] / 2)
                await page.mouse.down()
                await page.mouse.move(
                    row["x"] + row["width"] - 2, row["y"] + row["height"] / 2, steps = 6
                )
                await page.mouse.up()
                selected = await page.evaluate("() => String(window.getSelection())")
                assert prompt in selected, f"{reply}: selected {selected!r}"
                await page.mouse.move(890, 50, steps = 6)
                await page.wait_for_timeout(300)
                assert await card.count() == 0, f"{reply}: card outlived the pointer"
                await page.mouse.click(890, 50)
            assert (
                await page.locator(f'[data-message-id="older"] {TRIGGER}').count() == 0
            ), "older reply's bar stayed mounted after the pointer left the card"
            assert not errors, errors
            print("PASS: both replies keep the stats card open and selectable", flush = True)
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
                "tests/fixtures/message-timing-card/vite.config.ts",
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
