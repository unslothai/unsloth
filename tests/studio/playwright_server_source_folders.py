# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the production folder manager in a browser with HTTP fixtures, without a model."""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PORT = int(os.environ.get("SMOKE_PORT", "5439"))
BASE_URL = f"http://127.0.0.1:{PORT}"
URL = f"{BASE_URL}/tests/fixtures/server-source-folders/index.html"


async def scene(
    browser,
    width,
    *,
    available = True,
    reject = False,
):
    print(f"Browser scene: width={width}, available={available}, reject={reject}", flush = True)
    context = await browser.new_context(viewport = {"width": width, "height": 800})
    await context.add_init_script(
        "localStorage.setItem('unsloth_auth_token', 'review.' + btoa(JSON.stringify({sub:'unsloth',role:'owner'})) + '.review');"
    )
    requests = []
    linked = []
    output = Path(os.environ.get("SMOKE_OUT", "work/server-source-folders"))
    output.mkdir(parents = True, exist_ok = True)

    async def api(route):
        path = route.request.url.split("/api/", 1)[1]
        if path == "rag/server-source-folders":
            return await route.fulfill(
                json = {
                    "folders": [{"id": "configured-folder", "displayName": "Manuals"}]
                    if available
                    else []
                }
            )
        if path.startswith("rag/linked-folders"):
            return await route.fulfill(json = {"folders": linked})
        if path == "rag/knowledge-bases/knowledge/server-linked-folders":
            requests.append(route.request.post_data_json)
            if reject:
                return await route.fulfill(
                    status = 400, json = {"detail": "Server source folder is unavailable"}
                )
            folder = {
                "id": "linked",
                "displayName": "Manuals",
                "scopeType": "knowledge_base",
                "scopeId": "knowledge",
                "status": "idle",
                "activeJobId": None,
            }
            linked.append(folder)
            return await route.fulfill(
                json = {
                    "linkedFolder": folder,
                    "job": {
                        "id": "job",
                        "linkedFolderId": "linked",
                        "status": "completed",
                        "mode": "sync",
                    },
                }
            )
        if path.startswith("rag/linked-folder-jobs/job/events"):
            return await route.fulfill(
                content_type = "text/event-stream",
                body = 'data: {"type":"complete","id":"job","linkedFolderId":"linked","status":"completed"}\n\n',
            )
        return await route.fulfill(json = {})

    await context.route(f"{BASE_URL}/api/**", api)
    page = await context.new_page()
    pending = set()
    page.on("request", lambda request: pending.add(request.url))
    page.on("requestfinished", lambda request: pending.discard(request.url))
    page.on("requestfailed", lambda request: pending.discard(request.url))
    page.on("pageerror", lambda error: print(f"Browser error: {error}", flush = True))
    page.on(
        "console",
        lambda message: print(message.text, flush = True) if message.type == "error" else None,
    )
    await page.goto(URL, wait_until = "domcontentloaded", timeout = 90_000)
    print("Page loaded", flush = True)
    try:
        await page.wait_for_selector("button", timeout = 60_000)
    except Exception:
        print(f"Pending requests: {sorted(pending)}", flush = True)
        print(await page.locator("body").inner_text(), flush = True)
        raise
    add = page.get_by_role("button", name = "Add folder", exact = True)
    if not available:
        await expect(add).to_be_disabled()
    else:
        await expect(add).to_be_enabled()
        await add.click()
        dialog = page.get_by_role("dialog", name = "Link a server folder")
        await expect(dialog).to_be_visible()
        assert await dialog.evaluate("el => el.getBoundingClientRect().right <= innerWidth")
        await page.screenshot(path = str(output / f"{width}-picker.png"), animations = "disabled")
        await dialog.get_by_role("button", name = "Manuals").click()
        await expect(dialog).not_to_be_visible()
        assert requests == [{"serverFolderId": "configured-folder"}]
        if reject:
            await expect(page.get_by_text("Could not link folder", exact = True)).to_be_visible()
            assert linked == []
        else:
            await expect(page.get_by_text("Manuals", exact = True)).to_be_visible()
        await page.screenshot(path = str(output / f"{width}-{'error' if reject else 'linked'}.png"))
    await context.close()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless = True, channel = os.environ.get("PLAYWRIGHT_CHANNEL") or None
        )
        try:
            for width in [390, 1280]:
                await scene(browser, width)
            await scene(browser, 1280, reject = True)
            await scene(browser, 1280, available = False)
        finally:
            await browser.close()
    print(json.dumps({"scenes": 4, "status": "passed"}))


if __name__ == "__main__":
    server = start_vite(PORT)
    try:
        wait_for_smoke_page(URL, "main.tsx", proc = server)
        asyncio.run(main())
    finally:
        stop_process(server)
