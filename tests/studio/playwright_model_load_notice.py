# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the real chat load hook, Sonner toast, and inline status without a model.

Start Vite from studio/frontend:
  npm run dev -- --config tests/fixtures/model-load-notice/vite.config.ts --port 5197
Then, from the repository root with Playwright and Chromium installed:
  python tests/studio/playwright_model_load_notice.py
PW_BROWSER_EXECUTABLE optionally selects an installed Chromium executable.

For a negative control, point Vite's PW_SOURCE_FRONTEND_DIR at the base checkout's
studio/frontend directory and run --case late-load-progress. The original hook
recreates the cancelled toast when the held progress response arrives.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from playwright.async_api import Error, async_playwright, expect


class LoadFixture:
    def __init__(
        self,
        page,
        base_url,
        *,
        hold_progress = False,
    ):
        self.page = page
        self.base_url = base_url
        self.loads = asyncio.Queue()
        self.progress = asyncio.Queue()
        self.hold_progress = hold_progress
        self.aborted = asyncio.Event()
        self.unloads = 0
        self.loaded = False
        self.errors = []
        self.held_routes = []

    async def open(self, *, download = False):
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("requestfailed", self.request_failed)
        await self.page.route(f"{self.base_url}/api/**", self.route)
        await self.page.goto(self.base_url + ("/?download=1" if download else "/"))
        await expect(self.page.get_by_role("button", name = "Load model", exact = True)).to_be_visible()

    def request_failed(self, request):
        if urlsplit(request.url).path == "/api/inference/load":
            self.aborted.set()

    async def route(self, route):
        path = urlsplit(route.request.url).path
        if path == "/api/inference/load":
            self.held_routes.append(route)
            await self.loads.put(route)
            return
        if path in ("/api/inference/load-progress", "/api/models/download-progress"):
            if self.hold_progress:
                self.hold_progress = False
                self.held_routes.append(route)
                await self.progress.put(route)
                return
            body = self.progress_body(path)
        elif path == "/api/inference/unload":
            self.unloads += 1
            self.loaded = False
            body = {"status": "unloaded"}
        elif path == "/api/inference/status":
            body = {
                "active_model": "fixture/model" if self.loaded else None,
                "loaded": ["fixture/model"] if self.loaded else [],
                "loading": [],
                "is_gguf": True,
                "gguf_variant": "Q4_K_M",
            }
        elif path == "/api/models/list":
            body = {"models": []}
        elif path == "/api/models/loras":
            body = {"loras": []}
        elif path == "/api/inference/validate":
            body = {"valid": True, "is_gguf": True}
        elif path == "/api/inference/active-generations":
            body = {"active_generations": []}
        elif "gpu" in path:
            body = {"gpus": []}
        else:
            body = {}
        await route.fulfill(json = body)

    async def close(self):
        # Resolve intercepted requests even when the test deliberately leaves a load
        # pending. Some may already have settled or been aborted by the application.
        for route in self.held_routes:
            try:
                await route.abort()
            except Error:
                pass
        await self.page.unroute_all(behavior = "ignoreErrors")
        await self.page.close()

    @staticmethod
    def progress_body(path, fraction = 0.2):
        if path == "/api/inference/load-progress":
            return {
                "phase": "mmap",
                "bytes_total": 1_000_000_000,
                "bytes_loaded": int(fraction * 1_000_000_000),
                "fraction": fraction,
            }
        return {
            "expected_bytes": 1_000_000_000,
            "downloaded_bytes": int(fraction * 1_000_000_000),
            "progress": fraction,
        }

    async def start(self):
        await self.page.get_by_role("button", name = "Load model", exact = True).click()
        request = await asyncio.wait_for(self.loads.get(), timeout = 10)
        await expect(self.page.get_by_test_id("lifecycle")).to_have_text("Busy")
        return request

    async def cancel(self):
        # Also recognizes the old label, so the cancellation regression can run on base.
        await self.page.get_by_role("button", name = re.compile(r"^Cancel(?: loading)?$")).click()
        await expect(self.page.get_by_test_id("lifecycle")).to_have_text("Idle")
        await expect(self.page.locator(".chat-model-load-toast")).to_have_count(0)
        await expect(self.page.get_by_test_id("inline-status")).to_be_empty()
        assert self.unloads == 1


async def cancel_aborts(fixture, artifacts):
    page = fixture.page
    await fixture.open()
    await fixture.start()
    await expect(page.locator(".chat-model-load-toast [data-close-button]")).to_have_count(0)
    await page.screenshot(path = str(artifacts / "loading.png"))
    await fixture.cancel()
    await asyncio.wait_for(fixture.aborted.wait(), timeout = 5)
    assert not fixture.errors, fixture.errors


async def hide_then_stop(fixture, artifacts):
    page = fixture.page
    await fixture.open()
    await fixture.start()
    await page.get_by_role("button", name = "Hide", exact = True).click()
    await expect(page.locator(".chat-model-load-toast")).to_have_count(0)
    await expect(page.get_by_test_id("inline-status")).to_contain_text("Loading model")
    assert fixture.unloads == 0 and not fixture.aborted.is_set()
    await page.get_by_role("button", name = "Stop", exact = True).click()
    await expect(page.get_by_test_id("lifecycle")).to_have_text("Idle")
    await expect(page.get_by_test_id("inline-status")).to_be_empty()
    await asyncio.wait_for(fixture.aborted.wait(), timeout = 5)
    assert fixture.unloads == 1
    assert not fixture.errors, fixture.errors


async def late_progress(
    fixture,
    artifacts,
    *,
    download = False,
):
    page = fixture.page
    await fixture.open(download = download)
    await fixture.start()
    old_progress = await asyncio.wait_for(fixture.progress.get(), timeout = 5)
    await fixture.cancel()
    await fixture.start()
    # Deliver a stale poll only once a replacement load is active. The old hook checked
    # merely that SOME load existed, then recreated the cancelled toast with its old ID.
    await old_progress.fulfill(
        json = fixture.progress_body(urlsplit(old_progress.request.url).path, 0.9)
    )
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    await expect(page.locator(".chat-model-load-toast")).to_have_count(1)
    await expect(page.locator(".chat-model-load-toast")).not_to_contain_text("90%")
    await expect(page.get_by_test_id("lifecycle")).to_have_text("Busy")
    assert not fixture.errors, fixture.errors


async def settled(
    fixture,
    artifacts,
    *,
    failure = False,
):
    page = fixture.page
    await fixture.open()
    request = await fixture.start()
    fixture.loaded = not failure
    await request.fulfill(
        status = 500 if failure else 200,
        json = {"detail": "Fixture load failed"}
        if failure
        else {
            "status": "loaded",
            "model": "fixture/model",
            "display_name": "Fixture",
            "is_gguf": True,
            "max_seq_length": 4096,
        },
    )
    await expect(page.get_by_test_id("lifecycle")).to_have_text("Idle")
    await expect(page.get_by_role("button", name = "Hide", exact = True)).to_have_count(0)
    await expect(page.get_by_role("button", name = "Cancel loading", exact = True)).to_have_count(0)
    await expect(page.locator(".chat-model-load-toast")).to_have_count(0)
    expected_type = "error" if failure else "success"
    await expect(page.locator(f"[data-sonner-toast][data-type='{expected_type}']")).to_have_count(1)
    assert not fixture.errors, fixture.errors


async def layout(fixture, artifacts):
    page = fixture.page
    await page.set_viewport_size({"width": 375, "height": 700})
    await fixture.open()
    await page.evaluate("document.documentElement.classList.add('dark')")
    await fixture.start()
    toast = page.locator(".chat-model-load-toast")
    await expect(toast.get_by_role("progressbar")).to_be_visible()
    boxes = await toast.evaluate("""el => {
      const box = node => { const r = node.getBoundingClientRect(); return {x:r.x,y:r.y,right:r.right,bottom:r.bottom}; };
      return {toast:box(el),content:box(el.querySelector('[data-content]')),
        cancel:box(el.querySelector('[data-cancel]')),hide:box(el.querySelector('[data-action]'))};
    }""")
    assert boxes["cancel"]["y"] >= boxes["content"]["bottom"]
    assert boxes["hide"]["y"] >= boxes["content"]["bottom"]
    assert boxes["cancel"]["right"] <= boxes["hide"]["x"]
    assert boxes["toast"]["x"] >= 0 and boxes["toast"]["right"] <= 375
    await page.screenshot(path = str(artifacts / "mobile-dark.png"))
    assert not fixture.errors, fixture.errors


async def main(args):
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents = True, exist_ok = True)
    cases = {
        "cancel": cancel_aborts,
        "hide": hide_then_stop,
        "late-load-progress": late_progress,
        "late-download-progress": lambda *a: late_progress(*a, download = True),
        "success": settled,
        "failure": lambda *a: settled(*a, failure = True),
        "layout": layout,
    }
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless = True, executable_path = os.environ.get("PW_BROWSER_EXECUTABLE") or None
        )
        try:
            for name, run in cases.items():
                if args.case and args.case != name:
                    continue
                page = await browser.new_page(viewport = {"width": 1000, "height": 700})
                fixture = LoadFixture(
                    page, args.url.rstrip("/"), hold_progress = name.startswith("late-")
                )
                try:
                    await run(fixture, artifacts)
                    print(f"PASS {name}", flush = True)
                finally:
                    await fixture.close()
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--url", default = "http://127.0.0.1:5197")
    parser.add_argument("--artifacts", default = "/tmp/unsloth-model-load-notice")
    parser.add_argument("--case")
    asyncio.run(main(parser.parse_args()))
