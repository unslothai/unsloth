# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opening a Library item must not reorder Suggested or move its scroll position until the Library
is left."""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5227"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
# Suggested's default limit.
ITEMS = 40
SCROLL_TOP = 600
NOW_MS = 1_760_000_000_000


def info(message: str) -> None:
    print(f"[library-suggested-order] {message}", flush = True)


def _png() -> bytes:
    rows = b"".join(b"\x00" + bytes((x * 16) % 256 for x in range(8 * 3)) for _ in range(8))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


PNG = _png()


def _item(i: int) -> dict:
    return {
        "id": f"image:img{i:02d}",
        "name": f"Library image {i:02d}",
        "fileName": f"img{i:02d}.png",
        "source": "generated",
        "contentType": "image/png",
        "sizeBytes": len(PNG),
        "createdAt": NOW_MS - i * 60_000,
        "updatedAt": NOW_MS - i * 60_000,
        "fileUrl": f"/api/library/items/file/img{i:02d}.png",
        "threadId": None,
        "threadTitle": None,
        "textOnly": False,
        "favorite": False,
        "folderId": None,
        "openedAt": None,
        "model": None,
    }


class MockApi:
    """Saves openedAt, so a refresh returns the bumped activity as the backend does."""

    def __init__(self) -> None:
        self.items = [_item(i) for i in range(ITEMS)]
        self.opened: list[str] = []
        self.listings = 0

    def handle(self, route) -> None:
        request = route.request
        path = urlparse(request.url).path
        if path == "/api/auth/status":
            return self._json(
                route,
                {
                    "initialized": True,
                    "requires_password_change": False,
                    "login_mode": "single",
                    "full_access": True,
                },
            )
        if path == "/api/library":
            self.listings += 1
            return self._json(route, {"items": self.items, "folders": []})
        if path == "/api/library/items/opened" and request.method == "POST":
            item_id = json.loads(request.post_data or "{}")["id"]
            self.opened.append(item_id)
            for item in self.items:
                if item["id"] == item_id:
                    item["openedAt"] = NOW_MS + 60_000 * len(self.opened)
            return self._json(route, {})
        if path == "/api/library/items/thumbnail" or path.startswith("/api/library/items/file/"):
            return route.fulfill(status = 200, content_type = "image/png", body = PNG)
        route.fulfill(status = 404, content_type = "application/json", body = "{}")

    @staticmethod
    def _json(route, payload: dict) -> None:
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(payload))


LAYOUT = """() => {
  const scroller = document.querySelector('[data-slot="sidebar-inset"]');
  const cards = {};
  for (const button of document.querySelectorAll('main button[aria-label^="Library image"]')) {
    const r = button.getBoundingClientRect();
    cards[button.getAttribute('aria-label')] = [Math.round(r.x), Math.round(r.y)];
  }
  return { scrollTop: Math.round(scroller.scrollTop), cards };
}"""


def layout(page) -> dict:
    return page.evaluate(LAYOUT)


def assert_unchanged(before: dict, after: dict, step: str) -> None:
    assert (
        after["scrollTop"] == before["scrollTop"]
    ), f"{step}: the Library scrolled from {before['scrollTop']} to {after['scrollTop']}"
    moved = [name for name, at in before["cards"].items() if after["cards"].get(name) != at]
    assert (
        not moved
    ), f"{step}: {len(moved)} of {len(before['cards'])} cards moved, e.g. {moved[:3]}"


def reading_order(state: dict) -> list[str]:
    cards = state["cards"]
    return sorted(cards, key = lambda name: (cards[name][1], cards[name][0]))


def open_suggested(page) -> None:
    page.goto(f"{BASE}/library?show=suggested", timeout = 120_000)
    wait_for_cards(page)


def wait_for_cards(page) -> None:
    page.wait_for_function(
        f"() => document.querySelectorAll('main button[aria-label^=\"Library image\"]').length === {ITEMS}",
        timeout = 120_000,
    )
    # Cards near the viewport load thumbnails; wait so the layout is final.
    page.wait_for_function(
        "() => document.querySelectorAll('main button[aria-label^=\"Library image\"] img').length >= 10",
        timeout = 60_000,
    )


def check(page, api: MockApi) -> None:
    open_suggested(page)
    page.evaluate(
        f"() => document.querySelector('[data-slot=\"sidebar-inset\"]').scrollTo(0, {SCROLL_TOP})"
    )
    page.wait_for_timeout(1000)
    before = layout(page)
    assert before["scrollTop"] == SCROLL_TOP, f"the Library only scrolls to {before['scrollTop']}"

    # A first-column card's DOM node survives a move to the top, and scroll anchoring follows it.
    first_column = min(x for x, _ in before["cards"].values())
    target = min(
        (name for name, (x, y) in before["cards"].items() if x == first_column and y > 250),
        key = lambda name: before["cards"][name][1],
    )
    target_id = f"image:img{target.rsplit(' ', 1)[1]}"
    assert target != "Library image 00", "the target must not already rank first"
    info(f"opening {target} at {before['cards'][target]}")

    page.locator(f'main button[aria-label="{target}"]').click()
    page.get_by_role("dialog").wait_for(timeout = 30_000)
    for _ in range(100):
        if target_id in api.opened:
            break
        page.wait_for_timeout(50)
    assert api.opened == [target_id], f"the viewer saved openedAt for {api.opened}"
    page.wait_for_timeout(500)
    assert_unchanged(before, layout(page), "viewer open")

    listings = api.listings
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    for _ in range(100):
        if api.listings > listings:
            break
        page.wait_for_timeout(50)
    assert api.listings > listings, "the focus refresh never listed the Library"
    page.wait_for_timeout(500)
    assert_unchanged(before, layout(page), "focus refresh with the saved openedAt")

    page.keyboard.press("Escape")
    page.get_by_role("dialog").wait_for(state = "hidden", timeout = 30_000)
    page.wait_for_timeout(500)
    assert_unchanged(before, layout(page), "viewer closed")

    sections = page.get_by_role("navigation", name = "Library sections")
    sections.get_by_role("button", name = "Images", exact = True).click()
    page.wait_for_url("**show=images**", timeout = 30_000)
    sections.get_by_role("button", name = "Suggested", exact = True).click()
    page.wait_for_url("**show=suggested**", timeout = 30_000)
    page.wait_for_timeout(500)
    assert reading_order(layout(page)) == reading_order(before), "a tab switch reordered Suggested"

    page.get_by_text("Model hub", exact = True).first.click()
    page.wait_for_url("**/hub**", timeout = 30_000)
    # Until the hub replaces the grid, a click back cancels the navigation and the user never left.
    page.wait_for_function(
        "() => !document.querySelector('main button[aria-label^=\"Library image\"]')",
        timeout = 30_000,
    )
    page.get_by_text("Library", exact = True).first.click()
    page.wait_for_url("**/library**", timeout = 30_000)
    wait_for_cards(page)
    first = reading_order(layout(page))[0]
    assert first == target, f"back in the Library, {first} ranks first, not the opened {target}"


def main() -> int:
    proc = None
    try:
        if not _EXTERNAL:
            proc = start_vite(PORT)
            wait_for_smoke_page(f"{BASE}/library", "/src/main.tsx", proc = proc, info = info)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless = True, args = chromium_launch_args())
            try:
                api = MockApi()
                context = browser.new_context(
                    viewport = {"width": 1280, "height": 900}, locale = "en-US"
                )
                context.add_init_script(
                    "localStorage.setItem('unsloth_auth_token','x.eyJleHAiOjk5OTk5OTk5OTl9.y');"
                    "localStorage.setItem('unsloth_auth_refresh_token','r');"
                )
                context.route(
                    lambda url: urlparse(url).path.startswith(("/api/", "/v1/")), api.handle
                )
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                # Absorb the dev server's dependency-optimization reload before the check.
                open_suggested(page)
                page.wait_for_timeout(2000)
                check(page, api)
            finally:
                browser.close()
        # Unmocked endpoints answer 404.
        unexpected = [error for error in errors if not error.startswith("Request failed (404)")]
        assert not unexpected, unexpected
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
