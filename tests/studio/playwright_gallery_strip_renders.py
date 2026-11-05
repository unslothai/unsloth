# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render counts while the Images and Video strips page in and their thumbnails arrive.

A thumbnail arriving or a page being appended must re-render only the cards it changes, not the
actions menu of every card already in the strip. Then a card from a later page is deleted through
its own menu, which must remove that card and only that card. Counts come from a React DevTools
hook installed before the page loads, using DevTools' rule for which fibers rendered.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
# Six pages of 50, so the strip appends five times while thumbnails stream in.
ITEMS = 300

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
    print(f"[gallery-strip-renders] {message}", flush = True)


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


def _image(i: int) -> dict:
    return {
        "id": f"img{i:05d}",
        "url": f"/api/inference/images/gallery/file/img{i:05d}.png",
        "prompt": f"prompt number {i}",
        "negative_prompt": None,
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance": 0.0,
        "seed": i,
        "batch_index": 0,
        "batch_size": 1,
        "model": "stabilityai/sdxl-turbo",
        "created_at": 1760000000 - i,
    }


def _video(i: int) -> dict:
    return {
        "id": f"vid{i:05d}",
        "url": f"/api/inference/video/gallery/file/vid{i:05d}.mp4",
        "prompt": f"clip {i}",
        "negative_prompt": None,
        "width": 832,
        "height": 480,
        "num_frames": 33,
        "fps": 16,
        "duration_s": 2.0,
        "steps": 4,
        "guidance": 1.0,
        "seed": i,
        "has_audio": False,
        "model": "Wan-AI/Wan2.1-T2V-1.3B",
        "created_at": 1760000000 - i,
    }


class MockApi:
    """Paged galleries that honour deletes, so a refill after a delete sees the shorter shelf."""

    def __init__(self) -> None:
        self.images = [_image(i) for i in range(ITEMS)]
        self.videos = [_video(i) for i in range(ITEMS)]
        self.deleted: list[str] = []

    def handle(self, route) -> None:
        request = route.request
        parsed = urlparse(request.url)
        path, query = parsed.path, parse_qs(parsed.query)

        def page(items: list[dict], key: str) -> None:
            if query.get("archived", ["false"])[0] == "true":
                return self._json(route, {key: [], "has_more": False})
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["50"])[0])
            chunk = items[offset : offset + limit]
            self._json(route, {key: chunk, "has_more": offset + limit < len(items)})

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
        if path == "/api/system/hardware":
            return self._json(
                route, {"video_supported": True, "gpu": {"gpu_name": "Mock", "vram_total_gb": 24}}
            )
        if path.startswith("/api/inference/images/gallery/file/") or path.startswith("/v1/videos/"):
            return route.fulfill(status = 200, content_type = "image/png", body = PNG)
        for prefix, items, key in (
            ("/api/inference/images/gallery", self.images, "images"),
            ("/api/inference/video/gallery", self.videos, "videos"),
        ):
            if path == prefix and request.method == "GET":
                return page(items, key)
            if path.startswith(prefix + "/") and request.method == "DELETE":
                item_id = path.rsplit("/", 1)[1]
                self.deleted.append(item_id)
                items[:] = [item for item in items if item["id"] != item_id]
                return self._json(route, {})
        route.fulfill(status = 404, content_type = "application/json", body = "{}")

    @staticmethod
    def _json(route, payload: dict) -> None:
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(payload))


# A card's menu renders about twice as it pages in (once settling after mount, once when its
# thumbnail lands). Re-rendering every card per page append or per thumbnail is many times this.
MENU_RENDER_BOUND = 4 * ITEMS


def open_strip(page, route: str, tiles: str) -> None:
    page.goto(f"{BASE}{route}", timeout = 120_000)
    page.wait_for_function(
        f"() => document.querySelectorAll('{tiles}').length >= 50", timeout = 120_000
    )
    page.wait_for_function(
        f"() => document.querySelectorAll('{tiles} img').length >= 10", timeout = 60_000
    )


def check_strip(page, api: MockApi, *, route: str, tile_attr: str, noun: str, victim: str) -> int:
    tiles = f"[{tile_attr}]"
    open_strip(page, route, tiles)
    page.wait_for_timeout(1000)
    page.evaluate("() => { window.__renders = {}; }")
    for _ in range(600):
        count, left = page.evaluate(
            f"""() => {{
              const strip = document.querySelector('{tiles}')?.parentElement;
              if (!strip) return [0, 1];
              strip.scrollLeft += 600;
              return [document.querySelectorAll('{tiles}').length,
                      strip.scrollWidth - strip.scrollLeft - strip.clientWidth];
            }}"""
        )
        page.wait_for_timeout(30)
        if count >= ITEMS and left < 2:
            break
    page.wait_for_timeout(1000)
    counts = page.evaluate(
        "() => { const c = window.__renders; window.__renders = null; return c; }"
    )
    # A reload mid-measure (vite re-optimizing dependencies) resets the counter to null.
    assert counts is not None, f"{route}: the page reloaded while it was being measured"
    total = page.evaluate(f"() => document.querySelectorAll('{tiles}').length")
    loaded = page.evaluate(f"() => document.querySelectorAll('{tiles} img').length")
    menus = counts.get("GalleryItemMenu", 0)
    info(
        json.dumps({"route": route, "tiles": total, "thumbnails": loaded, "GalleryItemMenu": menus})
    )
    # Fails the run, not the bound, when the strip never paged or nothing loaded.
    assert total == ITEMS, f"{route}: the strip shows {total} of {ITEMS} items after scrolling"
    assert loaded >= 50, f"{route}: only {loaded} thumbnails loaded, so little was measured"
    # A card from the last page, after five appends: its own menu must delete it and only it.
    card = page.locator(f"[{tile_attr}='{victim}']")
    card.scroll_into_view_if_needed()
    card.get_by_role("button", name = f"More actions for this {noun}").click()
    page.get_by_role("menuitem", name = "Delete").click()
    page.wait_for_function(
        f"() => !document.querySelector(\"[{tile_attr}='{victim}']\")", timeout = 30_000
    )
    page.wait_for_timeout(500)
    assert api.deleted[-1:] == [
        victim
    ], f"{route}: the menu deleted {api.deleted[-1:]}, not {victim}"
    left = page.evaluate(f"() => document.querySelectorAll('{tiles}').length")
    assert left == ITEMS - 1, f"{route}: {left} cards left after deleting one of {ITEMS}"
    return menus


def main() -> int:
    proc = None
    try:
        if not _EXTERNAL:
            proc = start_vite(PORT)
            wait_for_smoke_page(f"{BASE}/images", "/src/main.tsx", proc = proc, info = info)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless = True, args = chromium_launch_args())
            try:
                api = MockApi()
                context = browser.new_context(viewport = {"width": 1280, "height": 900})
                context.add_init_script(
                    "localStorage.setItem('unsloth_auth_token','x.eyJleHAiOjk5OTk5OTk5OTl9.y');"
                    "localStorage.setItem('unsloth_auth_refresh_token','r');"
                )
                context.add_init_script(RENDER_COUNTER)
                context.route(
                    lambda url: urlparse(url).path.startswith(("/api/", "/v1/")), api.handle
                )
                page = context.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                # A first visit to each page lets the dev server finish optimizing dependencies, whose
                # reload would otherwise land in the middle of a measurement.
                open_strip(page, "/images", "[data-image-id]")
                open_strip(page, "/video", "[data-clip-id]")
                page.wait_for_timeout(2000)
                menus = {
                    "/images": check_strip(
                        page,
                        api,
                        route = "/images",
                        tile_attr = "data-image-id",
                        noun = "image",
                        victim = "img00287",
                    ),
                    "/video": check_strip(
                        page,
                        api,
                        route = "/video",
                        tile_attr = "data-clip-id",
                        noun = "video",
                        victim = "vid00287",
                    ),
                }
            finally:
                browser.close()
        # Endpoints this check does not mock answer 404, which the app reports and moves past.
        unexpected = [error for error in errors if not error.startswith("Request failed (404)")]
        assert not unexpected, unexpected
        over = {route: count for route, count in menus.items() if count > MENU_RENDER_BOUND}
        assert not over, (
            f"card menus rendered {over} times while {ITEMS} cards paged in (bound "
            f"{MENU_RENDER_BOUND}): a thumbnail or page append is re-rendering every card"
        )
        info("ok")
        return 0
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
