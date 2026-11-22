# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Route, expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, _api_payload, _json


def main() -> None:
    common = {
        "negative_prompt": None,
        "width": 64,
        "height": 64,
        "steps": 1,
        "guidance": 1,
        "seed": 1,
        "batch_index": 0,
        "batch_size": 1,
        "model": None,
        "created_at": 100,
        "pinned": False,
        "archived": False,
    }
    images = [
        {
            **common,
            "id": "new",
            "url": "/image-new",
            "prompt": "New image",
            "generation_seconds": 2.25,
            "generation_count": 2,
        },
        {**common, "id": "old", "url": "/image-old", "prompt": "Old image"},
    ]
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless = True, executable_path = os.environ.get("PW_CHROME")
        )
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');")

        def route_request(route: Route) -> None:
            parsed = urlparse(route.request.url)
            if parsed.netloc == "huggingface.co":
                _json(route, [])
            elif parsed.path == "/api/inference/images/gallery":
                _json(route, {"images": images, "has_more": False})
            elif parsed.path.startswith("/api/"):
                _json(
                    route, _api_payload(parsed.path, parse_qs(parsed.query), full_footprint = False)
                )
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{BASE_URL}/images", wait_until = "domcontentloaded")
        tiles = page.locator("[data-image-id] > button")
        expect(tiles).to_have_count(2, timeout = 30_000)
        page.get_by_role("button", name = "Recipe", exact = True).click()
        expect(page.get_by_text("2.3 s (batch of 2)")).to_be_visible()
        page.keyboard.press("Escape")
        tiles.nth(1).click()
        page.get_by_role("button", name = "Recipe", exact = True).click()
        expect(page.get_by_text("Time", exact = True)).to_have_count(0)
        assert not errors, errors
        browser.close()
    print("PASS image generation duration and legacy gallery records")


if __name__ == "__main__":
    main()
