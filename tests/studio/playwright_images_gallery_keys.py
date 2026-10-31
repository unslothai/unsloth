# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise Images gallery keyboard navigation in the real UI without a GPU."""

import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Route, expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, _api_payload, _json


IMAGES = [
    {
        "id": str(index),
        "url": f"/api/inference/images/gallery/{index}/image",
        "prompt": f"Image {index}",
        "negative_prompt": None,
        "width": 64,
        "height": 64,
        "steps": 1,
        "guidance": 1,
        "seed": index,
        "batch_index": 0,
        "batch_size": 1,
        "model": None,
        "created_at": f"2026-01-0{4 - index}T00:00:00Z",
        "pinned": False,
        "archived": False,
    }
    for index in range(1, 4)
]


def main() -> None:
    """Check ordinary arrows, boundary behavior, and the existing Alt+arrow reorder."""
    errors = []
    with sync_playwright() as playwright:
        executable_path = os.environ.get("PW_CHROME")
        browser = playwright.chromium.launch(headless = True, executable_path = executable_path)
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900}, reduced_motion = "reduce"
        )
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');")

        def route_request(route: Route) -> None:
            """Serve deterministic gallery data and reuse the existing image page API fixture."""
            parsed = urlparse(route.request.url)
            if parsed.netloc == "huggingface.co":
                _json(route, [])
            elif parsed.path == "/api/inference/images/gallery":
                _json(route, {"images": IMAGES, "has_more": False})
            elif parsed.path.endswith("/move"):
                image_id = parsed.path.split("/")[-2]
                _json(route, next(item for item in IMAGES if item["id"] == image_id))
            elif parsed.path.startswith("/api/inference/images/gallery/"):
                route.fulfill(status = 404)
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
        expect(tiles).to_have_count(3, timeout = 30_000)

        # Focusing a thumbnail should make arrow keys select its neighbour and keep focus there.
        first, second, third = (tiles.nth(index) for index in range(3))
        first.focus()
        page.keyboard.press("ArrowLeft")
        expect(first).to_be_focused()
        page.keyboard.press("ArrowRight")
        expect(second).to_be_focused()
        expect(second.locator("span.pointer-events-none")).to_have_count(1)
        page.keyboard.press("ArrowRight")
        expect(third).to_be_focused()
        page.keyboard.press("ArrowRight")
        expect(third).to_be_focused()
        page.keyboard.press("ArrowLeft")
        expect(second).to_be_focused()
        expect(second.locator("span.pointer-events-none")).to_have_count(1)

        # Alt+arrow is the existing reorder shortcut, not selection navigation.
        with page.expect_request(lambda request: request.url.endswith("/2/move")) as move:
            page.keyboard.press("Alt+ArrowRight")
        assert move.value.post_data_json == {"after_id": "3"}
        assert not errors, errors
        browser.close()
    print("PASS Images gallery keyboard navigation and Alt+arrow reordering")


if __name__ == "__main__":
    main()
