# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A slow Video eject shows progress, sends one request, and allows retry on failure."""

import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, _api_payload, _json

REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
IDLE = {
    "loaded": False,
    "repo_id": None,
    "family": None,
    "base_repo": None,
    "device": None,
    "dtype": None,
    "vae_tiling": False,
    "speed_optims": [],
    "has_audio": False,
    "supports_cfg": True,
}


def main():
    status = {**IDLE, "loaded": True, "repo_id": REPO_ID, "base_repo": REPO_ID, "family": "wan2.2"}
    generate = {"active": True, "phase": "denoising", "step": 1, "total": 4}
    unloads = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True, executable_path = os.getenv("PW_CHROME"))
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test')")

        def route_request(route):
            url = urlparse(route.request.url)
            if url.netloc != urlparse(BASE_URL).netloc:
                _json(route, [])
            elif url.path == "/api/inference/video/status":
                _json(route, status)
            elif url.path == "/api/inference/video/load-progress":
                _json(route, {"phase": None, "downloaded_bytes": 0, "total_bytes": 0})
            elif url.path == "/api/inference/video/generate-progress":
                _json(route, generate)
            elif url.path == "/api/inference/video/gallery":
                _json(route, {"videos": [], "has_more": False})
            elif url.path == "/api/inference/video/unload":
                assert route.request.method == "POST"
                unloads.append(route)
            elif url.path.startswith("/api/"):
                _json(route, _api_payload(url.path, parse_qs(url.query), full_footprint = False))
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.goto(f"{BASE_URL}/video", wait_until = "domcontentloaded")
        picker = page.locator('[data-tour="video-model"]')
        shortcut = picker.locator("[data-eject-hit]")
        menu_eject = page.get_by_role("button", name = "Eject model", exact = True)
        unloading = page.get_by_text("Unloading model", exact = True)
        expect(shortcut).to_be_visible(timeout = 30_000)
        # The mount probe must have resumed the generation, or the eject parks it behind the held unload.
        expect(page.get_by_role("button").filter(has_text = "Cancel").first).to_be_visible()

        # Two clicks in one event turn also exercise the guard before React updates the DOM.
        with page.expect_request("**/api/inference/video/unload"):
            shortcut.evaluate("(el) => { el.click(); el.click(); }")
        expect(unloading).to_be_visible()
        # A generation finishing during teardown must not clear the indicator.
        generate = {"active": False, "phase": "completed", "step": 4, "total": 4, "video": None}
        with page.expect_response("**/api/inference/video/generate-progress"):
            pass
        page.wait_for_timeout(1_000)
        expect(unloading).to_be_visible()
        expect(shortcut).to_have_count(0)
        picker.click()
        expect(page.locator(".unsloth-model-selector-menu")).to_be_visible()
        expect(menu_eject).to_have_count(0)
        page.keyboard.press("Escape")
        assert len(unloads) == 1, len(unloads)

        unloads[0].fulfill(status = 503, json = {"detail": "Test unload failure"})
        expect(page.get_by_text("Test unload failure", exact = True)).to_be_visible()
        expect(unloading).to_have_count(0)
        expect(shortcut).to_be_visible()

        picker.click()
        with page.expect_request("**/api/inference/video/unload"):
            menu_eject.click()
        expect(unloading).to_be_visible()
        assert len(unloads) == 2, len(unloads)
        status = IDLE
        _json(unloads[1], status)
        expect(picker).to_contain_text("Select video model")
        expect(shortcut).to_have_count(0)
        expect(unloading).to_have_count(0)
        browser.close()


if __name__ == "__main__":
    main()
