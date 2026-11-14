# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A slow Images eject shows progress, sends one request, and allows retry on failure."""

import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from playwright_image_model_footprint import BASE_URL, REPO_ID, _api_payload, _json


def main():
    status_path = "/api/inference/images/status"
    idle = _api_payload(status_path, {}, full_footprint = False)
    status = {
        **idle,
        "loaded": True,
        "repo_id": REPO_ID,
        "base_repo": "black-forest-labs/FLUX.2-klein-4B",
        "family": "flux.2-klein",
        "model_kind": "gguf",
    }
    generating = True
    unloads = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True, executable_path = os.getenv("PW_CHROME"))
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test')")

        def route_request(route):
            url = urlparse(route.request.url)
            if url.netloc != urlparse(BASE_URL).netloc:
                _json(route, [])
            elif url.path == status_path:
                _json(route, status)
            elif url.path == "/api/inference/images/generate-progress":
                _json(
                    route,
                    {
                        "active": generating,
                        "step": 1,
                        "fraction": 0.25,
                        "total_steps": 4,
                        "eta_seconds": 1,
                    },
                )
            elif url.path == "/api/inference/images/unload":
                assert route.request.method == "POST"
                unloads.append(route)
            elif url.path.startswith("/api/"):
                _json(route, _api_payload(url.path, parse_qs(url.query), full_footprint = False))
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.goto(f"{BASE_URL}/images", wait_until = "domcontentloaded")
        picker = page.locator('[data-tour="images-model"]')
        shortcut = picker.locator("[data-eject-hit]")
        menu_eject = page.get_by_role("button", name = "Eject model", exact = True)
        unloading = page.get_by_text("Unloading model", exact = True)
        expect(shortcut).to_be_visible(timeout = 30_000)
        expect(page.get_by_role("button").filter(has_text = "Stop")).to_be_visible()

        # Two clicks in one event turn also exercise the guard before React updates the DOM.
        with page.expect_request("**/api/inference/images/unload"):
            shortcut.evaluate("(el) => { el.click(); el.click(); }")
        expect(unloading).to_be_visible()
        # The generation poll refreshes the gallery when a run ends during teardown.
        with page.expect_request(
            lambda req: urlparse(req.url).path == "/api/inference/images/gallery"
        ):
            generating = False
        expect(unloading).to_be_visible()
        expect(shortcut).to_have_count(0)
        picker.click()
        expect(page.locator(".unsloth-model-selector-menu")).to_be_visible()
        expect(menu_eject).to_have_count(0)
        page.keyboard.press("Escape")
        assert len(unloads) == 1

        unloads[0].fulfill(status = 503, json = {"detail": "Test unload failure"})
        expect(page.get_by_text("Test unload failure", exact = True)).to_be_visible()
        expect(unloading).to_have_count(0)
        expect(shortcut).to_be_visible()

        picker.click()
        with page.expect_request("**/api/inference/images/unload"):
            menu_eject.click()
        expect(unloading).to_be_visible()
        assert len(unloads) == 2
        status = idle
        _json(unloads[1], status)
        expect(picker).to_contain_text("Select image model")
        expect(shortcut).to_have_count(0)
        expect(unloading).to_have_count(0)
        browser.close()


if __name__ == "__main__":
    main()
