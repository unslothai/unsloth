# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image VRAM budget controls in the real frontend with simulated APIs; no GPU required.

Run with BASE_URL pointing to Vite. Verifies persistence, unsupported hosts/backends,
failed saves, and flushing a pending edit when navigating away.
"""

import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright
from playwright_image_model_footprint import _api_payload

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:4173")


def run_case(browser, case):
    context = browser.new_context(viewport = {"width": 1280, "height": 900}, reduced_motion = "reduce")
    context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');")
    state = {"fraction": 0.97, "stored": False, "writes": []}
    errors = []

    def route(request):
        url = urlparse(request.request.url)
        if not url.path.startswith("/api/"):
            if url.netloc == urlparse(BASE).netloc:
                request.continue_()
            else:
                request.fulfill(json = [])
            return
        if url.path == "/api/settings/vram-budget":
            if case == "old-backend":
                request.fulfill(status = 404, json = {"detail": "Not found"})
                return
            if request.request.method == "PUT":
                value = request.request.post_data_json["fraction"]
                state["writes"].append(value)
                if case == "failed-save":
                    request.fulfill(status = 500, json = {"detail": "Budget write refused"})
                    return
                state.update(fraction = 0.97 if value is None else value, stored = value is not None)
            data = dict(
                fraction = state["fraction"],
                is_stored = state["stored"],
                default_fraction = 0.97,
                min_fraction = 0.1,
                max_fraction = 1,
                reload_required = False,
            )
        else:
            data = _api_payload(url.path, parse_qs(url.query), full_footprint = True)
            if url.path == "/api/system":
                for key in ("gpu", "inference_gpu"):
                    if case == "cpu":
                        data[key].update(available = False, devices = [])
                    else:
                        for device in data[key]["devices"]:
                            device.update(memory_total_gb = 8, vram_free_gb = 7)
                if case == "cpu":
                    data["device_backend"] = "cpu"
            if url.path == "/api/health" and case == "mac":
                data["device_type"] = "mac"
        request.fulfill(json = data)

    context.route("**/*", route)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{BASE}/images", wait_until = "domcontentloaded")
    page.get_by_role("button", name = "Advanced", exact = True).click()
    memory = page.get_by_role("combobox", name = "Memory", exact = True)
    memory.scroll_into_view_if_needed()
    expect(memory).to_be_visible()
    value = page.locator('input[aria-label="VRAM Budget"]')
    if case in {"cpu", "mac", "old-backend"}:
        # Let the GPU inventory and settings reads settle before checking absence.
        page.wait_for_timeout(800)
        expect(value).to_have_count(0)
    else:
        expect(value).to_have_value("97%")
        slider = page.get_by_role("slider", name = "VRAM Budget", exact = True)
        slider.focus()
        slider.press("ArrowLeft")
        if case == "navigate-away":
            # Before the 400 ms debounce fires: teardown must flush the changed value.
            page.get_by_role("button", name = "New chat", exact = True).first.click()
        if case == "failed-save":
            expect(page.get_by_text("Budget write refused", exact = True).first).to_be_visible()
            assert state["fraction"] == 0.97
        else:
            page.wait_for_timeout(800)
            assert state["writes"] == [0.969], state
            page.goto(f"{BASE}/images", wait_until = "domcontentloaded")
            expect(value).to_have_value("96.9%")
            page.get_by_role("button", name = "Reset to the server default", exact = True).click()
            expect(value).to_have_value("97%")
            assert state["writes"] == [0.969, None], state
    assert not errors, errors
    context.close()
    print(f"PASS image VRAM budget: {case}", flush = True)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True)
        for case in ("save-reset", "navigate-away", "failed-save", "cpu", "mac", "old-backend"):
            run_case(browser, case)
        browser.close()


if __name__ == "__main__":
    main()
