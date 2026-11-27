# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the chat layout setting in a real browser without a model."""

import os
import tempfile
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright
from playwright_image_model_footprint import _api_payload, _json

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:4173")


def main() -> None:
    """Toggle the preference, reload, and verify it remains scoped to chat."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless = True,
            executable_path = os.environ.get("PW_CHROMIUM_EXECUTABLE_PATH") or None,
        )
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900},
            reduced_motion = "reduce",
        )
        context.add_init_script("localStorage.setItem('unsloth_auth_token', 'rendered-ui-test');")

        def route_request(route) -> None:
            """Supply the existing deterministic API fixture for the UI smoke test."""
            parsed = urlparse(route.request.url)
            if parsed.netloc == "huggingface.co":
                _json(route, [])
            elif parsed.path == "/api/models/list":
                _json(route, {"models": []})
            elif parsed.path == "/api/settings/chat-memory":
                _json(route, {"text": "", "enabled": False})
            elif parsed.path.startswith("/api/"):
                _json(
                    route, _api_payload(parsed.path, parse_qs(parsed.query), full_footprint = False)
                )
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.goto(f"{BASE_URL}/chat", wait_until = "domcontentloaded", timeout = 90_000)
        root = page.locator("[data-chat-page-root]")
        expect(root).to_have_attribute("dir", "ltr", timeout = 90_000)
        tools = page.get_by_role("button", name = "Tools and attachments")
        send = page.get_by_role("button", name = "Send message", exact = True)
        assert tools.bounding_box()["x"] < send.bounding_box()["x"]
        page.get_by_role("button", name = "Settings", exact = True).last.click()
        dialog = page.locator("[data-settings-dialog]")
        dialog.get_by_role("button", name = "Chat", exact = True).click()
        switch = dialog.get_by_role("switch", name = "Right-to-left chat layout")
        expect(switch).not_to_be_checked(timeout = 90_000)
        switch.check()
        expect(root).to_have_attribute("dir", "rtl")
        assert page.evaluate("document.documentElement.dir") != "rtl"
        greeting = page.locator("h1.unsloth-welcome-title")
        expect(greeting).to_have_attribute("dir", "auto")
        assert greeting.evaluate("el => getComputedStyle(el).direction") == "ltr"
        dialog.get_by_role("button", name = "Close settings").click()
        assert tools.bounding_box()["x"] > send.bounding_box()["x"]
        page.screenshot(path = os.path.join(tempfile.gettempdir(), "unsloth-rtl-desktop.png"))
        page.reload(wait_until = "domcontentloaded", timeout = 90_000)
        expect(root).to_have_attribute("dir", "rtl", timeout = 90_000)
        expect(page.get_by_role("button", name = "Open run settings")).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        expect(root).to_have_attribute("dir", "rtl")
        page.screenshot(path = os.path.join(tempfile.gettempdir(), "unsloth-rtl-mobile.png"))
        composer = page.get_by_role("textbox", name = "Message input")
        composer.fill("مرحبا world")
        expect(composer).to_have_value("مرحبا world")
        assert composer.evaluate("el => getComputedStyle(el).direction") == "rtl"
        composer.fill("Hello مرحبا")
        assert composer.evaluate("el => getComputedStyle(el).direction") == "ltr"
        page.get_by_role("button", name = "Tools and attachments").click()
        expect(page.get_by_role("menu")).to_be_visible()
        context.close()
        browser.close()
    print("Chat layout direction browser smoke passed.")


if __name__ == "__main__":
    main()
