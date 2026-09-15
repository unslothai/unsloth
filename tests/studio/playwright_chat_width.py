# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check width presets against a running Studio with a saved conversation.

BASE_URL, STUDIO_PW and CHAT_THREAD_ID identify the disposable test instance.
Run with: python tests/studio/playwright_chat_width.py
"""

import os

from playwright.sync_api import sync_playwright


def check_widths(page):
    """Full must preserve the space available in Wide, including narrow panes."""
    thread_url = page.url
    measurements = {}
    for preset in ("Standard", "Wide", "Full width"):
        page.set_viewport_size({"width": 1440, "height": 900})
        page.keyboard.press("Control+,")
        page.get_by_role("dialog").get_by_role("button", name = "Appearance", exact = True).click()
        page.get_by_role("combobox", name = "Chat width").click()
        page.get_by_role("option", name = preset, exact = True).click()
        page.keyboard.press("Escape")
        measurements[preset] = {}
        for width in (390, 768, 900, 1280, 1536, 1920, 2560):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(300)
            measurements[preset][width] = page.evaluate(
                """() => {
                    const width = selector => document.querySelector(selector).getBoundingClientRect().width;
                    return {
                        message: width('.aui-assistant-message-root'),
                        composer: width('.unsloth-composer-shell'),
                        overflow: document.documentElement.scrollWidth > innerWidth,
                    };
                }"""
            )
        if preset == "Wide":
            page.set_viewport_size({"width": 1920, "height": 900})
            page.goto(thread_url.split("?")[0] + "?new=width-transition")
            shell = page.locator(".unsloth-composer-shell")
            shell.wait_for()
            page.wait_for_timeout(300)
            welcome_width = shell.bounding_box()["width"]
            assert abs(welcome_width - measurements[preset][1920]["composer"]) <= 1
            page.goto(thread_url)
            page.locator(".aui-assistant-message-root").wait_for()
    for width, full in measurements["Full width"].items():
        standard = measurements["Standard"][width]
        wide = measurements["Wide"][width]
        for surface in ("message", "composer"):
            assert full[surface] >= wide[surface] - 1, (width, surface, full, wide)
            assert wide[surface] >= standard[surface] - 1, (width, surface, wide, standard)
        assert not any(measurements[preset][width]["overflow"] for preset in measurements)
    return measurements


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport = {"width": 1440, "height": 900}, reduced_motion = "reduce")
        page.goto(os.environ["BASE_URL"])
        page.locator("input[type=password]").fill(os.environ["STUDIO_PW"])
        page.keyboard.press("Enter")
        page.locator(".aui-thread-root").wait_for()
        page.goto(f"{os.environ['BASE_URL']}/chat?thread={os.environ['CHAT_THREAD_ID']}")
        page.locator(".aui-assistant-message-root").wait_for()
        print(check_widths(page))
        browser.close()
