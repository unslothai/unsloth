# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real shared component coverage for all five decorative locations, including fallback.

Start the frontend Vite server, then run with PW_BASE_URL and optionally
PW_CHROMIUM_EXECUTABLE and PW_ART_DIR. No model download or training is required.
"""

import os
import re
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

out = Path(os.environ.get("PW_ART_DIR", "/tmp/unsloth-mascot-components"))
out.mkdir(parents = True, exist_ok = True)
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path = os.environ.get("PW_CHROMIUM_EXECUTABLE"),
        headless = True,
        args = ["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = browser.new_page(viewport = {"width": 1440, "height": 1000})
    # Exercise the bundled fallback on the canvas decoration as well as successful assets.
    page.route("**/*sloth%20w%20pc%20transparent.png*", lambda route: route.abort())
    page.goto(os.environ.get("PW_BASE_URL", "http://127.0.0.1:5173") + "/smoke-mascots.html")
    toggle = page.get_by_role("button", name = "Decorative mascots", exact = True)
    for enabled in (True, False, True):
        if toggle.get_attribute("aria-pressed") != str(enabled).lower():
            toggle.click()
        for location in ("greeting", "authentication", "notFound", "canvas", "training"):
            image = page.locator(f'[data-decoration="{location}"] img')
            if enabled:
                expect(image).to_be_visible()
            else:
                expect(image).to_have_count(0)
        for identity in ("user", "model-owner"):
            expect(page.locator(f'[data-identity="{identity}"] img')).to_be_visible()
        if enabled:
            expect(page.locator('[data-decoration="canvas"] img')).to_have_attribute(
                "src", re.compile(r"^data:image/")
            )
        page.screenshot(
            path = str(
                out / ("all-components-enabled.png" if enabled else "all-components-disabled.png")
            )
        )
        page.reload()
        expect(toggle).to_have_attribute("aria-pressed", str(enabled).lower())
    print(
        "PASS: all five decorative locations, failed-asset fallback, identity exclusions, and reload persistence"
    )
    browser.close()
