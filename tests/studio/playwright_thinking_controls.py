# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the production thinking control against its deterministic Vite smoke page.

Start Vite on PW_PORT (default 5418) first. Screenshots and the report go to PW_OUT.
"""

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main():
    out = Path(os.environ.get("PW_OUT", "logs/pr-thinking"))
    out.mkdir(parents = True, exist_ok = True)
    base = f"http://127.0.0.1:{os.environ.get('PW_PORT', '5418')}/smoke-thinking-controls.html"
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport = {"width": 960, "height": 720})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for theme in ("dark", "light"):
            for width in (960, 375):
                page.set_viewport_size({"width": width, "height": 720})
                for state in (
                    "adjustable",
                    "fixed",
                    "toggle",
                    "mandatory",
                    "unknown",
                    "unsupported",
                ):
                    page.goto(f"{base}?theme={theme}&state={state}")
                    if state == "unsupported":
                        expect(
                            page.get_by_role("button", name = "Thinking", exact = False)
                        ).to_have_count(0)
                        continue
                    trigger = page.get_by_role("button", name = "Thinking", exact = False)
                    trigger.click()
                    panel = page.get_by_role("dialog", name = "Thinking settings")
                    expect(panel).to_be_visible()
                    expect(panel.get_by_role("slider")).to_have_count(
                        1 if state == "adjustable" else 0
                    )
                    if state == "fixed":
                        expect(panel.get_by_text("This model supports High only.")).to_be_visible()
                    if state == "mandatory":
                        expect(panel.get_by_role("switch")).to_have_count(0)
                    if state == "adjustable":
                        slider = panel.get_by_role("slider")
                        slider.focus()
                        page.keyboard.press("End")
                        expect(page.get_by_label("Selected effort")).to_have_text("max")
                        page.keyboard.press("Home")
                        expect(page.get_by_label("Selected effort")).to_have_text("low")
                        page.keyboard.press("ArrowRight")
                        expect(slider).to_have_attribute("aria-valuetext", "Medium")
                        panel.get_by_role("combobox").click()
                        page.get_by_role("option", name = "Extra High", exact = True).click()
                        expect(page.get_by_label("Selected effort")).to_have_text("xhigh")
                        panel.get_by_role("switch", name = "Enable thinking").click()
                        expect(trigger).to_have_attribute("aria-label", "Thinking · Off")
                        panel.get_by_role("button", name = "Reset to model default").click()
                        expect(page.get_by_label("Selected effort")).to_have_text("medium")
                    bounds = panel.bounding_box()
                    assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
                    page.wait_for_timeout(200)  # Capture the settled 100ms popover animation.
                    page.screenshot(path = str(out / f"{state}-{theme}-{width}.png"))
                    page.keyboard.press("Escape")
                    expect(panel).not_to_be_visible()
                    expect(trigger).to_be_focused()
                    checks.append(f"{state}-{theme}-{width}")
        page.goto(f"{base}?preserve=1&state=unsupported")
        page.get_by_role("button", name = "Preserve thinking").click()
        page.get_by_role("switch", name = "Preserve thinking").click()
        expect(page.get_by_role("switch", name = "Preserve thinking")).to_be_checked()
        assert not errors, errors
        browser.close()
    (out / "report.json").write_text(
        json.dumps({"passed": checks, "errors": errors}, indent = 2), encoding = "utf-8"
    )
    print(
        f"PASS: {len(checks)} thinking scenarios, keyboard controls, viewport bounds, focus and preservation"
    )


if __name__ == "__main__":
    main()
