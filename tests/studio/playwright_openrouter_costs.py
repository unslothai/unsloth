"""Independent pricing control and saved cost details with deterministic fixtures."""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

out = Path(os.environ.get("PW_OUT", "logs/pr-costs-independent"))
out.mkdir(parents = True, exist_ok = True)
base = f"http://127.0.0.1:{os.environ.get('PW_PORT', '5420')}/smoke-openrouter-costs.html"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport = {"width": 960, "height": 1100}, device_scale_factor = 2, reduced_motion = "reduce"
    )
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    for theme in ("dark", "light"):
        page.goto(f"{base}?theme={theme}")
        trigger = page.get_by_role("button", name = "Published model rates")
        trigger.click()
        panel = page.get_by_role("dialog", name = "Published model rates")
        expect(panel.get_by_text("$1", exact = True)).to_be_visible()
        expect(panel.get_by_text("$2", exact = True)).to_be_visible()
        page.wait_for_timeout(250)
        panel.screenshot(path = str(out / f"rates-{theme}-960.png"))
        panel.locator("summary").click()
        expect(panel.get_by_text("min prompt tokens: 200000")).to_be_visible()
        expect(panel.get_by_text("Source: OpenRouter Models API")).to_be_visible()
        panel.screenshot(path = str(out / f"conditions-{theme}-960.png"))
        page.keyboard.press("Escape")
        expect(panel).not_to_be_visible()
        expect(trigger).to_be_focused()
        page.goto(f"{base}?receipt=1&theme={theme}")
        page.locator("summary").click()
        expect(page.get_by_text("Upstream BYOK charge (separate)")).to_be_visible()
        page.locator("details").screenshot(path = str(out / f"receipt-{theme}-960.png"))
    assert not errors, errors
    browser.close()
print("PASS: independent pricing, conditional rates, recorded receipt, BYOK, focus and screenshots")
