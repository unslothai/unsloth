# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reopen a dismissed research plan through the real message and activity panel.

Run: python tests/studio/playwright_research_review.py
Starts and stops Vite unless SMOKE_BASE_URL is supplied. SMOKE_PORT overrides
the local port; SMOKE_BROWSER selects chromium (CI/default) or webkit.
"""

import os
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import (
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5184"))
EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = EXTERNAL or f"http://127.0.0.1:{PORT}"


def check_review(page):
    errors = []
    requests = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def reject_api(route):
        requests.append(route.request.url)
        route.fulfill(status = 500, body = "Unexpected API request")

    page.route(lambda url: urlparse(url).path.startswith("/api/"), reject_api)
    page.goto(f"{BASE}/smoke-research-review.html")
    dialog = page.get_by_role("dialog", name = "Review the research plan")
    expect(dialog).to_be_visible(timeout = 30_000)
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    page.evaluate("() => { window.__review.setDraft(); window.__review.setError(); }")
    before = page.evaluate("() => window.__review.state()")

    # Scope to the message: the panel's own button already reopened on the base.
    button = page.get_by_test_id("research-message").get_by_role(
        "button", name = "Review plan", exact = True
    )
    button.click()
    expect(dialog).to_be_visible()
    after = page.evaluate("() => window.__review.state()")
    assert after["sessions"] == before["sessions"], "Review changed connection/run state"
    assert after["planReviewByRunId"]["review-run"] == {
        **before["planReviewByRunId"]["review-run"],
        "open": True,
    }, "Review discarded the draft"

    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    page.evaluate("() => window.__review.closePanel()")
    button.click()
    expect(dialog).to_be_visible()
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()

    page.evaluate("() => { window.__review.complete(); window.__review.closePanel(); }")
    completed = page.evaluate("() => window.__review.state()")
    page.get_by_test_id("research-message").get_by_role("button", name = "View activity").click()
    viewed = page.evaluate("() => window.__review.state()")
    assert viewed["openRunId"] == "review-run"
    assert viewed["sessions"] == completed["sessions"]
    assert viewed["planReviewByRunId"] == completed["planReviewByRunId"]
    expect(dialog).to_have_count(0)
    assert requests == [], f"Review made API requests: {requests}"
    assert errors == [], f"Page errors: {errors}"


def main():
    proc = None
    try:
        if not EXTERNAL:
            proc = start_vite(PORT)
        wait_for_smoke_page(
            f"{BASE}/smoke-research-review.html", "smoke-research-review-main.tsx", proc = proc
        )
        with sync_playwright() as pw:
            name = os.environ.get("SMOKE_BROWSER", "chromium")
            if name not in {"chromium", "webkit"}:
                raise ValueError(f"Unsupported SMOKE_BROWSER: {name}")
            browser_type = getattr(pw, name)
            executable = os.environ.get("PW_EXECUTABLE")
            browser = browser_type.launch(
                args = chromium_launch_args() if name == "chromium" else [],
                **({"executable_path": executable} if executable else {}),
            )
            try:
                check_review(browser.new_page())
            finally:
                browser.close()
        print("PASS: dismissed plan reopens; draft, connection and activity behavior preserved")
    finally:
        if proc is not None:
            stop_process(proc)


if __name__ == "__main__":
    main()
