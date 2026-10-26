# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DeferredTextarea (the Images prompt box) under a slow owner; PW_ENGINE picks the browser engine (#15221)."""

import json
import os

from playwright.sync_api import expect, sync_playwright
from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PAGE = "/smoke-deferred-prompt.html"
ENTRY = "/smoke-deferred-prompt-main.tsx"
TYPED = " with a small red boat near the shore"


def check(page):
    box = page.get_by_role("textbox", name = "Instruction", exact = True)
    owner = page.get_by_label("Owner prompt")
    generated = page.get_by_label("Generated")
    expect(box).to_have_value("seed prompt")

    # The slow owner must never render inside a key's input event.
    box.click()
    box.press("End")
    box.press_sequentially(TYPED)
    expect(box).to_have_value("seed prompt" + TYPED)
    expect(owner).to_have_text("seed prompt" + TYPED)
    in_key = page.evaluate("window.ownerRendersInKey")
    assert in_key == 0, f"the owner rendered inside {in_key} of {len(TYPED)} keystrokes"

    # A Generate pressed right after the last key reads the full text, not a pending transition.
    box.press_sequentially(" now")
    page.get_by_role("button", name = "Generate", exact = True).click()
    expect(generated).to_have_text(
        json.dumps(["seed prompt" + TYPED + " now"], separators = (",", ":"))
    )

    page.get_by_role("button", name = "Recall", exact = True).click()
    expect(box).to_have_value("from recall")
    page.get_by_role("button", name = "Append", exact = True).click()
    expect(box).to_have_value("from recall, transparent background")

    box.click()
    box.press("End")
    box.press_sequentially("!")
    expect(owner).to_have_text("from recall, transparent background!")


def main():
    server = None
    try:
        base = os.environ.get("BASE_URL")
        if not base:
            port = int(os.environ.get("PW_PORT", "5423"))
            server = start_vite(port)
            base = f"http://127.0.0.1:{port}"
        wait_for_smoke_page(base + PAGE, ENTRY, proc = server)
        with sync_playwright() as pw:
            options = {"headless": True}
            if os.environ.get("PW_EXECUTABLE"):
                options["executable_path"] = os.environ["PW_EXECUTABLE"]
            if os.environ.get("PW_CHANNEL"):
                options["channel"] = os.environ["PW_CHANNEL"]
            browser = getattr(pw, os.environ.get("PW_ENGINE", "chromium")).launch(**options)
            print(
                f"Browser: {os.environ.get('PW_CHANNEL', os.environ.get('PW_ENGINE', 'chromium'))} {browser.version}",
                flush = True,
            )
            try:
                page = browser.new_page(viewport = {"width": 900, "height": 700})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + PAGE)
                page.get_by_role("textbox", name = "Instruction", exact = True).wait_for(
                    state = "visible", timeout = 60_000
                )
                check(page)
                assert not errors, errors
                print("deferred prompt: OK", flush = True)
            finally:
                browser.close()
    finally:
        if server:
            stop_process(server)


if __name__ == "__main__":
    main()
