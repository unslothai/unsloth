# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Production settings/preview and assistant-ui input smoke test; no inference required.
Run: python tests/studio/playwright_composer_settings.py
PW_ENGINE=webkit selects WebKit. PW_OUTPUT optionally saves screenshots.
"""

import json
import os
import re
from pathlib import Path
from playwright.sync_api import expect, sync_playwright
from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PAGE = "/smoke-composer-settings.html"
ENTRY = "/smoke-composer-settings-main.tsx"


def check_settings(page):
    plain = page.get_by_role("switch", name = "Plain text composer", exact = True)
    context = page.get_by_role("switch", name = "Show context window usage", exact = True)
    select = page.get_by_role("combobox", name = "Send shortcut", exact = True)
    editor = page.get_by_role("textbox", name = "Message", exact = True)
    preview = page.get_by_role("region", name = "Formatted preview", exact = True)
    submitted = page.get_by_label("Submitted messages")
    expect(plain).to_be_checked()
    expect(context).to_be_checked()
    expect(page.get_by_role("button", name = "Queue", exact = True)).to_have_attribute(
        "aria-pressed", "true"
    )
    editor.fill("**Keep this literal**")
    expect(preview).to_have_count(0)
    plain.click()
    expect(preview.locator('[data-streamdown="strong"]')).to_have_text("Keep this literal")
    expect(editor).to_have_value("**Keep this literal**")
    requested = []
    page.on(
        "request",
        lambda req: requested.append(req.url) if "draft-only.invalid" in req.url else None,
    )
    editor.fill(
        "![private](https://draft-only.invalid/image.png)\n\n[private link](https://draft-only.invalid/)\n\n<script>alert(1)</script>"
    )
    expect(preview).to_contain_text("private link")
    expect(preview.locator("img, a, script")).to_have_count(0)
    assert not requested, requested
    context.click()
    page.get_by_role("button", name = "Steer", exact = True).click()
    select.click()
    option = page.get_by_role("option", name = re.compile(r"^(⌘|Ctrl\+)Enter$"))
    modified_label = option.inner_text()
    option.click()
    editor.fill("First line")
    editor.press("End")
    editor.press("Enter")
    editor.type("Second line")
    expect(editor).to_have_value("First line\nSecond line")
    expect(submitted).to_have_text("[]")
    editor.press("Meta+Enter")
    expect(editor).to_have_value("")
    assert json.loads(submitted.text_content()) == [
        {"text": "First line\nSecond line", "behavior": "steer"}
    ]
    editor.fill("Queue this once")
    editor.press("Meta+Shift+Enter")
    assert json.loads(submitted.text_content())[-1]["behavior"] == "queue"
    page.reload()
    expect(plain).not_to_be_checked()
    expect(context).not_to_be_checked()
    expect(select).to_contain_text(modified_label)
    expect(page.get_by_role("button", name = "Steer", exact = True)).to_have_attribute(
        "aria-pressed", "true"
    )
    select.click()
    page.get_by_role("option", name = "Enter", exact = True).click()
    editor.fill("Line one")
    editor.press("End")
    editor.press("Shift+Enter")
    editor.type("Line two")
    expect(editor).to_have_value("Line one\nLine two")
    editor.press("Enter")
    assert json.loads(submitted.text_content())[-1]["behavior"] == "steer"
    editor.fill("Queue override")
    editor.press("Control+Enter")
    assert json.loads(submitted.text_content())[-1]["behavior"] == "queue"
    editor.fill("Compose without sending")
    editor.dispatch_event(
        "keydown", {"key": "Enter", "code": "Enter", "isComposing": True, "keyCode": 229}
    )
    expect(editor).to_have_value("Compose without sending")
    editor.fill(
        "**Review the training results**\n\n- Compare accuracy and speed\n- Summarize the next steps"
    )
    expect(preview.locator('[data-streamdown="strong"]')).to_have_text(
        "Review the training results"
    )
    expect(preview.locator("li")).to_have_count(2)
    editor.blur()
    output = os.environ.get("PW_OUTPUT")
    if output:
        dest = Path(output)
        dest.mkdir(parents = True, exist_ok = True)
        page.screenshot(path = str(dest / "composer-settings-light.png"), full_page = True)
        page.evaluate("document.documentElement.classList.add('dark')")
        page.wait_for_timeout(250)  # Let the existing theme color transitions settle.
        page.screenshot(path = str(dest / "composer-settings-dark.png"), full_page = True)
    page.set_viewport_size({"width": 320, "height": 812})
    expect(plain).to_be_visible()
    assert page.evaluate(
        "document.documentElement.scrollWidth <= innerWidth"
    ), "mobile horizontal overflow"
    if output:
        page.screenshot(path = str(Path(output) / "composer-settings-mobile.png"), full_page = True)
    print(
        "PASS: settings, persistence, preview, raw draft, shortcuts, override, IME and 320px layout",
        flush = True,
    )


def main():
    server = None
    try:
        base = os.environ.get("BASE_URL")
        if not base:
            port = int(os.environ.get("PW_PORT", "5422"))
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
                page = browser.new_page(viewport = {"width": 1100, "height": 850})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base + PAGE)
                page.get_by_role("switch", name = "Plain text composer", exact = True).wait_for(
                    state = "visible", timeout = 60_000
                )
                check_settings(page)
                assert not errors, errors
            finally:
                browser.close()
    finally:
        if server:
            stop_process(server)


if __name__ == "__main__":
    main()
