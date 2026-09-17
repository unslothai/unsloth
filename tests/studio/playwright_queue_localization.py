# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queue view localization and the editor's send-shortcut chord; no backend needed.

Run: python tests/studio/playwright_queue_localization.py
PW_ENGINE=webkit selects WebKit.
"""

import os
from playwright.sync_api import expect, sync_playwright
from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PAGE = "/smoke-prompt-queue-actions.html"
ENTRY = "/smoke-prompt-queue-actions-main.tsx"

# `wait_for_smoke_page` proves vite ANSWERS, by fetching the raw HTML. The first navigation
# is what makes it WORK: vite transforms the page's whole module graph on demand, and the
# default `wait_until = "load"` waits out every one of those requests, which on a cold
# Windows runner runs past playwright's 30s default. The `wait_for` at the end of `seed`
# already carries 60s for the same reason; this is the navigation ahead of it.
NAV_TIMEOUT_MS = 90_000

# The queue view must not fall back to English while Settings shows the
# translated Queue/Steer wording from the same catalog.
JA = {
    "steer": "方向を変更",
    "edit": "メッセージを編集",
    "copy": "メッセージをコピー",
    "queueing": "キューをオフにする",
    "removed": "メッセージをキューから削除しました。",
}


def seed(
    page,
    base,
    *,
    locale = "en",
    shortcut = "enter",
):
    page.goto(base + PAGE, wait_until = "domcontentloaded", timeout = NAV_TIMEOUT_MS)
    page.evaluate(
        """([locale, shortcut]) => {
            localStorage.setItem("unsloth_locale", locale);
            localStorage.setItem(
                "unsloth_chat_preferences",
                JSON.stringify({ state: { sendShortcut: shortcut }, version: 0 }),
            );
        }""",
        [locale, shortcut],
    )
    page.reload(wait_until = "domcontentloaded", timeout = NAV_TIMEOUT_MS)
    page.get_by_role("button", name = "Reset fixture", exact = True).wait_for(
        state = "visible", timeout = 60_000
    )


def open_editor(page, more_name, edit_name):
    page.get_by_role("button", name = more_name, exact = True).click()
    page.get_by_role("menuitem", name = edit_name, exact = True).click()
    editor = page.locator('textarea[aria-label^="Edit queued prompt"]')
    expect(editor).to_be_visible()
    return editor


def check_localized(page, base):
    seed(page, base, locale = "ja")
    # setLocale loads the catalog lazily; the harness never calls initializeLocale.
    page.evaluate(
        "async () => { const m = await import('/src/i18n/index.ts'); await m.setLocale('ja'); }"
    )
    row = page.locator("[data-queue-item-id]").first
    expect(row).to_contain_text(JA["steer"])
    expect(page.locator('[aria-label^="キューのメッセージ 1"]').first).to_be_visible()
    page.get_by_role("button", name = "キューのメッセージ 1 のその他の操作", exact = True).click()
    for name in (JA["edit"], JA["copy"], JA["queueing"]):
        expect(page.get_by_role("menuitem", name = name, exact = True)).to_be_visible()
    page.keyboard.press("Escape")
    page.get_by_role("button", name = "キューのメッセージ 2 を削除", exact = True).click()
    expect(page.locator('div[role="status"][aria-live="polite"]')).to_have_text(JA["removed"])
    print("PASS: queue view, menu and announcements follow the selected locale", flush = True)


def check_editor_shortcut(page, base):
    # Default "Enter" sends in the composer, so it saves here; Shift+Enter is a newline.
    seed(page, base, shortcut = "enter")
    editor = open_editor(page, "More options for queued prompt 1", "Edit message")
    editor.fill("shift-stays-open")
    editor.press("Shift+Enter")
    expect(editor).to_be_visible()
    expect(page.locator("[data-queue-item-id]").first).to_have_attribute(
        "aria-label", "Queued prompt 1 of 3: First prompt"
    )
    editor.fill("saved-with-enter")
    editor.press("Enter")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").first).to_contain_text("saved-with-enter")
    expect(page.get_by_label("Composer submissions", exact = True)).to_have_text("0")

    # "mod-enter" moves the chord: plain Enter becomes a newline.
    seed(page, base, shortcut = "mod-enter")
    editor = open_editor(page, "More options for queued prompt 1", "Edit message")
    editor.fill("plain-enter-stays-open")
    editor.press("Enter")
    expect(editor).to_be_visible()
    # The row renders the open editor, so read the committed prompt off the label.
    expect(page.locator("[data-queue-item-id]").first).to_have_attribute(
        "aria-label", "Queued prompt 1 of 3: First prompt"
    )
    editor.fill("saved-with-mod-enter")
    editor.press("ControlOrMeta+Enter")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").first).to_contain_text("saved-with-mod-enter")
    expect(page.get_by_label("Composer submissions", exact = True)).to_have_text("0")
    print("PASS: the queued-prompt editor follows the send-shortcut preference", flush = True)


def check_escape_during_ime(page, base):
    # An IME consumes Escape to close its candidate window and reports the key
    # as composing, so the editor must keep the draft rather than cancel.
    seed(page, base)
    editor = open_editor(page, "More options for queued prompt 1", "Edit message")
    editor.fill("draft-survives-ime-escape")
    for init in ("{isComposing: true}", "{keyCode: 229}"):
        page.evaluate(
            f"""() => {{
                const ta = document.querySelector('textarea[aria-label^="Edit queued prompt"]');
                ta.dispatchEvent(new KeyboardEvent('keydown', Object.assign(
                    {{key: 'Escape', bubbles: true, cancelable: true}}, {init})));
            }}"""
        )
        expect(editor).to_be_visible()
        expect(editor).to_have_value("draft-survives-ime-escape")
    # Escape with no composition still cancels.
    editor.press("Escape")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").first).to_have_attribute(
        "aria-label", "Queued prompt 1 of 3: First prompt"
    )
    print("PASS: Escape keeps the draft while an IME is composing", flush = True)


def check_candidate_confirming_enter(page, base):
    # A candidate-confirming Enter can arrive with isComposing false and no key
    # code 229, so per-event flags alone would save the pre-edit text.
    seed(page, base, shortcut = "enter")
    editor = open_editor(page, "More options for queued prompt 1", "Edit message")
    editor.fill("composition-in-progress")
    page.evaluate(
        """() => {
            const ta = document.querySelector('textarea[aria-label^="Edit queued prompt"]');
            ta.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true}));
            ta.dispatchEvent(new KeyboardEvent('keydown',
                {key: 'Enter', bubbles: true, cancelable: true}));
        }"""
    )
    expect(editor).to_be_visible()
    expect(page.locator("[data-queue-item-id]").first).to_have_attribute(
        "aria-label", "Queued prompt 1 of 3: First prompt"
    )
    # After the composition ends the same Enter saves.
    page.evaluate(
        """() => {
            const ta = document.querySelector('textarea[aria-label^="Edit queued prompt"]');
            ta.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true}));
        }"""
    )
    editor.press("Enter")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").first).to_contain_text("composition-in-progress")
    print("PASS: a candidate-confirming Enter does not save the edit", flush = True)


def start_composition(page):
    page.evaluate(
        """() => {
            const ta = document.querySelector('textarea[aria-label^="Edit queued prompt"]');
            ta.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true}));
        }"""
    )


def check_stuck_composition_recovers(page, base):
    # Some IMEs never send compositionend. The gate must not wedge Enter.
    seed(page, base, shortcut = "enter")
    editor = open_editor(page, "More options for queued prompt 1", "Edit message")
    editor.fill("recovers-after-timeout")
    start_composition(page)
    editor.press("Enter")
    expect(editor).to_be_visible()
    # The watchdog drops the flag, so the next Enter saves.
    page.wait_for_timeout(2800)
    editor.press("Enter")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").first).to_contain_text("recovers-after-timeout")

    # Blur is the other reset point.
    editor = open_editor(page, "More options for queued prompt 2", "Edit message")
    editor.fill("recovers-after-blur")
    start_composition(page)
    editor.blur()
    editor.focus()
    editor.press("Enter")
    expect(editor).to_have_count(0)
    expect(page.locator("[data-queue-item-id]").nth(1)).to_contain_text("recovers-after-blur")
    print("PASS: a stuck composition recovers on timeout and on blur", flush = True)


def main():
    server = None
    try:
        base = os.environ.get("PW_BASE_URL", "").rstrip("/")
        if not base:
            port = int(os.environ.get("PW_PORT", "5424"))
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
                check_localized(page, base)
                check_editor_shortcut(page, base)
                check_escape_during_ime(page, base)
                check_candidate_confirming_enter(page, base)
                check_stuck_composition_recovers(page, base)
                assert not errors, errors
            finally:
                browser.close()
    finally:
        if server:
            stop_process(server)


if __name__ == "__main__":
    main()
