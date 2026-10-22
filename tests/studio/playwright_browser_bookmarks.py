# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bookmark submissions preserve the current page, its state, and browser history."""

import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import chromium_launch_args, install_view_transition_killer

BASE = os.environ["BASE_URL"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-bookmarks"))
FIRST = "https://bookmark.test/first"
SECOND = "https://bookmark.test/followed"


def check_page(browser, session, action, followed):
    context = browser.new_context(viewport = {"width": 1440, "height": 900}, locale = "en-US")
    install_view_transition_killer(context)
    context.add_init_script(
        f"localStorage.setItem('unsloth_auth_token', {json.dumps(session['access_token'])});"
        f"localStorage.setItem('unsloth_refresh_token', {json.dumps(session['refresh_token'])});"
    )
    page = context.new_page()
    label = f"{action}-{'followed' if followed else 'typed'}"
    try:

        def fetched_page(route):
            url = route.request.post_data_json["url"]
            title = "Typed page" if url == FIRST else "Followed page"
            route.fulfill(
                headers = {
                    "Content-Type": "application/json",
                    "X-Unsloth-Browser-Kind": "html",
                },
                json = {
                    "url": url,
                    "base": url,
                    "refresh": None,
                    "html": f'<title>{title}</title><a href="{SECOND}">Follow link</a>'
                    '<div style="height:600px"></div>'
                    '<label>Page input<input aria-label="Page input"></label>'
                    '<div style="height:3000px"></div>',
                },
            )

        page.route("**/api/browser/fetch", fetched_page)
        page.goto(f"{BASE}/chat", wait_until = "domcontentloaded")
        page.get_by_role("button", name = "New tab", exact = True).click()
        panel = page.get_by_role("region", name = "Browser", exact = True)
        address = panel.get_by_role("textbox", name = "Search or enter a URL")
        address.fill(FIRST)
        address.press("Enter")
        shell = panel.locator('iframe[src*="/api/browser/frame"]')
        content = panel.frame_locator('iframe[src*="/api/browser/frame"]').frame_locator("iframe")
        expect(content.get_by_role("link", name = "Follow link")).to_be_visible()
        if followed:
            content.get_by_role("link", name = "Follow link").click()
        current = SECOND if followed else FIRST
        title = "Followed page" if followed else "Typed page"
        expect(address).to_have_value(current)
        expect(panel.get_by_role("tab")).to_contain_text(title)
        field = content.get_by_role("textbox", name = "Page input")
        field.fill("keep this input")
        frame = next(frame for frame in page.frames if frame.url == "about:srcdoc")
        frame.evaluate("window.scrollTo(0, 500)")
        scroll = frame.evaluate("window.scrollY")
        assert scroll > 0
        shell_url = shell.get_attribute("src")
        loads = []
        page.on(
            "framenavigated", lambda frame: loads.append(frame.url) if frame.parent_frame else None
        )
        panel.get_by_role("button", name = "Bookmark this page", exact = True).click()
        editor = page.get_by_role("dialog").filter(has = page.locator("[data-bookmark-name]"))
        name = editor.get_by_role("textbox", name = "Name", exact = True)
        expect(name).to_have_value(title)
        if action == "Enter":
            name.press("Enter")
        else:
            editor.get_by_role("button", name = "Save", exact = True).click()
        expect(editor).to_be_hidden()
        expect(address).to_have_value(current)
        expect(field).to_have_value("keep this input")
        assert frame.evaluate("window.scrollY") == scroll
        assert shell.get_attribute("src") == shell_url
        assert not loads, f"bookmarking loaded another frame: {loads}"
        bookmarks = page.evaluate("""() => Object.keys(localStorage)
            .filter(key => key.startsWith('unsloth_browser_bookmarks'))
            .flatMap(key => JSON.parse(localStorage.getItem(key)).state.bookmarks)""")
        assert len(bookmarks) == 1
        assert bookmarks[0]["url"] == current
        assert bookmarks[0]["title"] == title
        if followed and action == "Save":

            def remove_bookmark():
                editor.get_by_role("button", name = "Remove bookmark", exact = True).click()
                expect(editor).to_be_hidden()
                toast = page.locator("[data-sonner-toast]").filter(has_text = "Bookmark removed")
                toast.get_by_role("button", name = "Close toast", exact = True).click()
                expect(toast).to_have_count(0)

            panel.get_by_role("button", name = "Edit bookmark", exact = True).click()
            name.fill("Renamed page")
            for folder in ("Other bookmarks", "Bookmarks toolbar"):
                editor.get_by_role("combobox").click()
                page.get_by_role("option", name = folder, exact = True).click()
                expect(address).to_have_value(current)
            checkbox = editor.get_by_role("checkbox", name = "Show editor when saving")
            checkbox.uncheck()
            checkbox.check()
            name.press("Escape")
            expect(editor).to_be_hidden()
            panel.get_by_role("button", name = "Edit bookmark", exact = True).click()
            expect(name).to_have_value("Renamed page")
            remove_bookmark()
            panel.get_by_role("button", name = "Bookmark this page", exact = True).click()
            editor.get_by_role("button", name = "Cancel", exact = True).click()
            expect(editor).to_be_hidden()
            for shortcut in ("Control+d", "Meta+d"):
                panel.get_by_role("button", name = "Bookmark this page", exact = True).press(shortcut)
                expect(name).to_have_value("Followed page")
                editor.get_by_role("button", name = "Save", exact = True).click()
                panel.get_by_role("button", name = "Edit bookmark", exact = True).click()
                remove_bookmark()
            panel.get_by_role("button", name = "Site information", exact = True).click()
            page.get_by_role("button", name = "Connection secure", exact = True).click()
            page.get_by_role("button", name = "Back", exact = True).last.click()
            page.get_by_role("button", name = "Clear browsing data", exact = True).click()
            page.get_by_role("button", name = "Cancel", exact = True).click()
            expect(address).to_have_value(current)
            expect(field).to_have_value("keep this input")
            assert frame.evaluate("window.scrollY") == scroll
            assert not loads, f"editor or site popover controls navigated: {loads}"
        page.screenshot(path = str(ART / f"{label}.png"))
        if followed:
            panel.get_by_role("button", name = "Back", exact = True).click()
            expect(address).to_have_value(FIRST)
            panel.get_by_role("button", name = "Forward", exact = True).click()
            expect(address).to_have_value(SECOND)
        print(f"PASS {label}", flush = True)
    except Exception:
        page.screenshot(path = str(ART / f"{label}-failed.png"))
        raise
    finally:
        context.close()


def main():
    ART.mkdir(parents = True, exist_ok = True)
    engine = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium")
    with sync_playwright() as playwright:
        request = playwright.request.new_context(base_url = BASE)
        response = request.post(
            "/api/auth/login",
            data = {"username": "unsloth", "password": os.environ["STUDIO_NEW_PW"]},
        )
        assert response.ok, "test account login succeeds"
        session = response.json()
        assert not session.get("must_change_password"), "test account must have completed setup"
        request.dispose()
        options = {"args": chromium_launch_args()} if engine == "chromium" else {}
        channel = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL")
        if channel:
            options["channel"] = channel
        browser = getattr(playwright, engine).launch(**options)
        failures = []
        try:
            for action in ("Save", "Enter"):
                for followed in (False, True):
                    try:
                        check_page(browser, session, action, followed)
                    except Exception as error:
                        failures.append(f"{action}, followed={followed}: {error}")
                        print(f"FAIL {failures[-1]}", flush = True)
        finally:
            browser.close()
        assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
