# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise mascot preferences against a running Studio with a disposable account.

BASE_URL=http://127.0.0.1:8890 STUDIO_TEST_PASSWORD=... \
PW_CHROMIUM_EXECUTABLE=/usr/bin/chromium python tests/studio/playwright_mascot_preferences.py

Uses the real UI and API. Restores the account's original personalization on exit.
PW_ART_DIR controls the screenshot/report directory; no auth state is written there.
"""

import base64
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = os.environ["BASE_URL"].rstrip("/")
OUT = Path(os.environ.get("PW_ART_DIR", "/tmp/unsloth-mascot-verification"))
OUT.mkdir(parents = True, exist_ok = True)
ENDPOINT = BASE + "/api/settings/personalization"
ASSET = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/public/Sloth emojis/large sloth heart.png"
)
AVATAR = "data:image/png;base64," + base64.b64encode(ASSET.read_bytes()).decode()
report = []
limitations = []

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path = os.environ.get("PW_CHROMIUM_EXECUTABLE"),
        headless = True,
        args = ["--no-sandbox", "--disable-dev-shm-usage"],
    )
    context = browser.new_context(viewport = {"width": 1440, "height": 1000})
    page = context.new_page()
    page.goto(BASE + "/login")
    page.get_by_label("Password", exact = True).fill(os.environ["STUDIO_TEST_PASSWORD"])
    page.get_by_role("button", name = "Login", exact = True).click()
    page.wait_for_url("**/chat")
    token = page.evaluate("localStorage.getItem('unsloth_auth_token')")
    headers = {"Authorization": "Bearer " + token}
    original = context.request.get(ENDPOINT, headers = headers).json()

    def get_settings():
        response = context.request.get(ENDPOINT, headers = headers)
        assert response.ok
        return response.json()

    def put_settings(payload):
        response = context.request.put(ENDPOINT, headers = headers, data = payload)
        assert response.ok, response.status
        return response.json()

    def settings_tab(name):
        if not page.get_by_role("dialog").is_visible():
            page.get_by_role("button", name = "Settings", exact = True).click()
        page.get_by_role("button", name = name, exact = True).click()

    def close_settings():
        page.keyboard.press("Escape")
        expect(page.get_by_role("dialog")).not_to_be_visible()

    def change(control):
        with page.expect_response(
            lambda r: r.url == ENDPOINT and r.request.method == "PUT"
        ) as response:
            control.click()
        assert response.value.status == 200

    def capture(name):
        page.screenshot(path = str(OUT / (name + ".png")))
        report.append(name)
        print("PASS", name, flush = True)

    try:
        put_settings(
            {
                "profile": {"showGreetingSloth": True, "avatarDataUrl": AVATAR},
                "appearance": {"customization": {"showMascots": True}, "language": "en"},
            }
        )
        page.reload()
        expect(page.locator(".unsloth-welcome-sloth")).to_be_visible()
        expect(page.locator('img[src^="data:image/png;base64,"]').first).to_be_visible()
        capture("chat-enabled-avatar")
        settings_tab("Appearance")
        toggle = page.get_by_role("switch", name = "Decorative mascots", exact = True)
        expect(toggle).to_be_checked()
        change(toggle)
        expect(toggle).not_to_be_checked()
        assert get_settings()["appearance"]["customization"]["showMascots"] is False
        assert get_settings()["mascotsSaved"] is True
        capture("appearance-disabled")
        close_settings()
        expect(page.locator(".unsloth-welcome-sloth")).to_have_count(0)
        expect(page.locator('img[src^="data:image/png;base64,"]').first).to_be_visible()
        capture("chat-disabled-avatar")
        page.reload()
        expect(page.locator(".aui-thread-welcome-message-inner").first).to_be_visible()
        expect(page.locator(".unsloth-welcome-sloth")).to_have_count(0)
        capture("disabled-after-refresh")
        page.get_by_role("button", name = "Model hub", exact = True).click()
        expect(
            page.locator(".hub-avatar-tile").first.or_(
                page.get_by_text("Can't reach Hugging Face", exact = True)
            )
        ).to_be_visible(timeout = 20000)
        if page.get_by_text("Can't reach Hugging Face", exact = True).is_visible():
            limitations.append(
                "Live model-author avatar check blocked: Hugging Face is unreachable."
            )
            page.screenshot(path = str(OUT / "model-hub-network-unavailable.png"))
        else:
            expect(page.locator(".hub-avatar-tile").first).to_be_visible()
            expect(page.locator(".hub-avatar-tile img").first).to_be_visible()
            capture("model-author-avatars-preserved")
        page.get_by_role("button", name = "New chat", exact = True).first.click()
        settings_tab("Chat")
        local = page.get_by_role("switch", name = "Sloth in greeting", exact = True)
        expect(local).to_be_checked()
        expect(local).to_be_disabled()
        capture("local-choice-preserved")
        search = page.get_by_placeholder("Search settings…")
        search.fill("mascot")
        page.get_by_role("button", name = "Decorative mascots", exact = False).click()
        expect(toggle).to_be_visible()
        capture("search-mascot")
        search.fill("")
        change(toggle)
        close_settings()
        expect(page.locator(".unsloth-welcome-sloth")).to_be_visible()
        page.reload()
        expect(page.locator(".unsloth-welcome-sloth")).to_be_visible()
        capture("enabled-after-refresh")
        settings_tab("Chat")
        change(local)
        settings_tab("Appearance")
        change(toggle)
        change(toggle)
        settings_tab("Chat")
        expect(local).not_to_be_checked()
        expect(local).to_be_enabled()
        close_settings()
        expect(page.locator(".unsloth-welcome-sloth")).to_have_count(0)
        capture("local-off-survives-global-cycle")
        settings_tab("Appearance")
        change(toggle)
        close_settings()
        # An independent browser context with only the login session must fetch the
        # choice from the server rather than inherit the old localStorage setting.
        auth = context.storage_state()
        for origin in auth["origins"]:
            origin["localStorage"] = [
                item for item in origin["localStorage"] if item["name"].startswith("unsloth_auth_")
            ]
        fresh = browser.new_context(storage_state = auth, viewport = {"width": 1440, "height": 1000})
        other = fresh.new_page()
        with other.expect_response(lambda r: r.url == ENDPOINT and r.request.method == "GET"):
            other.goto(BASE + "/chat")
        other.get_by_role("button", name = "Settings", exact = True).click()
        other.get_by_role("button", name = "Appearance", exact = True).click()
        expect(
            other.get_by_role("switch", name = "Decorative mascots", exact = True)
        ).not_to_be_checked()
        other.screenshot(path = str(OUT / "fresh-session-server-preference.png"))
        report.append("fresh-session-server-preference")
        fresh.close()
        # A pre-feature server response must not overwrite an explicit local opt-out.
        # Only this legacy-response check uses an intercepted GET; all normal saves
        # and the independent-session check above use the live backend unchanged.
        put_settings({"appearance": {"customization": {"showMascots": True}}})
        legacy = browser.new_context(storage_state = context.storage_state())
        legacy_page = legacy.new_page()

        def old_personalization(route):
            if route.request.method != "GET":
                route.continue_()
                return
            result = route.fetch()
            payload = result.json()
            payload["appearance"]["customization"].pop("showMascots", None)
            payload["mascotsSaved"] = False
            route.fulfill(response = result, json = payload)

        legacy_page.route(ENDPOINT, old_personalization)
        with legacy_page.expect_response(lambda r: r.url == ENDPOINT and r.request.method == "PUT"):
            legacy_page.goto(BASE + "/chat")
        legacy_page.get_by_role("button", name = "Settings", exact = True).click()
        legacy_page.get_by_role("button", name = "Appearance", exact = True).click()
        expect(
            legacy_page.get_by_role("switch", name = "Decorative mascots", exact = True)
        ).not_to_be_checked()
        assert get_settings()["appearance"]["customization"]["showMascots"] is False
        report.append("legacy-server-preserves-local-off")
        legacy.close()
        # Keep appearance storage but remove authentication in an isolated context.
        # The sign-in illustration is decorative too; the form must remain usable.
        signed_out = context.storage_state()
        signed_out["cookies"] = []
        for origin in signed_out["origins"]:
            origin["localStorage"] = [
                item
                for item in origin["localStorage"]
                if not item["name"].startswith("unsloth_auth_")
            ]
        login_context = browser.new_context(storage_state = signed_out)
        login_page = login_context.new_page()
        login_page.goto(BASE + "/login")
        expect(login_page.get_by_label("Password", exact = True)).to_be_visible()
        expect(login_page.locator('img[src*="Sloth%20emojis"]')).to_have_count(0)
        login_page.screenshot(path = str(OUT / "login-decoration-disabled.png"))
        report.append("login-decoration-disabled")
        login_context.close()
        for enabled in (False, True):
            put_settings({"appearance": {"customization": {"showMascots": enabled}}})
            # Hydrate through the authenticated shell before visiting the not-found page.
            page.goto(BASE + "/chat")
            settings_tab("Appearance")
            expect(toggle).to_be_checked(checked = enabled)
            close_settings()
            page.goto(BASE + "/images/mascot-verification-missing-page")
            sloth = page.locator('img[src*="sloth%20shy"]')
            if enabled:
                expect(sloth).to_be_visible()
            else:
                expect(sloth).to_have_count(0)
            capture("not-found-" + ("enabled" if enabled else "disabled"))
        print(json.dumps({"passed": report, "limitations": limitations}), flush = True)
        (OUT / "report.json").write_text(
            json.dumps({"passed": report, "limitations": limitations}, indent = 2)
        )
    except Exception:
        page.screenshot(path = str(OUT / "failure.png"))
        raise
    finally:
        put_settings({key: original[key] for key in ("version", "profile", "appearance")})
        browser.close()
