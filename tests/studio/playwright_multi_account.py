# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-account browser flow against a disposable Studio: the owner creates an account,
its setup code is private, and switching accounts in one browser clears account data.

    STUDIO_E2E_URL=http://127.0.0.1:8000 STUDIO_E2E_OWNER_PASSWORD=... \
        python tests/studio/playwright_multi_account.py

Skips (exit 0) without the owner password. The account it creates is deleted at the end.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request

from playwright.sync_api import Page, expect, sync_playwright

BASE_URL = os.environ.get("STUDIO_E2E_URL", "http://127.0.0.1:8000").rstrip("/")
OWNER_PASSWORD = os.environ.get("STUDIO_E2E_OWNER_PASSWORD")
ENGINE = os.environ.get("PW_ENGINE", "chromium")
STEP_TIMEOUT_MS = 30_000


def api(
    method: str,
    path: str,
    token: str | None = None,
    body: dict | None = None,
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE_URL + path, data = data, method = method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout = 60) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw) if raw else {}
        except ValueError:
            return error.code, {}


def login(page: Page, username: str, password: str) -> None:
    page.goto(f"{BASE_URL}/login")
    # Wait for the layout the server's login_mode implies rather than sampling before the page settles.
    _, status = api("GET", "/api/auth/status")
    username_field = page.get_by_role("textbox", name = "Username", exact = True)
    if status.get("login_mode") == "multi":
        expect(username_field).to_be_visible(timeout = STEP_TIMEOUT_MS)
        username_field.fill(username)
    else:
        expect(username_field).to_have_count(0)
    page.locator("#password").fill(password)
    page.get_by_role("button", name = "Login", exact = True).click()


def log_out(page: Page) -> None:
    page.evaluate(
        """async () => {
            await fetch("/api/auth/logout", {
                method: "POST",
                headers: { Authorization: `Bearer ${localStorage.getItem("unsloth_auth_token")}` },
            });
            for (const key of ["unsloth_auth_token", "unsloth_auth_refresh_token",
                               "unsloth_auth_must_change_password", "unsloth_auth_session_mark"]) {
                localStorage.removeItem(key);
            }
        }"""
    )


def local(page: Page, key: str):
    return page.evaluate("key => localStorage.getItem(key)", key)


def run(page: Page, context) -> None:
    username = f"e2e_{int(time.time())}"
    managed_password = f"Managed-{int(time.time())}-password"
    status, owner_login = api(
        "POST", "/api/auth/login", body = {"username": "unsloth", "password": OWNER_PASSWORD}
    )
    assert status == 200, ("owner login", status, owner_login)
    owner_token = owner_login["access_token"]
    account_id = None
    try:
        _, initial = api("GET", "/api/auth/status")
        page.goto(f"{BASE_URL}/login")
        page.evaluate(
            """() => {
                localStorage.setItem("unsloth_locale", "en");
                localStorage.setItem("theme", "dark");
                localStorage.setItem("unsloth_e2e_private", "owner-private");
                localStorage.setItem("chat-draft:e2e", "owner-draft");
                localStorage.setItem("unsloth_chat_permission_mode", "full");
            }"""
        )
        if initial.get("login_mode") == "single":
            expect(page.locator("#username")).to_have_count(0)
        login(page, "unsloth", OWNER_PASSWORD)
        expect(page).to_have_url(re.compile(r"/chat"), timeout = STEP_TIMEOUT_MS)
        assert local(page, "unsloth_e2e_private") == "owner-private"

        status, created = api("POST", "/api/accounts", owner_token, {"username": username})
        assert status == 201, ("create account", status, created)
        account_id = created["account"]["account_id"]
        setup_code = created["setup_code"]
        assert setup_code
        expires = datetime.fromisoformat(created["setup_code_expires_at"].replace("Z", "+00:00"))
        assert expires > datetime.now(timezone.utc), expires
        _, after = api("GET", "/api/auth/status")
        assert after.get("login_mode") == "multi", after

        log_out(page)
        page.goto(f"{BASE_URL}/login")
        second_tab = context.new_page()
        second_tab.goto(f"{BASE_URL}/login")
        expect(second_tab.locator("#username")).to_be_visible(timeout = STEP_TIMEOUT_MS)
        second_tab.evaluate("() => { window.oldAccountDocument = true; }")

        # Usernames are case-insensitive at login; the setup code is the first password.
        login(page, username.upper(), setup_code)
        expect(page).to_have_url(re.compile(r"/change-password"), timeout = STEP_TIMEOUT_MS)
        assert local(page, "unsloth_e2e_private") is None
        assert local(page, "chat-draft:e2e") is None
        # Appearance survives the switch; its value may have been replaced by the personalization sync.
        assert local(page, "theme") is not None, "the account switch cleared the theme"
        assert local(page, "unsloth_chat_permission_mode") != "full"
        deadline = time.monotonic() + STEP_TIMEOUT_MS / 1000
        while second_tab.evaluate("() => window.oldAccountDocument") is not None:
            assert time.monotonic() < deadline, "the other tab never reloaded on the account switch"
            time.sleep(0.2)

        page.locator("#current-password").fill(setup_code)
        page.locator("#new-password").fill(managed_password)
        page.locator("#confirm-password").fill(managed_password)
        page.get_by_role("button", name = "Change password", exact = True).click()
        expect(page).to_have_url(re.compile(r"/chat"), timeout = STEP_TIMEOUT_MS)
        managed_token = local(page, "unsloth_auth_token")
        status, _ = api("GET", "/api/accounts", managed_token)
        assert status == 403, ("managed account listing", status)
        page.evaluate("() => localStorage.setItem('unsloth_settings_active_tab', 'accounts')")
        page.reload()
        expect(page.get_by_test_id("settings-tab-accounts")).to_have_count(0)
        page.evaluate("() => localStorage.setItem('unsloth_e2e_private', 'managed-private')")
        log_out(page)
        login(page, "unsloth", OWNER_PASSWORD)
        expect(page).to_have_url(re.compile(r"/chat"), timeout = STEP_TIMEOUT_MS)
        assert local(page, "unsloth_e2e_private") is None
        second_tab.close()
    finally:
        if account_id:
            status, _ = api("DELETE", f"/api/accounts/{account_id}", owner_token)
            assert status == 204, ("delete account", status)


def main() -> int:
    if not OWNER_PASSWORD:
        print("SKIP: set STUDIO_E2E_URL and STUDIO_E2E_OWNER_PASSWORD for a disposable Studio")
        return 0
    with sync_playwright() as playwright:
        browser = getattr(playwright, ENGINE).launch(headless = True)
        context = browser.new_context(viewport = {"width": 1440, "height": 900}, color_scheme = "light")
        context.set_default_timeout(STEP_TIMEOUT_MS)
        page = context.new_page()
        try:
            run(page, context)
        finally:
            browser.close()
    print("PASS: multi-account browser flow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
