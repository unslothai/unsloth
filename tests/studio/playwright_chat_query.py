# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Query-link routing/composer regression against a built frontend, with no model or backend.

Uses the existing production build. BASE_URL can select an already-running preview.
Auth responses are fixtures; other APIs return 503.
The assertions cover the real router and composer, not backend authentication or inference.
"""

import json
import os
from urllib.parse import quote, urlparse

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PORT = int(os.environ.get("PW_PORT", "5417"))
BASE = os.environ.get("BASE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
BROWSER = os.environ.get("PW_BROWSER", "chromium")


def fixture(
    browser,
    *,
    authenticated = True,
    existing_draft = False,
    password_change = False,
):
    context = browser.new_context()
    values = {}
    if authenticated:
        values["unsloth_auth_token"] = "query-test-token"
    if existing_draft:
        values["chat-draft:latest"] = "my unsent message"
        values["chat-draft-pastes:latest"] = json.dumps(["my unsent pasted file"])
    context.add_init_script(
        """(() => {
      if (localStorage.getItem('query-test-seeded')) return;
      for (const [key, value] of Object.entries(%s)) localStorage.setItem(key, value);
      localStorage.setItem('query-test-seeded', '1');
    })();"""
        % json.dumps(values)
    )
    posts = []
    state = {"change": password_change}

    def api(route):
        path = urlparse(route.request.url).path
        if route.request.method == "POST":
            posts.append(path)
        if path == "/api/auth/status":
            body = {
                "initialized": True,
                "requires_password_change": state["change"],
                "login_mode": "single",
                "full_access": True,
            }
        elif path in ("/api/auth/login", "/api/auth/change-password", "/api/auth/refresh"):
            if path == "/api/auth/change-password":
                state["change"] = False
            body = {
                "access_token": "query-test-token",
                "refresh_token": "query-test-refresh",
                "must_change_password": state["change"],
                "account_id": "owner",
            }
        else:
            route.fulfill(status = 503, content_type = "application/json", body = "{}")
            return
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(body))

    context.route("**/api/**", api)
    return context, posts


def composer(page):
    return page.locator("form textarea").first


def assert_unsent(posts):
    allowed = {
        "/api/auth/login",
        "/api/auth/change-password",
        "/api/auth/refresh",
        "/api/settings/chat-preferences/migrate",
        "/api/chat/sidebar/rows",
    }
    assert not set(posts) - allowed, f"Unexpected automatic action: {set(posts) - allowed}"


def run(browser):
    for route, text in [
        ("/chat", "null"),
        ("/", '{"x":1}'),
        ("/", "1.00"),
        ("/chat", '"quoted" + café & a=b'),
    ]:
        context, posts = fixture(browser)
        page = context.new_page()
        page.goto(BASE + route + "?q=" + quote(text, safe = ""), wait_until = "domcontentloaded")
        expect(composer(page)).to_have_value(text, timeout = 20000)
        assert "q=" not in page.url and "new=" in page.url
        assert_unsent(posts)
        context.close()
    print("PASS literal queries on both routes, clean URL, no automatic send", flush = True)

    context, posts = fixture(browser, existing_draft = True)
    page = context.new_page()
    page.goto(BASE + "/chat?q=new+question", wait_until = "domcontentloaded")
    expect(composer(page)).to_have_value("new question", timeout = 20000)
    expect(page.get_by_role("button", name = "Pasted text:", exact = False)).to_have_count(0)
    assert page.evaluate("localStorage.getItem('chat-draft:latest')") == "my unsent message"
    assert json.loads(page.evaluate("localStorage.getItem('chat-draft-pastes:latest')")) == [
        "my unsent pasted file"
    ]
    composer(page).fill("edited query")
    page.reload(wait_until = "domcontentloaded")
    expect(composer(page)).to_have_value("edited query", timeout = 20000)
    page.get_by_role("button", name = "New chat", exact = True).first.click()
    expect(composer(page)).to_have_value("my unsent message", timeout = 20000)
    expect(
        page.get_by_role(
            "button", name = "Pasted text: my unsent pasted file.txt. Show in text field", exact = True
        )
    ).to_have_count(1)
    assert_unsent(posts)
    context.close()
    print("PASS existing text/files preserved, isolated query edits survive reload", flush = True)

    context, posts = fixture(browser, existing_draft = True)
    query_chunks = []
    context.on(
        "request",
        lambda request: query_chunks.append(request.url)
        if "/assets/query-draft-" in request.url
        else None,
    )
    page = context.new_page()
    page.goto(BASE + "/chat?new=ordinary-test", wait_until = "networkidle")
    expect(composer(page)).to_have_value("my unsent message", timeout = 20000)
    # Establish this session's draft by typing, independently of legacy cold-start restoration.
    composer(page).fill("message typed before the query")
    page.wait_for_function(
        "localStorage.getItem('chat-draft:latest') === 'message typed before the query'"
    )
    assert not query_chunks, "Query persistence was loaded on ordinary chat startup"
    page.get_by_role("button", name = "Turn on temporary chat", exact = True).click()
    # A same-document navigation exercises the persistent runtime as browser history does.
    page.evaluate(
        "history.pushState({}, '', '/chat?q=temporary+query'); dispatchEvent(new PopStateEvent('popstate'))"
    )
    expect(composer(page)).to_have_value("temporary query", timeout = 20000)
    assert query_chunks, "The on-use query chunk was not requested"
    expect(page.get_by_role("button", name = "Pasted text:", exact = False)).to_have_count(0)
    composer(page).fill("temporary edit")
    page.wait_for_timeout(400)
    assert (
        page.evaluate("localStorage.getItem('chat-draft:latest')")
        == "message typed before the query"
    )
    assert (
        page.evaluate("Object.keys(sessionStorage).filter(k => k.startsWith('chat-query-draft:'))")
        == []
    )
    page.get_by_role("button", name = "Turn off temporary chat", exact = True).click()
    expect(composer(page)).to_have_value("message typed before the query", timeout = 20000)
    assert_unsent(posts)
    context.close()
    print("PASS temporary query is visible and memory-only", flush = True)

    context, posts = fixture(browser, authenticated = False)
    page = context.new_page()
    page.goto(BASE + "/?q=" + quote('signed out {"x":1}', safe = ""), wait_until = "domcontentloaded")
    expect(page.get_by_label("Password", exact = True)).to_be_visible(timeout = 20000)
    assert "/login?" in page.url and "q=" in page.url
    page.get_by_label("Password", exact = True).fill("test-password")
    page.get_by_role("button", name = "Login", exact = True).click()
    expect(composer(page)).to_have_value('signed out {"x":1}', timeout = 20000)
    assert "q=" not in page.url
    assert_unsent(posts)
    context.close()
    print("PASS query survives sign-in and stays unsent", flush = True)

    context, posts = fixture(browser, authenticated = False, password_change = True)
    page = context.new_page()
    page.goto(BASE + "/chat?q=password+change+query", wait_until = "domcontentloaded")
    expect(page.get_by_label("Current password", exact = True)).to_be_visible(timeout = 20000)
    assert "/change-password?" in page.url and "q=" in page.url
    page.get_by_label("Current password", exact = True).fill("old-password")
    page.get_by_label("New password", exact = True).fill("new-password")
    page.get_by_label("Confirm password", exact = True).fill("new-password")
    page.locator("form button[type=submit]").click()
    expect(composer(page)).to_have_value("password change query", timeout = 20000)
    assert "q=" not in page.url
    assert_unsent(posts)
    context.close()
    print("PASS query survives required password change", flush = True)


if __name__ == "__main__":
    server = None
    try:
        if "BASE_URL" not in os.environ:
            server = start_vite(PORT, preview = True)
            wait_for_smoke_page(BASE, "/assets/index-", proc = server)
        with sync_playwright() as playwright:
            browser = getattr(playwright, BROWSER).launch()
            try:
                run(browser)
            finally:
                browser.close()
    finally:
        if server is not None:
            stop_process(server)
