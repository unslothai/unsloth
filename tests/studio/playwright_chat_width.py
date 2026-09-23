# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check width presets against a running Studio with a saved conversation.

BASE_URL is the disposable instance, STUDIO_NEW_PW (or STUDIO_PW) its password.

CHAT_THREAD_ID is optional; without it this seeds its own thread. That is what
lets CI run this at all: it measures a rendered assistant bubble, no CI step can
hand one over, so before this no step ran it and the presets shipped ungated.
Seeding costs no model, since the messages endpoint stores whatever roles it is
given and the thread renders from that.

Run with: python tests/studio/playwright_chat_width.py
"""

import os
import time
import uuid

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8888")
PASSWORD = os.environ.get("STUDIO_NEW_PW") or os.environ.get("STUDIO_PW", "")
TIMEOUT_MS = 60_000


def api(
    page,
    path,
    method="GET",
    body=None,
    token=None,
):
    """Call the backend from the page, so the request carries the session cookie."""
    result = page.evaluate(
        """async ([url, method, body, token]) => {
            const headers = { "Content-Type": "application/json" };
            if (token) headers.Authorization = `Bearer ${token}`;
            const response = await fetch(url, {
                method,
                headers,
                body: body === null ? undefined : JSON.stringify(body),
            });
            const text = await response.text();
            let parsed = null;
            try { parsed = JSON.parse(text); } catch { parsed = text; }
            return { status: response.status, body: parsed };
        }""",
        [f"{BASE}{path}", method, body, token],
    )
    if result["status"] >= 400:
        raise AssertionError(f"{method} {path} returned {result['status']}: {result['body']!r}")
    return result["body"]


def sign_in(page):
    """Rotate the bootstrap password if the instance still has one, else log in.

    Two branches, as in playwright_thread_scoped_settings.py: a fresh CI boot lands on
    /change-password, a re-run against the same server on /login.
    """
    page.goto(f"{BASE}/change-password", wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    try:
        page.locator("#new-password").wait_for(state="visible", timeout=15_000)
        rotating = True
    except Exception:  # noqa: BLE001 - already rotated, so the form is not there
        rotating = False
    if rotating:
        page.fill("#new-password", PASSWORD, timeout=TIMEOUT_MS)
        page.fill("#confirm-password", PASSWORD, timeout=TIMEOUT_MS)
        endpoint = "/api/auth/change-password"
    else:
        page.goto(f"{BASE}/login", wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        page.locator("#password").wait_for(state="visible", timeout=TIMEOUT_MS)
        page.fill("#password", PASSWORD, timeout=TIMEOUT_MS)
        endpoint = "/api/auth/login"
    with page.expect_response(
        lambda r: endpoint in r.url and r.request.method == "POST", timeout=TIMEOUT_MS
    ) as response:
        page.locator('button[type="submit"]').click()
    if response.value.status >= 400:
        raise AssertionError(f"POST {endpoint} returned {response.value.status}")
    page.goto(f"{BASE}/chat", wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    page.locator('button[data-pill-label="Search"]:visible').first.wait_for(timeout=TIMEOUT_MS)
    return page.evaluate("() => localStorage.getItem('unsloth_auth_token')")


def seed_thread(page, token):
    """A saved conversation carrying one user turn and one assistant turn.

    The assistant turn is the point: only that role renders the
    `.aui-assistant-message-root` every assertion below measures. Its text is long enough
    to reach the column cap at every viewport tested, so a preset that fails to widen
    reads as a narrower bubble rather than as one that was never wide enough to tell.
    """
    thread_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    api(
        page,
        "/api/chat/threads",
        method="POST",
        token=token,
        body={
            "id": thread_id,
            "title": "chat width presets",
            "modelType": "base",
            "modelId": "",
            "archived": False,
            "createdAt": now,
            "updatedAt": now,
        },
    )
    user_id = str(uuid.uuid4())
    api(
        page,
        f"/api/chat/threads/{thread_id}/messages",
        method="PUT",
        token=token,
        body={
            "messages": [
                {
                    "id": user_id,
                    "threadId": thread_id,
                    "parentId": None,
                    "role": "user",
                    "content": [{"type": "text", "text": "How wide does this column get?"}],
                    "createdAt": now,
                },
                {
                    "id": str(uuid.uuid4()),
                    "threadId": thread_id,
                    "parentId": user_id,
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Seeded reply for the width presets. " * 40}
                    ],
                    "createdAt": now + 1,
                },
            ]
        },
    )
    return thread_id


def check_widths(page):
    """Full must preserve the space available in Wide, including narrow panes."""
    thread_url = page.url
    measurements = {}
    for preset in ("Standard", "Wide", "Full width"):
        page.set_viewport_size({"width": 1440, "height": 900})
        page.keyboard.press("Control+,")
        page.get_by_role("dialog").get_by_role("button", name="Appearance", exact=True).click()
        page.get_by_role("combobox", name="Chat width").click()
        page.get_by_role("option", name=preset, exact=True).click()
        page.keyboard.press("Escape")
        measurements[preset] = {}
        for width in (390, 768, 900, 1280, 1536, 1920, 2560):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(300)
            measurements[preset][width] = page.evaluate(
                """() => {
                    const width = selector => document.querySelector(selector).getBoundingClientRect().width;
                    return {
                        message: width('.aui-assistant-message-root'),
                        composer: width('.unsloth-composer-shell'),
                        overflow: document.documentElement.scrollWidth > innerWidth,
                    };
                }"""
            )
        if preset == "Wide":
            page.set_viewport_size({"width": 1920, "height": 900})
            page.goto(thread_url.split("?")[0] + "?new=width-transition")
            shell = page.locator(".unsloth-composer-shell")
            shell.wait_for()
            page.wait_for_timeout(300)
            welcome_width = shell.bounding_box()["width"]
            assert abs(welcome_width - measurements[preset][1920]["composer"]) <= 1
            page.goto(thread_url)
            page.locator(".aui-assistant-message-root").wait_for()
    for width, full in measurements["Full width"].items():
        standard = measurements["Standard"][width]
        wide = measurements["Wide"][width]
        for surface in ("message", "composer"):
            assert full[surface] >= wide[surface] - 1, (width, surface, full, wide)
            assert wide[surface] >= standard[surface] - 1, (width, surface, wide, standard)
        assert not any(measurements[preset][width]["overflow"] for preset in measurements)
    # The ordering above is >=, so three presets pinned to one width satisfy all of it and
    # a setting that stopped working entirely would read as a pass. At the widest viewport
    # they are separated by construction (appearance-custom-store.ts: 48rem, 72rem,
    # max(72rem, 100% - 6rem)), so require that separation.
    widest = max(measurements["Standard"])
    for surface in ("message", "composer"):
        standard = measurements["Standard"][widest][surface]
        wide = measurements["Wide"][widest][surface]
        full = measurements["Full width"][widest][surface]
        assert wide > standard, (
            f"at {widest}px Wide gives {surface} {wide}px, no more than Standard's "
            f"{standard}px, so the preset is doing nothing"
        )
        assert full > wide, (
            f"at {widest}px Full width gives {surface} {full}px, no more than Wide's "
            f"{wide}px, so the preset is doing nothing"
        )
    return measurements


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        token = sign_in(page)
        thread_id = os.environ.get("CHAT_THREAD_ID") or seed_thread(page, token)
        page.goto(
            f"{BASE}/chat?thread={thread_id}", wait_until="domcontentloaded", timeout=TIMEOUT_MS
        )
        page.locator(".aui-assistant-message-root").wait_for(timeout=TIMEOUT_MS)
        print(check_widths(page))
        browser.close()
