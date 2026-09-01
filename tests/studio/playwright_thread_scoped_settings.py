# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-chat settings: the composer pills and permission level follow the chat.

Drives two saved chats through the real UI and asserts each keeps its own modes across a
switch and a reload, that a chat's edits leave the installation defaults (and so every new
chat) alone, and that a chat which stored nothing still opens on those defaults.

Needs no model: with nothing loaded the Search and Code pills stay clickable, which is what
lets this run in seconds rather than behind a GGUF download.

    BASE_URL=http://127.0.0.1:18921 STUDIO_NEW_PW=... \
        python tests/studio/playwright_thread_scoped_settings.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = os.environ["BASE_URL"]
NEW = os.environ["STUDIO_NEW_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright"))
ART.mkdir(parents = True, exist_ok = True)

TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TIMEOUT_MS", "30000"))

# The installation-wide slots the per-chat edits below must not touch. The legacy confirm toggle
GLOBAL_KEYS = (
    "unsloth_chat_tools_enabled",
    "unsloth_chat_code_tools_enabled",
    "unsloth_chat_permission_mode",
    "unsloth_chat_confirm_tool_calls",
)

_step = [0]


def step(message):
    _step[0] += 1
    print(f"[thread-settings] STEP {_step[0]}: {message}", flush = True)


def fail(message):
    print(f"[thread-settings] FAIL: {message}", flush = True)
    sys.exit(1)


def shoot(page, name):
    try:
        page.screenshot(path = str(ART / f"thread-settings-{name}.png"), full_page = False)
    except Exception:  # noqa: BLE001 - screenshots are diagnostics only
        pass


def api(
    page,
    path,
    method = "GET",
    body = None,
    token = None,
):
    """Call the backend from the page so the request carries the session cookie."""
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
        fail(f"{method} {path} returned {result['status']}: {result['body']!r}")
    return result["body"]


def sign_in(page):
    step("sign in, then land on /chat")
    page.goto(f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000)
    try:
        page.locator("#new-password").wait_for(state = "visible", timeout = 15_000)
        rotate = True
    except Exception:  # noqa: BLE001 - a rerun against the same server is already rotated
        rotate = False
    if rotate:
        page.fill("#new-password", NEW, timeout = TIMEOUT_MS)
        page.fill("#confirm-password", NEW, timeout = TIMEOUT_MS)
        endpoint = "/api/auth/change-password"
    else:
        page.goto(f"{BASE}/login", wait_until = "domcontentloaded", timeout = 60_000)
        page.locator("#password").wait_for(state = "visible", timeout = 60_000)
        page.fill("#password", NEW, timeout = TIMEOUT_MS)
        endpoint = "/api/auth/login"
    with page.expect_response(
        lambda r: endpoint in r.url and r.request.method == "POST",
        timeout = TIMEOUT_MS,
    ) as response_info:
        page.locator('button[type="submit"]').click()
    if response_info.value.status >= 400:
        fail(f"POST {endpoint} returned {response_info.value.status}")
    page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60_000)
    page.locator('button[data-pill-label="Search"]:visible').first.wait_for(
        state = "visible", timeout = 60_000
    )
    return page.evaluate("() => localStorage.getItem('unsloth_auth_token')")


def app_created_thread_id():
    """The id a chat started in the app really carries.

    assistant-ui mints `__LOCALID_<id>` for a thread before its first send, the thread list
    adapter hands that same string back as the remoteId, and the row keeps it as its primary
    key. The prefix therefore says nothing about whether a row exists.
    """
    return f"__LOCALID_{uuid.uuid4().hex}"


def seed_thread(
    page,
    token,
    title,
    thread_id = None,
):
    """Create a saved chat with one message, the state the sidebar and the loader expect."""
    thread_id = thread_id or str(uuid.uuid4())
    now = int(time.time() * 1000)
    api(
        page,
        "/api/chat/threads",
        method = "POST",
        token = token,
        body = {
            "id": thread_id,
            "title": title,
            "modelType": "base",
            "modelId": "",
            "archived": False,
            "createdAt": now,
            "updatedAt": now,
        },
    )
    api(
        page,
        f"/api/chat/threads/{thread_id}/messages",
        method = "PUT",
        token = token,
        body = {
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "threadId": thread_id,
                    "parentId": None,
                    "role": "user",
                    "content": [{"type": "text", "text": f"seed for {title}"}],
                    "createdAt": now,
                }
            ]
        },
    )
    return thread_id


def stored_settings(page, token, thread_id):
    return api(page, f"/api/chat/threads/{thread_id}", token = token).get("settings")


def wait_for_stored_settings(page, token, thread_id, key, value):
    """Block until the debounced snapshot write for `thread_id` has landed."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        settings = stored_settings(page, token, thread_id)
        if settings and settings.get(key) == value:
            return settings
        page.wait_for_timeout(250)
    fail(f"{thread_id} never stored {key}={value!r}: {stored_settings(page, token, thread_id)!r}")


def new_chat_in_page(page):
    """Start a new chat from the sidebar, which routes without reloading the document."""
    button = page.locator('[data-sidebar="menu-button"]').filter(has_text = "New Chat").first
    button.wait_for(state = "visible", timeout = TIMEOUT_MS)
    button.click()
    settle(page)


def open_thread_in_page(page, title):
    """Switch chats the way a user does, without reloading the document."""
    entry = page.locator('[data-testid="recent-thread"]').filter(has_text = title).first
    entry.wait_for(state = "visible", timeout = TIMEOUT_MS)
    entry.click()
    settle(page)


def open_thread(page, thread_id):
    page.goto(
        f"{BASE}/chat?thread={thread_id}",
        wait_until = "domcontentloaded",
        timeout = 60_000,
    )
    settle(page)


def unload_any_model(page, token):
    """Leave no model loaded, so the capability-gated pills stay clickable.

    The Search and Code pills are disabled when a model is loaded that cannot run tools
    (`modelLoaded && !(supportsTools || supportsBuiltinWebSearch)`), and this file drives
    both. In CI an earlier step in the same job leaves a small GGUF resident, which has
    no tool support, so every pill click here would time out on a disabled button. With
    nothing loaded the pills are pre-selectable, which is the state this test is about.
    """
    status = page.evaluate(
        """async ({ base, token }) => {
            const res = await fetch(base + "/api/inference/status", {
                headers: { Authorization: "Bearer " + token },
            });
            if (!res.ok) return null;
            return await res.json();
        }""",
        {"base": BASE, "token": token},
    )
    status = status or {}
    loaded = status.get("model_identifier") or (status.get("loaded") or [None])[0]
    if not loaded:
        return
    print(
        f"[thread-settings] unloading {loaded!r} so the pills are not capability-gated", flush = True
    )
    page.evaluate(
        """async ({ base, token, modelPath }) => {
            await fetch(base + "/api/inference/unload", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: "Bearer " + token,
                },
                body: JSON.stringify({ model_path: modelPath }),
            });
        }""",
        {"base": BASE, "token": token, "modelPath": loaded},
    )
    page.wait_for_timeout(1500)


def settle(page):
    """Wait for the composer, then for the thread's snapshot to have been applied."""
    page.locator('button[data-pill-label="Search"]:visible').first.wait_for(
        state = "visible", timeout = TIMEOUT_MS
    )
    # the snapshot arrives on a GET, and the pin write is debounced behind it.
    page.wait_for_timeout(1200)


def pill(page, label):
    return page.locator(f'button[data-pill-label="{label}"]:visible').first


def permission_pill(page):
    return page.locator('button[aria-label="Permission level for tool calls"]:visible').first


def choose_permission(page, label):
    permission_pill(page).click()
    menu = page.get_by_role("menu").last
    expect(menu).to_be_visible()
    menu.get_by_role("menuitem").filter(has_text = label).first.click()
    expect(permission_pill(page)).to_have_attribute("data-pill-label", label)


def expect_pills(page, where, search, code, permission):
    for label, wanted in (("Search", search), ("Code", code)):
        expect(pill(page, label)).to_have_attribute(
            "data-active", "true" if wanted else "false", timeout = TIMEOUT_MS
        )
    expect(permission_pill(page)).to_have_attribute(
        "data-pill-label", permission, timeout = TIMEOUT_MS
    )
    print(
        f"[thread-settings]   {where}: Search={search} Code={code} " f"permission={permission!r}",
        flush = True,
    )


def read_globals(page):
    return page.evaluate(
        """(keys) => Object.fromEntries(keys.map((k) => [k, localStorage.getItem(k)]))""",
        list(GLOBAL_KEYS),
    )


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args = ["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport = {"width": 1280, "height": 900})
        page = context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        page_errors = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        token = sign_in(page)
        if not token:
            fail("no auth token in localStorage after change-password")
        unload_any_model(page, token)
        page.goto(f"{BASE}/chat", wait_until = "domcontentloaded", timeout = 60_000)

        step("an unsaved chat still edits the installation defaults")
        # plain /chat runs on a runtime-made thread id with no row:
        settle(page)
        pill(page, "Search").click()
        page.wait_for_timeout(600)
        enabled_globals = read_globals(page)
        if enabled_globals["unsloth_chat_tools_enabled"] != "true":
            fail(f"an unsaved chat's edit never reached the defaults: {enabled_globals!r}")
        pill(page, "Search").click()
        page.wait_for_timeout(600)
        disabled_globals = read_globals(page)
        if disabled_globals["unsloth_chat_tools_enabled"] != "false":
            fail(f"toggling back never reached the defaults: {disabled_globals!r}")

        step("pin the installation default every later step compares against")
        # The install is shared, not fresh:
        choose_permission(page, "Approve for me")
        print(
            f"[thread-settings]   defaults now {read_globals(page)!r}",
            flush = True,
        )

        step("seed two saved chats")
        # Both id shapes are real: chats started in the app keep their `__LOCALID_` id as the row's primary key
        thread_a = seed_thread(page, token, "Chat A", app_created_thread_id())
        thread_b = seed_thread(page, token, "Chat B")
        print(f"[thread-settings]   A={thread_a} B={thread_b}", flush = True)

        step("a chat with no snapshot of its own opens on those defaults")
        open_thread(page, thread_a)
        expect_pills(page, "A on first open", False, False, "Approve for me")
        defaults = read_globals(page)

        step("set Chat A to Search on, Ask for approval")
        pill(page, "Search").click()
        choose_permission(page, "Ask for approval")
        expect_pills(page, "A after editing", True, False, "Ask for approval")
        shoot(page, "01-chat-a-edited")
        wait_for_stored_settings(page, token, thread_a, "toolsEnabled", True)

        step("Chat A's edits leave the installation defaults alone")
        after_edit = read_globals(page)
        if after_edit != defaults:
            fail(
                "editing inside a chat moved the installation defaults: "
                f"{defaults!r} -> {after_edit!r}"
            )

        step("Chat B opens on the defaults, not on Chat A's modes")
        open_thread(page, thread_b)
        expect_pills(page, "B on first open", False, False, "Approve for me")

        step("set Chat B to Code on, Run automatically")
        pill(page, "Code").click()
        choose_permission(page, "Run automatically")
        expect_pills(page, "B after editing", False, True, "Run automatically")
        shoot(page, "02-chat-b-edited")
        wait_for_stored_settings(page, token, thread_b, "codeToolsEnabled", True)

        step("switching back to Chat A restores Chat A's own modes")
        open_thread(page, thread_a)
        expect_pills(page, "A after switching back", True, False, "Ask for approval")

        step("and they survive a full reload")
        page.reload(wait_until = "domcontentloaded")
        settle(page)
        expect_pills(page, "A after reload", True, False, "Ask for approval")
        shoot(page, "03-chat-a-after-reload")

        step("switching back to Chat B restores Chat B's own modes")
        open_thread(page, thread_b)
        expect_pills(page, "B after switching back", False, True, "Run automatically")

        step("and a sidebar switch, with no reload, does the same")
        # The reload-free path is the one users take, and the only one where the store still holds the outgoing chat's
        open_thread_in_page(page, "Chat A")
        expect_pills(page, "A after an in-page switch", True, False, "Ask for approval")
        open_thread_in_page(page, "Chat B")
        expect_pills(page, "B after an in-page switch", False, True, "Run automatically")
        shoot(page, "03-in-page-switch")

        step("leaving a chat for a new one restores the installation defaults in place")
        # No reload here either, so the defaults have to come from the captured copy rather than from the store being
        new_chat_in_page(page)
        expect_pills(page, "new chat after an in-page switch", False, False, "Approve for me")

        step("a new chat still starts from the installation defaults")
        page.goto(
            f"{BASE}/chat?new={uuid.uuid4()}",
            wait_until = "domcontentloaded",
            timeout = 60_000,
        )
        settle(page)
        expect_pills(page, "new chat", False, False, "Approve for me")
        shoot(page, "04-new-chat")
        if read_globals(page) != defaults:
            fail("the installation defaults changed at some point during the run")

        step("a chat edited before it had modes of its own keeps the new defaults")
        # no chat is open, so this moves the defaults every snapshot-less chat follows.
        pill(page, "Search").click()
        page.wait_for_timeout(600)
        moved = read_globals(page)
        if moved["unsloth_chat_tools_enabled"] != "true":
            fail(f"a new-chat edit did not reach the defaults: {moved!r}")
        thread_c = seed_thread(page, token, "Chat C")
        open_thread(page, thread_c)
        expect_pills(page, "C on first open", True, False, "Approve for me")

        step("the pinned snapshots reached the backend")
        stored = {
            "A": api(page, f"/api/chat/threads/{thread_a}", token = token).get("settings"),
            "B": api(page, f"/api/chat/threads/{thread_b}", token = token).get("settings"),
            "C": api(page, f"/api/chat/threads/{thread_c}", token = token).get("settings"),
        }
        print(f"[thread-settings]   stored={json.dumps(stored, sort_keys = True)}", flush = True)
        if not stored["A"] or stored["A"].get("toolsEnabled") is not True:
            fail(f"Chat A's snapshot did not persist: {stored['A']!r}")
        if stored["A"].get("permissionMode") != "ask":
            fail(f"Chat A stored the wrong permission level: {stored['A']!r}")
        if not stored["B"] or stored["B"].get("codeToolsEnabled") is not True:
            fail(f"Chat B's snapshot did not persist: {stored['B']!r}")
        if stored["B"].get("toolsEnabled") is not False:
            fail(f"Chat B inherited Chat A's Search pill: {stored['B']!r}")
        if stored["B"].get("permissionMode") != "off":
            fail(f"Chat B stored the wrong permission level: {stored['B']!r}")
        if not stored["C"]:
            fail("opening a chat did not pin the modes it was showing")

        step("the thread listing stays free of the snapshot")
        listing = api(page, "/api/chat/threads", token = token)
        for thread in listing.get("threads", []):
            if thread.get("settings") is not None:
                fail(f"thread listing carries a settings snapshot: {thread['id']}")

        if page_errors:
            fail(f"page errors during the run: {page_errors[:3]!r}")

        context.close()
        browser.close()

    print("[thread-settings] PASS", flush = True)


if __name__ == "__main__":
    main()
