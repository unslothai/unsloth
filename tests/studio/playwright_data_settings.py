# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data settings against memory-only chat storage, with no backend or model.

PW_ENGINE=chromium|firefox|webkit selects the browser engine.
PW_CHANNEL=chrome|msedge optionally selects a branded Chromium browser.
PW_PORT and PW_OUT select the local port and JSON report.
"""

import itertools
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

FIXTURE = """(() => {
    localStorage.setItem('unsloth_chat_legacy_imported_to_studio_db', 'true');
    const fixture = { rows: [], requests: [], hold: false, fail: false, listFail: false };
    window.__dataFixture = fixture;
    window.fetch = async (input, init = {}) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.origin);
        const method = init.method || input?.method || 'GET';
        const record = { path: url.pathname, query: url.search, method, done: false };
        fixture.requests.push(record);
        let body = {}, status = 200;
        if (url.pathname === '/api/chat/threads') {
            if (fixture.listFail) { status = 503; body = { detail: 'Chat list unavailable' }; }
            else body = { threads: fixture.rows };
        } else if (url.pathname === '/api/chat/projects') body = { projects: [] };
        else if (url.pathname === '/api/chat' && method === 'DELETE') {
            if (fixture.hold) await new Promise(resolve => { fixture.release = resolve; });
            if (fixture.fail) { status = 503; body = { detail: 'Deletion unavailable' }; }
            else {
                body = { deletedThreadIds: fixture.rows.map(row => row.id), sandboxes_kept: [] };
                fixture.rows = [];
            }
        } else if (url.pathname.includes('knowledge-bases')) {
            body = { knowledgeBases: [], ragAvailable: false };
        } else if (url.pathname.includes('linked-folders')) body = { folders: [] };
        else if (url.pathname.includes('gallery')) body = { images: [], videos: [], clips: [], total: 0 };
        else if (url.pathname.includes('messages')) body = { messages: [] };
        record.done = true;
        return new Response(JSON.stringify(body), {
            status, headers: { 'Content-Type': 'application/json' },
        });
    };
})();"""


def run(page):
    checks = []

    def reset(**options):
        page.evaluate("window.__settingsSmoke.close()")
        expect(page.get_by_role("dialog")).to_have_count(0)
        page.evaluate(
            """options => {
            const f = window.__dataFixture;
            Object.assign(f, { requests: [], hold: false, fail: false, listFail: false }, options);
            f.rows = Array.from({ length: options.count ?? 3 }, (_, i) => ({
                id: crypto.randomUUID(), title: `Chat ${i}`, modelType: 'base', modelId: 'test',
                createdAt: 1700000000000 + i, updatedAt: 1700000000000 + i,
            }));
            window.__settingsSmoke.open('data');
        }""",
            options,
        )
        page.get_by_role("combobox", name = "Chat sandbox files", exact = True).wait_for()
        page.wait_for_function(
            "window.__dataFixture.requests.some(r => r.path === '/api/chat/threads' && r.done)"
        )

    def deletes():
        return page.evaluate("window.__dataFixture.requests.filter(r => r.method === 'DELETE')")

    def confirm(mode):
        page.get_by_role("button", name = "Delete all", exact = True).click()
        label = "Delete chats and sandboxes…" if mode else "Delete chats only…"
        page.get_by_role("menuitem", name = label, exact = True).click()
        return page.get_by_role("dialog").last

    for preference, mode, override in itertools.product([False, True], repeat = 3):
        reset()
        page.get_by_role("combobox", name = "Chat sandbox files", exact = True).click()
        page.get_by_role(
            "option",
            name = "Delete sandbox files" if preference else "Keep sandbox files",
            exact = True,
        ).click()
        dialog = confirm(mode)
        toggle = dialog.get_by_role("switch")
        expect(toggle).to_have_attribute("aria-checked", str(mode).lower())
        assert not deletes(), "The menu must only open confirmation"
        if override:
            toggle.click()
        dialog.get_by_role("button", name = "Clear 3 chats", exact = True).click()
        expect(page.get_by_role("button", name = "Delete all", exact = True)).to_be_disabled()
        requests = deletes()
        assert len(requests) == 1
        assert requests[0]["query"] == ("?delete_files=true" if mode != override else "")
        checks.append(f"choice-{preference}-{mode}-{override}")

    for dismissal in ["Cancel", "Close", "Escape"]:
        reset()
        dialog = confirm(True)
        if dismissal == "Escape":
            page.keyboard.press("Escape")
        else:
            dialog.get_by_role("button", name = dismissal, exact = True).click()
        expect(page.locator("#clear-chats-delete-files")).to_have_count(0)
        assert not deletes()
        checks.append(f"dismiss-{dismissal}")

    reset(hold = True)
    dialog = confirm(True)
    dialog.get_by_role("button", name = "Clear 3 chats", exact = True).click()
    page.wait_for_function("typeof window.__dataFixture.release === 'function'")
    expect(dialog.get_by_role("switch")).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Cancel", exact = True)).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Close", exact = True)).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.locator("#clear-chats-delete-files")).to_be_visible()
    page.mouse.click(5, 5)
    expect(page.locator("#clear-chats-delete-files")).to_be_visible()
    page.evaluate("window.__dataFixture.release()")
    expect(page.locator("#clear-chats-delete-files")).to_have_count(0)
    assert len(deletes()) == 1
    checks.append("pending-delete-locks-choice-and-dismissal")

    reset(fail = True)
    dialog = confirm(True)
    dialog.get_by_role("button", name = "Clear 3 chats", exact = True).click()
    expect(page.locator("#clear-chats-delete-files")).to_have_count(0)
    expect(page.get_by_role("button", name = "Delete all", exact = True)).to_be_enabled()
    assert page.evaluate("window.__dataFixture.rows.length") == 3
    assert len(deletes()) == 2
    checks.append("failed-delete-preserves-chats-and-reenables-action")

    for count in [0, 1, 1000]:
        reset(count = count)
        trigger = page.get_by_role("button", name = "Delete all", exact = True)
        if count == 0:
            expect(trigger).to_be_disabled()
        else:
            dialog = confirm(False)
            expect(dialog.get_by_role("heading")).to_contain_text(str(count))
            dialog.get_by_role("button", name = "Cancel", exact = True).click()
        checks.append(f"count-{count}")

    reset(listFail = True)
    expect(page.get_by_role("button", name = "Delete all", exact = True)).to_be_disabled()
    checks.append("failed-count-cannot-open-zero-chat-confirmation")

    reset()
    for label in ["Archived chats", "Archived images", "Archived videos", "Archived audio"]:
        button = page.get_by_role("button", name = label, exact = True)
        expect(button.locator("svg")).to_have_count(2)
        button.click()
        page.get_by_role("button", name = "Back to Data", exact = True).click()
        checks.append(f"archive-{label}")
    errors = page.evaluate("window.__settingsSmoke.errors()")
    resize_notice = "ResizeObserver loop completed with undelivered notifications."
    failures = [error for error in errors if error != resize_notice]
    assert not failures, failures
    return {"checks": checks, "browser_notices": errors}


def main():
    engine = os.environ.get("PW_ENGINE", "chromium")
    port = int(os.environ.get("PW_PORT", "5412"))
    output = Path(os.environ.get("PW_OUT", "logs/data_settings_report.json"))
    server = start_vite(port)
    try:
        url = f"http://127.0.0.1:{port}/smoke-settings.html"
        wait_for_smoke_page(url, "smoke-settings-main.tsx", proc = server, timeout_s = 60)
        with sync_playwright() as p:
            options = {"headless": True}
            if os.environ.get("PW_CHANNEL"):
                options["channel"] = os.environ["PW_CHANNEL"]
            browser = getattr(p, engine).launch(**options)
            page = browser.new_page(viewport = {"width": 1280, "height": 1000})
            page.add_init_script(FIXTURE)
            page.goto(url, wait_until = "domcontentloaded")
            page.wait_for_function("!!window.__settingsSmoke", timeout = 120000)
            output.parent.mkdir(parents = True, exist_ok = True)
            result = {"engine": engine, "version": browser.version}
            try:
                result.update(run(page))
            except Exception as error:
                result["error"] = str(error)
                page.screenshot(path = str(output.with_suffix(".png")))
                raise
            finally:
                output.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
            print(f"{engine}: {len(result['checks'])} Data settings checks passed", flush = True)
            browser.close()
    finally:
        stop_process(server)


if __name__ == "__main__":
    main()
