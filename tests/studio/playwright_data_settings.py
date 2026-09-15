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
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

FIXTURE = """(() => {
    localStorage.setItem('unsloth_chat_legacy_imported_to_studio_db', 'true');
    const fixture = { rows: [], requests: [], hold: false, fail: false, listFail: false, media: {}, projects: [{ id: "research", name: "Research notes", createdAt: 1, updatedAt: 1 }, { id: "other", name: "Other project", createdAt: 2, updatedAt: 2 }, { id: "archived-project", name: "Old experiments", createdAt: 3, updatedAt: 3, archived: true }] };
    window.__dataFixture = fixture;
    window.fetch = async (input, init = {}) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.origin);
        const method = init.method || input?.method || 'GET';
        const record = { path: url.pathname, query: url.search, method, done: false };
        record.body = init.body ? JSON.parse(init.body) : null;
        fixture.requests.push(record);
        let body = {}, status = 200;
        if (url.pathname === '/api/chat/threads' && method === 'DELETE') {
            if (fixture.holdMutation) await new Promise(resolve => { fixture.releaseMutation = resolve; });
            if (fixture.failMutation) { status = 503; body = { detail: 'Mutation unavailable' }; }
            else { fixture.rows = fixture.rows.filter(row => !record.body.ids.includes(row.id)); body = { sandboxes_kept: [] }; }
        } else if (url.pathname.startsWith('/api/chat/threads/') && method === 'PATCH') {
            const row = fixture.rows.find(row => row.id === decodeURIComponent(url.pathname.split('/').pop()));
            Object.assign(row, record.body); body = row;
        } else if (url.pathname === '/api/chat/threads') {
            if (fixture.listFail) { status = 503; body = { detail: 'Chat list unavailable' }; }
            else body = { threads: fixture.rows };
        } else if (url.pathname === '/api/chat/projects') body = { projects: fixture.projects.filter(project => url.searchParams.get('include_archived') !== 'false' || !project.archived) };
        else if (url.pathname === '/api/chat/export') {
            if (fixture.holdExport) await new Promise(resolve => { fixture.releaseExport = resolve; });
            body = { threads: fixture.rows, messages: [], projects: [] };
        }
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
        else if (url.pathname.includes('gallery')) {
            const kind = url.pathname.includes('/images/') ? 'images' : url.pathname.includes('/video/') ? 'videos' : 'audio';
            const entries = fixture.media[kind] ?? [];
            const suffix = url.pathname.split('/gallery')[1];
            if (suffix && (method === 'PATCH' || method === 'DELETE')) {
                if (fixture.holdMutation) await new Promise(resolve => { fixture.releaseMutation = resolve; });
                if (fixture.failMutation || fixture.failMutationId === decodeURIComponent(suffix.slice(1))) { status = 503; body = { detail: 'Mutation unavailable' }; }
                else {
                    const id = decodeURIComponent(suffix.slice(1));
                    body = entries.find(row => row.id === id) ?? {};
                    fixture.media[kind] = entries.filter(row => row.id !== id);
                }
            } else if (!suffix) {
                const before = url.searchParams.get('before_id');
                const offset = before ? entries.findIndex(row => row.id === before) + 1 : Number(url.searchParams.get('offset') ?? 0);
                const limit = Number(url.searchParams.get('limit') ?? 20);
                const pageRows = entries.slice(offset, offset + limit);
                if (offset > 0 && fixture.holdPage) await new Promise(resolve => { fixture.releasePage = resolve; });
                if (fixture.failPage && offset > 0) { status = 503; body = { detail: 'Page unavailable' }; }
                else {
                    const last = pageRows.at(-1);
                    body = { [kind]: fixture.stallPage && offset > 0 ? [] : pageRows, has_more: fixture.stallPage && offset > 0 || offset + limit < entries.length,
                        next_before_mtime: last ? offset + pageRows.length : null, next_before_id: last?.id ?? null };
                }
            } else {
                return new Response(Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZwoAAAAASUVORK5CYII='), c => c.charCodeAt(0)), { headers: { 'Content-Type': 'image/png' } });
            }
        }
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
        expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
        page.evaluate(
            """options => {
            const f = window.__dataFixture;
            Object.assign(f, { requests: [], hold: false, holdExport: false, fail: false, listFail: false }, options);
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

    for source, target in itertools.product(
        [
            "Manage chats",
            "Uploaded files",
            "Archived chats",
            "Archived images",
            "Archived videos",
            "Archived audio",
        ],
        [
            "Archived chats",
            "Archived images",
            "Archived videos",
            "Archived audio",
            "Chat sandbox files",
            "Uploaded files",
        ],
    ):
        reset()
        if source.startswith("Archived"):
            page.get_by_role("button", name = source, exact = True).click()
        else:
            page.locator(f'[data-settings-label="{source}"]').get_by_role(
                "button", name = "Manage", exact = True
            ).click()
        expect(page.get_by_role("button", name = "Back to Data", exact = True)).to_be_visible()
        page.locator("aside input").fill(target)
        page.locator("aside").get_by_role("button", name = target, exact = True).click()
        expect(page.locator(".settings-search-hit")).to_have_attribute(
            "data-settings-label", target
        )
        expect(page.get_by_role("button", name = "Back to Data", exact = True)).to_have_count(0)
        assert not deletes()
        checks.append(f"search-{source}-to-{target}")

    for shelf in ["chats", "images", "videos", "audio"]:
        page.evaluate("shelf => window.__settingsSmoke.openArchived(shelf)", shelf)
        expect(page.get_by_role("heading", name = f"Archived {shelf}", exact = True)).to_be_visible()
        page.locator("aside input").fill("Chat sandbox files")
        page.locator("aside").get_by_role("button", name = "Chat sandbox files", exact = True).click()
        expect(page.locator(".settings-search-hit")).to_have_attribute(
            "data-settings-label", "Chat sandbox files"
        )
        expect(page.get_by_role("button", name = "Back to Data", exact = True)).to_have_count(0)
        checks.append(f"archive-request-after-search-{shelf}")

    reset(holdExport = True)
    export_row = page.locator('[data-settings-label="Export chat history"]')
    export_row.get_by_role("button", name = "Export", exact = True).click()
    page.wait_for_function("typeof window.__dataFixture.releaseExport === 'function'")
    page.locator("aside input").fill("Chat sandbox files")
    page.locator("aside").get_by_role("button", name = "Chat sandbox files", exact = True).click()
    expect(page.locator(".settings-search-hit")).to_have_attribute(
        "data-settings-label", "Chat sandbox files"
    )
    expect(export_row.get_by_role("button", name = "Exporting...", exact = True)).to_be_disabled()
    with page.expect_download():
        page.evaluate("window.__dataFixture.releaseExport()")
    expect(export_row.get_by_role("button", name = "Export", exact = True)).to_be_enabled()
    assert (
        page.evaluate(
            "window.__dataFixture.requests.filter(r => r.path === '/api/chat/export').length"
        )
        == 1
    )
    checks.append("search-preserves-in-flight-export")

    checks.extend(run_libraries(page))
    checks.extend(run_library_locales(page))
    checks.extend(run_library_selection(page))
    checks.extend(run_library_collation(page))
    checks.extend(run_restore_notifications(page))
    checks.extend(run_thumbnail_retention(page))

    errors = page.evaluate("window.__settingsSmoke.errors()")
    resize_notice = "ResizeObserver loop completed with undelivered notifications."
    failures = [error for error in errors if error != resize_notice]
    assert not failures, failures
    return {"checks": checks, "browser_notices": errors}


def run_libraries(page):
    checks = []

    def seed(shelf, **options):
        page.evaluate("window.__settingsSmoke.close()")
        expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
        page.evaluate(
            """({shelf, options}) => {
            const f = window.__dataFixture;
            Object.assign(f, { requests: [], holdMutation: false, failMutation: false,
                holdPage: false, failPage: false, stallPage: false, listFail: false }, options);
            f.rows = Array.from({ length: 32 }, (_, i) => ({
                id: crypto.randomUUID(), title: i === 26 ? 'Café needle' : i === 0 ? 'Zulu sample' : `Chat ${String(i).padStart(2, '0')}`,
                modelType: 'base', modelId: 'test',
                projectId: i === 31 && shelf === 'chats' ? 'archived-project' : i % 2 === 0 || i === 1 ? 'research' : 'other',
                archived: shelf === 'chats', createdAt: 1700000000000 + i * 86400000,
                updatedAt: 1700000000000 + (32 - i) * 86400000,
            }));
            delete f.releaseMutation; delete f.releasePage;
            const pairId = crypto.randomUUID();
            for (const i of [1, 2]) Object.assign(f.rows[i], { pairId, title: 'Compare models' });
            for (const kind of ['images', 'videos', 'audio']) {
                const route = kind === 'videos' ? 'video' : kind;
                f.media[kind] = Array.from({ length: options.mediaCount ?? 447 }, (_, i) => ({
                    id: `${kind}-${i}`, prompt: i === 26 ? 'Café needle' : `Sample ${String(i).padStart(2, '0')}`,
                    created_at: kind === 'images' ? 1700000000 - i * 86400 : new Date(1700000000000 - i * 86400000).toISOString(),
                    url: `/api/inference/${route}/gallery/${kind}-${i}/content`, archived: true,
                }));
            }
            if (options.reorderDates) {
                for (const kind of ['images', 'videos', 'audio']) f.media[kind][300].created_at = kind === 'images' ? 1900000000 : new Date(1900000000000).toISOString();
            }
            if (shelf === 'manage') window.__settingsSmoke.open('data');
            else window.__settingsSmoke.openArchived(shelf);
        }""",
            {"shelf": shelf, "options": options},
        )
        if shelf == "manage":
            page.locator('[data-settings-label="Manage chats"]').get_by_role(
                "button", name = "Manage", exact = True
            ).click()
            label = "Search chats or projects"
        else:
            label = (
                "Search archived chats or projects"
                if shelf == "chats"
                else f"Search archived {shelf}"
            )
        search = page.get_by_role("searchbox", name = label, exact = True)
        search.wait_for()
        page.evaluate("window.dispatchEvent(new Event('unsloth-chat-projects-updated'))")
        return search

    def sort(label):
        page.get_by_role("button", name = "Filter and sort", exact = True).click()
        page.get_by_role("menuitemradio", name = label, exact = True).click()

    def mutation_requests():
        return page.evaluate(
            "window.__dataFixture.requests.filter(r => ['PATCH', 'DELETE'].includes(r.method))"
        )

    for shelf in ["manage", "chats"]:
        search = seed(shelf)
        search.fill("  RESEARCH   cafe ")
        expect(page.get_by_role("button", name = "Café needle", exact = True)).to_be_visible()
        expect(page.get_by_role("button", name = "Zulu sample", exact = True)).to_have_count(0)
        checks.append(f"{shelf}-search-title-and-project-across-pages-unicode")
        search.fill("missing query")
        expect(
            page.get_by_text(
                "No chats match your search."
                if shelf == "manage"
                else "No archived chats match your search.",
                exact = True,
            )
        ).to_be_visible()
        search.fill("")
        page.get_by_role("button", name = "Filter by project", exact = True).click()
        page.get_by_role("combobox", name = "Search projects", exact = True).fill("Research")
        page.get_by_role("option", name = "Research notes", exact = True).click()
        expect(page.get_by_role("heading", name = "Other project", exact = True)).to_have_count(0)
        checks.append(f"{shelf}-searchable-project-filter")
        sort("Compare chats")
        expect(page.get_by_role("button", name = "Compare models", exact = True)).to_have_count(1)
        expect(page.get_by_role("button", name = "Café needle", exact = True)).to_have_count(0)
        sort("Single chats")
        expect(page.get_by_role("button", name = "Compare models", exact = True)).to_have_count(0)
        checks.append(f"{shelf}-type-filter-paired-chats")
        sort("Alphabetical")
        expect(page.locator("section").last.locator("button[title]").first).to_have_text(
            "Café needle"
        )
        sort("Created")
        expect(page.get_by_role("button", name = "Chat 30", exact = True)).to_be_visible()
        sort("Updated")
        expect(page.get_by_role("button", name = "Zulu sample", exact = True)).to_be_visible()
        checks.append(f"{shelf}-sort-created-updated-alphabetical")
        assert page.get_by_text("Date created", exact = True).count() == 0
        title = page.get_by_role("button", name = "Zulu sample", exact = True)
        subtitle = title.locator("..").locator("p")
        assert subtitle.inner_text()
        assert subtitle.bounding_box()["y"] > title.bounding_box()["y"]
        checks.append(f"{shelf}-date-beneath-title")

    search = seed("chats")
    search.fill("Old experiments")
    expect(page.get_by_role("button", name = "Chat 31", exact = True)).to_be_visible()
    checks.append("archive-search-includes-archived-project-names")

    search = seed("manage")
    page.get_by_role("checkbox", name = "Select all visible chats", exact = True).click()
    expect(page.get_by_text("Selected chats: 20", exact = True)).to_be_visible()
    search.fill("needle")
    expect(
        page.get_by_role("checkbox", name = 'Select "Café needle"', exact = True)
    ).not_to_be_checked()
    expect(page.get_by_role("button", name = "Archive", exact = True)).to_have_count(0)
    assert not mutation_requests()
    checks.append("manage-filter-clears-hidden-selection")
    page.get_by_role("checkbox", name = 'Select "Café needle"', exact = True).click()
    page.get_by_role("button", name = "Archive", exact = True).click()
    page.wait_for_function(
        "window.__dataFixture.rows.find(r => r.title === 'Café needle').archived === true"
    )
    assert len(mutation_requests()) == 1
    checks.append("manage-filtered-archive-scope")

    search = seed("chats", holdMutation = True)
    search.fill("Compare models")
    page.get_by_role("button", name = "Delete results", exact = True).click()
    dialog = page.get_by_role("alertdialog")
    expect(dialog.get_by_role("heading")).to_have_text("Delete chat")
    toggle = dialog.get_by_role("switch")
    if toggle.get_attribute("aria-checked") != "true":
        toggle.click()
    dialog.get_by_role("button", name = "Delete", exact = True).click()
    page.wait_for_function("typeof window.__dataFixture.releaseMutation === 'function'")
    expect(toggle).to_be_disabled()
    expect(dialog.get_by_role("button", name = "Cancel", exact = True)).to_be_disabled()
    page.keyboard.press("Escape")
    expect(dialog).to_be_visible()
    page.evaluate("window.__dataFixture.releaseMutation()")
    expect(dialog).to_have_count(0)
    requests = mutation_requests()
    assert len(requests) == 1 and len(requests[0]["body"]["ids"]) == 2
    assert requests[0]["body"]["delete_files"] is True
    assert page.evaluate("window.__dataFixture.rows.length") == 30
    checks.append("archive-filtered-delete-pair-sandbox-choice-pending-lock")

    for kind in ["images", "videos", "audio"]:
        search = seed(kind)
        search.fill("CAFE needle")
        row = page.locator("[data-archived-id]")
        expect(row).to_have_count(1)
        expect(row).to_contain_text("Café needle")
        expect(page.get_by_text("Searching remaining items", exact = False)).to_have_count(0)
        assert (
            page.evaluate(
                "window.__dataFixture.requests.filter(r => r.path.endsWith('/gallery') && r.query.includes('archived=true')).length"
            )
            >= 3
        )
        assert page.get_by_text("Date created", exact = True).count() == 0
        checks.append(f"{kind}-search-full-archive-and-compact-row")
        search.fill("missing query")
        expect(page.get_by_text("No archived items match your search.", exact = True)).to_be_visible()
        search.fill("")
        sort("Alphabetical")
        expect(row.first).to_contain_text("Café needle")
        sort("Oldest first")
        expect(row.first).to_contain_text("Sample 446")
        checks.append(f"{kind}-empty-query-reset-and-sorts")
        search.fill("needle")
        page.get_by_role("button", name = "Delete results", exact = True).click()
        dialog = page.get_by_role("alertdialog")
        expect(dialog.get_by_role("heading")).to_have_text("Delete archived items (1)")
        dialog.get_by_role("button", name = "Cancel", exact = True).click()
        assert not mutation_requests()
        page.get_by_role("button", name = "Delete results", exact = True).click()
        dialog.get_by_role("button", name = "Delete", exact = True).click()
        expect(dialog).to_have_count(0)
        expect(row).to_have_count(0)
        assert len(mutation_requests()) == 1
        assert page.evaluate("kind => window.__dataFixture.media[kind].length", kind) == 446
        checks.append(f"{kind}-filtered-delete-confirm-cancel-scope")

        search = seed(kind, failPage = True)
        search.fill("needle")
        expect(
            page.get_by_text("Search is incomplete. Retry loading the remaining items.", exact = True)
        ).to_be_visible()
        assert page.get_by_text("No archived items match your search.", exact = True).count() == 0
        page.evaluate("window.__dataFixture.failPage = false")
        page.get_by_role("button", name = "Retry", exact = True).click()
        expect(
            page.get_by_role("button", name = "Unarchive: Café needle", exact = True)
        ).to_be_visible()
        checks.append(f"{kind}-failed-page-retry-retains-search")
        page.evaluate("window.__dataFixture.holdMutation = true")
        page.get_by_role("button", name = "Unarchive: Café needle", exact = True).click()
        page.wait_for_function("typeof window.__dataFixture.releaseMutation === 'function'")
        expect(
            page.get_by_role("button", name = "Unarchive: Café needle", exact = True)
        ).to_be_disabled()
        page.evaluate("window.__dataFixture.releaseMutation()")
        expect(page.locator("[data-archived-id]").filter(has_text = "Café needle")).to_have_count(0)
        assert len(mutation_requests()) == 1
        checks.append(f"{kind}-restore-pending-lock")

        search = seed(kind, failMutation = True)
        search.fill("needle")
        page.get_by_role("button", name = "Delete: Café needle", exact = True).click()
        dialog = page.get_by_role("alertdialog")
        dialog.get_by_role("button", name = "Delete", exact = True).click()
        expect(dialog).to_have_count(0)
        expect(page.get_by_role("button", name = "Delete: Café needle", exact = True)).to_be_enabled()
        assert page.evaluate("kind => window.__dataFixture.media[kind].length", kind) == 447
        checks.append(f"{kind}-failed-delete-retains-row")

        search = seed(kind, stallPage = True)
        search.fill("needle")
        expect(page.get_by_role("button", name = "Retry", exact = True)).to_be_visible()
        count = page.evaluate("window.__dataFixture.requests.length")
        page.wait_for_timeout(800)
        assert page.evaluate("window.__dataFixture.requests.length") == count
        checks.append(f"{kind}-stalled-page-stops-retrying")

    for kind in ["images", "videos", "audio"]:
        search = seed(kind, holdPage = True)
        page.get_by_role("button", name = "Show more", exact = True).click()
        page.wait_for_function("typeof window.__dataFixture.releasePage === 'function'")
        page.get_by_role("button", name = "Unarchive: Sample 00", exact = True).click()
        expect(page.locator("[data-archived-id]").filter(has_text = "Sample 00")).to_have_count(0)
        page.evaluate("window.__dataFixture.holdPage = false; window.__dataFixture.releasePage()")
        search.fill("Sample 20")
        expect(page.locator(f'[data-archived-id="{kind}-20"]')).to_contain_text("Sample 20")
        expect(page.get_by_text("Searching remaining items", exact = False)).to_have_count(0)
        checks.append(f"{kind}-restore-during-pagination-keeps-boundary-item")

        seed(kind, mediaCount = 23)
        page.get_by_role("button", name = "Delete all", exact = True).click()
        dialog = page.get_by_role("alertdialog")
        expect(dialog.get_by_role("heading")).to_have_text("Delete archived items (23)")
        assert not mutation_requests()
        dialog.get_by_role("button", name = "Cancel", exact = True).click()
        assert page.evaluate("kind => window.__dataFixture.media[kind].length", kind) == 23
        checks.append(f"{kind}-bulk-confirmation-loads-all-pages-before-delete")

    for kind in ["images", "videos", "audio"]:
        seed(kind, reorderDates = True)
        expect(page.locator("[data-archived-id]")).to_have_count(20)
        sort("Created")
        expect(page.locator("[data-archived-id]").first).to_contain_text("Sample 300")
        expect(page.get_by_text("Searching remaining items", exact = False)).to_have_count(0)
        expect(page.locator("[data-archived-id]")).to_have_count(20)
        thumbnails = page.evaluate(
            "window.__dataFixture.requests.filter(r => r.path.includes('/content')).length"
        )
        assert thumbnails < 60, thumbnails
        checks.append(f"{kind}-global-date-sort-with-bounded-rows-and-thumbnails")

    for width, theme in itertools.product([360, 768, 1280], ["light", "dark"]):
        page.set_viewport_size({"width": width, "height": 1000})
        page.evaluate(
            "theme => document.documentElement.classList.toggle('dark', theme === 'dark')", theme
        )
        search = seed("chats")
        search.fill("needle")
        row = page.get_by_role("button", name = "Café needle", exact = True)
        expect(row).to_be_visible()
        assert row.bounding_box()["width"] > 10
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        expect(
            page.get_by_role("button", name = "Unarchive: Café needle", exact = True)
        ).to_be_visible()
        search.focus()
        # macOS WebKit uses Option-Tab to include buttons in keyboard navigation.
        key = (
            "Alt+Tab"
            if sys.platform == "darwin" and os.environ.get("PW_ENGINE") == "webkit"
            else "Tab"
        )
        page.keyboard.press(key)
        expect(page.get_by_role("button", name = "Filter and sort", exact = True)).to_be_focused()
        page.keyboard.press("Enter")
        expect(page.get_by_role("menuitemradio", name = "Updated", exact = True)).to_be_visible()
        page.keyboard.press("Escape")
        checks.append(f"archive-layout-keyboard-{width}-{theme}")
    page.set_viewport_size({"width": 1280, "height": 1000})
    return checks


def run_library_locales(page):
    checks = []
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.evaluate("document.documentElement.classList.remove('dark')")
    locales = ["en", "es", "fr", "de", "it", "pt-BR", "ru", "zh-CN", "ja", "ko", "hi", "ar"]
    for locale in locales:
        text = page.evaluate(
            """async locale => {
            const api = await import('/src/i18n/index.ts');
            await api.setLocale(locale);
            return {
                ...api.messages[locale].settings.data.library,
                manage: api.translate('settings.data.manageChats'),
                manageAction: api.translate('settings.data.manageAction'),
                cancel: api.translate('common.cancel'),
                delete: api.translate('common.delete'),
                deleteAll: api.translate('settings.data.deleteAllAction'),
                data: api.translate('settings.data.title'),
                back: api.translate('settings.data.backToData'),
                archived: api.translate('settings.data.archivedChats'),
            };
            }""",
            locale,
        )
        if locale == "es":
            assert text["noProject"] == "Sin proyecto"
        for shelf in ["manage", "chats", "images", "videos", "audio"]:
            page.evaluate("window.__settingsSmoke.close()")
            expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
            page.evaluate(
                """shelf => {
                const f = window.__dataFixture;
                Object.assign(f, {failMutation: false, holdMutation: false, failPage: false,
                    stallPage: false, holdPage: false, listFail: false, requests: []});
                f.rows = [null, 'gone', 'research'].map((projectId, index) => ({
                    id: `locale-${index}`, title: `Sample ${index}`, modelType: 'base', modelId: 'test',
                    projectId, createdAt: 1700000000000, updatedAt: 1700000000000,
                    archived: shelf === 'chats',
                }));
                f.media = Object.fromEntries(['images', 'videos', 'audio'].map(kind => [kind, [{
                    id: 'locale-media', prompt: 'Sample media', url: '/unused',
                    created_at: kind === 'images' ? 1700000000 : '2023-11-14T22:13:20Z',
                }]]));
                if (shelf === 'manage') window.__settingsSmoke.open('data');
                else window.__settingsSmoke.openArchived(shelf);
                }""",
                shelf,
            )
            if shelf == "manage":
                page.locator(f'[data-settings-label="{text["manage"]}"]').get_by_role(
                    "button", name = text["manageAction"], exact = True
                ).click()
            placeholder = text[
                {
                    "manage": "searchChats",
                    "chats": "searchArchivedChats",
                    "images": "searchImages",
                    "videos": "searchVideos",
                    "audio": "searchAudio",
                }[shelf]
            ]
            search = page.get_by_role("searchbox", name = placeholder, exact = True)
            expect(search).to_be_visible()
            toolbar = page.get_by_role("button", name = text["filterSort"], exact = True)
            toolbar.click()
            expect(
                page.get_by_role("menuitemradio", name = text["alphabetical"], exact = True)
            ).to_be_visible()
            if shelf in ["manage", "chats"]:
                expect(
                    page.get_by_role("menuitemradio", name = text["singleChats"], exact = True)
                ).to_be_visible()
            page.keyboard.press("Escape")
            if shelf in ["manage", "chats"]:
                expect(page.get_by_role("button", name = "Sample 0", exact = True)).to_be_visible()
                search.fill(text["noProject"])
                expect(page.get_by_role("button", name = "Sample 0", exact = True)).to_be_visible()
                expect(page.get_by_role("button", name = "Sample 1", exact = True)).to_have_count(0)
                search.fill(text["unavailableProject"])
                expect(page.get_by_role("button", name = "Sample 1", exact = True)).to_be_visible()
                search.fill("")
                page.get_by_role("button", name = text["filterProject"], exact = True).click()
                page.get_by_role("combobox", name = text["searchProjects"], exact = True).fill(
                    text["noProject"]
                )
                page.get_by_role("option", name = text["noProject"], exact = True).click()
                expect(
                    page.get_by_role("heading", name = text["noProject"], exact = True)
                ).to_be_visible()
                expect(page.get_by_role("button", name = "Sample 1", exact = True)).to_have_count(0)
                if shelf == "manage":
                    page.get_by_role("checkbox", name = text["selectAll"], exact = True).click()
                    expect(
                        page.get_by_role("button", name = text["archive"], exact = True)
                    ).to_be_visible()
                    page.get_by_role("button", name = text["delete"], exact = True).click()
                    expect(page.get_by_role("alertdialog")).to_contain_text(
                        text["deleteChatsWarning"].replace("{count}", "1")
                    )
                else:
                    page.get_by_role("button", name = text["deleteResults"], exact = True).click()
                    expect(page.get_by_role("alertdialog")).to_contain_text(
                        text["deleteArchivedWarning"].replace("{count}", "1")
                    )
            else:
                row = page.locator("[data-archived-id]")
                expect(row).to_contain_text("Sample media")
                expected_date = page.evaluate(
                    "locale => new Date(1700000000000).toLocaleString(locale, {year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'})",
                    locale,
                )
                expect(row).to_contain_text(expected_date)
                expect(
                    page.get_by_role(
                        "button",
                        name = text["unarchiveItem"].replace("{title}", "Sample media"),
                        exact = True,
                    )
                ).to_be_visible()
                page.get_by_role("button", name = text["deleteAll"], exact = True).click()
                expect(page.get_by_role("alertdialog").get_by_role("heading")).to_have_text(
                    text["deleteItemsTitle"].replace("{count}", "1")
                )
                expect(page.get_by_role("alertdialog")).to_contain_text(text["deleteFilesWarning"])
            page.get_by_role("alertdialog").get_by_role(
                "button", name = text["cancel"], exact = True
            ).click()
            assert not page.evaluate(
                "window.__dataFixture.requests.some(r => r.method === 'DELETE' || r.method === 'PATCH')"
            )
            search.fill("NoMatchingTitle")
            empty_key = (
                "noChats"
                if shelf == "manage"
                else "noArchivedMatches"
                if shelf == "chats"
                else "noMediaMatches"
            )
            expect(page.get_by_text(text[empty_key], exact = True)).to_be_visible()
            if locale == "es" and shelf == "chats":
                page.locator("aside input").fill(text["archived"])
                page.locator("aside").get_by_role(
                    "button", name = text["archived"], exact = True
                ).click()
                expect(page.locator(".settings-search-hit")).to_have_attribute(
                    "data-settings-label", text["archived"]
                )
                expect(page.get_by_role("button", name = text["back"], exact = True)).to_have_count(0)
                checks.append("localized-settings-search-exits-archive")
            checks.append(f"library-locale-{locale}-{shelf}")
    # Change locale with the archive and its query still mounted.
    page.evaluate("async () => {await (await import('/src/i18n/index.ts')).setLocale('es');}")
    search = page.get_by_role("searchbox", name = "Buscar audio archivado", exact = True)
    expect(search).to_have_value("NoMatchingTitle")
    expect(page.get_by_role("button", name = "Filtrar y ordenar", exact = True)).to_be_visible()
    checks.append("library-locale-live-switch-retains-query")
    page.evaluate("async () => {await (await import('/src/i18n/index.ts')).setLocale('en');}")
    return checks


def run_library_selection(page):
    page.evaluate("window.__settingsSmoke.close()")
    expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
    page.evaluate("""() => {
        const f = window.__dataFixture;
        Object.assign(f, {requests: [], failMutation: false, holdMutation: false, listFail: false});
        f.rows = Array.from({length: 32}, (_, i) => ({
            id: `selection-${i}`, title: `Chat ${i}`, modelType: 'base', modelId: 'test',
            createdAt: 1700000000000, updatedAt: 1700000000000 + (32-i)*1000,
        }));
        window.__settingsSmoke.open('data');
    }""")
    page.locator('[data-settings-label="Manage chats"]').get_by_role(
        "button", name = "Manage", exact = True
    ).click()
    target = page.get_by_role("checkbox", name = 'Select "Chat 19"', exact = True)
    target.click()
    expect(page.get_by_text("Selected chats: 1", exact = True)).to_be_visible()
    page.evaluate("""() => {
        window.__dataFixture.rows[20].updatedAt = 1900000000000;
        window.dispatchEvent(new Event('unsloth-chat-history-updated'));
    }""")
    expect(target).to_have_count(0)
    expect(page.get_by_role("button", name = "Chat 20", exact = True)).to_be_visible()
    expect(page.get_by_text("Selected chats: 1", exact = True)).to_be_visible()
    checks = ["selected-chat-survives-live-page-reorder"]
    select_all = page.get_by_role("checkbox", name = "Select all visible chats", exact = True)
    expect(select_all).not_to_be_checked()
    select_all.click()
    expect(page.get_by_text("Selected chats: 21", exact = True)).to_be_visible()
    select_all.click()
    expect(page.get_by_text("Selected chats: 1", exact = True)).to_be_visible()
    checks.append("select-visible-preserves-hidden-selection")
    page.get_by_role("button", name = "Show more (12)", exact = True).click()
    expect(target).to_be_checked()
    checks.append("revealed-chat-keeps-its-selection")
    page.evaluate("""() => {
        const f = window.__dataFixture;
        f.rows.push(...Array.from({length: 40}, (_, i) => ({
            id: `incoming-${i}`, title: `Incoming ${i}`, modelType: 'base', modelId: 'test',
            createdAt: 1700000000000, updatedAt: 1900000000000 + i + 1,
        })));
        window.dispatchEvent(new Event('unsloth-chat-history-updated'));
    }""")
    expect(target).to_have_count(0)
    expect(page.get_by_text("Selected chats: 1", exact = True)).to_be_visible()
    page.get_by_role("button", name = "Delete", exact = True).click()
    expect(page.get_by_role("alertdialog").get_by_role("heading")).to_have_text("Delete chats (1)")
    page.get_by_role("alertdialog").get_by_role("button", name = "Cancel", exact = True).click()
    page.get_by_role("button", name = "Archive", exact = True).click()
    page.wait_for_function(
        "window.__dataFixture.rows.find(r => r.id === 'selection-19').archived === true"
    )
    assert page.evaluate(
        "window.__dataFixture.requests.filter(r => r.method === 'PATCH').map(r => r.path)"
    ) == ["/api/chat/threads/selection-19"]
    checks.append("bulk-action-retains-original-selected-id")
    # Explicit filter changes still clear selections.
    page.get_by_role("checkbox", name = 'Select "Incoming 18"', exact = True).click()
    page.get_by_role("searchbox", name = "Search chats or projects", exact = True).fill("Incoming 18")
    expect(
        page.get_by_role("checkbox", name = 'Select "Incoming 18"', exact = True)
    ).not_to_be_checked()
    expect(page.get_by_role("button", name = "Archive", exact = True)).to_have_count(0)
    checks.append("explicit-filter-change-clears-selection")
    return checks


def run_library_collation(page):
    checks = []
    titles = ["阿", "八", "中", "张", "曾", "Zebra", "苹果", "橙子", "東京", "大阪"]
    for shelf in ["manage", "chats", "images", "videos", "audio"]:
        page.evaluate("window.__settingsSmoke.close()")
        expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
        page.evaluate(
            """async ({shelf, titles}) => {
            await (await import('/src/i18n/index.ts')).setLocale('en');
            const f = window.__dataFixture;
            Object.assign(f, {requests: [], failMutation: false, holdMutation: false, listFail: false, failPage: false, stallPage: false});
            f.rows = titles.map((title, i) => ({id: `collation-${i}`, title, modelType: 'base', modelId: 'test',
                createdAt: 1700000000000, updatedAt: 1700000000000, archived: shelf === 'chats'}));
            f.projects = titles.map((name, i) => ({id: `project-${i}`, name, createdAt: 1, updatedAt: 1}));
            f.media = Object.fromEntries(['images', 'videos', 'audio'].map(kind => [kind, titles.map((prompt, i) => ({
                id: `${kind}-${i}`, prompt, url: '/unused',
                created_at: kind === 'images' ? 1700000000 : '2023-11-14T22:13:20Z',
            }))]));
            if(shelf === 'manage') window.__settingsSmoke.open('data');
            else window.__settingsSmoke.openArchived(shelf);
        }""",
            {"shelf": shelf, "titles": titles},
        )
        if shelf == "manage":
            page.locator('[data-settings-label="Manage chats"]').get_by_role(
                "button", name = "Manage", exact = True
            ).click()
        page.get_by_role("button", name = "Filter and sort", exact = True).click()
        page.get_by_role("menuitemradio", name = "Alphabetical", exact = True).click()
        for locale in ["en", "zh-CN", "ja"]:
            info = page.evaluate(
                """async ({locale, titles}) => {
                const api = await import('/src/i18n/index.ts');
                await api.setLocale(locale);
                return {expected: [...titles].sort(new Intl.Collator(locale).compare),
                    projectFilter: api.translate('settings.data.library.filterProject')};
            }""",
                {"locale": locale, "titles": titles},
            )
            if shelf in ["manage", "chats"]:
                rows = page.locator("main section .divide-y button[title]:not([aria-label])")
            else:
                rows = page.locator("[data-archived-id] p[title]")
            expect(rows).to_have_text(info["expected"])
            checks.append(f"{shelf}-alphabetical-{locale}")
            if shelf in ["manage", "chats"]:
                page.get_by_role("button", name = info["projectFilter"], exact = True).click()
                assert page.get_by_role("option").all_text_contents()[2:] == info["expected"]
                page.keyboard.press("Escape")
                checks.append(f"{shelf}-projects-{locale}")
    page.evaluate("async () => {await (await import('/src/i18n/index.ts')).setLocale('en');}")
    return checks


def run_restore_notifications(page):
    checks = []
    page.evaluate("""() => {
        window.addEventListener('unsloth:gallery-changed', event => {
            window.__dataFixture.notifications.push(event.detail.kind);
        });
    }""")
    for kind in ["images", "videos", "audio"]:
        for count, action, failure_index in [
            (23, "restore", None),
            (23, "restore", 2),
            (23, "restore", 0),
            (1, "restore", None),
            (3, "delete", None),
        ]:
            page.evaluate("window.__settingsSmoke.close()")
            expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
            page.evaluate(
                """({kind, count, failureIndex}) => {
                    const f = window.__dataFixture;
                    Object.assign(f, {requests: [], notifications: [], failMutation: false,
                        holdMutation: false, failPage: false, stallPage: false,
                        failMutationId: failureIndex === null ? null : `${kind}-${failureIndex}`});
                    f.media[kind] = Array.from({length: count}, (_, i) => ({
                        id: `${kind}-${i}`, prompt: `Restore ${i}`, url: '/unused',
                        created_at: kind === 'images' ? 1700000000 : '2023-11-14T22:13:20Z',
                    }));
                    window.__settingsSmoke.openArchived(kind);
                }""",
                {"kind": kind, "count": count, "failureIndex": failure_index},
            )
            trigger = "Unarchive all" if action == "restore" else "Delete all"
            page.get_by_role("button", name = trigger, exact = True).click()
            dialog = page.get_by_role("alertdialog")
            heading = "Unarchive items" if action == "restore" else "Delete archived items"
            expect(dialog.get_by_role("heading")).to_have_text(f"{heading} ({count})")
            confirm = "Unarchive" if action == "restore" else "Delete"
            dialog.get_by_role("button", name = confirm, exact = True).click()
            expect(dialog).to_have_count(0)
            completed = count if failure_index is None else failure_index
            expected_events = [kind] if completed and action == "restore" else []
            observed = page.evaluate("window.__dataFixture.notifications")
            assert observed == expected_events, {
                "kind": kind,
                "completed": completed,
                "events": observed,
            }
            assert (
                page.evaluate("kind => window.__dataFixture.media[kind].length", kind)
                == count - completed
            )
            mutations = page.evaluate(
                "window.__dataFixture.requests.filter(r => ['PATCH', 'DELETE'].includes(r.method)).length"
            )
            assert mutations == completed + (failure_index is not None)
            checks.append(f"{kind}-{action}-{count}-notification-failure-{failure_index}")
    return checks


def run_thumbnail_retention(page):
    checks = []
    for kind in ["images", "videos"]:
        page.evaluate("window.__settingsSmoke.close()")
        expect(page.get_by_role("dialog", include_hidden = True)).to_have_count(0)
        page.evaluate(
            """kind => {
                const f = window.__dataFixture;
                Object.assign(f, {requests: [], thumbnailRequests: 0, failPage: false, stallPage: false});
                const originalFetch = window.fetch;
                f.restoreFetch = () => { window.fetch = originalFetch; };
                window.fetch = async (input, init) => {
                    const url = new URL(typeof input === 'string' ? input : input.url, location.origin);
                    if (url.pathname.endsWith('/content')) {
                        f.thumbnailRequests += 1;
                        const png = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAIAAAADnC86AAAAMklEQVR4nO3NAQ0AMAgAoPtMhrOGjY3h5qAAkV1vw19ZxWKxWCwWi8VisVgsFovFR+MBbzgBjTOVo70AAAAASUVORK5CYII='), c => c.charCodeAt(0));
                        const bytes = new Uint8Array(6 * 1024 * 1024);
                        bytes.set(png);
                        return new Response(bytes, {headers: {'Content-Type': 'image/png'}});
                    }
                    return originalFetch(input, init);
                };
                f.media[kind] = Array.from({length: 6}, (_, i) => ({
                    id: `thumbnail-${i}`, prompt: `Sample ${5-i}`,
                    url: `/api/inference/images/gallery/thumbnail-${i}/content`,
                    created_at: kind === 'images' ? 1700000000 : '2023-11-14T22:13:20Z',
                }));
                window.__settingsSmoke.openArchived(kind);
            }""",
            kind,
        )
        try:
            page.wait_for_function("""() => {
                const images = [...document.querySelectorAll('[data-archived-id] img')];
                return images.length === 6 && images.every(img => img.complete && img.naturalWidth > 0);
            }""")
            assert page.evaluate("window.__dataFixture.thumbnailRequests") == 6, page.evaluate(
                "window.__dataFixture.thumbnailRequests"
            )
            page.get_by_role("button", name = "Filter and sort", exact = True).click()
            page.get_by_role("menuitemradio", name = "Alphabetical", exact = True).click()
            expect(page.locator("[data-archived-id] p[title]").first).to_have_text("Sample 0")
            page.wait_for_timeout(200)
            assert page.evaluate("window.__dataFixture.thumbnailRequests") == 6, page.evaluate(
                "window.__dataFixture.thumbnailRequests"
            )
            checks.append(f"{kind}-sort-retains-visible-thumbnails-over-budget")
            page.get_by_role("searchbox").fill("Sample")
            expect(page.locator("[data-archived-id]")).to_have_count(6)
            page.wait_for_timeout(200)
            assert page.evaluate("window.__dataFixture.thumbnailRequests") == 6, page.evaluate(
                "window.__dataFixture.thumbnailRequests"
            )
            checks.append(f"{kind}-search-retains-visible-thumbnails-over-budget")
        finally:
            page.evaluate("window.__dataFixture.restoreFetch()")
    return checks


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
            if os.environ.get("PW_EXECUTABLE"):
                options["executable_path"] = os.environ["PW_EXECUTABLE"]
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
