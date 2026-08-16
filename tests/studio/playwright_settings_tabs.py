# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings dialog behaviour harness: every tab renders, search jumps land, deep-open works.

Drives smoke-settings.html (the real SettingsDialog against the real store, no backend) so the
result is comparable between a tree where the tab panels are static imports and one where they
are behind React.lazy + Suspense.

Emits a JSON report to $PW_OUT so two trees can be diffed field by field.

    PW_ENGINE=chromium PW_PORT=5399 PW_OUT=out.json python tests/studio/playwright_settings_tabs.py

PW_CHUNK_DELAY_MS delays every tab module response, to widen the window a lazy panel is
in-flight for; PW_CHUNK_FAIL=<tab> aborts one tab's module outright.
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
)

TABS = [
    "general",
    "profile",
    "appearance",
    "resources",
    "chat",
    "voice",
    "connections",
    "data",
    "api-keys",
    "agents",
    "debugging",
    "about",
]

ENGINE = os.environ.get("PW_ENGINE", "chromium")
PORT = int(os.environ.get("PW_PORT", "5399"))
OUT = Path(os.environ.get("PW_OUT", "logs/settings_tabs_report.json"))
CHUNK_DELAY_MS = int(os.environ.get("PW_CHUNK_DELAY_MS", "0"))
CHUNK_FAIL = os.environ.get("PW_CHUNK_FAIL", "")
SETTLE_MS = 600
SETTLE_TIMEOUT_S = 25.0
# Panel container: the scroller `mainScrollRef` points at, inside the dialog.
PANEL = 'div[role="dialog"] main div.hover-scrollbar'

report: dict = {
    "engine": ENGINE,
    "chunk_delay_ms": CHUNK_DELAY_MS,
    "chunk_fail": CHUNK_FAIL,
    "steps": [],
    "tabs": {},
    "failures": [],
}


def log(msg: str) -> None:
    print(f"[settings-tabs] {msg}", flush = True)


def fail(msg: str) -> None:
    log(f"FAIL {msg}")
    report["failures"].append(msg)


SNAPSHOT_JS = """(sel) => {
    const root = document.querySelector(sel);
    if (!root) return { present: false, sig: 'none' };
    const labels = [...root.querySelectorAll('[data-settings-label]')]
        .map((el) => el.dataset.settingsLabel);
    const text = (root.innerText || '').trim();
    return {
        present: true,
        labels,
        elements: root.querySelectorAll('*').length,
        textLength: text.length,
        // Identity of what is on screen, insensitive to live values (versions, sizes).
        sig: labels.join('|') + '#' + root.querySelectorAll('*').length,
    };
}"""


def snapshot(page) -> dict:
    return page.evaluate(SNAPSHOT_JS, PANEL)


def settle(page, timeout_s: float = SETTLE_TIMEOUT_S) -> dict:
    """Poll until the panel signature holds still for SETTLE_MS, then return it."""
    deadline = time.time() + timeout_s
    last = snapshot(page)
    stable_since = time.time()
    while time.time() < deadline:
        page.wait_for_timeout(50)
        now = snapshot(page)
        if now.get("sig") != last.get("sig"):
            last = now
            stable_since = time.time()
            continue
        if (time.time() - stable_since) * 1000 >= SETTLE_MS:
            return now
    last["settle_timeout"] = True
    return last


def click_tab_and_observe(page, tab: str) -> dict:
    """Click a tab; record how long the panel keeps the old content and whether it blanks."""
    before = snapshot(page)
    page.locator(f'[data-testid="settings-tab-{tab}"]').click(force = True, timeout = 15000)
    started = time.time()
    changed_ms = None
    blank_frames = 0
    frames = 0
    deadline = started + SETTLE_TIMEOUT_S
    while time.time() < deadline:
        now = snapshot(page)
        frames += 1
        if now.get("present") and now.get("elements", 0) < 5:
            blank_frames += 1
        if changed_ms is None and now.get("sig") != before.get("sig"):
            changed_ms = round((time.time() - started) * 1000)
            break
        page.wait_for_timeout(20)
    final = settle(page)
    return {
        "before_sig": before.get("sig"),
        "changed_ms": changed_ms,
        "blank_frames": blank_frames,
        "frames_sampled": frames,
        "settled": final,
    }


def open_dialog(page, tab: str | None = None) -> None:
    page.evaluate("(t) => window.__settingsSmoke.open(t || undefined)", tab)
    page.wait_for_selector('div[role="dialog"]', timeout = 15000)


def run_chunk_fail(page) -> None:
    """One panel's module is blocked. The dialog must survive it, and so must the app."""
    open_dialog(page)
    settle(page)
    page.locator('[data-testid="settings-tab-general"]').click(force = True, timeout = 15000)
    settle(page)
    page.locator(f'[data-testid="settings-tab-{CHUNK_FAIL}"]').click(force = True, timeout = 15000)
    page.wait_for_timeout(3000)
    state = page.evaluate(
        """() => ({
            dialog: !!document.querySelector('div[role="dialog"]'),
            nav: document.querySelectorAll('[data-testid^="settings-tab-"]').length,
            harness: !!document.querySelector('[data-testid="harness-root"]'),
            boundary: !!document.querySelector('[data-testid="harness-error-boundary"]'),
            panelText: (document.querySelector('div[role="dialog"] main div.hover-scrollbar')
                || {}).innerText || null,
            bodyText: (document.body.innerText || '').trim().length,
        })"""
    )
    report["chunk_fail_state"] = state
    log(f"after blocking {CHUNK_FAIL}: {state}")
    # The idle prefetch pulls every panel, so a blocked module must not surface as an
    # unhandled rejection for a tab the user never opened.
    errors = page.evaluate("() => window.__settingsSmoke.errors()")
    report["chunk_fail_window_errors"] = errors
    unhandled = [e for e in errors if "dynamically imported module" in e or "Importing a module" in e]
    if unhandled:
        fail(f"blocking the {CHUNK_FAIL} panel left an unhandled rejection: {unhandled}")
    else:
        log("no unhandled rejection from the idle prefetch")
    if state["boundary"]:
        fail(
            f"blocking the {CHUNK_FAIL} panel unmounted the app: the throw reached the "
            "harness root boundary, and the real app has none"
        )
    if not state["dialog"] or state["nav"] != 12:
        fail(f"blocking the {CHUNK_FAIL} panel took the dialog down ({state})")
    else:
        log("the dialog and its twelve nav entries survived")
    # Another tab must still work.
    page.locator('[data-testid="settings-tab-about"]').click(force = True, timeout = 15000)
    after = settle(page)
    report["chunk_fail_recovery"] = after
    if not after.get("present") or after.get("elements", 0) < 5:
        fail(f"after a failed panel, another tab no longer renders ({after})")
    else:
        log("another tab still renders after the failure")


def run(page) -> None:
    if CHUNK_FAIL:
        run_chunk_fail(page)
        return
    # --- 1. every tab renders when selected -----------------------------------------
    open_dialog(page)
    settle(page)
    # Start from a tab that is not the persisted one, so the first iteration is a real switch.
    page.locator('[data-testid="settings-tab-about"]').click(force = True, timeout = 15000)
    settle(page)
    for tab in TABS:
        obs = click_tab_and_observe(page, tab)
        report["tabs"][tab] = obs
        snap = obs["settled"]
        if not snap.get("present"):
            fail(f"tab {tab}: panel container missing")
        elif snap.get("elements", 0) < 5:
            fail(f"tab {tab}: panel settled empty ({snap})")
        elif snap.get("sig") == obs["before_sig"]:
            fail(f"tab {tab}: panel never changed from the previous tab ({snap['sig']})")
        else:
            log(
                f"tab {tab}: changed after {obs['changed_ms']}ms, "
                f"{snap['elements']} elements, {len(snap['labels'])} labels, "
                f"blank frames {obs['blank_frames']}/{obs['frames_sampled']}"
            )
    report["steps"].append("all-tabs-render")

    # --- 2. close/reopen, and deep-open straight to a tab ----------------------------
    page.evaluate("() => window.__settingsSmoke.close()")
    page.wait_for_timeout(300)
    for tab in ("voice", "api-keys", "data", "about", "connections"):
        open_dialog(page, tab)
        state = page.evaluate("() => window.__settingsSmoke.state()")
        snap = settle(page)
        report.setdefault("deep_open", {})[tab] = {"state": state, "settled": snap}
        if state["activeTab"] != tab:
            fail(f"deep-open {tab}: store activeTab is {state['activeTab']}")
        if not snap.get("present") or snap.get("elements", 0) < 5:
            fail(f"deep-open {tab}: panel did not render ({snap})")
        elif snap["sig"] != report["tabs"][tab]["settled"]["sig"]:
            fail(
                f"deep-open {tab}: panel differs from the same tab reached by clicking "
                f"({snap['sig']} vs {report['tabs'][tab]['settled']['sig']})"
            )
        else:
            log(f"deep-open {tab}: matches the clicked-to panel")
        page.evaluate("() => window.__settingsSmoke.close()")
        page.wait_for_timeout(200)
    report["steps"].append("deep-open")

    # --- 3. search, then jump to a result and confirm the scroll target flashed ------
    # Pick a real setting that lives well down a long panel, so a jump that does not
    # happen is visible in scrollTop as well as in the flash.
    target_tab = "general"
    target_label = report["tabs"][target_tab]["settled"]["labels"][-1]
    query = target_label.split()[0]
    open_dialog(page, "about")
    settle(page)
    search = page.locator('div[role="dialog"] aside input').first
    search.fill(query)
    page.wait_for_timeout(400)
    entries = page.locator('div[role="dialog"] aside button:not([data-testid])')
    texts = [entries.nth(i).inner_text().strip() for i in range(entries.count())]
    report["search"] = {"query": query, "target_label": target_label, "result_texts": texts}
    log(f"search '{query}' -> {texts}")
    index = next((i for i, txt in enumerate(texts) if txt == target_label), None)
    if index is None:
        fail(f"search '{query}': '{target_label}' not among results {texts}")
    else:
        entries.nth(index).click(force = True)
        flashed = None
        deadline = time.time() + 15
        while time.time() < deadline:
            state = page.evaluate(
                """(sel) => {
                    const hit = document.querySelector('.settings-search-hit');
                    const root = document.querySelector(sel);
                    return {
                        hit: hit ? (hit.dataset.settingsLabel || hit.tagName) : null,
                        scrollTop: root ? root.scrollTop : null,
                    };
                }""",
                PANEL,
            )
            if state["hit"]:
                flashed = state
                break
            page.wait_for_timeout(30)
        report["search"]["flashed"] = flashed
        report["search"]["settled"] = settle(page)
        if not flashed:
            fail(
                f"search jump to '{target_label}': never flashed "
                f"(panel {report['search']['settled'].get('sig')})"
            )
        elif flashed["hit"] != target_label:
            fail(f"search jump: flashed '{flashed['hit']}', expected '{target_label}'")
        else:
            log(f"search jump to '{target_label}': flashed, scrollTop {flashed['scrollTop']}")
    report["steps"].append("search-jump")

    report["page_errors"] = page.evaluate("() => window.__settingsSmoke.errors()")


def main() -> int:
    vite = start_vite(PORT)
    try:
        url = f"http://127.0.0.1:{PORT}/smoke-settings.html"
        with sync_playwright() as pw:
            launcher = getattr(pw, ENGINE)
            kwargs: dict = {"headless": True}
            if ENGINE == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = launcher.launch(**kwargs)
            ctx = browser.new_context(
                viewport = {"width": 1440, "height": 900},
                reduced_motion = "reduce",
            )
            page = ctx.new_page()
            if CHUNK_DELAY_MS or CHUNK_FAIL:

                def handle(route):
                    path = route.request.url
                    if "/tabs/" not in path:
                        return route.continue_()
                    if CHUNK_FAIL and f"/{CHUNK_FAIL}-tab" in path:
                        return route.abort("failed")
                    if CHUNK_DELAY_MS:
                        time.sleep(CHUNK_DELAY_MS / 1000)
                    return route.continue_()

                page.route("**/*", handle)
            console: list[str] = []
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            for attempt in range(40):
                try:
                    page.goto(url, wait_until = "domcontentloaded", timeout = 30000)
                    break
                except Exception:
                    if attempt == 39:
                        raise
                    time.sleep(2)
            page.wait_for_function("() => !!window.__settingsSmoke", timeout = 120000)
            # Vite dev re-optimizes deps on first sight and full-reloads; let that settle, then
            # reload once so every module is served from a stable dep graph.
            page.wait_for_timeout(4000)
            page.reload(wait_until = "domcontentloaded")
            page.wait_for_function("() => !!window.__settingsSmoke", timeout = 120000)
            page.wait_for_timeout(1500)
            try:
                run(page)
            finally:
                report["console"] = [
                    c
                    for c in console
                    if ("error" in c.lower() or "pageerror" in c.lower())
                    and "502" not in c
                    and "403" not in c
                ][:40]
                boundary = page.locator('[data-testid="harness-error-boundary"]')
                report["error_boundary"] = boundary.inner_text() if boundary.count() else None
                if report["error_boundary"] and not CHUNK_FAIL:
                    fail(f"error boundary tripped: {report['error_boundary']}")
            browser.close()
    finally:
        stop_process(vite)

    OUT.parent.mkdir(parents = True, exist_ok = True)
    OUT.write_text(json.dumps(report, indent = 2))
    log(f"report -> {OUT}")
    if report["failures"]:
        log(f"{len(report['failures'])} FAILURES")
        for f in report["failures"]:
            log(f"  - {f}")
        return 1
    log("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
