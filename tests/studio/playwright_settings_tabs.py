# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings dialog behaviour harness: every tab renders, search jumps land, deep-open works.

Drives smoke-settings.html (the real SettingsDialog and store, no backend), so a static-import
tree and a React.lazy one are directly comparable. Emits a JSON report to $PW_OUT for a
field-by-field diff.

    PW_ENGINE=chromium PW_PORT=5399 PW_OUT=out.json python tests/studio/playwright_settings_tabs.py

PW_CHUNK_DELAY_MS delays every tab module response, widening the window a lazy panel is
in-flight for; PW_CHUNK_FAIL=<tab> aborts one tab's module outright.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    click_forced,
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
    "remote-lan",
    "agents",
    "keyboard-shortcuts",
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
LAN_URLS = ("http://192.168.1.24:8888", "http://10.0.0.7:8888")
LAN_STATUS = {
    "state": "online",
    "urls": list(LAN_URLS),
    "public_urls": [],
    "error": None,
    "auto_start": False,
    "managed_by": "settings",
    "can_start": False,
    "can_stop": True,
    "block_reason": None,
    "bind_host": "0.0.0.0",
    "wildcard_bind": True,
    "serves_web_ui": True,
    "keyless_lan_eligible": True,
    "keyless_scope": "off",
    "keyless_tools": False,
}
# The panel scroller inside the dialog, the element `mainScrollRef` points at.
PANEL = 'div[role="dialog"] main div.hover-scrollbar'
# How long the Data module is held, and how far into that hold the deep-open is abandoned.
DEEP_OPEN_HOLD_MS = 2500
DEEP_OPEN_ABANDON_MS = 300
# The Data panel's own module, however the server spells it (vite appends ?t=, ?v=).
DATA_MODULE = re.compile(r"/data-tab(\.tsx)?(\?|$)")

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


def settle_panel(page, timeout_s: float = SETTLE_TIMEOUT_S) -> dict:
    """Settle, but do not accept a placeholder as the answer.

    The panel renders from a deferred value, so the outgoing content stays on screen until
    the incoming one is ready, and under load that hand-off can hold still past SETTLE_MS.
    """
    deadline = time.time() + timeout_s
    latest = settle(page, timeout_s = timeout_s)
    while latest.get("elements", 0) < 5 and time.time() < deadline:
        page.wait_for_timeout(100)
        latest = settle(page, timeout_s = max(1.0, deadline - time.time()))
    return latest


def click_tab_and_observe(page, tab: str) -> dict:
    """Click a tab; record how long the panel keeps the old content and whether it blanks."""
    before = snapshot(page)
    click_forced(page.locator(f'[data-testid="settings-tab-{tab}"]'), timeout = 15000)
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
    final = settle_panel(page)
    return {
        "before_sig": before.get("sig"),
        "changed_ms": changed_ms,
        "blank_frames": blank_frames,
        "frames_sampled": frames,
        "settled": final,
    }


def require_harness(page) -> None:
    """Fail with the cause rather than a selector timeout if the page has moved on.

    Vite dev proxies /api to 127.0.0.1:8888. With a real Unsloth listening there and no
    token, those calls come back 401 and the app's auth handling navigates, which takes
    the harness's window with it. Every later step then times out on a dialog that cannot
    exist. This is not the dialog's doing: it happens on main too, where the harness is
    unmounted before the first open. So say so.
    """
    if not page.evaluate("() => !!window.__settingsSmoke"):
        raise RuntimeError(
            "the harness page is gone (window.__settingsSmoke undefined). This smoke page "
            f"is served by vite on {PORT} and proxies /api to 127.0.0.1:8888; an Unsloth "
            "listening there answers 401 and the app navigates away. Run this without a "
            "backend on 8888."
        )


def open_dialog(page, tab: str | None = None) -> None:
    require_harness(page)
    page.evaluate("(t) => window.__settingsSmoke.open(t || undefined)", tab)
    page.wait_for_selector('div[role="dialog"]', timeout = 15000)


# Only subpages put a back button in the panel header, so this is locale-independent.
ON_SUBPAGE_JS = """() => ({
    subpage: !!document.querySelector('div[role="dialog"] main header button'),
    elements: (document.querySelector('div[role="dialog"] main div.hover-scrollbar')
        || { querySelectorAll: () => [] }).querySelectorAll('*').length,
})"""

# Deep-open the archived chats, then walk away partway through the hold.
ABANDON_DEEP_OPEN_JS = """(delay) => {
    window.__abandonedAt = null;
    window.__settingsSmoke.openArchived('chats');
    setTimeout(() => {
        const panel = document.querySelector('div[role="dialog"] main div.hover-scrollbar');
        window.__abandonedAt = {
            elements: panel ? panel.querySelectorAll('*').length : null,
            subpage: !!document.querySelector('div[role="dialog"] main header button'),
        };
        window.__settingsSmoke.close();
    }, delay);
}"""


def run_abandoned_deep_open(page) -> None:
    """A deep-open the panel never mounted for must not outlive the navigation.

    `openArchivedChats` sets `archivedRequested`, and DataTab is the only thing that clears
    it. Now that the panel is fetched on first view, closing the dialog while that fetch is
    in flight leaves the request set with nothing to consume it, and the next ordinary visit
    to Data opens an archive listing nobody asked for. Runs before anything else opens the
    dialog, so the Data module is still cold and the hold is what decides when it arrives.
    """

    # below past its timeout even though nothing was wrong with the dialog.
    # Matched on the Data module alone rather than on everything.
    def hold_data(route):
        time.sleep(DEEP_OPEN_HOLD_MS / 1000)
        return route.continue_()

    page.route(DATA_MODULE, hold_data)
    try:
        page.evaluate(ABANDON_DEEP_OPEN_JS, DEEP_OPEN_ABANDON_MS)
        page.wait_for_timeout(DEEP_OPEN_HOLD_MS + 1500)
    finally:
        page.unroute(DATA_MODULE, hold_data)
    abandoned = page.evaluate("() => window.__abandonedAt")
    open_dialog(page, "data")
    settled = settle_panel(page)
    landed = page.evaluate(ON_SUBPAGE_JS)
    report["abandoned_deep_open"] = {
        "abandoned_at": abandoned,
        "landed": landed,
        "state": page.evaluate("() => window.__settingsSmoke.state()"),
        "settled_sig": settled.get("sig"),
    }
    log(f"abandoned deep-open: at close {abandoned}, later visit {landed}")
    if not abandoned or abandoned.get("subpage"):
        fail(f"deep-open abandon never happened mid-load, so nothing was tested ({abandoned})")
    elif landed["elements"] < 5:
        fail(f"after an abandoned deep-open, Data did not render at all ({landed})")
    elif landed["subpage"]:
        fail("an abandoned archive deep-open reopened the archive on the next visit to Data")
    else:
        log("an abandoned deep-open leaves the next visit to Data on the main page")
    page.evaluate("() => window.__settingsSmoke.close()")
    page.wait_for_timeout(200)

    # Dropping the abandoned ones must not drop the honoured ones.
    page.evaluate("() => window.__settingsSmoke.openArchived('chats')")
    page.wait_for_selector('div[role="dialog"]', timeout = 15000)
    settle_panel(page)
    honoured = page.evaluate(ON_SUBPAGE_JS)
    report["abandoned_deep_open"]["honoured"] = honoured
    if not honoured["subpage"]:
        fail(f"a deep-open the panel did reach no longer opens the archive ({honoured})")
    else:
        log("a deep-open the panel reaches still opens the archive")
    page.evaluate("() => window.__settingsSmoke.close()")
    page.wait_for_timeout(200)
    report["steps"].append("abandoned-deep-open")


def run_chunk_fail(page) -> None:
    """One panel's module is blocked. The dialog must survive it, and so must the app."""
    open_dialog(page)
    settle_panel(page)
    click_forced(page.locator('[data-testid="settings-tab-general"]'), timeout = 15000)
    settle_panel(page)
    # Read the nav size instead of hardcoding it. The invariant is that blocking a
    # error-handling regression and was a stale constant.
    nav_before = page.evaluate(
        "() => document.querySelectorAll('[data-testid^=\"settings-tab-\"]').length"
    )
    if nav_before < 2:
        fail(f"the settings nav was already empty before blocking anything ({nav_before})")
    click_forced(page.locator(f'[data-testid="settings-tab-{CHUNK_FAIL}"]'), timeout = 15000)
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
    # The idle prefetch pulls every panel, so a blocked one must not surface as a rejection.
    errors = page.evaluate("() => window.__settingsSmoke.errors()")
    report["chunk_fail_window_errors"] = errors
    unhandled = [
        e for e in errors if "dynamically imported module" in e or "Importing a module" in e
    ]
    if unhandled:
        fail(f"blocking the {CHUNK_FAIL} panel left an unhandled rejection: {unhandled}")
    else:
        log("no unhandled rejection from the idle prefetch")
    if state["boundary"]:
        fail(
            f"blocking the {CHUNK_FAIL} panel unmounted the app: the throw reached the "
            "harness root boundary, and the real app has none"
        )
    if not state["dialog"] or state["nav"] != nav_before:
        fail(
            f"blocking the {CHUNK_FAIL} panel took the dialog down "
            f"(nav was {nav_before} before, {state})"
        )
    else:
        log("the dialog and its twelve nav entries survived")
    # Another tab must still work.
    click_forced(page.locator('[data-testid="settings-tab-about"]'), timeout = 15000)
    after = settle_panel(page)
    report["chunk_fail_recovery"] = after
    if not after.get("present") or after.get("elements", 0) < 5:
        fail(f"after a failed panel, another tab no longer renders ({after})")
    else:
        log("another tab still renders after the failure")


def run_lan_address_actions(page) -> None:
    """Every listed address must own the actions that operate on it."""
    click_forced(page.locator('[data-testid="settings-tab-remote-lan"]'), timeout = 15000)
    settle_panel(page)

    rows = {}
    for url in LAN_URLS:
        address = page.get_by_text(url, exact = True)
        if address.count() != 1:
            fail(f"LAN address {url}: expected one visible row, found {address.count()}")
            continue
        labels = address.evaluate(
            """(node) => [...node.parentElement.querySelectorAll('button')]
                .map((button) => button.getAttribute('aria-label'))"""
        )
        rows[url] = labels
        expected = [f"Show QR code for {url}", f"Copy {url}"]
        if labels != expected:
            fail(f"LAN address {url}: row actions {labels}, expected {expected}")

    if len(rows) != len(LAN_URLS):
        return

    target = LAN_URLS[1]
    click_forced(page.get_by_role("button", name = f"Show QR code for {target}", exact = True))
    qr_dialog = page.locator('div[role="dialog"]').last
    qr_dialog.wait_for(state = "visible", timeout = 15000)
    qr_value = qr_dialog.locator("code").inner_text().strip()
    if qr_value != target:
        fail(f"LAN QR action opened {qr_value}, expected {target}")
    page.keyboard.press("Escape")
    page.get_by_role("heading", name = "Open on your phone").wait_for(state = "hidden", timeout = 15000)

    page.evaluate(
        """() => Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: { writeText: async (text) => { window.__copiedLanUrl = text; } },
        })"""
    )
    click_forced(page.get_by_role("button", name = f"Copy {target}", exact = True))
    page.wait_for_function("(url) => window.__copiedLanUrl === url", arg = target, timeout = 15000)
    report["lan_address_actions"] = {"rows": rows, "qr": qr_value, "copied": target}
    report["steps"].append("lan-address-actions")
    log("each LAN address keeps its own QR and copy target")


def run(page) -> None:
    if CHUNK_FAIL:
        run_chunk_fail(page)
        return
    run_abandoned_deep_open(page)

    open_dialog(page)
    settle_panel(page)
    click_forced(page.locator('[data-testid="settings-tab-about"]'), timeout = 15000)
    settle_panel(page)
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

    run_lan_address_actions(page)

    page.evaluate("() => window.__settingsSmoke.close()")
    page.wait_for_timeout(300)
    for tab in ("voice", "api-keys", "data", "about", "connections"):
        open_dialog(page, tab)
        state = page.evaluate("() => window.__settingsSmoke.state()")
        snap = settle_panel(page)
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

    # A real setting well down a long panel, so a jump that never happens shows in scrollTop.
    # search, then jump to a result and confirm the scroll target flashed ------ A real setting well down a long panel
    target_tab = "general"
    target_label = report["tabs"][target_tab]["settled"]["labels"][-1]
    query = target_label.split()[0]
    open_dialog(page, "about")
    settle_panel(page)
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
        click_forced(entries.nth(index))
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
        report["search"]["settled"] = settle_panel(page)
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


def write_report() -> None:
    OUT.parent.mkdir(parents = True, exist_ok = True)
    OUT.write_text(json.dumps(report, indent = 2), encoding = "utf-8")
    log(f"report -> {OUT}")


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
            page.route(
                "**/api/settings/lan-access*",
                lambda route: route.fulfill(
                    status = 200,
                    content_type = "application/json",
                    body = json.dumps(LAN_STATUS),
                ),
            )
            if CHUNK_DELAY_MS or CHUNK_FAIL:

                def handle(route):
                    path = route.request.url
                    if "/tabs/" not in path:
                        return route.fallback()
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
            # Vite dev re-optimizes deps on first sight and full-reloads;
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
    except Exception as exc:
        # A cold-start dev server can fail to serve a module before anything is under test.
        report["aborted"] = f"{type(exc).__name__}: {exc}"
        fail(f"harness aborted: {report['aborted'].splitlines()[0]}")
        write_report()
        raise
    finally:
        stop_process(vite)

    write_report()
    if report["failures"]:
        log(f"{len(report['failures'])} FAILURES")
        for f in report["failures"]:
            log(f"  - {f}")
        return 1
    log("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
