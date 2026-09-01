# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Comprehensive Unsloth chat UI test, run locally + in CI."""

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

# Tests run as plain `python tests/studio/playwright_chat_ui.py` (not via pytest/import), so prepend this dir to
# sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    evaluate_fetch,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_console_error,
    is_benign_page_error,
    recover_or_replace_page as _robust_recover_or_replace_page,
    robust_evaluate,
    wait_for_health,
    click_forced,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ["STUDIO_NEW_PW"]
NEW2 = os.environ.get("STUDIO_NEW2_PW", NEW + "X9!")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)

# When on (default in CI), fail loudly on any missing button/nav/dialog instead of logging a WARN;
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"

# Per-turn assistant-bubble wait.
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))
# cannot close the gap, and paid once per run.
# How long the rapid-submit step holds the first turn's response.
RAPID_FIRST_TURN_HOLD_S = 3.0

# Wall-clock cap for the whole script (healthy run is 5-9 min).
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))

PERMISSION_ONLY = os.environ.get("STUDIO_UI_PERMISSION_ONLY", "0") == "1"

# Default stays Chromium for CI.
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None

# Render like the 4 vCPU boxes users and Kaggle sessions actually run on.
CPU_THROTTLE = float(os.environ.get("STUDIO_UI_CPU_THROTTLE", "0") or 0)

# Per-fetch budget; /api/inference/load is the slowest (cold-cache GGUF load).
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
LOAD_FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_LOAD_TIMEOUT_MS", "180000"))

_n = [0]


def step(s):
    print(f"[ui] STEP {s}", flush = True)


def info(s):
    print(f"[ui] {s}", flush = True)


def fail(m):
    raise AssertionError(f"[ui] FAIL: {m}")


def apply_cpu_throttle(ctx, page):
    """Throttle this page, if the option is set. No-op otherwise."""
    if CPU_THROTTLE <= 1:
        return page
    ctx.new_cdp_session(page).send("Emulation.setCPUThrottlingRate", {"rate": CPU_THROTTLE})
    info(f"CPU throttled {CPU_THROTTLE}x")
    return page


def new_throttled_page(ctx):
    """Every page this driver opens, with the settings common to all of them.

    The throttle is scoped to the page TARGET, so a page opened directly runs
    at full speed and the steps after it pass under exactly the conditions the
    throttle exists to reproduce. The 60s default rides along for the reason it
    always did: macos-14 renders, webfonts and lazy routes crowd 30s.
    """
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    apply_cpu_throttle(ctx, page)
    return page


def recover_or_replace_page(page, ctx, **kwargs):
    """The shared recovery, with the throttle carried onto a replacement page.

    `Emulation.setCPUThrottlingRate` is scoped to the PAGE TARGET, so a page
    from `ctx.new_page()` runs at full speed however the option was set, and
    every remaining step would pass under exactly the conditions the throttle
    exists to reproduce. Wrapping the import rather than each call site means a
    fourth recovery point cannot forget it.
    """
    replacement = _robust_recover_or_replace_page(page, ctx, **kwargs)
    if replacement is not page:
        apply_cpu_throttle(ctx, replacement)
    return replacement


def expected_default_model():
    override = os.environ.get("EXPECTED_DEFAULT_MODEL")
    if override:
        return override

    # importing it: the --no-torch Playwright install can't import the
    # Parse DEFAULT_MODELS_GGUF as a literal out of defaults.py instead of importing it:
    import ast

    defaults_path = (
        Path(__file__).resolve().parents[2]
        / "studio"
        / "backend"
        / "core"
        / "inference"
        / "defaults.py"
    )
    try:
        tree = ast.parse(defaults_path.read_text(encoding = "utf-8"))
    except Exception as exc:
        fail(f"could not read {defaults_path}: {exc}")
    models = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "DEFAULT_MODELS_GGUF" for t in node.targets):
            continue
        try:
            models = ast.literal_eval(node.value)
        except Exception as exc:
            fail(f"could not eval DEFAULT_MODELS_GGUF literal: {exc}")
        break
    if not models:
        fail("DEFAULT_MODELS_GGUF not found or empty in defaults.py")
    return models[0]


def soft_fail(m):
    """Hard fail in STRICT mode, info-warn otherwise."""
    if STRICT:
        fail(m)
    info(f"WARN (strict-off): {m}")


def exercise_permission_mode_controls(page, shoot):
    """Exercise labels, migration, persistence, confirmation, and focus."""
    step("permission levels: labels, persistence, confirmation, and focus")
    pill = page.locator('button[aria-label="Permission level for tool calls"]:visible').first
    expect(pill).to_be_visible()

    def expect_mode(label):
        expect(pill).to_have_attribute("data-pill-label", label)
        expect(pill).to_contain_text(label)

    def open_menu():
        pill.click()
        menu = page.get_by_role("menu").last
        expect(menu).to_be_visible()
        return menu

    def choose(label):
        menu = open_menu()
        item = menu.get_by_role("menuitem").filter(has_text = label).first
        expect(item).to_be_visible()
        item.click()

    def set_legacy_confirm(legacy_value):
        page.evaluate(
            """(legacyValue) => {
                localStorage.removeItem("unsloth_chat_permission_mode");
                if (legacyValue === null) {
                    localStorage.removeItem("unsloth_chat_confirm_tool_calls");
                } else {
                    localStorage.setItem(
                        "unsloth_chat_confirm_tool_calls",
                        legacyValue,
                    );
                }
            }""",
            legacy_value,
        )

    # The level is an installation setting, mirrored through /api/chat/settings, so "fresh profile" is no longer "fresh
    def refuse_settings_hydration(route):
        if route.request.method == "GET":
            route.fulfill(
                status = 503,
                content_type = "application/json",
                body = json.dumps({"detail": "hydration disabled for this step"}),
            )
        else:
            route.continue_()

    # either side -- a race, not a regression.
    # Every reload in this block used to be followed by a bare `expect(pill).to_be_visible()` on the default 5s expect
    def reload_and_wait_for_pill():
        page.reload(wait_until = "domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout = 30_000)
        except Exception:
            pass  # best-effort -- proceed even if network never idles
        expect(pill).to_be_visible(timeout = 30_000)

    # choose() only drives THIS tab.
    # The mirror to /api/chat/settings is a 400ms trailing-edge debounce (SETTINGS_DEBOUNCE_MS, chat-runtime-store.ts)
    # whose only early flush is the beforeunload keepalive, so the pill turning over proves the click landed locally,
    # not that the installation stored it.
    # Measured on webkit, timed from the choose() below: choose returns at t+238ms, set_legacy_confirm at t+248ms, and
    # the reload starts there.
    # The debounce would not have fired until ~t+630ms, so the only PUT that goes out at all is the beforeunload
    # keepalive at t+252ms, and the reloaded page's hydrating GET arrives at t+697ms.
    # It was betting that an unload-time keepalive beats a hydrating GET by 445ms of loopback, on every engine, every
    # run.
    def expect_server_mode(expected, timeout_ms = 15_000):
        deadline = time.monotonic() + timeout_ms / 1000.0
        seen = "<never read>"
        while True:
            seen = page.evaluate(
                """async () => {
                    const token = localStorage.getItem("unsloth_auth_token");
                    const res = await fetch("/api/chat/settings", {
                        headers: token ? { Authorization: "Bearer " + token } : {},
                        cache: "no-store",
                    });
                    if (!res.ok) return "<http " + res.status + ">";
                    const body = await res.json();
                    return (body && body.settings && body.settings.permissionMode) ?? null;
                }"""
            )
            if seen == expected:
                return
            if time.monotonic() >= deadline:
                break
            page.wait_for_timeout(100)
        fail(
            f"permission level never reached the installation: /api/chat/settings "
            f"reports permissionMode={seen!r} after {timeout_ms}ms, expected "
            f"{expected!r} -- the debounced mirror never landed"
        )

    page.route("**/api/chat/settings", refuse_settings_hydration)
    set_legacy_confirm(None)
    reload_and_wait_for_pill()

    expect_mode("Approve for me")
    menu = open_menu()
    for label in (
        "Ask for approval",
        "Approve for me",
        "Run automatically",
        "Full access",
    ):
        expect(menu.get_by_role("menuitem").filter(has_text = label).first).to_be_visible()
    if menu.get_by_text("Off", exact = True).count() != 0:
        fail("legacy Off label is still visible")
    if menu.locator('[role="menuitem"] button, [role="menuitem"] [role="button"]').count():
        fail("permission menu contains nested interactive controls")
    page.keyboard.press("Escape")
    expect(pill).to_be_focused()

    choose("Approve for me")
    expect_mode("Approve for me")
    expect(page.get_by_role("alertdialog")).to_have_count(0)

    compact_width = 390
    page.set_viewport_size({"width": compact_width, "height": 844})
    expect(pill).to_be_visible()

    # Let the reflow land before measuring.
    # set_viewport_size returns once the viewport is set, not once the layout has responded to it, and to_be_visible
    # does not cover the gap: the pill is already visible, at its old width.
    def fits_compact(box) -> bool:
        return box is not None and box["x"] >= 0 and box["x"] + box["width"] <= compact_width

    deadline = time.time() + 5
    box = pill.bounding_box()
    while not fits_compact(box) and time.time() < deadline:
        page.wait_for_timeout(50)
        box = pill.bounding_box()
    if not fits_compact(box):
        fail(f"permission pill is clipped in compact layout: {box!r}")
    page.set_viewport_size({"width": 1280, "height": 900})

    # Legacy setting migration:
    migration_cases = (
        ("true", "Ask for approval"),
        ("false", "Run automatically"),
        (None, "Approve for me"),
    )
    try:
        for legacy_value, expected_label in migration_cases:
            set_legacy_confirm(legacy_value)
            reload_and_wait_for_pill()
            expect_mode(expected_label)
    finally:
        page.unroute("**/api/chat/settings", refuse_settings_hydration)

    # The other half of that contract:
    choose("Ask for approval")
    expect_mode("Ask for approval")
    expect_server_mode("ask")
    set_legacy_confirm("false")
    reload_and_wait_for_pill()
    expect_mode("Ask for approval")
    cached = page.evaluate("() => localStorage.getItem('unsloth_chat_permission_mode')")
    if cached != "ask":
        fail(f"hydration left the local cache at {cached!r}, expected 'ask'")

    choose("Run automatically")
    expect_mode("Run automatically")
    expect(page.locator('button[data-pill-label="Search"]:visible').first).to_be_visible()
    expect(page.locator('button[data-pill-label="Code"]:visible').first).to_be_visible()
    stored = page.evaluate("() => localStorage.getItem('unsloth_chat_permission_mode')")
    if stored != "off":
        fail(f"Run automatically persisted {stored!r}, expected 'off'")

    choose("Full access")
    dialog = page.get_by_role("alertdialog")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("heading", name = "Enable Full access?")).to_be_visible()
    expect(dialog).to_contain_text("the code sandbox")
    dialog.get_by_role("button", name = "Cancel").click()
    expect(dialog).to_be_hidden()
    expect_mode("Run automatically")

    # Full access requires explicit consent and never overwrites persistence.
    choose("Full access")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name = "I understand").click()
    expect_mode("Full access")
    expect(pill).to_have_attribute("data-variant", "danger")
    active_icon = pill.locator(".composer-pill-glyph > :first-child")
    pill.hover()
    page.wait_for_timeout(200)
    icon_opacity = float(active_icon.evaluate("el => getComputedStyle(el).opacity"))
    if icon_opacity < 0.5:
        fail(f"Full access icon disappeared on hover (opacity={icon_opacity})")
    stored = page.evaluate("() => localStorage.getItem('unsloth_chat_permission_mode')")
    if stored != "off":
        fail(f"Full access overwrote persisted mode with {stored!r}")

    reload_and_wait_for_pill()
    expect_mode("Run automatically")

    # The active row is a no-op and must not open the Full access dialog.
    choose("Approve for me")
    # Fresh profiles default to Approve for me.
    expect_mode("Approve for me")
    shoot("04-permission-levels")


def login_via_api(pw):
    req = urllib.request.Request(
        f"{BASE}/api/auth/login",
        data = json.dumps({"username": "unsloth", "password": pw}).encode(),
        method = "POST",
        headers = {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout = 10) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def parse_rgb(s):
    m = re.search(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", s or "")
    return tuple(int(x) for x in m.groups()) if m else None


def exercise_floating_monitor_geometry(page):
    """Exercise content, drag, native resize, and viewport geometry."""
    monitor = page.get_by_test_id("floating-monitor")
    monitor.wait_for(state = "visible", timeout = 10_000)
    monitor_handle = page.get_by_test_id("floating-monitor-drag-handle")
    viewport = page.viewport_size
    if viewport is None:
        fail("Playwright viewport unavailable for floating monitor check")
    inset = 16
    tolerance = 1

    def monitor_box(label):
        box = monitor.bounding_box()
        if box is None:
            fail(f"floating monitor has no bounding box during {label}")
        return box

    def wait_for_box(label, predicate):
        deadline = time.time() + 5
        box = monitor_box(label)
        while not predicate(box) and time.time() < deadline:
            page.wait_for_timeout(50)
            box = monitor_box(label)
        if not predicate(box):
            fail(f"floating monitor did not settle during {label}: {box!r}")
        return box

    def pointer_drag(start_x, start_y, end_x, end_y):
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(end_x, end_y, steps = 10)
        page.mouse.up()
        page.wait_for_timeout(100)

    def drag_monitor_to(x, y):
        box = monitor_handle.bounding_box()
        if box is None:
            fail("floating monitor handle has no bounding box")
        pointer_drag(
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] / 2,
            x,
            y,
        )
        return monitor_box("drag")

    def resize_monitor_to(
        x,
        y,
        grip_inset = 8,
    ):
        before = monitor_box("resize")
        pointer_drag(
            before["x"] + before["width"] - grip_inset,
            before["y"] + before["height"] - grip_inset,
            x,
            y,
        )
        return before, monitor_box("resize")

    def expect_close(actual, expected, label):
        if abs(actual - expected) > tolerance:
            fail(f"{label}: expected {expected!r}, got {actual!r}")

    def is_inside(box, surface):
        return (
            box["x"] >= inset - tolerance
            and box["y"] >= inset - tolerance
            and box["x"] + box["width"] <= surface["width"] - inset + tolerance
            and box["y"] + box["height"] <= surface["height"] - inset + tolerance
        )

    # Every assertion below compares heights sampled seconds apart against this baseline, so the panel must already be
    # showing its final row set.
    # Until the first /api/system response is applied the panel paints use-system.ts's zero-filled DEFAULT_SYSTEM, which
    # has no GPU: on a host that reports one (macos-14 reports a single MLX device) the VRAM row then appears and adds
    # ~59px permanently.
    try:
        page.wait_for_function(
            r"""() => {
                const monitor = document.querySelector(
                    '[data-testid="floating-monitor"]'
                );
                const content = document.querySelector(
                    '[data-testid="floating-monitor-content"]'
                );
                if (!(monitor && content)) return false;
                // DEFAULT_SYSTEM reports a 0 GiB RAM total; a real payload never does.
                const readout = content.innerText.match(
                    /([\d.]+)\s*GiB\s*\/\s*([\d.]+)\s*GiB/
                );
                if (!readout || !(Number.parseFloat(readout[2]) > 0)) return false;
                // The rows commit a pass before the panel resizes to them, so the
                // panel is only done reacting once its scroll region exactly fits
                // the content it was reconciled against.
                const scroll = content.parentElement;
                const monitorHeight = monitor.getBoundingClientRect().height;
                const contentHeight = content.getBoundingClientRect().height;
                const scrollHeight = scroll.getBoundingClientRect().height;
                if (Math.abs(scrollHeight - contentHeight) > 1) return false;
                // Row insertion also lands a frame before the gap between rows
                // does, and that intermediate state is self-consistent. Require
                // the geometry to hold for two consecutive animation frames --
                // this runs under the default polling="raf", and it is how
                // Playwright itself defines a stable element.
                //
                // Position belongs in the signature as well as size. An undragged
                // panel is bottom-anchored by re-clamping its top against the new
                // height, and that lands a frame AFTER the height it reacts to, so
                // a size-only signature reports settled while the panel is still
                // where the shorter version put it. Sampling there reads a bottom
                // inset that is exactly the growth too low. Deliberately not
                // waiting on the expected inset itself: that would gate on the
                // very thing the assertions below check and turn a real
                // misplacement into a timeout instead of a failure.
                const rect = monitor.getBoundingClientRect();
                const signature = [
                    monitorHeight, contentHeight,
                    Math.round(rect.top), Math.round(rect.left),
                ].join("x");
                const settled = window.__unslothMonitorGeometry === signature;
                window.__unslothMonitorGeometry = signature;
                return settled;
            }""",
            timeout = 30_000,
        )
    except Exception as exc:
        fail(f"floating monitor never settled on an /api/system payload: {exc!r}")

    initial_box = monitor_box("initial placement")
    expect_close(
        initial_box["x"] + initial_box["width"],
        viewport["width"] - inset,
        "initial right inset",
    )
    expect_close(
        initial_box["y"] + initial_box["height"],
        viewport["height"] - inset,
        "initial bottom inset",
    )

    # Delayed GPU rows must expand upward and retain the initial bottom anchor.
    monitor.get_by_test_id("floating-monitor-content").evaluate(
        """node => {
            const probe = document.createElement("div");
            probe.dataset.testid = "floating-monitor-growth-probe";
            probe.style.height = "48px";
            node.appendChild(probe);
        }"""
    )
    grown_box = wait_for_box(
        "content growth",
        lambda box: box["height"] >= initial_box["height"] + 47,
    )
    expect_close(
        grown_box["y"] + grown_box["height"],
        viewport["height"] - inset,
        "content growth bottom inset",
    )
    monitor.get_by_test_id("floating-monitor-growth-probe").evaluate("node => node.remove()")
    initial_box = wait_for_box(
        "content shrink",
        lambda box: (
            abs(box["height"] - initial_box["height"]) <= tolerance
            and abs(box["y"] + box["height"] - viewport["height"] + inset) <= tolerance
        ),
    )

    # Chromium retains a blocked inline resize request.
    _, blocked_box = resize_monitor_to(
        viewport["width"] - 2,
        viewport["height"] - 2,
    )
    expect_close(blocked_box["width"], initial_box["width"], "blocked width")
    expect_close(blocked_box["height"], initial_box["height"], "blocked height")
    left_box = drag_monitor_to(0, viewport["height"] / 2)
    expect_close(left_box["x"], inset, "left inset")
    expect_close(left_box["width"], initial_box["width"], "post-drag width")
    expect_close(left_box["height"], initial_box["height"], "post-drag height")
    right_box = drag_monitor_to(viewport["width"], viewport["height"] / 2)
    expect_close(
        right_box["x"] + right_box["width"],
        viewport["width"] - inset,
        "right inset",
    )

    # Constraint changes during pointer capture must rebase the active drag.
    handle_box = monitor_handle.bounding_box()
    if handle_box is None:
        fail("floating monitor handle has no active-drag bounding box")
    page.mouse.move(
        handle_box["x"] + handle_box["width"] / 2,
        handle_box["y"] + handle_box["height"] / 2,
    )
    page.mouse.down()
    reduced_viewport = {"width": 500, "height": 400}
    page.set_viewport_size(reduced_viewport)
    page.mouse.move(498, 398, steps = 10)
    page.mouse.up()
    wait_for_box(
        "active viewport shrink",
        lambda box: is_inside(box, reduced_viewport),
    )

    narrow_viewport = {"width": 260, "height": 400}
    page.set_viewport_size(narrow_viewport)
    wait_for_box("narrow viewport", lambda box: is_inside(box, narrow_viewport))
    page.set_viewport_size(viewport)
    wait_for_box("viewport restore", lambda box: is_inside(box, viewport))

    resize_start = drag_monitor_to(0, 0)
    _, resized_box = resize_monitor_to(
        resize_start["x"] + resize_start["width"] - 8 + 40,
        resize_start["y"] + resize_start["height"] - 8 + 30,
    )
    expect_close(resized_box["width"], resize_start["width"] + 40, "resize width")
    expect_close(resized_box["height"], resize_start["height"] + 30, "resize height")
    expect_close(resized_box["x"], resize_start["x"], "resize left edge")
    expect_close(resized_box["y"], resize_start["y"], "resize top edge")

    _, minimum_box = resize_monitor_to(
        resized_box["x"] + resized_box["width"] - 102,
        resized_box["y"] + resized_box["height"] - 102,
        grip_inset = 2,
    )
    expect_close(minimum_box["width"], resize_start["width"], "minimum width")
    expect_close(minimum_box["height"], resize_start["height"], "minimum height")

    drag_monitor_to(0, 0)
    _, maximum_box = resize_monitor_to(
        viewport["width"] - 2,
        viewport["height"] - 2,
    )
    expect_close(
        maximum_box["x"] + maximum_box["width"],
        viewport["width"] - inset,
        "maximum resize right inset",
    )
    expect_close(
        maximum_box["y"] + maximum_box["height"],
        viewport["height"] - inset,
        "maximum resize bottom inset",
    )

    # Do not leave the maximum-size overlay above the shutdown controls.
    monitor.get_by_role("button", name = "Close").click()
    monitor.wait_for(state = "hidden")
    info(
        "OK floating monitor preserves native resize and stays stable across "
        "content, drag, and viewport changes"
    )


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ui",
        info = info,
    )
    # Pre-flight: macos-14 can surface a 200 /api/health while the auth DB is still migrating;
    # this 30s probe catches that gap before we sink 60s into a change-password timeout.
    wait_for_health(BASE, timeout = 30.0, info = info)
    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        fail(f"unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
    browser_type = getattr(p, PLAYWRIGHT_BROWSER)
    launch_kwargs = {"headless": True}
    if PLAYWRIGHT_BROWSER == "chromium":
        launch_kwargs["args"] = chromium_launch_args()
        if PLAYWRIGHT_CHANNEL:
            launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL
    elif PLAYWRIGHT_CHANNEL:
        fail("STUDIO_PLAYWRIGHT_CHANNEL requires chromium")
    if CPU_THROTTLE > 1 and PLAYWRIGHT_BROWSER != "chromium":
        # Refused here rather than at the call: `new_cdp_session` is Chromium only, so firefox/webkit would abort
        # mid-run with a Playwright error about CDP that says nothing about the option that caused it.
        fail(f"STUDIO_UI_CPU_THROTTLE requires chromium, not {PLAYWRIGHT_BROWSER}")
    browser = browser_type.launch(**launch_kwargs)
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        # Reduce motion so view-transition animations don't intercept pointer events and break Playwright's
        reduced_motion = "reduce",
    )
    # Hard-disable CSS view-transitions: Unsloth's theme toggle + sidebar collapse run startViewTransition() which can
    # leave <html> intercepting pointer events for a beat after each route swap.
    # See _playwright_robust.py.
    install_view_transition_killer(ctx)
    system_requests: list[str] = []
    ctx.on(
        "request",
        lambda request: (
            system_requests.append(request.url)
            if request.url.split("?", 1)[0].endswith("/api/system")
            else None
        ),
    )
    page = new_throttled_page(ctx)
    page_errors = []
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    console_errors: list[str] = []

    def _on_console(m):
        if m.type != "error":
            return
        try:
            text = m.text
        except Exception:
            return
        console_errors.append(text)

    page.on("console", _on_console)

    # Capture /v1/chat/completions statuses so a mid-test 4xx (which surfaces only as a hung wait_for_function) is
    chat_completions_responses: list[tuple[int, str]] = []
    page.on(
        "response",
        lambda r: (
            chat_completions_responses.append((r.status, r.url))
            if "/v1/chat/completions" in r.url
            else None
        ),
    )

    def shoot(name):
        # Screenshots are diagnostic only
        # Page.screenshot waits for webfonts, which on macos-14 can crowd the default;
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as _shoot_err:
            info(f"WARN: screenshot {name} failed: {_shoot_err}")

    # ─────────────────────────────────────────────────────
    step("change-password through UI (Setup your account)")
    # Settle the network before touching the form:
    form_err: Exception | None = None
    for _form_attempt in range(3):
        try:
            page.goto(f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000)
            try:
                page.wait_for_load_state("networkidle", timeout = 30_000)
            except Exception:
                pass
            pw_field = page.locator("#new-password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            # Do NOT shoot() between wait_for and fill -- the screenshot's font-load wait can let a background poll
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            shoot("01-change-password-filled")
            # Click submit AND wait for the POST response together so a macos-14 net::ERR_NO_BUFFER_SPACE buffer-fail
            # surfaces now, not at the next composer.wait_for.
            status, _ = click_and_wait_for_response(
                page,
                url_substr = "/api/auth/change-password",
                method = "POST",
                do_click = lambda: page.locator('button[type="submit"]').click(),
                timeout_ms = 30_000,
                info = lambda m: print(f"[ui]   {m}", flush = True),
            )
            if status is not None and status >= 400:
                raise AssertionError(
                    f"change-password POST returned {status}; "
                    f"see console_errors={console_errors[:1]!r}"
                )
            form_err = None
            break
        except Exception as e:
            form_err = e
            try:
                cur_url = page.url
            except Exception:
                cur_url = "<page closed>"
            print(
                f"[ui]   change-password form attempt {_form_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                f"page_errors={len(page_errors)} console_errors={len(console_errors)}",
                flush = True,
            )
            if console_errors:
                print(
                    f"[ui]   first console.error: {console_errors[0][:200]!r}",
                    flush = True,
                )
            if page_errors:
                print(f"[ui]   first pageerror:    {page_errors[0][:200]!r}", flush = True)
            try:
                shoot(f"01-change-password-attempt-{_form_attempt + 1}-fail")
            except Exception:
                pass
            if _form_attempt < 2:
                # ERR_NO_BUFFER_SPACE needs the OS to recover socket buffers;
                # back off 5s then 15s before retrying.
                if "ERR_NO_BUFFER_SPACE" in str(e):
                    backoff_s = 5 if _form_attempt == 0 else 15
                    print(
                        f"[ui]   ENOBUFS detected; sleeping {backoff_s}s "
                        f"before retry to let OS recover socket buffers...",
                        flush = True,
                    )
                    time.sleep(backoff_s)
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ui]   recovery: {m}", flush = True),
                )
    if form_err is not None:
        raise form_err

    # ─────────────────────────────────────────────────────
    step("wait for composer to mount")
    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass  # best-effort -- proceed even if network never idles

    composer = page.locator('textarea[aria-label="Message input"]')
    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            composer.wait_for(state = "visible", timeout = 60_000)
            last_err = None
            break
        except Exception as e:
            last_err = e
            try:
                cur_url = page.url
            except Exception:
                cur_url = "<page closed>"
            print(
                f"[ui]   composer.wait_for attempt {_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                f"page_errors={len(page_errors)} console_errors={len(console_errors)}",
                flush = True,
            )
            if console_errors:
                print(
                    f"[ui]   first console.error: {console_errors[0][:200]!r}",
                    flush = True,
                )
            if page_errors:
                print(f"[ui]   first pageerror:    {page_errors[0][:200]!r}", flush = True)
            try:
                shoot(f"03-composer-wait-attempt-{_attempt + 1}-fail")
            except Exception:
                pass  # best-effort -- proceed even if network never idles
            if _attempt == 0:
                # Replace the page if it died;
                # otherwise next iteration's page.goto() handles the reload.
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    goto_url = BASE,
                    settle_networkidle = True,
                    info = lambda m: print(f"[ui]   recovery: {m}", flush = True),
                )
                composer = page.locator('textarea[aria-label="Message input"]')
    if last_err is not None:
        raise last_err
    shoot("03-chat-loaded")

    exercise_permission_mode_controls(page, shoot)
    if PERMISSION_ONLY:
        info(
            "permission-only run passed "
            f"(browser={PLAYWRIGHT_BROWSER}, channel={PLAYWRIGHT_CHANNEL or 'bundled'})"
        )
        browser.close()
        sys.exit(0)

    # /api/models/list and /api/inference/load need a bearer;
    # the frontend stores it under "unsloth_auth_token" (auth/session.ts).
    token = robust_evaluate(
        page,
        "() => localStorage.getItem('unsloth_auth_token')",
    )
    if not token:
        # Fall back: exchange the refresh token via /api/auth/refresh.
        refresh_token = robust_evaluate(
            page,
            "() => localStorage.getItem('unsloth_auth_refresh_token')",
        )
        if refresh_token:
            refresh_resp = evaluate_fetch(
                page,
                f"{BASE}/api/auth/refresh",
                method = "POST",
                headers = {"Content-Type": "application/json"},
                body = {"refresh_token": refresh_token},
                timeout_ms = FETCH_TIMEOUT_MS,
            )
            if refresh_resp.get("error"):
                fail(f"/api/auth/refresh wedged: {refresh_resp['error']!r}")
            refresh = refresh_resp.get("body") or {}
            token = (refresh or {}).get("access_token")
            next_refresh_token = (refresh or {}).get("refresh_token")
            if token and next_refresh_token:
                robust_evaluate(
                    page,
                    """([accessToken, refreshToken]) => {
                        localStorage.setItem('unsloth_auth_token', accessToken);
                        localStorage.setItem('unsloth_auth_refresh_token', refreshToken);
                    }""",
                    [token, next_refresh_token],
                )
            elif token:
                fail("/api/auth/refresh returned access_token but no refresh_token")
    if not token:
        fail("could not obtain auth token after change-password")

    # Verify the chat page's default model matches DEFAULT_MODELS_GGUF[0] (defaults.py) -- guards the first-launch UX
    step("default_models[0] matches DEFAULT_MODELS_GGUF[0]")
    EXPECTED_DEFAULT = expected_default_model()
    defaults_resp = evaluate_fetch(
        page,
        f"{BASE}/api/models/list",
        headers = {"Authorization": f"Bearer {token}"},
        timeout_ms = FETCH_TIMEOUT_MS,
    )
    if defaults_resp.get("error") or defaults_resp.get("status") != 200:
        fail(
            f"/api/models/list failed: status={defaults_resp.get('status')!r} "
            f"error={defaults_resp.get('error')!r}"
        )
    defaults = defaults_resp["body"] or {}
    if not defaults.get("default_models"):
        fail(f"/api/models/list returned no default_models: {defaults}")
    if defaults["default_models"][0] != EXPECTED_DEFAULT:
        fail(
            f"default_models[0]={defaults['default_models'][0]!r}, "
            f"expected {EXPECTED_DEFAULT!r}; defaults.py drift?"
        )
    info(f"OK default_models[0] = {EXPECTED_DEFAULT}")

    # The selector button should show the default model's name even before a model is loaded ("Select model" if none).
    selector_btn = page.locator(
        'button:has-text("Select model"), '
        'button:has-text("gemma"), '
        'button:has-text("Qwen"), '
        'button:has-text("Llama")'
    ).first
    # Best-effort: selector re-mounts as /api/models/list resolves, so use a short timeout and skip the snapshot on
    sel_text = ""
    # After change-password the router rebuilds login -> chat shell;
    try:
        sel_text = (selector_btn.text_content(timeout = 2_000) or "").strip()
    except Exception as _sel_err:
        info(f"WARN: model-selector probe skipped: {type(_sel_err).__name__}: {_sel_err}")
    if sel_text:
        info(f"model selector button text: {sel_text!r}")
        shoot("03b-default-model-button")

    # ─────────────────────────────────────────────────────
    step("load GGUF via /api/inference/load (uses session cookie)")
    # AbortSignal-bounded: macos-14 has been seen wedging on this fetch.
    load_resp = evaluate_fetch(
        page,
        f"{BASE}/api/inference/load",
        method = "POST",
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        body = {
            "model_path": GGUF_REPO,
            "gguf_variant": GGUF_VARIANT,
            "is_lora": False,
            "max_seq_length": 2048,
        },
        timeout_ms = LOAD_FETCH_TIMEOUT_MS,
    )
    if load_resp.get("error"):
        fail(f"/api/inference/load wedged: {load_resp['error']!r}")
    if load_resp["status"] != 200:
        fail(f"/api/inference/load returned {load_resp['status']}: {load_resp.get('body')!r}")
    info(f"loaded model: {(load_resp['body'] or {}).get('display_name')}")

    # Unsloth caches model state in zustand;
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # this just catches picker-mount / debounced HF-search regressions.
    # ─────────────────────────────────────────────────────
    step("model picker: open + drive search bar")
    # Prefer the guided-tour anchor [data-tour="chat-model-selector"] (app-sidebar.tsx) -- as stable as anything in the
    picker_btn = page.locator('[data-tour="chat-model-selector"]').first
    if picker_btn.count() == 0:
        # Fall back to text-based locators for older Unsloth builds.
        picker_btn = page.locator(
            'button:has-text("gemma-3-270m"), '
            'button:has-text("Gemma 3"), '
            'button:has-text("Select model")'
        ).first
    if picker_btn.count() == 0:
        soft_fail("model picker button not found")
    else:
        picker_btn.click()
        page.wait_for_timeout(500)
        shoot("03c-model-picker-open")
        search = page.get_by_placeholder(
            re.compile(r"Search.*models?", re.I),
        ).first
        if search.count() == 0:
            soft_fail("model picker search input not found")
        else:
            # typeahead actually filters (else an ignored-input regression
            # "qwen" then "llama" popover text must DIFFER, proving the typeahead actually filters (else an
            def picker_visible_text():
                return robust_evaluate(
                    page,
                    """() => {
                    const el = document.querySelector(
                        '[role="dialog"], [role="listbox"], [role="menu"]'
                    );
                    return el ? (el.innerText || '').trim() : '';
                }""",
                )

            search.fill("qwen")
            page.wait_for_timeout(800)
            qwen_text = picker_visible_text()
            shoot("03d-model-picker-search-qwen")
            search.fill("")
            page.wait_for_timeout(300)
            search.fill("llama")
            page.wait_for_timeout(800)
            llama_text = picker_visible_text()
            shoot("03e-model-picker-search-llama")
            if qwen_text and llama_text and qwen_text == llama_text:
                soft_fail(
                    "model picker text was identical for qwen + llama "
                    "queries -- typeahead may not be filtering"
                )
            else:
                info("OK search bar filtered (qwen text != llama text)")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    # ─────────────────────────────────────────────────────
    prompts = [
        "Reply with exactly: hello",
        "What is 1+1? Reply with the digit only.",
        "Reply with exactly: world",
        "Reply with exactly: tree",
        "What is 2+2? Reply with the digit only.",
    ]

    def _bubble_count():
        """Total [data-role='assistant'] elements (empty or not)."""
        return robust_evaluate(
            page,
            """() => {
            return document.querySelectorAll('[data-role="assistant"]').length;
        }""",
        )

    def send_and_wait(prompt, idx):
        # Wait until the previous turn fully stopped:
        page.wait_for_selector(
            'button[aria-label="Send message"]',
            state = "attached",
            timeout = TURN_TIMEOUT_MS,
        )
        try:
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = 5_000,
            )
        except Exception:
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = TURN_TIMEOUT_MS,
            )

        # Snapshot total bubble count before send;
        bubbles_before = _bubble_count()
        # The llama.cpp and web update banners are fixed bottom-right toasts (z-9998 / z-9999) that can overlap the
        for prefix in ("llama", "web"):
            snooze_btn = page.locator(f'[data-testid="{prefix}-update-snooze-button"]')
            if snooze_btn.count():
                try:
                    snooze_btn.first.click(timeout = 2_000)
                    page.wait_for_selector(
                        f'[data-testid="{prefix}-update-banner"]',
                        state = "detached",
                        timeout = 5_000,
                    )
                except Exception:
                    pass
        composer.click()
        composer.fill(prompt)
        page.locator('button[aria-label="Send message"]').click()

        # Wait for the new placeholder bubble to render -- confirms the click was actionable and the request issued.
        page.wait_for_function(
            """(want) => {
                return document.querySelectorAll(
                    '[data-role="assistant"]'
                ).length >= want;
            }""",
            arg = bubbles_before + 1,
            timeout = TURN_TIMEOUT_MS,
        )

        try:
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "attached",
                timeout = 3_000,
            )
        except Exception:
            pass
        try:
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = TURN_TIMEOUT_MS,
            )
        except Exception:
            shoot(f"04-turn-{idx}-still-streaming")
            raise

    step("rapid submit: 100 ms follow-up queues behind the first turn")
    rapid_bubbles_before = _bubble_count()
    composer_form = page.locator('form:has(textarea[aria-label="Message input"])').first
    # How long a reply takes is not ours to decide: sampling settings, whatever GGUF_REPO points at and an early EOS all
    # move it, and a short answer can finish inside the follow-up delay on a fast runner, leaving nothing to queue
    # behind.
    # The sync Playwright route handler runs on this thread, so a wait inside it blocks the test: the handler would fire
    # during a wait_for_timeout, finish, and release the request before a Python-side second submit could happen, which
    page.evaluate(
        """(args) => {
            const [secondPrompt, holdMs] = args;
            window.__unslothRapid = {
                intercepted: false, preparing: false, submitted: false, queueSeen: false,
                observed: false, error: null, seen: [], holdUntil: 0,
            };
            const state = window.__unslothRapid;
            const realFetch = window.fetch;
            const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

            const sendFollowUp = (deadline) => {
                if (state.preparing || state.submitted || state.error) return;
                if (deadline === undefined) deadline = Date.now() + 5000;
                if (state.holdUntil) {
                    deadline = Math.min(deadline, state.holdUntil - 250);
                }
                // Re-query, and retry: this is the chat's first message, so
                // sending it swaps the welcome composer for the dock composer.
                // A node captured earlier is detached, and for a short window
                // there is no connected composer at all.
                const composer = document.querySelector(
                    'textarea[aria-label="Message input"]'
                );
                if (!composer || !composer.isConnected || !composer.form) {
                    // Never retry past the hold. The response is released when
                    // it expires, so a submit after that races a buffered reply
                    // finishing first and would report a queue failure for an
                    // application that behaved correctly.
                    if (Date.now() > deadline) {
                        state.error = "no connected composer for the follow-up";
                        return;
                    }
                    setTimeout(() => sendFollowUp(deadline), 25);
                    return;
                }
                // React tracks the value on the node, so a plain assignment is
                // reverted on the next render.
                const setValue = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, "value"
                ).set;
                setValue.call(composer, secondPrompt);
                composer.dispatchEvent(new Event("input", { bubbles: true }));
                // Synthetic input and requestSubmit in the same JS turn can make
                // the form callback read the previous controlled value. That is
                // not how a user types, and it leaves the new text visible while
                // the test incorrectly records a submit. Let React publish the
                // input, then re-check the composer because the welcome bar can
                // be replaced by the dock bar during this frame.
                state.preparing = true;
                requestAnimationFrame(() => {
                    state.preparing = false;
                    const current = document.querySelector(
                        'textarea[aria-label="Message input"]'
                    );
                    if (
                        !current || !current.isConnected || !current.form ||
                        current.value !== secondPrompt
                    ) {
                        if (Date.now() > deadline) {
                            state.error = "follow-up composer did not settle";
                            return;
                        }
                        setTimeout(() => sendFollowUp(deadline), 25);
                        return;
                    }
                    current.form.requestSubmit();
                    state.submitted = true;
                });
            };

            window.fetch = async (...a) => {
                const url = String(
                    (a[0] && a[0].url) ? a[0].url : a[0]
                );
                const isTurn = url.includes("chat/completions");
                if (isTurn && !state.intercepted) {
                    state.intercepted = true;
                    state.holdUntil = Date.now() + holdMs;
                    state.seen.push(url);
                    const response = realFetch(...a);
                    // The request is out and the turn is running. Send the
                    // follow-up now, then keep the response pending so it
                    // cannot complete first.
                    setTimeout(() => sendFollowUp(), 0);
                    await sleep(holdMs);
                    return response;
                }
                return realFetch(...a);
            };

            // Always restore. A wrapper left in place for the rest of the
            // run is a monkeypatch with no teardown, and every later turn would
            // pay for it.
            window.__unslothRapidArm = () => setTimeout(() => sendFollowUp(), 100);

            window.__unslothRapidRestore = () => {
                window.fetch = realFetch;
                clearInterval(poll);
            };

            const poll = setInterval(() => {
                if (document.querySelector(
                    'button[aria-label="Remove queued prompt 1"]'
                )) {
                    state.queueSeen = true;
                    clearInterval(poll);
                }
            }, 25);
            setTimeout(() => {
                clearInterval(poll);
                if (!state.queueSeen && !state.error) {
                    // Resolve the wait rather than let it die on a generic
                    // Playwright timeout. That path is the regression this step
                    // exists to catch, and leaving it unresolved makes the
                    // screenshot, the explicit message and the fetch teardown
                    // below unreachable in exactly that case.
                    state.observed = true;
                }
            }, 20000);
        }""",
        ["Reply with exactly: rapid-second", int(RAPID_FIRST_TURN_HOLD_S * 1000)],
    )

    composer.fill("Reply with exactly: rapid-first")
    composer_form.evaluate("form => form.requestSubmit()")
    # Arm the 100 ms path now that the first turn has been submitted.
    page.evaluate("() => window.__unslothRapidArm && window.__unslothRapidArm()")

    page.wait_for_function(
        """() => window.__unslothRapid
            && (window.__unslothRapid.queueSeen
                || window.__unslothRapid.observed
                || window.__unslothRapid.error)""",
        timeout = 60_000,
    )
    state = page.evaluate("() => window.__unslothRapid")
    page.evaluate("() => { if (window.__unslothRapidRestore) window.__unslothRapidRestore(); }")
    if state["error"]:
        shoot("04-rapid-submit-no-composer")
        fail(f"could not send the follow-up: {state['error']}")
    # when it did not, so an unheld run cannot report a silent pass.
    # queueSeen is the property under test;
    if not state["queueSeen"] and not state["intercepted"]:
        fail(
            "the first turn's request was never seen, so it was never held, "
            f"and no queue formed; saw {state['seen']}"
        )
    # missing queue control is therefore a real regression, not timing.
    # The follow-up went out after the first turn's request was issued and while its response was still held, so that
    if not state["queueSeen"]:
        shoot("04-rapid-submit-no-queue")
        fail(
            "follow-up sent during a held first turn did not appear as queued "
            f"work (submitted={state['submitted']}, intercepted="
            f"{state['intercepted']})"
        )

    # Settle before the five-turn sequence below: two bubbles, nothing streaming, nothing queued.
    # What the turns SAID is not checked here and never was;
    # the queue behaviour this step exists to prove is `state.queueSeen` above.
    page.wait_for_function(
        """(want) => {
            const replies = Array.from(
                document.querySelectorAll('[data-role="assistant"]')
            ).slice(-2);
            return replies.length === 2 &&
                document.querySelectorAll('[data-role="assistant"]').length >= want &&
                !document.querySelector('button[aria-label="Stop generating"]') &&
                !document.querySelector('button[aria-label="Remove queued prompt 1"]');
        }""",
        arg = rapid_bubbles_before + 2,
        timeout = TURN_TIMEOUT_MS * 2,
    )
    shoot("04-rapid-submit-queued")
    info("OK 100 ms follow-up waited and both assistant turns completed")

    for i, p_ in enumerate(prompts, start = 1):
        step(f"turn {i}: {p_!r}")
        send_and_wait(p_, i)
    shoot("04-after-five-turns")

    texts = robust_evaluate(
        page,
        """() => Array.from(document.querySelectorAll('[data-role="assistant"]'))
        .map(e => (e.innerText || '').trim())""",
    )
    if len(texts) < len(prompts):
        fail(f"expected >= {len(prompts)} assistant bubbles, got {len(texts)}")
    info(f"five turn lengths = {[len(t) for t in texts[:5]]}")
    # Surface /v1/chat/completions status distribution:
    if chat_completions_responses:
        statuses = [code for code, _ in chat_completions_responses]
        bad = [code for code in statuses if code >= 400]
        info(
            f"/v1/chat/completions: {len(statuses)} request(s); "
            f"statuses={statuses}; 4xx/5xx={len(bad)}"
        )

    # ─────────────────────────────────────────────────────
    step("regenerate last assistant turn")
    last_assistant = page.locator('[data-role="assistant"]').last
    last_assistant.hover()
    page.wait_for_timeout(400)
    # Exclude disabled controls:
    regen_btn = (
        page.get_by_role(
            "button",
            name = re.compile(r"(reload|regenerate)", re.I),
        )
        .and_(page.locator("button:not([disabled])"))
        .first
    )
    if regen_btn.count() > 0:
        regen_btn.click()
        # Wait for this turn's streaming to finish.
        try:
            # Stop still on -- prior turn mid-stream.
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = 90_000,
            )
        except Exception:
            pass
        shoot("05-after-regenerate")
        info("regenerate completed")
    else:
        # Don't strict-fail: ActionBarPrimitive.Reload has no stable aria-label so the locator relies on icon-tied
        # tooltip text.
        # Soft-skip until we add a data-testid (TODO).
        info("WARN regenerate button not visible (known-fragile locator, skipped)")

    # ─────────────────────────────────────────────────────
    extra = ["Reply with: yes", "Reply with: no"]
    for j, p_ in enumerate(extra, start = 1):
        step(f"extra turn {j}: {p_!r}")
        before_count = len(page.locator('[data-role="assistant"]').all())
        send_and_wait(p_, before_count + 1)
    shoot("06-after-extra-turns")

    # ─────────────────────────────────────────────────────
    step("composer toggle buttons (Thinking / Web search / Code execution)")
    for feature in ("thinking", "web search", "code execution"):
        # Match whichever of "Disable X" / "Enable X" is rendered.
        toggle = page.locator(
            f'button[aria-label="Disable {feature}"], button[aria-label="Enable {feature}"]'
        ).first
        if toggle.count() == 0:
            info(f"toggle '{feature}' not present on this layout")
            continue
        # Skip if the button is disabled (model lacks the capability;
        if toggle.is_disabled():
            info(f"toggle '{feature}' is disabled for this model -- skip")
            continue
        before = toggle.get_attribute("aria-label") or ""
        toggle.click()
        page.wait_for_timeout(200)
        after = (
            page.locator(
                f'button[aria-label="Disable {feature}"], button[aria-label="Enable {feature}"]'
            ).first.get_attribute("aria-label")
            or ""
        )
        if before == after:
            info(f"WARN '{feature}' aria-label did not flip ({before!r})")
        else:
            info(f"OK '{feature}': {before!r} -> {after!r}")
        try:
            page.locator(
                f'button[aria-label="Disable {feature}"], button[aria-label="Enable {feature}"]'
            ).first.click()
        except Exception:
            pass
        page.wait_for_timeout(200)
    shoot("07-toggles-cycled")

    # ─────────────────────────────────────────────────────
    cfg_open = page.locator('button[aria-label="Open configuration"]').first
    if cfg_open.count() > 0:
        step("Configuration sheet: drive Temperature + Top P + extras")
        cfg_open.click()
        page.wait_for_timeout(500)
        shoot("08-config-open")
        # Walk every Radix slider (role="slider") by index, focus it, press Home (-> min) for deterministic state;
        sliders = page.locator('[role="slider"]')
        n_sliders = sliders.count()
        info(f"configuration sheet exposes {n_sliders} slider(s)")
        for idx in range(n_sliders):
            try:
                s = sliders.nth(idx)
                s.scroll_into_view_if_needed()
                s.focus()
                page.keyboard.press("Home")
                page.wait_for_timeout(80)
            except Exception as exc:
                info(f"  slider[{idx}] focus/Home failed: {exc!r}")
        shoot("09-config-all-min")
        # Temperature is the first slider (configuration-sheet.tsx), so Home already pinned it to 0 for determinism.
        info("Temperature set to slider min (0.0) for determinism")
        close_btn = page.locator('button[aria-label="Close configuration"]').first
        if close_btn.count() > 0:
            close_btn.click()
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    def read_chat_typography():
        """Read message typography after a user-driven theme transition."""
        return robust_evaluate(
            page,
            """() => {
                const root = document.documentElement;
                const assistant = Array.from(
                    document.querySelectorAll('.aui-assistant-message-root')
                );
                const user = Array.from(
                    document.querySelectorAll('.aui-user-message-root')
                );
                if (assistant.length === 0 || user.length === 0) {
                    return { error: 'chat message roots are missing' };
                }
                const ua = navigator.userAgent.toLowerCase();
                const role = (nodes) => {
                    const styles = nodes.map((node) => getComputedStyle(node));
                    return {
                        fontWeight: [...new Set(styles.map((style) => style.fontWeight))],
                        letterSpacing: [...new Set(styles.map((style) => style.letterSpacing))],
                        // The tracking is authored in em, so it only means anything next to the
                        // size it resolved against.
                        fontSize: [...new Set(styles.map((style) => style.fontSize))],
                    };
                };
                return {
                    // The scale the size tokens are multiplied by: index.css sets the 15px
                    // product default, and the appearance store overrides it inline as
                    // preference / 16 for any other size.
                    uiFontScale: getComputedStyle(root)
                        .getPropertyValue('--ui-font-scale').trim(),
                    actualRenderLinux: root.classList.contains('render-linux'),
                    isDesktopLinux: ua.includes('linux') && !ua.includes('android'),
                    isDark: root.classList.contains('dark'),
                    usesBaselineTypography: (
                        root.classList.contains('no-font-smoothing') ||
                        root.hasAttribute('data-chat-font') ||
                        root.hasAttribute('data-ui-font')
                    ),
                    assistant: role(assistant),
                    user: role(user),
                };
            }""",
        )

    # text-ui-15p5 unscaled (index.css:
    _TEXT_UI_15P5_PX = 15.5

    def assert_chat_typography(label, typography):
        if typography.get("error"):
            fail(typography["error"])
        if typography["actualRenderLinux"] != typography["isDesktopLinux"]:
            fail(f"desktop Linux detection mismatch: {typography!r}")
        is_dark = typography["isDark"]
        # Tracking is authored in em (thread.tsx tracking-[0.01em] / dark:tracking-[0.02em], and 0.023em for the lighter
        # dark-mode instance on Linux), so assert the em and let the size come from the element.
        # Pinning px assumed a 16px base and broke the moment the product default became 15px (--ui-font-scale in
        expected_em = 0.02 if is_dark else 0.01
        if typography["isDesktopLinux"] and not typography["usesBaselineTypography"]:
            expected_weight = "350" if is_dark else "390"
            if is_dark:
                expected_em = 0.023
        else:
            expected_weight = "410"
        for role in ("assistant", "user"):
            actual = typography[role]
            if actual["fontWeight"] != [expected_weight]:
                fail(
                    f"chat font weight {label}/{role}: expected {expected_weight}, "
                    f"got {actual['fontWeight']!r}"
                )
            if len(actual["fontSize"]) != 1:
                fail(f"chat font size {label}/{role}: not uniform, got {actual['fontSize']!r}")
            font_size = float(actual["fontSize"][0].removesuffix("px"))
            try:
                ui_font_scale = float(typography.get("uiFontScale") or "1")
            except ValueError:
                ui_font_scale = None
            if ui_font_scale is None:
                fail(f"chat font size {label}/{role}: unreadable --ui-font-scale")
            expected_size = _TEXT_UI_15P5_PX * ui_font_scale
            if abs(font_size - expected_size) > 0.01:
                fail(
                    f"chat font size {label}/{role}: expected text-ui-15p5, "
                    f"{_TEXT_UI_15P5_PX}px * {ui_font_scale} = {expected_size:g}px, "
                    f"got {font_size}px"
                )
            expected_spacing = expected_em * font_size
            # float() raises on "normal", which is how zero tracking is reported.
            spacings = [
                0.0 if v.strip() == "normal" else float(v.removesuffix("px"))
                for v in actual["letterSpacing"]
            ]
            # Sub-pixel tolerance only:
            if len(spacings) != 1 or abs(spacings[0] - expected_spacing) > 0.005:
                fail(
                    f"chat letter spacing {label}/{role}: expected {expected_em}em of "
                    f"{font_size}px = {expected_spacing:g}px, got {actual['letterSpacing']!r}"
                )

    # ─────────────────────────────────────────────────────
    acct = page.locator('button[aria-label$=" account menu"]').first
    if acct.count() > 0:
        step("theme toggle x3 with computed-color assertion")
        observed = []
        typography_states = []
        for cycle in range(3):
            try:
                page.wait_for_function(
                    """() => {
                        const m = document.querySelector('[role="menu"]');
                        if (!m) return true;
                        // Radix sets data-state="closed" during the
                        // close animation; treat that as already gone.
                        return m.getAttribute('data-state') === 'closed';
                    }""",
                    timeout = 7_000,
                )
            except Exception:
                pass
            page.wait_for_timeout(250)
            # Retry once (after Escape to clear stray popups) if the first click is silently swallowed
            opened = False
            for attempt in range(2):
                try:
                    click_forced(acct)
                except Exception as exc:
                    if attempt == 1:
                        soft_fail(f"theme cycle {cycle + 1}: account-menu click failed ({exc!r})")
                    continue
                try:
                    page.wait_for_selector(
                        '[role="menu"][data-state="open"]',
                        timeout = 5_000,
                    )
                    opened = True
                    break
                except Exception:
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
            if not opened:
                soft_fail(f"theme cycle {cycle + 1}: account menu didn't open")
                break
            theme_item = page.get_by_role(
                "menuitem",
                name = re.compile(r"^(Light Mode|Dark Mode)$", re.I),
            ).first
            if theme_item.count() == 0:
                page.keyboard.press("Escape")
                soft_fail(f"theme cycle {cycle + 1}: theme menuitem missing")
                break
            # Click with fallbacks: a small CI viewport can push the item off-screen (force=True still needs it in
            click_err = None
            for click_attempt in range(3):
                try:
                    if click_attempt == 0:
                        click_forced(theme_item, timeout = 3_000)
                    elif click_attempt == 1:
                        theme_item.scroll_into_view_if_needed(timeout = 2_000)
                        click_forced(theme_item, timeout = 3_000)
                    else:
                        theme_item.evaluate("el => el.click()")
                    click_err = None
                    break
                except Exception as exc:
                    click_err = exc
                    page.wait_for_timeout(200)
            if click_err is not None:
                page.keyboard.press("Escape")
                soft_fail(f"theme cycle {cycle + 1}: theme menuitem click failed ({click_err!r})")
                break
            # Settle. The ".dark" class on <html> is the ground truth (theme-store toggles only that);
            page.wait_for_timeout(700)
            bg = robust_evaluate(
                page,
                """() => {
                const root = document.documentElement;
                return {
                    cls:    root.className,
                    isDark: root.classList.contains('dark'),
                    bg:     getComputedStyle(document.body).backgroundColor,
                    rbg:    getComputedStyle(root).backgroundColor,
                };
            }""",
            )
            observed.append(bg)
            typography = read_chat_typography()
            assert_chat_typography(f"theme-cycle-{cycle + 1}", typography)
            typography_states.append(typography)
            shoot(f"10-theme-cycle-{cycle + 1}")
            info(f"  cycle {cycle + 1}: dark={bg['isDark']} body bg={bg['bg']!r}")
        # Across cycles we should see both a near-white (light) and a near-black (dark) body bg;
        rgbs = [parse_rgb(o["bg"]) for o in observed if parse_rgb(o["bg"])]
        light_seen = any(min(r) > 220 for r in rgbs)
        dark_seen = any(max(r) < 60 for r in rgbs)
        if len(observed) < 3:
            soft_fail(f"theme toggle ran only {len(observed)} cycle(s), expected 3")
        # completion above is the real invariant.
        # Don't strict-fail on both polarities:
        if light_seen and dark_seen:
            info("OK light + dark computed background colors observed")
        else:
            info(
                f"WARN observed only one polarity across {len(rgbs)} "
                f"cycles: light_seen={light_seen}, dark_seen={dark_seen} "
                "(toggle may not flip on this runner's color-scheme)"
            )

        # These are user-driven theme transitions, not synthetic class changes.
        if len(typography_states) != 3:
            soft_fail(
                f"chat typography observed {len(typography_states)} theme state(s), expected 3"
            )
        elif {state["isDark"] for state in typography_states} != {False, True}:
            soft_fail(f"chat typography did not observe both themes: {typography_states!r}")
        else:
            info("OK chat typography platform and theme behavior")
    else:
        soft_fail("chat typography requires the account-menu theme control")

    # ─────────────────────────────────────────────────────
    def click_nav(label, expected_url_pat = None):
        # Resolve the sidebar nav button.
        # get_by_role(name=...) works on Linux but the tooltip-derived name can be empty on macOS when the sidebar
        # collapses to icons, so fall back to more permissive locators.
        candidates = [
            page.get_by_role("button", name = re.compile(rf"^\s*{label}\s*$", re.I)).first,
            page.locator(f'button:has-text("{label}")').first,
            page.locator(f'a:has-text("{label}")').first,
            page.locator(f'[data-sidebar="menu-button"]:has-text("{label}")').first,
        ]
        btn = None
        for c in candidates:
            if c.count() > 0:
                btn = c
                break
        if btn is None:
            # Unpinned rows (Video, Recipes, Export by default) live in the sidebar's "More" flyout, which opens on
            more_btn = page.get_by_role("button", name = re.compile(r"^\s*More\s*$", re.I)).first
            if more_btn.count() > 0:
                more_btn.hover()
                page.wait_for_timeout(500)
                item = page.get_by_role("menuitem", name = re.compile(label, re.I)).first
                if item.count() == 0:
                    click_forced(more_btn)
                    page.wait_for_timeout(500)
                    item = page.get_by_role("menuitem", name = re.compile(label, re.I)).first
                if item.count() > 0:
                    btn = item
        if btn is None:
            soft_fail(f"nav '{label}' not found")
            return False
        try:
            click_forced(btn, timeout = 5_000)
        except Exception as exc:
            soft_fail(f"nav '{label}' click failed: {exc!r}")
            return False
        page.wait_for_timeout(800)
        if expected_url_pat and not re.search(expected_url_pat, page.url):
            soft_fail(
                f"clicking '{label}' didn't change url to /{expected_url_pat}; current: {page.url}"
            )
            return False
        return True

    step("sidebar nav: New Chat -> Compare -> Search -> Recipes")
    click_nav("New Chat", r"/chat")
    shoot("11-new-chat")
    # Compare moved into the composer "Tools and attachments" menu.
    plus_btn = page.get_by_role("button", name = re.compile(r"Tools and attachments", re.I)).first
    if plus_btn.count() > 0:
        click_forced(plus_btn)
        page.wait_for_timeout(400)
        compare_item = page.get_by_role("menuitem", name = re.compile(r"Compare chat", re.I)).first
        if compare_item.count() == 0:
            # Compare chat moved into the "More" submenu;
            more_trigger = page.get_by_role("menuitem", name = re.compile(r"^More$", re.I)).first
            if more_trigger.count() > 0:
                more_trigger.hover()
                page.wait_for_timeout(400)
                compare_item = page.get_by_role(
                    "menuitem", name = re.compile(r"Compare chat", re.I)
                ).first
                if compare_item.count() == 0:
                    click_forced(more_trigger)
                    page.wait_for_timeout(400)
                    compare_item = page.get_by_role(
                        "menuitem", name = re.compile(r"Compare chat", re.I)
                    ).first
        if compare_item.count() > 0:
            click_forced(compare_item)
            page.wait_for_timeout(800)
            if not re.search(r"/chat\?", page.url):
                soft_fail(f"'Compare chat' didn't open compare; current: {page.url}")
        else:
            soft_fail("composer + menu: 'Compare chat' item not found")
    else:
        soft_fail("composer + menu: plus button not found")
    shoot("12-compare")
    # Search opens a dialog (not a route change).
    search_btn = page.get_by_role("button", name = re.compile(r"^search$", re.I)).first
    if search_btn.count() > 0:
        search_btn.click()
        page.wait_for_timeout(500)
        shoot("13-search-dialog")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    click_nav("Recipes", r"/data-recipes")
    shoot("14-recipes")
    page.goto(f"{BASE}/chat")
    composer.wait_for(state = "visible", timeout = 60_000)

    if acct.count() > 0:
        step("Developer (API) tab via account menu")
        acct.click()
        page.wait_for_timeout(400)
        dev = page.get_by_role("menuitem", name = re.compile(r"developer|api", re.I)).first
        if dev.count() > 0:
            dev.click()
            page.wait_for_timeout(800)
            shoot("15-developer-tab")
            create_btn = page.get_by_role(
                "button",
                name = re.compile(r"create.*key|generate.*key|add.*key|new key", re.I),
            ).first
            if create_btn.count() > 0:
                info("OK 'create API key' affordance visible")
            keys_section = page.get_by_text(
                re.compile(r"api keys|developer", re.I),
            ).first
            if keys_section.count() > 0:
                info(f"OK API tab text: {(keys_section.text_content() or '').strip()[:80]!r}")
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        else:
            page.keyboard.press("Escape")

    # ─────────────────────────────────────────────────────
    step("Recipes tab: cards render + click first card")
    page.goto(f"{BASE}/data-recipes")
    page.wait_for_timeout(1500)
    headings = page.locator("main h2, main h3, [data-recipe], a[href*='/data-recipes/']")
    n_cards = headings.count()
    info(f"Recipes route headings/cards: {n_cards}")
    shoot("15b-recipes-cards")
    if n_cards > 0:
        try:
            headings.first.scroll_into_view_if_needed()
            headings.first.click()
            page.wait_for_timeout(1200)
            shoot("15c-recipes-first-card")
            info("OK clicked first recipe card")
        except Exception as exc:
            info(f"WARN click first recipe failed: {exc!r}")
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    step("Recents: click previous chat in sidebar")
    # The persisted thread title is usually a snippet of the first user message, so accept any of our prompt keywords.
    PROMPT_KEYWORDS = ("hello", "world", "tree", "yes", "1+1", "2+2")
    # misbehaving selector can't blow up wallclock.
    # Use the structural data-testid (thread-sidebar.tsx):
    threads = page.locator('[data-testid="recent-thread"]')
    deadline = time.monotonic() + 30
    clicked_recent = False
    try:
        threads.first.wait_for(state = "visible", timeout = 5_000)
    except Exception as _wait_err:
        info(f"WARN no recent-thread testid surfaced within 5s: {_wait_err!s}")
    n_threads = threads.count()
    for i in range(min(n_threads, 5)):
        if time.monotonic() > deadline:
            break
        try:
            t = (threads.nth(i).text_content() or "").strip()
            threads.nth(i).scroll_into_view_if_needed()
            threads.nth(i).click(timeout = 5_000)
            page.wait_for_timeout(500)
            shoot("15d-recent-clicked")
            info(f"OK clicked recent entry: {t[:60]!r}")
            # The landed thread must include at least one of our prompts.
            turns_text = robust_evaluate(
                page,
                """() => {
                const els = document.querySelectorAll(
                    '[data-role="user"], [data-role="assistant"]'
                );
                return Array.from(els).map(e => (e.innerText || '')
                    .toLowerCase()).join(' ');
            }""",
            )
            clicked_recent = True
            if any(k in turns_text for k in PROMPT_KEYWORDS):
                info("OK landed on a thread that includes our prompts")
                break
            else:
                soft_fail(
                    "Recents-clicked thread doesn't contain any of our "
                    f"sent prompts; turns_text={turns_text[:120]!r}"
                )
                break
        except Exception as _click_err:
            info(f"recent-thread click {i} failed: {_click_err!s}")
            continue
    if not clicked_recent:
        soft_fail(f"no Recents entry was clickable within 30s deadline (n_threads={n_threads})")
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # Image attachment UI reachable.
    # ─────────────────────────────────────────────────────
    step("attachment widget reachable")
    attach = page.locator('button[aria-label="Add Attachment"]').first
    if attach.count() > 0:
        # Only hover -- clicking would block on the native file dialog.
        attach.hover()
        page.wait_for_timeout(200)
        shoot("16-attachment-hover")

    # ─────────────────────────────────────────────────────
    step("reload + session survives")
    page.reload()
    composer.wait_for(state = "visible", timeout = 60_000)
    if "/login" in page.url:
        fail(f"unexpected redirect to /login after reload: {page.url}")
    shoot("17-after-reload")

    # ─────────────────────────────────────────────────────
    health = evaluate_fetch(
        page,
        f"{BASE}/api/health",
        timeout_ms = FETCH_TIMEOUT_MS,
    )
    if health.get("error"):
        fail(f"/api/health wedged: {health['error']!r}")
    if health["status"] != 200:
        fail(f"/api/health returned {health['status']}")

    # ─────────────────────────────────────────────────────
    step("post-rotation auth check (after UI change-password)")
    if (s_old := login_via_api(OLD)) != 401:
        fail(f"old bootstrap pw should be 401, got {s_old}")
    if (s_new := login_via_api(NEW)) != 200:
        fail(f"rotated pw should be 200, got {s_new}")
    info("OK old=401, new=200")

    # 16.
    # ─────────────────────────────────────────────────────
    step("rotate password via subprocess(curl) -- the 'terminal' path")
    login_proc = subprocess.run(
        [
            "curl",
            "-fsS",
            "-X",
            "POST",
            f"{BASE}/api/auth/login",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"username": "unsloth", "password": NEW}),
        ],
        capture_output = True,
        text = True,
        timeout = 15,
    )
    if login_proc.returncode != 0:
        fail(f"curl login failed: {login_proc.stderr!r}")
    login_body = json.loads(login_proc.stdout)
    cli_token = login_body.get("access_token")
    if not cli_token:
        fail(f"curl login returned no access_token: {login_body!r}")
    info("CLI obtained an access token")

    browser_refresh_token = robust_evaluate(
        page,
        "() => localStorage.getItem('unsloth_auth_refresh_token')",
    )
    if not browser_refresh_token:
        fail("browser refresh token missing before CLI rotation")

    change_proc = subprocess.run(
        [
            "curl",
            "-fsS",
            "-X",
            "POST",
            f"{BASE}/api/auth/change-password",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {cli_token}",
            "-d",
            json.dumps({"current_password": NEW, "new_password": NEW2}),
        ],
        capture_output = True,
        text = True,
        timeout = 15,
    )
    if change_proc.returncode != 0:
        fail(
            f"curl change-password failed: rc={change_proc.returncode} "
            f"stderr={change_proc.stderr!r} stdout={change_proc.stdout!r}"
        )
    info("CLI rotated password NEW -> NEW2 successfully")

    if (s_new1 := login_via_api(NEW)) != 401:
        fail(f"after CLI rotation, NEW pw should be 401, got {s_new1}")
    if (s_new2 := login_via_api(NEW2)) != 200:
        fail(f"after CLI rotation, NEW2 pw should be 200, got {s_new2}")
    info("OK after CLI rotation: NEW=401, NEW2=200 -- old studio creds dead")

    # /change-password revoked refresh tokens server-side (auth.py), so the browser's /api/auth/refresh must now fail.
    refresh_proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-o",
            os.devnull,
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            f"{BASE}/api/auth/refresh",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"refresh_token": browser_refresh_token}),
        ],
        capture_output = True,
        text = True,
        timeout = 15,
    )
    if refresh_proc.returncode != 0:
        fail(
            f"curl refresh-token check failed: rc={refresh_proc.returncode} "
            f"stderr={refresh_proc.stderr!r} stdout={refresh_proc.stdout!r}"
        )
    try:
        refresh_status = int(refresh_proc.stdout.strip())
    except ValueError:
        fail(f"curl refresh-token check returned invalid status: {refresh_proc.stdout!r}")
    if refresh_status == 200:
        fail(f"/api/auth/refresh should fail after CLI rotation; got 200")
    info(
        f"OK browser /api/auth/refresh now {refresh_status} "
        "(refresh token revoked) -- old studio session can no longer renew"
    )

    # Persisted monitor auth boundary, then shutdown.
    # ─────────────────────────────────────────────────────
    step("persisted monitor stays dormant on /login and resumes after auth")
    try:
        ctx.clear_cookies()
    except Exception as exc:
        info(f"WARN clearing stale session cookies failed: {exc!r}")
    robust_evaluate(
        page,
        """() => localStorage.setItem(
            "unsloth_monitor_overlay",
            JSON.stringify({ state: { isOpen: true, isMinimized: false }, version: 0 })
        )""",
    )
    try:
        page.evaluate(
            "['unsloth_auth_token', 'unsloth_auth_refresh_token']"
            ".forEach((key) => localStorage.removeItem(key))"
        )
    except Exception as exc:
        info(f"WARN clearing stale auth tokens failed: {exc!r}")
    _fresh_page = new_throttled_page(ctx)
    _fresh_page.on("pageerror", lambda e: page_errors.append(str(e)))
    _fresh_page.on("console", _on_console)
    try:
        page.close()
    except Exception:
        pass
    page = _fresh_page
    login_system_request_count = len(system_requests)

    # Re-login with NEW2 for a valid /api/shutdown token.
    _tolerated_nav = ("ERR_ABORTED", "interrupted by another navigation")
    # A slow CI runner can make this re-login navigation time out even with the server healthy, so retry the whole
    # goto/wait/fill/submit sequence (mirrors the change-password retry above).
    # wait_for_health is a diagnostic pre-gate.
    wait_for_health(BASE, timeout = 30.0, info = info)
    relogin_err: Exception | None = None
    for _relogin_attempt in range(3):
        # force=True bypasses the actionability check:
        try:
            # Pin the token, not a range: one spanning every preference (12 to 20, so 11.625px to 19.375px) also admits
            # the neighbouring tokens.
            try:
                page.goto(f"{BASE}/login", wait_until = "domcontentloaded", timeout = 60_000)
            except Exception as exc:
                if not any(t in str(exc) for t in _tolerated_nav):
                    raise
                info(f"goto /login interrupted ({exc!r}); password-field wait will confirm /login")
            pw_field = page.locator("#password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            page.keyboard.press("Control+,")
            page.wait_for_timeout(5_500)
            if len(system_requests) != login_system_request_count:
                raise AssertionError(
                    "persisted monitor requested /api/system while /login was active"
                )
            if "/login" not in page.url:
                raise AssertionError(f"login route reloaded or redirected unexpectedly: {page.url}")
            pw_field.fill(NEW2)
            # Wait on the login POST so a transient 4xx/5xx is caught and retried here, not swallowed until the
            status, _ = click_and_wait_for_response(
                page,
                url_substr = "/api/auth/login",
                method = "POST",
                do_click = lambda: page.locator('button[type="submit"]').click(),
                timeout_ms = 30_000,
                info = lambda m: print(f"[ui]   {m}", flush = True),
            )
            if status is not None and status >= 400:
                raise AssertionError(
                    f"login POST returned {status}; see console_errors={console_errors[:1]!r}"
                )
            relogin_err = None
            break
        except Exception as e:
            relogin_err = e
            try:
                cur_url = page.url
            except Exception:
                cur_url = "<page closed>"
            print(
                f"[ui]   re-login attempt {_relogin_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                f"page_errors={len(page_errors)} console_errors={len(console_errors)}",
                flush = True,
            )
            if console_errors:
                print(
                    f"[ui]   first console.error: {console_errors[0][:200]!r}",
                    flush = True,
                )
            if page_errors:
                print(f"[ui]   first pageerror:    {page_errors[0][:200]!r}", flush = True)
            try:
                shoot(f"18-relogin-attempt-{_relogin_attempt + 1}-fail")
            except Exception:
                pass
            if _relogin_attempt < 2:
                # ERR_NO_BUFFER_SPACE needs the OS to recover socket buffers;
                if "ERR_NO_BUFFER_SPACE" in str(e):
                    backoff_s = 5 if _relogin_attempt == 0 else 15
                    print(
                        f"[ui]   ENOBUFS detected; sleeping {backoff_s}s "
                        f"before retry to let OS recover socket buffers...",
                        flush = True,
                    )
                    time.sleep(backoff_s)
                # Replace the page if it died;
                old_page = page
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ui]   recovery: {m}", flush = True),
                )
                # A freshly created replacement page loses the pageerror/console listeners;
                if page is not old_page:
                    page.on("pageerror", lambda e: page_errors.append(str(e)))
                    page.on("console", _on_console)
    if relogin_err is not None:
        raise relogin_err
    # Composer mount confirms the rotated session is authenticated.
    # Kept OUTSIDE the retry: the loop breaks right after submit, so we never re-goto /login once login has set tokens
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)
    monitor_deadline = time.time() + 10
    while len(system_requests) == login_system_request_count and time.time() < monitor_deadline:
        page.wait_for_timeout(100)
    if len(system_requests) == login_system_request_count:
        fail("persisted monitor did not resume /api/system polling after login")
    if page.get_by_role("dialog", name = re.compile(r"^Settings$")).count() != 0:
        fail("settings shortcut on /login left the dialog open after authentication")
    info("OK persisted monitor stayed dormant on /login and resumed after authentication")

    exercise_floating_monitor_geometry(page)

    shoot("18-relogin-with-NEW2")

    step("Shutdown via account menu")
    acct_btn = page.locator('button[aria-label$=" account menu"]').first
    if acct_btn.count() == 0:
        fail("account menu button missing -- can't reach Shutdown")
    acct_btn.click()
    page.wait_for_timeout(400)
    shutdown_item = page.get_by_role(
        "menuitem",
        name = re.compile(r"^\s*Shutdown\s*$", re.I),
    ).first
    if shutdown_item.count() == 0:
        fail("Shutdown menuitem not in account menu")
    shutdown_item.click()
    shoot("19-shutdown-dialog")
    stop_btn = page.get_by_role(
        "button",
        name = re.compile(r"^\s*Stop server\s*$", re.I),
    ).first
    stop_btn.wait_for(state = "visible", timeout = 5_000)
    stop_btn.click()

    # Start fresh after the CLI rotation invalidates this browser session.
    # Stay in the SAME context: it keeps the init script and costs nothing to reuse.
    try:
        page.wait_for_function(
            """() => /Unsloth has stopped/.test(document.body.innerText)""",
            timeout = 15_000,
        )
        shoot("20-shutdown-placeholder")
        info("OK 'Unsloth has stopped' placeholder rendered")
    except Exception as exc:
        info(f"WARN shutdown placeholder didn't render: {exc!r}")

    # /api/health must now be unreachable;
    # poll for up to 15s.
    host = re.sub(r"^https?://", "", BASE).split(":")[0]
    port = int(re.search(r":(\d+)", BASE).group(1)) if ":" in BASE else 80
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout = 1):
                pass
            time.sleep(0.5)
        except (ConnectionRefusedError, OSError):
            info("OK port closed -- server process is gone")
            break
    else:
        # Connection still works -> shutdown didn't take effect.
        try:
            r = urllib.request.urlopen(f"{BASE}/api/health", timeout = 2)
            fail(f"server still up after Shutdown click; /api/health={r.status}")
        except urllib.error.URLError as exc:
            info(f"OK /api/health unreachable: {exc!r}")

    # Some pageerrors are benign: chat-completions 422s (network-layer bubble-up, not a JS bug;
    # Full list in `_playwright_robust.BENIGN_PAGE_ERROR_PATTERNS`.
    real_errors = [e for e in page_errors if not is_benign_page_error(e)]
    real_console_errors = [e for e in console_errors if not is_benign_console_error(e)]
    if page_errors:
        info(
            f"WARN page errors: {len(page_errors)} total "
            f"({len(real_errors)} non-benign); first: {page_errors[0]!r}"
        )
    if real_errors:
        fail(f"{len(real_errors)} non-benign pageerror events")
    info(
        f"console.error events: {len(console_errors)} total ({len(real_console_errors)} non-benign)"
    )

    info("PASS comprehensive UI flow")
    _watchdog.cancel()
    browser.close()
