# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth extra-UI Playwright test: Compare tab, Recipes editor, /export, /studio, Settings tabs."""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

# Run as a plain script (not via pytest), so prepend the dir to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    evaluate_fetch,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    recover_or_replace_page,
    robust_evaluate,
    wait_for_first,
    wait_for_health,
    click_forced,
)

BASE = os.environ["BASE_URL"]
OLD = os.environ["STUDIO_OLD_PW"]
NEW = os.environ.get("STUDIO_NEW_PW", "ExtraUi-NEW-2026!")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_extra")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
# The Voice-picker media-access crash is specific to headless Chromium on macos-14; only there is a renderer crash
# downgraded to a warning. Linux/Windows keep hard crash coverage.
MACOS_RUNNER = os.environ.get("RUNNER_OS", "").lower() == "macos" or sys.platform == "darwin"
# Longer turn timeout: gemma-3-270m CPU inference is 3-5x slower on macos-14 runners.
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
LOAD_FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_LOAD_TIMEOUT_MS", "180000"))
# Declares a runner with no route to the Hub, for the voice-picker wheel step. Egress that is blackholed rather than
# refused can leave the search hanging with no transport failure to observe, so the fallback needs a way to be asserted
# as well as detected.
HF_OFFLINE = os.environ.get("STUDIO_UI_HF_OFFLINE", "0") == "1"
# Voice-picker wheel budget, both halves set by the frontend rather than picked round.
# The searched rows are up to 15.3s away on a healthy runner: the query is debounced 300ms and the Hugging Face search
# is then given 15s (HF_SEARCH_TIMEOUT_MS, studio/frontend/src/features/hub/hooks/use-hub-model-search.ts). Waiting
# 15.5s for them clears that, so a slow-but-working Hub is not red, and it also outlives the abort at 15.3s that a
# blackholed runner's search ends in, so the transport failure that permits the fallback is observed before the wait
# gives up.
# Those 200ms of headroom only hold while the debounce fires on time, so when the wait runs out with the search still
# open the budget is re-based onto the request itself (search_abort_extension) rather than spent.
# The 30s ceiling is that wait plus the re-basing a starved runner needs, plus what is left to do after it: a list swap
# landing mid-wheel costs one 2s wheel round, and an unreachable-Hub run has already spent its first 15.5s searching
# when it clears the query and wheels the built-in list. Only a failing run pays either; a passing run leaves on the
# first wheel, 1.5s end to end in CI.
WHEEL_ROWS_TIMEOUT_MS = 15_500
WHEEL_DEADLINE_S = 30.0
# The ceiling the deadline may be pushed to when the wait is re-based onto a request that is still open.
# WHEEL_DEADLINE_S covers one search, and the picker runs two in sequence, so re-basing onto the second has to be
# allowed to outlast it. Measured from the step's start so a page that keeps opening requests cannot hold the step open
# indefinitely.
WHEEL_DEADLINE_MAX_S = 75.0

_n = [0]
_failed: list[str] = []


def step(s: str) -> None:
    print(f"[ui-extra] STEP {s}", flush = True)


def info(s: str) -> None:
    print(f"[ui-extra] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[ui-extra] FAIL: {m}", flush = True)
    _failed.append(m)


def soft_fail(m: str) -> None:
    if STRICT:
        fail(m)
    else:
        info(f"WARN (strict-off): {m}")


def runtime_warn(m: str) -> None:
    """Warn about a runtime-coupled assertion (Compare-pane streaming) that STRICT does not gate."""
    info(f"WARN (runtime): {m}")


def page_crashed(pg, exc: Exception) -> bool:
    """True when the browser/page/context died (a macos-14 renderer crash) rather than a live-page
    assertion failing -- so the caller can downgrade CI-environment flakiness to a runtime warning."""
    try:
        if pg.is_closed():
            return True
    except Exception:
        return True
    msg = str(exc).lower()
    return "has been closed" in msg or "target closed" in msg or "crash" in msg


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ui-extra",
        info = info,
    )
    # Health pre-flight: bash-side health wait can pass before the auth DB migrates on macos-14.
    wait_for_health(BASE, timeout = 30.0, info = info)
    # Chromium launch args: see tests/studio/_playwright_robust.py.
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    install_view_transition_killer(ctx)

    # Evidence that this runner cannot reach the Hub, collected for the whole session because the frontend backs off for
    # 30s after a failed Hub request (REMOTE_OFFLINE_TTL_MS in studio/frontend/src/features/hub/lib/network.ts) and may
    # not retry inside a later step. Bound to the context, not the page, so a replacement page is covered too.
    hf_unreachable: list[str] = []
    # Set while the wheel step owns the picker, so an aborted Hub request can be attributed.
    wheel_step_active = [False]
    # Hub requests still in flight, by start time, so a wait that runs out while the frontend's own search timeout is
    # still running can wait for its abort instead of guessing.
    hf_inflight: dict[object, float] = {}

    def _is_hub_url(url: str) -> bool:
        """Only the origin the picker itself queries counts as Hub connectivity.

        A substring test also matches datasets-server.huggingface.co, which the training
        split lookup calls. The frontend keys its backoff by exact origin
        (HUGGING_FACE_ORIGIN in studio/frontend/src/features/hub/lib/network.ts), so a
        failure at a sibling host says nothing about the picker's search, and counting it
        would let an unrelated lookup hand a real search regression the built-in list.
        """
        try:
            return urllib.parse.urlsplit(url).netloc.lower() == "huggingface.co"
        except Exception:
            return False

    def _note_hf_unreachable(why: str) -> None:
        if not hf_unreachable:
            info(f"WARN Hugging Face unreachable from this runner: {why}")
        hf_unreachable.append(why)

    def _on_request(req) -> None:
        try:
            if _is_hub_url(req.url):
                hf_inflight[req] = time.monotonic()
        except Exception:
            pass

    def _on_requestfailed(req) -> None:
        try:
            hf_inflight.pop(req, None)
            if not _is_hub_url(req.url):
                return
            failure = req.failure or ""
            # net::ERR_ABORTED is how a blackholed request ends, at the frontend's own 15s search timeout, and equally
            # how a superseded query or an unmounting picker ends. It only says "unreachable" while the wheel step holds
            # the picker open on a single query, where nothing else can be cancelling anything. Every other failure is a
            # transport error and counts wherever it happens.
            if "ERR_ABORTED" in failure and not wheel_step_active[0]:
                return
            _note_hf_unreachable(f"request failed: {failure}")
        except Exception:
            pass

    def _on_requestfinished(req) -> None:
        try:
            hf_inflight.pop(req, None)
        except Exception:
            pass

    def _on_response(resp) -> None:
        # 429 and 5xx are the Hub refusing to serve this runner. Every other 4xx is a request the app itself built
        # wrong, which is a real defect and must not excuse anything.
        try:
            if not _is_hub_url(resp.url):
                return
            if resp.status == 429 or resp.status >= 500:
                _note_hf_unreachable(f"HTTP {resp.status}")
            elif hf_unreachable:
                # A served response proves this runner has a route to the Hub, so the earlier failures are stale and
                # must stop excusing anything: the frontend drops its own offline state on exactly this signal
                # (markRemoteNetworkOnline, studio/frontend/src/features/hub/lib/network.ts). Keeping them would let one
                # transient failure hand a later search-rendering regression the built-in list.
                info(
                    f"Hugging Face reachable again (HTTP {resp.status}); dropping "
                    f"{len(hf_unreachable)} earlier failure(s)"
                )
                hf_unreachable.clear()
        except Exception:
            pass

    ctx.on("request", _on_request)
    ctx.on("requestfailed", _on_requestfailed)
    ctx.on("requestfinished", _on_requestfinished)
    ctx.on("response", _on_response)
    page = ctx.new_page()
    # 60s default for the slow macos-14 runner (second Unsloth boot of the job).
    page.set_default_timeout(60_000)
    page_errors = []

    # Filter known-benign React errors (slow-CI timing artefacts); base list in _playwright_robust.
    def _on_pageerror(e):
        msg = str(e)
        if is_benign_page_error(msg):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    def shoot(name: str) -> None:
        # Screenshots are diagnostic; never fail the test on a font-load timeout.
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

    # Setup: change-password through the UI + model load.
    step("setup: change-password + model load")
    # 3-attempt retry: form re-renders mid-fill on macos-14 can detach the password fields.
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
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            # Click submit AND wait for the POST response together so a server-side reject surfaces now.
            status, _ = click_and_wait_for_response(
                page,
                url_substr = "/api/auth/change-password",
                method = "POST",
                do_click = lambda: page.locator('button[type="submit"]').click(),
                timeout_ms = 30_000,
                info = lambda m: print(f"[ui-extra]   {m}", flush = True),
            )
            if status is not None and status >= 400:
                raise AssertionError(
                    f"change-password POST returned {status}; see page_errors={page_errors[:1]!r}"
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
                f"[extra-ui]   change-password form attempt {_form_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                f"page_errors={len(page_errors)}",
                flush = True,
            )
            if _form_attempt < 2:
                # ERR_NO_BUFFER_SPACE needs the OS to recover socket buffers; back off 5s then 15s.
                if "ERR_NO_BUFFER_SPACE" in str(e):
                    backoff_s = 5 if _form_attempt == 0 else 15
                    print(
                        f"[extra-ui]   ENOBUFS detected; sleeping {backoff_s}s "
                        f"before retry to let OS recover socket buffers...",
                        flush = True,
                    )
                    time.sleep(backoff_s)
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[extra-ui]   recovery: {m}", flush = True),
                )
    if form_err is not None:
        raise form_err
    # Settle network, then wait_for with one recovery cycle: the post-submit re-render can crash macos-14.
    try:
        page.wait_for_load_state("networkidle", timeout = 30_000)
    except Exception:
        pass
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
                f"[extra-ui]   composer.wait_for attempt {_attempt + 1} failed: "
                f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                f"page_errors={len(page_errors)}",
                flush = True,
            )
            try:
                shoot(f"01-composer-wait-attempt-{_attempt + 1}-fail")
            except Exception:
                pass
            if _attempt == 0:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    goto_url = BASE,
                    settle_networkidle = True,
                    info = lambda m: print(f"[extra-ui]   recovery: {m}", flush = True),
                )
                composer = page.locator('textarea[aria-label="Message input"]')
    if last_err is not None:
        raise last_err
    shoot("01-chat-loaded")

    token = robust_evaluate(page, "() => localStorage.getItem('unsloth_auth_token')")
    if not token:
        fail("no access token after change-password")
        sys.exit(1)
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
        sys.exit(1)
    if load_resp["status"] != 200:
        fail(f"/api/inference/load -> {load_resp['status']}: {load_resp.get('body')!r}")
        sys.exit(1)
    info(f"loaded model: {(load_resp['body'] or {}).get('display_name')}")
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # Detect chat-only mode (/api/health.chat_only): /studio redirects to /chat while /export stays reachable and
    # self-gated.
    health_resp = evaluate_fetch(
        page,
        f"{BASE}/api/health",
        timeout_ms = FETCH_TIMEOUT_MS,
    )
    if health_resp.get("error"):
        fail(f"/api/health wedged: {health_resp['error']!r}")
        sys.exit(1)
    health = health_resp.get("body") or {}
    chat_only = bool(health.get("chat_only"))
    info(f"chat_only mode: {chat_only}")

    # 1. Compare tab.
    step("Compare tab: send to two panes")
    # Compare lives in the composer "Tools and attachments" menu.
    compare_opened = False
    # Waited for, not counted. This is the first step after load, so it is the one
    # that pays for anything slowing first paint: #9251's reload snapshot overlay
    # opened a window where the composer is on screen but not yet in the
    # accessibility tree, and `count()` answered 0 six milliseconds in and called
    # it "Compare nav not found". See wait_for_first().
    plus_btn = wait_for_first(
        page.get_by_role("button", name = re.compile(r"Tools and attachments", re.I))
    )
    if plus_btn is not None:
        click_forced(plus_btn)
        # The menu items get a short wait rather than the full one: a miss here is a real branch (the item lives under
        # "More"), not a slow render, and the fallbacks below must stay quick.
        compare_item = wait_for_first(
            page.get_by_role("menuitem", name = re.compile(r"Compare chat", re.I)),
            timeout_ms = 2000,
        )
        if compare_item is None:
            # Fallback: Compare chat may be under the "More" submenu.
            more_trigger = wait_for_first(
                page.get_by_role("menuitem", name = re.compile(r"^More$", re.I)),
                timeout_ms = 2000,
            )
            if more_trigger is not None:
                more_trigger.hover()
                compare_item = wait_for_first(
                    page.get_by_role("menuitem", name = re.compile(r"Compare chat", re.I)),
                    timeout_ms = 2000,
                )
                if compare_item is None:
                    click_forced(more_trigger)
                    compare_item = wait_for_first(
                        page.get_by_role("menuitem", name = re.compile(r"Compare chat", re.I)),
                        timeout_ms = 2000,
                    )
        if compare_item is not None:
            click_forced(compare_item)
            compare_opened = True
    if not compare_opened:
        # Which of the two was missing, because "Compare nav not found" sent the
        # last reader looking for a removed menu item that was never removed.
        missing = (
            "the composer's Tools and attachments button"
            if plus_btn is None
            else "the Compare chat menu item"
        )
        soft_fail(f"Compare nav not found: {missing} never appeared")
    else:
        page.wait_for_timeout(1500)
        shoot("02-compare-opened")
        view = page.locator('[data-tour="chat-compare-view"]').first
        if view.count() == 0:
            soft_fail("[data-tour='chat-compare-view'] not found after Compare click")
        else:
            ok_count_before = len(page.locator('[data-role="assistant"]').all())
            # Composer placeholder in compare-mode is "Send to both models...".
            cmp_composer = page.get_by_placeholder(
                re.compile(r"Send to both models", re.I),
            ).first
            if cmp_composer.count() == 0:
                # Fall back to any textarea inside the compare view.
                cmp_composer = view.locator("textarea").first
            if cmp_composer.count() == 0:
                soft_fail("compare composer textarea not found")
            else:
                cmp_composer.click()
                cmp_composer.fill("Reply with: A")
                # Prefer Enter: onKeyDown maps plain Enter to send(); the Send button's aria-label came late.
                cmp_composer.press("Enter")
                # Expect 2 new assistant bubbles (one per pane). Panes have no explicit model in this CI flow so the
                # backend may reject; downgrade to runtime_warn but keep the structural assertions.
                try:
                    page.wait_for_function(
                        """(want) => {
                            return document.querySelectorAll(
                                '[data-role="assistant"]'
                            ).length >= want;
                        }""",
                        arg = ok_count_before + 2,
                        timeout = 60_000,
                    )
                    info("OK Compare: 2 new assistant bubbles after first prompt")
                except Exception as exc:
                    runtime_warn(
                        f"Compare: 2 bubbles didn't appear (panes likely "
                        f"have no model selected): {exc!r}"
                    )
                shoot("03-compare-after-A")

                # Second prompt -> 4 total new bubbles (same runtime-flaky caveat).
                cmp_composer.fill("Reply with: B")
                cmp_composer.press("Enter")
                try:
                    page.wait_for_function(
                        """(want) => {
                            return document.querySelectorAll(
                                '[data-role="assistant"]'
                            ).length >= want;
                        }""",
                        arg = ok_count_before + 4,
                        timeout = 60_000,
                    )
                    info("OK Compare: 4 total new assistant bubbles after second prompt")
                except Exception as exc:
                    runtime_warn(
                        f"Compare: 4 bubbles didn't appear (panes likely "
                        f"have no model selected): {exc!r}"
                    )
                shoot("04-compare-after-B")

    # Back to single chat for subsequent steps.
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # 2. Recipes editor.
    step("Recipes editor: click first template + Preview dialog")
    page.goto(f"{BASE}/data-recipes")
    page.wait_for_timeout(1500)
    shoot("05-recipes-list")
    # Template cards render as <button> elements.
    templates = page.locator("main button").filter(has_not_text = re.compile(r"^(\+|Create)"))
    n_templates = templates.count()
    info(f"recipe templates visible: {n_templates}")
    if n_templates == 0:
        soft_fail("no recipe template cards found")
    else:
        try:
            templates.first.scroll_into_view_if_needed()
            templates.first.click()
            page.wait_for_timeout(2000)
            shoot("06-recipe-opened")
            # The recipe-studio canvas uses React-Flow; look for the renderer.
            canvas = page.locator(
                ".react-flow__renderer, .react-flow, [data-testid*='react-flow']"
            ).first
            if canvas.count() == 0:
                # Some templates open as dialogs instead of a route.
                info("(no React-Flow canvas; template may have opened a dialog)")
            else:
                info("OK React-Flow canvas mounted")
        except Exception as exc:
            soft_fail(f"recipe template click failed: {exc!r}")

    # 3. Export route.
    step(f"Export route ({'chat-only self-gated' if chat_only else 'form fields'})")
    page.goto(f"{BASE}/export")
    page.wait_for_timeout(1500)
    shoot("07-export")
    if chat_only:
        if "/export" not in page.url:
            soft_fail(f"chat-only mode should keep /export reachable; url={page.url}")
        else:
            unavailable = page.get_by_text(re.compile(r"Export unavailable", re.I)).first
            if unavailable.count() == 0:
                soft_fail("chat-only /export did not show the export unavailable gate")
            else:
                info("OK chat-only /export rendered the unavailable gate")
    else:
        # Non-chat-only: verify the export-cta button + HF token field.
        cta = page.locator('[data-tour="export-cta"]').first
        if cta.count() == 0:
            soft_fail("[data-tour='export-cta'] not found in /export")
        else:
            info("OK [data-tour='export-cta'] visible")
        # HF-token field is lazy-loaded behind a disclosure; poll for ~8s and log at info (non-blocking).
        hf_token = None
        for _try in range(8):
            page.wait_for_timeout(1000)
            for cand in (
                page.get_by_placeholder(re.compile(r"hf[_\\.\\-]", re.I)).first,
                page.locator(
                    'input[placeholder*="token" i], input[placeholder*="huggingface" i]'
                ).first,
                page.locator('input[name="hf_token"], input[id*="hf-token"]').first,
            ):
                if cand.count() > 0:
                    hf_token = cand
                    break
            if hf_token is not None:
                break
        if hf_token is not None:
            info("OK HF token input visible")
        else:
            info(
                "WARN HF token input not located in /export after 8s "
                "(likely lazy-loaded behind a disclosure section -- "
                "non-blocking for upload flow)"
            )

    # 4. Unsloth training route.
    step(f"Unsloth route ({'chat-only redirect' if chat_only else 'tabs + sections'})")
    page.goto(f"{BASE}/studio")
    page.wait_for_timeout(1500)
    shoot("08-studio")
    if chat_only:
        if "/studio" in page.url:
            soft_fail(f"chat-only mode should redirect /studio -> /chat; url={page.url}")
        else:
            info(f"OK chat-only redirected /studio -> {page.url}")
    else:
        for tab_name in ("Configure", "Current run", "History"):
            tab = page.get_by_role("tab", name = re.compile(rf"^\s*{tab_name}\s*$", re.I)).first
            if tab.count() == 0:
                soft_fail(f"tab '{tab_name}' not found in /studio")
            else:
                info(f"OK tab '{tab_name}' visible")
        for anchor in ("studio-model-picker", "studio-dataset", "studio-params"):
            el = page.locator(f'[data-tour="{anchor}"]').first
            if el.count() == 0:
                soft_fail(f"[data-tour='{anchor}'] not found")
            else:
                info(f"OK [data-tour='{anchor}'] visible")

    # 5. Settings dialog tabs.
    step("Settings dialog: cycle through tabs")
    page.goto(f"{BASE}/chat")
    composer.wait_for(state = "visible", timeout = 60_000)
    dictate = page.get_by_role("button", name = "Dictate").first
    if dictate.count() == 0:
        fail("Chat Dictate button not found")
    elif dictate.get_attribute("type") != "button":
        fail("Chat Dictate control must use type=button, not submit the composer")
    else:
        info("OK Chat Dictate control is type=button")

    page.keyboard.press("Control+,")
    page.wait_for_timeout(800)
    settings = page.get_by_role("dialog").first
    if settings.count() == 0:
        # macOS shortcut is Cmd-,.
        page.keyboard.press("Meta+,")
        page.wait_for_timeout(800)
        settings = page.get_by_role("dialog").first
    if settings.count() == 0:
        soft_fail("Settings dialog didn't open with Cmd/Ctrl-,")
    else:
        shoot("09-settings-open")
        # Each tab is a button named by its visible text; availability depends on chat_only mode.
        candidate_tabs = (
            "General",
            "Profile",
            "Appearance",
            "Chat",
            "Developer",
            "Voice",
            "About",
        )
        seen_tabs = []
        for tab_name in candidate_tabs:
            btn = page.get_by_role(
                "button",
                name = re.compile(rf"^\s*{tab_name}\s*$", re.I),
            ).first
            if btn.count() == 0:
                continue
            try:
                btn.click()
                page.wait_for_timeout(400)
                # Tab body must be non-empty.
                body_text = page.evaluate(
                    """() => {
                        const dialog = document.querySelector('[role="dialog"]');
                        return dialog ? (dialog.innerText || '').trim().length : 0;
                    }"""
                )
                if body_text > 30:
                    info(f"OK Settings tab '{tab_name}' body length={body_text}")
                    seen_tabs.append(tab_name)
                else:
                    soft_fail(f"Settings tab '{tab_name}' body suspiciously short: {body_text}")
            except Exception as exc:
                soft_fail(f"Settings tab '{tab_name}' click failed: {exc!r}")
        step("Voice model picker: real mouse-wheel scrolling")
        # By test id: the tab label is translated.
        voice_tab = page.get_by_test_id("settings-tab-voice").first
        if voice_tab.count() == 0:
            fail("Voice settings tab not found")
        else:
            # The dictation-engine dropdown touches a media-access path that can crash headless Chromium on macos-14
            # (CheckMediaAccessPermission), so there a crash is a runtime warning + page recovery; on Linux/Windows a
            # crash and any live-page failure stay a hard fail.
            try:
                voice_tab.click()
                # By test id: these were bound to translated copy, which caused #7835.
                page.get_by_test_id("dictation-engine-trigger").click()
                page.get_by_test_id("dictation-engine-model").click()
                page.get_by_test_id("stt-model-trigger").click()
                wheel_step_active[0] = True
                results = page.get_by_test_id("stt-model-results")
                # Wheel at the searched rows, not at whatever overflows first. The query is debounced 300ms and the
                # list is then replaced by a one-line spinner for as long as the Hugging Face search takes, so the
                # first paint that overflows is the pre-search built-in list: on macos-15 the hover + wheel lands after
                # the swap, on a container that is one spinner row tall and has nothing to scroll. Requiring rendered
                # model rows (the loading and empty states are plain divs, every row is a button) pins the assertion to
                # the state a user scrolls, and re-wheeling until the deadline absorbs a swap that lands mid-wheel.
                #
                # Rows alone are not enough, though: the built-in list is rows, and it overflows from the moment the
                # popover opens, so a fast runner can satisfy that inside the 300ms debounce and never wheel a searched
                # row at all. Snapshot the built-in rows first and require the list to have become something else, so
                # the search is what is being scrolled.
                # The built-in list is accepted only on proof that the Hub is unreachable (see below), which is also the
                # only branch that clears the query and so the only one that waits on `rows_overflow`.
                builtin_rows_js = """() => {
                    const node = document.querySelector('[data-testid="stt-model-results"]');
                    if (!node) return "";
                    return Array.from(node.querySelectorAll('button'))
                        .map((row) => row.innerText).join("\\u0000");
                }"""
                try:
                    results.locator("button").first.wait_for(state = "attached", timeout = 10_000)
                except Exception as builtin_err:
                    info(f"WARN built-in model rows never rendered: {builtin_err!r}")
                builtin_rows = robust_evaluate(page, builtin_rows_js)
                page.get_by_test_id("stt-model-search").fill("whisper")
                query_typed_at = time.monotonic()
                searched_rows_overflow = """(builtin) => {
                    const node = document.querySelector('[data-testid="stt-model-results"]');
                    if (!node || node.scrollHeight <= node.clientHeight) return false;
                    const rows = Array.from(node.querySelectorAll('button'));
                    if (rows.length === 0) return false;
                    return rows.map((row) => row.innerText).join("\\u0000") !== builtin;
                }"""
                rows_overflow = """() => {
                    const node = document.querySelector('[data-testid="stt-model-results"]');
                    return !!node
                        && node.querySelectorAll('button').length > 0
                        && node.scrollHeight > node.clientHeight;
                }"""
                scrolled_js = """() => {
                    const node = document.querySelector('[data-testid="stt-model-results"]');
                    return !!node && node.scrollTop > 0;
                }"""
                wheel_started_at = time.monotonic()
                wheel_deadline = wheel_started_at + WHEEL_DEADLINE_S
                wheel_scrolled = False
                cleared_search = False
                extended_for: set = set()
                next_rows_ms = float(WHEEL_ROWS_TIMEOUT_MS)

                def search_abort_extension() -> tuple:
                    """How much of the frontend's own search timeout is still to run.

                    WHEEL_ROWS_TIMEOUT_MS is counted from `fill`, but the frontend starts its 15s
                    from the debounced request, which a CPU-starved runner can schedule well past
                    the nominal 300ms. While that request is in flight the abort that proves the
                    Hub unreachable has not happened yet, so the budget is re-based onto the
                    request rather than the step deciding the Hub is healthy without it.

                    Only requests issued after the query was typed count: an unrelated Hub
                    request left hanging from an earlier step started long ago and would anchor
                    the budget to a deadline that has already passed.

                    Returns the request as well, because the picker searches twice in sequence
                    (unsloth-owned, then general: mergedModelIterator in
                    studio/frontend/src/features/hub/hooks/use-hub-model-search.ts). A slow but
                    healthy first search can spend the extension, and the second then starts with
                    its own full budget, so the caller has to be able to re-base onto that one
                    rather than treat the step as already extended.
                    """
                    live = [(at, req) for req, at in hf_inflight.items() if at >= query_typed_at]
                    if not live:
                        return None, 0.0
                    started, req = min(live, key = lambda pair: pair[0])
                    return req, max(
                        0.0, (started - time.monotonic()) * 1000 + WHEEL_ROWS_TIMEOUT_MS
                    )

                while not wheel_scrolled:
                    remaining_ms = (wheel_deadline - time.monotonic()) * 1000
                    if remaining_ms <= 0:
                        break
                    rows_ms = min(remaining_ms, next_rows_ms)
                    next_rows_ms = float(WHEEL_ROWS_TIMEOUT_MS)
                    try:
                        if cleared_search:
                            page.wait_for_function(rows_overflow, timeout = rows_ms)
                        else:
                            page.wait_for_function(
                                searched_rows_overflow,
                                arg = builtin_rows,
                                timeout = rows_ms,
                            )
                    except Exception as row_err:
                        # A dead renderer must reach the crash handler below, exactly as in the wheel wait; swallowed
                        # here it becomes a hard "did not wheel-scroll".
                        if page_crashed(page, row_err):
                            raise
                        if not (cleared_search or hf_unreachable or HF_OFFLINE):
                            open_req, extra_ms = search_abort_extension()
                            if (
                                open_req is not None
                                and open_req not in extended_for
                                and extra_ms > 0
                            ):
                                extended_for.add(open_req)
                                next_rows_ms = extra_ms
                                # The extension is worth nothing if the step deadline still ends inside it: rows_ms is
                                # min()ed against what is left, so the second search would be cut off mid-flight and
                                # reported as a scroll failure. Push the deadline past the request just re-based onto,
                                # up to the ceiling.
                                wheel_deadline = min(
                                    wheel_started_at + WHEEL_DEADLINE_MAX_S,
                                    max(wheel_deadline, time.monotonic() + extra_ms / 1000),
                                )
                                info(
                                    "WARN search rows are not in and a Hugging Face request is "
                                    f"still open; waiting {extra_ms / 1000:.1f}s more for it to "
                                    "answer or abort"
                                )
                                continue
                        # Falling back to the built-in list means asserting the wheel against the pre-search list this
                        # step was rewritten to stop accepting, so it takes proof that the Hub is what is missing: a
                        # failed huggingface.co request (or 429/5xx), or a runner that declares itself offline. Search
                        # rendering that breaks with the Hub answering normally has no such proof and fails here with
                        # the geometry, instead of passing on the built-in list.
                        offline = bool(hf_unreachable) or HF_OFFLINE
                        if cleared_search or not offline:
                            break  # never overflowed with rows; geometry is reported below
                        why = hf_unreachable[0] if hf_unreachable else "STUDIO_UI_HF_OFFLINE=1"
                        info(
                            f"WARN no 'whisper' search rows and the Hub is unreachable ({why}); "
                            "wheeling the built-in list instead"
                        )
                        cleared_search = True
                        page.get_by_test_id("stt-model-search").fill("")
                        continue
                    results.hover()
                    page.mouse.wheel(0, 700)
                    try:
                        page.wait_for_function(scrolled_js, timeout = 2_000)
                    except Exception as wheel_err:
                        # A dead renderer must still reach the crash handler below, not be retried until the deadline
                        # and reported as a scroll failure.
                        if page_crashed(page, wheel_err):
                            raise
                        continue
                    wheel_scrolled = True
                if wheel_scrolled:
                    info("OK Voice model picker mouse wheel changed scrollTop")
                else:
                    # Geometry in the message: the next failure says whether the list was short, empty or
                    # scrollable-but-unscrolled without a second CI run.
                    try:
                        geom = robust_evaluate(
                            page,
                            """() => {
                                const node = document.querySelector('[data-testid="stt-model-results"]');
                                if (!node) return null;
                                return {
                                    scrollTop: node.scrollTop,
                                    scrollHeight: node.scrollHeight,
                                    clientHeight: node.clientHeight,
                                    rows: node.querySelectorAll('button').length,
                                };
                            }""",
                        )
                    except Exception as geom_err:
                        geom = f"<unreadable: {geom_err!r}>"
                    fail(
                        f"Voice model picker did not wheel-scroll: {geom} "
                        f"hub_unreachable={hf_unreachable[:1] or False} "
                        f"cleared_search={cleared_search}"
                    )
            except Exception as exc:
                if page_crashed(page, exc) and MACOS_RUNNER:
                    runtime_warn(f"Voice model picker aborted (browser/page unstable): {exc!r}")
                    page = recover_or_replace_page(
                        page,
                        ctx,
                        default_timeout_ms = 60_000,
                        info = lambda m: info(f"recovery: {m}"),
                    )
                else:
                    fail(f"Voice model picker did not wheel-scroll: {exc!r}")
            finally:
                wheel_step_active[0] = False
        # When the crash closed the context/browser, recover_or_replace_page hands back the closed page; skip the
        # cosmetic teardown rather than re-raise TargetClosedError on it.
        if not page.is_closed():
            shoot("10-settings-tabs-visited")
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        info(f"visited Settings tabs: {seen_tabs}")
        if not seen_tabs:
            soft_fail("no Settings tabs were visitable")

    # Done.
    if page_errors:
        info(f"WARN {len(page_errors)} pageerror events; first: {page_errors[0]!r}")
        fail(f"{len(page_errors)} pageerror events")

    if _failed:
        info(f"FAILED: {len(_failed)} assertion(s)")
        for m in _failed:
            info(f"  - {m}")
        sys.exit(1)
    info("PASS extra UI flow")
    _watchdog.cancel()
    try:
        browser.close()
    except Exception:
        pass  # a crashed browser may already be gone; never fail teardown after PASS
