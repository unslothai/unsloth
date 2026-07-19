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

# Tests run as plain `python tests/studio/playwright_chat_ui.py` (not
# via pytest/import), so prepend this dir to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    evaluate_fetch,
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_console_error,
    is_benign_page_error,
    recover_or_replace_page,
    robust_evaluate,
    wait_for_health,
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

# When on (default in CI), fail loudly on any missing button/nav/dialog
# instead of logging a WARN; off locally to run against a partial install.
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"

# Per-turn assistant-bubble wait. The free macos-14 runner is ~3-5x
# slower at gemma-3-270m CPU inference; this lets it bump the timeout.
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))

# Wall-clock cap for the whole script (healthy run is 5-9 min).
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))

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


def expected_default_model():
    override = os.environ.get("EXPECTED_DEFAULT_MODEL")
    if override:
        return override

    # Parse DEFAULT_MODELS_GGUF as a literal out of defaults.py instead of
    # importing it: the --no-torch Playwright install can't import the
    # inference package or defaults.py's hardware deps.
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
        tree = ast.parse(defaults_path.read_text())
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


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ui",
        info = info,
    )
    # Pre-flight: macos-14 can surface a 200 /api/health while the auth
    # DB is still migrating; this 30s probe catches that gap before we
    # sink 60s into a change-password timeout. Diagnostic only.
    wait_for_health(BASE, timeout = 30.0, info = info)
    # Chromium launch args: see `tests/studio/_playwright_robust.py`.
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        # Reduce motion so view-transition animations don't intercept
        # pointer events and break Playwright's actionability check.
        reduced_motion = "reduce",
    )
    # Hard-disable CSS view-transitions: Unsloth's theme toggle + sidebar
    # collapse run startViewTransition() which can leave <html> intercepting
    # pointer events for a beat after each route swap. See _playwright_robust.py.
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
    page = ctx.new_page()
    # 60s default (was 30s): macos-14 under --single-process Chromium is
    # slow enough that renders/webfonts/lazy routes routinely crowd 30s.
    page.set_default_timeout(60_000)
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

    # Capture /v1/chat/completions statuses so a mid-test 4xx (which
    # surfaces only as a hung wait_for_function) is debuggable from the log.
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
        # Screenshots are diagnostic only -- never fail on a screenshot
        # timeout. Page.screenshot waits for webfonts, which on macos-14
        # can crowd the default; bump the timeout and swallow errors.
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
    # 1. Change-password through the UI ("Setup your account").
    # Bootstrap state pre-seeds the current password; we enter the
    # new password twice and submit -- the user's first-run experience.
    # ─────────────────────────────────────────────────────
    step("change-password through UI (Setup your account)")
    # Settle the network before touching the form: a late bootstrap poll
    # can rerender the page (dropping #new-password) mid-test. The whole
    # goto/wait/fill/submit sequence is wrapped in a 3-attempt retry with
    # a fresh page/reload between tries so a mid-try rerender doesn't
    # poison the next.
    form_err: Exception | None = None
    for _form_attempt in range(3):
        try:
            page.goto(f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000)
            try:
                page.wait_for_load_state("networkidle", timeout = 30_000)
            except Exception:
                pass  # best-effort -- proceed even if network never idles
            pw_field = page.locator("#new-password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            # Do NOT shoot() between wait_for and fill -- the screenshot's
            # font-load wait can let a background poll detach the form.
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            shoot("01-change-password-filled")
            # Click submit AND wait for the POST response together so a
            # macos-14 net::ERR_NO_BUFFER_SPACE buffer-fail surfaces now,
            # not at the next composer.wait_for.
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
                # ERR_NO_BUFFER_SPACE needs the OS to recover socket
                # buffers; back off 5s then 15s before retrying.
                if "ERR_NO_BUFFER_SPACE" in str(e):
                    backoff_s = 5 if _form_attempt == 0 else 15
                    print(
                        f"[ui]   ENOBUFS detected; sleeping {backoff_s}s "
                        f"before retry to let OS recover socket buffers...",
                        flush = True,
                    )
                    time.sleep(backoff_s)
                # Replace the page if it died; otherwise next iteration's
                # page.goto() handles the reload.
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ui]   recovery: {m}", flush = True),
                )
    if form_err is not None:
        raise form_err

    # ─────────────────────────────────────────────────────
    # 2. Chat surface mounts, default model surface is visible.
    # ─────────────────────────────────────────────────────
    step("wait for composer to mount")
    # After change-password the router rebuilds login -> chat shell; on
    # macos-14 racing straight into wait_for() either burns the timeout
    # or crashes the renderer mid-mount. Settle network first, then
    # wait_for with one recovery cycle on failure.
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
                pass
            if _attempt == 0:
                # Re-navigate: open a fresh page in the same context if
                # the renderer died (localStorage auth survives), else
                # re-goto to force a clean re-render.
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

    # /api/models/list and /api/inference/load need a bearer; the
    # frontend stores it under "unsloth_auth_token" (auth/session.ts).
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

    # Verify the chat page's default model matches DEFAULT_MODELS_GGUF[0]
    # (defaults.py) -- guards the first-launch UX against list reorders.
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

    # The selector button should show the default model's name even
    # before a model is loaded ("Select model" if none).
    selector_btn = page.locator(
        'button:has-text("Select model"), '
        'button:has-text("gemma"), '
        'button:has-text("Qwen"), '
        'button:has-text("Llama")'
    ).first
    # Best-effort: selector re-mounts as /api/models/list resolves, so
    # use a short timeout and skip the snapshot on miss.
    sel_text = ""
    try:
        sel_text = (selector_btn.text_content(timeout = 2_000) or "").strip()
    except Exception as _sel_err:
        info(f"WARN: model-selector probe skipped: {type(_sel_err).__name__}: {_sel_err}")
    if sel_text:
        info(f"model selector button text: {sel_text!r}")
        shoot("03b-default-model-button")

    # ─────────────────────────────────────────────────────
    # 3. Trigger model load via the same endpoint the picker uses.
    # ─────────────────────────────────────────────────────
    step("load GGUF via /api/inference/load (uses session cookie)")
    # AbortSignal-bounded: macos-14 has been seen wedging on this fetch.
    # The 3-min budget is generous for a cold-cache load; a wedge fails
    # cleanly instead of forcing a 30-min runner cancel.
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

    # Unsloth caches model state in zustand; reload so the composer picks
    # up the loaded model.
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 3b. Model picker search bar -- exercise the typeahead filter.
    # We don't actually select a different model (multi-GB download);
    # this just catches picker-mount / debounced HF-search regressions.
    # ─────────────────────────────────────────────────────
    step("model picker: open + drive search bar")
    # Prefer the guided-tour anchor [data-tour="chat-model-selector"]
    # (app-sidebar.tsx) -- as stable as anything in the codebase.
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
            # "qwen" then "llama" popover text must DIFFER, proving the
            # typeahead actually filters (else an ignored-input regression
            # would silently pass).
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
        # Close picker without changing selection.
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    # ─────────────────────────────────────────────────────
    # 4. Five chat turns, all non-empty.
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
        # 1. Wait until the previous turn fully stopped: Send attached
        #    AND Stop detached. The composer hot-swaps both in one DOM
        #    slot, so Stop's detached state alone is racy.
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
            # Stop still on -- prior turn mid-stream. Wait it out at the
            # full per-turn budget.
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = TURN_TIMEOUT_MS,
            )

        # 2. Snapshot total bubble count before send; we wait for it to
        #    grow by exactly 1. We do NOT require non-empty text: an
        #    empty assistant response is legitimate (gemma-3-270m does
        #    this at temp 0), and the old non-empty predicate got stuck
        #    on such bubbles.
        bubbles_before = _bubble_count()
        # The llama.cpp and web update banners are fixed bottom-right toasts
        # (z-9998 / z-9999) that can overlap the composer's Send button and
        # intercept the click. Snooze whichever is showing before sending.
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

        # 3. Wait for the new placeholder bubble to render -- confirms
        #    the click was actionable and the request issued.
        page.wait_for_function(
            """(want) => {
                return document.querySelectorAll(
                    '[data-role="assistant"]'
                ).length >= want;
            }""",
            arg = bubbles_before + 1,
            timeout = TURN_TIMEOUT_MS,
        )

        # 4. Wait for this turn's streaming to finish. Stop may never
        #    appear (gemma-3-270m can finish before it paints), so its
        #    appearance is best-effort; then wait for it to detach.
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
    # Surface /v1/chat/completions status distribution: a 4xx here is
    # usually the cause of a hung wait_for_function downstream.
    if chat_completions_responses:
        statuses = [code for code, _ in chat_completions_responses]
        bad = [code for code in statuses if code >= 400]
        info(
            f"/v1/chat/completions: {len(statuses)} request(s); "
            f"statuses={statuses}; 4xx/5xx={len(bad)}"
        )

    # ─────────────────────────────────────────────────────
    # 5. Regenerate the last assistant turn.
    # ─────────────────────────────────────────────────────
    step("regenerate last assistant turn")
    last_assistant = page.locator('[data-role="assistant"]').last
    last_assistant.hover()
    page.wait_for_timeout(400)
    # Exclude disabled controls: the picker's new disabled "Reload model"
    # button also matches and sorts first, so .first would target it.
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
        try:
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
        # Don't strict-fail: ActionBarPrimitive.Reload has no stable
        # aria-label so the locator relies on icon-tied tooltip text.
        # Soft-skip until we add a data-testid (TODO).
        info("WARN regenerate button not visible (known-fragile locator, skipped)")

    # ─────────────────────────────────────────────────────
    # 6. Add two more turns AFTER regenerate.
    # ─────────────────────────────────────────────────────
    extra = ["Reply with: yes", "Reply with: no"]
    for j, p_ in enumerate(extra, start = 1):
        step(f"extra turn {j}: {p_!r}")
        before_count = len(page.locator('[data-role="assistant"]').all())
        send_and_wait(p_, before_count + 1)
    shoot("06-after-extra-turns")

    # ─────────────────────────────────────────────────────
    # 7. Composer toggle buttons. Each aria-label flips between
    # "Disable X" / "Enable X" with state (shared-composer.tsx).
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
        # e.g. gemma-3-270m has no reasoning, so thinking stays disabled).
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
        # Flip back so test state is unchanged.
        try:
            page.locator(
                f'button[aria-label="Disable {feature}"], button[aria-label="Enable {feature}"]'
            ).first.click()
        except Exception:
            pass
        page.wait_for_timeout(200)
    shoot("07-toggles-cycled")

    # ─────────────────────────────────────────────────────
    # 8. Configuration sheet: open, drive Temperature slider, close.
    # ─────────────────────────────────────────────────────
    cfg_open = page.locator('button[aria-label="Open configuration"]').first
    if cfg_open.count() > 0:
        step("Configuration sheet: drive Temperature + Top P + extras")
        cfg_open.click()
        page.wait_for_timeout(500)
        shoot("08-config-open")
        # Walk every Radix slider (role="slider") by index, focus it,
        # press Home (-> min) for deterministic state; a locked slider
        # surfaces an error here.
        sliders = page.locator('[role="slider"]')
        n_sliders = sliders.count()
        info(f"configuration sheet exposes {n_sliders} slider(s)")
        for idx in range(n_sliders):
            try:
                s = sliders.nth(idx)
                s.scroll_into_view_if_needed()
                s.focus()
                page.keyboard.press("Home")  # -> min
                page.wait_for_timeout(80)
            except Exception as exc:
                info(f"  slider[{idx}] focus/Home failed: {exc!r}")
        shoot("09-config-all-min")
        # Temperature is the first slider (configuration-sheet.tsx), so
        # Home already pinned it to 0 for determinism.
        info("Temperature set to slider min (0.0) for determinism")
        # Close.
        close_btn = page.locator('button[aria-label="Close configuration"]').first
        if close_btn.count() > 0:
            close_btn.click()
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    # ─────────────────────────────────────────────────────
    # 9. Theme toggle -- multiple cycles + computed-bg-color check
    # (light is near-white >240; dark is near-black <40).
    # ─────────────────────────────────────────────────────
    acct = page.locator('button[aria-label$=" account menu"]').first
    if acct.count() > 0:
        step("theme toggle x3 with computed-color assertion")
        observed = []
        for cycle in range(3):
            # Wait for any prior dropdown to fully detach: clicking while
            # the view-transition is still open no-ops silently. The
            # transition can run >700ms on slow CI, so use a roomy budget.
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
            # Retry once (after Escape to clear stray popups) if the first
            # click is silently swallowed mid-view-transition.
            opened = False
            for attempt in range(2):
                try:
                    acct.click(force = True)
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
            # Click with fallbacks: a small CI viewport can push the item
            # off-screen (force=True still needs it in viewport). Fall back
            # to scroll-into-view, then a synthetic evaluate() .click() that
            # skips Playwright's viewport check.
            click_err = None
            for click_attempt in range(3):
                try:
                    if click_attempt == 0:
                        theme_item.click(force = True, timeout = 3_000)
                    elif click_attempt == 1:
                        theme_item.scroll_into_view_if_needed(timeout = 2_000)
                        theme_item.click(force = True, timeout = 3_000)
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
            # Settle. The ".dark" class on <html> is the ground truth
            # (theme-store toggles only that); don't gate on ".light".
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
            shoot(f"10-theme-cycle-{cycle + 1}")
            info(f"  cycle {cycle + 1}: dark={bg['isDark']} body bg={bg['bg']!r}")
        # Across cycles we should see both a near-white (light) and a
        # near-black (dark) body bg; one polarity means the toggle stuck.
        rgbs = [parse_rgb(o["bg"]) for o in observed if parse_rgb(o["bg"])]
        light_seen = any(min(r) > 220 for r in rgbs)
        dark_seen = any(max(r) < 60 for r in rgbs)
        if len(observed) < 3:
            soft_fail(f"theme toggle ran only {len(observed)} cycle(s), expected 3")
        # Don't strict-fail on both polarities: the runner's
        # prefers-color-scheme + Unsloth's "system" default can collapse
        # to one polarity even when .dark toggles correctly. The 3-cycle
        # completion above is the real invariant.
        if light_seen and dark_seen:
            info("OK light + dark computed background colors observed")
        else:
            info(
                f"WARN observed only one polarity across {len(rgbs)} "
                f"cycles: light_seen={light_seen}, dark_seen={dark_seen} "
                "(toggle may not flip on this runner's color-scheme)"
            )

    # ─────────────────────────────────────────────────────
    # 10. Sidebar nav: New Chat, Compare, Search, Recipes.
    # ─────────────────────────────────────────────────────
    def click_nav(label, expected_url_pat = None):
        # Resolve the sidebar nav button. get_by_role(name=...) works on
        # Linux but the tooltip-derived name can be empty on macOS when
        # the sidebar collapses to icons, so fall back to more permissive
        # locators.
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
            soft_fail(f"nav '{label}' not found")
            return False
        # force=True bypasses the actionability check: the post-toggle
        # view-transition can briefly report <html> as topmost even
        # though the button is visible + enabled (belt-and-suspenders
        # atop the startViewTransition neutraliser).
        try:
            btn.click(force = True, timeout = 5_000)
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
        plus_btn.click(force = True)
        page.wait_for_timeout(400)
        compare_item = page.get_by_role("menuitem", name = re.compile(r"Compare chat", re.I)).first
        if compare_item.count() == 0:
            # Compare chat moved into the "More" submenu; hover (then
            # click as fallback) to open it.
            more_trigger = page.get_by_role("menuitem", name = re.compile(r"^More$", re.I)).first
            if more_trigger.count() > 0:
                more_trigger.hover()
                page.wait_for_timeout(400)
                compare_item = page.get_by_role(
                    "menuitem", name = re.compile(r"Compare chat", re.I)
                ).first
                if compare_item.count() == 0:
                    more_trigger.click(force = True)
                    page.wait_for_timeout(400)
                    compare_item = page.get_by_role(
                        "menuitem", name = re.compile(r"Compare chat", re.I)
                    ).first
        if compare_item.count() > 0:
            compare_item.click(force = True)
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
    # Back to chat for subsequent steps.
    page.goto(f"{BASE}/chat")
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 11. API / Developer tab via account menu -> Settings dialog,
    # api-keys tab. Guards against the management UI being hidden.
    # ─────────────────────────────────────────────────────
    if acct.count() > 0:
        step("Developer (API) tab via account menu")
        acct.click()
        page.wait_for_timeout(400)
        dev = page.get_by_role("menuitem", name = re.compile(r"developer|api", re.I)).first
        if dev.count() > 0:
            dev.click()
            page.wait_for_timeout(800)
            shoot("15-developer-tab")
            # Look for the create-key affordance.
            create_btn = page.get_by_role(
                "button",
                name = re.compile(r"create.*key|generate.*key|add.*key|new key", re.I),
            ).first
            if create_btn.count() > 0:
                info("OK 'create API key' affordance visible")
            # Look for the api-keys list section title.
            keys_section = page.get_by_text(
                re.compile(r"api keys|developer", re.I),
            ).first
            if keys_section.count() > 0:
                info(f"OK API tab text: {(keys_section.text_content() or '').strip()[:80]!r}")
            # Close dialog with Escape.
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        else:
            page.keyboard.press("Escape")

    # ─────────────────────────────────────────────────────
    # 11b. Recipes tab: cards render + we can click one. A broken
    # loader would render zero cards or crash the route.
    # ─────────────────────────────────────────────────────
    step("Recipes tab: cards render + click first card")
    page.goto(f"{BASE}/data-recipes")
    page.wait_for_timeout(1500)
    # Count clickable headings/cards under main, then screenshot.
    headings = page.locator("main h2, main h3, [data-recipe], a[href*='/data-recipes/']")
    n_cards = headings.count()
    info(f"Recipes route headings/cards: {n_cards}")
    shoot("15b-recipes-cards")
    if n_cards > 0:
        # Try clicking the first one to confirm it navigates / opens.
        try:
            headings.first.scroll_into_view_if_needed()
            headings.first.click()
            page.wait_for_timeout(1200)
            shoot("15c-recipes-first-card")
            info("OK clicked first recipe card")
        except Exception as exc:
            info(f"WARN click first recipe failed: {exc!r}")
    # Back to chat.
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 11c. Recents: click the most-recent thread (we persisted one
    # via the turns above). Guards the thread-history loader / route.
    # ─────────────────────────────────────────────────────
    step("Recents: click previous chat in sidebar")
    # The persisted thread title is usually a snippet of the first user
    # message, so accept any of our prompt keywords.
    PROMPT_KEYWORDS = ("hello", "world", "tree", "yes", "1+1", "2+2")
    # Use the structural data-testid (thread-sidebar.tsx): the old
    # text-filtered selector matched coalesced nav text and burned
    # 13-23 min per platform. Also bound the whole step at 30s so a
    # misbehaving selector can't blow up wallclock.
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
    # Back to chat.
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 12. Image attachment UI reachable. The current model is text-only,
    # so just check the button exists (CI's gemma-4-E2B covers vision).
    # ─────────────────────────────────────────────────────
    step("attachment widget reachable")
    attach = page.locator('button[aria-label="Add Attachment"]').first
    if attach.count() > 0:
        # Only hover -- clicking would block on the native file dialog.
        attach.hover()
        page.wait_for_timeout(200)
        shoot("16-attachment-hover")

    # ─────────────────────────────────────────────────────
    # 13. Reload + verify session JWT survives.
    # ─────────────────────────────────────────────────────
    step("reload + session survives")
    page.reload()
    composer.wait_for(state = "visible", timeout = 60_000)
    if "/login" in page.url:
        fail(f"unexpected redirect to /login after reload: {page.url}")
    shoot("17-after-reload")

    # ─────────────────────────────────────────────────────
    # 14. /api/health stays healthy throughout.
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
    # 15. Negative-auth post-UI-rotation.
    # ─────────────────────────────────────────────────────
    step("post-rotation auth check (after UI change-password)")
    if (s_old := login_via_api(OLD)) != 401:
        fail(f"old bootstrap pw should be 401, got {s_old}")
    if (s_new := login_via_api(NEW)) != 200:
        fail(f"rotated pw should be 200, got {s_new}")
    info("OK old=401, new=200")

    # ─────────────────────────────────────────────────────
    # 16. Out-of-band ("terminal") password rotation via subprocess(curl).
    # Rotating from a shell must invalidate the old creds and revoke
    # refresh tokens server-side (auth.py:152), so the browser's
    # /api/auth/refresh must fail too.
    # ─────────────────────────────────────────────────────
    step("rotate password via subprocess(curl) -- the 'terminal' path")
    # Log in via the API for a fresh token (what an admin does from a shell).
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

    # NEW must now be 401, NEW2 must be 200.
    if (s_new1 := login_via_api(NEW)) != 401:
        fail(f"after CLI rotation, NEW pw should be 401, got {s_new1}")
    if (s_new2 := login_via_api(NEW2)) != 200:
        fail(f"after CLI rotation, NEW2 pw should be 200, got {s_new2}")
    info("OK after CLI rotation: NEW=401, NEW2=200 -- old studio creds dead")

    # /change-password revoked refresh tokens server-side (auth.py), so
    # the browser's /api/auth/refresh must now fail.
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
        fail(f"curl refresh-token check returned invalid status: " f"{refresh_proc.stdout!r}")
    if refresh_status == 200:
        fail(f"/api/auth/refresh should fail after CLI rotation; got 200")
    info(
        f"OK browser /api/auth/refresh now {refresh_status} "
        "(refresh token revoked) -- old studio session can no longer renew"
    )

    # ─────────────────────────────────────────────────────
    # 17. Persisted monitor auth boundary, then shutdown. A monitor left open
    # must stay dormant on /login and resume after successful authentication.
    # ─────────────────────────────────────────────────────
    step("persisted monitor stays dormant on /login and resumes after auth")
    # Start fresh after the CLI rotation invalidates this browser session.
    # Stay in the SAME context: macOS Chromium runs --single-process, where
    # closing the last context kills the browser and a second context cannot
    # be created. Open the new page before closing the old one; the context
    # init script covers the new page.
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
    # Auth tokens live in localStorage, and /login's guest guard redirects on
    # their mere presence, so drop them before navigating.
    try:
        page.evaluate(
            "['unsloth_auth_token', 'unsloth_auth_refresh_token']"
            ".forEach((key) => localStorage.removeItem(key))"
        )
    except Exception as exc:
        info(f"WARN clearing stale auth tokens failed: {exc!r}")
    _fresh_page = ctx.new_page()
    _fresh_page.set_default_timeout(60_000)
    _fresh_page.on("pageerror", lambda e: page_errors.append(str(e)))
    _fresh_page.on("console", _on_console)
    try:
        page.close()
    except Exception:
        pass
    page = _fresh_page
    login_system_request_count = len(system_requests)

    # Re-login with NEW2 for a valid /api/shutdown token. Route changes can
    # still abort or interrupt this navigation, so the field wait below is the
    # final confirmation that we reached /login.
    _tolerated_nav = ("ERR_ABORTED", "interrupted by another navigation")
    # A slow CI runner can make this re-login navigation time out even with the
    # server healthy, so retry the whole goto/wait/fill/submit sequence (mirrors
    # the change-password retry above). wait_for_health is a diagnostic pre-gate.
    wait_for_health(BASE, timeout = 30.0, info = info)
    relogin_err: Exception | None = None
    for _relogin_attempt in range(3):
        try:
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
            # Wait on the login POST so a transient 4xx/5xx is caught and retried
            # here, not swallowed until the out-of-loop composer wait.
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
                # ERR_NO_BUFFER_SPACE needs the OS to recover socket
                # buffers; back off 5s then 15s before retrying.
                if "ERR_NO_BUFFER_SPACE" in str(e):
                    backoff_s = 5 if _relogin_attempt == 0 else 15
                    print(
                        f"[ui]   ENOBUFS detected; sleeping {backoff_s}s "
                        f"before retry to let OS recover socket buffers...",
                        flush = True,
                    )
                    time.sleep(backoff_s)
                # Replace the page if it died; otherwise next iteration's
                # page.goto() handles the reload.
                old_page = page
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    info = lambda m: print(f"[ui]   recovery: {m}", flush = True),
                )
                # A freshly created replacement page loses the pageerror/console
                # listeners; re-attach so error tracking survives recovery.
                if page is not old_page:
                    page.on("pageerror", lambda e: page_errors.append(str(e)))
                    page.on("console", _on_console)
    if relogin_err is not None:
        raise relogin_err
    # Composer mount confirms the rotated session is authenticated. Kept OUTSIDE the
    # retry: the loop breaks right after submit, so we never re-goto /login once login
    # has set tokens -- that would hit the guest guard, redirect to /chat, and make a
    # merely-slow composer look like a broken login.
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

    # Wait for the post-shutdown placeholder body (the component swaps in
    # "Unsloth Studio has stopped." once /api/shutdown returns ok).
    try:
        page.wait_for_function(
            """() => /Unsloth Studio has stopped/.test(document.body.innerText)""",
            timeout = 15_000,
        )
        shoot("20-shutdown-placeholder")
        info("OK 'Unsloth Studio has stopped' placeholder rendered")
    except Exception as exc:
        info(f"WARN shutdown placeholder didn't render: {exc!r}")

    # /api/health must now be unreachable; poll for up to 15s.
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

    # Some pageerrors are benign: chat-completions 422s (network-layer
    # bubble-up, not a JS bug; per-turn flow already validates each turn)
    # and fetch failures after Shutdown (server is dead by design). Full
    # list in `_playwright_robust.BENIGN_PAGE_ERROR_PATTERNS`.
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
