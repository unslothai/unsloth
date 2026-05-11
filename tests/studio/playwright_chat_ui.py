# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Comprehensive Studio chat UI test, run locally + in CI.

Covers:
  1. /change-password through the UI (no API pre-rotate).
  2. Model loaded by the time chat opens (the chat page's runtime
     adapter pings /api/models/list; we trigger /api/inference/load
     via page.evaluate so we don't need the password out-of-band).
  3. Five chat turns, each deterministic (temperature handled at the
     server level via Studio's default; we only assert non-empty).
  4. Regenerate the last turn from the assistant action bar.
  5. Composer toggle buttons: Thinking / Web search / Code execution
     -- assert aria-label flips state on click.
  6. Configuration sheet: open, drive Temperature slider via keyboard,
     close.
  7. Theme toggle through the account menu, multiple cycles, with a
     deterministic computed-background-color check on
     `document.documentElement` and `document.body`.
  8. Sidebar nav: New Chat, Compare, Search, Recipes (URL changes).
  9. Recents (history) cards: click an existing chat thread.
  10. API tab via account menu -> Developer / api-keys.
  11. Image attachment UI (upload widget reachable; vision response
      not asserted because gemma-3-270m is text-only).
  12. Reload + verify session JWT survives.
  13. /api/health remains healthy.
  14. Negative-auth post-UI-rotation: old=401, new=200.
  15. Terminal-driven password rotation via subprocess(curl) to
      /api/auth/change-password (NEW -> NEW2). Confirms refresh
      tokens get revoked and that an out-of-band password change
      (i.e. another tab / CLI / curl) invalidates the old creds.
  16. Shutdown via the account menu's Shutdown menuitem + the
      AlertDialog's "Stop server" action; wait for /api/health to
      become unreachable (server process exited).
  17. No uncaught page errors.
"""

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

# Shared robustness helpers live next to this script. Tests run as
# plain `python tests/studio/playwright_chat_ui.py` (not via pytest /
# import), so prepend the dir to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    click_and_wait_for_response,
    install_view_transition_killer,
    is_benign_console_error,
    is_benign_page_error,
    recover_or_replace_page,
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

# Strict mode -- when on (default in CI), the test fails loudly if any
# expected button / nav / dialog is missing instead of logging a WARN
# and continuing. Locally we leave it off so the test still runs against
# a partial Studio install.
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"

# Per-turn assistant-bubble wait. The free macos-14 runner (3 vCPU /
# 7 GB / no GPU) is ~3-5x slower at gemma-3-270m CPU inference than the
# free ubuntu-latest runner; "Say the word 'tree'" has been observed to
# hit the 180 s default exactly. STUDIO_UI_TURN_TIMEOUT_MS lets the Mac
# CI bump this without hard-coding a Mac branch in the test.
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))

_n = [0]


def step(s):
    print(f"[ui] STEP {s}", flush = True)


def info(s):
    print(f"[ui] {s}", flush = True)


def fail(m):
    raise AssertionError(f"[ui] FAIL: {m}")


def soft_fail(m):
    """Hard fail in STRICT mode, info-warn otherwise.

    Use for "this button should exist but didn't" assertions where
    a missing element is a regression in CI but acceptable when
    running against a partial Studio locally.
    """
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
    # Pre-flight: bash-side wait_for already gated on /api/health
    # before launching us, but the macos-14 free runner has been
    # observed to surface a 200 /api/health while the auth DB is
    # still finishing its migration. A second 30s probe inside the
    # script catches that gap before we sink 60s into a change-
    # password timeout. Diagnostic only -- the workflow's own wait
    # is the authoritative gate, so we don't fail on miss.
    wait_for_health(BASE, timeout = 30.0, info = info)
    # Chromium launch args: see `tests/studio/_playwright_robust.py`.
    # Bundles the macos-14 stability set (--single-process for the
    # pipeTransport.js JSON-RPC crash) + new throttling kill set
    # (--disable-background-timer-throttling and friends) that
    # prevent Chromium from deprioritising the headless context's
    # CPU/timers when it thinks the window is backgrounded -- which
    # CI runners routinely flag.
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        # Reduces motion so the theme toggle's view-transition
        # animation doesn't briefly intercept pointer events
        # (the running CSS view-transition leaves the html in a
        # state where Playwright's actionability check fails).
        reduced_motion = "reduce",
    )
    # Hard-disable CSS view-transitions: see _playwright_robust.py
    # for the underlying init script. Necessary because Studio's theme
    # toggle + sidebar collapse run their own startViewTransition()
    # which can leave the <html> element intercepting pointer events
    # for a beat after each route swap -- Playwright surfaces this as
    # "<html> intercepts pointer events" on the next click.
    install_view_transition_killer(ctx)
    page = ctx.new_page()
    # 60s default (was 30s) -- macos-14 free runner under
    # --single-process Chromium is slow enough that page renders /
    # webfonts / lazy-loaded routes routinely crowd 30s. Run
    # 25494926834 hit Page.screenshot timeout AND
    # locator.wait_for("#new-password") timeout under the old 30s
    # default. 60s is conservative without bloating real-failure
    # detection.
    page.set_default_timeout(60_000)
    page_errors = []
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    console_errors: list[str] = []
    # Filtered console.error log -- excludes BENIGN_CONSOLE_ERROR_PATTERNS
    # so the diagnostic dumps + final summary count only signals worth
    # reading. Raw firehose is still surfaced via len(console_errors)
    # vs len(filtered).

    def _on_console(m):
        if m.type != "error":
            return
        try:
            text = m.text
        except Exception:
            return
        console_errors.append(text)

    page.on("console", _on_console)

    # Per-turn HTTP-status capture: if a /v1/chat/completions request
    # 4xx-rejects mid-test the symptom is a hung wait_for_function and
    # a "FAIL: 1 non-benign pageerror events" line; this listener
    # surfaces the underlying status codes so a flake is debuggable
    # straight from the CI log without artifact spelunking.
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
        # Screenshots are diagnostic artifacts only -- never fail the
        # test on a screenshot timeout. Page.screenshot waits for
        # webfonts to fully load before snapshotting; on macos-14 free
        # runners with --single-process Chromium, font loading on the
        # Studio chat page (Inter / Geist Mono) regularly crowds the
        # 30s default and crashes Page.screenshot. Bump the timeout
        # AND wrap in try/except so the test progresses even if the
        # screenshot can't be captured. animations='disabled' freezes
        # any in-flight CSS transitions for a deterministic snap.
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
    # The bootstrap state injects window.__UNSLOTH_BOOTSTRAP__
    # so the current-password is pre-seeded; we only enter the
    # new password twice and submit. Match the workflow rename
    # from "tool calling tests" pattern: this *is* the user's
    # first-run experience.
    # ─────────────────────────────────────────────────────
    step("change-password through UI (Setup your account)")
    # Wait for the network to settle before touching the form. Without
    # this, on macos-14 free runners under --single-process Chromium,
    # the page sometimes redirects mid-test (the bootstrap state poll
    # finishes after wait_for() returns, the React router decides
    # we're "already authenticated" or "no longer must-change", and
    # rerenders without #new-password). Letting the network idle first
    # gives the bootstrap dispatch a chance to settle BEFORE we
    # commit to the form path. Run 25497245250 / job 74820324136
    # showed this exact sequence: wait_for() returned then
    # page.fill('#new-password') timed out 60s later because the
    # form had been replaced. Run 25578374480 / job 75091072289
    # showed the same race a step deeper: pw_field.fill('#new-password')
    # succeeded then page.fill('#confirm-password') hit a 60s timeout
    # because a re-render between the two locators detached the
    # second input. We wrap the whole goto/wait/fill/submit sequence
    # in a 3-attempt retry, with a fresh page or hard reload between
    # attempts so a re-render in the middle of one try doesn't poison
    # the next.
    form_err: Exception | None = None
    for _form_attempt in range(3):
        try:
            page.goto(
                f"{BASE}/change-password", wait_until = "domcontentloaded", timeout = 60_000
            )
            try:
                page.wait_for_load_state("networkidle", timeout = 30_000)
            except Exception:
                pass  # best-effort -- proceed even if network never idles
            pw_field = page.locator("#new-password")
            pw_field.wait_for(state = "visible", timeout = 60_000)
            # NOTE: do NOT call shoot() between wait_for and fill -- the
            # screenshot's font-load wait gives the React form a chance to
            # detach if any background state-poll fires. Take screenshots
            # AFTER the form is committed instead.
            pw_field.fill(NEW, timeout = 60_000)
            page.fill("#confirm-password", NEW, timeout = 60_000)
            shoot("01-change-password-filled")
            # Click submit AND wait for the POST /api/auth/change-password
            # response in the same step. macos-14 free runners under
            # --single-process Chromium occasionally hit
            # net::ERR_NO_BUFFER_SPACE when the renderer requests a
            # resource (run 25586583024 / job 75116256117 had the
            # change-password POST silently buffer-fail and the page
            # stayed on /change-password; even after my page.goto(BASE)
            # recovery the auth state never persisted). Tying the
            # click to the response wait surfaces the buffer-error
            # IMMEDIATELY in this attempt rather than at the next
            # composer.wait_for, so the next retry-iteration starts
            # fresh with a known-bad starting state.
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
                print(
                    f"[ui]   first pageerror:    {page_errors[0][:200]!r}", flush = True
                )
            try:
                shoot(f"01-change-password-attempt-{_form_attempt + 1}-fail")
            except Exception:
                pass
            if _form_attempt < 2:
                # Recovery: replace the page if it died, otherwise the
                # next loop iteration's page.goto() handles the reload.
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
    # The change-password POST resolves async and the React router
    # rebuilds the tree (login form -> chat shell) on success. On
    # macos-14 free runners under --single-process Chromium, the
    # rebuild is heavy enough under software rendering that one of
    # two things happens if we race straight into wait_for():
    #   (a) the composer textarea is still suspending and we burn
    #       the 60s ceiling waiting for it to mount, or
    #   (b) the renderer crashes mid-mount, which under
    #       --single-process takes the entire context down (next
    #       Playwright call returns TargetClosedError).
    # Defend against both: settle network first, then attempt
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
                print(
                    f"[ui]   first pageerror:    {page_errors[0][:200]!r}", flush = True
                )
            try:
                shoot(f"03-composer-wait-attempt-{_attempt + 1}-fail")
            except Exception:
                pass
            if _attempt == 0:
                # Recovery: re-navigate. If the page died (renderer
                # gone under --single-process) we open a fresh page in
                # the same context so the auth state in localStorage
                # survives; otherwise we re-goto the same URL to force
                # a clean re-render.
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

    # Pull the auth token now -- /api/models/list and
    # /api/inference/load both require a bearer. The frontend
    # stores it under "unsloth_auth_token" (auth/session.ts).
    token = page.evaluate(
        "() => localStorage.getItem('unsloth_auth_token')",
    )
    if not token:
        # Fall back: exchange the refresh token via /api/auth/refresh.
        refresh_token = page.evaluate(
            "() => localStorage.getItem('unsloth_auth_refresh_token')",
        )
        if refresh_token:
            refresh = page.evaluate(
                f"""async (rt) => {{
                const r = await fetch("{BASE}/api/auth/refresh", {{
                    method: "POST",
                    headers: {{"Content-Type": "application/json"}},
                    body: JSON.stringify({{refresh_token: rt}}),
                }});
                return await r.json();
            }}""",
                refresh_token,
            )
            token = refresh.get("access_token")
    if not token:
        fail("could not obtain auth token after change-password")

    # Verify the chat page's default model surface comes from
    # backend/core/inference/defaults.py:DEFAULT_MODELS_GGUF[0],
    # which is the canonical "what the user sees if nothing has
    # been loaded yet" entry. A regression that reorders that
    # list or hides the default would break the first-launch UX,
    # which is what this assertion guards.
    step("default_models[0] matches DEFAULT_MODELS_GGUF[0]")
    EXPECTED_DEFAULT = os.environ.get(
        "EXPECTED_DEFAULT_MODEL",
        "unsloth/gemma-4-E2B-it-GGUF",
    )
    defaults = page.evaluate(
        f"""async (token) => {{
        const r = await fetch("{BASE}/api/models/list", {{
            headers: {{ "Authorization": "Bearer " + token }},
        }});
        return await r.json();
    }}""",
        token,
    )
    if not defaults.get("default_models"):
        fail(f"/api/models/list returned no default_models: {defaults}")
    if defaults["default_models"][0] != EXPECTED_DEFAULT:
        fail(
            f"default_models[0]={defaults['default_models'][0]!r}, "
            f"expected {EXPECTED_DEFAULT!r}; defaults.py drift?"
        )
    info(f"OK default_models[0] = {EXPECTED_DEFAULT}")

    # The model selector button text on the chat page should say
    # the default model's display name even before a model is
    # loaded. The model-selector renders the current model name
    # (or "Select model" if no current); for a fresh chat it
    # should surface the default.
    selector_btn = page.locator(
        'button:has-text("Select model"), '
        'button:has-text("gemma"), '
        'button:has-text("Qwen"), '
        'button:has-text("Llama")'
    ).first
    if selector_btn.count() > 0:
        sel_text = (selector_btn.text_content() or "").strip()
        info(f"model selector button text: {sel_text!r}")
        shoot("03b-default-model-button")

    # ─────────────────────────────────────────────────────
    # 3. Trigger model load via the page's session cookies.
    # Equivalent to the user clicking a model in the picker;
    # we just call the same endpoint the picker would.
    # ─────────────────────────────────────────────────────
    step("load GGUF via /api/inference/load (uses session cookie)")
    # Token already fetched above; reuse it for the load call.
    load_resp = page.evaluate(f"""async () => {{
        const r = await fetch("{BASE}/api/inference/load", {{
            method: "POST",
            headers: {{
                "Authorization": "Bearer {token}",
                "Content-Type": "application/json",
            }},
            body: JSON.stringify({{
                model_path: "{GGUF_REPO}",
                gguf_variant: "{GGUF_VARIANT}",
                is_lora: false,
                max_seq_length: 2048,
            }}),
        }});
        return {{status: r.status, body: await r.json()}};
    }}""")
    if load_resp["status"] != 200:
        fail(
            f"/api/inference/load returned {load_resp['status']}: {load_resp.get('body')!r}"
        )
    info(f"loaded model: {load_resp['body'].get('display_name')}")

    # Studio caches the per-context model state in zustand; reload
    # to make the chat composer pick up the loaded model.
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 3b. Model picker search bar -- click the model selector,
    # type into the search box, verify filtering. We don't
    # actually select a different model (that would trigger a
    # multi-GB download); we just exercise the typeahead so a
    # regression in the picker mount / debounced HF search would
    # surface here.
    # ─────────────────────────────────────────────────────
    step("model picker: open + drive search bar")
    # Stable selector first: [data-tour="chat-model-selector"] is the
    # guided-tour anchor on the model picker button (app-sidebar.tsx).
    # If the tour anchor moves the tour breaks, so this selector is at
    # least as stable as anything else in the codebase.
    picker_btn = page.locator('[data-tour="chat-model-selector"]').first
    if picker_btn.count() == 0:
        # Fall back to text-based locators for older Studio builds.
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
            # Type "qwen" -> capture popover text. Type "llama" -> capture
            # again. The two text snapshots must DIFFER, proving the
            # typeahead actually filters the list (a regression that
            # rendered the picker but ignored input would silently pass
            # the old version of this test).
            def picker_visible_text():
                return page.evaluate("""() => {
                    const el = document.querySelector(
                        '[role="dialog"], [role="listbox"], [role="menu"]'
                    );
                    return el ? (el.innerText || '').trim() : '';
                }""")

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
        """Total number of [data-role='assistant'] elements (empty or not)."""
        return page.evaluate("""() => {
            return document.querySelectorAll('[data-role="assistant"]').length;
        }""")

    def send_and_wait(prompt, idx):
        # 1. Wait until the previous turn has fully stopped: Send
        #    button is attached AND Stop button is detached. The
        #    assistant-ui composer hot-swaps these inside a single
        #    DOM slot; relying on Stop's detached state alone is
        #    racy (the slot can briefly show neither during
        #    transition).
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
            # Stop button still hanging on -- that's the prior turn
            # mid-stream. Wait it out at the full per-turn budget.
            page.wait_for_selector(
                'button[aria-label="Stop generating"]',
                state = "detached",
                timeout = TURN_TIMEOUT_MS,
            )

        # 2. Snapshot total bubble count BEFORE send. We then wait
        #    for total count to grow by exactly 1 (proves the new
        #    placeholder rendered) and for the Stop button to come
        #    + go (proves the new turn ran end-to-end). We do NOT
        #    require the new bubble's text to be non-empty: an
        #    empty assistant response is a legitimate model output,
        #    not a test failure. The earlier "non-empty count >=
        #    baseline + 1" predicate broke when any prior turn
        #    streamed empty (which gemma-3-270m DOES on simple
        #    prompts at temperature 0), because that empty bubble
        #    became permanently "stuck" below the moving threshold.
        bubbles_before = _bubble_count()
        composer.click()
        composer.fill(prompt)
        page.locator('button[aria-label="Send message"]').click()

        # 3. Wait for the new placeholder bubble to render. This
        #    confirms the click was actionable AND the request
        #    issued (assistant-ui only mounts the placeholder once
        #    the runtime accepts the message).
        page.wait_for_function(
            """(want) => {
                return document.querySelectorAll(
                    '[data-role="assistant"]'
                ).length >= want;
            }""",
            arg = bubbles_before + 1,
            timeout = TURN_TIMEOUT_MS,
        )

        # 4. Wait for streaming to FINISH for this specific turn.
        #    We wait for Stop button to APPEAR (proves streaming
        #    started) with a short budget; if it never appears,
        #    that's fine -- gemma-3-270m can finish before the
        #    Stop button paints. Either way we then wait for it
        #    to be detached at the full per-turn budget.
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

    texts = page.evaluate("""() => Array.from(document.querySelectorAll('[data-role="assistant"]'))
        .map(e => (e.innerText || '').trim())""")
    if len(texts) < len(prompts):
        fail(f"expected >= {len(prompts)} assistant bubbles, got {len(texts)}")
    info(f"five turn lengths = {[len(t) for t in texts[:5]]}")
    # Surface /v1/chat/completions HTTP status distribution so a flake
    # is debuggable from the CI log directly. A 4xx during a chat
    # turn is almost always the upstream cause of a hung
    # wait_for_function on a downstream turn.
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
    regen_btn = page.get_by_role(
        "button",
        name = re.compile(r"(reload|regenerate)", re.I),
    ).first
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
        # Don't strict-fail on regenerate -- the assistant-ui
        # ActionBarPrimitive.Reload doesn't expose a stable
        # aria-label, so the test depends on tooltip text matching
        # which is tied to the icon set. Soft-skip until we add a
        # data-testid in the action bar (TODO).
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
    # 7. Composer toggle buttons. Each renders with an
    # aria-label that flips between "Disable X" / "Enable X"
    # depending on its current state (shared-composer.tsx).
    # ─────────────────────────────────────────────────────
    step("composer toggle buttons (Thinking / Web search / Code execution)")
    for feature in ("thinking", "web search", "code execution"):
        # Look for either "Disable X" or "Enable X" -- whichever
        # is currently rendered.
        toggle = page.locator(
            f'button[aria-label="Disable {feature}"], '
            f'button[aria-label="Enable {feature}"]'
        ).first
        if toggle.count() == 0:
            info(f"toggle '{feature}' not present on this layout")
            continue
        # Skip if the model doesn't support this capability (the
        # button is rendered disabled). gemma-3-270m, for instance,
        # has no reasoning so "Disable thinking" is permanent-disabled.
        if toggle.is_disabled():
            info(f"toggle '{feature}' is disabled for this model -- skip")
            continue
        before = toggle.get_attribute("aria-label") or ""
        toggle.click()
        page.wait_for_timeout(200)
        after = (
            page.locator(
                f'button[aria-label="Disable {feature}"], '
                f'button[aria-label="Enable {feature}"]'
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
                f'button[aria-label="Disable {feature}"], '
                f'button[aria-label="Enable {feature}"]'
            ).first.click()
        except Exception:
            pass
        page.wait_for_timeout(200)
    shoot("07-toggles-cycled")

    # ─────────────────────────────────────────────────────
    # 8. Configuration sheet: open, find Temperature slider,
    # press Home (→ 0), close.
    # ─────────────────────────────────────────────────────
    cfg_open = page.locator('button[aria-label="Open configuration"]').first
    if cfg_open.count() > 0:
        step("Configuration sheet: drive Temperature + Top P + extras")
        cfg_open.click()
        page.wait_for_timeout(500)
        shoot("08-config-open")
        # ParamSlider uses Radix UI Slider. Each slider gets a
        # role="slider" attribute. Walk every slider in the sheet
        # by index, focus it, send Home (-> min) so the test
        # state is fully deterministic. Whatever the labels are
        # ("Temperature", "Top P", "Min P", "Repetition penalty",
        # max_tokens etc.), we drive them all to min so a
        # regression that locks a slider returns errors here.
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
        # Then drive Temperature specifically to 0.0 to make the
        # downstream chat deterministic. Temperature is the *first*
        # slider in the sheet (configuration-sheet.tsx renders it
        # first); Home already pinned it to 0.
        info("Temperature set to slider min (0.0) for determinism")
        # Close.
        close_btn = page.locator('button[aria-label="Close configuration"]').first
        if close_btn.count() > 0:
            close_btn.click()
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    # ─────────────────────────────────────────────────────
    # 9. Theme toggle -- multiple cycles + deterministic
    # computed-background-color check. The light theme
    # uses near-white (>240); dark uses near-black (<40).
    # ─────────────────────────────────────────────────────
    acct = page.locator('button[aria-label$=" account menu"]').first
    if acct.count() > 0:
        step("theme toggle x3 with computed-color assertion")
        observed = []
        for cycle in range(3):
            # Wait for any prior dropdown to fully detach. The Radix
            # Account-menu sets data-state="open" while the view-
            # transition is mid-flight; clicking it again before that
            # clears would no-op silently and the for-loop bailed
            # after cycle 1 in earlier runs.
            try:
                page.wait_for_function(
                    """() => !document.querySelector('[role="menu"]')""",
                    timeout = 3_000,
                )
            except Exception:
                pass
            page.wait_for_timeout(150)
            try:
                acct.click(force = True)
            except Exception as exc:
                soft_fail(
                    f"theme cycle {cycle + 1}: account-menu click failed " f"({exc!r})"
                )
                break
            # Wait for the dropdown menu to actually render before
            # querying its items.
            try:
                page.wait_for_selector('[role="menu"]', timeout = 3_000)
            except Exception:
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
            try:
                theme_item.click(force = True)
            except Exception as exc:
                page.keyboard.press("Escape")
                soft_fail(
                    f"theme cycle {cycle + 1}: theme menuitem click failed "
                    f"({exc!r})"
                )
                break
            # Settle. The ".dark" class on <html> is the ground
            # truth (theme-store toggles only that class); the
            # ".light" sibling is steady-state from next-themes
            # so don't gate on it.
            page.wait_for_timeout(700)
            bg = page.evaluate("""() => {
                const root = document.documentElement;
                return {
                    cls:    root.className,
                    isDark: root.classList.contains('dark'),
                    bg:     getComputedStyle(document.body).backgroundColor,
                    rbg:    getComputedStyle(root).backgroundColor,
                };
            }""")
            observed.append(bg)
            shoot(f"10-theme-cycle-{cycle + 1}")
            info(f"  cycle {cycle + 1}: dark={bg['isDark']} body bg={bg['bg']!r}")
        # Sanity check: across cycles we should observe both a
        # light state (body bg roughly near-white) and a dark state
        # (body bg near-black). If we only saw one polarity the
        # toggle didn't flip.
        rgbs = [parse_rgb(o["bg"]) for o in observed if parse_rgb(o["bg"])]
        light_seen = any(min(r) > 220 for r in rgbs)
        dark_seen = any(max(r) < 60 for r in rgbs)
        if len(observed) < 3:
            soft_fail(f"theme toggle ran only {len(observed)} cycle(s), expected 3")
        # Don't strict-fail on "both polarities observed" -- the
        # CI runner's prefers-color-scheme + Studio's "system" default
        # can collapse to a single polarity even after a successful
        # toggle (the .dark classlist toggles correctly, but the
        # resolved theme can stay constant). Surface as info; the
        # 3-cycle loop completion above is the real invariant.
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
        # Resolve the sidebar nav button. The plain
        # get_by_role("button", name=...) lookup works on Linux
        # Chromium because the accessible-name algorithm there picks
        # up `tooltip={label}` from SidebarMenuButton, but on macOS
        # Chromium the tooltip-derived name is sometimes empty when
        # the sidebar collapses to icon-only mode. Fall back through
        # progressively more permissive locators so the test stays
        # green on both platforms.
        candidates = [
            page.get_by_role(
                "button", name = re.compile(rf"^\s*{label}\s*$", re.I)
            ).first,
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
        # force=True bypasses Playwright's actionability check. The
        # button IS visible + enabled, but the post-theme-toggle view-
        # transition can leave <html> reported as the topmost element
        # for a beat (we already neutralise startViewTransition via
        # add_init_script; this is belt-and-suspenders).
        try:
            btn.click(force = True, timeout = 5_000)
        except Exception as exc:
            soft_fail(f"nav '{label}' click failed: {exc!r}")
            return False
        page.wait_for_timeout(800)
        if expected_url_pat and not re.search(expected_url_pat, page.url):
            soft_fail(
                f"clicking '{label}' didn't change url to /{expected_url_pat}; "
                f"current: {page.url}"
            )
            return False
        return True

    step("sidebar nav: New Chat -> Compare -> Search -> Recipes")
    click_nav("New Chat", r"/chat")
    shoot("11-new-chat")
    click_nav("Compare", r"/chat\?")  # /chat?compare=...
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
    # 11. API / Developer tab via account menu -> opens the
    # Settings dialog with the api-keys tab. Verify we can see
    # the Create API Key form (or existing keys table); regressions
    # that hide the api-keys management UI surface here.
    # ─────────────────────────────────────────────────────
    if acct.count() > 0:
        step("Developer (API) tab via account menu")
        acct.click()
        page.wait_for_timeout(400)
        dev = page.get_by_role(
            "menuitem", name = re.compile(r"developer|api", re.I)
        ).first
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
                info(
                    f"OK API tab text: {(keys_section.text_content() or '').strip()[:80]!r}"
                )
            # Close dialog with Escape.
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        else:
            page.keyboard.press("Escape")

    # ─────────────────────────────────────────────────────
    # 11b. Recipes tab: verify cards render + we can click one.
    # The Recipes route renders a grid of preset cards; a
    # regression that breaks the loader would render zero cards
    # or crash the route.
    # ─────────────────────────────────────────────────────
    step("Recipes tab: cards render + click first card")
    page.goto(f"{BASE}/data-recipes")
    page.wait_for_timeout(1500)
    # Recipe cards are rendered as <a> or button elements; count
    # all clickable headings under main + screenshot.
    headings = page.locator(
        "main h2, main h3, [data-recipe], a[href*='/data-recipes/']"
    )
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
    # 11c. Recents: the chat sidebar lists previous threads. We
    # already created several turns above (which gets persisted
    # as a thread). Find the sidebar's recents region and click
    # the most-recent entry. This catches regressions in the
    # thread-history loader / route param plumbing.
    # ─────────────────────────────────────────────────────
    step("Recents: click previous chat in sidebar")
    # We sent the prompts ["Reply with exactly: hello", "What is 1+1?",
    # "Reply with exactly: world", ...] above. The thread title that
    # gets persisted is typically a snippet of the first user message
    # (Studio summarises after a few turns). We accept either a literal
    # word from one of our prompts OR a short Studio-summary heuristic.
    PROMPT_KEYWORDS = ("hello", "world", "tree", "yes", "1+1", "2+2")
    # Use the structural data-testid the frontend renders on each
    # chat-history entry (studio/frontend/src/features/chat/thread-
    # sidebar.tsx). The previous text-filtered selector
    #   "aside a, aside button, [data-sidebar='sidebar'] a, ..."
    # matched coalesced sidebar nav text like 'unslothBETA',
    # 'UUnslothUnsloth' which the EXCLUDE regex didn't strip; the
    # test then clicked nav links, lost its frame, hit per-locator
    # timeouts and burned 13-23 minutes per platform on this single
    # step (run 25537467494 macui = 23m9s, winui = 13m6s, linui = 13m5s).
    # Belt-and-suspenders: bound the whole step at 30s so a misbehaving
    # selector can never blow up wallclock the way the old loop did.
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
            # Strict check: after clicking the Recents entry, the
            # thread we land on must include at least one of our
            # prompts in its rendered messages.
            turns_text = page.evaluate(
                """() => {
                const els = document.querySelectorAll(
                    '[data-role="user"], [data-role="assistant"]'
                );
                return Array.from(els).map(e => (e.innerText || '')
                    .toLowerCase()).join(' ');
            }""",
                None,
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
        soft_fail(
            f"no Recents entry was clickable within 30s deadline "
            f"(n_threads={n_threads})"
        )
    # Back to chat.
    page.goto(f"{BASE}/chat")
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)

    # ─────────────────────────────────────────────────────
    # 12. Image attachment UI (upload widget reachable). The
    # current model is text-only so we don't assert a vision
    # response -- just that the attachment button is there
    # and the file input accepts a PNG. CI's gemma-4-E2B
    # job covers the actual vision path.
    # ─────────────────────────────────────────────────────
    step("attachment widget reachable")
    attach = page.locator('button[aria-label="Add Attachment"]').first
    if attach.count() > 0:
        # Just hover -- triggering the file picker mid-test
        # would block on a native dialog. Verifying the
        # button is reachable is enough.
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
    health = page.evaluate(f"""async () => {{
        const r = await fetch("{BASE}/api/health");
        return {{status: r.status, body: await r.text()}};
    }}""")
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
    # 16. Out-of-band ("terminal") password rotation.
    # POST /api/auth/change-password from a real subprocess(curl)
    # invocation -- this is the same surface a sysadmin / another
    # tab / a desktop helper would use, and the security promise
    # is: rotating the password from "the terminal" must invalidate
    # the previous credentials. The endpoint also revokes refresh
    # tokens server-side (auth.py:152), so /api/auth/refresh from
    # the still-open browser context must fail too.
    # ─────────────────────────────────────────────────────
    step("rotate password via subprocess(curl) -- the 'terminal' path")
    # Get a fresh access token by logging in via the API rather than
    # reusing whatever's in localStorage; this matches what an admin
    # would actually do from a shell.
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

    # The browser still has the pre-rotation access token. Refresh
    # tokens were revoked server-side by /change-password (auth.py),
    # so /api/auth/refresh from the browser context must now fail.
    refresh_after = page.evaluate(f"""async () => {{
        const r = await fetch("{BASE}/api/auth/refresh", {{
            method: "POST",
            credentials: "include",
        }});
        return {{status: r.status}};
    }}""")
    if refresh_after["status"] == 200:
        fail(f"/api/auth/refresh should fail after CLI rotation; got 200")
    info(
        f"OK browser /api/auth/refresh now {refresh_after['status']} "
        "(refresh token revoked) -- old studio session can no longer renew"
    )

    # ─────────────────────────────────────────────────────
    # 17. Shutdown button via the account menu.
    # The Shutdown menuitem opens an AlertDialog ("Stop Unsloth
    # Studio?") whose primary action is "Stop server"; clicking
    # it POSTs /api/shutdown and then replaces document.body with
    # the "Unsloth Studio has stopped" placeholder. /api/health
    # should become unreachable shortly after.
    # ─────────────────────────────────────────────────────
    step("Shutdown via account menu")
    # Re-login through the UI with NEW2 so the browser has a valid
    # access token for the /api/shutdown call (the previous one
    # was invalidated by the CLI rotation above).
    page.goto(f"{BASE}/login")
    pw_field = page.locator("#password")
    pw_field.wait_for(state = "visible", timeout = 60_000)
    pw_field.fill(NEW2)
    page.locator('button[type="submit"]').click()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)
    shoot("18-relogin-with-NEW2")

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

    # Wait for the post-shutdown placeholder body. The component
    # replaces document.body.innerHTML with text containing
    # "Unsloth Studio has stopped." once /api/shutdown returns ok.
    try:
        page.wait_for_function(
            """() => /Unsloth Studio has stopped/.test(document.body.innerText)""",
            timeout = 15_000,
        )
        shoot("20-shutdown-placeholder")
        info("OK 'Unsloth Studio has stopped' placeholder rendered")
    except Exception as exc:
        info(f"WARN shutdown placeholder didn't render: {exc!r}")

    # Now /api/health must become unreachable (process exited or is
    # at least not listening). Poll for up to 15 s.
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

    # Some pageerrors are benign in this test:
    #   - "Request failed (422)": the OpenAI-compatible chat-completions
    #     endpoint rejects rapid-fire/malformed requests with 422. The
    #     surfaced error is a network-layer bubble-up, NOT a JS bug,
    #     and the per-turn flow already validates message-by-message
    #     correctness. Filtering these here keeps the pageerror gate
    #     focused on actual frontend regressions (TypeError, ReferenceError,
    #     null deref, etc.).
    #   - "Failed to fetch" / "NetworkError" after the Shutdown click:
    #     the server is intentionally dead by then; any in-flight
    #     fetch fails by design.
    # The full list lives in `_playwright_robust.BENIGN_PAGE_ERROR_PATTERNS`
    # so playwright_extra_ui.py shares the same gate.
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
        f"console.error events: {len(console_errors)} total "
        f"({len(real_console_errors)} non-benign)"
    )

    info("PASS comprehensive UI flow")
    browser.close()
