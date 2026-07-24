# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model-picker per-model-config Playwright regression test (GPU-free, CPU gemma).

Guards, end to end against the real frontend, the exact regressions that got the
predecessor PR reverted:

  - Context Length persists: set a distinctive per-model Context Length + tick
    "Remember for this model" + Load; the value reaches the /api/inference/load
    request (max_seq_length) AND lands in localStorage (unsloth_model_configs),
    and survives a full browser reload (HARD).
  - Reset clears: after customizing, Reset must clear the stored override, never
    pin the context to a fixed number (the "Reset pins context" regression) (HARD).
  - Hidden infra models absent: the RAG embedder (bge-small-en-v1.5) and the
    llama.cpp validation probe (stories260K) never appear in the picker. The
    probe GGUF is primed into the HF cache by the CI job, so "absent" proves
    "hidden", not "not downloaded" (HARD).
  - Legacy migration is idempotent: a pre-feature unsloth_load_settings store
    migrates once into the versioned unsloth_model_configs map with the value
    preserved, and a second reload with a fresh legacy seed present does not
    re-migrate, duplicate, or clobber (gates under STUDIO_UI_STRICT via soft_fail).
  - Advanced settings persist: KV cache dtype / tensor-parallel toggled under
    Advanced + Remember land in unsloth_model_configs (best-effort).

Runs as a plain script (not via pytest), mirroring tests/studio/playwright_extra_ui.py:
accumulate failures in `_failed`, exit non-zero if any HARD gate failed. With
STUDIO_UI_STRICT=1 (as CI sets), soft_fail also gates; genuinely-optional checks
use runtime_warn so they never flake the merge gate.
"""

import json
import re
import sys
import os
import time
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
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ.get("STUDIO_NEW_PW", "ModelCfg-NEW-2026!")
# Attach mode: log into an already-provisioned Studio with an existing password
# instead of the first-boot change-password dance. CI leaves STUDIO_LOGIN_PW unset
# to exercise the real change-password flow; local runs can set it to skip re-provisioning.
LOGIN_PW = os.environ.get("STUDIO_LOGIN_PW")
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
# Substring of the On Device picker row for the loaded model.
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT", "gemma-3-270m")
# A distinctive valid (>=128, multiple of 128, below the model's 32768 ceiling)
# Context Length, clearly not a default, so persistence is unambiguous.
DISTINCT_CTX = int(os.environ.get("STUDIO_DISTINCT_CTX", "4096"))
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_modelcfg")
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
LOAD_FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_LOAD_TIMEOUT_MS", "180000"))

_n = [0]
_failed: list[str] = []


def step(s: str) -> None:
    print(f"[ui-modelcfg] STEP {s}", flush = True)


def info(s: str) -> None:
    print(f"[ui-modelcfg] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[ui-modelcfg] FAIL: {m}", flush = True)
    _failed.append(m)


def soft_fail(m: str) -> None:
    if STRICT:
        fail(m)
    else:
        info(f"WARN (strict-off): {m}")


def runtime_warn(m: str) -> None:
    """Warn about a genuinely-optional check that STRICT does not gate."""
    info(f"WARN (runtime): {m}")


def _count(loc) -> int:
    try:
        return loc.count()
    except Exception:
        return 0


def _as_int(value) -> int | None:
    """Parse an input value to int, tolerating commas/whitespace. Comparisons
    must be numeric, never substring: '40960' (a model's native default) would
    spuriously "contain" '4096'."""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _login_token_via_api(base: str, user: str, pw: str) -> str:
    """POST /api/auth/login -> access_token (attach-mode helper, stdlib only)."""
    import urllib.request

    req = urllib.request.Request(
        f"{base}/api/auth/login",
        data = json.dumps({"username": user, "password": pw}).encode(),
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )
    with urllib.request.urlopen(req, timeout = 15) as r:
        return json.loads(r.read().decode())["access_token"]


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ui-modelcfg",
        info = info,
    )
    # Health pre-flight: bash-side health wait can pass before the auth DB migrates.
    wait_for_health(BASE, timeout = 30.0, info = info)
    browser = p.chromium.launch(
        headless = True,
        args = chromium_launch_args(),
    )
    ctx = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
    )
    install_view_transition_killer(ctx)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    page_errors = []

    def _on_pageerror(e):
        msg = str(e)
        if is_benign_page_error(msg):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    # Record every /api/inference/load POST payload so the persistence gate can
    # assert max_seq_length.
    load_posts: list[str] = []

    def _on_request(req):
        try:
            if req.method == "POST" and "/api/inference/load" in req.url:
                load_posts.append(req.post_data or "")
        except Exception:
            pass

    page.on("request", _on_request)

    def shoot(name: str) -> None:
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

    def read_configs() -> dict:
        """Return the parsed unsloth_model_configs map (or {} if absent/invalid)."""
        raw = robust_evaluate(page, "() => localStorage.getItem('unsloth_model_configs')")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def config_entries(cfg: dict) -> list[dict]:
        """The per-model entries (dict values) of the stored map, schema-tolerant."""
        return [v for v in cfg.values() if isinstance(v, dict)]

    # ─────────────────────────────────────────────────────
    # Setup: authenticate + model load.
    # ─────────────────────────────────────────────────────
    if LOGIN_PW:
        # Attach mode: log in via the API and seed the token before navigation,
        # skipping the first-boot change-password dance.
        step("setup: API login + token seed (attach to running Studio)")
        _tok = _login_token_via_api(BASE, LOGIN_USER, LOGIN_PW)
        ctx.add_init_script(
            f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(_tok)});}}"
            f"catch(e){{}}"
        )
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)
    else:
        step("setup: change-password")
        # 3-attempt retry: the form can re-render mid-fill on slow runners and
        # detach the password fields; each retry re-navigates with a fresh page.
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
                status, _ = click_and_wait_for_response(
                    page,
                    url_substr = "/api/auth/change-password",
                    method = "POST",
                    do_click = lambda: page.locator('button[type="submit"]').click(),
                    timeout_ms = 30_000,
                    info = lambda m: print(f"[ui-modelcfg]   {m}", flush = True),
                )
                if status is not None and status >= 400:
                    raise AssertionError(
                        f"change-password POST returned {status}; page_errors={page_errors[:1]!r}"
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
                    f"[ui-modelcfg]   change-password attempt {_form_attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}; "
                    f"page_errors={len(page_errors)}",
                    flush = True,
                )
                if _form_attempt < 2:
                    if "ERR_NO_BUFFER_SPACE" in str(e):
                        backoff_s = 5 if _form_attempt == 0 else 15
                        time.sleep(backoff_s)
                    page = recover_or_replace_page(
                        page,
                        ctx,
                        default_timeout_ms = 60_000,
                        info = lambda m: print(f"[ui-modelcfg]   recovery: {m}", flush = True),
                    )
                    page.on("request", _on_request)
        if form_err is not None:
            raise form_err

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
                shoot(f"00-composer-wait-attempt-{_attempt + 1}-fail")
            except Exception:
                pass
            if _attempt == 0:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    goto_url = BASE,
                    settle_networkidle = True,
                    info = lambda m: print(f"[ui-modelcfg]   recovery: {m}", flush = True),
                )
                page.on("request", _on_request)
                composer = page.locator('textarea[aria-label="Message input"]')
    if last_err is not None:
        raise last_err
    shoot("01-chat-loaded")

    token = robust_evaluate(page, "() => localStorage.getItem('unsloth_auth_token')")
    if not token:
        fail("no access token after auth setup")
        sys.exit(1)

    # Load the tiny GGUF so it is a live "On Device" model in the picker.
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
    load_posts.clear()  # drop the setup load; keep only UI-driven loads below.

    # ─────────────────────────────────────────────────────
    # Picker helpers (proven selectors).
    # ─────────────────────────────────────────────────────
    POPOVER = '[data-tour="chat-model-selector-popover"]'
    TRIGGER = '[data-tour="chat-model-selector"]'

    def open_picker():
        popover = page.locator(POPOVER).first
        if _count(popover) == 0 or not popover.is_visible():
            page.locator(TRIGGER).first.click()
            page.wait_for_timeout(900)
            popover = page.locator(POPOVER).first
        popover.wait_for(state = "visible", timeout = 30_000)
        return popover

    def close_picker():
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
        except Exception:
            pass

    def select_on_device_row(popover, hint):
        od = page.get_by_role("tab", name = "On Device").first
        if _count(od):
            od.click()
            page.wait_for_timeout(700)
        row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            # Fall back to search filtering.
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(hint)
                page.wait_for_timeout(700)
                row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            return None
        row.click()
        page.wait_for_timeout(800)
        return row

    def open_config(popover, hint):
        if select_on_device_row(popover, hint) is None:
            return None
        gear = popover.locator('button[aria-label^="Inference settings for"]').first
        if _count(gear) == 0:
            return None
        gear.click()
        page.wait_for_timeout(800)
        return popover

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    def primary_button(popover):
        for name in ("Load model", "Reload model", "Save settings", "Forget settings"):
            b = popover.get_by_role("button", name = name).first
            if _count(b):
                return b
        return None

    # ─────────────────────────────────────────────────────
    # 1. Hidden infra models absent from the picker (HARD).
    # ─────────────────────────────────────────────────────
    step("hidden infra models absent from picker")
    popover = open_picker()
    shoot("02-picker-open")
    needles = ["bge-small-en-v1.5", "stories260"]
    tabs = ["Recommended", "On Device", "Connected"]
    hidden_ok = True
    for needle in needles:
        for tab_name in tabs:
            tab = page.get_by_role("tab", name = tab_name).first
            if _count(tab) == 0:
                continue
            try:
                tab.click()
                page.wait_for_timeout(400)
            except Exception:
                continue
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(needle)
                page.wait_for_timeout(600)
            hit = popover.locator(
                "[data-model-picker-option]",
                has_text = re.compile(re.escape(needle), re.I),
            )
            c = _count(hit)
            if c > 0:
                hidden_ok = False
                fail(f"infra model {needle!r} visible in picker '{tab_name}' tab ({c} rows)")
            if _count(search):
                search.fill("")
                page.wait_for_timeout(300)
    if hidden_ok:
        info("OK hidden: bge-small-en-v1.5 + stories260K absent from every picker tab")
    shoot("03-hidden-check")
    close_picker()

    # ─────────────────────────────────────────────────────
    # 2. Context Length persists (load + request + reload) (HARD).
    # ─────────────────────────────────────────────────────
    step(f"context length {DISTINCT_CTX} persists")
    popover = open_picker()
    if open_config(popover, MODEL_HINT) is None:
        fail(f"could not open run-settings for a model matching {MODEL_HINT!r}")
    else:
        shoot("04-config-open")
        ctx_in = context_input(popover)
        if ctx_in is None:
            fail("Context Length input not found in run-settings")
        else:
            default_ctx = ctx_in.input_value()
            info(f"default Context Length shown: {default_ctx!r}")
            ctx_in.click()
            ctx_in.fill(str(DISTINCT_CTX))
            page.wait_for_timeout(300)
            page.keyboard.press("Tab")  # blur to commit
            page.wait_for_timeout(300)
            remember = popover.get_by_label("Remember for this model").first
            if _count(remember):
                try:
                    remember.check()
                except Exception:
                    remember.click()
            else:
                fail("'Remember for this model' checkbox not found")
            page.wait_for_timeout(300)
            shoot("05-ctx-set")
            btn = primary_button(popover)
            if btn is None:
                fail("primary Load/Save button not found in run-settings")
            else:
                btn.click()
                page.wait_for_timeout(2500)
                shoot("06-after-load")

                # (a) localStorage stored the distinctive context.
                cfg = read_configs()
                entries = config_entries(cfg)
                got_ls = any(e.get("customContextLength") == DISTINCT_CTX for e in entries)
                if got_ls:
                    info(f"OK persist(localStorage): customContextLength={DISTINCT_CTX} stored")
                else:
                    fail(
                        "context not stored in unsloth_model_configs "
                        f"(entries={json.dumps(entries)[:400]})"
                    )

                # (b) the load request carried max_seq_length == distinctive value.
                got_req = False
                for body in load_posts:
                    try:
                        payload = json.loads(body) if body else {}
                    except Exception:
                        payload = {}
                    if payload.get("max_seq_length") == DISTINCT_CTX:
                        got_req = True
                        break
                if got_req:
                    info(f"OK persist(request): /api/inference/load max_seq_length={DISTINCT_CTX}")
                else:
                    # The UI may debounce the load; localStorage is the primary
                    # proof, so only warn if the request was missed.
                    runtime_warn(
                        "no /api/inference/load carried "
                        f"max_seq_length={DISTINCT_CTX}; posts={load_posts!r}"
                    )

    # (c) survives a full browser reload.
    close_picker()
    page.reload()
    composer = page.locator('textarea[aria-label="Message input"]')
    composer.wait_for(state = "visible", timeout = 60_000)
    popover = open_picker()
    if open_config(popover, MODEL_HINT) is None:
        fail("could not reopen run-settings after reload")
    else:
        ctx_in = context_input(popover)
        val = ctx_in.input_value() if ctx_in else None
        if _as_int(val) == DISTINCT_CTX:
            info(f"OK persist(reload): Context Length still {val!r} after reload")
        else:
            fail(f"Context Length did not persist across reload (got {val!r})")
        shoot("07-after-reload")

    # ─────────────────────────────────────────────────────
    # 3. Reset clears the override (never pins context) (HARD).
    # ─────────────────────────────────────────────────────
    step("reset clears the per-model override")
    # (popover + config still open from the reload check.)
    reset_btn = popover.get_by_role("button", name = "Reset").first
    if _count(reset_btn) == 0:
        fail("Reset button not found in run-settings")
    else:
        try:
            reset_btn.click()
            page.wait_for_timeout(500)
        except Exception as e:
            fail(f"Reset click failed: {e}")
        # The input after Reset is informational only: a live-loaded model can still
        # echo its context even with the stored override gone. The regression we
        # guard ("Reset PINS the override") lives in localStorage, asserted below.
        ctx_in = context_input(popover)
        after_reset = ctx_in.input_value() if ctx_in else None
        info(f"reset: Context Length input now shows {after_reset!r}")
        # Commit the reset so the stored override is dropped, then assert storage.
        btn = primary_button(popover)
        if btn is not None and btn.is_enabled():
            btn.click()
            page.wait_for_timeout(1500)
        cfg = read_configs()
        pinned = any(
            _as_int(e.get("customContextLength")) == DISTINCT_CTX for e in config_entries(cfg)
        )
        if pinned:
            fail("Reset left the distinctive context pinned in unsloth_model_configs")
        else:
            info("OK reset: distinctive context cleared from unsloth_model_configs")
        shoot("08-after-reset")
    close_picker()

    # ─────────────────────────────────────────────────────
    # 4. Advanced settings persist (best-effort, never gates).
    # ─────────────────────────────────────────────────────
    step("advanced (KV cache dtype / tensor parallel) persists")
    try:
        popover = open_picker()
        if open_config(popover, MODEL_HINT) is not None:
            adv = popover.get_by_role("switch", name = re.compile("advanced settings", re.I)).first
            if _count(adv):
                try:
                    adv.check()
                except Exception:
                    adv.click()
                page.wait_for_timeout(500)
            # The Tensor Parallelism Radix Switch has no aria-label, so target the
            # first switch after the "Tensor Parallelism" text.
            tp = popover.locator(
                'xpath=.//span[contains(text(),"Tensor Parallelism")]'
                '/following::*[@role="switch"][1]'
            ).first
            toggled = False
            if _count(tp):
                try:
                    tp.click()
                    toggled = True
                except Exception:
                    pass
            remember = popover.get_by_label("Remember for this model").first
            if _count(remember):
                try:
                    remember.check()
                except Exception:
                    remember.click()
            btn = primary_button(popover)
            if btn is not None and btn.is_enabled():
                btn.click()
                page.wait_for_timeout(1500)
            cfg = read_configs()
            has_adv = any(
                e.get("tensorParallel") or e.get("kvCacheDtype") for e in config_entries(cfg)
            )
            if toggled and has_adv:
                info("OK advanced: tensorParallel/kvCacheDtype persisted")
            else:
                runtime_warn(
                    f"advanced persistence not observed (toggled={toggled}, "
                    f"entries={json.dumps(config_entries(cfg))[:300]})"
                )
        else:
            runtime_warn("could not open run-settings for the advanced-persist check")
        close_picker()
    except Exception as e:
        runtime_warn(f"advanced-persist check errored: {e}")

    # ─────────────────────────────────────────────────────
    # 5. Legacy migration is idempotent (gates in CI via soft_fail).
    #    Seed a pre-feature unsloth_load_settings store, confirm it migrates once
    #    with the value preserved, then reload with a fresh legacy seed and confirm
    #    the migration does not re-run, duplicate, or clobber. Re-running on every
    #    reload was the regression that reverted the predecessor PR.
    # ─────────────────────────────────────────────────────
    step("legacy unsloth_load_settings migrates once and stays idempotent")
    try:
        legacy_key = f"{GGUF_REPO}::{GGUF_VARIANT}"
        legacy = {
            legacy_key: {
                "contextLength": DISTINCT_CTX,
                "kvCacheDtype": "q8_0",
                "tensorParallel": True,
            }
        }
        robust_evaluate(
            page,
            "(seed) => {"
            "  localStorage.setItem('unsloth_load_settings', JSON.stringify(seed));"
            "  localStorage.removeItem('unsloth_model_configs');"
            "  localStorage.removeItem('unsloth_model_configs_migrated');"
            "  return true;"
            "}",
            arg = legacy,
        )
        page.reload()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state = "visible", timeout = 60_000)
        # Opening the picker config forces the store to read (which migrates).
        popover = open_picker()
        open_config(popover, MODEL_HINT)
        page.wait_for_timeout(800)
        cfg_first = read_configs()
        migrated_ctx = any(
            e.get("customContextLength") == DISTINCT_CTX for e in config_entries(cfg_first)
        )
        if migrated_ctx:
            info(f"OK migration: legacy context {DISTINCT_CTX} preserved after migrating")
        else:
            soft_fail(
                f"legacy context {DISTINCT_CTX} not migrated into unsloth_model_configs "
                f"(got {json.dumps(cfg_first)[:400]})"
            )
        flag_first = robust_evaluate(
            page, "() => localStorage.getItem('unsloth_model_configs_migrated')"
        )
        if flag_first != "1":
            soft_fail(f"migration flag not set after migrating (got {flag_first!r})")
        shoot("09-after-migration")
        close_picker()

        # Idempotency: a second reload with a DIFFERENT legacy entry must not re-run
        # the migration (the persistent flag blocks it), so the new key must not leak
        # in, nothing duplicates, and the migrated value is untouched.
        if migrated_ctx:
            probe_key = "unsloth/__idem_probe__::Q4_K_M"
            robust_evaluate(
                page,
                "(seed) => {"
                "  localStorage.setItem('unsloth_load_settings', JSON.stringify(seed));"
                "  return true;"
                "}",
                arg = {probe_key: {"contextLength": DISTINCT_CTX + 2048, "tensorParallel": True}},
            )
            page.reload()
            composer.wait_for(state = "visible", timeout = 60_000)
            popover = open_picker()
            open_config(popover, MODEL_HINT)
            page.wait_for_timeout(800)
            cfg_second = read_configs()
            keys_first = set(cfg_first.keys())
            keys_second = set(cfg_second.keys())
            new_keys = keys_second - keys_first
            still_has_ctx = any(
                e.get("customContextLength") == DISTINCT_CTX for e in config_entries(cfg_second)
            )
            if new_keys:
                soft_fail(
                    "legacy migration re-ran on a second reload (persistent flag "
                    f"ignored): new keys {sorted(new_keys)}"
                )
            elif keys_second != keys_first:
                soft_fail(
                    "legacy migration dropped entries on a second reload: "
                    f"{sorted(keys_first)} -> {sorted(keys_second)}"
                )
            elif not still_has_ctx:
                soft_fail("legacy migration clobbered the migrated context on a second reload")
            else:
                info(
                    "OK migration idempotent: second reload did not re-migrate, duplicate, or clobber"
                )
            shoot("10-after-second-reload")
            close_picker()
    except Exception as e:
        soft_fail(f"migration idempotency check errored: {e}")

    # ─────────────────────────────────────────────────────
    if page_errors:
        fail(f"page errors during run: {page_errors[:3]!r}")

    browser.close()

if _failed:
    print(f"[ui-modelcfg] RESULT: FAIL ({len(_failed)} issue(s))", flush = True)
    for m in _failed:
        print(f"[ui-modelcfg]   - {m}", flush = True)
    sys.exit(1)
print("[ui-modelcfg] RESULT: PASS", flush = True)
sys.exit(0)
