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
    dump_diagnostics,
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
# Attach mode: log into an already-provisioned Unsloth with an existing password instead of the first-boot
# change-password dance. CI leaves STUDIO_LOGIN_PW unset to exercise the real change-password flow; local runs can set
# it to skip re-provisioning.
LOGIN_PW = os.environ.get("STUDIO_LOGIN_PW")
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
# Substring of the On Device picker row for the loaded model.
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT", "gemma-3-270m")
# A distinctive valid (>=128, multiple of 128, below the model's 32768 ceiling) Context Length, clearly not a default
DISTINCT_CTX = int(os.environ.get("STUDIO_DISTINCT_CTX", "4096"))
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_modelcfg")
# Settle window after run-settings opens, before staging an edit. An edit made in the panel's first moments is
# silently discarded: it re-derives its baseline once mount-time work lands and drops whatever was staged, so Save
# reports "Default settings kept" and stores nothing.
# Measured on gemma-3-270m: fails at 0ms, passes from 500ms.
# The panel exposes no readiness signal to poll (the input value, the Reset state and the primary button label are
# all identical before and after), so this is a bounded wait rather than a condition.
CONFIG_SETTLE_MS = int(os.environ.get("STUDIO_CONFIG_SETTLE_MS", "1000"))
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
LOAD_FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_LOAD_TIMEOUT_MS", "180000"))

_n = [0]
_failed: list[str] = []


# Ported from features/hub/lib/local-path.ts and features/hub/lib/model-identity.ts.
_LOCAL_PATH_PREFIX_RE = re.compile(
    r"^(?:/|\.{1,2}(?:$|[\\/])|~(?:$|[\\/])|~[^\\/]+[\\/]|[A-Za-z]:[\\/]|\\\\)"
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WSL_DRIVE_PATH_RE = re.compile(r"^/mnt/[A-Za-z](?:/|$)")


def _normalize_case_insensitive_path(path: str, min_length: int) -> str:
    slashed = path.replace("\\", "/")
    end = len(slashed)
    while end > min_length and slashed[end - 1] == "/":
        end -= 1
    return slashed[:end].lower()


def _normalize_model_identity(model_id: str) -> str:
    """Mirror of normalizeModelIdentity in features/hub/lib/model-identity.ts.

    Case folding is not unconditional there: a plain POSIX path keeps its case,
    because /models/Foo.gguf and /models/foo.gguf are different files. Only a hub
    id and the case-insensitive roots -- a Windows drive, a UNC share, a WSL mount
    -- fold. Lowercasing everything here merged paths the app keeps apart, so an
    entry belonging to one could satisfy a check that the other had saved.
    """
    trimmed = model_id.strip()
    if not (trimmed and _LOCAL_PATH_PREFIX_RE.match(trimmed)):
        return trimmed.lower()
    slash_path = trimmed.replace("\\", "/")
    if _WINDOWS_DRIVE_PATH_RE.match(trimmed):
        return _normalize_case_insensitive_path(trimmed, 3)
    if slash_path.startswith("//"):
        return _normalize_case_insensitive_path(trimmed, 2)
    if _WSL_DRIVE_PATH_RE.match(slash_path):
        return _normalize_case_insensitive_path(trimmed, 6)
    return trimmed


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
    """Number of matches, or 0.

    A raise here is not the same as no match: a closed page or a lost execution
    context also throws, and reporting that as "selector missing" sends the reader
    after the markup instead of the crash. Say so, then still return 0 so callers
    that only branch on emptiness keep working.
    """
    try:
        return loc.count()
    except Exception as exc:
        info(f"WARN: locator raised (not a missing element): {type(exc).__name__}: {exc}")
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
    if PLAYWRIGHT_BROWSER not in ("chromium", "firefox", "webkit"):
        fail(f"unsupported STUDIO_PLAYWRIGHT_BROWSER={PLAYWRIGHT_BROWSER!r}")
        sys.exit(1)
    browser_type = getattr(p, PLAYWRIGHT_BROWSER)
    launch_kwargs = {"headless": True}
    if PLAYWRIGHT_BROWSER == "chromium":
        launch_kwargs["args"] = chromium_launch_args()
        if PLAYWRIGHT_CHANNEL:
            launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL
    elif PLAYWRIGHT_CHANNEL:
        fail("STUDIO_PLAYWRIGHT_CHANNEL requires chromium")
        sys.exit(1)
    browser = browser_type.launch(**launch_kwargs)
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

    # Record every /api/inference/load POST payload so the persistence gate can assert max_seq_length.
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
        """Return the parsed unsloth_model_configs map (or {} if absent/invalid).

        Absent and unreadable are not the same: "no entry" is what several assertions
        below treat as success, so storage that failed to read must be said out loud
        rather than passed off as a clean slate.
        """
        raw = robust_evaluate(page, "() => localStorage.getItem('unsloth_model_configs')")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception as exc:
            fail(f"unsloth_model_configs is unreadable ({exc}); raw={str(raw)[:200]!r}")
            return {}
        if not isinstance(data, dict):
            fail(f"unsloth_model_configs is not an object: {type(data).__name__}")
            return {}
        return data

    def config_entries(cfg: dict) -> list[dict]:
        """The per-model entries (dict values) of the stored map, schema-tolerant."""
        return [v for v in cfg.values() if isinstance(v, dict)]

    def entries_for_model(cfg: dict) -> list[dict]:
        """Only the entries keyed to the model under test.

        The keys embed the repo id and quant (`v2:["<repo>","<quant>"]`), so scanning
        every entry lets a value belonging to a different model -- or to another quant
        of this one -- satisfy a persistence, reset or migration assertion. Both halves
        have to match. Falls back to all entries only when no key has the versioned
        shape, so a schema change degrades to the old behaviour rather than silently
        asserting nothing.
        """
        want = (_normalize_model_identity(GGUF_REPO), GGUF_VARIANT.strip().lower())
        recognised = [k for k in cfg if re.match(r"^v\d+:\[", str(k))]
        if not recognised:
            return config_entries(cfg)
        matched = []
        for key in recognised:
            # Parse the key rather than substring-searching its serialised form: the repo alone also matches this
            # repo's *other* quants, so a stale entry for one quant could stand in for the one under test and mask
            # its failed save.
            try:
                parts = json.loads(str(key).split(":", 1)[1])
            except Exception:
                continue
            if not isinstance(parts, list) or not parts:
                continue
            raw = (list(parts) + [""])[:2]
            got = (_normalize_model_identity(str(raw[0])), str(raw[1]).strip().lower())
            if got == want and isinstance(cfg[key], dict):
                matched.append(cfg[key])
        # Scoping is meaningful, so an empty result is a real answer: returning every entry here is what let another
        # model's value satisfy these checks.
        return matched

    # ─────────────────────────────────────────────────────
    if LOGIN_PW:
        # Attach mode: log in via the API and seed the token before navigation, skipping the first-boot change-password
        # dance.
        step("setup: API login + token seed (attach to running Unsloth)")
        _tok = _login_token_via_api(BASE, LOGIN_USER, LOGIN_PW)
        ctx.add_init_script(
            f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(_tok)});}}"
            f"catch(e){{}}"
        )
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)
    else:
        step("setup: change-password")
        # 3-attempt retry: the form can re-render mid-fill on slow runners and detach the password fields; each retry
        # re-navigates with a fresh page.
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
    # Unfiltered, for diagnostics: which gears exist at all when the one being looked for did not.
    # Kept as CSS rather than reusing row_gear's role lookup, because the point here is to report what is there, not to
    # match anything.
    GEAR_ANY = 'button[aria-label^="Inference settings for" i]'

    def diagnose(name, selector):
        """Screenshot + JSON sidecar (URL, body, storage) for a selector that missed.

        Without this a miss reaches the log as one line naming a selector, and the
        artifact holds no record of what the picker was actually showing -- which is
        how a picker that had closed itself read as a missing gear.
        """
        rows = []
        try:
            opts = page.locator("[data-model-picker-option]")
            rows = [
                (opts.nth(i).inner_text() or "").strip()[:60] for i in range(min(opts.count(), 12))
            ]
        except Exception:
            pass
        gears = []
        try:
            g = page.locator(GEAR_ANY)
            gears = [g.nth(i).get_attribute("aria-label") for i in range(min(g.count(), 12))]
        except Exception:
            pass
        dump_diagnostics(
            page,
            ART,
            name,
            info = info,
            extra = {"missed_selector": selector, "option_rows": rows, "gear_labels": gears},
        )
        info(f"DIAG {name}: {len(rows)} option row(s), {len(gears)} gear(s); see {name}.json")

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

    def reveal_on_device_row(popover, hint):
        """Bring the row into view without clicking it.

        Since single-quant rows collapse (#7736) the row loads its quant in one
        click and the picker closes, so selecting first would dismiss the gear
        this is about to press.
        """
        od = page.get_by_role("tab", name = "On Device").first
        if _count(od):
            od.click()
            page.wait_for_timeout(700)
        row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(hint)
                page.wait_for_timeout(700)
                row = popover.locator("[data-model-picker-option]", has_text = hint).first
        return row if _count(row) else None

    def select_on_device_row(popover, hint):
        row = reveal_on_device_row(popover, hint)
        if row is None:
            return None
        row.click()
        page.wait_for_timeout(800)
        return row

    def config_is_open(popover):
        """Back is unique to the config page and always rendered inside the picker."""
        return _count(popover.get_by_role("button", name = "Back to model list")) > 0

    # The collapsed sole-quant row appears only after an async probe lands, so an absent gear means either a multi-quant
    # repo or a probe in flight, with no DOM state to tell them apart.
    # Only a multi-quant repo pays the full wait, once per open_config.
    SOLE_QUANT_SETTLE_MS = 30_000

    # Long enough for the probe, short enough that naming a quant that is not there does not spend the whole settle
    # window before falling back to the repo.
    QUANT_GEAR_MS = 2_000

    def row_gear(
        popover,
        hint,
        quant = None,
        timeout_ms = SOLE_QUANT_SETTLE_MS,
    ):
        # The gear is a sibling of the row, not inside [data-model-picker-option], so scope it by repo id;
        # case-insensitive to match the has_text row lookup.
        #
        # The quant, when given, is anchored to the end rather than searched for anywhere in
        # the label. Every label is "<repo> <quant>", so an unanchored match lets F16 find
        # BF16, and `.first` among variants the expander orders by fit rather than by name
        # then opens the other one, after which the exact-key storage checks fail on a quant
        # that was working.
        pattern = f"^Inference settings for .*{re.escape(hint)}"
        if quant:
            pattern += f".* {re.escape(quant)}$"
        gear = popover.get_by_role(
            "button",
            name = re.compile(pattern, re.IGNORECASE),
        ).first
        try:
            gear.wait_for(state = "visible", timeout = timeout_ms)
        except Exception:
            return None
        return gear

    def open_config(popover, hint):
        if reveal_on_device_row(popover, hint) is None:
            return None
        # A sole-quant repo is a collapsed row whose click selects the model and closes
        # the picker, so click its gear without touching the row; a multi-quant repo
        # shows gears only once the row is expanded.
        #
        # The quant first at each step: with "Expand quantizations" on, the expander
        # is already mounted, so a repo-only lookup finds some gear straight away and
        # never reaches the expansion branch -- and which one it finds is then
        # arbitrary. Repo-only stays as the fallback, for the collapsed single-quant
        # row whose label carries its own quant and need not carry this one.
        gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS)
        if gear is None:
            gear = row_gear(popover, hint)
        if gear is None:
            if select_on_device_row(popover, hint) is None:
                return None
            if not popover.is_visible():
                # The probe landed mid-click, so the row selected the model; reopen for the gear that is now there.
                popover = open_picker()
                if reveal_on_device_row(popover, hint) is None:
                    return None
            gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS) or (
                row_gear(popover, hint)
            )
        if gear is None:
            return None
        gear.click()
        # Gate on the page itself rather than a sleep, so a slow mount is waited out and a failed open is not mistaken
        # for a missing Context Length input below.
        for _ in range(20):
            if config_is_open(popover):
                page.wait_for_timeout(CONFIG_SETTLE_MS)
                return popover
            page.wait_for_timeout(250)
        diagnose("open-config-not-open", 'button[name="Back to model list"]')
        return None

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    def primary_button(popover):
        # exact: get_by_role matches the accessible name as a substring by default, so
        # "Load model" also matches "Reload model" -- and it is swept first, so the
        # reload case would be found under the wrong name. The panel shows exactly one
        # of these four.
        for name in ("Load model", "Reload model", "Save settings", "Forget settings"):
            b = popover.get_by_role("button", name = name, exact = True).first
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
    # This step asserts an absence, so it passes for free if the picker renders no rows at all -- which is exactly the
    # state a broken picker is in. Prove it is populated first, or "hidden" means nothing.
    od_tab = page.get_by_role("tab", name = "On Device").first
    if _count(od_tab):
        od_tab.click()
        page.wait_for_timeout(400)
    # Waited for, not counted once: until cachedReady flips the picker renders the loading state with no rows at all,
    # so a fixed pause turns a slow cache scan into a hard failure. A populated picker attaches a row as soon as it has
    # one, so this returns immediately in the normal case and only spends the timeout when there is genuinely nothing.
    try:
        popover.locator("[data-model-picker-option]").first.wait_for(
            state = "attached", timeout = 20_000
        )
    except Exception:
        pass
    populated = _count(popover.locator("[data-model-picker-option]"))
    if populated == 0:
        fail("picker shows no rows at all, so the hidden-model check below proves nothing")
        diagnose("hidden-check-empty-picker", "[data-model-picker-option]")
    else:
        info(f"picker populated: {populated} option row(s) before the hidden check")
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
            remember = popover.get_by_label("Remember for this model").first
            if _count(remember):
                try:
                    remember.check()
                except Exception:
                    remember.click()
            else:
                fail("'Remember for this model' checkbox not found")
            ctx_in.click()
            ctx_in.fill(str(DISTINCT_CTX))
            page.wait_for_timeout(300)
            shoot("05-ctx-set")
            btn = primary_button(popover)
            if btn is None:
                fail("primary Load/Save button not found in run-settings")
            else:
                # Keep the input focused. The button click must commit the draft and use it in the same load request.
                btn.click()
                page.wait_for_timeout(2500)
                shoot("06-after-load")

                cfg = read_configs()
                entries = entries_for_model(cfg)
                got_ls = any(e.get("customContextLength") == DISTINCT_CTX for e in entries)
                if got_ls:
                    info(f"OK persist(localStorage): customContextLength={DISTINCT_CTX} stored")
                else:
                    fail(
                        "context not stored in unsloth_model_configs "
                        f"(entries={json.dumps(entries)[:400]})"
                    )

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
                    # The UI may debounce the load; localStorage is the primary proof, so only warn if the request was
                    # missed.
                    runtime_warn(
                        "no /api/inference/load carried "
                        f"max_seq_length={DISTINCT_CTX}; posts={load_posts!r}"
                    )

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
    reset_btn = popover.get_by_role("button", name = "Reset").first
    if _count(reset_btn) == 0:
        fail("Reset button not found in run-settings")
    else:
        try:
            reset_btn.click()
            page.wait_for_timeout(500)
        except Exception as e:
            fail(f"Reset click failed: {e}")
        # The input after Reset is informational only: a live-loaded model can still echo its context even with the
        # stored override gone. The regression we guard ("Reset PINS the override") lives in localStorage, asserted
        # below.
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
            _as_int(e.get("customContextLength")) == DISTINCT_CTX for e in entries_for_model(cfg)
        )
        if pinned:
            fail("Reset left the distinctive context pinned in unsloth_model_configs")
        else:
            info("OK reset: distinctive context cleared from unsloth_model_configs")
        shoot("08-after-reset")

    # ─────────────────────────────────────────────────────
    # 3b. Re-typing the value already shown must not pin an override (HARD).
    # Entering the currently displayed context commits no onChange (the value is
    # unchanged), so the cached blur value must not be replayed into a stored
    # override on Load. Otherwise re-typing the shown number, or doing so before a
    # Reset, recreates a phantom context pin. The box shows "Auto" while nothing is
    # pinned, so the number it edits is read from the focused input below.
    # ─────────────────────────────────────────────────────
    step("re-typing the shown context does not pin an override")
    # Own its state instead of inheriting the step above: the previous step commits a
    # Reset, which can close the picker, and inheriting turned that into a silent skip
    # that let this regression go unchecked.
    popover = open_picker()
    if open_config(popover, MODEL_HINT) is None:
        fail("could not open run-settings for the re-type-shown check")
        ctx_in = None
        native_default = None
    else:
        ctx_in = context_input(popover)
        # With no override stored the box reads "Auto", and only reveals the number it
        # would edit (the fitted context) once it has focus. Click first, or there is
        # nothing numeric to re-type and the step skips the regression it guards.
        if ctx_in is not None:
            ctx_in.click()
            page.wait_for_timeout(200)
        native_default = _as_int(ctx_in.input_value()) if ctx_in else None
    if ctx_in is None or native_default is None:
        # A skip here is not a pass: this step is the only guard on the phantom-pin
        # regression, so say so at the level STRICT gates rather than as prose.
        soft_fail("re-type-shown did not run: Context Length input has no numeric default")
    else:
        remember = popover.get_by_label("Remember for this model").first
        if _count(remember):
            try:
                remember.check()
            except Exception:
                remember.click()
        ctx_in.click()
        ctx_in.fill(str(native_default))
        page.wait_for_timeout(200)
        btn = primary_button(popover)
        if btn is not None and btn.is_enabled():
            # Same-click Load: the button click must commit the draft, but a draft equal to the shown value carries
            # no override, so the click must still commit the reset and leave no stored `customContextLength`.
            btn.click()
            page.wait_for_timeout(1500)
        cfg = read_configs()
        entries = entries_for_model(cfg)
        pinned = [e for e in entries if _as_int(e.get("customContextLength")) == native_default]

        # Not every stored context here is a phantom pin. model-config-page.tsx pins the
        # active context ON PURPOSE when the placement is fixed:
        #
        #   const pinFixedLayerContext =
        #     target.isGguf && loadableConfig.gpuMemoryMode === "manual" &&
        #     loadableConfig.gpuLayers != null && loadableConfig.gpuLayers >= 0 &&
        #     loadableConfig.customContextLength == null && activeLoadedContext != null;
        #
        # with the reason stated above it: "If the user fixes GPU Layers (Manual) and
        # remembers, pin that shown context so a later fresh load keeps the fitted
        # placement instead of sending native/0 and recreating the OOM." Storing the
        # context is the feature; not storing it is the bug it was written to prevent.
        #
        # This step could not tell the two apart, so on a runner where the placement IS
        # manual -- which is every CPU-only CI runner, gpuLayers 0 -- it reported the
        # documented behaviour as a regression. It went unnoticed because it inherited a
        # closed popover from the Reset step and silently skipped until #7760 made it own
        # its state; the first time it actually ran, it failed.
        expected_pin = [
            e
            for e in pinned
            if e.get("gpuMemoryMode") == "manual"
            and isinstance(e.get("gpuLayers"), int)
            and e.get("gpuLayers") >= 0
        ]
        if pinned and len(expected_pin) == len(pinned):
            info(
                f"OK re-type-shown: context {native_default} is stored, and every entry "
                f"storing it has the fixed-layer placement that pins it deliberately"
            )
        elif pinned:
            fail(
                "re-typing the shown context pinned it as an override "
                f"(customContextLength={native_default}) with no fixed-layer placement to "
                f"justify it; entries={[{k: e.get(k) for k in ('gpuMemoryMode', 'gpuLayers', 'customContextLength')} for e in pinned]}"
            )
        else:
            info("OK re-type-shown: shown context not stored as an override")
        shoot("08b-after-retype-shown")
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
            # The Tensor Parallelism Radix Switch has no aria-label, so target the first switch after the
            # "Tensor Parallelism" text.
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
                e.get("tensorParallel") or e.get("kvCacheDtype") for e in entries_for_model(cfg)
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
    #    Seed a pre-feature unsloth_load_settings store, confirm it migrates once with the value preserved, then
    #    reload with a fresh legacy seed and confirm the migration does not re-run, duplicate, or clobber. Re-running
    #    on every reload was the regression that reverted the predecessor PR.
    # ─────────────────────────────────────────────────────
    step("legacy unsloth_load_settings migrates once and stays idempotent")

    _seed_marks = [0]

    def seed_legacy_for_next_document(seed: dict, *, wipe_migrated: bool) -> None:
        """Put the legacy store in place for the NEXT document, before any app code runs.

        Not `evaluate` on the live page, which is what this used to do. Writing the
        seed into the document that is about to be discarded makes the assertion race
        that document's own in-flight work: `savePerModelConfig` in
        model-config-page.tsx runs from the `.then()` of a GET
        /api/settings/openai-auto-switch/overrides, so a response that arrives in the
        window between `removeItem('unsloth_model_configs')` and the navigation
        RE-CREATES the key from the server row. The reloaded page then finds that key
        already present, `mergeLegacyEntries` skips the legacy entry it was seeded to
        migrate (`Object.hasOwn(map, key)`), and `migrateLegacyLoadSettingsOnce`
        latches `unsloth_model_configs_migrated` anyway -- so the step reported the
        migration dropping a value when nothing had migrated at all.

        That window is roughly 20ms wide and only opens when the previous step's Load
        outruns its own wait, which is why this failed intermittently rather than
        every time: three reds in fourteen main runs, with the commit that introduced
        the write-back itself green.

        An init script runs at document start on the new page, after every write from
        the old document has gone with it, so the store the migration reads is the one
        this step asked for. The sessionStorage mark keeps it to a single document:
        init scripts cannot be removed, and re-seeding on the reload below would
        re-arm the very legacy entry the idempotency half needs to stay absent.
        """
        _seed_marks[0] += 1
        mark = f"__ui_modelcfg_seed_{_seed_marks[0]}"
        page.add_init_script(
            "(() => {\n"
            f"  const MARK = {json.dumps(mark)};\n"
            "  try {\n"
            "    if (sessionStorage.getItem(MARK)) { return; }\n"
            "    sessionStorage.setItem(MARK, '1');\n"
            f"    localStorage.setItem('unsloth_load_settings', {json.dumps(json.dumps(seed))});\n"
            + (
                "    localStorage.removeItem('unsloth_model_configs');\n"
                "    localStorage.removeItem('unsloth_model_configs_migrated');\n"
                if wipe_migrated
                else ""
            )
            + "  } catch (e) {}\n"
            "})();"
        )

    def clear_server_overrides_for_model() -> None:
        """Drop the server-side override rows for the model under test.

        This step clears localStorage and nothing else, which was fine while local
        storage was the only thing feeding the panel. It is not any more: the shared
        override row now outranks the local record, and opening the panel writes the
        row back down into `unsloth_model_configs`. Steps 2 to 4 above each mirror a
        save to the server and nothing removes it, so what this step actually measured
        was the migration racing that write-back -- and the migrated value lost
        whenever the row carried a context of its own.

        Seen on main as two different-looking failures from the same cause: an entry
        with nothing from the legacy seed in it (row written over a wiped map), and an
        entry carrying the seed's kvCacheDtype but step 3b's context (row written over
        the migrated map). Removing the row leaves the legacy import as the only thing
        that can put a value in this key, which is what the step is about.
        """
        # MODEL AND QUANT NORMALISED SEPARATELY, which is what `entries_for_model` already does to these same two
        # values. Folding the whole `<model>:<quant>` string as one identity only works while the model half folds too:
        # `normalizeModelIdentity` deliberately keeps a plain POSIX path's case, so with a local-path GGUF_REPO the row
        # `/models/Foo.gguf:UD-Q4_K_XL` normalises to itself and never matched the lowercased `...:ud-q4_k_xl` this was
        # comparing against. The stale row then survived the cleanup and the migration check went on to measure server
        # precedence instead.
        want_model = _normalize_model_identity(GGUF_REPO)
        want_quant = GGUF_VARIANT.strip().lower()

        def _is_row_for_model(key: str) -> bool:
            if _normalize_model_identity(key) == want_model:
                return True
            model, sep, quant = key.rpartition(":")
            return bool(sep) and (
                _normalize_model_identity(model) == want_model
                and quant.strip().lower() == want_quant
            )

        def rows_for_model() -> list[str] | None:
            """The override rows for this model, or None if the inventory could not be READ.

            None is not the empty list. `evaluate_fetch` reports a timeout or an HTTP error by
            returning `status == 0` / a non-None `error` rather than raising, so a request that
            never landed used to come back as "there are no override rows" -- the cleanup did
            nothing, its own post-delete verification passed on the same silence, and stale rows
            went on to contaminate the migration check with no line of output saying so.
            """
            resp = evaluate_fetch(
                page,
                f"{BASE}/api/settings/openai-auto-switch/overrides",
                headers = {"Authorization": f"Bearer {token}"},
            )
            if not resp.get("status") or resp.get("error") is not None:
                return None
            body = resp.get("body")
            if not isinstance(body, dict) or not isinstance(body.get("overrides"), dict):
                return None
            return [k for k in body["overrides"] if _is_row_for_model(str(k))]

        def remove_rows(keys: list[str]) -> None:
            for key in keys:
                evaluate_fetch(
                    page,
                    f"{BASE}/api/settings/openai-auto-switch/overrides",
                    method = "PUT",
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    body = {"model_id": key, "remove": True},
                )

        stale = rows_for_model()
        if stale is None:
            runtime_warn(
                "could not read the server override inventory, so the rows left by the earlier "
                "steps were not cleared; the migration check below may be measuring server "
                "precedence instead"
            )
            return
        remove_rows(stale)
        # AND IT HAS TO STAY REMOVED. `syncModelOverride` is fire-and-forget, so a mirror PUT
        # from steps 2 to 4 can still be in flight when this runs and recreate the row moments
        # after a single post-delete read found it gone -- which puts back exactly the
        # contamination this cleanup exists to remove. So absence is confirmed over a short
        # window rather than at one instant, and a row that comes back is removed again.
        # EVERY INTERVAL, not until the first empty one. Breaking on the first empty sample
        # confirms absence at one instant plus 250 ms, which is the same single-read weakness one
        # step further along: a queued PUT arriving in the third interval still recreates the row
        # before hydration reads it. The window is only a window if it is sampled to the end.
        left: list[str] | None = []
        for _ in range(4):
            page.wait_for_timeout(250)
            seen_now = rows_for_model()
            if seen_now is None:
                left = None
                break
            if seen_now:
                remove_rows(seen_now)
            left = seen_now
        if left is None:
            runtime_warn(
                "could not re-read the server override inventory after clearing it, so whether "
                "the rows stayed removed is unknown; the migration check below may be measuring "
                "server precedence instead"
            )
        elif left:
            # Not fatal on its own: say so rather than let the migration assertion below report the leftover row as the
            # migration losing a value.
            runtime_warn(
                f"server override rows for the model under test survived removal: {left}; "
                "the migration check below may be measuring server precedence instead"
            )
        elif stale:
            info(f"cleared {len(stale)} server override row(s) left by the earlier steps")

    def wait_for_migration_settled(timeout_ms: int = 15_000) -> str | None:
        """Block until the legacy import has actually run, and return its flag.

        The condition, not a sleep. `migrateLegacyLoadSettingsOnce` sets
        `unsloth_model_configs_migrated` as its last act on every path it takes --
        imported, nothing to import, legacy store unreadable -- so the flag appearing
        is exactly "the import has been and gone", which is what the assertions below
        need to be true before they read anything. A fixed wait either reads too early
        on a slow runner, which is how this step reported a value missing that was
        about to be written, or pads every green run with time it does not need.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        flag = None
        while time.monotonic() < deadline:
            flag = robust_evaluate(
                page, "() => localStorage.getItem('unsloth_model_configs_migrated')"
            )
            if flag is not None:
                return flag
            page.wait_for_timeout(100)
        return flag

    try:
        clear_server_overrides_for_model()
        legacy_key = f"{GGUF_REPO}::{GGUF_VARIANT}"
        legacy = {
            legacy_key: {
                "contextLength": DISTINCT_CTX,
                "kvCacheDtype": "q8_0",
                "tensorParallel": True,
                # A fingerprint, and the reason it is this field: no other step here sets Disable Vision, and
                # toApiOverride only sends disable_vision when true, so a record carrying it can only have come from
                # this seed.
                "disableVision": True,
            }
        }
        seed_legacy_for_next_document(legacy, wipe_migrated = True)
        page.reload()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state = "visible", timeout = 60_000)
        # Opening the picker config forces the store to read (which migrates).
        popover = open_picker()
        open_config(popover, MODEL_HINT)
        flag_first = wait_for_migration_settled()
        cfg_first = read_configs()
        model_entries = entries_for_model(cfg_first)
        migrated_ctx = any(e.get("customContextLength") == DISTINCT_CTX for e in model_entries)
        # Did the import run at all? Two very different failures were being reported as one: "it ran and lost the
        # context" is a bug in the migration, while "nothing from this seed is here" means the key was written by
        # something else and the import skipped it, which is what a racing write produces.
        migrated_any = any(e.get("disableVision") is True for e in model_entries)
        # AND THE PASS REQUIRES THE FINGERPRINT TOO. The context alone does not say where it came from: a server
        # override row that survived the cleanup carries DISTINCT_CTX as well -- step 3b wrote it -- and hydration can
        # put that into `model_entries` with the legacy import never having applied the seed at all.
        if migrated_ctx and migrated_any:
            info(f"OK migration: legacy context {DISTINCT_CTX} preserved after migrating")
        elif migrated_ctx:
            soft_fail(
                f"legacy context {DISTINCT_CTX} is present but NOTHING ELSE from the seed is: "
                f"disableVision is the fingerprint that only this seed sets, so the context was "
                f"put here by something other than the legacy import -- a surviving server "
                f"override row hydrating over the map is what produces this "
                f"(entries={json.dumps(model_entries)[:400]})"
            )
        elif migrated_any:
            soft_fail(
                f"legacy context {DISTINCT_CTX} was DROPPED by the migration: other fields "
                f"of the same legacy entry did land, so migrateLegacyLoadSettingsOnce ran "
                f"and lost the context (entries={json.dumps(model_entries)[:400]})"
            )
        else:
            soft_fail(
                f"legacy entry never migrated at all: nothing from the seed reached "
                f"unsloth_model_configs, so the key was already occupied when "
                f"migrateLegacyLoadSettingsOnce ran (cfg={json.dumps(cfg_first)[:400]})"
            )
        if flag_first != "1":
            soft_fail(f"migration flag not set after migrating (got {flag_first!r})")
        shoot("09-after-migration")
        close_picker()

        # Idempotency: a second reload with a DIFFERENT legacy entry must not re-run the migration (the persistent flag
        # blocks it), so the new key must not leak in, nothing duplicates, and the migrated value is untouched.
        if migrated_ctx:
            probe_key = "unsloth/__idem_probe__::Q4_K_M"
            # Same document-start seeding as the first half, and for the same reason: this one deliberately leaves
            # `unsloth_model_configs` alone, so a write racing the navigation would land in the very map the assertions
            # below compare key-for-key.
            seed_legacy_for_next_document(
                {probe_key: {"contextLength": DISTINCT_CTX + 2048, "tensorParallel": True}},
                wipe_migrated = False,
            )
            page.reload()
            composer.wait_for(state = "visible", timeout = 60_000)
            popover = open_picker()
            open_config(popover, MODEL_HINT)
            # The flag is already "1" here, so this waits on the store having been read
            # by the reloaded page rather than on the import: `readMap` is what would
            # re-run the import if the flag were being ignored, which is the regression
            # this half exists to catch.
            wait_for_migration_settled()
            cfg_second = read_configs()
            keys_first = set(cfg_first.keys())
            keys_second = set(cfg_second.keys())
            new_keys = keys_second - keys_first
            still_has_ctx = any(
                e.get("customContextLength") == DISTINCT_CTX for e in entries_for_model(cfg_second)
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
