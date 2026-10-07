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

from playwright.sync_api import expect, sync_playwright

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
    report_failing_step,
    robust_evaluate,
    step_budget_s,
    wait_for_first,
    wait_for_health,
    wait_until,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ.get("STUDIO_NEW_PW", "ModelCfg-NEW-2026!")
LOGIN_PW = os.environ.get("STUDIO_LOGIN_PW")
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT", "gemma-3-270m")
# A valid (>=128, multiple of 128, below the 32768 ceiling) non-default Context Length.
DISTINCT_CTX = int(os.environ.get("STUDIO_DISTINCT_CTX", "4096"))
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_modelcfg")
# An edit in the panel's first moments is discarded when it re-derives its baseline (fails at 0ms,
# passes from 500ms). The panel exposes no readiness signal, so this is a bounded wait.
CONFIG_SETTLE_MS = int(os.environ.get("STUDIO_CONFIG_SETTLE_MS", "1000"))
# The picker debounces its query by 300 ms (useDebouncedValue), plus margin.
PICKER_DEBOUNCE_MS = 450
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None
TURN_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_TURN_TIMEOUT_MS", "180000"))
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "720"))
FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_FETCH_TIMEOUT_MS", "30000"))
LOAD_FETCH_TIMEOUT_MS = int(os.environ.get("STUDIO_UI_LOAD_TIMEOUT_MS", "180000"))

_SLOW_LANE = max(1.0, TURN_TIMEOUT_MS / 180_000)
UI_STEP_BUDGET_S = step_budget_s(180 * _SLOW_LANE)
LOAD_STEP_BUDGET_S = step_budget_s((LOAD_FETCH_TIMEOUT_MS / 1000 + 180) * _SLOW_LANE)
NO_STEP_CEILING = 0

_n = [0]
_watchdog = None
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


def step(s: str, budget_s: float | None = None) -> None:
    """Start step `s`; it may run `budget_s` (default UI_STEP_BUDGET_S) before the run stops."""
    print(f"[ui-modelcfg] STEP {s}", flush = True)
    if _watchdog is not None:
        _watchdog.begin_step(s, UI_STEP_BUDGET_S if budget_s is None else budget_s)


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
    # Named steps kick the watchdog; total_deadline_s keeps WALL_TIMEOUT_S a whole-run cap.
    _watchdog = install_wall_clock_watchdog(
        WALL_TIMEOUT_S,
        label = "ui-modelcfg",
        info = info,
        total_deadline_s = WALL_TIMEOUT_S,
    )
    report_failing_step(_watchdog, label = "ui-modelcfg")
    # A shell health wait can pass before the auth DB migrates.
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

    load_posts: list[str] = []

    def _on_request(req):
        try:
            if req.method == "POST" and "/api/inference/load" in req.url:
                load_posts.append(req.post_data or "")
        except Exception:
            pass

    page.on("request", _on_request)

    # Settings-committing requests, tracked on the context so a replacement page is covered.
    _commits = {"started": 0, "inflight": set(), "loads": [], "loads_ended": set()}

    # validate and unload precede /load in the Load flow; untracked, their gap read as quiet and the
    # next reload aborted the /load.
    def _is_commit(req) -> bool:
        return req.method != "GET" and (
            "/api/inference/load" in req.url
            or "/api/inference/validate" in req.url
            or "/api/inference/unload" in req.url
            or "/api/settings/" in req.url
        )

    def _is_load(req) -> bool:
        return req.method == "POST" and "/api/inference/load" in req.url

    def _commit_started(req):
        try:
            if _is_commit(req):
                _commits["started"] += 1
                _commits["inflight"].add(req)
                if _is_load(req):
                    _commits["loads"].append(req)
        except Exception:
            pass

    def _commit_ended(req):
        _commits["inflight"].discard(req)
        try:
            if _is_load(req):
                _commits["loads_ended"].add(req)
        except Exception:
            pass

    ctx.on("request", _commit_started)
    ctx.on("requestfinished", _commit_ended)
    ctx.on("requestfailed", _commit_ended)

    def click_and_wait_for_commit(btn, what: str) -> None:
        """Click a Load/Save button and return once what it sent has been answered.

        Replaces a fixed 1.5-2.5 s pause. The click writes localStorage synchronously, then
        mirrors the override to the server and (for Load/Reload) loads the model; this waits
        for those requests to start and for none to be in flight for three polls in a row
        (a staged VRAM budget PUT runs before the load, so one quiet poll is not enough).
        Never raises: a click that sends nothing is logged after 10 s and the assertions after
        it decide, as they did after the fixed pause.
        """
        # A Load/Reload click is answered only by its /load; a flow stopping short goes quiet for 3 s.
        label = " ".join((btn.text_content() or "").split())
        expects_load = label in ("Load model", "Reload model")
        # Snapshot after a Playwright call: request events only arrive during one, so a restore /load
        # already sent would otherwise count as this click's.
        started = _commits["started"]
        loads_before = len(_commits["loads"])
        clicked_at = time.monotonic()
        btn.click()
        quiet = [0]
        # A request whose finish event never arrives would leave `inflight` non-empty forever.
        progress = [None, time.monotonic()]

        def _pending_loads():
            return [req for req in _commits["loads"] if req not in _commits["loads_ended"]]

        def settled():
            if _commits["started"] == started:
                return "nothing sent" if time.monotonic() - clicked_at > 10 else None
            seen = (_commits["started"], len(_commits["inflight"]), len(_commits["loads_ended"]))
            if seen != progress[0]:
                progress[:] = [seen, time.monotonic()]
            # 30 s with no change while only settings writes are open is a lost event. Open /load, /validate
            # or /unload keep the full timeout: a large GGUF teardown can take minutes.
            if (
                _commits["inflight"]
                and all("/api/settings/" in req.url for req in _commits["inflight"])
                and not _pending_loads()
                and time.monotonic() - progress[1] > 30
            ):
                return "stalled"
            if _commits["inflight"]:
                quiet[0] = 0
                return None
            quiet[0] += 1
            if expects_load:
                ours = _commits["loads"][loads_before:]
                if not any(req in _commits["loads_ended"] for req in ours):
                    if ours:
                        quiet[0] = 0
                        return None
                    return "stopped before load" if quiet[0] >= 12 else None
            return "answered" if quiet[0] >= 3 else None

        try:
            outcome = wait_until(
                settled,
                timeout_s = LOAD_FETCH_TIMEOUT_MS / 1000,
                what = f"{what}: settings requests answered",
                interval_s = 0.25,
                page = page,
            )
            if outcome == "nothing sent":
                info(f"WARN {what}: the click sent no load or settings request within 10s")
            elif outcome == "stopped before load":
                info(f"WARN {what}: the load flow ended without sending /api/inference/load")
            elif outcome == "stalled":
                info(
                    f"WARN {what}: nothing changed for 30s with only settings writes open; still counted in "
                    f"flight: {sorted(f'{req.method} {req.url}' for req in _commits['inflight'])}"
                )
                _commits["inflight"].clear()
        except TimeoutError as exc:
            info(
                f"WARN {exc}; in flight: "
                f"{sorted(f'{req.method} {req.url}' for req in _commits['inflight'])}, "
                f"/load not ended: {[req.url for req in _pending_loads()]}"
            )

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
            # Parse the key: a substring match on the repo also matches its other quants.
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
        # An empty result is a real answer; returning everything let another model satisfy the checks.
        return matched

    if LOGIN_PW:
        step("setup: API login + token seed (attach to running Unsloth)", NO_STEP_CEILING)
        _tok = _login_token_via_api(BASE, LOGIN_USER, LOGIN_PW)
        ctx.add_init_script(
            f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(_tok)});}}"
            f"catch(e){{}}"
        )
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)
    else:
        step("setup: change-password", NO_STEP_CEILING)
        # The form can re-render mid-fill on slow runners and detach the password fields.
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
                        # ENOBUFS has nothing to poll; only time helps.
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
    load_posts.clear()

    POPOVER = '[data-tour="chat-model-selector-popover"]'
    TRIGGER = '[data-tour="chat-model-selector"]'
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
            popover = page.locator(POPOVER).first
        popover.wait_for(state = "visible", timeout = 30_000)
        return popover

    def close_picker():
        try:
            page.keyboard.press("Escape")
            page.locator(POPOVER).first.wait_for(state = "hidden", timeout = 10_000)
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
        # Until the cache scan lands the tab renders no rows at all.
        wait_for_first(popover.locator("[data-model-picker-option]"), timeout_ms = 20_000)
        row = popover.locator("[data-model-picker-option]", has_text = hint).first
        if _count(row) == 0:
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(hint)
                wait_for_first(
                    popover.locator("[data-model-picker-option]", has_text = hint),
                    timeout_ms = 10_000,
                )
                row = popover.locator("[data-model-picker-option]", has_text = hint).first
        return row if _count(row) else None

    def select_on_device_row(popover, hint):
        row = reveal_on_device_row(popover, hint)
        if row is None:
            return None
        row.click()
        try:
            wait_until(
                lambda: not popover.is_visible() or _count(popover.locator(GEAR_ANY)) > 0,
                timeout_s = 10,
                what = "the row click to close the picker or show its gears",
                interval_s = 0.1,
                page = page,
            )
        except TimeoutError as exc:
            info(f"WARN {exc}")
        return row

    def config_is_open(popover):
        """Back is unique to the config page and always rendered inside the picker."""
        return _count(popover.get_by_role("button", name = "Back to model list")) > 0

    # The sole-quant row appears only after an async probe, indistinguishable from a multi-quant repo.
    SOLE_QUANT_SETTLE_MS = 30_000

    QUANT_GEAR_MS = 2_000

    def row_gear(
        popover,
        hint,
        quant = None,
        timeout_ms = SOLE_QUANT_SETTLE_MS,
    ):
        # The gear is a sibling of the row. Anchor the quant at the end, or F16 matches BF16.
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
        # A sole-quant row click selects the model and closes the picker, so click its gear directly.
        # Quant first: with "Expand quantizations" on, a repo-only lookup picks an arbitrary gear.
        gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS)
        if gear is None:
            gear = row_gear(popover, hint)
        if gear is None:
            if select_on_device_row(popover, hint) is None:
                return None
            if not popover.is_visible():
                popover = open_picker()
                if reveal_on_device_row(popover, hint) is None:
                    return None
            gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS) or (
                row_gear(popover, hint)
            )
        if gear is None:
            return None
        gear.click()
        if (
            wait_for_first(
                popover.get_by_role("button", name = "Back to model list"), timeout_ms = 5_000
            )
            is not None
        ):
            page.wait_for_timeout(CONFIG_SETTLE_MS)
            return popover
        diagnose("open-config-not-open", 'button[name="Back to model list"]')
        return None

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    PRIMARY_BUTTON_NAMES = re.compile(r"^(Load model|Reload model|Save settings|Forget settings)$")

    def primary_button(popover):
        # Anchored: name matching is substring by default, so "Load model" matches "Reload model".
        # One locator for all names: the label flips once the runtime store fills in.
        b = popover.get_by_role("button", name = PRIMARY_BUTTON_NAMES).first
        return b if _count(b) else None

    # 1. Hidden infra models absent from the picker (HARD).
    step("hidden infra models absent from picker")
    popover = open_picker()
    shoot("02-picker-open")
    needles = ["bge-small-en-v1.5", "stories260"]
    tabs = ["Recommended", "On Device", "Connected"]
    hidden_ok = True
    # An absence check passes on an empty picker, so prove it is populated first.
    od_tab = page.get_by_role("tab", name = "On Device").first
    if _count(od_tab):
        od_tab.click()
    # Until cachedReady flips the picker renders no rows.
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
                expect(tab).to_have_attribute("aria-selected", "true", timeout = 5_000)
            except Exception:
                continue
            search = popover.locator("[data-model-picker-search-input]").first
            if _count(search):
                search.click()
                search.fill(needle)
                # Outlast the 300 ms debounce, then wait until every row matches the needle: an empty list cannot
                # tell filtered from not-yet-applied.
                page.wait_for_timeout(PICKER_DEBOUNCE_MS)
                try:
                    page.wait_for_function(
                        """([sel, needle]) => {
                            const root = document.querySelector(sel);
                            if (!root) return true;
                            const rows = root.querySelectorAll("[data-model-picker-option]");
                            return Array.from(rows).every(
                                (r) => (r.innerText || "").toLowerCase().includes(needle)
                            );
                        }""",
                        arg = [POPOVER, needle.lower()],
                        timeout = 5_000,
                    )
                except Exception as exc:
                    info(
                        f"WARN filter for {needle!r} in '{tab_name}' did not settle: {type(exc).__name__}"
                    )
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
    if hidden_ok:
        info("OK hidden: bge-small-en-v1.5 + stories260K absent from every picker tab")
    shoot("03-hidden-check")
    close_picker()

    # 2. Context Length persists (load + request + reload) (HARD).
    step(f"context length {DISTINCT_CTX} persists", LOAD_STEP_BUDGET_S)
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
            expect(ctx_in).to_have_value(str(DISTINCT_CTX), timeout = 5_000)
            shoot("05-ctx-set")
            btn = primary_button(popover)
            if btn is None:
                fail("primary Load/Save button not found in run-settings")
            else:
                # Keep the input focused: the click must commit the draft into the same load request.
                click_and_wait_for_commit(btn, "context length Load")
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
        # The popover can render before the stored config is applied, so wait for the stored value.
        val = None
        deadline = time.monotonic() + 15
        while True:
            ctx_in = context_input(popover)
            val = ctx_in.input_value() if ctx_in else None
            if _as_int(val) == DISTINCT_CTX or time.monotonic() >= deadline:
                break
            page.wait_for_timeout(250)
        # With the model loaded the field is seeded from the runtime, so re-read the stored entry.
        stored = [
            e
            for e in entries_for_model(read_configs())
            if e.get("customContextLength") == DISTINCT_CTX
        ]
        if not stored:
            fail(
                "the remembered per-model entry lost customContextLength across reload "
                f"(entries={json.dumps(entries_for_model(read_configs()))[:400]})"
            )
        elif _as_int(val) == DISTINCT_CTX:
            info(f"OK persist(reload): stored entry and Context Length both {val!r} after reload")
        else:
            fail(f"Context Length did not persist across reload (got {val!r} after 15s)")
        shoot("07-after-reload")

    # 3. Reset clears the override (never pins context) (HARD).
    step("reset clears the per-model override", LOAD_STEP_BUDGET_S)
    reset_btn = popover.get_by_role("button", name = "Reset").first
    if _count(reset_btn) == 0:
        fail("Reset button not found in run-settings")
    else:
        try:
            reset_btn.click()
            try:
                expect(reset_btn).to_be_disabled(timeout = 5_000)
            except AssertionError:
                info("WARN Reset did not report the defaults restored within 5s")
        except Exception as e:
            fail(f"Reset click failed: {e}")
        # A live-loaded model can echo its context after Reset; storage is what is asserted.
        ctx_in = context_input(popover)
        after_reset = ctx_in.input_value() if ctx_in else None
        info(f"reset: Context Length input now shows {after_reset!r}")
        btn = primary_button(popover)
        if btn is not None and btn.is_enabled():
            click_and_wait_for_commit(btn, "reset commit")
        cfg = read_configs()
        pinned = any(
            _as_int(e.get("customContextLength")) == DISTINCT_CTX for e in entries_for_model(cfg)
        )
        if pinned:
            fail("Reset left the distinctive context pinned in unsloth_model_configs")
        else:
            info("OK reset: distinctive context cleared from unsloth_model_configs")
        shoot("08-after-reset")

    # 3b. Re-typing the shown value must not pin an override (HARD): it commits no onChange, so the
    # cached blur value must not be replayed. The box shows "Auto" until focused.
    step("re-typing the shown context does not pin an override", LOAD_STEP_BUDGET_S)
    # Own its state: the Reset step can close the picker, which turned this into a silent skip.
    popover = open_picker()
    if open_config(popover, MODEL_HINT) is None:
        fail("could not open run-settings for the re-type-shown check")
        ctx_in = None
        native_default = None
    else:
        ctx_in = context_input(popover)
        if ctx_in is not None:
            ctx_in.click()
            try:
                wait_until(
                    lambda: _as_int(ctx_in.input_value()) is not None,
                    timeout_s = 5,
                    what = "Context Length to show a number once focused",
                    interval_s = 0.05,
                    page = page,
                )
            except TimeoutError as exc:
                info(f"WARN {exc}")
        native_default = _as_int(ctx_in.input_value()) if ctx_in else None
    if ctx_in is None or native_default is None:
        # A skip is not a pass: this step is the only guard on the phantom-pin regression.
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
        expect(ctx_in).to_have_value(str(native_default), timeout = 5_000)
        btn = primary_button(popover)
        if btn is not None and btn.is_enabled():
            click_and_wait_for_commit(btn, "re-typed context Load")
        cfg = read_configs()
        entries = entries_for_model(cfg)
        pinned = [e for e in entries if _as_int(e.get("customContextLength")) == native_default]

        # Not every stored context is a phantom pin: model-config-page.tsx pins the active context on purpose
        # for a manual GPU-layer placement (pinFixedLayerContext), which includes every CPU-only runner.
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

    # 4. Advanced settings persist (best-effort, never gates).
    step("advanced (KV cache dtype / tensor parallel) persists", LOAD_STEP_BUDGET_S)
    try:
        popover = open_picker()
        if open_config(popover, MODEL_HINT) is not None:
            adv = popover.get_by_role("switch", name = re.compile("advanced settings", re.I)).first
            if _count(adv):
                try:
                    adv.check()
                except Exception:
                    adv.click()
                try:
                    expect(adv).to_be_checked(timeout = 5_000)
                except AssertionError:
                    pass
            # The Tensor Parallelism switch has no aria-label, so target it by the preceding text.
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
                click_and_wait_for_commit(btn, "advanced settings Load")
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

    # 5. Legacy unsloth_load_settings migrates once and stays idempotent (soft_fail gates in CI).
    step("legacy unsloth_load_settings migrates once and stays idempotent", LOAD_STEP_BUDGET_S)

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
        # Normalise model and quant separately: normalizeModelIdentity keeps a POSIX path's case.
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
        # syncModelOverride is fire-and-forget, so a late PUT can recreate the row; sample the whole
        # window and remove it again each time.
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
                # Fingerprint: only this seed sets disableVision (toApiOverride sends it only when true).
                "disableVision": True,
            }
        }
        seed_legacy_for_next_document(legacy, wipe_migrated = True)
        page.reload()
        composer = page.locator('textarea[aria-label="Message input"]')
        composer.wait_for(state = "visible", timeout = 60_000)
        # Opening the picker config makes the store read, which migrates.
        popover = open_picker()
        open_config(popover, MODEL_HINT)
        flag_first = wait_for_migration_settled()
        cfg_first = read_configs()
        model_entries = entries_for_model(cfg_first)
        migrated_ctx = any(e.get("customContextLength") == DISTINCT_CTX for e in model_entries)
        # Separate "ran and lost the context" from "skipped because something else wrote the key".
        migrated_any = any(e.get("disableVision") is True for e in model_entries)
        # Require the fingerprint too: a surviving server override row also carries DISTINCT_CTX.
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

        # Idempotency: a second reload with a different legacy entry must not re-run the migration.
        if migrated_ctx:
            probe_key = "unsloth/__idem_probe__::Q4_K_M"
            # Seed at document start so no racing write lands in the map compared key-for-key.
            seed_legacy_for_next_document(
                {probe_key: {"contextLength": DISTINCT_CTX + 2048, "tensorParallel": True}},
                wipe_migrated = False,
            )
            page.reload()
            composer.wait_for(state = "visible", timeout = 60_000)
            popover = open_picker()
            open_config(popover, MODEL_HINT)
            # The flag is already "1", so wait for readMap, which would re-run the import if it were ignored.
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
