# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

""" "Estimated Memory Usage" row Playwright regression test (GPU-free, no model load).

The row lives in the model-picker's run-settings panel (MemoryEstimateRow in
features/model-picker/components/model-config-page.tsx) and it renders ONLY when the
selected target is a GGUF and POST /api/inference/estimate-memory answers
`available: true`. When it cannot answer it returns `null` -- the row HIDES rather
than erroring -- which is why a screenshot of this panel proves nothing on its own:
an unavailable estimate, a 404 from a backend predating the route, and a row that
was never implemented all look identical on screen.

So this drives the row against a RECORDED API exchange. Every
/api/inference/estimate-memory call is intercepted with `context.route`, its POST body
is captured, and the response this test hands back is the one the row is then asserted
to be displaying. What that buys, gate by gate:

  - Request carries the settings: typing a distinctive Context Length must produce a
    NEW estimate POST whose `n_ctx` is that number, and whose body still carries the
    rest of the load settings the panel is supposed to price (model_path, cache_type_kv,
    n_parallel, gpu_memory_mode). A row that re-renders without re-pricing, or one wired
    to a stale request object, fails here (HARD).
  - Response reaches the screen: the figures shown are the ones the stub returned, to
    the byte -- `formatBytesGiB` of the stubbed totals, and, once the row is expanded,
    the breakdown lines plus the KV note built from the echoed context and cache dtype
    (HARD).
  - The row hides rather than errors: `available: false` hides it, restoring the
    available response brings it BACK -- without that half, "hidden" is satisfied by a
    panel that simply died -- and an HTTP 404, the answer from a backend predating the
    route, hides it too, with no page error (HARD).

Stubbing is deliberate. The real endpoint needs a GGUF whose header is on this disk and
answers with whatever that machine's memory happens to be, so a test asserting real
figures would either assert nothing specific or be a hardware report. The request half
is not stubbed: it is the panel's own, unmodified.

Runs as a plain script (not via pytest), mirroring tests/studio/playwright_model_config.py:
accumulate failures in `_failed`, exit non-zero if any HARD gate failed. With
STUDIO_UI_STRICT=1 (as CI sets), soft_fail also gates; genuinely-optional checks use
runtime_warn so they never flake the merge gate.

Honours STUDIO_PLAYWRIGHT_BROWSER in {chromium, firefox, webkit}, like every sibling
scene. The locale is pinned to en-US because one assertion reads a number the app
formatted with `toLocaleString`.
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
    install_view_transition_killer,
    install_wall_clock_watchdog,
    is_benign_page_error,
    recover_or_replace_page,
    wait_for_health,
)

BASE = os.environ["BASE_URL"]
NEW = os.environ.get("STUDIO_NEW_PW", "MemEst-NEW-2026!")
# Attach mode: log into an already-provisioned Unsloth with an existing password instead of the first-boot
# change-password dance. The CI step runs this after the model-config scene against the SAME server, whose bootstrap
# password that scene has already rotated, so there is no change-password flow left to drive there.
LOGIN_PW = os.environ.get("STUDIO_LOGIN_PW")
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/gemma-3-270m-it-GGUF")
GGUF_VARIANT = os.environ.get("GGUF_VARIANT", "UD-Q4_K_XL")
# Substring of the picker row for the model whose run settings this drives.
#
# From GGUF_REPO rather than a bare family name, and not cosmetically: the row lookup
# takes `.first`, so a hint matching a NON-GGUF sibling opens whichever the picker orders
# first. On a cache holding both `gemma-3-270m-it-GGUF` and
# `gemma-3-270m-it-unsloth-bnb-4bit`, "gemma-3-270m" opened the bnb-4bit safetensors
# panel -- no Context Length control, no memory row, the row being GGUF-only -- and the
# suite reported the feature missing. The repo's own name cannot collide that way.
MODEL_HINT = os.environ.get("STUDIO_MODEL_HINT") or GGUF_REPO.rsplit("/", 1)[-1]
# Context Lengths for the four re-prices. Each must be valid (>=128, a multiple of 128, under the model's ceiling) and
# DIFFERENT from what the control is showing at the time. That last part is a property of the control, not a precaution:
# the box reads "Auto" until focused and then reveals the context the load would fit to (8192 here), and typing the
# number already displayed commits no onChange, so the request key never moves and no re-price happens. Measured: typing
# 8192 into a box showing 8192 produced zero requests; every other value produced one. Hence chosen at the moment of
# typing, against what the box says, rather than fixed per step.
CTX_CANDIDATES = [
    int(part)
    for part in os.environ.get("STUDIO_CTX_CANDIDATES", "6144,5120,4096,3072,2048,1536").split(",")
    if part.strip()
]
ART_DIR = os.environ.get("PW_ART_DIR", "logs/playwright_memory_estimate")
# Settle window after run-settings opens, before staging an edit.
# Same reason as playwright_model_config.py: an edit made in the panel's first moments is discarded when the panel
# re-derives its baseline once mount-time work lands.
CONFIG_SETTLE_MS = int(os.environ.get("STUDIO_CONFIG_SETTLE_MS", "1000"))
ART = Path(ART_DIR)
ART.mkdir(parents = True, exist_ok = True)
STRICT = os.environ.get("STUDIO_UI_STRICT", "0") == "1"
PLAYWRIGHT_BROWSER = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium").lower()
PLAYWRIGHT_CHANNEL = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL") or None
WALL_TIMEOUT_S = float(os.environ.get("STUDIO_UI_WALL_TIMEOUT_S", "600"))
# The hook debounces at 250ms and the fetch is intercepted in-process, so a re-price
# lands in well under a second; this is the "it is never coming" bound.
ESTIMATE_WAIT_MS = int(os.environ.get("STUDIO_UI_ESTIMATE_WAIT_MS", "20000"))

TRANSCRIPT_NAME = "memory-estimate-exchanges.json"

GIB = 1024**3
# Exact quarter-GiB figures on purpose: `formatBytesGiB` is `(bytes / 1024**3).toFixed(2)`, and a quarter of a GiB is
# exactly representable, so the string the app renders is predictable to the last digit on every engine rather than a
# rounding argument.
STUB_WEIGHTS_BYTES = int(3.25 * GIB)
STUB_KV_BYTES = int(1.50 * GIB)
STUB_COMPUTE_BYTES = int(0.75 * GIB)
STUB_TOTAL_BYTES = STUB_WEIGHTS_BYTES + STUB_KV_BYTES + STUB_COMPUTE_BYTES  # 5.50 GiB
STUB_GPU_BYTES = int(4.75 * GIB)
STUB_LAYER_COUNT = 27
STUB_GPU_LAYERS = 12


def _gib(num_bytes: int) -> str:
    """Python-side mirror of `formatBytesGiB` in `lib/memory/format.ts`.

    The label is GiB, not GB, and that is the assertion rather than a detail.
    This mirror used to print "GB" because the panel did, so the test agreed
    with the app about a divide by 1024**3 that both of them called a decimal
    gigabyte. Consolidating the formatters corrected the app; correcting the
    mirror to match is what keeps this test measuring the app rather than
    re-stating whatever the app currently happens to do.

    Only the unit moved. Every figure here is byte-for-byte what it was.
    """
    return f"{num_bytes / GIB:.2f} GiB"


_n = [0]
_failed: list[str] = []


def step(s: str) -> None:
    print(f"[ui-memest] STEP {s}", flush = True)


def info(s: str) -> None:
    print(f"[ui-memest] {s}", flush = True)


def fail(m: str) -> None:
    print(f"[ui-memest] FAIL: {m}", flush = True)
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

    A raise here is not the same as no match: a closed page or a lost execution context
    also throws, and reporting that as "selector missing" sends the reader after the
    markup instead of the crash. Say so, then still return 0.
    """
    try:
        return loc.count()
    except Exception as exc:
        info(f"WARN: locator raised (not a missing element): {type(exc).__name__}: {exc}")
        return 0


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


# ─────────────────────────────────────────────────────
# The stubbed endpoint. `_mode` decides what the NEXT call answers; every call is
# recorded, request and response, and the whole transcript is written to the artifact
# directory so a failure can be read after the fact.
# ─────────────────────────────────────────────────────
_mode = ["available"]
exchanges: list[dict] = []

UNAVAILABLE_BODY = {
    "available": False,
    # The reason a real backend gives for a GGUF whose file is not on this disk, which is the commonest way a user meets
    # the hidden row.
    "reason": "not_downloaded",
    "weights_bytes": 0,
    "kv_bytes": 0,
    "compute_bytes": 0,
    "drafter_runtime_bytes": 0,
    "drafter_runtime_gpu_bytes": 0,
    "projector_runtime_bytes": 0,
    "drafter_kv_unsized": False,
    "total_bytes": 0,
    "gpu_bytes": 0,
    "kv_estimable": False,
    "kv_on_gpu": True,
    "n_ctx": 0,
    "cache_type_kv": None,
    "n_parallel": 1,
    "layer_count": None,
    "gpu_layers": None,
    "moe_offload_unmodelled": False,
}


def _available_body(request_payload: dict) -> dict:
    """An available estimate whose context / cache dtype / slot count ECHO the request.

    Echoing is what turns the KV note into a round-trip assertion: the note is built
    from `nCtx` and `cacheTypeKv` off the RESPONSE, so seeing the context that was typed
    into the control appear there proves the value travelled control -> request ->
    response -> DOM, not merely that a number rendered.
    """
    raw_ctx = request_payload.get("n_ctx")
    n_ctx = int(raw_ctx) if isinstance(raw_ctx, (int, float)) and raw_ctx else 0
    raw_parallel = request_payload.get("n_parallel")
    n_parallel = int(raw_parallel) if isinstance(raw_parallel, (int, float)) and raw_parallel else 1
    return {
        "available": True,
        "reason": None,
        "weights_bytes": STUB_WEIGHTS_BYTES,
        "kv_bytes": STUB_KV_BYTES,
        "compute_bytes": STUB_COMPUTE_BYTES,
        "drafter_runtime_bytes": 0,
        "drafter_runtime_gpu_bytes": 0,
        "projector_runtime_bytes": 0,
        "drafter_kv_unsized": False,
        "total_bytes": STUB_TOTAL_BYTES,
        "gpu_bytes": STUB_GPU_BYTES,
        "kv_estimable": True,
        "kv_on_gpu": True,
        "n_ctx": n_ctx,
        "cache_type_kv": request_payload.get("cache_type_kv") or "f16",
        "n_parallel": n_parallel,
        "layer_count": STUB_LAYER_COUNT,
        "gpu_layers": STUB_GPU_LAYERS,
        "moe_offload_unmodelled": False,
    }


def _handle_estimate(route) -> None:
    request = route.request
    raw = ""
    try:
        raw = request.post_data or ""
    except Exception:
        raw = ""
    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    mode = _mode[0]
    record: dict = {
        "ts": time.time(),
        "mode": mode,
        "method": request.method,
        "url": request.url,
        "request": payload,
        "raw_request": raw[:2000],
    }
    exchanges.append(record)
    if mode == "http404":
        body = {"detail": "Not Found"}
        record["response_status"] = 404
        record["response"] = body
        route.fulfill(
            status = 404,
            content_type = "application/json",
            body = json.dumps(body),
        )
        return
    body = dict(UNAVAILABLE_BODY) if mode == "unavailable" else _available_body(payload)
    record["response_status"] = 200
    record["response"] = body
    route.fulfill(status = 200, content_type = "application/json", body = json.dumps(body))


def write_transcript() -> None:
    """The recorded API exchange, in the artifact directory the sibling scenes use."""
    try:
        (ART / TRANSCRIPT_NAME).write_text(
            json.dumps(
                {
                    "base_url": BASE,
                    "browser": PLAYWRIGHT_BROWSER,
                    "model": {"repo": GGUF_REPO, "variant": GGUF_VARIANT},
                    "stub": {
                        "weights_bytes": STUB_WEIGHTS_BYTES,
                        "kv_bytes": STUB_KV_BYTES,
                        "compute_bytes": STUB_COMPUTE_BYTES,
                        "total_bytes": STUB_TOTAL_BYTES,
                        "gpu_bytes": STUB_GPU_BYTES,
                    },
                    "exchanges": exchanges,
                },
                indent = 2,
                default = str,
            ),
            encoding = "utf-8",
        )
        info(f"wrote {len(exchanges)} estimate exchange(s) to {ART / TRANSCRIPT_NAME}")
    except Exception as exc:
        info(f"WARN: could not write the exchange transcript: {exc}")


with sync_playwright() as p:
    _watchdog = install_wall_clock_watchdog(WALL_TIMEOUT_S, label = "ui-memest", info = info)
    # Health pre-flight: a bash-side health wait can pass before the auth DB migrates.
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
        # One assertion reads a number the app formatted with `toLocaleString`, whose group separator is a property of
        # the locale. Left to the engine's default, 8192 renders "8,192" on one runner and "8 192" on another, and the
        # gate would be measuring the runner.
        locale = "en-US",
    )
    install_view_transition_killer(ctx)
    # On the CONTEXT, not the page: a page replaced by the recovery helper below would otherwise lose the
    # interception and reach the real endpoint, which on a runner with no GGUF answers unavailable, so the row would
    # hide and this would read as the feature being broken.
    ctx.route("**/api/inference/estimate-memory*", _handle_estimate)
    page = ctx.new_page()
    page.set_default_timeout(60_000)
    page_errors: list[str] = []

    def _on_pageerror(e):
        msg = str(e)
        if is_benign_page_error(msg):
            info(f"WARN ignoring benign pageerror: {msg!r}")
            return
        page_errors.append(msg)

    page.on("pageerror", _on_pageerror)

    def shoot(name: str) -> None:
        _n[0] += 1
        try:
            page.screenshot(
                path = str(ART / f"{_n[0]:02d}-{name}.png"),
                full_page = True,
                timeout = 90_000,
                animations = "disabled",
            )
        except Exception as exc:
            info(f"WARN: screenshot {name} failed: {exc}")

    def diagnose(name: str, missed: str) -> None:
        rows = []
        try:
            opts = page.locator("[data-model-picker-option]")
            rows = [
                (opts.nth(i).inner_text() or "").strip()[:60] for i in range(min(opts.count(), 12))
            ]
        except Exception:
            pass
        dump_diagnostics(
            page,
            ART,
            name,
            info = info,
            extra = {
                "missed_selector": missed,
                "option_rows": rows,
                "estimate_exchanges": exchanges[-4:],
                "mode": _mode[0],
            },
        )

    # ─────────────────────────────────────────────────────
    if LOGIN_PW:
        step("setup: API login + token seed (attach to running Unsloth)")
        _tok = _login_token_via_api(BASE, LOGIN_USER, LOGIN_PW)
        ctx.add_init_script(
            f"try{{localStorage.setItem('unsloth_auth_token', {json.dumps(_tok)});}}catch(e){{}}"
        )
        page.goto(BASE, wait_until = "domcontentloaded", timeout = 60_000)
    else:
        step("setup: change-password")
        form_err: Exception | None = None
        for _attempt in range(3):
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
                    info = lambda m: print(f"[ui-memest]   {m}", flush = True),
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
                    f"[ui-memest]   change-password attempt {_attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}; page.url={cur_url}",
                    flush = True,
                )
                if _attempt < 2:
                    page = recover_or_replace_page(
                        page,
                        ctx,
                        default_timeout_ms = 60_000,
                        info = lambda m: print(f"[ui-memest]   recovery: {m}", flush = True),
                    )
                    page.on("pageerror", _on_pageerror)
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
            shoot(f"00-composer-wait-attempt-{_attempt + 1}-fail")
            if _attempt == 0:
                page = recover_or_replace_page(
                    page,
                    ctx,
                    default_timeout_ms = 60_000,
                    goto_url = BASE,
                    settle_networkidle = True,
                    info = lambda m: print(f"[ui-memest]   recovery: {m}", flush = True),
                )
                page.on("pageerror", _on_pageerror)
                composer = page.locator('textarea[aria-label="Message input"]')
    if last_err is not None:
        raise last_err
    shoot("01-chat-loaded")

    # ─────────────────────────────────────────────────────
    # Picker helpers (same proven selectors as playwright_model_config.py).
    # ─────────────────────────────────────────────────────
    POPOVER = '[data-tour="chat-model-selector-popover"]'
    TRIGGER = '[data-tour="chat-model-selector"]'
    SOLE_QUANT_SETTLE_MS = 30_000
    QUANT_GEAR_MS = 2_000

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
        """Bring the row into view without clicking it: a single-quant row loads its
        quant on click and closes the picker, taking the gear with it."""
        od = page.get_by_role("tab", name = "On Device").first
        if _count(od):
            od.click()
            page.wait_for_timeout(700)
        try:
            popover.locator("[data-model-picker-option]").first.wait_for(
                state = "attached", timeout = 20_000
            )
        except Exception:
            pass
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

    def row_gear(
        popover,
        hint,
        quant = None,
        timeout_ms = SOLE_QUANT_SETTLE_MS,
    ):
        # The gear is a sibling of the row, not inside [data-model-picker-option]. The
        # quant is anchored to the end: every label is "<repo> <quant>", so an unanchored
        # match lets F16 find BF16 and `.first` then opens a variant this test never named.
        pattern = f"^Inference settings for .*{re.escape(hint)}"
        if quant:
            pattern += f".* {re.escape(quant)}$"
        gear = popover.get_by_role("button", name = re.compile(pattern, re.IGNORECASE)).first
        try:
            gear.wait_for(state = "visible", timeout = timeout_ms)
        except Exception:
            return None
        return gear

    def config_is_open(popover):
        """Back is unique to the config page and always rendered inside the picker."""
        return _count(popover.get_by_role("button", name = "Back to model list")) > 0

    def open_config(popover, hint):
        if reveal_on_device_row(popover, hint) is None:
            diagnose("no-picker-row", f"[data-model-picker-option] has_text={hint!r}")
            return None
        # The quant first: with "Expand quantizations" on, a repo-only lookup finds some gear straight away and which
        # one is arbitrary. Repo-only stays as the fallback, for the collapsed sole-quant row whose label carries its
        # own quant.
        gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS)
        if gear is None:
            gear = row_gear(popover, hint)
        if gear is None:
            # A multi-quant repo shows gears only once the row is expanded, and clicking a collapsed sole-quant row
            # loads it and closes the picker -- so reopen and look again rather than treating a closed picker as a
            # missing gear.
            if select_on_device_row(popover, hint) is None:
                diagnose("no-row-gear", f"Inference settings for ...{hint}")
                return None
            if not popover.is_visible():
                popover = open_picker()
                if reveal_on_device_row(popover, hint) is None:
                    diagnose("no-row-gear-after-reopen", f"[data-model-picker-option] {hint!r}")
                    return None
            gear = row_gear(popover, hint, quant = GGUF_VARIANT, timeout_ms = QUANT_GEAR_MS) or (
                row_gear(popover, hint)
            )
        if gear is None:
            diagnose("no-row-gear", f"Inference settings for ...{hint}")
            return None
        gear.click()
        for _ in range(40):
            if config_is_open(popover):
                page.wait_for_timeout(CONFIG_SETTLE_MS)
                return popover
            page.wait_for_timeout(250)
        diagnose("config-not-open", 'button[name="Back to model list"]')
        return None

    def context_input(popover):
        for role in ("textbox", "spinbutton"):
            loc = popover.get_by_role(role, name = "Context Length").first
            if _count(loc):
                return loc
        loc = popover.locator('input[aria-label="Context Length"]').first
        return loc if _count(loc) else None

    # ─────────────────────────────────────────────────────
    # Row helpers. The toggle button is the anchor for everything: the header row is its
    # parent, and the breakdown is whatever its aria-controls names -- both stated by the
    # component itself, so neither depends on a class name that a restyle can move.
    # ─────────────────────────────────────────────────────
    ESTIMATE_LABEL = re.compile(r"Estimated Memory Usage", re.I)
    # The run-settings panel exists more than once in this document -- the picker keeps its own copy mounted behind the
    # one on screen -- so a bare `.first` is a coin toss, and the copy it landed on in CI was one the user cannot see:
    # the row was painted, screenshotted, and reported missing for twenty seconds. Take the first match that is actually
    # on screen, and say which strategy found it, because "the row is not there" and "the row is there twice" are the
    # same failure otherwise.
    _match_note: list[str] = []

    def _first_visible(loc, label: str):
        """The first on-screen match of `loc`, or None. Never raises."""
        try:
            total = loc.count()
        except Exception:
            return None
        for i in range(total):
            candidate = loc.nth(i)
            try:
                if candidate.is_visible():
                    if label not in _match_note:
                        _match_note.append(label)
                        info(f"row located by {label} ({total} match(es) in the document)")
                    return candidate
            except Exception:
                continue
        return None

    def estimate_button():
        """The visible toggle, by role first and by markup second.

        The role query is the one that carries a claim worth making -- a row a screen
        reader cannot announce is a row that is not there for some users -- so it is
        tried first and its success is recorded. The structural fallback exists because
        this scene must be able to tell "the panel never rendered the row" from "the
        row is unreachable by role", and a single locator that answers no to both
        cannot.
        """
        found = _first_visible(page.get_by_role("button", name = ESTIMATE_LABEL), "role=button")
        if found is not None:
            return found
        return _first_visible(page.locator("button").filter(has_text = ESTIMATE_LABEL), "button+text")

    def estimate_visible() -> bool:
        return estimate_button() is not None

    def wait_for_row(present: bool, timeout_ms: int = ESTIMATE_WAIT_MS) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            if estimate_visible() == present:
                return True
            page.wait_for_timeout(200)
        return estimate_visible() == present

    def _readable(raw: str | None) -> str:
        """`inner_text` with the row's layout glue normalised back to plain spaces.

        The breakdown captions join their items with U+00A0 so a narrow panel breaks
        between "262,144 tokens" and "4 slots" rather than inside either. That is a
        line-breaking detail and not something a reader distinguishes, but it does
        defeat a plain `"6,144 tokens" in text` check, so every assertion below reads
        the caption the way it looks rather than the way it is encoded.
        """
        return (raw or "").replace(" ", " ").strip()

    def header_text() -> str:
        button = estimate_button()
        if button is None:
            return ""
        try:
            return _readable(button.locator("xpath=..").inner_text())
        except Exception:
            return ""

    def breakdown_text() -> str:
        button = estimate_button()
        if button is None:
            return ""
        try:
            content_id = button.get_attribute("aria-controls")
        except Exception:
            content_id = None
        if not content_id:
            return ""
        # Attribute selector, not `#id`: React's useId mints ids containing ':', which is not a valid CSS id selector.
        panel = page.locator(f'[id="{content_id}"]').first
        if _count(panel) == 0:
            return ""
        try:
            return _readable(panel.inner_text())
        except Exception:
            return ""

    def wait_for_estimate_post(
        predicate,
        *,
        since: int,
        timeout_ms: int = ESTIMATE_WAIT_MS,
    ):
        """The first recorded exchange at or after `since` matching `predicate`, or None.

        Driven off the recorded transcript rather than `expect_request`, because these
        gates are about WHAT was asked as well as that something was.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            for record in exchanges[since:]:
                if predicate(record):
                    return record
            page.wait_for_timeout(200)
        for record in exchanges[since:]:
            if predicate(record):
                return record
        return None

    _used_contexts: set[int] = set()

    def reprice(popover, label: str):
        """Move the Context Length and return `(value, record)` once it has been priced.

        `value` is picked here rather than fixed per step, against what the box is showing
        at this moment: typing the number already displayed is a no-op the panel does not
        re-price (see the note on CTX_CANDIDATES), and the box reveals its number only once
        it has focus, so the choice cannot be made before the click.

        `record` is None when nothing was priced, which every caller reports as its own
        failure rather than carrying on against a stale row.
        """
        box = context_input(popover)
        if box is None:
            fail(f"{label}: the Context Length control is not in the run-settings panel")
            return None, None
        attempted: list[int] = []
        for _try in range(3):
            try:
                box.click()
                page.wait_for_timeout(200)
                shown = box.input_value()
            except Exception as exc:
                fail(f"{label}: could not focus the Context Length control: {exc}")
                return None, None
            try:
                shown_int = int(str(shown).replace(",", "").strip())
            except Exception:
                shown_int = None
            value = next(
                (
                    candidate
                    for candidate in CTX_CANDIDATES
                    if candidate != shown_int
                    and candidate not in _used_contexts
                    and candidate not in attempted
                ),
                None,
            )
            if value is None:
                fail(
                    f"{label}: ran out of Context Length values to type; the control shows "
                    f"{shown!r} and {sorted(_used_contexts)} are spent"
                )
                return None, None
            attempted.append(value)
            before = len(exchanges)
            try:
                box.fill(str(value))
                # Commit the draft: a value left focused mid-edit has been seen to stay a draft on the slower engines.
                box.press("Tab")
            except Exception as exc:
                fail(f"{label}: could not type {value} into the Context Length control: {exc}")
                return None, None
            record = wait_for_estimate_post(
                lambda rec: rec["request"].get("n_ctx") == value, since = before
            )
            if record is not None:
                _used_contexts.add(value)
                info(f"{label}: Context Length {shown!r} -> {value}, priced")
                return value, record
            info(
                f"{label}: typing {value} over {shown!r} produced no estimate request; "
                f"retrying with another value"
            )
        fail(
            f"{label}: the panel never re-priced after the Context Length was changed "
            f"(tried {attempted}); every n_ctx asked for so far="
            f"{[r['request'].get('n_ctx') for r in exchanges]!r}"
        )
        return None, None

    # ─────────────────────────────────────────────────────
    # 1. Open the run-settings panel for the GGUF and prove the row is there (HARD).
    # ─────────────────────────────────────────────────────
    step("open run-settings for the GGUF target")
    popover = open_picker()
    shoot("02-picker-open")
    if open_config(popover, MODEL_HINT) is None:
        fail(f"could not open run-settings for a model matching {MODEL_HINT!r}")
        write_transcript()
        shoot("03-config-failed")
        browser.close()
        print(f"[ui-memest] RESULT: FAIL ({len(_failed)} issue(s))", flush = True)
        for m in _failed:
            print(f"[ui-memest]   - {m}", flush = True)
        sys.exit(1)
    shoot("03-config-open")

    if not wait_for_row(True):
        fail(
            "the Estimated Memory Usage row never appeared for a GGUF target whose "
            f"estimate was stubbed available (exchanges={len(exchanges)}, "
            f"last={exchanges[-1:]!r})"
        )
        diagnose("row-missing", "button[name=/Estimated Memory Usage/]")
    else:
        info("OK row: Estimated Memory Usage rendered for the GGUF target")
        if "role=button" not in _match_note:
            runtime_warn(
                "the row was found only by markup, not by role=button: a screen "
                "reader would not announce this toggle"
            )

    if not exchanges:
        fail(
            "the panel never called POST /api/inference/estimate-memory, so nothing on "
            "screen can be attributed to the estimate at all"
        )

    # ─────────────────────────────────────────────────────
    # 2. The request carries the load settings, and a changed control re-prices (HARD).
    # ─────────────────────────────────────────────────────
    step("the Context Length on screen reaches the estimate request")
    priced_ctx, priced = reprice(popover, "context reaches the request")
    if priced is not None:
        info(f"OK request: estimate re-priced at n_ctx={priced_ctx}")
        body = priced["request"]
        # The rest of what the panel is supposed to price.
        required = ("model_path", "n_ctx", "cache_type_kv", "n_parallel", "gpu_memory_mode")
        missing = [key for key in required if key not in body]
        if missing:
            fail(f"the estimate request no longer carries {missing}; body keys={sorted(body)}")
        else:
            info(f"OK request: carries {list(required)}")
        model_path = str(body.get("model_path") or "")
        if MODEL_HINT.lower() not in model_path.lower():
            fail(
                f"the estimate priced {model_path!r}, which is not the model whose run "
                f"settings are open ({MODEL_HINT!r})"
            )
        else:
            info(f"OK request: priced model_path={model_path!r}")
        variant = body.get("gguf_variant")
        if (
            variant is not None
            and GGUF_VARIANT
            and str(variant).strip().lower() != (GGUF_VARIANT.strip().lower())
        ):
            # Only the structural locator found it, so the toggle is not reachable by role -- an ancestor is hiding it
            # from the accessibility tree, or its accessible name is not what it reads as. Reported rather than gated
            # while it is unproven which; the line above names the strategy that won.
            runtime_warn(
                f"the estimate priced quant {variant!r}, not {GGUF_VARIANT!r}; the picker row "
                f"may have collapsed onto a different variant"
            )

    # ─────────────────────────────────────────────────────
    # 3. The stubbed numbers are the ones on screen (HARD).
    # ─────────────────────────────────────────────────────
    step("the row displays the numbers the endpoint returned")
    if not wait_for_row(True):
        fail("the row is gone after the re-price, so its figures cannot be read")
    else:
        head = header_text()
        info(f"row header text: {head!r}")
        # Case-insensitively: the pill is written "Beta" in the source and rendered "BETA" by a CSS `uppercase`, so
        # `inner_text` returns the transformed string and an exact-case check fails on a pill that is right there on
        # screen.
        if not re.search(r"\bbeta\b", head, re.I):
            soft_fail(f"the row lost its Beta pill (header text={head!r})")
        total_gib = _gib(STUB_TOTAL_BYTES)
        gpu_gib = _gib(STUB_GPU_BYTES)
        # The total is shown in BOTH memory topologies: as the sole figure where the GPU and the host share one pool,
        # and beside the GPU share where they do not. The GPU figure only exists in the second, so it is asserted
        # only when its label is there; a CPU-only runner and an Apple machine legitimately show neither.
        if total_gib not in head:
            fail(
                f"the row does not show the returned total {total_gib!r} "
                f"(total_bytes={STUB_TOTAL_BYTES}); header text={head!r}"
            )
        else:
            info(f"OK figures: total {total_gib} is on screen")
        if re.search(r"\bGPU\b", head):
            if gpu_gib not in head:
                fail(
                    f"the row shows a GPU figure but not the returned gpu_bytes {gpu_gib!r}; "
                    f"header text={head!r}"
                )
            else:
                info(f"OK figures: GPU {gpu_gib} is on screen")
        else:
            info("single-pool layout: no separate GPU figure to check")
        shoot("04-row-collapsed")

        try:
            expander = estimate_button()
            if expander is None:
                raise RuntimeError("the row is no longer on screen")
            expander.click()
            page.wait_for_timeout(400)
        except Exception as exc:
            fail(f"could not expand the Estimated Memory Usage row: {exc}")
        detail = breakdown_text()
        info(f"row breakdown text: {detail!r}")
        if not detail:
            fail(
                "expanding the row produced no breakdown panel (the button's aria-controls "
                "names nothing on the page)"
            )
        else:
            for label, value in (
                ("Weights", _gib(STUB_WEIGHTS_BYTES)),
                ("KV cache", _gib(STUB_KV_BYTES)),
                ("Compute buffers", _gib(STUB_COMPUTE_BYTES)),
            ):
                if label not in detail:
                    fail(f"the breakdown has no {label!r} line; text={detail!r}")
                elif value not in detail:
                    fail(
                        f"the {label!r} line does not show the returned {value!r}; "
                        f"text={detail!r}"
                    )
                else:
                    info(f"OK breakdown: {label} = {value}")
            # The KV note is assembled from the RESPONSE's context and cache dtype, and the stub echoes the request's.
            if priced is not None and priced_ctx is not None:
                echoed = f"{priced_ctx:,} tokens"
                if echoed not in detail:
                    fail(
                        f"the KV note does not quote the priced context {echoed!r}, so the "
                        f"response's n_ctx is not what the row is displaying; text={detail!r}"
                    )
                else:
                    info(f"OK round trip: the KV note quotes {echoed}")
            layers_note = f"{STUB_GPU_LAYERS} of {STUB_LAYER_COUNT + 1} layers on GPU"
            if layers_note not in detail:
                runtime_warn(
                    f"the Weights line does not carry the placement note {layers_note!r}; "
                    f"text={detail!r}"
                )
            else:
                info(f"OK breakdown: placement note {layers_note!r}")
        shoot("05-row-expanded")

    # ─────────────────────────────────────────────────────
    # 4. An unavailable estimate HIDES the row (HARD).
    # ─────────────────────────────────────────────────────
    step("available:false hides the row")
    _mode[0] = "unavailable"
    _unavailable_ctx, unavailable_record = reprice(popover, "available:false hides the row")
    if unavailable_record is not None and unavailable_record["mode"] != "unavailable":
        fail(
            "the re-price was served before the endpoint was switched to unavailable, so "
            "the hide below would be measuring the wrong response"
        )
    if wait_for_row(False):
        info("OK hide: available:false removed the row")
    else:
        fail(
            "the row is still on screen after the estimate came back available:false; "
            f"header={header_text()!r}"
        )
    shoot("06-unavailable")

    # ─────────────────────────────────────────────────────
    # 5. Restoring the available response brings the row BACK (HARD).
    #    Without this the hide gate above is satisfied by a panel that simply died.
    # ─────────────────────────────────────────────────────
    step("restoring an available estimate brings the row back")
    _mode[0] = "available"
    restored_ctx, restored = reprice(popover, "restoring brings the row back")
    if restored is not None and restored["mode"] != "available":
        fail(
            "the restoring re-price was served before the endpoint was switched back, so "
            "the row coming back below would not be attributable to it"
        )
    if wait_for_row(True):
        head = header_text()
        if _gib(STUB_TOTAL_BYTES) in head:
            info("OK restore: the row came back with the returned total")
        else:
            fail(f"the row came back without the returned total; header={head!r}")
        detail = breakdown_text()
        echoed = f"{restored_ctx:,} tokens" if restored_ctx is not None else ""
        if detail and echoed and echoed not in detail:
            soft_fail(
                f"the restored row still quotes an older context; expected {echoed!r} in "
                f"{detail!r}"
            )
    else:
        fail(
            "the row did not come back once the estimate was available again, so the hide "
            "gate above proves nothing about the response and everything about the panel"
        )
    shoot("07-restored")

    # ─────────────────────────────────────────────────────
    # 6. A 404 from a backend predating the route also HIDES it (HARD).
    #
    # LAST, and it has to be. A non-OK answer stops the panel re-pricing for the rest of
    # its life: measured here, after one 404 the Context Length control keeps taking and
    # committing values -- 3072, 2048, 1536 -- and no further POST is ever made.
    # `available: false` does not do that, which is why the restore gate above sits with
    # the unavailable case rather than this one.
    #
    # Harmless in the situation this gate is about, since a backend predating the route
    # answers 404 to everything and the row is meant to stay hidden; it is only reachable
    # by an endpoint that fails and then recovers. Ordered around rather than asserted:
    # this suite is about the row, and pinning the freeze would pin behaviour nobody has
    # decided on.
    # ─────────────────────────────────────────────────────
    step("HTTP 404 hides the row")
    _mode[0] = "http404"
    _not_found_ctx, not_found_record = reprice(popover, "404 hides the row")
    if not_found_record is not None and not_found_record["mode"] != "http404":
        fail(
            "the re-price was served before the endpoint was switched to 404, so the hide "
            "below would be measuring the wrong response"
        )
    if wait_for_row(False):
        info("OK hide: a 404 removed the row rather than surfacing an error")
    else:
        fail(f"the row survived a 404 from the estimate endpoint; header={header_text()!r}")
    if page_errors:
        fail(f"a 404 estimate produced page errors instead of a hidden row: {page_errors[:3]!r}")
    shoot("08-http404")

    close_picker()

    if page_errors:
        fail(f"page errors during run: {page_errors[:3]!r}")

    write_transcript()
    browser.close()

if _failed:
    print(f"[ui-memest] RESULT: FAIL ({len(_failed)} issue(s))", flush = True)
    for m in _failed:
        print(f"[ui-memest]   - {m}", flush = True)
    sys.exit(1)
print(f"[ui-memest] RESULT: PASS ({len(exchanges)} estimate exchange(s) recorded)", flush = True)
sys.exit(0)
